"""OpenCode execution environment — run autonomous coding tasks via the OpenCode CLI.

Unlike shell-based backends (LocalEnvironment, DockerEnvironment, …), this
backend does **not** execute arbitrary shell commands.  Instead, the command
string passed to ``execute()`` is treated as a task prompt and forwarded to
``opencode run`` for autonomous code generation.

Environment variables (honoured at construction time):
    OPENCODE_MODEL     — Model to use (default: opencode/deepseek-v4-flash-free)
    OPENCODE_TIMEOUT   — Timeout in seconds (default: 600)
    OPENCODE_PROVIDER  — Provider name for custom providers (default: opencode)
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import IO

from hermes_constants import get_hermes_home
from tools.environments.base import BaseEnvironment, ProcessHandle, _popen_bash

logger = logging.getLogger(__name__)


def _find_opencode() -> str | None:
    """Locate the opencode binary on PATH."""
    return (
        subprocess.run(
            ["which", "opencode"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        or None
    )


def _resolve_opencode_binary() -> str:
    """Return the opencode binary path or raise."""
    binary = _find_opencode()
    if binary:
        return binary
    # Fallback to common npm-global locations
    for candidate in (
        os.path.expanduser("~/.npm-global/bin/opencode"),
        "/usr/local/bin/opencode",
        "/usr/bin/opencode",
    ):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        "opencode binary not found. Install it with: npm install -g opencode-ai"
    )


# ---------------------------------------------------------------------------
# Session log helper
# ---------------------------------------------------------------------------


def _write_session_log(session_id: str, content: str, exit_code: int) -> str:
    """Write a session log to {HERMES_HOME}/logs/opencode/session_{timestamp}.log.

    Returns the absolute path to the log file.
    """
    logs_dir = get_hermes_home() / "logs" / "opencode"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"session_{timestamp}_{session_id}.log"
    log_content = (
        f"=== OpenCode Session Log ===\n"
        f"Session ID: {session_id}\n"
        f"Timestamp:  {timestamp}\n"
        f"Exit Code:  {exit_code}\n"
        f"{'=' * 40}\n\n"
        f"{content}\n"
    )
    log_path.write_text(log_content, encoding="utf-8")
    return str(log_path)


# ---------------------------------------------------------------------------
# ProcessHandle adapter for opencode subprocess
# ---------------------------------------------------------------------------


class _OpenCodeProcessHandle:
    """Minimal ProcessHandle-compatible wrapper for an opencode subprocess."""

    def __init__(self, proc: subprocess.Popen):
        self._proc = proc

    @property
    def stdout(self) -> IO[str] | None:
        return self._proc.stdout

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode

    def poll(self) -> int | None:
        return self._proc.poll()

    def kill(self) -> None:
        try:
            self._proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def wait(self, timeout: float | None = None) -> int:
        self._proc.wait(timeout=timeout)
        return self._proc.returncode


# ---------------------------------------------------------------------------
# OpenCodeEnvironment
# ---------------------------------------------------------------------------


class OpenCodeEnvironment(BaseEnvironment):
    """Run autonomous coding tasks via the OpenCode CLI (``opencode run``).

    The ``execute()`` method treats the incoming command as a task prompt
    and passes it to OpenCode for autonomous handling.  Stdout/stderr are
    captured and a session log is written.

    Unlike LocalEnvironment / DockerEnvironment, this backend does **not**
    maintain a persistent shell session with env-var snapshots.  Each call
    to ``execute()`` is an independent OpenCode invocation.
    """

    # Do not try to embed stdin as a heredoc — we pass the prompt via CLI arg.
    _stdin_mode: str = "pipe"

    # Snapshot creation is a no-op; keep timeout short.
    _snapshot_timeout: int = 5

    def __init__(
        self,
        cwd: str = "",
        timeout: int = 600,
        env: dict | None = None,
    ):
        self._binary = _resolve_opencode_binary()

        # Read OpenCode-specific env vars (precedence: constructor kwargs > env vars > defaults)
        self._model = (env or {}).get("OPENCODE_MODEL") or os.getenv(
            "OPENCODE_MODEL", "opencode/deepseek-v4-flash-free"
        )
        self._provider = (env or {}).get("OPENCODE_PROVIDER") or os.getenv(
            "OPENCODE_PROVIDER", "opencode"
        )
        _env_timeout = os.getenv("OPENCODE_TIMEOUT")
        if _env_timeout:
            try:
                self._env_timeout = int(_env_timeout)
            except ValueError:
                self._env_timeout = timeout
        else:
            self._env_timeout = timeout

        # Expand ~ in cwd
        if cwd:
            cwd = os.path.expanduser(cwd)

        # Determine effective cwd
        effective_cwd = cwd or os.getcwd()
        if not os.path.isdir(effective_cwd):
            logger.warning(
                "OpenCodeEnvironment cwd %r does not exist; using current dir %r",
                cwd,
                os.getcwd(),
            )
            effective_cwd = os.getcwd()

        super().__init__(cwd=effective_cwd, timeout=timeout, env=env or {})
        self._session_counter = 0

        # init_session is a no-op for OpenCode; mark it ready so the base
        # class doesn't try to run a bash snapshot.
        self._snapshot_ready = False

        logger.info(
            "OpenCodeEnvironment initialized (binary=%s, model=%s, cwd=%s, timeout=%s)",
            self._binary,
            self._model,
            self.cwd,
            self.timeout,
        )

    # ------------------------------------------------------------------
    # Abstract method overrides
    # ------------------------------------------------------------------

    def init_session(self):
        """No-op — OpenCode does not use bash session snapshots."""
        self._snapshot_ready = False
        logger.info("OpenCodeEnvironment init_session (no-op)")

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ) -> ProcessHandle:
        """Run the opencode CLI instead of bash.

        *cmd_string* is treated as the task prompt for OpenCode.
        The base class may call this during init_session with bootstrap
        code — we detect that and return a no-op handle.
        """
        # During init_session the base class sends env-dump bootstrap code.
        # OpenCode doesn't need bash snapshots, so just return an empty handle.
        if login and "export -p" in cmd_string:
            proc = subprocess.Popen(
                ["true"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            return _OpenCodeProcessHandle(proc)

        # Normal invocation: run opencode with the given prompt.
        effective_timeout = min(timeout, self._env_timeout)
        context_files = self._resolve_context_files(cmd_string)

        cmd: list[str] = [
            self._binary,
            "run",
            cmd_string,
            "--model",
            self._model,
        ]
        if context_files:
            cmd.extend(["-f"] + context_files)

        logger.info("OpenCode run: %s", " ".join(cmd[:4]))
        logger.debug("Full command (cwd=%s): %s", self.cwd, " ".join(cmd))

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=self.cwd,
            env=self._build_env(),
        )
        return _OpenCodeProcessHandle(proc)

    def _build_env(self) -> dict:
        """Build the environment dict for the opencode subprocess."""
        run_env = dict(os.environ)
        # Forward any extra env passed at construction time
        if self.env:
            run_env.update(self.env)
        return run_env

    def _resolve_context_files(self, _prompt: str) -> list[str]:
        """Determine optional context files to pass to OpenCode.

        Currently returns an empty list — future work could scan the CWD
        for relevant files or use file lists from the prompt.
        """
        return []

    # ------------------------------------------------------------------
    # Execute override — bypasses bash session wrapping
    # ------------------------------------------------------------------

    def execute(
        self,
        command: str,
        cwd: str = "",
        *,
        timeout: int | None = None,
        stdin_data: str | None = None,
    ) -> dict:
        """Execute a task via OpenCode, returning {output, exit_code, session_log}.

        *command* is the task prompt sent to ``opencode run``.
        *cwd* overrides the working directory for this call.
        *timeout* overrides the default timeout.
        """
        effective_timeout = timeout or self._env_timeout or self.timeout
        effective_cwd = cwd or self.cwd

        switched_cwd = effective_cwd and os.path.isdir(effective_cwd)
        if switched_cwd:
            orig_cwd = self.cwd
            self.cwd = effective_cwd

        self._session_counter += 1
        session_id = f"{self._session_id}-{self._session_counter}"

        try:
            proc = self._run_bash(command, timeout=effective_timeout)
            result = self._wait_for_process(proc, timeout=effective_timeout)
        finally:
            if switched_cwd:
                self.cwd = orig_cwd

        output = result.get("output", "")
        exit_code = result.get("returncode", 1)

        # Write session log
        try:
            log_path = _write_session_log(session_id, output, exit_code)
        except Exception as exc:
            logger.warning("Failed to write OpenCode session log: %s", exc)
            log_path = ""

        # Return the OpenCode-specific result contract
        return {
            "output": output,
            "exit_code": exit_code,
            "session_log": log_path,
            # Also include returncode for base-class compatibility
            "returncode": exit_code,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """No persistent resources to clean up."""
        pass
