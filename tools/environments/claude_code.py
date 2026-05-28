"""Claude Code CLI execution environment — delegate tasks to Claude Code.

Instead of running shell commands through ``bash``, this backend hands off a
task description directly to the Claude Code CLI (``claude --bare -p``),
letting Claude Code work autonomously and then returning the aggregated
output.

Use case: complex, multi-step coding tasks where you want Claude Code's
agentic loop (tool use, file editing, iteration) to handle the work in a
single delegated call.

Configuration via environment variables:
    - ``ANTHROPIC_API_KEY`` — API key for Claude Code
    - ``ANTHROPIC_BASE_URL`` — API endpoint override (e.g. DeepSeek compat)
    - ``CLAUDE_CODE_MODEL`` — model to use (default: ``haiku``)
    - ``CLAUDE_CODE_MAX_TURNS`` — maximum agentic turns (default: ``10``)
    - ``CLAUDE_CODE_TIMEOUT`` — wall-clock timeout in seconds (default: ``300``)
    - ``MAX_THINKING_TOKENS`` — set to ``0`` to disable thinking for
      DeepSeek/compatible API backends

IMPORTANT KNOWN LIMITATION
--------------------------
DeepSeek's Anthropic-compatible API (``ANTHROPIC_BASE_URL`` set to
``https://api.deepseek.com/anthropic``) exhibits a protocol mismatch with
Claude Code v2.x when tools are used.  The API returns ``thinking`` blocks
but Claude Code passes them back in a different format than expected,
causing tool-use failures.  Text-only (no-tool) responses work fine with
DeepSeek.

When using a real Anthropic API key (https://api.anthropic.com), full tool
support works as intended.
"""

import logging
import os
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from tools.environments.base import BaseEnvironment, ProcessHandle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_claude() -> str | None:
    """Locate the ``claude`` CLI binary."""
    # Check explicit override first
    override = os.environ.get("CLAUDE_CODE_PATH")
    if override and os.path.isfile(override):
        return override
    # Then PATH
    found = shutil.which("claude")
    if found:
        return found
    # Common npm-style global install path
    npm_global = os.path.expanduser("~/.npm-global/bin/claude")
    if os.path.isfile(npm_global):
        return npm_global
    return None


def _make_session_log_dir() -> Path:
    """Create and return the session log directory under HERMES_HOME."""
    log_dir = get_hermes_home() / "logs" / "claude-code"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _build_claude_command(
    prompt: str,
    *,
    model: str,
    max_turns: int,
    allowed_tools: str,
    cwd: str,
) -> list[str]:
    """Build the ``claude`` CLI invocation."""
    cmd = [
        "claude",
        "--bare",
        "-p",
        prompt,
        "--max-turns",
        str(max_turns),
        "--model",
        model,
        "--allowedTools",
        allowed_tools,
        "--no-session-persistence",
    ]

    # Debug mode: --verbose if requested
    if os.environ.get("CLAUDE_CODE_VERBOSE", "").lower() in {"1", "true", "yes"}:
        cmd.append("--verbose")

    return cmd


def _build_claude_env(env_override: dict | None = None) -> dict[str, str]:
    """Build the environment dict for the Claude Code subprocess.

    We forward the host env but inject/override the Claude-specific variables
    that the task spec requires honoring.
    """
    env = dict(os.environ)

    # Forward explicit env overrides (these are the Claude Code-specific ones)
    if env_override:
        env.update(env_override)

    # MAX_THINKING_TOKENS=0 disables thinking blocks for DeepSeek compatibility
    if "MAX_THINKING_TOKENS" not in env:
        env["MAX_THINKING_TOKENS"] = "0"

    # ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL — pass through if already set
    # (usually from the Hermes .env loading).  Don't strip these for Claude Code
    # subprocesses since it needs them to authenticate.

    return env


def _parse_turns_used(output: str) -> int:
    """Try to extract the number of turns from Claude Code output.

    Claude Code emits a summary line like ``[N turn(s) completed]`` or
    ``turns: N`` near the end.  Fall back to 0 if not found.
    """
    patterns = [
        r"(\d+)\s+turns?\s+completed",
        r"turns:\s*(\d+)",
        r"total turns:\s*(\d+)",
        r"completed\s+(\d+)\s+turns?",
    ]
    for pattern in patterns:
        m = re.search(pattern, output, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return 0


def _parse_cost_usd(output: str) -> float:
    """Try to extract the cost from Claude Code output.

    Claude Code sometimes emits a cost line.  Fall back to 0.0 if not found.
    """
    patterns = [
        r"\$\s*([\d.]+)",
        r"cost:\s*\$\s*([\d.]+)",
        r"total cost:\s*\$\s*([\d.]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, output, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return 0.0


# ---------------------------------------------------------------------------
# ClaudeCodeEnvironment
# ---------------------------------------------------------------------------


class ClaudeCodeEnvironment(BaseEnvironment):
    """Run tasks via the Claude Code CLI in non-interactive print mode.

    Unlike other backends that execute arbitrary shell commands through bash,
    this backend delegates a *task description* to Claude Code, which then
    autonomously works on it (reading files, running commands, editing, etc.)
    up to the configured max turns.

    The ``execute()`` method takes the ``command`` parameter as a task prompt
    and runs ``claude --bare -p "<prompt>"`` inside the target directory.

    Session logs are written to ``{HERMES_HOME}/logs/claude-code/`` with
    timestamped filenames for later review.
    """

    def __init__(
        self,
        cwd: str = "",
        timeout: int = 300,
        env: dict | None = None,
        *,
        model: str | None = None,
        max_turns: int | None = None,
        allowed_tools: str | None = None,
        claude_binary: str | None = None,
    ):
        # Read config from environment variables, with constructor overrides
        self._model = model or os.environ.get("CLAUDE_CODE_MODEL", "haiku")
        self._max_turns = max_turns or int(os.environ.get("CLAUDE_CODE_MAX_TURNS", "10"))
        self._allowed_tools = allowed_tools or os.environ.get(
            "CLAUDE_CODE_ALLOWED_TOOLS", "Read,Edit,Write,Bash"
        )
        self._claude_binary = claude_binary or _find_claude()

        self._timeout = int(os.environ.get("CLAUDE_CODE_TIMEOUT", str(timeout)))

        # Effective env vars that get passed to the subprocess
        self._claude_env: dict[str, str] = _build_claude_env(env)

        # Resolve cwd
        if cwd:
            cwd = os.path.expanduser(cwd)
        else:
            cwd = os.getcwd()

        # Call parent __init__ (sets up temp paths etc. we don't fully use,
        # but the abstract base requires them).
        super().__init__(cwd=cwd, timeout=self._timeout, env=env)

        # Mark snapshot as "ready" so base class machinery doesn't try to
        # source a snapshot file before our commands.
        self._snapshot_ready = True

        if not self._claude_binary:
            logger.warning(
                "Claude Code binary not found. Set CLAUDE_CODE_PATH or ensure "
                "'claude' is on PATH. Install with: npm install -g @anthropic-ai/claude-code"
            )

        logger.info(
            "ClaudeCodeEnvironment initialized: model=%s, max_turns=%s, timeout=%ss, cwd=%s",
            self._model,
            self._max_turns,
            self._timeout,
            self.cwd,
        )

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ) -> ProcessHandle:
        """Not used by this backend — the Claude Code path bypasses bash entirely.

        This stub satisfies the abstract method requirement.  In normal use,
        :meth:`execute` overrides the base class flow and calls Claude Code
        directly without ever invoking ``_run_bash``.
        """
        raise NotImplementedError(
            "ClaudeCodeEnvironment does not support direct bash execution. "
            "Use execute() to delegate tasks to Claude Code."
        )

    def cleanup(self) -> None:
        """Release resources (no-op — there are no persistent processes)."""
        pass

    def _write_session_log(self, output: str, exit_code: int, *, started: float) -> str:
        """Write the session output to a timestamped log file.

        Returns the path to the written log file.
        """
        log_dir = _make_session_log_dir()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"session_{ts}.log"

        elapsed = time.monotonic() - started

        lines = [
            f"=== Claude Code Session Log ===",
            f"Timestamp: {datetime.now().isoformat()}",
            f"Model: {self._model}",
            f"Max turns: {self._max_turns}",
            f"Working directory: {self.cwd}",
            f"Exit code: {exit_code}",
            f"Elapsed: {elapsed:.1f}s",
            "=== Output ===",
            output,
            "",
        ]

        log_path.write_text("\n".join(lines), encoding="utf-8")
        logger.debug("Claude Code session log written to %s", log_path)
        return str(log_path)

    # ------------------------------------------------------------------
    # Overridden execute — runs Claude Code instead of bash
    # ------------------------------------------------------------------

    def execute(
        self,
        command: str,
        cwd: str = "",
        *,
        timeout: int | None = None,
        stdin_data: str | None = None,
    ) -> dict[str, Any]:
        """Delegate a task to Claude Code CLI.

        *command* is treated as a task prompt (not a shell command).  Claude
        Code will be invoked in print mode (``--bare -p``) with the prompt
        and allowed to work up to ``max_turns`` iterations.

        Returns a result dict compatible with the terminal tool:
            {
                "output": str,        # Claude Code stdout/stderr
                "returncode": int,    # exit status
                "session_log": str,   # path to session log file
                "model_used": str,    # model name
                "turns_used": int,    # turns consumed (best-effort parse)
                "cost_usd": float,    # cost (best-effort parse)
            }
        """
        effective_timeout = timeout or self._timeout
        effective_cwd = cwd or self.cwd

        started = time.monotonic()

        # Validate prerequisites
        if not self._claude_binary:
            return {
                "output": "Claude Code CLI not found. Install with: "
                "npm install -g @anthropic-ai/claude-code\n"
                "or set CLAUDE_CODE_PATH to the binary location.",
                "returncode": 127,
                "session_log": "",
                "model_used": self._model,
                "turns_used": 0,
                "cost_usd": 0.0,
            }

        # Build the claude command
        cmd = _build_claude_command(
            command,
            model=self._model,
            max_turns=self._max_turns,
            allowed_tools=self._allowed_tools,
            cwd=effective_cwd,
        )

        run_env = _build_claude_env(self.env or {})

        logger.info(
            "Claude Code executing: turns=%s, model=%s, timeout=%ss, cwd=%s",
            self._max_turns,
            self._model,
            effective_timeout,
            effective_cwd,
        )

        # Spawn the Claude Code process
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=effective_cwd,
                env=run_env,
            )
        except FileNotFoundError:
            return {
                "output": f"Failed to execute Claude Code: binary not found at "
                f"{self._claude_binary}",
                "returncode": 127,
                "session_log": "",
                "model_used": self._model,
                "turns_used": 0,
                "cost_usd": 0.0,
            }
        except OSError as exc:
            return {
                "output": f"Failed to execute Claude Code: {exc}",
                "returncode": 1,
                "session_log": "",
                "model_used": self._model,
                "turns_used": 0,
                "cost_usd": 0.0,
            }

        # Wait with timeout, collecting output
        output_chunks: list[str] = []
        timed_out = False

        try:
            output, _ = proc.communicate(timeout=effective_timeout)
            output_chunks.append(output)
            exit_code = proc.returncode or 0
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            try:
                output, _ = proc.communicate(timeout=5)
                if output:
                    output_chunks.append(output)
            except Exception:
                pass
            exit_code = 124

        full_output = "".join(output_chunks)

        if timed_out:
            full_output += f"\n\n[Claude Code timed out after {effective_timeout}s]"
            logger.warning(
                "Claude Code timed out after %ds (model=%s)",
                effective_timeout,
                self._model,
            )

        # Parse metadata from output
        turns_used = _parse_turns_used(full_output)
        cost_usd = _parse_cost_usd(full_output)

        # Write session log
        session_log = self._write_session_log(
            full_output, exit_code, started=started
        )

        elapsed = time.monotonic() - started
        logger.info(
            "Claude Code done: exit=%s, turns=%s, cost=$%.4f, elapsed=%.1fs",
            exit_code,
            turns_used,
            cost_usd,
            elapsed,
        )

        return {
            "output": full_output,
            "returncode": exit_code,
            "session_log": session_log,
            "model_used": self._model,
            "turns_used": turns_used,
            "cost_usd": cost_usd,
        }
