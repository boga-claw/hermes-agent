#!/usr/bin/env python3
"""Per-file parallel test runner.

The minimum-viable replacement for pytest-xdist + a subprocess-isolation
plugin. Discovers test files under ``tests/`` (excluding integration/e2e
unless explicitly requested), then runs one ``python -m pytest <file>``
subprocess per file, with bounded parallelism (default: ``os.cpu_count()``).

Why per-file rather than per-test?
    Per-test spawn overhead (~250ms × 17k tests = 70min CPU minimum)
    swamped the actual work. Per-file spawn (~250ms × ~850 files = ~3.5min)
    fits in the budget while still giving every file a fresh Python
    interpreter — the only isolation boundary that actually matters
    (cross-file module-level state leakage was the original flake source;
    intra-file state is the test author's responsibility).

Why drop xdist entirely?
    xdist's persistent workers accumulate state across files, which is
    exactly the leakage we wanted to fix. xdist also adds complexity
    (loadfile vs loadscope, --max-worker-restart, internal control plane)
    that we don't need when the unit of work is "run pytest on one file".
    A subprocess.Popen pool gated by a semaphore is ~60 lines and does
    the job.

Usage:
    python scripts/run_tests_parallel.py [pytest_args...]

    Common pytest args pass through (e.g. ``-v``, ``-x``, ``--tb=long``,
    ``-k 'pattern'``, ``--lf``).

Environment:
    HERMES_TEST_WORKERS  Override worker count (default: os.cpu_count())
    HERMES_TEST_PATHS    Override discovery roots (colon-sep, default: 'tests')

Exit code: 0 if every file's pytest exited 0; 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import List, Tuple


# Default test discovery roots.
_DEFAULT_ROOTS = ["tests"]

# Directories to skip during discovery — the e2e + integration suites
# require real services and are run separately. Match exactly the
# ``--ignore=`` flags the previous CI command used.
_SKIP_PARTS = {"integration", "e2e"}

# Per-file wall-clock cap. Generous default — pytest-timeout still
# enforces per-test caps inside each subprocess; this is just an outer
# safety net so a single hung file can't stall the whole suite. Override
# via --file-timeout or HERMES_TEST_FILE_TIMEOUT.
_DEFAULT_FILE_TIMEOUT_SECONDS = 600.0  # 10 minutes


def _discover_files(roots: List[Path]) -> List[Path]:
    """Return every ``test_*.py`` under the given roots (sorted).

    Roots may be directories (recursed for ``test_*.py``) or explicit
    ``.py`` files (included as-is, even if they don't match the
    ``test_*`` prefix — caller knows what they want).

    Exclude any file whose path contains a component in ``_SKIP_PARTS``,
    UNLESS the user explicitly named it as a root (in which case the
    user's intent overrides the skip filter).
    """
    seen: set[Path] = set()
    out: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            # Explicit file: include it as-is, skip the _SKIP_PARTS filter
            # since the user named it directly.
            real = root.resolve()
            if real not in seen:
                seen.add(real)
                out.append(root)
            continue
        for path in root.rglob("test_*.py"):
            if any(part in _SKIP_PARTS for part in path.parts):
                continue
            real = path.resolve()
            if real in seen:
                continue
            seen.add(real)
            out.append(path)
    return sorted(out)


def _kill_tree(proc: "subprocess.Popen", pgid: int | None = None) -> None:
    """Kill the pytest subprocess and every descendant it spawned.

    A test run can spin up uvicorn servers, async runtimes, or other
    long-running grandchildren that survive the pytest subprocess exit
    if we don't kill the whole tree. ``subprocess.Popen.kill()`` only
    targets the immediate child; grandchildren reparent to PID 1
    (Linux) / get adopted by services.exe (Windows) and leak.

    POSIX: the caller must pass ``pgid`` — the process group id captured
    immediately after Popen (via ``os.getpgid(proc.pid)``). We can't
    look it up here in the happy path because by the time we get
    called the leader process has already been reaped and its pid is
    gone from the kernel's process table, even though descendants in
    the group are still alive. SIGKILL'ing the captured pgid takes out
    everything in that group atomically.

    Windows: ``taskkill /F /T /PID`` walks the recorded ppid chain and
    terminates the whole tree, even when the root has already exited.

    Why not psutil: psutil walks the parent-child tree, but in the
    happy path the root has already been reaped so ``psutil.Process(pid)``
    can't find it; grandchildren reparented to PID 1 are also
    unreachable by tree walk at that point. The platform-native
    primitives (process groups / taskkill) handle both cases correctly
    without an extra abstraction layer.
    """
    if proc.pid is None:
        return

    if sys.platform == "win32":
        try:
            
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )  # windows-footgun: ok
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
    else:
        # POSIX: kill the captured pgid. Local-import signal so the
        # SIGKILL attribute is never referenced on Windows.
        if pgid is not None:
            try:
                import signal as _signal
                os.killpg(pgid, _signal.SIGKILL)  # windows-footgun: ok
            except (ProcessLookupError, PermissionError, OSError):
                pass

    # Belt-and-suspenders: ensure subprocess.communicate() sees the exit.
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass


def _run_one_file(
    file: Path,
    pytest_args: List[str],
    repo_root: Path,
    file_timeout: float,
) -> Tuple[Path, int, str]:
    """Run ``python -m pytest <file> <pytest_args>`` in a fresh subprocess.

    Returns (file, returncode, captured_combined_output).

    pytest exit codes (https://docs.pytest.org/en/stable/reference/exit-codes.html):
        0 = all tests passed
        1 = some tests failed
        2 = test execution interrupted
        3 = internal error
        4 = pytest CLI usage error
        5 = no tests collected

    We treat exit 5 as a pass: it just means every test in the file was
    skipped or filtered by a marker (e.g. ``-m 'not integration'`` skips
    files where every test is marked integration). That's intentional and
    not a failure mode.

    On per-file timeout (``file_timeout`` seconds) or any other exception
    during ``communicate()``, we kill the whole process group / process
    tree so grandchildren (uvicorn servers, async runtimes, etc.) do not
    orphan onto PID 1. The pytest-timeout plugin enforces per-test
    timeouts inside the subprocess; this outer timeout exists only to
    bound a pathologically slow or hung file as a whole.
    """
    cmd = [sys.executable, "-m", "pytest", str(file), *pytest_args]
    proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        # POSIX: place the child at the head of its own process group so
        # _kill_tree can SIGKILL the group atomically.
        # Windows: this maps to CREATE_NEW_PROCESS_GROUP in CPython 3.12+;
        # _kill_tree handles the Windows path via taskkill /F /T.
        start_new_session=True,
    )

    # Capture the pgid NOW, before the leader can exit and be reaped.
    # Once the leader is reaped, os.getpgid(proc.pid) raises
    # ProcessLookupError even though grandchildren in that group are
    # still alive — defeating the whole cleanup. None on Windows where
    # the pgid concept doesn't apply (taskkill walks ppid chain instead).
    pgid: int | None = None
    if sys.platform != "win32":
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, PermissionError):
            # Astonishingly fast child? Already dead. _kill_tree's
            # fallback will handle this case as a no-op.
            pgid = None

    try:
        output, _ = proc.communicate(timeout=file_timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        _kill_tree(proc, pgid=pgid)
        # Drain whatever the child wrote before we killed it so we have
        # something to surface in the failure dump.
        try:
            output, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            output = "(file timeout exceeded; output unavailable)"
        rc = 124  # de facto convention for "killed by timeout".
        output = (
            f"(per-file timeout: {file_timeout:.0f}s exceeded; "
            f"process tree SIGKILL'd)\n{output}"
        )
    except BaseException:
        # KeyboardInterrupt / runner crash — make sure no zombie
        # grandchildren outlive us.
        _kill_tree(proc, pgid=pgid)
        raise
    else:
        # Happy path: pytest exited on its own. The child process already
        # cleaned up its grandchildren if it's well-behaved, but
        # well-behaved is not universal — kill the group anyway. Already-
        # dead processes are a no-op.
        _kill_tree(proc, pgid=pgid)

    if rc == 5:
        # No tests collected — every test in the file was filtered out.
        # Treat as a pass; surface info in a slightly distinct status
        # so the operator can spot it.
        rc = 0
    return file, rc, output


def _format_file(file: Path, repo_root: Path) -> str:
    """Render a test-file path for display: strip the repo-root prefix
    when possible so output reads ``tests/acp/test_auth.py`` instead of
    ``/home/runner/work/hermes-agent/hermes-agent/tests/acp/test_auth.py``.

    Falls back to the absolute path for anything outside the repo root.
    """
    try:
        return str(file.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(file)


def _print_progress(
    done: int,
    total: int,
    file: Path,
    rc: int,
    dur: float,
    repo_root: Path,
    passed: int,
    failed: int,
) -> None:
    """Single-line live progress.

    Format:
      [done/total  ✓passed ✗failed ⏲inflight] status path (Xs)

    The running tally lets you see at a glance how many files succeeded /
    failed / are still in flight without scanning previous output.
    """
    status = "✓" if rc == 0 else "✗"
    in_flight = total - done
    msg = (
        f"[{done:>4}/{total}  ✓{passed} ✗{failed} ⏲{in_flight}] "
        f"{status} {_format_file(file, repo_root)} ({dur:.1f}s)"
    )
    # Truncate to terminal width if available (no clobbering ANSI lines).
    try:
        cols = os.get_terminal_size().columns
        if len(msg) > cols:
            msg = msg[: cols - 1] + "…"
    except OSError:
        pass
    print(msg, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=int(os.environ.get("HERMES_TEST_WORKERS") or os.cpu_count() or 4),
        help="Parallel worker count (default: $HERMES_TEST_WORKERS or os.cpu_count())",
    )
    parser.add_argument(
        "--paths",
        default=os.environ.get("HERMES_TEST_PATHS", ":".join(_DEFAULT_ROOTS)),
        help="Colon-separated discovery roots (default: 'tests')",
    )
    parser.add_argument(
        "--include-integration",
        action="store_true",
        help="Don't skip integration/ e2e/ during discovery",
    )
    parser.add_argument(
        "--file-timeout",
        type=float,
        default=float(
            os.environ.get("HERMES_TEST_FILE_TIMEOUT", _DEFAULT_FILE_TIMEOUT_SECONDS)
        ),
        help=(
            "Per-file wall-clock cap in seconds. On timeout, the pytest "
            "subprocess and its full process tree are SIGKILL'd. "
            "Default: 600 (10 min), env: HERMES_TEST_FILE_TIMEOUT."
        ),
    )
    parser.add_argument(
        "paths_positional",
        nargs="*",
        metavar="PATH",
        help=(
            "Restrict discovery to these paths (directories or .py files). "
            "Mutually exclusive with --paths. Anything after a literal '--' "
            "separator is passed through to each per-file pytest invocation."
        ),
    )
    # Manually split argv on '--' so positional paths and pytest passthrough
    # args don't fight over each other. argparse's nargs="*" positional is
    # greedy and will swallow everything after '--' including the pytest
    # flags, defeating the convention.
    argv = sys.argv[1:]
    if "--" in argv:
        sep = argv.index("--")
        our_args, pytest_passthrough = argv[:sep], argv[sep + 1 :]
    else:
        our_args, pytest_passthrough = argv, []
    args = parser.parse_args(our_args)

    repo_root = Path(__file__).resolve().parent.parent

    # Resolve discovery roots: positional path args override --paths if any
    # were supplied, otherwise --paths (which itself defaults to 'tests').
    if args.paths_positional:
        # Positionals can be directories OR explicit .py files. Either is
        # fine — _discover_files handles both via rglob('test_*.py') for
        # dirs and direct inclusion for files.
        roots = [repo_root / p for p in args.paths_positional]
    else:
        roots = [repo_root / p for p in args.paths.split(":") if p]

    if args.include_integration:
        # Caller takes responsibility — typically used via explicit -k filter.
        global _SKIP_PARTS  # noqa: PLW0603 — config knob
        _SKIP_PARTS = set()

    files = _discover_files(roots)
    if not files:
        print(f"No test files discovered under {[str(r) for r in roots]}", file=sys.stderr)
        return 1

    print(
        f"Discovered {len(files)} test files under "
        f"{[str(r.relative_to(repo_root)) if r.is_relative_to(repo_root) else str(r) for r in roots]}; "
        f"running with -j {args.jobs}",
        flush=True,
    )

    # Capture and print on completion (out-of-order is fine — keeps the
    # terminal clean rather than interleaving N parallel pytest outputs).
    failures: List[Tuple[Path, str]] = []
    started = time.monotonic()
    done_count = 0
    pass_count = 0
    fail_count = 0
    lock = threading.Lock()

    def _on_done(file: Path, started_at: float, fut: "Future[Tuple[Path, int, str]]") -> None:
        nonlocal done_count, pass_count, fail_count
        try:
            fpath, rc, output = fut.result()
        except Exception as exc:  # noqa: BLE001 — must always advance counter
            with lock:
                done_count += 1
                fail_count += 1
                failures.append((file, f"runner crashed: {exc!r}"))
                _print_progress(
                    done_count, len(files), file, 1,
                    time.monotonic() - started_at,
                    repo_root, pass_count, fail_count,
                )
            return
        with lock:
            done_count += 1
            if rc == 0:
                pass_count += 1
            else:
                fail_count += 1
                failures.append((fpath, output))
            _print_progress(
                done_count, len(files), fpath, rc,
                time.monotonic() - started_at,
                repo_root, pass_count, fail_count,
            )

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures: List[Future] = []
        for file in files:
            t0 = time.monotonic()
            fut = pool.submit(
                _run_one_file, file, pytest_passthrough, repo_root, args.file_timeout
            )
            fut.add_done_callback(lambda f, file=file, t0=t0: _on_done(file, t0, f))
            futures.append(fut)
        # Block until everything's done. ThreadPoolExecutor.__exit__ waits
        # for all submitted work, but doing it explicitly here makes the
        # control flow obvious.
        for fut in futures:
            fut.result() if fut.exception() is None else None

    elapsed = time.monotonic() - started
    print()
    print(f"=== Summary: {len(files)} files in {elapsed:.1f}s ({args.jobs} workers) ===")
    print(f"  Passed: {len(files) - len(failures)}")
    print(f"  Failed: {len(failures)}")

    if failures:
        print()
        print("=== Failure output ===")
        for file, output in failures:
            print()
            print(f"--- {_format_file(file, repo_root)} ---")
            print(output.rstrip())
        print()
        print("=== Failed files ===")
        for file, _ in failures:
            print(f"  {_format_file(file, repo_root)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
