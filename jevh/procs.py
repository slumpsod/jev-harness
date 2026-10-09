"""Running other programs without letting them hang the harness.

`subprocess.run(cmd, capture_output=True, timeout=...)` does not end on time on Windows. On a timeout it kills only the
process it started (with shell=True, the shell), then waits with no limit for the output pipes to close, and a
grandchild still holds them: npm test's node, a test's server, pytest-xdist's workers. The call then returns only when
that grandchild ends, if it ever does. `run` kills the whole process tree on a timeout and stops waiting for the pipes a
few seconds later.

`run` also gives the program an empty stdin unless it is handed input. A program that asks a question (pip uninstall's
"Proceed (Y/n)?", a test calling input(), npx asking to install) gets an end of file at once instead of waiting forever
on whatever stdin the harness has.
"""
from __future__ import annotations

import os
import signal
import subprocess
from typing import Any

DRAIN = 5.0   # seconds a killed program's pipes may take to close before its last output is given up


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill `proc` and everything it started. On Windows, taskkill /T follows the processes it started. Elsewhere this
    needs `proc` to lead its own process group (start_new_session=True, as `run` does); otherwise only `proc` dies."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass


def run(cmd: str | list[str], *, shell: bool = False, cwd: str | os.PathLike | None = None,
        env: dict[str, str] | None = None, timeout: float | None = None, input: str | None = None,
        encoding: str = "utf-8", errors: str = "replace") -> subprocess.CompletedProcess:
    """subprocess.run(cmd, capture_output=True, text=True, ...) that ends on time. On `timeout`, the program and
    everything it started are killed, and subprocess.TimeoutExpired is raised with whatever output had been read
    (`.stdout`, `.stderr`). Without `input`, the program's stdin is empty."""
    kw: dict[str, Any] = {}
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kw["start_new_session"] = True   # its own process group, so kill_tree takes its children too
    proc = subprocess.Popen(cmd, shell=shell, cwd=cwd, env=env,
                            stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding=encoding,
                            errors=errors, **kw)
    try:
        out, err = proc.communicate(input, timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        try:
            out, err = proc.communicate(timeout=DRAIN)
        except (subprocess.TimeoutExpired, ValueError):
            out, err = "", ""   # something it started outlived the kill and still holds the pipes: leave them
        raise subprocess.TimeoutExpired(cmd, timeout, output=out, stderr=err) from None
    except BaseException:
        kill_tree(proc)   # (a KeyboardInterrupt or a Stop must not leave it running)
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
