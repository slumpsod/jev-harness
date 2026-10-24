"""Restart the dashboard (or the desktop app that serves it) so it runs the code on disk.

An update changes the files, but a running server keeps the code it imported: only a new process loads the new version.
The server can't start its successor itself, since the port (and the app's single-instance port) stay its own until it
exits. So `schedule()` starts a small helper (`python -m jevh.restart`), detached from the server, which waits for the
server's process to exit (ending it after WAIT_EXIT if it hangs) and for its port to free up, then runs the same command
line again in the same folder with the same environment, and exits.

Where the new process's output goes: a server started from a console (`jevh dashboard` in a terminal) opens again in a
new console window on Windows; any other (the app, a hidden dashboard, anything on Linux or macOS) runs in the background
with its output in `output()` beside the log the caller names (the dashboard's: workspace/restart.log says what each restart
did, workspace/restart-output.log has what the harness started by the last one printed).
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

WAIT_EXIT = 30.0   # seconds for the old process to exit before it is ended
WAIT_PORT = 15.0   # then for its port to be free (Windows can hold a closed listener's port a moment)
START_CWD = os.getcwd()   # (read at import, when the server starts: the folder it was started in)

_DETACHED = 0x00000008 | 0x00000200   # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
_BREAKAWAY = 0x01000000                # CREATE_BREAKAWAY_FROM_JOB: outlive a job that ends the server's children with it
_NEW_CONSOLE = 0x00000010


def output(log: Path) -> Path:
    """Where a harness started in the background by a restart writes (its own file: two processes appending to one file
    through handles of their own write over each other's lines on Windows)."""
    return Path(log).with_name("restart-output.log")


def command() -> list[str]:
    """The command line that started this process: the same interpreter (pythonw stays pythonw), options and arguments."""
    return [sys.executable, *sys.orig_argv[1:]]


def has_console() -> bool:
    """Is this process writing to a console window someone can see (a terminal), rather than a file or nothing?"""
    try:
        if not (sys.stdout and sys.stdout.isatty()):
            return False
        if os.name == "nt":
            import ctypes
            return bool(ctypes.windll.kernel32.GetConsoleWindow())
        return True
    except (AttributeError, OSError, ValueError):
        return False


def schedule(port: int, log: Path, cmd: list[str] | None = None, pid: int | None = None) -> list[str]:
    """Start the helper that runs `cmd` (this process's own command line) again once process `pid` (this one) has exited
    and `port` is free. The caller then stops its server. Returns the command that will run."""
    cmd = cmd or command()
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, "-m", "jevh.restart", "--pid", str(pid or os.getpid()), "--port", str(port),
            "--cwd", START_CWD, "--output", str(output(log))]
    if os.name == "nt" and has_console():
        args.append("--console")
    args += ["--", *cmd]
    kw: dict = {"cwd": START_CWD, "stdin": subprocess.DEVNULL, "close_fds": True}
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} restart requested by pid {pid or os.getpid()}: {subprocess.list2cmdline(cmd)}\n")
        f.flush()
        kw.update(stdout=f, stderr=f)   # (what the helper does)
        if os.name == "nt":
            try:
                subprocess.Popen(args, creationflags=_DETACHED | _BREAKAWAY, **kw)
            except OSError:   # the job doesn't allow breaking away: most don't end their children anyway
                subprocess.Popen(args, creationflags=_DETACHED, **kw)
        else:
            subprocess.Popen(args, start_new_session=True, **kw)
    return cmd


# ---------------------------------------------------------------------- the helper
def _alive(pid: int) -> bool:
    if pid <= 0:   # (on POSIX 0 and below name process groups, never one process)
        return False
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x00100000 | 0x1000, False, pid)   # SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            return k.WaitForSingleObject(h, 0) == 0x102   # WAIT_TIMEOUT: still running
        finally:
            k.CloseHandle(h)
    try:   # (Linux) exited, but its parent hasn't collected it yet: a zombie still answers kill(pid, 0)
        with open(f"/proc/{pid}/stat", "rb") as f:
            if f.read().rsplit(b")", 1)[1].split()[0] == b"Z":
                return False
    except (OSError, IndexError):
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _end(pid: int) -> None:
    if pid <= 0:
        return
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x0001, False, pid)   # PROCESS_TERMINATE
        if h:
            k.TerminateProcess(h, 1)
            k.CloseHandle(h)
    else:
        import signal
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def wait_and_run(pid: int, port: int, cmd: list[str], cwd: str, console: bool = False, out: Path | None = None,
                 wait_exit: float = WAIT_EXIT, wait_port: float = WAIT_PORT) -> subprocess.Popen:
    def say(msg: str) -> None:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)

    t0 = time.monotonic()
    while _alive(pid) and time.monotonic() - t0 < wait_exit:
        time.sleep(0.2)
    if _alive(pid):
        say(f"pid {pid} did not exit within {wait_exit:.0f} s: ending it")
        _end(pid)
        time.sleep(1.0)
    else:
        say(f"pid {pid} exited after {time.monotonic() - t0:.1f} s")
    t1 = time.monotonic()
    while port and not port_free(port) and time.monotonic() - t1 < wait_port:
        time.sleep(0.25)
    if port and not port_free(port):
        say(f"port {port} is still taken; starting anyway (the app retries)")
    elif time.monotonic() - t1 > 0.5:
        say(f"port {port} free after {time.monotonic() - t1:.1f} s")
    kw: dict = {"cwd": cwd, "stdin": subprocess.DEVNULL, "close_fds": True}
    if os.name == "nt" and console:
        kw["creationflags"] = _NEW_CONSOLE
    else:
        sink = open(out, "w", encoding="utf-8") if out else sys.stdout   # noqa: SIM115 - (the new process keeps it)
        kw.update(stdout=sink, stderr=subprocess.STDOUT if out else sys.stderr)
        if os.name == "nt":
            kw["creationflags"] = _DETACHED
        else:
            kw["start_new_session"] = True
    p = subprocess.Popen(cmd, **kw)
    say(f"started pid {p.pid}: {subprocess.list2cmdline(cmd)}" + (f" (output: {out})" if out and not console else ""))
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jevh.restart",
                                 description="(started by the dashboard) run a command again once a process has exited")
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--cwd", default=None)
    ap.add_argument("--console", action="store_true", help="(Windows) run it in a new console window")
    ap.add_argument("--output", default=None, help="the file its output goes to (when not in a console)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command to run")
    wait_and_run(a.pid, a.port, cmd, a.cwd or os.getcwd(), a.console, Path(a.output) if a.output else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
