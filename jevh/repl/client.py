"""Host side of a worker REPL: spawn, exec with timeout, serve callbacks, kill."""
from __future__ import annotations

import collections
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..config import ROOT, shell_env
from ..procs import kill_tree as _kill_tree   # the REPL and everything it started (Playwright, Chromium, jobs)

Handler = Callable[..., Any]
START_TIMEOUT = 30.0   # seconds a new REPL may take to say it is ready (imports, a cold disk, an antivirus scan)


@dataclass
class ExecResult:
    output: str
    error: bool = False
    final: dict | None = None
    timed_out: bool = False
    usage: dict | None = None      # Jev calls the worker's tools made during this exec


def truncate(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    head = int(cap * 0.6)
    tail = cap - head
    return f"{text[:head]}\n... [{len(text) - cap} chars omitted; use peek() or slice] ...\n{text[-tail:]}"


class Repl:
    def __init__(self, cwd: Path, artifacts: Path, handlers: dict[str, Handler] | None = None,
                 tools: dict[str, str] | None = None, timeout: float = 120.0, output_cap: int = 4000,
                 env: dict[str, str] | None = None):
        self.cwd = Path(cwd)
        self.extra_env = dict(env or {})
        self.artifacts = Path(artifacts)
        self.handlers = handlers or {}
        self.tools = tools or {}
        self.timeout = timeout
        self.output_cap = output_cap
        self.proc: subprocess.Popen | None = None
        self._q: queue.Queue = queue.Queue()
        self.stderr_tail: collections.deque[str] = collections.deque(maxlen=200)

    # -- lifecycle -------------------------------------------------------------------
    def start(self) -> None:
        self.cwd.mkdir(parents=True, exist_ok=True)
        self.artifacts.mkdir(parents=True, exist_ok=True)
        env = dict(shell_env(), PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
                   JEVH_ARTIFACTS=str(self.artifacts),
                   PYTHONPATH=os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]), **self.extra_env)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen([sys.executable, "-u", "-m", "jevh.repl.server"], cwd=self.cwd,
                                     env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, creationflags=flags,
                                     start_new_session=os.name != "nt")
        self._q = queue.Queue()
        threading.Thread(target=self._read_stdout, args=(self.proc, self._q), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(self.proc,), daemon=True).start()
        try:
            msg = self._q.get(timeout=START_TIMEOUT)
        except queue.Empty:
            msg = None
        if not msg or msg.get("type") != "ready":
            # (a REPL that never said it was ready is killed, not left waiting on its stdin for the rest of the run)
            _kill_tree(self.proc)
            self.proc = None
            raise RuntimeError(f"REPL failed to start: {''.join(self.stderr_tail)[-1000:] or 'no reply'}")
        if self.tools:
            res = self._roundtrip({"op": "init", "tools": self.tools}, self.timeout)
            if res.error:
                self.stderr_tail.append(res.output)

    def close(self, kill: bool = False) -> None:
        proc = self.proc
        if proc and proc.poll() is None:
            try:
                if kill:
                    raise TimeoutError
                proc.stdin.close()
                proc.wait(timeout=3)
            except Exception:  # noqa: BLE001
                _kill_tree(proc)
        self.proc = None

    def __enter__(self) -> "Repl":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- ops -------------------------------------------------------------------------
    def set(self, name: str, value: Any) -> None:
        self._roundtrip({"op": "set", "name": name, "value": value}, self.timeout)

    def snapshot(self, path: Path) -> dict:
        """Save the worker's variables, imports and defs (for resume). {} if it failed."""
        return self._state_op("snapshot", path)

    def restore(self, path: Path) -> dict:
        return self._state_op("restore", path)

    def _state_op(self, op: str, path: Path) -> dict:
        res = self._roundtrip({"op": op, "path": str(path)}, min(self.timeout, 120))
        try:
            return json.loads(res.output) if not res.timed_out else {}
        except ValueError:
            return {}

    def run(self, code: str) -> ExecResult:
        res = self._roundtrip({"op": "exec", "code": code}, self.timeout)
        res.output = truncate(res.output, self.output_cap)
        return res

    # -- internals -------------------------------------------------------------------
    def _write(self, msg: dict) -> None:
        proc = self.proc   # (a Stop on another thread may clear self.proc at any moment)
        if proc is None:
            raise RuntimeError("the REPL was stopped")
        # "replace": a lone surrogate (half an emoji from a cut-off JSON string) in a callback's answer cannot be encoded
        # as UTF-8; strict, it raised here and failed the whole task over one character
        proc.stdin.write((json.dumps(msg, ensure_ascii=False, default=str) + "\n").encode("utf-8", "replace"))
        proc.stdin.flush()

    def _roundtrip(self, msg: dict, timeout: float) -> ExecResult:
        if self.proc is None or self.proc.poll() is not None:
            self.start()
        self._write(msg)
        budget = timeout
        while True:
            started = time.monotonic()
            try:
                reply = self._q.get(timeout=max(budget, 0.01))
            except queue.Empty:
                self.close(kill=True)
                self.start()
                return ExecResult(f"[timed out after {timeout:.0f}s; the REPL was restarted and all "
                                  f"variables were lost]", error=True, timed_out=True)
            budget -= time.monotonic() - started
            if reply is None:
                err = "".join(self.stderr_tail)[-1500:]
                if self.proc is not None:   # what it started (a browser, a job) must not outlive it
                    _kill_tree(self.proc)
                self.proc = None
                return ExecResult(f"[REPL process died]\n{err}", error=True)
            if reply.get("type") == "call":
                self._write(self._dispatch(reply))  # callback time is not charged to the budget
                continue
            return ExecResult(reply.get("output", ""), bool(reply.get("error")), reply.get("final"),
                              usage=reply.get("usage"))

    def _dispatch(self, call: dict) -> dict:
        fn = self.handlers.get(call.get("name", ""))
        if fn is None:
            return {"op": "reply", "value": None, "error": f"{call.get('name')}() is not available here"}
        try:
            return {"op": "reply", "value": fn(*call.get("args", []), **call.get("kwargs", {})),
                    "error": None}
        except Exception as e:  # noqa: BLE001 - surfaced to the worker's code as RuntimeError
            return {"op": "reply", "value": None, "error": f"{type(e).__name__}: {e}"}

    @staticmethod
    def _read_stdout(proc: subprocess.Popen, q: queue.Queue) -> None:
        try:
            for line in proc.stdout:
                try:
                    msg = json.loads(line.decode("utf-8"))
                except ValueError:   # not JSON, or not UTF-8 (UnicodeDecodeError): not a protocol line
                    continue
                if isinstance(msg, dict):
                    q.put(msg)
        except (OSError, ValueError):   # the pipe went away under us (a Stop killed the REPL)
            pass
        finally:
            # always: without it the host would wait the whole repl_timeout for a reply that can never come
            q.put(None)

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        for line in proc.stderr:
            self.stderr_tail.append(line.decode("utf-8", "replace"))
