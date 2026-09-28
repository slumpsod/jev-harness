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

from ..config import ROOT

Handler = Callable[..., Any]


@dataclass
class ExecResult:
    output: str
    error: bool = False
    final: dict | None = None
    timed_out: bool = False


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
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
                   JEVH_ARTIFACTS=str(self.artifacts),
                   PYTHONPATH=os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]), **self.extra_env)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen([sys.executable, "-u", "-m", "jevh.repl.server"], cwd=self.cwd,
                                     env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, creationflags=flags)
        self._q = queue.Queue()
        threading.Thread(target=self._read_stdout, args=(self.proc, self._q), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(self.proc,), daemon=True).start()
        msg = self._q.get(timeout=30)
        if not msg or msg.get("type") != "ready":
            raise RuntimeError(f"REPL failed to start: {''.join(self.stderr_tail)[-1000:]}")
        if self.tools:
            res = self._roundtrip({"op": "init", "tools": self.tools}, self.timeout)
            if res.error:
                self.stderr_tail.append(res.output)

    def close(self, kill: bool = False) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                if kill:
                    raise TimeoutError
                self.proc.stdin.close()
                self.proc.wait(timeout=3)
            except Exception:  # noqa: BLE001
                self.proc.kill()
        self.proc = None

    def __enter__(self) -> "Repl":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- ops -------------------------------------------------------------------------
    def set(self, name: str, value: Any) -> None:
        self._roundtrip({"op": "set", "name": name, "value": value}, self.timeout)

    def run(self, code: str) -> ExecResult:
        res = self._roundtrip({"op": "exec", "code": code}, self.timeout)
        res.output = truncate(res.output, self.output_cap)
        return res

    # -- internals -------------------------------------------------------------------
    def _write(self, msg: dict) -> None:
        self.proc.stdin.write((json.dumps(msg, ensure_ascii=False, default=str) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

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
                self.proc = None
                return ExecResult(f"[REPL process died]\n{err}", error=True)
            if reply.get("type") == "call":
                self._write(self._dispatch(reply))  # callback time is not charged to the budget
                continue
            return ExecResult(reply.get("output", ""), bool(reply.get("error")), reply.get("final"))

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
        for line in proc.stdout:
            try:
                q.put(json.loads(line.decode("utf-8")))
            except json.JSONDecodeError:
                continue
        q.put(None)

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        for line in proc.stderr:
            self.stderr_tail.append(line.decode("utf-8", "replace"))
