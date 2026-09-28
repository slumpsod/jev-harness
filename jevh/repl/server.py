"""Worker REPL process. Run as `python -u -m jevh.repl.server`.

Protocol: JSON lines. The host writes ops to our stdin; we answer on a private dup of
the original stdout. fd 1 is then pointed at stderr, so prints from subprocesses the
worker's code starts cannot corrupt the protocol.

host -> repl   {"op": "init", "tools": {"alias": "module"}}
               {"op": "set", "name": ..., "value": ...}
               {"op": "exec", "code": ...}
               {"op": "reply", "value": ..., "error": str|null}      (answer to a call)
repl -> host   {"type": "ready"}
               {"type": "call", "name": ..., "args": [...], "kwargs": {...}}
               {"type": "result", "output": str, "error": bool, "final": {...}|null}
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import importlib
import io
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

_PROTO_OUT = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
os.dup2(2, 1)
_PROTO_IN = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
sys.stdin = io.StringIO("")
ARTIFACTS = Path(os.environ.get("JEVH_ARTIFACTS", "artifacts")).resolve()


def _send(msg: dict) -> None:
    _PROTO_OUT.write(json.dumps(msg, ensure_ascii=False, default=str) + "\n")
    _PROTO_OUT.flush()


def _recv() -> dict:
    line = _PROTO_IN.readline()
    if not line:
        sys.exit(0)
    return json.loads(line)


def _call(name: str, *args, **kwargs):
    _send({"type": "call", "name": name, "args": list(args), "kwargs": kwargs})
    msg = _recv()
    if msg.get("error"):
        raise RuntimeError(msg["error"])
    return msg.get("value")


def _no_input(*_a, **_k):
    raise RuntimeError("input() is not available; ask the user by finishing with final().")


builtins.input = _no_input

# ---------------------------------------------------------------- worker-facing API


def peek(x, n: int = 2000) -> str:
    """Preview any value without flooding your context."""
    s = x if isinstance(x, str) else repr(x)
    return s if len(s) <= n else f"{s[:n]}... [{len(s) - n} more chars]"


def llm_query(prompt: str, data=None) -> str:
    """Ask your own model a focused sub-question (e.g. about a slice of a big variable)."""
    return _call("llm_query", prompt, None if data is None else peek(data, 60000))


def jev(state, questions: dict) -> dict:
    """Fast typed decisions (noul/choice/score) from the System One model."""
    return _call("jev", state, questions)


def mcp(tool: str, **arguments) -> str:
    """Call an MCP tool listed in your prompt: mcp("server.tool", arg=value)."""
    return _call("mcp", tool, arguments)


def recall(query: str, k: int = 5) -> list:
    """Search ALL past conversations and remembered facts (not just what you were given);
    Jev keeps the snippets that match. -> ["[date persona] User: ... Assistant: ...", ...]"""
    return _call("recall", query, k)


def remember(fact: str) -> str:
    """Save a lasting fact about the user or their world for future conversations."""
    return _call("remember", fact)


def ask_user(question: str, options: list | None = None) -> str:
    """Ask the user something only they can answer and wait for the reply, keeping all your
    state (variables, an open browser page). options: short choices shown as buttons."""
    return _call("ask_user", question, options)


def request(persona: str, task: str, inputs: list | None = None) -> dict:
    """Hand a sub-task to another assistant and wait. Returns {status, answer, artifacts}."""
    return _call("request", persona, task, inputs or [])


def request_async(persona: str, task: str, inputs: list | None = None) -> str:
    """Start a sub-task without waiting; returns a handle for wait(). Several run in parallel."""
    return _call("request_async", persona, task, inputs or [])


def wait(handles, timeout: float | None = None):
    """Wait for one handle (-> result dict) or a list (-> list of dicts, same order).
    With a timeout, unfinished ones come back as {"status": "running"}."""
    return _call("wait", handles, timeout)


def request_all(jobs: list) -> list:
    """Run several hand-offs in parallel and wait for all:
    request_all([("front_end_designer", brief), ("comfyui_manager", brief2, [path])])"""
    return wait([request_async(*job) for job in jobs])


def save_artifact(name: str, data=None, path: str | None = None) -> str:
    """Save bytes/str/JSON-able data, or copy a file, into the shared artifact store."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    dest = ARTIFACTS / Path(name).name
    if path is not None and not dest.suffix and Path(path).suffix:
        dest = dest.with_suffix(Path(path).suffix)  # keep ".html" etc. so the file opens
    if path is not None:
        shutil.copyfile(path, dest)
    elif isinstance(data, bytes):
        dest.write_bytes(data)
    elif isinstance(data, str):
        dest.write_text(data, encoding="utf-8")
    else:
        dest.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return str(dest)


def load_artifact(ref: str, binary: bool = False):
    p = Path(ref)
    if not p.is_absolute():
        p = ARTIFACTS / ref
    return p.read_bytes() if binary else p.read_text(encoding="utf-8", errors="replace")


def list_artifacts() -> list[str]:
    return sorted(str(p) for p in ARTIFACTS.glob("*")) if ARTIFACTS.exists() else []


_FINAL: dict | None = None


def _resolve_artifact(a) -> str:
    """Workers often list a deliverable by the bare name they saved it under ("page.html"):
    point that at the artifact store, or copy a file from the working dir into it. (Seen live:
    the final() gate sent correct work back twice because "page.html" wasn't found as given.)"""
    p = Path(str(a))
    if p.is_absolute():
        return str(p)
    if (ARTIFACTS / p).exists():
        return str((ARTIFACTS / p).resolve())
    if p.exists():
        return save_artifact(p.name, path=str(p))
    return str(p)


def final(answer: str, artifacts: list | None = None) -> str:
    """Finish the task. `answer` goes back to whoever asked (user or another assistant)."""
    global _FINAL
    proj = sys.modules.get("jevh.tools.project")
    changed = proj.changed() if proj is not None and getattr(proj, "root", None) else []
    _FINAL = {"answer": str(answer), "artifacts": [_resolve_artifact(a) for a in (artifacts or [])],
              "changed": changed}
    return "[final answer recorded]"


G: dict = {"__name__": "__main__", "ctx": {}, "peek": peek, "llm_query": llm_query, "jev": jev,
           "request": request, "request_async": request_async, "wait": wait, "request_all": request_all,
           "save_artifact": save_artifact, "load_artifact": load_artifact,
           "list_artifacts": list_artifacts, "final": final, "mcp": mcp, "recall": recall,
           "remember": remember, "ask_user": ask_user}


def _exec(code: str) -> tuple[str, bool]:
    buf = io.StringIO()
    error = False
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            tree = ast.parse(code, "<cell>", "exec")
            last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            if last is not None:
                tree.body = tree.body[:-1]
            exec(compile(tree, "<cell>", "exec"), G)
            if last is not None:
                value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), G)
                if value is not None:
                    print(repr(value))
        except SystemExit:
            print("[exit() ignored; use final() to finish]")
        except BaseException:  # noqa: BLE001 - report everything to the model
            error = True
            etype, evalue, tb = sys.exc_info()
            frames = [f for f in traceback.extract_tb(tb) if f.filename == "<cell>"]
            lines = traceback.format_list(frames[-3:]) + traceback.format_exception_only(etype, evalue)
            print("".join(lines).rstrip())
    return buf.getvalue(), error


def main() -> None:
    global _FINAL
    _send({"type": "ready"})
    while True:
        msg = _recv()
        op = msg.get("op")
        if op == "init":
            out, err = [], False
            for alias, module in (msg.get("tools") or {}).items():
                try:
                    G[alias] = importlib.import_module(module)
                    sys.modules.setdefault(alias, G[alias])  # so `import comfy` works too
                except Exception as e:  # noqa: BLE001
                    out.append(f"tool {alias} ({module}) failed to import: {e}")
                    err = True
            _send({"type": "result", "output": "\n".join(out), "error": err, "final": None})
        elif op == "set":
            G[msg["name"]] = msg.get("value")
            _send({"type": "result", "output": "", "error": False, "final": None})
        elif op == "exec":
            _FINAL = None
            output, error = _exec(msg.get("code", ""))
            _send({"type": "result", "output": output, "error": error, "final": _FINAL})
        else:
            _send({"type": "result", "output": f"unknown op {op!r}", "error": True, "final": None})


if __name__ == "__main__":
    main()
