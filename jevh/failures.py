"""Failure log: every way a task went wrong, collected in workspace/failures.db so the harness can
be improved from them later (prompts, thresholds, tools, skills). Logging never changes how a task
runs: it watches the harness's events and writes down what failed.

Where a failure comes from (`source`):
  harness  the harness saw it: a crash, the step limit, a stall, a reply cut off or empty, a worker
           going in circles, Jev unreachable, a hand-off refused, the user pressing Stop
  gate     the final() gate sent work back, or accepted it with problems still listed
  jev      after a task, Jev read the request, the answer and the work log and judged that it failed
           (config failures.jev_check)
  model    a worker called report_failure() in its REPL
  hook     a hook in config failures.hooks flagged it

A row keeps what is needed to study it without the session: the request, the answer, the work log,
persona and model, and the task's working folder (its checkpoint.json has the whole conversation).
`status` (open | reviewed | fixed | ignored) and `note` are for triage.

Hooks (config failures.hooks) run on harness events and decide for themselves what a failure is:

    failures:
      hooks:
        - name: no_sources                    # the failure kind when the hook doesn't give one
          on: [done]                          # event kinds ("*" = every event)
          command: python checks/sources.py   # gets the event as JSON on stdin
          timeout: 30
        - on: [output]
          python: checks.repl:bad_output      # fn(event) -> None | str | dict | list

The event is {"event", "session", "turn", "task", "persona", "data"}; a "done" event's data also
carries "work_log". A hook flags a failure by printing (or returning) a summary string, a
{"kind", "summary", "detail", "scores"} object, a list of them, or {"failures": [...]}; nothing
printed means no failure. A command can also exit with code 2 and print a summary as plain text.
Any other non-zero exit, or an exception, is the hook's own error: shown, never logged as a failure.

    python -m jevh.failures                          # recent failures and counts by kind
    python -m jevh.failures --kind gate_bounce -n 50
    python -m jevh.failures export failures.jsonl    # everything, one JSON object per line
    python -m jevh.failures status 12 fixed "gate wording changed"
    python -m jevh.failures add bad_artifact "handed over the file name, not the file" --workdir <task folder>
"""
from __future__ import annotations

import argparse
import importlib
import json
import random
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from . import procs, store
from .bus import Task, jsonable
from .config import shell_env
from .decide import noul

STATUSES = ("open", "reviewed", "fixed", "ignored")
# How a task ended -> the failure kind (final and plain are the normal ends)
STOPS = {"error": ("crash", "the worker crashed"),
         "max_steps": ("step_limit", "reached its step limit without calling final()"),
         "stalled": ("stalled", "stopped for going in circles"),
         "cancelled": ("stopped_by_user", "the user stopped the task")}
# The Jev check after a task, on evals/failure_cases.py (semif, 21 cases): 0.60-0.90 all catch 11/11 failures with
# 0/10 good answers flagged. It sees the files handed over: asked only about the answer text, 2 of 5 correct
# "give me a CSV file" runs were logged as missed_request (0.89, 0.97; live 2026-09-30).
GAVE_UP_Q = "Does `answer` say that `request` could not be done, or was only partly done?"
MISSED_Q = "Do `answer` and the files in `artifacts` (handed over with it) fail to deliver what `request` asked for?"
CAP = 8000   # chars kept of each text field


class FailureLog:
    """The failure rows, in SQLite (thread-safe)."""

    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db = store.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("""CREATE TABLE IF NOT EXISTS failures(
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, session TEXT, turn INTEGER, task INTEGER,
            persona TEXT, model TEXT, source TEXT, kind TEXT, summary TEXT, detail TEXT, request TEXT,
            answer TEXT, work_log TEXT, workdir TEXT, scores TEXT, status TEXT DEFAULT 'open', note TEXT DEFAULT '')""")
        self.db.execute("CREATE INDEX IF NOT EXISTS failures_kind ON failures(kind, ts)")
        self.db.commit()

    def record(self, *, source: str, kind: str, summary: str, session: str = "", turn: int | None = None,
               task: int | None = None, persona: str = "", model: str = "", detail: str = "", request: str = "",
               answer: str = "", work_log: str = "", workdir: str = "", scores: dict | None = None) -> dict[str, Any]:
        row = {"ts": time.time(), "session": session, "turn": turn, "task": task, "persona": persona,
               "model": model, "source": source, "kind": kind, "summary": str(summary)[:500],
               "detail": str(detail)[:CAP], "request": str(request)[:CAP], "answer": str(answer)[:CAP],
               "work_log": str(work_log)[-CAP:], "workdir": str(workdir or ""),
               "scores": json.dumps(scores or {}, default=str)}
        with self._lock:
            cur = self.db.execute(f"INSERT INTO failures({', '.join(row)}) VALUES ({', '.join(':' + k for k in row)})",
                                  row)
            self.db.commit()
        return {"id": cur.lastrowid, **row, "scores": scores or {}, "status": "open", "note": ""}

    def list(self, session: str | None = None, kind: str | None = None, source: str | None = None,
             status: str | None = None, since: float | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Newest first."""
        where, args = [], []
        for col, val in (("session", session), ("kind", kind), ("source", source), ("status", status)):
            if val is not None:
                where.append(f"{col}=?")
                args.append(val)
        if since is not None:
            where.append("ts>=?")
            args.append(since)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        with self._lock:
            rows = self.db.execute(f"SELECT * FROM failures {clause} ORDER BY id DESC LIMIT ?", [*args, limit]).fetchall()
        return [self._row(r) for r in rows]

    def get(self, fid: int) -> dict[str, Any] | None:
        with self._lock:
            r = self.db.execute("SELECT * FROM failures WHERE id=?", (fid,)).fetchone()
        return self._row(r) if r else None

    def counts(self, session: str | None = None, since: float | None = None) -> dict[str, dict[str, int]]:
        """Open failures by kind, source and persona."""
        where, args = ["status='open'"], []
        if session is not None:
            where.append("session=?")
            args.append(session)
        if since is not None:
            where.append("ts>=?")
            args.append(since)
        clause = "WHERE " + " AND ".join(where)
        with self._lock:
            return {f"by_{g}": {(r[0] or "-"): r[1] for r in self.db.execute(
                f"SELECT {g}, COUNT(*) FROM failures {clause} GROUP BY {g} ORDER BY COUNT(*) DESC", args)}
                for g in ("kind", "source", "persona")}

    def set_status(self, fid: int, status: str, note: str | None = None) -> bool:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {', '.join(STATUSES)}")
        with self._lock:
            cur = self.db.execute("UPDATE failures SET status=?, note=COALESCE(?, note) WHERE id=?", (status, note, fid))
            self.db.commit()
        return cur.rowcount > 0

    def export(self, path: Path, status: str | None = None) -> int:
        """Every row (or those with `status`), oldest first, as JSON lines. -> rows written."""
        rows = list(reversed(self.list(status=status, limit=10 ** 9)))
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return len(rows)

    @staticmethod
    def _row(r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        try:
            d["scores"] = json.loads(d.get("scores") or "{}")
        except ValueError:
            d["scores"] = {}
        return d

    def close(self) -> None:
        with self._lock:
            self.db.close()


class HookError(RuntimeError):
    pass


class Hook:
    def __init__(self, spec: dict[str, Any], root: Path):
        on = spec.get("on") or ["done"]
        self.on = {on} if isinstance(on, str) else set(on)
        self.command = spec.get("command")
        self.python = spec.get("python")
        if bool(self.command) == bool(self.python):
            raise ValueError(f"failure hook needs exactly one of command / python: {spec}")
        self.name = str(spec.get("name") or self.command or self.python)
        self.kind = str(spec.get("name") or "hook")
        self.timeout = float(spec.get("timeout", 30))
        self.root = root
        self._fn: Callable | None = None

    def wants(self, kind: str) -> bool:
        return "*" in self.on or kind in self.on

    def __call__(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        if self.python:
            return as_failures(self._func()(event))
        try:   # (procs.run: a hook that starts something in the background must not hang the turn past its timeout)
            p = procs.run(self.command, shell=True, input=json.dumps(event, default=str), timeout=self.timeout,
                          cwd=self.root, env=shell_env())
        except subprocess.TimeoutExpired:
            raise HookError(f"timed out after {self.timeout:.0f}s") from None
        out = p.stdout.strip()
        if p.returncode == 2:   # "this is a failure", with the summary as text (or JSON)
            found = as_failures(_json_or_text(out)) or [{"summary": out or p.stderr.strip() or self.name}]
            return found
        if p.returncode != 0:
            raise HookError(f"exit {p.returncode}: {(p.stderr or out).strip()[-300:]}")
        return as_failures(_json_or_text(out)) if out else []

    def _func(self) -> Callable:
        if self._fn is None:
            mod, _, fn = str(self.python).partition(":")
            if str(self.root) not in sys.path:   # hook modules can live next to config.yaml
                sys.path.insert(0, str(self.root))
            self._fn = getattr(importlib.import_module(mod), fn or "check")
        return self._fn


def _json_or_text(out: str) -> Any:
    try:
        return json.loads(out)
    except ValueError:
        return out


def as_failures(out: Any) -> list[dict[str, Any]]:
    """What a hook gave back -> failure dicts (kind, summary, detail, scores); [] = no failure."""
    if not out:
        return []
    if isinstance(out, str):
        return [{"summary": out.strip()}] if out.strip() else []
    if isinstance(out, (list, tuple)):
        return [f for x in out for f in as_failures(x)]
    if isinstance(out, dict):
        if "failures" in out:
            return as_failures(out["failures"])
        if "failure" in out:
            return as_failures(out["failure"])
        if out.get("summary") or out.get("kind"):
            f = {k: out[k] for k in ("kind", "summary", "detail", "scores") if out.get(k)}
            f.setdefault("summary", f.get("kind", ""))
            return [f]
        return []
    return [{"summary": str(out)}]


class FailureWatch:
    """Reads every harness event, logs what failed, and runs the hooks.

    `where()`: the calling thread's {turn, persona, task}; `lookup(task_id)`: that task on the board;
    `ask(step, state, questions)`: a Jev call (the Jev check); `emit(kind, data)`: to the UI."""

    def __init__(self, log: FailureLog, settings: dict[str, Any], session: str, where: Callable[[], dict],
                 lookup: Callable[[int], Task | None], ask: Callable | None, emit: Callable[[str, dict], None],
                 thresholds: Callable[[str], float], root: Path, spawn: Callable | None = None,
                 audit_llm: Callable | None = None):
        """`spawn(kind, fn, task)`: run fn in the background for that task (None: inline);
        `audit_llm(persona, own_llm)`: the model a spot check uses (failures.audit_model)."""
        self.log, self.session = log, session
        self.where, self.lookup, self.ask, self.emit, self.t = where, lookup, ask, emit, thresholds
        self.enabled = bool(settings.get("enabled", True))
        self.jev_check = bool(settings.get("jev_check", True)) and ask is not None
        self.audit_rate = float(settings.get("audit") or 0)   # share of finished lead tasks spot-checked
        self.spawn, self.audit_llm = spawn, audit_llm
        self._rng = random.Random()
        self.hooks = [Hook(h, root) for h in settings.get("hooks") or []]
        self._models: dict[tuple, str] = {}   # (turn, task id) -> model name, from "start" events

    # -- recording ---------------------------------------------------------------------
    def record(self, source: str, kind: str, summary: str, task: Task | None = None, **extra) -> dict | None:
        if not self.enabled:
            return None
        w = self.where()
        tid = task.id if task is not None else w.get("task")
        task = task or (self.lookup(tid) if tid is not None else None)
        extra.setdefault("persona", task.to if task is not None else w.get("persona", ""))
        extra.setdefault("model", self._models.get((w.get("turn"), tid), ""))
        extra.setdefault("request", task.text if task is not None else "")
        if task is not None and task.checkpoint:
            extra.setdefault("workdir", str(Path(task.checkpoint).parent))
        try:
            row = self.log.record(source=source, kind=kind, summary=summary, session=self.session,
                                  turn=w.get("turn"), task=tid, **extra)
        except Exception as e:  # noqa: BLE001 - the log must never break a turn
            print(f"failure log: {type(e).__name__}: {e}", file=sys.stderr)
            return None
        self.emit("failure", {"failure": {k: row[k] for k in ("id", "source", "kind", "summary", "persona", "task")}})
        return row

    def report(self, task: Task, what: str, detail: str = "") -> str:
        """A worker's report_failure()."""
        if not self.enabled:
            return "not logged: failure logging is off"
        row = self.record("model", "reported", str(what), task, detail=str(detail or ""))
        return f"logged failure #{row['id']}; carry on with the task" if row else "not logged"

    # -- events --------------------------------------------------------------------------
    def on_event(self, kind: str, data: dict) -> None:
        if not self.enabled or kind in ("failure", "hook_error"):
            return
        try:
            self._builtin(kind, data)
        except Exception as e:  # noqa: BLE001
            print(f"failure watch ({kind}): {type(e).__name__}: {e}", file=sys.stderr)
        for hook in self.hooks:
            if hook.wants(kind):
                self._run_hook(hook, kind, data)

    def _task(self, data: dict) -> Task | None:
        """The event's task: harness events carry it, worker events its id."""
        if isinstance(data.get("task"), Task):
            return data["task"]
        return self.lookup(data["task_id"]) if data.get("task_id") is not None else None

    def _builtin(self, kind: str, data: dict) -> None:
        task = self._task(data)
        if kind == "start" and task is not None:
            self._models[(self.where().get("turn"), task.id)] = str(data.get("model", ""))
        elif kind == "verify" and not data.get("ok"):
            problems = [str(p) for p in data.get("problems") or []]
            first = problems[0].splitlines()[0] if problems else "problems found"
            self.record("gate", "gate_bounce" if data.get("bounced") else "gate_unresolved",
                        f"final() {'sent back' if data.get('bounced') else 'accepted with problems'} at step "
                        f"{data.get('step')}: {first}", task, detail="\n".join(problems),
                        answer=data.get("answer", ""), scores=data.get("scores") or {})
        elif kind == "progress_check" and data.get("stuck"):
            self.record("harness", "going_in_circles", f"progress check at step {data.get('step')}: repeating the "
                        f"same attempt (strike {data.get('strike')})", task, scores={"circles": data.get("circles")})
        elif kind == "output" and data.get("error"):
            out = str(data.get("output", ""))
            if out.startswith("(reply cut off"):
                self.record("harness", "reply_cut_off", f"step {data.get('step')}: reply cut off before its code "
                            "was complete", task)
            elif out.startswith("(empty reply"):
                self.record("harness", "empty_reply", f"step {data.get('step')}: reply was only thinking, no text",
                            task)
        elif kind == "jev_error":
            self.record("harness", "jev_unavailable", f"Jev unavailable during {data.get('step')}", task,
                        detail=str(data.get("error", "")))
        elif kind == "request" and data.get("approved") is False and task is not None:
            parent = task.parent
            self.record("harness", "handoff_refused", f"{task.requester} -> {task.to} refused: {task.answer}",
                        parent, detail=task.text)
        elif kind == "done" and task is not None:
            self._done(task, data.get("result"))

    def _done(self, task: Task, result) -> None:
        if result is None:
            return
        from .learn import work_log   # (learn imports the worker lazily; kept lazy here too)
        log = work_log(result.transcript, answer=result.answer) if result.transcript else ""
        model = (getattr(result.llm, "_model", "") or getattr(result.llm, "name", "")) if result.llm else ""
        common = dict(answer=result.answer, work_log=log, **({"model": model} if model else {}))
        if result.stopped in STOPS:
            kind, what = STOPS[result.stopped]
            self.record("harness", kind, f"{task.to} {what} (step {result.steps})", task, **common)
            return
        if not self.jev_check:
            self._maybe_audit(task, result, model)
            return
        try:
            r = self.ask("failure_check", {"request": task.text[:2000], "answer": result.answer[:3000],
                                           "artifacts": [Path(a).name for a in result.artifacts] or "(none)",
                                           "work_log": (log or "(no code was run)")[-4000:]},
                         {"gave_up": noul(GAVE_UP_Q), "missed": noul(MISSED_Q)})
            scores = {"gave_up": round(r["gave_up"].noul, 3), "missed": round(r["missed"].noul, 3)}
        except Exception:  # noqa: BLE001 - Jev down: no check this time (the outage itself is logged elsewhere)
            return
        if scores["gave_up"] >= self.t("failure_gave_up"):
            self.record("jev", "gave_up", f"{task.to}'s answer says the request was not (fully) done", task,
                        scores=scores, **common)
        elif scores["missed"] >= self.t("failure_missed"):
            self.record("jev", "missed_request", f"{task.to}'s answer doesn't deliver what was asked", task,
                        scores=scores, **common)
        else:
            self._maybe_audit(task, result, model)

    def _maybe_audit(self, task: Task, result, model: str) -> None:
        """A spot check of a lead task that ended normally and nothing flagged: a model rereads it all
        (jevh/audit.py) and a wrong answer is logged as wrong_answer. In the background, on a sample."""
        if not (self.audit_rate and task.parent is None and result.llm is not None and result.answer
                and self._rng.random() < self.audit_rate):
            return
        from .audit import audit
        from .learn import work_log
        log = work_log(result.transcript, cap=12000, answer=result.answer) if result.transcript else ""
        llm = self.audit_llm(task.to, result.llm) if self.audit_llm else result.llm

        def run() -> None:
            try:
                v = audit(llm, task.text, result.answer, list(result.artifacts or []), log)
            except Exception as e:  # noqa: BLE001 - a spot check never breaks anything
                print(f"spot check: {type(e).__name__}: {e}", file=sys.stderr)
                return
            self.emit("audit", {"task_id": task.id, "persona": task.to, "verdict": v["verdict"],
                                "problems": v["problems"][:3]})
            if v["verdict"] == "wrong":
                first = (v["problems"] or ["the answer is wrong"])[0]
                self.record("audit", "wrong_answer", f"{task.to}'s answer is wrong: {first}"[:500], task,
                            detail="\n".join(v["problems"]), answer=result.answer, work_log=log,
                            **({"model": model} if model else {}))
        if self.spawn is not None:
            self.spawn("audit", run, task)
        else:
            run()

    def _run_hook(self, hook: Hook, kind: str, data: dict) -> None:
        w = self.where()
        payload = {k: v for k, v in data.items() if k != "result"}
        task = self._task(data)
        result = data.get("result")
        if result is not None and hasattr(result, "transcript"):
            from .learn import work_log
            payload["result"] = result
            payload["work_log"] = work_log(result.transcript, answer=result.answer) if result.transcript else ""
        event = {"event": kind, "session": self.session, "turn": w.get("turn"),
                 "task": task.id if task is not None else w.get("task"),
                 "persona": task.to if task is not None else data.get("persona", w.get("persona", "")),
                 "data": jsonable(payload)}
        try:
            found = hook(event)
        except Exception as e:  # noqa: BLE001 - a broken hook is shown, never logged as the task's failure
            self.emit("hook_error", {"hook": hook.name, "event": kind, "error": f"{type(e).__name__}: {e}"})
            print(f"failure hook {hook.name} ({kind}): {type(e).__name__}: {e}", file=sys.stderr)
            return
        for f in found:
            scores = f.get("scores") if isinstance(f.get("scores"), dict) else {}
            self.record("hook", str(f.get("kind") or hook.kind), str(f.get("summary") or hook.kind), task,
                        detail=str(f.get("detail", "")), scores=scores,
                        **({"answer": result.answer} if result is not None and hasattr(result, "answer") else {}),
                        **({"work_log": payload["work_log"]} if payload.get("work_log") else {}))


# ---------------------------------------------------------------- command line

def from_workdir(workdir: Path) -> dict[str, Any]:
    """What a task's folder tells about it (its checkpoint.json): request, persona, session, work log,
    and the answer it handed to final() - to log a failure that no detector caught."""
    from .learn import work_log
    cp = json.loads((Path(workdir) / "checkpoint.json").read_text(encoding="utf-8"))
    msgs = cp.get("messages") or []
    answer = ""
    for m in msgs:
        for call in m.get("tool_calls") or []:
            fn = call.get("function") or {}
            if fn.get("name") == "final":
                try:
                    answer = str(json.loads(fn.get("arguments") or "{}").get("answer") or answer)
                except ValueError:
                    pass
    return {"request": cp.get("task_text", ""), "persona": cp.get("persona", ""), "answer": answer,
            "session": Path(workdir).resolve().parent.name, "workdir": str(Path(workdir).resolve()),
            # (the checkpoint keeps no system message; work_log skips the first two)
            "work_log": work_log([{"role": "system", "content": ""}, *msgs], answer=answer or None) if msgs else ""}


def main(argv: list[str] | None = None) -> int:
    from .config import load_config
    ap = argparse.ArgumentParser(prog="jevh failures", description="The harness's failure log.")
    ap.add_argument("--config")
    ap.add_argument("--kind")
    ap.add_argument("--source")
    ap.add_argument("--status", default=None)
    ap.add_argument("-n", type=int, default=20)
    sub = ap.add_subparsers(dest="cmd")
    ex = sub.add_parser("export", help="write failures as JSON lines")
    ex.add_argument("path")
    st = sub.add_parser("status", help="mark a failure reviewed / fixed / ignored / open")
    st.add_argument("id", type=int)
    st.add_argument("status", choices=STATUSES)
    st.add_argument("note", nargs="?")
    ad = sub.add_parser("add", help="log a failure no detector caught (source: user)")
    ad.add_argument("kind", help="a short slug, e.g. bad_artifact")
    ad.add_argument("summary")
    ad.add_argument("--workdir", help="the task's folder (has checkpoint.json): fills request, persona, work log")
    for opt in ("request", "answer", "persona", "model", "detail", "session"):
        ad.add_argument(f"--{opt}")
    a = ap.parse_args(argv)
    log = FailureLog(load_config(a.config).workspace / "failures.db")
    if a.cmd == "add":
        row = from_workdir(Path(a.workdir)) if a.workdir else {}
        row.update({k: getattr(a, k) for k in ("request", "answer", "persona", "model", "detail", "session")
                    if getattr(a, k) is not None})
        if not (row.get("request") or "").strip():
            print("give --request or --workdir: refine replays the request")
            return 1
        f = log.record(source="user", kind=a.kind, summary=a.summary, **row)
        print(f"#{f['id']} {a.kind} logged")
        return 0
    if a.cmd == "export":
        print(f"{log.export(Path(a.path), a.status)} failures -> {a.path}")
        return 0
    if a.cmd == "status":
        ok = log.set_status(a.id, a.status, a.note)
        print(f"#{a.id} -> {a.status}" if ok else f"no failure #{a.id}")
        return 0 if ok else 1
    for f in reversed(log.list(kind=a.kind, source=a.source, status=a.status, limit=a.n)):
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(f["ts"]))
        print(f"#{f['id']:<5} {when}  {f['source']:<7} {f['kind']:<18} {f['persona'] or '-':<18} "
              f"[{f['status']}] {f['summary'][:110]}")
    c = log.counts()
    if c["by_kind"]:
        print("\nopen, by kind: " + ", ".join(f"{k} {n}" for k, n in c["by_kind"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
