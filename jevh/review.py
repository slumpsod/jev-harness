"""Session review: flag a session for a closer look, write notes on what looked wrong, and have a model
read the whole session against those notes to find the problems and say whether it was a failure.

workspace/reviews.db
  flags  one row per flagged session: status (open | decided | dismissed), the verdict (failure |
         not_failure), who gave it (model | user) and why, and the model's last review
  notes  the user's notes on a flag, oldest first (a note can point at one turn)

What the reviewing model reads (session_record()): the notes, the conversation (every turn in memory),
each task's request, how it ended and its work log (from the task folders under
workspace/sessions/<session>/), and the failures the harness already logged for the session. It answers
with the problems it found, which note each one bears on, and a verdict. A failure verdict, the model's
or the user's, is also written to the failure log (source "review"), so refinement works on it like any
other failure; overruling it with not_failure marks that row ignored.

The reviewer is config review.model ("provider:model" or a model block), else models.strong, else the
default persona's model.

    python -m jevh.review                                   # flagged sessions
    python -m jevh.review sessions                          # recent sessions, to pick one to flag
    python -m jevh.review flag <session> "the totals were wrong" [--turn 2]
    python -m jevh.review note <id|session> "it also never cited a source" [--turn 3]
    python -m jevh.review show <id|session>
    python -m jevh.review run <id|session> [--model lmstudio:qwen3.8-27b]
    python -m jevh.review verdict <id|session> failure|not_failure ["why"]
    python -m jevh.review dismiss <id|session>
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any

STATUSES = ("open", "decided", "dismissed")
VERDICTS = ("failure", "not_failure")
TASK_DIR = re.compile(r"^u(\d+)t(\d+)-(.+)$")   # a task's folder: u<turn>t<task>-<persona>
CAP = 8000   # chars kept of a model's raw reply
NORMAL_ENDS = ("final", "plain")


class ReviewError(RuntimeError):
    pass


class ReviewLog:
    """Flagged sessions and their notes, in SQLite (thread-safe)."""

    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS flags(id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT UNIQUE, ts REAL,
                updated REAL, title TEXT DEFAULT '', status TEXT DEFAULT 'open', verdict TEXT DEFAULT '',
                verdict_by TEXT DEFAULT '', reason TEXT DEFAULT '', review TEXT DEFAULT '', failure_id INTEGER);
            CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY AUTOINCREMENT, flag_id INTEGER, ts REAL,
                turn INTEGER, text TEXT);
            CREATE INDEX IF NOT EXISTS notes_flag ON notes(flag_id, id);
        """)
        self.db.commit()

    # -- flags ---------------------------------------------------------------------------
    def flag(self, session: str, note: str = "", turn: int | None = None, title: str = "") -> dict[str, Any]:
        """Flag `session` (or return its flag; a dismissed one opens again), with an optional first note."""
        session = str(session).strip()
        if not session:
            raise ValueError("no session to flag")
        now = time.time()
        with self._lock:
            row = self.db.execute("SELECT id, status, title FROM flags WHERE session=?", (session,)).fetchone()
            if row is None:
                fid = self.db.execute("INSERT INTO flags(session, ts, updated, title) VALUES (?,?,?,?)",
                                      (session, now, now, str(title)[:300])).lastrowid
            else:
                fid = row["id"]
                self.db.execute("UPDATE flags SET updated=?, status=CASE status WHEN 'dismissed' THEN 'open' "
                                "ELSE status END, title=CASE title WHEN '' THEN ? ELSE title END WHERE id=?",
                                (now, str(title)[:300], fid))
            self.db.commit()
        if str(note).strip():
            self.add_note(fid, note, turn)
        return self.get(fid)

    def add_note(self, flag_id: int, text: str, turn: int | None = None) -> dict[str, Any]:
        text = str(text).strip()
        if not text:
            raise ValueError("empty note")
        now = time.time()
        with self._lock:
            if not self.db.execute("SELECT 1 FROM flags WHERE id=?", (flag_id,)).fetchone():
                raise KeyError(f"no flag #{flag_id}")
            nid = self.db.execute("INSERT INTO notes(flag_id, ts, turn, text) VALUES (?,?,?,?)",
                                  (flag_id, now, turn, text)).lastrowid
            self.db.execute("UPDATE flags SET updated=? WHERE id=?", (now, flag_id))
            self.db.commit()
        return {"id": nid, "flag_id": flag_id, "ts": now, "turn": turn, "text": text}

    def get(self, flag_id: int) -> dict[str, Any] | None:
        with self._lock:
            r = self.db.execute("SELECT * FROM flags WHERE id=?", (flag_id,)).fetchone()
        return self._full(r) if r else None

    def find(self, session: str) -> dict[str, Any] | None:
        with self._lock:
            r = self.db.execute("SELECT * FROM flags WHERE session=?", (str(session),)).fetchone()
        return self._full(r) if r else None

    def resolve(self, ref: str | int) -> dict[str, Any] | None:
        """A flag by its id or its session name."""
        ref = str(ref).strip().lstrip("#")
        return (self.get(int(ref)) if ref.isdigit() else None) or self.find(ref)

    def list(self, status: str | None = None, verdict: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Most recently changed first, each with its notes."""
        where, args = [], []
        for col, val in (("status", status), ("verdict", verdict)):
            if val is not None:
                where.append(f"{col}=?")
                args.append(val)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        with self._lock:
            rows = self.db.execute(f"SELECT * FROM flags {clause} ORDER BY updated DESC, id DESC LIMIT ?",
                                   [*args, limit]).fetchall()
        return [self._full(r) for r in rows]

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self.db.execute("SELECT CASE status WHEN 'decided' THEN verdict ELSE status END, COUNT(*) "
                                   "FROM flags GROUP BY 1").fetchall()
        return {r[0]: r[1] for r in rows}

    def set_review(self, flag_id: int, review: dict[str, Any]) -> dict[str, Any]:
        """The model's review; its verdict becomes the flag's unless the user already decided."""
        now = time.time()
        with self._lock:
            self.db.execute("UPDATE flags SET review=?, updated=? WHERE id=?",
                            (json.dumps(review, ensure_ascii=False, default=str), now, flag_id))
            self.db.execute("UPDATE flags SET verdict=?, verdict_by='model', reason=?, status='decided' "
                            "WHERE id=? AND verdict_by<>'user'", (review["verdict"], review.get("summary", ""), flag_id))
            self.db.commit()
        return self.get(flag_id)

    def set_verdict(self, flag_id: int, verdict: str, by: str = "user", reason: str = "") -> dict[str, Any]:
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {', '.join(VERDICTS)}")
        with self._lock:
            cur = self.db.execute("UPDATE flags SET verdict=?, verdict_by=?, reason=?, status='decided', updated=? "
                                  "WHERE id=?", (verdict, by, str(reason), time.time(), flag_id))
            self.db.commit()
        if not cur.rowcount:
            raise KeyError(f"no flag #{flag_id}")
        return self.get(flag_id)

    def set_status(self, flag_id: int, status: str) -> bool:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {', '.join(STATUSES)}")
        with self._lock:
            cur = self.db.execute("UPDATE flags SET status=?, updated=? WHERE id=?", (status, time.time(), flag_id))
            self.db.commit()
        return cur.rowcount > 0

    def link_failure(self, flag_id: int, failure_id: int | None) -> None:
        with self._lock:
            self.db.execute("UPDATE flags SET failure_id=? WHERE id=?", (failure_id, flag_id))
            self.db.commit()

    def _full(self, r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        try:
            d["review"] = json.loads(d.get("review") or "null")
        except ValueError:
            d["review"] = None
        with self._lock:
            d["notes"] = [dict(n) for n in self.db.execute(
                "SELECT id, ts, turn, text FROM notes WHERE flag_id=? ORDER BY id", (d["id"],))]
        return d

    def close(self) -> None:
        with self._lock:
            self.db.close()


# ---------------------------------------------------------------- what happened in a session

def session_tasks(workspace: Path, session: str) -> list[dict[str, Any]]:
    """Every task of the session from its folder's checkpoint.json, in turn and task order."""
    from .failures import from_workdir
    from .worker import FIRST_STEP_NUDGE
    root = Path(workspace) / "sessions" / session
    out = []
    for d in root.iterdir() if root.is_dir() else []:
        m = TASK_DIR.match(d.name)
        if not m or not (d / "checkpoint.json").is_file():
            continue
        try:
            info = from_workdir(d)
            cp = json.loads((d / "checkpoint.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        request = info["request"].split("\n\n(In your REPL:")[0]
        request = request[:-len(FIRST_STEP_NUDGE)] if FIRST_STEP_NUDGE and request.endswith(FIRST_STEP_NUDGE) else request
        out.append({"turn": int(m[1]), "task": int(m[2]), "persona": info["persona"] or m[3],
                    "request": request, "answer": info["answer"],
                    # how it ended, when not normally (final() or a plain reply): step limit, stalled, crash...
                    "stopped": "" if cp.get("stopped") in NORMAL_ENDS else
                    cp.get("stopped") or ("" if cp.get("finished") else "interrupted"),
                    "steps": cp.get("step"), "work_log": info["work_log"], "workdir": str(d)})
    return sorted(out, key=lambda t: (t["turn"], t["task"]))


def session_record(workspace: Path, memory, failures, session: str) -> dict[str, Any]:
    """What a reviewer needs to judge a session: its conversation, its tasks and their work logs, and the
    failures the harness logged for it."""
    tasks = session_tasks(workspace, session)
    # memory keeps no turn number, and a crashed turn isn't in memory: number each remembered turn by the
    # lead task (task 1) whose request it is, so turns, task folders and the failure log agree
    leads = sorted((t["turn"], t["request"].strip()) for t in tasks if t["task"] == 1)
    turns, n = [], 1
    for t in memory.session_turns(session, 10 ** 6):
        head = t.user.strip()[:200]
        n = next((k for k, req in leads if k >= n and req.startswith(head)), n)
        turns.append({"n": n, "persona": t.persona, "user": t.user, "assistant": t.assistant, "ts": t.ts})
        n += 1
    logged = [{k: f.get(k) for k in ("id", "turn", "task", "persona", "source", "kind", "summary", "status")}
              for f in failures.list(session=session, limit=200)][::-1]
    return {"session": session, "turns": turns, "tasks": tasks, "failures": logged}


def list_sessions(workspace: Path, memory, limit: int = 30) -> list[dict[str, Any]]:
    """Recent sessions (newest first) with their first message and turn count, from memory and the
    session folders (a session whose turns all failed is only in the folders)."""
    with memory._lock:
        rows = memory.db.execute("SELECT session, COUNT(*), MIN(ts), MAX(ts) FROM turns GROUP BY session").fetchall()
        firsts = dict(memory.db.execute("SELECT session, user FROM turns WHERE id IN "
                                        "(SELECT MIN(id) FROM turns GROUP BY session)").fetchall())
    found = {s: {"session": s, "turns": n, "started": a, "last": b, "title": (firsts.get(s) or "")[:200]}
             for s, n, a, b in rows}
    sessions = Path(workspace) / "sessions"
    for d in sessions.iterdir() if sessions.is_dir() else []:
        if d.is_dir() and d.name not in found and any(TASK_DIR.match(x.name) for x in d.iterdir()):
            st = d.stat().st_mtime
            found[d.name] = {"session": d.name, "turns": 0, "started": st, "last": st, "title": ""}
    return sorted(found.values(), key=lambda s: -(s["last"] or 0))[:limit]


def session_title(record: dict[str, Any]) -> str:
    if record["turns"]:
        return record["turns"][0]["user"][:200]
    return record["tasks"][0]["request"][:200] if record["tasks"] else ""


# ---------------------------------------------------------------- the model's review

PROMPT = """You are reviewing one session of an AI assistant harness (jev-harness: a router picks an assistant \
persona, the worker model acts by running Python in a REPL with tools, hands work to other personas, and \
calls final() with its answer; a gate checks the answer before it goes out). The user flagged this session \
for review and wrote the notes below about what they think went wrong. Read the whole session, find the \
problems - first the ones the notes point at, then any others - and decide whether the session was a failure.

A session is a failure when the user did not get what they asked for: a wrong, made-up or incomplete answer \
presented as done, a file that is missing or broken, an action that did the wrong thing, or a task that \
stopped without finishing. It is not a failure when the result was right and complete, even if the work \
took detours, or when the assistant plainly and correctly said what it could not do and why. Check the \
notes against the evidence: a note can be mistaken.

## The user's notes
{notes}

## Conversation
{turns}

## Tasks (each task's request, how it ended, and its work log: the code run and what came back)
{tasks}

## Failures the harness already logged in this session
{failures}

Reply with ONLY a JSON object:
{{"verdict": "failure" or "not_failure",
  "confidence": <0 to 1>,
  "summary": "<one or two sentences: what happened and why that verdict>",
  "problems": [{{"what": "<the problem>", "turn": <turn number or null>, "note": <the number of the note it \
bears on, or null>, "evidence": "<the line of the answer or work log that shows it>", "severity": "high" or \
"medium" or "low"}}],
  "notes_checked": [{{"note": <number>, "holds": true or false, "why": "<one sentence>"}}]}}"""


def _clip(text: str, n: int) -> str:
    text = str(text or "")
    return text if len(text) <= n else text[:n] + f" [...{len(text) - n} more chars]"


def review_prompt(record: dict[str, Any], notes: list[dict[str, Any]], budget: int = 60000) -> str:
    """The prompt, with the work logs cut to fit `budget` characters (the latest of each kept)."""
    note_txt = "\n".join(f"{i}. " + (f"(turn {n['turn']}) " if n.get("turn") else "") + n["text"]
                         for i, n in enumerate(notes, 1)) or "(no notes: look for anything that went wrong)"
    turns = "\n\n".join(f"### Turn {t['n']} ({t['persona']})\nUser: {_clip(t['user'], 3000)}\n"
                        f"Assistant: {_clip(t['assistant'], 6000)}" for t in record["turns"]) or "(none in memory)"
    fails = "\n".join(f"- #{f['id']} turn {f['turn']}, {f['persona'] or '-'}: {f['kind']} ({f['source']}) - "
                      f"{f['summary']}" for f in record["failures"]) or "(none)"
    fixed = len(PROMPT) + len(note_txt) + len(turns) + len(fails)
    per = max(800, (budget - fixed) // max(len(record["tasks"]), 1))
    tasks = "\n\n".join(
        f"### Turn {t['turn']} task {t['task']}: {t['persona']} - "
        f"{'ended: ' + t['stopped'] if t['stopped'] else 'finished'} after {t['steps']} steps\n"
        f"Request: {_clip(t['request'], 1500)}\n" + (f"Answer: {_clip(t['answer'], 1500)}\n" if t["answer"] else "")
        + "Work log:\n" + ((t["work_log"] if len(t["work_log"]) <= per else "[...]\n" + t["work_log"][-per:])
                           or "(no code was run)")
        for t in record["tasks"]) or "(no task folders)"
    return PROMPT.format(notes=note_txt, turns=turns, tasks=tasks, failures=fails)


def parse_review(text: str) -> dict[str, Any]:
    """The model's JSON reply -> a review; ReviewError when there is no usable verdict in it."""
    from .refine.generate import parse_json
    data = parse_json(text, "{")
    if not isinstance(data, dict):
        raise ReviewError("the model's reply had no JSON object")
    verdict = str(data.get("verdict", "")).strip().lower().replace("-", "_").replace(" ", "_")
    if verdict not in VERDICTS:
        raise ReviewError(f"the model's verdict {data.get('verdict')!r} is not failure / not_failure")
    try:
        conf = max(0.0, min(1.0, float(data.get("confidence"))))
    except (TypeError, ValueError):
        conf = None
    problems = [p if isinstance(p, dict) else {"what": str(p)} for p in data.get("problems") or []]
    checked = [c for c in data.get("notes_checked") or [] if isinstance(c, dict)]
    return {"verdict": verdict, "confidence": conf, "summary": str(data.get("summary", "")).strip(),
            "problems": problems, "notes_checked": checked}


def reviewer_block(cfg, model: str | dict | None = None, model_override: str | None = None) -> dict[str, Any]:
    """The reviewing model's block: `model` ("provider:model" or a block), else config review.model,
    else the harness's --model, else models.strong, else the default persona's model."""
    from .personas import load_personas
    settings = cfg.raw.get("review") or {}
    m = model or settings.get("model") or model_override
    if isinstance(m, dict):
        block = dict(m)
    elif m:
        provider, _, name = str(m).partition(":")
        if not name:
            raise ValueError(f"model must be provider:model, got {m!r}")
        block = {"provider": provider, "model": name}
    elif cfg.models.get("strong"):
        block = dict(cfg.models["strong"])
    else:
        personas = load_personas(cfg.personas_dir)
        p = personas.get(cfg.default_persona) or next(iter(personas.values()))
        block = {k: v for k, v in p.model.items() if k != "fallback"}
    block.pop("think", None)   # a review must end; a finite budget instead of unlimited thinking
    block["max_tokens"] = int(settings.get("max_tokens", block.get("max_tokens") or 32768))
    return block


def run_review(llm, record: dict[str, Any], notes: list[dict[str, Any]], budget: int = 60000) -> dict[str, Any]:
    """Ask `llm` to review the session. -> the parsed review, with the model's name and raw reply."""
    prompt = review_prompt(record, notes, budget)
    res = llm.chat([{"role": "system", "content": "You review AI assistant sessions for failures. "
                                                  "You answer with one JSON object."},
                    {"role": "user", "content": prompt}])
    text = getattr(res, "text", str(res))
    try:
        review = parse_review(text)
    except ReviewError as e:
        raise ReviewError(f"{e}; reply began: {text.strip()[:300]!r}") from None
    review.update(model=getattr(res, "model", "") or getattr(llm, "name", ""), ts=time.time(), raw=text[-CAP:])
    return review


# ---------------------------------------------------------------- tying it to the failure log

def sync_failure(reviews: ReviewLog, failures, flag: dict[str, Any], record: dict[str, Any] | None = None) -> dict:
    """Keep the failure log in step with a flag's verdict: a failure gets one row (source "review"), and
    not_failure marks that row ignored. -> the flag as it is now."""
    linked = failures.get(flag["failure_id"]) if flag.get("failure_id") else None
    record = record or {}
    if flag["verdict"] == "failure" and not (record.get("turns") or record.get("tasks")):
        return flag   # nothing recorded: refinement replays the request, and there is none
    if flag["verdict"] == "failure":
        if linked is not None:
            if linked["status"] == "ignored":
                failures.set_status(linked["id"], "open", "the review says failure again")
            return reviews.get(flag["id"])
        review = flag.get("review") or {}
        notes = "\n".join(f"- {n['text']}" for n in flag["notes"])
        problems = "\n".join(f"- {p.get('what', '')}" + (f" (evidence: {p['evidence']})" if p.get("evidence") else "")
                             for p in review.get("problems") or [])
        lead = record.get("tasks") or []
        turns = record.get("turns") or []
        row = failures.record(
            source="review", kind="review_failure",
            summary=flag.get("reason") or review.get("summary") or (flag["notes"][0]["text"] if flag["notes"] else
                                                                     "flagged in review"),
            session=flag["session"], persona=lead[0]["persona"] if lead else (turns[0]["persona"] if turns else ""),
            request=turns[0]["user"] if turns else (lead[0]["request"] if lead else ""),
            answer=turns[-1]["assistant"] if turns else "",
            work_log=lead[0]["work_log"] if lead else "", workdir=lead[0]["workdir"] if lead else "",
            detail=f"Review #{flag['id']} ({flag['verdict_by']}).\nNotes:\n{notes or '-'}\nProblems:\n{problems or '-'}",
            scores={"confidence": review.get("confidence")} if review.get("confidence") is not None else None)
        reviews.link_failure(flag["id"], row["id"])
    elif linked is not None and linked["status"] == "open":
        failures.set_status(linked["id"], "ignored", f"review #{flag['id']}: not a failure")
    return reviews.get(flag["id"])


class Reviewer:
    """The pieces a review needs, for the CLI and the dashboard: the flag store, the session material,
    the reviewing model, and the failure log."""

    def __init__(self, cfg, memory, failures, reviews: ReviewLog | None = None, llm_factory=None,
                 model_override: str | None = None):
        self.cfg, self.memory, self.failures = cfg, memory, failures
        self.reviews = reviews or ReviewLog(cfg.workspace / "reviews.db")
        self.model_override = model_override
        self._factory = llm_factory
        self.running: set[int] = set()   # flags a model is reviewing right now
        self._lock = threading.Lock()

    def record(self, session: str) -> dict[str, Any]:
        return session_record(self.cfg.workspace, self.memory, self.failures, session)

    def flag(self, session: str, note: str = "", turn: int | None = None) -> dict[str, Any]:
        title = "" if self.reviews.find(session) else session_title(self.record(session))
        return self.reviews.flag(session, note, turn, title)

    def llm(self, model: str | dict | None = None):
        block = reviewer_block(self.cfg, model, self.model_override)
        if self._factory is not None:
            return self._factory(block)
        from .llm import make_llm
        return make_llm(block, self.cfg.providers)

    def run(self, flag_id: int, model: str | dict | None = None) -> dict[str, Any]:
        """Review a flagged session with the model. -> the flag with the review (and its verdict)."""
        flag = self.reviews.get(flag_id)
        if flag is None:
            raise KeyError(f"no flag #{flag_id}")
        with self._lock:
            if flag_id in self.running:
                raise ReviewError("a review of this session is already running")
            self.running.add(flag_id)
        try:
            record = self.record(flag["session"])
            if not record["turns"] and not record["tasks"]:
                raise ReviewError(f"nothing recorded for session {flag['session']}")
            budget = int((self.cfg.raw.get("review") or {}).get("prompt_chars", 60000))
            review = run_review(self.llm(model), record, flag["notes"], budget)
            flag = self.reviews.set_review(flag_id, review)
            return sync_failure(self.reviews, self.failures, flag, record)
        finally:
            with self._lock:
                self.running.discard(flag_id)

    def decide(self, flag_id: int, verdict: str, reason: str = "") -> dict[str, Any]:
        """The user's verdict (it overrules the model's)."""
        flag = self.reviews.set_verdict(flag_id, verdict, "user", reason)
        return sync_failure(self.reviews, self.failures, flag, self.record(flag["session"]))


# ---------------------------------------------------------------- command line

def _when(ts: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "-"


def describe(flag: dict[str, Any]) -> str:
    state = (f"{flag['verdict']} ({flag['verdict_by']})" if flag["status"] == "decided" else flag["status"])
    return f"#{flag['id']:<4} {_when(flag['updated'])}  {flag['session']:<26} [{state}] {flag['title'][:70]}"


def show(flag: dict[str, Any], record: dict[str, Any] | None = None) -> str:
    lines = [describe(flag)]
    for i, n in enumerate(flag["notes"], 1):
        lines.append(f"  note {i}" + (f" (turn {n['turn']})" if n["turn"] else "") + f": {n['text']}")
    r = flag.get("review")
    if r:
        conf = f", confidence {r['confidence']:.2f}" if r.get("confidence") is not None else ""
        lines.append(f"  model review ({r.get('model') or '?'}, {_when(r.get('ts'))}): {r['verdict']}{conf}")
        if r.get("summary"):
            lines.append(f"    {r['summary']}")
        for p in r.get("problems") or []:
            where = ", ".join(x for x in (f"turn {p['turn']}" if p.get("turn") else "",
                                          f"note {p['note']}" if p.get("note") else "") if x)
            lines.append(f"    - [{p.get('severity') or '?'}] {p.get('what', '')}" + (f" ({where})" if where else ""))
            if p.get("evidence"):
                lines.append(f"        evidence: {p['evidence']}")
    if flag["verdict_by"] == "user":
        lines.append(f"  your verdict: {flag['verdict']}" + (f" - {flag['reason']}" if flag["reason"] else ""))
    if flag.get("failure_id"):
        lines.append(f"  failure log: #{flag['failure_id']}")
    if record is not None:
        lines.append(f"  session: {len(record['turns'])} turns, {len(record['tasks'])} tasks, "
                     f"{len(record['failures'])} failures logged")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from .config import load_config
    from .failures import FailureLog
    from .memory import Memory
    ap = argparse.ArgumentParser(prog="python -m jevh.review", description="Flag sessions for review.")
    ap.add_argument("--config")
    ap.add_argument("--status", choices=STATUSES)
    ap.add_argument("-n", type=int, default=30)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("sessions", help="recent sessions, to pick one to flag")
    fl = sub.add_parser("flag", help="flag a session for review, with a note")
    fl.add_argument("session")
    fl.add_argument("note", nargs="?", default="")
    fl.add_argument("--turn", type=int)
    nt = sub.add_parser("note", help="add a note to a flagged session (flags it if needed)")
    nt.add_argument("ref", help="flag id or session")
    nt.add_argument("text")
    nt.add_argument("--turn", type=int)
    sh = sub.add_parser("show", help="a flag's notes and review")
    sh.add_argument("ref")
    rn = sub.add_parser("run", help="have a model review the session against the notes")
    rn.add_argument("ref")
    rn.add_argument("--model", help="provider:model (default: config review.model, models.strong, ...)")
    vd = sub.add_parser("verdict", help="your own verdict: failure | not_failure")
    vd.add_argument("ref")
    vd.add_argument("verdict", choices=VERDICTS)
    vd.add_argument("reason", nargs="?", default="")
    ds = sub.add_parser("dismiss", help="drop a flag from the review list")
    ds.add_argument("ref")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    memory, failures = Memory(cfg.workspace / "memory"), FailureLog(cfg.workspace / "failures.db")
    rv = Reviewer(cfg, memory, failures)

    def pick(ref: str) -> dict[str, Any] | None:
        f = rv.reviews.resolve(ref)
        if f is None:
            print(f"no flagged session {ref}")
        return f

    if a.cmd == "sessions":
        for s in list_sessions(cfg.workspace, memory, a.n):
            mark = " [flagged]" if rv.reviews.find(s["session"]) else ""
            print(f"{s['session']:<26} {_when(s['last'])}  {s['turns']:>3} turns{mark}  {s['title'][:80]}")
        return 0
    if a.cmd == "flag":
        f = rv.flag(a.session, a.note, a.turn)
        print(f"flagged #{f['id']} {f['session']} ({len(f['notes'])} notes)")
        return 0
    if a.cmd == "note":
        f = rv.reviews.resolve(a.ref) or rv.flag(a.ref)
        rv.reviews.add_note(f["id"], a.text, a.turn)
        print(f"note added to #{f['id']} {f['session']}")
        return 0
    if a.cmd in ("show", "run", "verdict", "dismiss"):
        f = pick(a.ref)
        if f is None:
            return 1
        if a.cmd == "run":
            b = reviewer_block(cfg, a.model)
            print(f"reviewing {f['session']} on {b['provider']}:{b['model']}...")
            try:
                f = rv.run(f["id"], a.model)
            except (ReviewError, ValueError) as e:
                print(f"review failed: {e}")
                return 1
        elif a.cmd == "verdict":
            f = rv.decide(f["id"], a.verdict, a.reason)
        elif a.cmd == "dismiss":
            rv.reviews.set_status(f["id"], "dismissed")
            print(f"#{f['id']} dismissed")
            return 0
        print(show(f, rv.record(f["session"])))
        return 0
    for f in rv.reviews.list(status=a.status, limit=a.n):
        print(describe(f))
    c = rv.reviews.counts()
    print(("\n" + ", ".join(f"{k} {n}" for k, n in c.items())) if c else "no flagged sessions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
