"""Past sessions, categorized by Jev: the dashboard's History tab.

A session is what one run of the dashboard (or the CLI) did. Its turns are in memory.db, its task folders and files in
workspace/sessions/<id>/. Jev puts each session in a category, and the categories are the assistants (personas): "which
assistant is best suited to handle this whole conversation?" is the kind of Choice that routes a request, asked once for
the conversation (Jev.session_category). The assistants' descriptions are what Jev already routes on, a persona the
harness creates later is a new category by itself, and the names match the ones you see elsewhere.

    workspace/history.db     session -> Jev's probabilities, the best category, how many turns it had when asked, and
                             which assistants were offered

Jev is asked in the background, once per session, the first time the tab lists it (about 0.1-0.3 s each), and again
for a session that has more turns than when it was asked (the one you are in). What shows is decided at read time from the
saved probabilities, so changing `min_confidence` or `also_at` needs no new questions. A session whose best category is
under `min_confidence` is listed under "No single field"; a second category at `also_at` or more shows as a second tag.

Settings (config.yaml `history:`):

    history:
      enabled: true        # false: the tab lists nothing and Jev is never asked
      min_confidence: 0.6  # under this the session is listed under "No single field" (evals/session_category_cases.py)
      also_at: 0.25        # another category this likely shows as a second tag
      max_requests: 6      # how many of a session's requests Jev reads: the first ones and the last two
      request_chars: 280   # each one cut to this
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .review import list_sessions, session_record, session_tasks

DEFAULTS = {"enabled": True, "min_confidence": 0.6, "also_at": 0.25, "max_requests": 6, "request_chars": 280}
MIXED = "mixed"      # the best category was not likely enough: no single field
EMPTY = "empty"      # nothing to read: a session with no request in it
SESSION_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$")
RETRY_AFTER = 30.0   # seconds before Jev is asked again after it could not be


def settings(raw: dict[str, Any] | None) -> dict[str, Any]:
    return {**DEFAULTS, **((raw or {}).get("history") or {})}


def pretty(name: str) -> str:
    """A category's name for people: high_end_dev -> High end dev."""
    words = name.replace("_", " ").strip()
    return words[:1].upper() + words[1:]


# -- what is saved ----------------------------------------------------------------------------------------------

class SessionIndex:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS categories(session TEXT PRIMARY KEY, category TEXT, confidence REAL, "
                        "probs TEXT, turns INTEGER, roster TEXT, asked REAL)")
        self.db.commit()

    def all(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            rows = self.db.execute("SELECT session, category, confidence, probs, turns, roster, asked FROM categories").fetchall()
        return {s: {"category": c, "confidence": conf, "probs": json.loads(p or "{}"), "turns": t,
                    "roster": json.loads(r or "[]"), "asked": a} for s, c, conf, p, t, r, a in rows}

    def put(self, session: str, category: str, confidence: float, probs: dict[str, float], turns: int,
            roster: list[str]) -> None:
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO categories VALUES (?,?,?,?,?,?,?)",
                            (session, category, float(confidence), json.dumps(probs), int(turns), json.dumps(roster),
                             time.time()))
            self.db.commit()

    def forget(self, session: str | None = None) -> None:
        """Drop one session's category (or every one): Jev is asked again."""
        with self._lock:
            if session is None:
                self.db.execute("DELETE FROM categories")
            else:
                self.db.execute("DELETE FROM categories WHERE session=?", (session,))
            self.db.commit()

    def close(self) -> None:
        with self._lock:
            self.db.close()


# -- reading a session --------------------------------------------------------------------------------------------

def requests_of(workspace: Path, memory, session: str) -> tuple[list[str], int]:
    """(the user's requests of a session, in order; how many turns memory has of it). A session whose turns all failed
    is only in its folder: then the requests are the lead tasks'."""
    turns = memory.session_turns(session, 10 ** 6)
    if turns:
        return [t.user for t in turns], len(turns)
    return [t["request"] for t in session_tasks(workspace, session) if t["task"] == 1], 0


def conversation(requests: list[str], limit: int = 6, chars: int = 280) -> str:
    """What Jev reads: the requests numbered in order, each cut short; a long session shows its first ones and its last
    two with a line saying how many are left out (the start says what the work is, the end where it went)."""
    reqs = [" ".join(r.split())[:chars] for r in requests if r and r.strip()]
    if len(reqs) <= limit:
        return "\n".join(f"{i}. {r}" for i, r in enumerate(reqs, 1))
    head, tail = max(limit - 2, 1), 2
    lines = [f"{i}. {r}" for i, r in enumerate(reqs[:head], 1)]
    lines.append(f"[... {len(reqs) - head - tail} more requests ...]")
    lines += [f"{i}. {r}" for i, r in enumerate(reqs[-tail:], len(reqs) - tail + 1)]
    return "\n".join(lines)


def view(row: dict[str, Any] | None, s: dict[str, Any]) -> tuple[str | None, float, list[str]]:
    """(category, how sure, other categories) as the tab shows a saved row under the current settings. None: not asked yet."""
    if row is None:
        return None, 0.0, []
    probs, top = row["probs"], row["category"]
    if not top or not probs:
        return EMPTY, 0.0, []
    conf = probs.get(top, row["confidence"])
    ranked = [k for k, v in sorted(probs.items(), key=lambda kv: -kv[1]) if k != top and v >= s["also_at"]]
    if conf < s["min_confidence"]:
        return MIXED, conf, [top] + ranked[:2]
    return top, conf, ranked[:2]


# -- asking Jev, in the background ------------------------------------------------------------------------------

class Categorizer:
    """Asks Jev about the sessions that have no category (or more turns than when it was asked), one at a time on one
    background thread. `ensure` says how many are still waiting; the tab asks again until it is none."""

    def __init__(self, get_harness: Callable[[], Any], index: SessionIndex, publish: Callable[..., None] | None = None):
        self._h, self.index, self.publish = get_harness, index, publish or (lambda *a, **k: None)
        self._lock = threading.Lock()
        self._queue: list[str] = []
        self._thread: threading.Thread | None = None
        self._failed_at = 0.0
        self.error = ""

    def ensure(self, sessions: list[dict[str, Any]]) -> int:
        rows = self.index.all()
        with self._lock:
            if self._failed_at and time.time() - self._failed_at < RETRY_AFTER:
                return 0   # Jev could not be reached a moment ago: not again at once
            for s in sessions:
                row = rows.get(s["session"])
                if s["session"] not in self._queue and (row is None or row["turns"] != s["turns"]):
                    self._queue.append(s["session"])
            if self._queue and (self._thread is None or not self._thread.is_alive()):
                self._thread = threading.Thread(target=self._run, name="history", daemon=True)
                self._thread.start()
            return len(self._queue)

    def pending(self) -> int:
        with self._lock:
            return len(self._queue)

    def _run(self) -> None:
        while True:
            with self._lock:
                if not self._queue:
                    self._thread = None
                    return
                session = self._queue[0]
            try:
                self._ask(session)
            except Exception as e:  # noqa: BLE001 - Jev down: say so, and try again later
                with self._lock:
                    self.error = f"Jev could not be asked ({type(e).__name__}: {e})"[:240]
                    self._failed_at = time.time()
                    self._queue.clear()
                    self._thread = None
                self.publish("history", {"error": self.error})
                return
            with self._lock:
                if self._queue and self._queue[0] == session:
                    self._queue.pop(0)
                self.error = ""
                self._failed_at = 0.0
            self.publish("history", {"session": session})

    def _ask(self, session: str) -> None:
        h = self._h()
        s = settings(h.cfg.raw)
        reqs, turns = requests_of(h.cfg.workspace, h.memory, session)
        roster = sorted(h.personas)
        if not any(r.strip() for r in reqs) or not roster:
            self.index.put(session, "", 0.0, {}, turns, roster)
            return
        probs, resp = h.jev.session_category(conversation(reqs, int(s["max_requests"]), int(s["request_chars"])),
                                             {n: p.description for n, p in h.personas.items()})
        try:   # metered as its own kind, on no turn: a running turn's Jev calls stay its own
            h.usage.record(session=h.session, turn=0, kind="jev_history", model=getattr(resp, "model", ""), calls=1,
                           input_tokens=getattr(resp, "input_tokens", 0), seconds=getattr(resp, "latency_s", 0.0))
        except Exception:  # noqa: BLE001 - accounting must never break the tab
            pass
        top = max(probs, key=lambda k: probs[k])
        self.index.put(session, top, probs[top], probs, turns, roster)


# -- what the tab shows -------------------------------------------------------------------------------------------

class History:
    def __init__(self, get_harness: Callable[[], Any], publish: Callable[..., None] | None = None):
        self._h = get_harness
        self.index = SessionIndex(Path(get_harness().cfg.workspace) / "history.db")
        self.categorizer = Categorizer(get_harness, self.index, publish)

    def overview(self, limit: int = 500) -> dict[str, Any]:
        """Every session, newest first, each with its category (None while Jev has not been asked), and the categories that
        have sessions, most recently used first."""
        h = self._h()
        s = settings(h.cfg.raw)
        if not s["enabled"]:
            return {"enabled": False, "categories": [], "sessions": [], "pending": 0, "error": "", "stale": 0}
        sessions = list_sessions(h.cfg.workspace, h.memory, limit)
        flagged = {f["session"]: f["id"] for f in h.review.reviews.list(limit=10 ** 6)}
        pending = self.categorizer.ensure(sessions)
        rows, roster = self.index.all(), sorted(h.personas)
        out, last, count, stale = [], {}, {}, 0
        for ses in sessions:
            row = rows.get(ses["session"])
            cat, conf, also = view(row, s)
            stale += bool(row and row["roster"] != roster)
            out.append({**ses, "title": " ".join((ses.get("title") or "").split()), "category": cat, "confidence": round(conf, 3),
                        "also": also, "current": ses["session"] == h.session, "flag": flagged.get(ses["session"])})
            if cat:
                count[cat] = count.get(cat, 0) + 1
                last[cat] = max(last.get(cat, 0), ses["last"] or 0)
        named = {n: p.description for n, p in h.personas.items()}
        cats = [{"name": c, "title": {MIXED: "No single field", EMPTY: "Empty sessions"}.get(c) or pretty(c),
                 "description": named.get(c, {MIXED: "Jev could not tell one field: the conversation spans several, or fits none of your assistants.",
                                              EMPTY: "No request in them."}.get(c, "An assistant this harness no longer has.")),
                 "count": n} for c, n in sorted(count.items(), key=lambda kv: (kv[0] in (MIXED, EMPTY), -last[kv[0]]))]
        return {"enabled": True, "categories": cats, "sessions": out, "pending": pending, "stale": stale,
                "error": self.categorizer.error}

    def recategorize(self, session: str | None = None) -> int:
        """Ask Jev again about one session, or all of them. -> how many are waiting."""
        self.index.forget(session)
        self.categorizer._failed_at = 0.0
        h = self._h()
        return self.categorizer.ensure(list_sessions(h.cfg.workspace, h.memory, 10 ** 6))

    def exists(self, session: str) -> bool:
        """Whether `session` is one this workspace has (only ids of that form are looked at: a path can't get through)."""
        h = self._h()
        return bool(SESSION_ID.match(session)) and session in {x["session"] for x in list_sessions(h.cfg.workspace, h.memory, 10 ** 6)}

    def transcript(self, session: str) -> dict[str, Any] | None:
        """One session read-only: its conversation, who ran what, its files, its cost, and Jev's category with the odds.
        None: no such session."""
        if not self.exists(session):
            return None
        h = self._h()
        rec = session_record(h.cfg.workspace, h.memory, h.failures, session)
        s = settings(h.cfg.raw)
        row = self.index.all().get(session)
        cat, conf, also = view(row, s)
        art = Path(h.cfg.workspace) / "sessions" / session / "artifacts"
        files = [{"name": f.name, "path": str(f), "size": f.stat().st_size}
                 for f in sorted(art.iterdir())[:200] if f.is_file()] if art.is_dir() else []
        try:
            usage = h.usage.summary(session=session)
        except Exception:  # noqa: BLE001
            usage = None
        return {"session": session, "category": cat, "confidence": round(conf, 3), "also": also,
                "probs": (row or {}).get("probs", {}), "current": session == h.session,
                "turns": rec["turns"],
                "tasks": [{k: (t[k][:300] if isinstance(t[k], str) else t[k]) for k in ("turn", "task", "persona", "request", "stopped", "steps")}
                          for t in rec["tasks"]],
                "failures": rec["failures"], "artifacts": files, "usage": usage}
