"""Notebook: a place to jot down ideas, problems and other notes, to come back to later.

workspace/notebook.db
  entries  one row per jotted thing: its kind (idea | problem | note), the text (the first line is its
           title), a status (open | done), whether it is pinned, and the session it was jotted in, if any

It is for you: nothing in the notebook reaches a worker or Jev. A problem that turns out to be a session
going wrong can be flagged for review (jevh/review.py) from the dashboard, with the entry as its note.

    jevh notebook                                  # open entries (pinned first, then newest)
    jevh notebook --kind problem -q "wrong units"  # only problems that mention these words
    jevh notebook add "try a second model as the checker"          # a note
    jevh notebook idea "show token cost per turn in the header"
    jevh notebook problem "the chart axis had no units"
    jevh notebook show 4 | edit 4 "new text" | done 4 | reopen 4 | pin 4 | unpin 4 | rm 4

In the chat: /idea <text>, /problem <text>, /jot <text>, /notebook [ideas|problems|notes|done|all|words].
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any

KINDS = ("idea", "problem", "note")
STATUSES = ("open", "done")
MAX_CHARS = 20000


def title_of(text: str, cap: int = 100) -> str:
    """The entry's first line without markdown marks, cut to `cap` characters."""
    for line in str(text).splitlines():
        line = re.sub(r"^\s*(?:#{1,6}|[-*>]|\d+\.)\s*", "", line).strip()
        if line:
            return line if len(line) <= cap else line[:cap - 1].rstrip() + "…"
    return ""


class Notebook:
    """The entries, in SQLite (thread-safe)."""

    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS entries(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, updated REAL,
                kind TEXT DEFAULT 'note', text TEXT, status TEXT DEFAULT 'open', pinned INTEGER DEFAULT 0,
                session TEXT DEFAULT '');
            CREATE INDEX IF NOT EXISTS entries_status ON entries(status, kind);
        """)
        self.db.commit()

    # -- writing ---------------------------------------------------------------------------
    def add(self, text: str, kind: str = "note", session: str = "", pinned: bool = False) -> dict[str, Any]:
        text, kind = self._text(text), self._kind(kind)
        now = time.time()
        with self._lock:
            eid = self.db.execute("INSERT INTO entries(ts, updated, kind, text, pinned, session) VALUES (?,?,?,?,?,?)",
                                  (now, now, kind, text, int(bool(pinned)), str(session or ""))).lastrowid
            self.db.commit()
        return self.get(eid)

    def update(self, entry_id: int, *, text: str | None = None, kind: str | None = None, status: str | None = None,
               pinned: bool | None = None) -> dict[str, Any] | None:
        """Change the given fields. -> the entry, or None when there is no such entry."""
        sets: dict[str, Any] = {}
        if text is not None:
            sets["text"] = self._text(text)
        if kind is not None:
            sets["kind"] = self._kind(kind)
        if status is not None:
            if status not in STATUSES:
                raise ValueError(f"status must be one of {', '.join(STATUSES)}")
            sets["status"] = status
        if pinned is not None:
            sets["pinned"] = int(bool(pinned))
        if sets:
            sets["updated"] = time.time()
            with self._lock:
                cur = self.db.execute(f"UPDATE entries SET {', '.join(k + '=?' for k in sets)} WHERE id=?",
                                      [*sets.values(), entry_id])
                self.db.commit()
            if not cur.rowcount:
                return None
        return self.get(entry_id)

    def delete(self, entry_id: int) -> bool:
        with self._lock:
            cur = self.db.execute("DELETE FROM entries WHERE id=?", (entry_id,))
            self.db.commit()
        return cur.rowcount > 0

    # -- reading ---------------------------------------------------------------------------
    def get(self, entry_id: int) -> dict[str, Any] | None:
        with self._lock:
            r = self.db.execute("SELECT * FROM entries WHERE id=?", (entry_id,)).fetchone()
        return self._row(r) if r else None

    def list(self, kind: str | None = None, status: str | None = None, query: str | None = None,
             session: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        """Pinned first, then newest first. `query`: entries whose text has every one of its words."""
        where: list[str] = []
        args: list[Any] = []
        for col, val, ok in (("kind", kind, KINDS), ("status", status, STATUSES)):
            if val is not None:
                if val not in ok:
                    raise ValueError(f"{col} must be one of {', '.join(ok)}")
                where.append(f"{col}=?")
                args.append(val)
        if session is not None:
            where.append("session=?")
            args.append(session)
        for word in str(query or "").split():
            where.append("text LIKE ? ESCAPE '\\'")
            args.append("%" + word.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        with self._lock:
            rows = self.db.execute(f"SELECT * FROM entries {clause} ORDER BY pinned DESC, ts DESC, id DESC LIMIT ?",
                                   [*args, limit]).fetchall()
        return [self._row(r) for r in rows]

    def counts(self) -> dict[str, int]:
        """open, done and all entries, and the open ones of each kind."""
        with self._lock:
            rows = self.db.execute("SELECT status, kind, COUNT(*) FROM entries GROUP BY status, kind").fetchall()
        out = {"open": 0, "done": 0, "all": 0, **{k: 0 for k in KINDS}}
        for status, kind, n in rows:
            out["all"] += n
            out[status] = out.get(status, 0) + n
            if status == "open":
                out[kind] = out.get(kind, 0) + n
        return out

    def close(self) -> None:
        with self._lock:
            self.db.close()

    # -- helpers ---------------------------------------------------------------------------
    @staticmethod
    def _text(text: str) -> str:
        text = str(text or "").strip()
        if not text:
            raise ValueError("nothing to write down")
        if len(text) > MAX_CHARS:
            raise ValueError(f"too long: {len(text)} characters (the limit is {MAX_CHARS})")
        return text

    @staticmethod
    def _kind(kind: str) -> str:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}")
        return kind

    @staticmethod
    def _row(r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        d["pinned"] = bool(d["pinned"])
        d["title"] = title_of(d["text"])
        return d


# ---------------------------------------------------------------- command line

def _when(ts: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "-"


def describe(e: dict[str, Any]) -> str:
    """One line: id, when, kind, state and the title."""
    mark = ("done" if e["status"] == "done" else "open") + (", pinned" if e["pinned"] else "")
    return f"#{e['id']:<4} {_when(e['ts'])}  {e['kind']:<8} [{mark}] {e['title']}"


def show(e: dict[str, Any]) -> str:
    lines = [describe(e), ""]
    lines += ["  " + ln for ln in e["text"].splitlines()]
    if e["session"]:
        lines += ["", f"  jotted in session {e['session']}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from .config import load_config
    ap = argparse.ArgumentParser(prog="jevh notebook", description="Jot down ideas and problems.")
    ap.add_argument("--config")
    ap.add_argument("--kind", choices=KINDS)
    ap.add_argument("--status", choices=(*STATUSES, "all"), default="open")
    ap.add_argument("-q", "--query", help="only entries that mention all of these words")
    ap.add_argument("-n", type=int, default=50)
    sub = ap.add_subparsers(dest="cmd")
    for name, helptext in (("add", "jot a note"), ("idea", "jot an idea"), ("problem", "jot a problem")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("text", nargs="+")
        if name == "add":
            p.add_argument("--kind", dest="add_kind", choices=KINDS, default="note")
    ed = sub.add_parser("edit", help="replace an entry's text")
    ed.add_argument("id", type=int)
    ed.add_argument("text", nargs="+")
    for name, helptext in (("show", "an entry in full"), ("done", "mark it done (a problem: solved)"),
                           ("reopen", "mark it open again"), ("pin", "keep it at the top"),
                           ("unpin", "stop keeping it at the top"), ("rm", "delete it")):
        sub.add_parser(name, help=helptext).add_argument("id", type=int)
    a = ap.parse_args(argv)

    nb = Notebook(load_config(a.config).workspace / "notebook.db")
    if a.cmd in ("add", "idea", "problem"):
        try:
            e = nb.add(" ".join(a.text), a.add_kind if a.cmd == "add" else a.cmd)
        except ValueError as err:
            print(err)
            return 1
        print(f"#{e['id']} {e['kind']} jotted down")
        return 0
    if a.cmd:
        e = nb.get(a.id)
        if e is None:
            print(f"no entry #{a.id}")
            return 1
        if a.cmd == "show":
            print(show(e))
            return 0
        if a.cmd == "rm":
            nb.delete(a.id)
            print(f"#{a.id} deleted")
            return 0
        change = {"edit": {"text": " ".join(getattr(a, "text", []))}, "done": {"status": "done"},
                  "reopen": {"status": "open"}, "pin": {"pinned": True}, "unpin": {"pinned": False}}[a.cmd]
        try:
            print(describe(nb.update(a.id, **change)))
        except ValueError as err:
            print(err)
            return 1
        return 0
    entries = nb.list(kind=a.kind, status=None if a.status == "all" else a.status, query=a.query, limit=a.n)
    for e in entries:
        print(describe(e))
    c = nb.counts()
    print(("\n" if entries else "") + (f"open: {c['idea']} ideas, {c['problem']} problems, {c['note']} notes; "
                                       f"{c['done']} done" if c["all"] else "the notebook is empty"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
