"""Conversation memory and remembered facts on disk. Nothing here reaches a worker unless
Jev passes it (gate_context up front, or recall() when the worker asks).

workspace/memory/memory.db        every turn and fact, with a full-text index (SQLite FTS5)
workspace/memory/<persona>.jsonl  human-readable log per persona (+ timeline.jsonl for all)

Candidates for a request are the persona's recent turns, a few recent turns of other
personas, turns from ANY time that share words with the request (so a project from last
month is reachable, not just the last 20 turns), and matching remembered facts.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

CHUNK_CHARS = 900
STOP = set("""a an and are as at be been but by can could did do does for from had has have how i if in into is it
its just me my no not of on or our please so some than that the their them then there these they this to up us
was we what when where which who why will with would you your yours hey hi thanks thank ok okay make get give
want need like tell show help about also any all more most very really can't don't i'm it's what's""".split())


@dataclass
class Turn:
    persona: str
    user: str
    assistant: str
    session: str
    ts: float

    def chunk(self) -> str:
        text = f"[{self.persona}] User: {self.user}\nAssistant: {self.assistant}"
        return text if len(text) <= CHUNK_CHARS else text[: CHUNK_CHARS - 3] + "..."


@dataclass
class Fact:
    id: int
    text: str
    ts: float
    source: str

    def chunk(self) -> str:
        return f"[remembered {time.strftime('%Y-%m-%d', time.localtime(self.ts))}] {self.text}"


def fts_query(text: str, max_terms: int = 12) -> str | None:
    """Words of `text` as an FTS5 OR-query (quoted, so punctuation can't break the syntax)."""
    words, seen = [], set()
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9_'-]{2,}", text.lower()):
        w = w.strip("'-")
        if w not in STOP and w not in seen and len(w) >= 3:
            seen.add(w)
            words.append(f'"{w}"')
    return " OR ".join(words[:max_terms]) or None


class Memory:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(self.root / "memory.db", check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS turns(id INTEGER PRIMARY KEY, persona TEXT, session TEXT, ts REAL,
                                             user TEXT, assistant TEXT);
            CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(user, assistant, content='turns',
                content_rowid='id', tokenize='porter unicode61');
            CREATE TABLE IF NOT EXISTS facts(id INTEGER PRIMARY KEY, ts REAL, text TEXT, source TEXT,
                                             active INTEGER DEFAULT 1);
            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(text, content='facts', content_rowid='id',
                tokenize='porter unicode61');
        """)
        if not self.db.execute("SELECT 1 FROM turns LIMIT 1").fetchone():
            self._import_jsonl()

    def _import_jsonl(self) -> None:
        path = self.root / "timeline.jsonl"
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                self._insert(Turn(**json.loads(line)))
        self.db.commit()

    def _insert(self, t: Turn) -> None:
        cur = self.db.execute("INSERT INTO turns(persona, session, ts, user, assistant) VALUES (?,?,?,?,?)",
                              (t.persona, t.session, t.ts, t.user, t.assistant))
        self.db.execute("INSERT INTO turns_fts(rowid, user, assistant) VALUES (?,?,?)",
                        (cur.lastrowid, t.user, t.assistant))

    def _turns(self, sql: str, args: tuple = ()) -> list[Turn]:
        with self._lock:
            rows = self.db.execute(sql, args).fetchall()
        return [Turn(p, u, a, s, ts) for p, s, ts, u, a in rows]

    def close(self) -> None:
        with self._lock:
            self.db.close()

    # ------------------------------------------------------------------ turns
    def append(self, persona: str, user: str, assistant: str, session: str) -> None:
        turn = Turn(persona, user, assistant, session, time.time())
        with self._lock:
            self._insert(turn)
            self.db.commit()
        line = json.dumps(asdict(turn), ensure_ascii=False) + "\n"
        for name in (persona, "timeline"):
            with open(self.root / f"{name}.jsonl", "a", encoding="utf-8") as f:
                f.write(line)

    def own(self, persona: str, n: int) -> list[Turn]:
        rows = self._turns("SELECT persona, session, ts, user, assistant FROM turns WHERE persona=? "
                           "ORDER BY id DESC LIMIT ?", (persona, max(n, 0)))
        return rows[::-1]

    def shared(self, exclude: str, n: int) -> list[Turn]:
        """Recent turns from *other* personas."""
        rows = self._turns("SELECT persona, session, ts, user, assistant FROM turns WHERE persona<>? "
                           "ORDER BY id DESC LIMIT ?", (exclude, max(n, 0)))
        return rows[::-1]

    def recent_text(self, session: str, n: int = 2) -> str:
        rows = self._turns("SELECT persona, session, ts, user, assistant FROM turns WHERE session=? "
                           "ORDER BY id DESC LIMIT ?", (session, n))
        return "\n\n".join(t.chunk() for t in rows[::-1])

    def session_turns(self, session: str, n: int = 2) -> list[Turn]:
        rows = self._turns("SELECT persona, session, ts, user, assistant FROM turns WHERE session=? "
                           "ORDER BY id DESC LIMIT ?", (session, n))
        return rows[::-1]

    def search(self, text: str, n: int = 10) -> list[Turn]:
        """Turns from any time that share words with `text`, best match first (BM25)."""
        q = fts_query(text)
        if not q or n <= 0:
            return []
        return self._turns("SELECT t.persona, t.session, t.ts, t.user, t.assistant FROM turns_fts f "
                           "JOIN turns t ON t.id = f.rowid WHERE turns_fts MATCH ? ORDER BY bm25(turns_fts) "
                           "LIMIT ?", (q, n))

    def count(self) -> int:
        with self._lock:
            return self.db.execute("SELECT count(*) FROM turns").fetchone()[0]

    # ------------------------------------------------------------------ facts
    def add_fact(self, text: str, source: str = "") -> int:
        with self._lock:
            cur = self.db.execute("INSERT INTO facts(ts, text, source) VALUES (?,?,?)", (time.time(), text, source))
            self.db.execute("INSERT INTO facts_fts(rowid, text) VALUES (?,?)", (cur.lastrowid, text))
            self.db.commit()
            return cur.lastrowid

    def retire_fact(self, fact_id: int) -> None:
        with self._lock:
            self.db.execute("UPDATE facts SET active=0 WHERE id=?", (fact_id,))
            self.db.commit()

    def facts(self, text: str | None = None, n: int = 8) -> list[Fact]:
        """Active facts matching `text` (all recent ones when text is None)."""
        with self._lock:
            if text is None:
                rows = self.db.execute("SELECT id, text, ts, source FROM facts WHERE active=1 "
                                       "ORDER BY id DESC LIMIT ?", (n,)).fetchall()
            else:
                q = fts_query(text)
                rows = [] if not q else self.db.execute(
                    "SELECT f.id, f.text, f.ts, f.source FROM facts_fts x JOIN facts f ON f.id = x.rowid "
                    "WHERE facts_fts MATCH ? AND f.active=1 ORDER BY bm25(facts_fts) LIMIT ?", (q, n)).fetchall()
        return [Fact(*r) for r in rows]

    # ------------------------------------------------------------------ what Jev chooses from
    def candidates(self, persona: str, n_own: int, n_shared: int, request: str | None = None,
                   n_search: int = 0, n_facts: int = 0) -> list[str]:
        """Chunks Jev may choose from for this persona: recent ones oldest first, then older
        turns and facts that share words with the request. De-duplicated."""
        turns = sorted(self.own(persona, n_own) + self.shared(persona, n_shared), key=lambda t: t.ts)
        if request and n_search:
            turns += self.search(request, n_search)
        facts = []
        if request and n_facts:  # matching facts, topped up with the newest (meaning beats wording)
            facts = self.facts(request, n_facts)
            ids = {f.id for f in facts}
            facts += [f for f in self.facts(None, n_facts) if f.id not in ids][:max(0, n_facts - len(facts))]
        seen, out = set(), []
        for c in [t.chunk() for t in turns] + [f.chunk() for f in facts]:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out
