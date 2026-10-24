"""Cron jobs: tasks the assistant does on a schedule, without you asking each time ("every weekday at 8:30, summarize
my unread mail", "every Sunday evening, check the llama.cpp releases"). The dashboard's Cron page lists them, makes
them, pauses them and shows what each run did.

workspace/cron.db
  jobs   one per job: its name, the request (`prompt`) it sends, when (`schedule`, a cron expression), the assistant it
         is pinned to (or none: Jev routes it as for any request), whether it is on, and when it runs next
  runs   one per run: when it was due, started and finished, how it went, the session it ran in (so History and the
         Review page have it like any other session) and the answer

A run is a turn of its own: a fresh Harness and session, so a job never reads or disturbs the conversation in the chat. It
sees what any new session sees (memory, skills, the tools its assistant has). Nobody is there to answer, so a worker that
asks the user something is told to use its best judgement.

The dashboard's `Runner` checks every `tick` seconds. A job whose time has come is queued and run one at a time (local
models serve one task well, not several). It waits while a turn of yours is running, as long as `grace_minutes`; past that
the run is recorded as MISSED and not made up later (a job due every morning that finds the dashboard off at 8:30 should
not run at noon). Two dashboards on one workspace do not both run a job: whoever moves `next_run` on gets the run.
Without a dashboard running nothing runs; `jevh cron run <id>` runs one job by hand.

    jevh cron                                    # the jobs
    jevh cron add "Morning mail" "Summarize my unread mail" --schedule "weekdays at 8:30"
    jevh cron run 1                              # run it now, here
    jevh cron runs 1                             # what it did
    jevh cron pause 1 | resume 1 | rm 1

A SCHEDULE is a cron expression (`minute hour day-of-month month day-of-week`, local time: `*/15 * * * *`, `30 8 * * 1-5`,
`@daily`) or a plain phrase `from_text` understands ("every 15 minutes", "hourly", "daily at 9am", "weekdays at 8:30",
"every monday at 14:00", "monthly on the 1st at noon"). Jobs are stored as the expression.

Settings (config.yaml `cron:`): see there. NOT VERIFIED on live models yet: `python evals/bench/cron.py`.
"""
from __future__ import annotations

import argparse
import queue
import re
import sqlite3
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from . import store

DEFAULTS = {"enabled": True, "tick": 20, "grace_minutes": 10, "min_minutes": 5, "keep_runs": 30, "max_jobs": 40,
            "timeout_minutes": 60}
NAME_CAP, PROMPT_CAP, ANSWER_CAP = 80, 8000, 6000
STALE_SECONDS = 180   # a run whose process has not marked it alive for this long has been lost (a restart, a crash)
NOBODY = ("(this is a scheduled job and nobody is here to answer: use your best judgement; if you really cannot go on "
          "without something from them, finish and say what you need)")
FINISHED = ("ok", "error", "cancelled", "interrupted")   # a run that did its work or tried to (not missed or skipped)
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DAYS = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"]
DAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
MACROS = {"@yearly": "0 0 1 1 *", "@annually": "0 0 1 1 *", "@monthly": "0 0 1 * *", "@weekly": "0 0 * * 0",
          "@daily": "0 0 * * *", "@midnight": "0 0 * * *", "@hourly": "0 * * * *"}


def settings(raw: dict | None) -> dict:
    return {**DEFAULTS, **((raw or {}).get("cron") or {})}


class CronError(ValueError):
    """Something the user can fix: a schedule that is not one, a job that is not there. Its text is shown as it is."""


# ---------------------------------------------------------------- schedules

def _field(text: str, lo: int, hi: int, names: list[str] | None, what: str) -> tuple[frozenset[int], bool]:
    """One cron field -> (the values it allows, whether it starts with `*`)."""
    values: set[int] = set()

    def number(s: str) -> int:
        s = s.strip().lower()
        if names and s in names:
            return names.index(s) + (lo if what == "month" else 0)
        if not s.isdigit():
            raise CronError(f"{what}: '{s}' is not a number" + (f" or a name ({', '.join(names)})" if names else ""))
        return int(s)

    for part in text.split(","):
        if not part:
            raise CronError(f"{what}: an empty entry in '{text}'")
        rng, _, step_s = part.partition("/")
        step = 1
        if step_s:
            if not step_s.isdigit() or int(step_s) < 1:
                raise CronError(f"{what}: the step in '{part}' must be a positive number")
            step = int(step_s)
        if rng == "*":
            a, b = lo, hi
        elif "-" in rng:
            x, _, y = rng.partition("-")
            a, b = number(x), number(y)
        else:
            a = number(rng)
            b = hi if step_s else a   # ("5/10" = from 5 to the end, every 10th)
        top = 7 if what == "day of week" else hi
        if not (lo <= a <= top and lo <= b <= top) or a > b:
            raise CronError(f"{what}: '{part}' is outside {lo}-{top}")
        values.update(range(a, b + 1, step))
    if what == "day of week" and 7 in values:
        values.discard(7)
        values.add(0)
    return frozenset(values), text.startswith("*")


class CronExpr:
    """A parsed cron expression. Local time, as cron is."""

    def __init__(self, text: str):
        text = " ".join(str(text).split())
        text = MACROS.get(text.lower(), text)
        parts = text.split(" ")
        if len(parts) != 5:
            raise CronError(f"'{text}' is not a schedule: a cron expression has five fields (minute hour day-of-month month "
                            f"day-of-week), like '30 8 * * 1-5'")
        self.text = text
        self.minutes, _ = _field(parts[0], 0, 59, None, "minute")
        self.hours, _ = _field(parts[1], 0, 23, None, "hour")
        self.doms, self.dom_star = _field(parts[2], 1, 31, None, "day of month")
        self.months, _ = _field(parts[3], 1, 12, MONTHS, "month")
        self.dows, self.dow_star = _field(parts[4], 0, 6, DAYS, "day of week")

    def _day_ok(self, t: datetime) -> bool:
        dom, dow = t.day in self.doms, (t.weekday() + 1) % 7 in self.dows
        return (dom and dow) if (self.dom_star or self.dow_star) else (dom or dow)   # (cron: both given -> either)

    def next_after(self, after: float) -> float:
        """The first minute strictly after `after` (a timestamp) that the expression allows."""
        t = datetime.fromtimestamp(after).replace(second=0, microsecond=0) + timedelta(minutes=1)
        limit = t + timedelta(days=366 * 8)
        while t < limit:
            if t.month not in self.months:
                t = datetime(t.year + (t.month == 12), t.month % 12 + 1, 1)
            elif not self._day_ok(t):
                t = datetime(t.year, t.month, t.day) + timedelta(days=1)
            elif t.hour not in self.hours:
                t = t.replace(minute=0) + timedelta(hours=1)
            else:
                m = next((m for m in sorted(self.minutes) if m >= t.minute), None)
                if m is None:
                    t = t.replace(minute=0) + timedelta(hours=1)
                    continue
                ts = t.replace(minute=m).timestamp()
                if ts > after:
                    return ts
                t = t.replace(minute=m) + timedelta(minutes=1)   # (a clock change put it back: go on)
        raise CronError(f"'{self.text}' never runs (a day that does not exist, like February 30?)")

    def min_gap(self, now: float, samples: int = 5) -> float:
        """The shortest time between two of the next few runs, in seconds."""
        ts, last, gap = now, None, float("inf")
        for _ in range(samples):
            ts = self.next_after(ts)
            if last is not None:
                gap = min(gap, ts - last)
            last = ts
        return gap


def _clock(text: str) -> tuple[int, int]:
    """'9', '9:30', '9am', '9:30 pm', '17:00', 'noon', 'midnight' -> (hour, minute)."""
    s = text.strip().lower().replace(".", "")
    if s == "noon":
        return 12, 0
    if s == "midnight":
        return 0, 0
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", s)
    if not m:
        raise CronError(f"'{text}' is not a time of day (try 8:30, 9am, 17:00, noon)")
    h, mi, ap = int(m[1]), int(m[2] or 0), m[3]
    if ap:
        if not 1 <= h <= 12:
            raise CronError(f"'{text}' is not a time of day")
        h = h % 12 + (12 if ap == "pm" else 0)
    if h > 23 or mi > 59:
        raise CronError(f"'{text}' is not a time of day")
    return h, mi


def _day_list(words: str) -> str:
    out = []
    for w in re.split(r"[,\s]+(?:and\s+)?|\s+and\s+", words.strip().lower()):
        if not w:
            continue
        key = w[:3]
        if key not in DAYS or not DAY_NAMES[DAYS.index(key)].lower().startswith(w.rstrip("s")):
            raise CronError(f"'{w}' is not a day of the week")
        out.append(str(DAYS.index(key)))
    if not out:
        raise CronError("which day of the week?")
    return ",".join(sorted(set(out)))


def from_text(text: str) -> str:
    """A cron expression, or a phrase for one -> the cron expression (checked)."""
    raw = " ".join(str(text or "").split())
    if not raw:
        raise CronError("say when: a cron expression like '30 8 * * 1-5' or a phrase like 'weekdays at 8:30'")
    s = raw.lower().rstrip(".")
    if s.startswith("@") or re.fullmatch(r"[\d*/,\-]+ [\d*/,\-]+ [\d*/,\-]+ [\w*/,\-]+ [\w*/,\-]+", raw):
        return CronExpr(raw).text
    m = re.fullmatch(r"(?:every|each)\s+(\d+)\s*(?:min(?:ute)?s?|m)", s)
    if m:
        n = int(m[1])
        if n >= 60 and n % 60 == 0:
            return from_text(f"every {n // 60} hours")
        if not 1 <= n <= 59:
            raise CronError("every N minutes: N must be 1 to 59 (or a whole number of hours)")
        return CronExpr(f"*/{n} * * * *").text
    m = re.fullmatch(r"(?:every|each)\s+(\d+)\s*(?:h|hrs?|hours?)(?:\s+at\s+:?(\d{1,2}))?", s)
    if m:
        n, mi = int(m[1]), int(m[2] or 0)
        if not 1 <= n <= 23:
            raise CronError("every N hours: N must be 1 to 23")
        return CronExpr(f"{mi} {'*' if n == 1 else f'*/{n}'} * * *").text
    if s in ("hourly", "every hour", "each hour"):
        return "0 * * * *"
    m = re.fullmatch(r"(?:hourly|every hour|each hour) at :?(\d{1,2})", s)
    if m:
        return CronExpr(f"{int(m[1])} * * * *").text
    if s in ("daily", "every day", "each day"):
        return "0 9 * * *"   # (no time said: 9 in the morning, as for weekdays and the rest)
    if s in ("nightly", "every night"):
        return "0 0 * * *"
    m = re.fullmatch(r"(?:daily|every day|each day|nightly|every night|every morning|every evening)?\s*at\s+(.+)", s)
    if m:
        h, mi = _clock(m[1])
        return CronExpr(f"{mi} {h} * * *").text
    m = re.fullmatch(r"(weekdays?|week days?|weekends?|every weekday|every weekend)(?:\s+at\s+(.+))?", s)
    if m:
        h, mi = _clock(m[2] or "9:00")
        return CronExpr(f"{mi} {h} * * {'1-5' if 'weekday' in m[1] or 'week day' in m[1] else '6,0'}").text
    m = re.fullmatch(r"(?:monthly|every month)(?:\s+on(?:\s+the)?\s+(\d{1,2})(?:st|nd|rd|th)?)?(?:\s+at\s+(.+))?", s)
    if m:
        h, mi = _clock(m[2] or "9:00")
        return CronExpr(f"{mi} {h} {int(m[1] or 1)} * *").text
    m = re.fullmatch(r"(?:every|each|on)?\s*([a-z, ]+?)(?:\s+(?:at|@)\s+(.+))?", s)
    if m and any(m[1].strip().startswith(d) for d in DAYS):
        dows = _day_list(m[1])
        h, mi = _clock(m[2] or "9:00")
        return CronExpr(f"{mi} {h} * * {dows}").text
    raise CronError(f"I could not read '{raw}' as a schedule. Use a cron expression ('30 8 * * 1-5') or a phrase like "
                    f"'every 15 minutes', 'daily at 9am', 'weekdays at 8:30', 'every monday at 14:00', 'monthly on the 1st'")


def _hm(h: int, m: int) -> str:
    if m == 0 and h in (0, 12):
        return "midnight" if h == 0 else "noon"
    return f"{(h % 12) or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def _one(values: frozenset[int]) -> int | None:
    return next(iter(values)) if len(values) == 1 else None


def describe(expr: str) -> str:
    """A cron expression in words, for the page; an unusual one is shown as it is."""
    try:
        e = CronExpr(expr)
    except CronError:
        return expr
    parts = e.text.split(" ")
    mi, hr, dom, mon, dow = parts
    m, h = _one(e.minutes), _one(e.hours)
    plain_day = dom == "*" and mon == "*"
    if mon == "*" and dom == "*" and dow == "*":
        if mi.startswith("*/") and hr == "*" and mi[2:].isdigit():
            n = int(mi[2:])
            return "every minute" if n == 1 else f"every {n} minutes"
        if mi == "*" and hr == "*":
            return "every minute"
        if m is not None and hr == "*":
            return "every hour" if m == 0 else f"every hour at :{m:02d}"
        if m is not None and hr.startswith("*/") and hr[2:].isdigit():
            return f"every {int(hr[2:])} hours" + (f" at :{m:02d}" if m else "")
        if m is not None and h is not None:
            return f"every day at {_hm(h, m)}"
    if m is not None and h is not None and plain_day:
        days = sorted(e.dows)
        if days == [1, 2, 3, 4, 5]:
            return f"weekdays at {_hm(h, m)}"
        if days == [0, 6]:
            return f"weekends at {_hm(h, m)}"
        if len(days) < 7:
            names = [DAY_NAMES[d] for d in days]
            return f"every {', '.join(names[:-1]) + ' and ' + names[-1] if len(names) > 1 else names[0]} at {_hm(h, m)}"
    if m is not None and h is not None and mon == "*" and dow == "*" and _one(e.doms) is not None:
        d = _one(e.doms)
        suffix = "th" if 10 <= d % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")
        return f"monthly on the {d}{suffix} at {_hm(h, m)}"
    return f"cron {e.text}"


def check(schedule: str, min_minutes: int = 0, now: float | None = None) -> str:
    """The schedule as a cron expression, or a CronError saying what is wrong with it (including: it fires more often
    than `min_minutes`)."""
    text = from_text(schedule)
    e = CronExpr(text)
    now = now or time.time()
    e.next_after(now)   # (raises when it never runs)
    if min_minutes and e.min_gap(now) < min_minutes * 60:
        raise CronError(f"'{text}' runs more often than every {min_minutes} minutes, the shortest a job may repeat "
                        f"(cron.min_minutes in config.yaml). A model needs minutes to do a task")
    return text


# ---------------------------------------------------------------- the store

def ago(ts: float | None, now: float | None = None) -> str:
    if not ts:
        return ""
    d = max(0, (now or time.time()) - ts)
    for n, u in ((86400, "d"), (3600, "h"), (60, "m")):
        if d >= n:
            return f"{int(d // n)}{u} ago"
    return "just now"


class CronJobs:
    """The jobs and their runs, in SQLite (safe across threads and across processes)."""

    def __init__(self, path: Path, cfg: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self._lock = threading.RLock()
        self.db = store.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, prompt TEXT, schedule TEXT,
                persona TEXT DEFAULT '', enabled INTEGER DEFAULT 1, created REAL, next_run REAL);
            CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER, trigger TEXT, due REAL,
                started REAL, finished REAL, status TEXT, session TEXT DEFAULT '', answer TEXT DEFAULT '',
                error TEXT DEFAULT '', seconds REAL DEFAULT 0, seen INTEGER DEFAULT 0, beat REAL);
            CREATE INDEX IF NOT EXISTS runs_job ON runs(job_id, id);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value INTEGER);
            INSERT OR IGNORE INTO meta VALUES('rev', 0);
        """)
        self.db.commit()

    # -- plumbing ----------------------------------------------------------------------------
    def _run(self, sql: str, args=(), touch: bool = True) -> sqlite3.Cursor:
        with self._lock:
            cur = self.db.execute(sql, args)
            if touch:
                self.db.execute("UPDATE meta SET value=value+1 WHERE key='rev'")
            self.db.commit()
            return cur

    def _all(self, sql: str, args=()) -> list[sqlite3.Row]:
        with self._lock:
            return self.db.execute(sql, args).fetchall()

    def rev(self) -> int:
        """Goes up on every change, by any process: a page that holds the last one knows nothing changed."""
        return self._all("SELECT value FROM meta WHERE key='rev'")[0][0]

    def close(self) -> None:
        with self._lock:
            self.db.close()

    # -- jobs --------------------------------------------------------------------------------
    @staticmethod
    def _job(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["enabled"] = bool(d["enabled"])
        d["persona"] = d["persona"] or None
        d["human"] = describe(d["schedule"])
        return d

    def _fields(self, name: Any, prompt: Any, schedule: Any, now: float) -> tuple[str, str, str]:
        name, prompt = " ".join(str(name or "").split()), str(prompt or "").strip()
        if not name:
            raise CronError("give the job a name")
        if not prompt:
            raise CronError("say what the assistant should do each time")
        if len(name) > NAME_CAP:
            raise CronError(f"the name is longer than {NAME_CAP} characters")
        if len(prompt) > PROMPT_CAP:
            raise CronError(f"the request is longer than {PROMPT_CAP} characters")
        return name, prompt, check(schedule, int(self.cfg["min_minutes"]), now)

    def add(self, name: str, prompt: str, schedule: str, *, persona: str | None = None, enabled: bool = True,
            now: float | None = None) -> dict:
        now = now or time.time()
        name, prompt, sched = self._fields(name, prompt, schedule, now)
        if self._all("SELECT COUNT(*) FROM jobs")[0][0] >= int(self.cfg["max_jobs"]):
            raise CronError(f"there are already {self.cfg['max_jobs']} jobs (cron.max_jobs in config.yaml): remove one first")
        cur = self._run("INSERT INTO jobs(name, prompt, schedule, persona, enabled, created, next_run) VALUES (?,?,?,?,?,?,?)",
                        (name, prompt, sched, persona or "", int(bool(enabled)), now,
                         CronExpr(sched).next_after(now) if enabled else None))
        return self.job(cur.lastrowid)

    def job(self, jid: int) -> dict | None:
        rows = self._all("SELECT * FROM jobs WHERE id=?", (jid,))
        return self._job(rows[0]) if rows else None

    def need(self, jid: int) -> dict:
        j = self.job(jid)
        if j is None:
            raise CronError(f"no job {jid}")
        return j

    def jobs(self) -> list[dict]:
        """Every job, with whether it is running (or waiting to), and how its last run went."""
        out = []
        busy = {r["job_id"] for r in self._all("SELECT DISTINCT job_id FROM runs WHERE status IN ('queued','running')")}
        for r in self._all("SELECT * FROM jobs ORDER BY id"):
            j = self._job(r)
            j["running"] = j["id"] in busy
            last = self._all("SELECT id, status, due, started, finished, seconds, session, error, seen FROM runs "
                             "WHERE job_id=? AND status NOT IN ('queued','running') ORDER BY id DESC LIMIT 1", (j["id"],))
            j["last"] = dict(last[0]) if last else None
            j["unseen"] = self._all("SELECT COUNT(*) FROM runs WHERE job_id=? AND seen=0 AND status IN "
                                    "('ok','error','interrupted')", (j["id"],))[0][0]
            out.append(j)
        return out

    def update(self, jid: int, *, name: Any = None, prompt: Any = None, schedule: Any = None, persona: Any = "keep",
               enabled: Any = None, now: float | None = None) -> dict:
        now = now or time.time()
        j = self.need(jid)
        name, prompt, sched = self._fields(j["name"] if name is None else name, j["prompt"] if prompt is None else prompt,
                                           j["schedule"] if schedule is None else schedule, now)
        on = j["enabled"] if enabled is None else bool(enabled)
        persona = j["persona"] if persona == "keep" else (persona or None)
        # a job that was off, or whose schedule changed, is next due from now; one that only got a new name keeps its turn
        nxt = (j["next_run"] if on and j["enabled"] and sched == j["schedule"] and j["next_run"]
               else CronExpr(sched).next_after(now) if on else None)
        self._run("UPDATE jobs SET name=?, prompt=?, schedule=?, persona=?, enabled=?, next_run=? WHERE id=?",
                  (name, prompt, sched, persona or "", int(on), nxt, jid))
        return self.need(jid)

    def delete(self, jid: int) -> bool:
        with self._lock:
            if not self.db.execute("DELETE FROM jobs WHERE id=?", (jid,)).rowcount:
                return False
            self.db.execute("DELETE FROM runs WHERE job_id=?", (jid,))
            self.db.execute("UPDATE meta SET value=value+1 WHERE key='rev'")
            self.db.commit()
        return True

    # -- what is due -------------------------------------------------------------------------
    def due(self, now: float | None = None) -> list[dict]:
        now = now or time.time()
        return [self._job(r) for r in self._all(
            "SELECT * FROM jobs WHERE enabled=1 AND next_run IS NOT NULL AND next_run<=? ORDER BY next_run, id", (now,))]

    def claim(self, job: dict, now: float | None = None) -> bool:
        """Move a due job on to its next time. True for the one process that did: it owns this run (another dashboard on
        the same workspace sees the new time and leaves the job alone)."""
        now = now or time.time()
        nxt = CronExpr(job["schedule"]).next_after(now)
        cur = self._run("UPDATE jobs SET next_run=? WHERE id=? AND enabled=1 AND next_run=?", (nxt, job["id"], job["next_run"]))
        return cur.rowcount == 1

    # -- runs --------------------------------------------------------------------------------
    @staticmethod
    def _runrow(r: sqlite3.Row) -> dict:
        d = dict(r)
        d.pop("beat", None)
        return d

    def add_run(self, job_id: int, status: str, *, trigger: str = "schedule", due: float | None = None, error: str = "",
                now: float | None = None) -> int:
        now = now or time.time()
        finished = now if status not in ("queued", "running") else None
        cur = self._run("INSERT INTO runs(job_id, trigger, due, started, finished, status, error, beat, seen) VALUES (?,?,?,?,?,?,?,?,?)",
                        (job_id, trigger, due or now, now if status == "running" else None, finished, status, error, now,
                         0 if status in ("queued", "running") else int(status in ("missed", "skipped"))))
        self._trim(job_id)
        return cur.lastrowid

    def _trim(self, job_id: int) -> None:
        keep = int(self.cfg["keep_runs"])
        self._run("DELETE FROM runs WHERE job_id=? AND status NOT IN ('queued','running') AND id NOT IN "
                  "(SELECT id FROM runs WHERE job_id=? AND status NOT IN ('queued','running') ORDER BY id DESC LIMIT ?)",
                  (job_id, job_id, keep), touch=False)

    def start_run(self, run_id: int, now: float | None = None) -> None:
        now = now or time.time()
        self._run("UPDATE runs SET status='running', started=?, beat=? WHERE id=? AND status='queued'", (now, now, run_id))

    def set_session(self, run_id: int, session: str) -> None:
        self._run("UPDATE runs SET session=? WHERE id=?", (session, run_id))

    def finish_run(self, run_id: int, status: str, answer: str = "", error: str = "", session: str | None = None,
                   now: float | None = None) -> None:
        now = now or time.time()
        with self._lock:
            row = self.db.execute("SELECT started FROM runs WHERE id=?", (run_id,)).fetchone()
        started = (row["started"] if row else None) or now
        self._run("UPDATE runs SET status=?, finished=?, seconds=?, answer=?, error=?, session=COALESCE(?, session) WHERE id=?",
                  (status, now, max(0.0, now - started), answer[:ANSWER_CAP], error[:1000], session, run_id))

    def beat(self, run_ids: list[int], now: float | None = None) -> None:
        if run_ids:
            now = now or time.time()
            with self._lock:
                self.db.executemany("UPDATE runs SET beat=? WHERE id=?", [(now, i) for i in run_ids])
                self.db.commit()

    def reap(self, now: float | None = None) -> int:
        """Runs nothing is alive to finish (their dashboard was closed or crashed) are ended as `interrupted`."""
        now = now or time.time()
        n = self._run("UPDATE runs SET status='interrupted', finished=?, error='the dashboard stopped before it finished' "
                      "WHERE status IN ('queued','running') AND beat<?", (now, now - STALE_SECONDS), touch=False).rowcount
        if n:
            self._run("UPDATE meta SET value=value+1 WHERE key='rev'", touch=False)
        return n

    def run(self, run_id: int) -> dict | None:
        rows = self._all("SELECT * FROM runs WHERE id=?", (run_id,))
        return self._runrow(rows[0]) if rows else None

    def runs(self, job_id: int, limit: int = 20) -> list[dict]:
        return [self._runrow(r) for r in self._all("SELECT * FROM runs WHERE job_id=? ORDER BY id DESC LIMIT ?", (job_id, limit))]

    def active(self, job_id: int) -> bool:
        return bool(self._all("SELECT 1 FROM runs WHERE job_id=? AND status IN ('queued','running') LIMIT 1", (job_id,)))

    def running(self) -> int:
        """How many runs are going now (a restart waits for them)."""
        return self._all("SELECT COUNT(*) FROM runs WHERE status='running'")[0][0]

    def mark_seen(self, job_id: int | None = None) -> None:
        self._run("UPDATE runs SET seen=1 WHERE seen=0" + (" AND job_id=?" if job_id else ""), (job_id,) if job_id else ())

    def unseen(self) -> int:
        return self._all("SELECT COUNT(*) FROM runs WHERE seen=0 AND status IN ('ok','error','interrupted')")[0][0]


# ---------------------------------------------------------------- running a job

def run_job(job: dict, make_harness: Callable[[Callable[[str, dict], None]], Any], cancel: threading.Event,
            timeout: float = 3600.0, on_session: Callable[[str], None] | None = None) -> dict:
    """Do one job on a harness of its own: -> {status, answer, error, session}. `make_harness(on_event)` builds it (a fresh
    session). Never raises."""
    holder: dict[str, Any] = {}
    timed_out = threading.Event()

    def on_event(kind: str, data: dict) -> None:
        if kind == "question" and holder.get("h") is not None:   # nobody to ask: carry on
            try:
                holder["h"].answer(data["question"]["id"], NOBODY)
            except Exception:  # noqa: BLE001
                pass
    try:
        h = holder["h"] = make_harness(on_event)
        if on_session:
            on_session(h.session)

        def stop(why: threading.Event | None = None) -> None:
            if why is not None:
                why.set()
            h.cancel()
        timer = threading.Timer(timeout, stop, (timed_out,))
        timer.daemon = True
        timer.start()
        done = threading.Event()

        def watch() -> None:   # the Stop button
            while not done.is_set():
                if cancel.wait(0.5):
                    h.cancel()
                    return
        threading.Thread(target=watch, name="cron-stop", daemon=True).start()
        try:
            r = h.turn(job["prompt"], force=job.get("persona") or None)
        finally:
            timer.cancel()
            done.set()
        lead = r.tasks[0] if r.tasks else None
        if timed_out.is_set():
            return {"status": "error", "answer": r.answer, "session": h.session,
                    "error": f"stopped after {int(timeout // 60)} minutes (cron.timeout_minutes)"}
        if r.cancelled:
            return {"status": "cancelled", "answer": r.answer, "session": h.session, "error": ""}
        if lead is not None and lead.status in ("failed", "rejected"):
            return {"status": "error", "answer": r.answer, "session": h.session, "error": "the assistant could not do it"}
        return {"status": "ok", "answer": r.answer, "session": h.session, "error": ""}
    except Exception as e:  # noqa: BLE001 - a job that cannot run is a failed run, never a dead timer
        h = holder.get("h")
        return {"status": "error", "answer": "", "session": getattr(h, "session", "") or "", "error": f"{type(e).__name__}: {e}"}


class Runner:
    """Starts the jobs whose time has come, every `tick` seconds, on a thread of its own, and runs them one at a time.
    `execute(job, cancel, on_session) -> {status, answer, error, session}` does a job (the dashboard gives `run_job`);
    `ready()` says whether now is a good time (no turn of the user's is running); `on_change()` is called when a run
    started or ended, `on_done(job, run)` when one ended."""

    def __init__(self, jobs: CronJobs, execute: Callable[..., dict], ready: Callable[[], bool] | None = None,
                 on_change: Callable[[], None] | None = None, on_done: Callable[[dict, dict], None] | None = None,
                 tick: float | None = None):
        self.jobs, self.execute, self.ready, self.on_change, self.on_done = jobs, execute, ready, on_change, on_done
        self.tick = float(tick if tick is not None else jobs.cfg["tick"])
        self.grace = float(jobs.cfg["grace_minutes"]) * 60
        self.timeout = float(jobs.cfg["timeout_minutes"]) * 60
        self._q: queue.Queue = queue.Queue()
        self._mine: dict[int, tuple[int, threading.Event]] = {}   # job id -> (run id, its Stop)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._worker: threading.Thread | None = None

    # -- deciding ----------------------------------------------------------------------------
    def once(self, now: float | None = None) -> list[int]:
        """One check: the run ids of the jobs queued by it. (Missed and skipped runs are recorded too.)"""
        now = now or time.time()
        changed = bool(self.jobs.reap(now))
        with self._lock:
            self.jobs.beat([rid for rid, _ in self._mine.values()], now)
        queued = []
        for job in self.jobs.due(now):
            late = now - job["next_run"]
            if late <= self.grace and self.ready is not None and not self.ready():
                continue   # the user is in the middle of something: try again at the next tick (until it is too late)
            if not self.jobs.claim(job, now):
                continue   # another dashboard on this workspace has it
            changed = True
            if late > self.grace:
                self.jobs.add_run(job["id"], "missed", due=job["next_run"], now=now,
                                  error=f"it was due {ago(job['next_run'], now)}; the dashboard was off or busy until after "
                                        f"the {int(self.grace // 60)} minutes cron.grace_minutes allows")
            elif self.jobs.active(job["id"]):
                self.jobs.add_run(job["id"], "skipped", due=job["next_run"], now=now,
                                  error="the run before it was still going")
            else:
                queued.append(self._queue(job, "schedule", job["next_run"], now))
        if changed and self.on_change:
            self.on_change()
        return queued

    def run_now(self, job_id: int) -> int:
        """Queue a run by hand (the schedule is untouched). Raises CronError when it is already running."""
        job = self.jobs.need(job_id)
        if self.jobs.active(job_id):
            raise CronError(f"'{job['name']}' is already running")
        rid = self._queue(job, "manual", None, time.time())
        if self.on_change:
            self.on_change()
        return rid

    def stop_run(self, job_id: int) -> bool:
        with self._lock:
            mine = self._mine.get(job_id)
        if mine:
            mine[1].set()
            return True
        return False

    def _queue(self, job: dict, trigger: str, due: float | None, now: float) -> int:
        rid = self.jobs.add_run(job["id"], "queued", trigger=trigger, due=due, now=now)
        with self._lock:
            self._mine[job["id"]] = (rid, threading.Event())
        self._q.put((job["id"], rid))
        self._ensure_worker()
        return rid

    # -- doing -------------------------------------------------------------------------------
    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker and self._worker.is_alive():
                return
            self._stop.clear()
            self._worker = threading.Thread(target=self._work, name="cron-worker", daemon=True)
            self._worker.start()

    def _work(self) -> None:
        while not self._stop.is_set():
            try:
                jid, rid = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            with self._lock:
                cancel = self._mine.get(jid, (rid, threading.Event()))[1]
            job = self.jobs.job(jid)
            if job is None or cancel.is_set():   # deleted, or stopped, while it waited
                self.jobs.finish_run(rid, "cancelled", error="stopped before it started" if job else "the job was deleted")
            else:
                self.jobs.start_run(rid)
                if self.on_change:
                    self.on_change()
                try:
                    out = self.execute(job, cancel, lambda s, rid=rid: self.jobs.set_session(rid, s))
                except Exception as e:  # noqa: BLE001
                    out = {"status": "error", "answer": "", "error": f"{type(e).__name__}: {e}", "session": ""}
                self.jobs.finish_run(rid, out.get("status") or "error", out.get("answer") or "", out.get("error") or "",
                                     out.get("session") or None)
            with self._lock:
                if self._mine.get(jid, (None,))[0] == rid:
                    del self._mine[jid]
            run = self.jobs.run(rid) or {}
            if self.on_change:
                self.on_change()
            if self.on_done and job is not None and run.get("status") != "cancelled":
                try:
                    self.on_done(job, run)
                except Exception:  # noqa: BLE001
                    pass

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    self.once()
                except Exception:  # noqa: BLE001 - never let the timer die
                    pass
                self._stop.wait(self.tick)
        self._thread = threading.Thread(target=loop, name="cron", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            for _, ev in self._mine.values():
                ev.set()


# ---------------------------------------------------------------- the command line

def _line(j: dict) -> str:
    state = "paused" if not j["enabled"] else f"next {datetime.fromtimestamp(j['next_run']):%a %d %b %H:%M}" if j["next_run"] else "off"
    last = j.get("last")
    return f"{j['id']:>3}  {j['name']:<28} {j['human']:<30} {state:<22} " + (f"last: {last['status']} {ago(last['finished'])}" if last else "never run")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="jevh cron", description="Jobs the assistant does on a schedule (docs/architecture/23-cron.md)")
    sub = ap.add_subparsers(dest="cmd")
    add = sub.add_parser("add", help="add a job")
    add.add_argument("name")
    add.add_argument("prompt", help="what the assistant should do each time")
    add.add_argument("--schedule", "-s", required=True, help="a cron expression or a phrase: 'weekdays at 8:30'")
    add.add_argument("--as", dest="persona", default=None, help="pin the job to one assistant (default: Jev routes it)")
    for name, helptext in (("rm", "remove a job and its runs"), ("pause", "stop a job running until resumed"),
                           ("resume", "turn a paused job back on"), ("run", "run a job now, here, and print the answer"),
                           ("runs", "what a job's runs did")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("id", type=int)
    args = ap.parse_args(argv)
    from .config import load_config
    cfg = load_config()
    jobs = CronJobs(cfg.workspace / "cron.db", settings(cfg.raw))
    try:
        if args.cmd in (None, "list"):
            rows = jobs.jobs()
            print("\n".join(_line(j) for j in rows) if rows else "no cron jobs yet (jevh cron add ..., or the dashboard's Cron page)")
        elif args.cmd == "add":
            j = jobs.add(args.name, args.prompt, args.schedule, persona=args.persona)
            print(f"job {j['id']}: {j['name']} - {j['human']}")
        elif args.cmd == "rm":
            print("removed" if jobs.delete(args.id) else f"no job {args.id}")
        elif args.cmd in ("pause", "resume"):
            j = jobs.update(args.id, enabled=args.cmd == "resume")
            print(_line(next(x for x in jobs.jobs() if x["id"] == j["id"])))
        elif args.cmd == "runs":
            jobs.need(args.id)
            for r in jobs.runs(args.id):
                when = datetime.fromtimestamp(r["started"] or r["due"] or 0)
                print(f"{r['id']:>4}  {when:%a %d %b %H:%M}  {r['status']:<11} {r['seconds']:>5.0f}s  {r['session']}  {(r['error'] or r['answer'] or '')[:80]!r}")
        elif args.cmd == "run":
            from .orchestrator import Harness
            job = jobs.need(args.id)
            rid = jobs.add_run(job["id"], "running", trigger="manual")
            out = run_job(job, lambda on_event: Harness(cfg, on_event=on_event), threading.Event(),
                          float(cfg.raw.get("cron", {}).get("timeout_minutes", DEFAULTS["timeout_minutes"])) * 60)
            jobs.finish_run(rid, out["status"], out["answer"], out["error"], out["session"] or None)
            print(out["answer"] or out["error"])
            return 0 if out["status"] == "ok" else 1
    except CronError as e:
        print(f"jevh cron: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
