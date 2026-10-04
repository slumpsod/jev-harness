"""What LM Studio's llama.cpp did with each request, read from its server log (a bench helper).

LM Studio keeps its log in ~/.lmstudio/server-logs/<year-month>/<date>.<n>.log (set LMSTUDIO_LOG_DIR to point
elsewhere). For every request llama.cpp prints which slot took it and how much of the prompt that slot already held
("selected slot by LCP similarity, f_sim_best = 0.94": the share of the new prompt shared with the slot's cache),
how many prompt tokens it then had to read ("prompt eval time = ... / 183 tokens"), what it generated, and when a
client hung up ("stop: cancel task"). That is exactly what a redirect or a quick reply costs the server, so the
bench reads it back. Without LM Studio's log (another server) the numbers are simply missing.

    python evals/bench/lmstudio_log.py 2026-10-03 23:17:00 23:18:30     # the requests of that window
"""
from __future__ import annotations

import glob
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable

STAMP = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]")
SELECT = re.compile(r"slot get_availabl: id\s+(\d+) \| task -1 \| selected slot by (\w+)(?: similarity)?"
                    r"(?:, f_sim_best = ([\d.]+) \(> [\d.]+ thold\), f_keep = ([\d.]+))?")
LAUNCH = re.compile(r"slot launch_slot_: id\s+(\d+) \| task (\d+) \| processing task")
PROMPT = re.compile(r"slot print_timing: id\s+(\d+) \| task (\d+) \| prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
EVAL = re.compile(r"slot print_timing: id\s+(\d+) \| task (\d+) \|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens "
                  r"\(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second")
RELEASE = re.compile(r"slot\s+release: id\s+(\d+) \| task (\d+) \| stop processing: n_tokens = (\d+)")
CANCEL = re.compile(r"srv\s+stop: cancel task, id_task = (\d+)")


def parse(lines: Iterable[str]) -> list[dict]:
    """The requests in these log lines, oldest first: {slot, task, at (launch time), how ("LCP" | "LRU"), sim, keep,
    processed (prompt tokens read), generated, gen_tps, n_tokens (the slot's context at the end), cancelled}."""
    tasks: dict[int, dict] = {}
    pending: dict[int, dict] = {}   # slot -> its latest "selected slot" line, for the task that launches next
    order: list[int] = []
    stamp = ""
    for line in lines:
        m = STAMP.match(line)
        if m:
            stamp = m.group(1)
        if (m := SELECT.search(line)):
            pending[int(m.group(1))] = {"how": "LCP" if m.group(2).upper().startswith("LCP") else m.group(2).upper(),
                                        "sim": float(m.group(3)) if m.group(3) else None,
                                        "keep": float(m.group(4)) if m.group(4) else None}
        elif (m := LAUNCH.search(line)):
            slot, task = int(m.group(1)), int(m.group(2))
            tasks[task] = {"slot": slot, "task": task, "at": stamp, **pending.pop(slot, {}), "processed": None,
                           "generated": None, "gen_tps": None, "n_tokens": None, "cancelled": False}
            order.append(task)
        elif (m := PROMPT.search(line)) and int(m.group(2)) in tasks:
            tasks[int(m.group(2))]["processed"] = int(m.group(4))
        elif (m := EVAL.search(line)) and int(m.group(2)) in tasks:
            tasks[int(m.group(2))].update(generated=int(m.group(4)), gen_tps=float(m.group(5)))
        elif (m := RELEASE.search(line)) and int(m.group(2)) in tasks:
            tasks[int(m.group(2))]["n_tokens"] = int(m.group(3))
        elif (m := CANCEL.search(line)) and int(m.group(1)) in tasks:
            tasks[int(m.group(1))]["cancelled"] = True
    return [tasks[t] for t in order]


def prompt_tokens(t: dict) -> int | None:
    """The request's prompt length: the slot's context at the end less what was generated (None if unknown)."""
    if t.get("n_tokens") is None:
        return None
    return t["n_tokens"] - (t.get("generated") or 0)


def log_files(day_from: str, day_to: str | None = None) -> list[str]:
    root = Path(os.environ.get("LMSTUDIO_LOG_DIR") or Path.home() / ".lmstudio" / "server-logs")
    days = {day_from[:10], (day_to or day_from)[:10]}
    return sorted(f for d in days for f in glob.glob(str(root / "**" / f"{d}*.log"), recursive=True))


def between(start: float, end: float) -> list[dict]:
    """The requests LM Studio launched between two times (time.time() values)."""
    a, b = (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)) for x in (start, end))
    lines: list[str] = []
    for f in log_files(a, b):
        lines += Path(f).read_text(encoding="utf-8", errors="replace").splitlines()
    got = parse(lines)
    return [t for t in got if a <= t["at"] <= b]


if __name__ == "__main__":
    day, t1, t2 = sys.argv[1:4]
    s = time.mktime(time.strptime(f"{day} {t1}", "%Y-%m-%d %H:%M:%S"))
    e = time.mktime(time.strptime(f"{day} {t2}", "%Y-%m-%d %H:%M:%S"))
    for t in between(s, e):
        pl = prompt_tokens(t)
        print(f"{t['at'][11:]} slot {t['slot']} task {t['task']:>6} {t.get('how', '?'):3s} sim {t.get('sim')} "
              f"read {t['processed']} of {pl} prompt tokens, generated {t['generated']}"
              + (" (cancelled)" if t["cancelled"] else ""))
