"""The thinking level (config `thinking.level`, the Settings tab), live: what each level costs and what it gets right.

Jed, 2026-10-24: "add in a thinking level selector? especially for the strata engine, qwen3.8 flash next seems to think a
lot on xhigh". auto (the old behaviour: nothing sent, Qwen3.8 on Strata thinks at xhigh) is the old arm; none, low,
medium and high are what the selector adds. Every level runs the same cases through the harness's own code paths.

Stages:
  questions  one model call per question, as a worker's model is built (make_llm + with_thinking): word problems,
             logic, date arithmetic, reading code, each with one right answer. Counted per level: right answers, seconds,
             thinking tokens.
  turns      full harness turns on everyday tasks (compute and report, write and test a function, fix a bug in a file,
             draft a note, read an email), routed as in the chat; graded on the result (the number, the file, the test
             passing, the words that must be there). Counted per level: passed, seconds, thinking and output tokens, model
             calls. The regression set is auto: it must do what it did before (no field is sent).

    python evals/bench/thinking_level.py                              # both stages, every level, 3 repeats (~1.5 h on Strata)
    python evals/bench/thinking_level.py --stage questions --reps 2
    python evals/bench/thinking_level.py --levels auto,medium --holdout
    python evals/bench/thinking_level.py --offline [raw.json]         # the summary again from saved results

Designed for: Qwen3.8-Flash-Next (IQ3_S) on Strata, which reads `reasoning_effort` (none, low, medium, high = xhigh); runs
on whatever models.strong says. Light Jev (Laya) for routing. Needs nothing secret, no account, no browser or desktop.
The cases are authored (the harness's own sessions are mostly tasks that need accounts or a screen); questions are
checked by exact answer, turns by what they leave behind. Results: runs/thinking_level/<stamp>/, thinking_level_raw.json
and thinking_level_summary.txt (a live run writes to workspace/evals/<backend>/; --offline re-sweeps the committed ones).
What would show a level is wrong to offer: it gets clearly fewer right than auto at no saving, or a saving that costs
most of the right answers.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))
import _out  # noqa: E402  (evals/_out.py: where a run's results go)

RUNS = HERE / "runs" / "thinking_level"
RAW = HERE / "thinking_level_raw.json"
SUMMARY = HERE / "thinking_level_summary.txt"
LEVELS = ("auto", "none", "low", "medium", "high")
LIMIT = 8 * 60

# (name, question, the right answer as a regex on the reply)
QUESTIONS = [
    ("pens", "A shop sells pens at 3 for $2 and notebooks at $1.75 each. I buy 7 pens and 3 notebooks and pay with a $20 "
             "bill. How much change do I get? Reply with just the amount.", r"10\.08"),
    ("trains", "A train leaves at 9:40 and travels 210 km at 84 km/h, then waits 25 minutes, then travels 96 km at 64 "
               "km/h. At what time does it arrive? Reply with just the time (HH:MM, 24-hour).", r"14:05"),
    ("dates", "What day of the week was 29 February 2024? Reply with just the day.", r"thursday"),
    ("ages", "Ana is twice as old as Ben was when Ana was as old as Ben is now. Ben is 24. How old is Ana? Reply with "
             "just the number.", r"\b32\b"),
    ("code", "What does this Python print?\n\nxs = [3, 1, 4, 1, 5, 9, 2, 6]\nprint(sum(x for i, x in enumerate(xs) if i % 2 "
             "== 0 and x > 2) * len(set(xs)))\n\nReply with just the number.", r"\b84\b"),
    ("insulate", "An attic is 38 ft by 26 ft. Blown insulation costs $1.85 per sq ft and a rebate pays $0.49 per sq ft up "
                 "to $890. What does the job cost after the rebate? Reply with just the amount.", r"1,?343\.68"),
    ("weighing", "You have 12 coins; one is lighter or heavier. With a balance scale, what is the fewest weighings that "
                 "always finds it and says whether it is lighter or heavier? Reply with just the number.", r"\b3\b"),
    ("percent", "A price rises 25%, then falls 20%, then rises 10%. What is the overall change, in percent? Reply with "
                "just the number with its sign.", r"\+?10(\.0+)?\s*%?"),
    ("clock", "What is the smaller angle between the hands of a clock at 3:40? Reply with just the number of degrees.",
     r"\b130\b"),
    ("binary", "How many 1 bits are in the binary form of 2024? Reply with just the number.", r"\b7\b"),
    ("series", "What is the next number: 2, 6, 12, 20, 30, 42, ? Reply with just the number.", r"\b56\b"),
    ("overlap", "Of 40 students, 23 play football, 17 play chess and 9 play neither. How many play both? Reply with just "
                "the number.", r"\b9\b"),
]

BUGGY = '''def average_positive(xs):
    """The mean of the positive numbers in xs (0 when there are none)."""
    total = 0
    for x in xs:
        if x > 0:
            total += x
    return total / len(xs)
'''


def _ans(rec, *pats):
    a = rec["answer"].replace(",", "")
    miss = [p for p in pats if not re.search(p, a, re.I)]
    return (False, f"the answer lacks {miss[0]!r}") if miss else (True, "right")


def _file(name, pat):
    def check(rec):
        t = rec["files"].get(name)
        return (True, f"{name} ok") if t and re.search(pat, t, re.I | re.S) else (False, f"no {name} with {pat!r}")
    return check


def _fixed(rec):
    t = rec["files"].get("stats.py") or ""
    if not t:
        return False, "no stats.py"
    ns: dict = {}
    try:
        exec(compile(t, "stats.py", "exec"), ns)   # (the worker's own file, in this bench's process: a few lines)
        f = ns["average_positive"]
        ok = f([1, -2, 3]) == 2 and f([-1, -5]) == 0 and f([]) == 0
    except Exception as e:  # noqa: BLE001
        return False, f"stats.py fails: {type(e).__name__}: {e}"
    return (True, "fixed") if ok else (False, "still wrong")


TASKS = [   # (name, prompt, check, setup files)
    ("fib_sha", "Using Python, work out the 40th Fibonacci number (F(1)=1, F(2)=1) and the SHA-256 hex digest of the text "
                "'strata on the 3090'. Give me both results.",
     lambda r: _ans(r, "102334155", "58cfdd685d3bf2dce0a0b426a79f6e8bf6ba59071f4c3a42753b7f74028c53b4"), {}),
    ("primes", "Make a CSV of the first 20 primes and their squares, save it as primes.csv, and tell me the sum of the "
               "squares.", lambda r: _ans(r, "30007"), {}),
    ("bugfix", "average_positive in stats.py (in your working folder) gives the wrong answer for [1, -2, 3]. Fix it, keep "
               "it returning 0 when there are no positive numbers, and test it.", _fixed, {"stats.py": BUGGY}),
    ("rev", "Write a python function that reverses the order of the words in a sentence, save it as rev.py with a few "
            "asserts that test it, and run them.", _file("rev.py", r"def .*assert"), {}),
    ("rebate", "My attic is 38 ft by 26 ft. Blown insulation is $1.85 per sq ft and the gas company's rebate is $0.49 per "
               "sq ft up to $890. What will the job cost me after the rebate? Use Python.", lambda r: _ans(r, r"1343\.68"), {}),
    ("landlord", "Draft a short polite note to my landlord saying the rent will be 3 days late, firm but friendly, signed "
                 "Jed.", lambda r: _ans(r, r"\bJed\b", r"rent", r"\b(3|three)\b"), {}),
    ("schedule", "I have three home inspections on Saturday: Springville at 9:30 (1 h), Provo at 11:00 (1.5 h) and Orem at "
                 "13:30 (1 h). Driving takes 20 min Springville-Provo and 15 min Provo-Orem. When will I be free, and how "
                 "much slack do I have between jobs? Use Python.", lambda r: _ans(r, r"14:30|2:30"), {}),
    ("csvfix", "Write a CSV people.csv with a header name,city,note and three rows where one note contains a comma, then "
               "read it back with the csv module and tell me how many rows it has.",
     _file("people.csv", r"name,city,note.*\""), {}),
]


def held(name: str) -> bool:
    """Every third case, in the order written (questions and tasks each): a fixed split, set before any run. (A hash of
    the name, as the other benches use, held out no question at all here.) Nothing in this bench is tuned on its cases;
    the split shows whether a level's result holds on cases nobody looked at while choosing a default."""
    for group in ([n for n, _, _ in QUESTIONS], [t[0] for t in TASKS]):
        if name in group:
            return group.index(name) % 3 == 2
    return False


# ---------------------------------------------------------------- stages
def run_questions(levels, reps, holdout) -> list[dict]:
    from jevh.config import load_config
    from jevh.llm import make_llm, with_thinking
    cfg = load_config()
    out = []
    for rep in range(reps):
        for name, q, pat in QUESTIONS:
            if holdout and not held(name):
                continue
            for lv in levels:
                llm = make_llm(with_thinking(dict(cfg.models["strong"]), lv, cfg.providers), cfg.providers)
                t0 = time.time()
                try:
                    r = llm.chat([{"role": "user", "content": q}])
                    text, err = r.text or "", None
                    toks = (r.output_tokens, r.thinking_tokens)
                except Exception as e:  # noqa: BLE001
                    text, err, toks = "", f"{type(e).__name__}: {e}", (0, 0)
                ok = bool(re.search(pat, text.replace("$", ""), re.I))
                rec = {"stage": "questions", "case": name, "level": lv, "rep": rep, "holdout": held(name),
                       "answer": text[:300], "error": err, "seconds": round(time.time() - t0, 1),
                       "output_tokens": toks[0], "thinking_tokens": toks[1], "passed": ok,
                       "why": "right" if ok else f"wrong: {text.strip()[:60]!r}"}
                out.append(rec)
                print(f"  q {name:9} {lv:6} rep {rep + 1}: {'PASS' if ok else 'FAIL'} {rec['seconds']:5.1f}s "
                      f"think {toks[1]}", flush=True)
    return out


def make_harness(where: Path, level: str, on_event=None):
    from jevh.config import load_config
    from jevh.orchestrator import Harness
    cfg = load_config()
    cfg.workspace = where / "ws"
    cfg.personas_dir = where / "personas"
    cfg.personas_dirs = [cfg.personas_dir]
    if not cfg.personas_dir.exists():
        shutil.copytree(ROOT / "personas", cfg.personas_dir, ignore=shutil.ignore_patterns("learned"))
    cfg.raw = {**cfg.raw, "learning": False, "failures": {**(cfg.raw.get("failures") or {}), "audit": 0},
               "thinking": {"level": level}}
    h = Harness(cfg, on_event=on_event)
    h.learning, h.tier_mode = False, "strong"
    h._ask_user = lambda *a, **k: "No - please don't change anything on this computer. Just do what you can."
    return h


def run_turns(levels, reps, holdout, where: Path) -> list[dict]:
    out = []
    for rep in range(reps):
        for name, prompt, check, files in TASKS:
            if holdout and not held(name):
                continue
            for lv in levels:
                h = make_harness(where / f"{name}-{lv}-{rep}", lv)
                if files:   # the files the task starts from go where the worker's REPL starts
                    orig = h.run_task

                    def run_task(task, goal, allow_collab, resume=None, _o=orig, _h=h, _f=files):
                        d = _h.session_dir / f"u{_h.turn_no}t{task.id}-{task.to}"
                        d.mkdir(parents=True, exist_ok=True)
                        for fn, body in _f.items():
                            (d / fn).write_text(body, encoding="utf-8")
                        return _o(task, goal, allow_collab, resume)
                    h.run_task = run_task
                watchdog = threading.Timer(LIMIT, h.cancel)
                watchdog.daemon = True
                watchdog.start()
                t0 = time.time()
                try:
                    r = h.turn(prompt)
                    answer, err, usage = r.answer, None, (r.usage or {})
                except Exception as e:  # noqa: BLE001
                    answer, err, usage = "", f"{type(e).__name__}: {e}", {}
                finally:
                    watchdog.cancel()
                found = session_files(h.session_dir)
                llm = (usage.get("by_kind") or {}).get("llm") or {}
                rec = {"stage": "turns", "case": name, "level": lv, "rep": rep, "holdout": held(name), "answer": answer[:3000],
                       "error": err, "seconds": round(time.time() - t0, 1), "files": found, "dir": str(h.session_dir),
                       "output_tokens": llm.get("output", 0), "thinking_tokens": llm.get("thinking", 0),
                       "input_tokens": llm.get("input", 0), "calls": llm.get("calls", 0)}
                grade_turn(rec)
                out.append(rec)
                print(f"  t {name:9} {lv:6} rep {rep + 1}: {'PASS' if rec['passed'] else 'FAIL'} {rec['seconds']:5.1f}s "
                      f"think {rec['thinking_tokens']} calls {rec['calls']} ({rec['why']})", flush=True)
    return out


def session_files(session_dir: Path) -> dict[str, str]:
    """The files a turn left (by name), not its thinking logs. The test is on the path inside the session: first live run
    (2026-10-24) tested the whole path, and a checkout in a folder named "thinking" (this worktree) dropped every file."""
    found = {}
    for p in Path(session_dir).rglob("*"):
        if p.is_file() and p.suffix in (".py", ".csv", ".txt", ".md") and "thinking" not in p.relative_to(session_dir).parts:
            found[p.name] = p.read_text(encoding="utf-8", errors="replace")[:20000]
    return found


def refill(rec: dict, run_dir: Path | None) -> dict:
    """A saved turn's files read again from its session folder, when it is still on disk (a raw file saved before the
    fix above has none)."""
    d = Path(rec["dir"]) if rec.get("dir") else None
    if (d is None or not d.is_dir()) and run_dir:
        d = next((x for x in (run_dir / f"{rec['case']}-{rec['level']}-{rec['rep']}" / "ws" / "sessions").glob("*")
                  if x.is_dir()), None) if (run_dir / f"{rec['case']}-{rec['level']}-{rec['rep']}").is_dir() else None
    if d and d.is_dir():
        rec["files"] = session_files(d)
    return rec


def grade_turn(rec: dict) -> dict:
    check = next(c for n, _, c, _ in TASKS if n == rec["case"])
    if rec["error"] and not rec["answer"]:
        rec["passed"], rec["why"] = False, rec["error"][:200]
        return rec
    ok, why = check(rec)
    rec["passed"], rec["why"] = bool(ok), why
    return rec


# ---------------------------------------------------------------- summary
def summarize(data: dict, levels=LEVELS) -> str:
    lines = [f"Thinking levels, live ({time.strftime('%Y-%m-%d %H:%M')})"]
    for stage, label in (("questions", "one call per question"), ("turns", "full harness turns")):
        rs = data.get(stage) or []
        if not rs:
            continue
        lines.append(f"\n[{stage}] {label}; per level: passed, median seconds, median thinking tokens"
                     + (", median model calls" if stage == "turns" else ""))
        for split in (False, True):
            s = [r for r in rs if r["holdout"] == split]
            if not s:
                continue
            lines.append(f"  {'held-out' if split else 'dev'}:")
            for lv in levels:
                x = [r for r in s if r["level"] == lv]
                if x:
                    lines.append(f"    {lv:6} {sum(r['passed'] for r in x):>3}/{len(x):<3}  "
                                 f"{statistics.median(r['seconds'] for r in x):6.1f}s  "
                                 f"think {statistics.median(r['thinking_tokens'] for r in x):7.0f}"
                                 + (f"  calls {statistics.median(r['calls'] for r in x):4.1f}" if stage == "turns" else ""))
        base = [r for r in rs if r["level"] == "auto"]
        if base:
            bt = statistics.median(r["seconds"] for r in base)
            for lv in levels:
                x = [r for r in rs if r["level"] == lv]
                if x and lv != "auto":
                    lines.append(f"    {lv} vs auto: {sum(r['passed'] for r in x) - sum(r['passed'] for r in base):+d} passed, "
                                 f"median time {statistics.median(r['seconds'] for r in x) / bt:.0%} of auto's")
        cases = sorted({r["case"] for r in rs})
        lines.append("    by case (passed per level " + "/".join(levels) + "):")
        for c in cases:
            lines.append(f"      {c:10} " + "  ".join(f"{sum(r['passed'] for r in rs if r['case'] == c and r['level'] == lv)}/"
                                                     f"{sum(1 for r in rs if r['case'] == c and r['level'] == lv)}"
                                                     for lv in levels))
    bad = [r for st in ("questions", "turns") for r in data.get(st) or [] if not r["passed"]]
    if bad:
        lines.append("\nfailures:")
        for r in bad:
            lines.append(f"  {r['stage']} {r['case']} {r['level']} rep {r['rep'] + 1}: {r['why']}")
    return "\n".join(lines)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", default="questions,turns")
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--holdout", action="store_true", help="only the held-out cases")
    ap.add_argument("--offline", nargs="?", const="", metavar="RAW.json",
                    help="summarize saved results again (default: the committed raw file, or --out's)")
    _out.add_args(ap)
    a = ap.parse_args()
    dest = _out.Out.of(a)
    levels = [x for x in a.levels.split(",") if x in LEVELS]
    if a.offline is not None:
        data = json.loads((Path(a.offline) if a.offline else dest.read(RAW)).read_text(encoding="utf-8"))
        run_dir = Path(a.offline).parent if a.offline else None   # (a run's own raw.json sits beside its turn folders)
        data["turns"] = [grade_turn(refill(r, run_dir)) for r in data.get("turns") or []]
        for r in (data.get("questions") or []) + data["turns"]:
            r["holdout"] = held(r["case"])   # (the split as defined now)
        print(summarize(data, [lv for lv in LEVELS if any(r["level"] == lv for st in ("questions", "turns")
                                                           for r in data.get(st) or [])]))
        return 0
    stages = a.stage.split(",")
    where = RUNS / time.strftime("%Y%m%d-%H%M%S")
    data: dict = {"levels": levels}

    def save():
        where.mkdir(parents=True, exist_ok=True)
        (where / "raw.json").write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    if "questions" in stages:
        data["questions"] = run_questions(levels, a.reps, a.holdout)
        save()
    if "turns" in stages:
        data["turns"] = run_turns(levels, a.reps, a.holdout, where)
        save()
    text = summarize(data, levels)
    print(text)
    blob = json.dumps(data, indent=1, ensure_ascii=False, default=str)
    dest.write(RAW).write_text(blob, encoding="utf-8")
    dest.write(SUMMARY).write_text(text + "\n", encoding="utf-8")
    print(f"raw: {dest.where(dest.write(RAW))}   summary: {dest.where(dest.write(SUMMARY))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
