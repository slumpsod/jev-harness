"""Cron jobs on live models: does a scheduled job do its work, in a session of its own, with nobody to answer?

    python evals/bench/cron.py                    # every case, 3 times each, a summary and raw results
    python evals/bench/cron.py --repeat 5         # more repeats (local models vary from run to run)
    python evals/bench/cron.py --holdout          # the held-out cases (different jobs, written after the main ones were tuned)
    python evals/bench/cron.py --stage jobs|timer # only the jobs a real harness does, or only the timer
    python evals/bench/cron.py --offline          # re-sweep the saved raw results (no models)

Needs what a normal turn needs: the models in config.yaml / config.local.yaml (LM Studio or llama-server) and a Jev backend.
No browser, Nango or network: every job works on files and arithmetic. It runs in a throwaway workspace, never yours.

What it checks (jevh/cron.py `run_job`, the same function the dashboard's Cron page and `jevh cron run` use):
  jobs    - a plain job finishes `ok`, in a new session folder, with the file or number it was asked for;
          - a job that tempts the worker to ask the user something (a vague request, a missing file) does not hang waiting:
            the question is answered by the harness ("nobody is here"), and the run ends within the time limit;
          - Stop ends a running job as `cancelled` within seconds; a job past its time limit ends as an `error` saying so;
          - a job does not touch another session: the folder list gains exactly one session per run.
  timer   - a due job is started once by the Runner, with the real harness behind it, and finishes `ok`; two Runners on one
            database start it once between them (the dashboard and the app hub share a workspace); a job missed past its
            grace is recorded `missed` and not run.
  Not measured: how *useful* the answers are (read a few: the summary prints them), and the Cron page (tests/test_cron_ui.py).

UNTUNED: there is no threshold in the feature to tune; a case is a pass or a fail. The numbers that would mean the feature is
wrong: a plain job under ~80% ok over repeats; any hang (a run past its limit by more than a minute); any run that left a
question pending; two Runners starting one job twice.

Results: cron_last_raw.json and cron_last_summary.txt (workspace/evals/<backend>/bench/, see evals/_out.py).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

import _out  # noqa: E402  (evals/_out.py: where a run's results go)
from jevh import cron  # noqa: E402
from jevh.config import load_config  # noqa: E402
from jevh.orchestrator import Harness  # noqa: E402

RAW = HERE / "cron_last_raw.json"
SUMMARY = HERE / "cron_last_summary.txt"

# name, split, prompt, what must be true of a finished run: (status in, a check on the answer/folder or None)
CASES = [
    {"name": "note_file", "split": "main", "prompt": "Write a one-line status note saying all systems are normal, and save it as status.txt.",
     "ok": ["ok"], "file": "status.txt"},
    {"name": "sum_of_squares", "split": "main", "prompt": "Work out the sum of the squares of the whole numbers from 1 to 20 and tell me the number.",
     "ok": ["ok"], "answer_has": "2870"},
    {"name": "vague_rename", "split": "main", "prompt": "Rename my report to something sensible.",
     "ok": ["ok", "error"], "no_hang": True},
    {"name": "which_one", "split": "main", "prompt": "Pick the best option for me and tell me which you chose.",
     "ok": ["ok", "error"], "no_hang": True},
    {"name": "csv_total", "split": "held", "prompt": "Make a small CSV of five products with prices in a file called prices.csv, then tell me the total price.",
     "ok": ["ok"], "file": "prices.csv"},
    {"name": "primes", "split": "held", "prompt": "List the prime numbers below 40 and say how many there are.",
     "ok": ["ok"], "answer_has": "12"},
    {"name": "ask_first", "split": "held", "prompt": "Email my landlord about the heating. Ask me what to say before you do anything.",
     "ok": ["ok", "error"], "no_hang": True},
]


def fresh_cfg(base):
    cfg = load_config(base)
    cfg.workspace = Path(tempfile.mkdtemp(prefix="jevh-cron-bench-")) / "ws"
    return cfg


def one_job(cfg, case: dict, timeout: float) -> dict:
    """A job done through the same path the dashboard uses. -> the case's row."""
    sessions = cfg.workspace / "sessions"
    before = {p.name for p in sessions.iterdir()} if sessions.is_dir() else set()
    asked, held = [], []

    def make(on_event):
        def watch(kind, data):
            if kind == "question":
                asked.append(data["question"]["question"][:160])
            on_event(kind, data)
        h = Harness(cfg, on_event=watch)
        held.append(h)
        return h
    t0 = time.time()
    out = cron.run_job({"id": 0, "name": case["name"], "prompt": case["prompt"], "persona": None}, make, threading.Event(), timeout)
    secs = time.time() - t0
    after = {p.name for p in sessions.iterdir()} if sessions.is_dir() else set()
    folder = sessions / out["session"] if out["session"] else None
    files = sorted(str(p.relative_to(folder)) for p in (folder.rglob("*") if folder and folder.is_dir() else []) if p.is_file())
    pending = held[0].pending_questions() if held else []
    problems = []
    if out["status"] not in case["ok"]:
        problems.append(f"status {out['status']} ({out['error']})")
    if case.get("answer_has") and case["answer_has"] not in out["answer"]:
        problems.append(f"the answer lacks {case['answer_has']!r}")
    if case.get("file") and not any(f.endswith(case["file"]) for f in files):
        problems.append(f"no {case['file']} was made")
    if len(after - before) != 1:
        problems.append(f"{len(after - before)} new session folders, not 1")
    if pending:
        problems.append("a question was left waiting")
    if secs > timeout + 60:
        problems.append(f"ran {secs:.0f}s past a {timeout:.0f}s limit")
    return {"case": case["name"], "split": case["split"], "status": out["status"], "seconds": round(secs, 1), "asked": asked,
            "answer": out["answer"][:300], "error": out["error"], "files": files[:6], "problems": problems, "pass": not problems}


def stop_and_limit(cfg) -> list[dict]:
    """Stop ends a running job; a job past its time limit is an error that says so."""
    rows = []
    slow = "Count slowly from 1 to 600, one number every second, by waiting a second between each, and print each one."
    cancel = threading.Event()
    threading.Timer(8, cancel.set).start()
    t0 = time.time()
    out = cron.run_job({"id": 0, "name": "stop", "prompt": slow, "persona": None}, lambda ev: Harness(cfg, on_event=ev), cancel, 600)
    secs = time.time() - t0
    rows.append({"case": "stop_ends_a_job", "split": "main", "status": out["status"], "seconds": round(secs, 1), "problems":
                 ([] if out["status"] == "cancelled" and secs < 8 + 45 else [f"{out['status']} after {secs:.0f}s (wanted cancelled within ~45s of Stop)"])})
    t0 = time.time()
    out = cron.run_job({"id": 0, "name": "limit", "prompt": slow, "persona": None}, lambda ev: Harness(cfg, on_event=ev), threading.Event(), 20)
    secs = time.time() - t0
    rows.append({"case": "time_limit_is_an_error", "split": "main", "status": out["status"], "seconds": round(secs, 1), "problems":
                 ([] if out["status"] == "error" and "stopped after" in out["error"] and secs < 20 + 60 else [f"{out['status']} {out['error']!r} after {secs:.0f}s"])})
    for r in rows:
        r["pass"] = not r["problems"]
    return rows


def timer(cfg) -> list[dict]:
    """The Runner with a real harness behind it: starts a due job once (even with two Runners), records a missed one."""
    rows = []
    path = cfg.workspace / "cron.db"
    a, b = cron.CronJobs(path, {"grace_minutes": 10}), cron.CronJobs(path, {"grace_minutes": 10})

    def execute(job, cancel, on_session):
        return cron.run_job(job, lambda ev: Harness(cfg, on_event=ev), cancel, 600, on_session)
    ra, rb = cron.Runner(a, execute, tick=1), cron.Runner(b, execute, tick=1)
    job = a.add("Bench note", "Write a one-line note saying hello and save it as hello.txt.", "daily at 9am")
    late = a.add("Bench missed", "Say hello.", "daily at 9am")
    now = time.time()
    a._run("UPDATE jobs SET next_run=? WHERE id=?", (now - 5, job["id"]))
    a._run("UPDATE jobs SET next_run=? WHERE id=?", (now - 3 * 3600, late["id"]))
    queued = ra.once(now) + rb.once(now)           # (two dashboards on one workspace, in the same second)
    end = time.time() + 600
    while time.time() < end and a.active(job["id"]):
        time.sleep(1)
    runs, missed = a.runs(job["id"]), a.runs(late["id"])
    problems = []
    if len(runs) != 1:
        problems.append(f"{len(runs)} runs of one due job from two runners (wanted exactly 1)")
    elif runs[0]["status"] != "ok" or runs[0]["trigger"] != "schedule" or not runs[0]["session"]:
        problems.append(f"the run was {runs[0]['status']}/{runs[0]['trigger']} ({runs[0]['error']})")
    rows.append({"case": "due_job_runs_once", "split": "main", "status": runs[0]["status"] if runs else "none", "queued": len(queued),
                 "problems": problems, "pass": not problems})
    problems = [] if len(missed) == 1 and missed[0]["status"] == "missed" else [f"runs: {[r['status'] for r in missed]}"]
    rows.append({"case": "late_job_is_missed", "split": "main", "status": missed[0]["status"] if missed else "none", "problems": problems,
                 "pass": not problems})
    return rows


def summarize(rows: list[dict], repeat: int, holdout: bool) -> str:
    lines = [f"cron bench: {len(rows)} runs, {'held-out' if holdout else 'main'} cases, repeat {repeat}"]
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["case"], []).append(r)
    for name, rs in by.items():
        ok = sum(r["pass"] for r in rs)
        secs = sorted(r.get("seconds", 0) for r in rs)
        lines.append(f"  {name:<26} {ok}/{len(rs)} pass   median {secs[len(secs) // 2]:.0f}s")
        for r in rs:
            if not r["pass"]:
                lines.append(f"      FAIL: {'; '.join(r['problems'])}")
    asked = [q for r in rows for q in r.get("asked", [])]
    lines.append(f"questions put to the user (each answered by the harness): {len(asked)}")
    lines.append(f"total: {sum(r['pass'] for r in rows)}/{len(rows)} pass")
    for r in rows[:6]:
        if r.get("answer"):
            lines.append(f"  sample answer ({r['case']}): {r['answer'][:140]!r}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--holdout", action="store_true", help="the held-out cases instead of the main ones")
    ap.add_argument("--stage", choices=["all", "jobs", "timer"], default="all")
    ap.add_argument("--timeout", type=float, default=300, help="seconds a job may take in the jobs stage")
    ap.add_argument("--offline", action="store_true", help="re-sweep the saved raw results")
    _out.add_args(ap)
    a = ap.parse_args(argv)
    out = _out.Out.of(a)
    if a.offline:
        saved = json.loads(out.read(RAW).read_text(encoding="utf-8"))
        print(summarize(saved["rows"], saved["repeat"], saved["holdout"]))
        return 0
    cfg = fresh_cfg(a.config)
    rows: list[dict] = []
    try:
        if a.stage in ("all", "jobs"):
            for case in [c for c in CASES if (c["split"] == "held") == a.holdout]:
                for i in range(a.repeat):
                    r = one_job(cfg, case, a.timeout)
                    rows.append(r)
                    print(f"{case['name']} #{i + 1}: {'pass' if r['pass'] else 'FAIL ' + '; '.join(r['problems'])} ({r['seconds']}s)", flush=True)
            if not a.holdout:
                rows += stop_and_limit(cfg)
        if a.stage in ("all", "timer") and not a.holdout:
            rows += timer(cfg)
    finally:
        shutil.rmtree(cfg.workspace.parent, ignore_errors=True)
    text = summarize(rows, a.repeat, a.holdout)
    print("\n" + text)
    out.write(RAW).write_text(json.dumps({"rows": rows, "repeat": a.repeat, "holdout": a.holdout}, indent=1), encoding="utf-8")
    out.write(SUMMARY).write_text(text + "\n", encoding="utf-8")
    return 0 if all(r["pass"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
