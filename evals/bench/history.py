"""Live check of the History tab's categories on your own sessions, against the real Jev. Nothing of yours is written: it
copies the workspace's memory database (read-only) and the session folder names into a temp folder and runs on that.

    python evals/bench/history.py                  # one line per check, a summary, raw results beside it
    python evals/bench/history.py --config path    # another config (its personas, workspace and Jev)
    python evals/bench/history.py --workspace path # the sessions of another workspace, sorted among this config's assistants

What it checks, over every session the workspace has:
  - Jev sorts all of them, with no error, and each lands in one of your assistants, "No single field" or "Empty sessions";
  - how long a question takes (the tab asks once per session, on a background thread);
  - looking again asks Jev nothing (a session is asked about once), and "sort again" asks about every session again;
  - the dashboard's endpoints serve the same: /api/history lists every session and /api/history/<id> reads each one with as
    many turns as the list says;
  - what you get, as counts per category (the titles stay on your screen: the summary file holds counts only).
The eval of how *right* the categories are, on labelled sessions, is evals/session_category_cases.py.

Results: history_last_summary.txt (counts only; in workspace/evals/<backend>/bench/, see evals/_out.py) and evals/bench/runs/history/<time>/raw.json (git-ignored: it has
your session titles).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

from jevh.config import load_config  # noqa: E402
from jevh.history import EMPTY, MIXED, History  # noqa: E402
from jevh.orchestrator import Harness  # noqa: E402
import _out  # noqa: E402  (evals/_out.py: where a run's results go)


def copy_workspace(src: Path, dst: Path) -> None:
    """The memory database (a consistent read-only copy) and the session folders' names: enough to list every session."""
    (dst / "memory").mkdir(parents=True)
    db = src / "memory" / "memory.db"
    if db.is_file():
        a = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
        b = sqlite3.connect(dst / "memory" / "memory.db")
        a.backup(b)
        b.close()
        a.close()
    for d in (src / "sessions").iterdir() if (src / "sessions").is_dir() else []:
        if d.is_dir():
            for x in d.iterdir():
                if x.is_dir() and x.name != "artifacts":
                    (dst / "sessions" / d.name / x.name).mkdir(parents=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config")
    ap.add_argument("--workspace", help="read this workspace's sessions instead of the config's")
    _out.add_args(ap)
    a = ap.parse_args(argv)
    dest = _out.Out.of(a)
    cfg = load_config(a.config)
    real = Path(a.workspace) if a.workspace else Path(cfg.workspace)
    tmp = Path(tempfile.mkdtemp(prefix="jevh-history-"))
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    try:
        copy_workspace(real, tmp)
        cfg.workspace = tmp
        h = Harness(cfg)
        asked: list[float] = []
        original = h.jev.session_category

        def timed(*args, **kw):
            t0 = time.perf_counter()
            out = original(*args, **kw)
            asked.append((time.perf_counter() - t0) * 1000)
            return out
        h.jev.session_category = timed
        hist = History(lambda: h)

        first = hist.overview(10 ** 6)
        n = len(first["sessions"])
        check("there are sessions to sort", n > 0, f"{n} sessions in the workspace")
        t0 = time.time()
        while hist.categorizer.pending() and time.time() - t0 < 300:
            time.sleep(0.2)
        took = time.time() - t0
        o = hist.overview(10 ** 6)
        roster = set(h.personas) | {MIXED, EMPTY}
        check("Jev sorted every session without an error", o["pending"] == 0 and not o["error"] and all(s["category"] for s in o["sessions"]),
              f"{took:.1f}s for {n} sessions; pending {o['pending']}; error {o['error']!r}")
        check("every category is one of your assistants, 'No single field' or 'Empty sessions'",
              {s["category"] for s in o["sessions"]} <= roster, ", ".join(sorted({s["category"] for s in o["sessions"] if s["category"]})))
        if asked:
            check("a question takes well under a second (median)", statistics.median(asked) < 1000,
                  f"median {statistics.median(asked):.0f} ms, worst {max(asked):.0f} ms over {len(asked)} questions")
        before = len(asked)
        hist.overview(10 ** 6)
        time.sleep(0.5)
        check("looking again asks Jev nothing", len(asked) == before, f"{len(asked) - before} new questions")
        again = hist.recategorize()
        t1 = time.time()
        while hist.categorizer.pending() and time.time() - t1 < 300:
            time.sleep(0.2)
        time.sleep(0.3)
        check("'sort again' asks about every session again", len(asked) - before == sum(1 for s in o["sessions"] if s["category"] != EMPTY),
              f"{len(asked) - before} questions for {n} sessions (pending right after: {again})")
        bad = [s["session"] for s in o["sessions"] if (hist.transcript(s["session"]) or {}).get("turns") is None
               or len(hist.transcript(s["session"])["turns"]) != s["turns"]]
        check("every session can be read, with the turns the list says", not bad, f"{len(bad)} that cannot" if bad else f"{n} read")

        counts: dict[str, int] = {}
        for s in o["sessions"]:
            counts[s["category"]] = counts.get(s["category"], 0) + 1
        raw = {"started": time.strftime("%Y%m%d-%H%M%S"), "sessions": o["sessions"], "categories": o["categories"],
               "ask_ms": [round(x, 1) for x in asked], "checks": checks}
        for s in o["sessions"]:
            if s["category"] == MIXED:
                print(f"  (no single field: {s['confidence']:.2f}, candidates {', '.join(s['also'])}) {s['title'][:70]!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    failed = [c for c in checks if not c["ok"]]
    lines = [f"HISTORY CATEGORIES  {raw['started']}  config={a.config or 'config.yaml'}", ""]
    lines += [f"  {'ok  ' if c['ok'] else 'FAIL'} {c['check']}" + (f"   [{c['detail']}]" if c["detail"] else "") for c in checks]
    lines += ["", "sessions per category: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])),
              f"{len(checks) - len(failed)}/{len(checks)} checks passed" + (f"; failing: {', '.join(c['check'] for c in failed)}" if failed else "")]
    text = "\n".join(lines)
    print(text)
    run_dir = ROOT / "evals" / "bench" / "runs" / "history" / raw["started"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    summary = dest.write(ROOT / "evals" / "bench" / "history_last_summary.txt")
    print(f"summary: {dest.where(summary)}   raw: {dest.where(run_dir / 'raw.json')}")
    summary.write_text(text + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
