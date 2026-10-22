"""Light Jev, live: full harness turns on real models with Jev in full mode (semif) against light mode (a small model, Laya,
on the CPU), with the checks light mode hands off answered by the task's model (light-worker) or skipped (light-off).

What it checks, per arm, on the same requests:
  kept      decisions light Jev keeps: the right answer, the persona the request belongs to, the skill and the tool guide
            it needs (the small model makes these)
  handoff   decisions it hands off: a follow-up stays with the assistant of the conversation, a lasting fact the user
            states is learned, a request that cannot be done (a file that is not there) lands in the failure log
  cost      seconds a turn, and how many decisions went to the small model, to the task's model, or were skipped

    python evals/bench/light_jev.py                       # all arms, all cases, 1 repeat (about 90 min on Qwen 27B)
    python evals/bench/light_jev.py --arm light-worker --reps 2
    python evals/bench/light_jev.py --holdout             # the held-out cases only (written apart; don't tune on them)
    python evals/bench/light_jev.py --offline             # the summary again from the saved results

Needs: the worker model (config models / config.local.yaml; designed for Qwen3.8-27B on llama-server), and an adapter
that serves both backends (openjev/systemone.py from this checkout, with semif_bridge.py and laya_bridge.py behind it;
point --jev-url at it). Laya was meant to run on the CPU here (LAYA_DEVICE=cpu). UNVERIFIED: run once, short (6 cases,
1 repeat, 2026-10-22: full 5/6, light-worker 4/6, light-off 3/6; openjev/README.md "Light Jev"); no committed results.
Light mode is wrong if a kept decision passes clearly less often than in full mode, or if light-worker passes the
handoff cases clearly less often than full while costing more than ~10 s a turn.
Results: light_jev_raw.json and light_jev_summary.txt (a live run writes to workspace/evals/<backend>/bench/; see
evals/_out.py), each run's workspace under runs/light_jev/<stamp>/.
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
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))
import _out  # noqa: E402  (evals/_out.py: where a run's results go)

RUNS = HERE / "runs" / "light_jev"
RAW = HERE / "light_jev_raw.json"
SUMMARY = HERE / "light_jev_summary.txt"
LIMIT = 8 * 60
ARMS = {"full": {"mode": "full"}, "light-worker": {"mode": "light", "light": {"checks": "worker"}},
        "light-off": {"mode": "light", "light": {"checks": "off"}}}
D, DEV, FIT, FE = "daily_assistant", "high_end_dev", "fitness_coach", "front_end_designer"


@dataclass
class Case:
    name: str
    kind: str            # answer | persona | skill | tool | followup | fact | failure
    prompt: str
    expect: str          # a regex for the answer, a persona, a skill, a tool, or a fact's text
    group: str           # kept | handoff
    before: list[str] = field(default_factory=list)   # earlier turns of the same conversation
    holdout: bool = False


CASES = [
    Case("percent", "answer", "What's 17% of 2,340? Just the number.", r"397\.8", "kept"),
    Case("sqft", "answer", "Convert 2,400 square feet to square meters, rounded to one decimal.", r"22[23]\.\d", "kept"),
    Case("fib", "answer", "Write and run Python that prints the 30th Fibonacci number (fib(1) = fib(2) = 1).", r"832,?040",
         "kept"),
    Case("route-fit", "persona", "Give me a 20-minute bodyweight workout I can do at home.", FIT, "kept"),
    Case("skill-quotes", "skill", "I need quotes for blown-in attic insulation for my 1,800 sq ft house.",
         "eco-insulation-quotes", "kept"),
    Case("tool-web", "tool", "What's the weather in Denver right now?", "web", "kept"),
    Case("follow-gym", "followup", "make day 2 harder", FIT, "handoff", before=["Give me a 3-day beginner gym split."]),
    Case("fact-slc", "fact", "I run my insulation business out of Salt Lake City. Suggest a short name for my newsletter.",
         "salt lake", "handoff"),
    Case("fail-pdf", "failure", r"Summarize the PDF at D:\missing\q3-report.pdf for me.", "", "handoff"),
    # held out: written apart from the ones above
    Case("h-tax", "answer", "What's 8% sales tax on $1,249.99? Just the amount.", r"100(\.00)?\b", "kept", holdout=True),
    Case("h-route-fe", "persona", "Pick a color palette for a bakery website.", FE, "kept", holdout=True),
    Case("h-follow-crew", "followup", "make it friendlier", D, "handoff",
         before=["Draft a two-line text telling my crew we start at 8 tomorrow."], holdout=True),
    Case("h-fact-dog", "fact", "My dog is named Max, by the way. What's a good name for a second dog?", "max", "handoff",
         holdout=True),
    Case("h-fail-tax", "failure", r"Open my tax return at D:\taxes\2025-return.pdf and tell me my refund amount.", "",
         "handoff", holdout=True),
]


def make_harness(where: Path, arm: str, jev_url: str, events: list, learn: bool):
    from jevh.config import load_config
    from jevh.orchestrator import Harness
    cfg = load_config()
    cfg.workspace = where / "ws"
    cfg.personas_dir = where / "personas"
    cfg.personas_dirs = [cfg.personas_dir]
    if not cfg.personas_dir.exists():
        shutil.copytree(ROOT / "personas", cfg.personas_dir, ignore=shutil.ignore_patterns("learned"))
    jev = {k: v for k, v in cfg.jev.items() if k not in ("mode", "light")}
    light = {**(cfg.jev.get("light") or {}), **ARMS[arm].get("light", {})}
    cfg.jev = {**jev, "url": jev_url, "backend": "semif", "mode": ARMS[arm]["mode"], "light": light}
    cfg.raw = {**cfg.raw, "jev": cfg.jev, "learning": learn,
               "failures": {**(cfg.raw.get("failures") or {}), "audit": 0, "jev_check": True}}
    t0 = time.time()
    h = Harness(cfg, on_event=lambda k, d: events.append((time.time() - t0, k, d)))
    h.learning, h.learn_sync, h.tier_mode = learn, True, "strong"
    h._ask_user = lambda *a, **k: "Yes, go ahead - your call."
    return h


def timed_turn(h, prompt: str):
    watchdog = threading.Timer(LIMIT, h.cancel)
    watchdog.daemon = True
    watchdog.start()
    try:
        return h.turn(prompt), None
    except Exception as e:  # noqa: BLE001 - a crash is a result
        return None, f"{type(e).__name__}: {e}"
    finally:
        watchdog.cancel()


def where_decided(trace: list[dict]) -> dict[str, int]:
    out = {"jev": 0, "worker": 0}
    for t in trace:
        out["worker" if str(t.get("model", "")).startswith("worker:") else "jev"] += 1
    return out


def run_case(case: Case, arm: str, where: Path, jev_url: str) -> dict:
    events: list = []
    h = make_harness(where, arm, jev_url, events, learn=case.kind == "fact")
    personas = []
    for before in case.before:
        r0, _ = timed_turn(h, before)
        personas.append(r0.persona if r0 else None)
    events.clear()
    started = time.time()
    r, err = timed_turn(h, case.prompt)
    h.wait_learning(120)
    seconds = round(time.time() - started, 1)
    answer = (r.answer if r else "") or ""
    trace = list(r.trace if r else [])
    skills = [s for _, k, d in events if k == "skills" for s in getattr(d.get("decision"), "skills", [])]
    tools = [t for _, k, d in events if k == "tools" for t in getattr(d.get("decision"), "tools", [])]
    failures = [f["kind"] for f in h.failures.list()]
    facts = [f.text for f in h.memory.facts(None, 20)]
    persona = r.persona if r else None
    if case.kind == "answer":
        ok = bool(re.search(case.expect, answer.replace("\u202f", " ")))
    elif case.kind == "persona":
        ok = persona == case.expect
    elif case.kind == "skill":
        ok = case.expect in skills
    elif case.kind == "tool":
        ok = case.expect in tools
    elif case.kind == "followup":
        ok = persona == case.expect and all(p == case.expect for p in personas)
    elif case.kind == "fact":
        ok = any(case.expect in f.lower() for f in facts)
    else:   # failure: the request could not be done, so the failure log should say so
        ok = any(k in ("gave_up", "missed_request", "wrong_answer") for k in failures)
    return {"case": case.name, "kind": case.kind, "group": case.group, "holdout": case.holdout, "arm": arm, "ok": ok,
            "error": err, "seconds": seconds, "persona": persona, "skills": skills, "tools": tools,
            "failures": failures, "facts": facts[:5], "answer": answer[:400], "decided": where_decided(trace),
            "handoff_seconds": round(sum(t.get("latency_s", 0) for t in trace
                                         if str(t.get("model", "")).startswith("worker:")), 1),
            "outages": sum(1 for _, k, d in events if k == "jev_error")}   # (a decision light Jev skips is not one)


def summary(rows: list[dict]) -> list[str]:
    arms = [a for a in ARMS if any(r["arm"] == a for r in rows)]
    lines = [f"Light Jev live bench: {len(rows)} turns ({', '.join(arms)})", ""]
    lines.append(f"{'':22}" + "".join(f"{a:>16}" for a in arms))
    for label, keep in (("kept decisions", lambda r: r["group"] == "kept"),
                        ("handed-off checks", lambda r: r["group"] == "handoff"),
                        ("held-out cases", lambda r: r["holdout"]), ("all", lambda r: True)):
        cells = []
        for a in arms:
            rs = [r for r in rows if r["arm"] == a and keep(r)]
            cells.append(f"{sum(r['ok'] for r in rs)}/{len(rs)}" if rs else "-")
        lines.append(f"{label:22}" + "".join(f"{c:>16}" for c in cells))
    for label, f in (("seconds a turn (mean)", lambda rs: statistics.mean(r["seconds"] for r in rs)),
                     ("handoff seconds/turn", lambda rs: statistics.mean(r["handoff_seconds"] for r in rs)),
                     ("small-model decisions", lambda rs: statistics.mean(r["decided"]["jev"] for r in rs)),
                     ("task-model decisions", lambda rs: statistics.mean(r["decided"]["worker"] for r in rs))):
        lines.append(f"{label:22}" + "".join(
            f"{f([r for r in rows if r['arm'] == a]):>16.1f}" for a in arms))
    lines += ["", "not passed:"]
    for r in rows:
        if not r["ok"]:
            why = {"answer": r["answer"][:80], "persona": r["persona"], "skill": r["skills"], "tool": r["tools"],
                   "followup": r["persona"], "fact": r["facts"][:2], "failure": r["failures"]}[r["kind"]]
            lines.append(f"  {r['arm']:13} {r['case']:15} {r['kind']:9} got {why}" + (f"  ERROR {r['error']}" if r["error"] else ""))
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", default="all", help="full, light-worker, light-off, or all (comma list)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--holdout", action="store_true", help="only the held-out cases")
    ap.add_argument("-k", default="", help="only cases whose name contains one of these (comma-separated)")
    ap.add_argument("--jev-url", default="http://127.0.0.1:33001", help="an adapter serving semif and laya")
    ap.add_argument("--offline", action="store_true")
    _out.add_args(ap)
    a = ap.parse_args()
    out = _out.Out.of(a)
    if a.offline:
        rows = json.loads(out.read(RAW).read_text(encoding="utf-8"))
        print("\n".join(summary(rows)))
        return 0
    arms = list(ARMS) if a.arm == "all" else a.arm.split(",")
    keys = [k for k in a.k.split(",") if k]
    cases = [c for c in CASES if (c.holdout or not a.holdout) and (not keys or any(k in c.name for k in keys))]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out.write(RAW)
    rows: list[dict] = []
    for rep in range(a.reps):
        for case in cases:
            for arm in arms:   # interleaved: the arms see the same server load
                where = RUNS / stamp / f"{case.name}-{arm}-{rep}"
                row = run_case(case, arm, where, a.jev_url)
                row["rep"] = rep
                rows.append(row)
                print(f"  {arm:13} {case.name:15} {'ok ' if row['ok'] else 'NO '} {row['seconds']:6.1f}s "
                      f"jev={row['decided']['jev']} worker={row['decided']['worker']}", flush=True)
                path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = summary(rows)
    out.write(SUMMARY).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved {out.where(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
