"""Live eval: is a long think SEARCHING (theory after theory) or CHECKING one idea?

The thinking watch's first question (TESTABLE_Q) catches hand-work code does exactly - and when
the model is checking one idea, "compute this in code" worked every time (the 8.3-min ARC run: 3
pauses, 3 code replies). But when it was searching among theories, it ignored those pauses (4 in a
row, 19 minutes), and TESTABLE_Q mostly didn't fire at all (0.17-0.22 on a 28 KB theorising step).
So searching gets its own question and its own pause: "write your best theory as a function and
print where it mismatches the examples".

Cases: real Qwen3.8 thinking at the watch's check points (last 2500 chars, as the watch sends it)
from the ARC bench runs, labelled by hand; frozen in search_cases_data.json. Tuning and holdout
are different runs (holdout: hv6temp1, hwatchtest, hwatchtest2, hwatchtest3 - incl. the 8.3-min run).

    python evals/search_cases.py            # tuning
    python evals/search_cases.py --holdout
    python evals/search_cases.py --offline [--holdout]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevh.config import load_config  # noqa: E402
from jevh.decide import SystemOneClient, noul  # noqa: E402
from jevh.orchestrator import TESTABLE_Q  # noqa: E402

DATA = Path(__file__).with_name("search_cases_data.json")
RAW = {"tuning": Path(__file__).with_name("search_cases_raw.json"),
       "holdout": Path(__file__).with_name("search_holdout_raw.json")}
ARC_TASK = ("Solve this ARC-AGI puzzle. The file puzzle.json holds JSON {\"train\": [{\"input\": grid, \"output\": "
            "grid}, ...], \"test\": [{\"input\": grid}]}; grids are lists of rows of colours 0-9. Every train input "
            "maps to its output by one hidden rule. Find the rule, apply it to the test input, save the output grid.")

QUESTIONS = {
    "searching": ("Is `thinking` going back and forth between different theories or approaches without settling "
                  "on one - proposing an idea, finding it doesn't fit, and trying another?"),
    "stuck": ("Is `thinking` stuck searching for a rule - trying idea after idea - rather than carefully checking "
              "one specific idea against the data?"),
    "checking": "Is `thinking` checking one specific idea step by step against the data?",
    "testable": TESTABLE_Q,
}


def cases(which: str) -> list[dict]:
    return [c for c in json.loads(DATA.read_text(encoding="utf-8")) if c["set"] == which]


def run_live(cs, raw: Path) -> list[dict]:
    cfg = load_config()
    client = SystemOneClient(cfg.jev["url"], cfg.jev["backend"], timeout=300)
    rows = []
    for c in cs:
        r = client.ask({"task": ARC_TASK, "thinking": c["thinking"]}, {k: noul(q) for k, q in QUESTIONS.items()})
        rows.append({k: r[k].noul for k in QUESTIONS})
    raw.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return rows


def report(cs, rows) -> None:
    labels = [c["searching"] for c in cs]
    for c, s in zip(cs, rows):
        print(f"  {'SEARCH' if c['searching'] else 'check '}  " + "  ".join(f"{k}={v:.2f}" for k, v in s.items())
              + f"  #{c['id']} {c['run']} {c['step']}@{c['at']}")
    for key in QUESTIONS:
        if key == "testable":
            continue
        high = key != "checking"
        print(f"\n== {key}")
        for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            flag = [(s[key] >= t) if high else (s[key] < 1 - t) for s in rows]
            hit = sum(f and l for f, l in zip(flag, labels))
            false = sum(f and not l for f, l in zip(flag, labels))
            print(f"  t={t:.1f}: searching caught {hit}/{labels.count(True)}, checking flagged {false}/{labels.count(False)}")
    print("\n== testable (the watch's current question) >= 0.55:")
    for name, want in (("searching", True), ("checking", False)):
        n = sum(1 for s, l in zip(rows, labels) if l == want and s["testable"] >= 0.55)
        print(f"  {name}: {n}/{labels.count(want)} would pause")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    which = "holdout" if a.holdout else "tuning"
    cs = cases(which)
    rows = json.loads(RAW[which].read_text(encoding="utf-8")) if a.offline else run_live(cs, RAW[which])
    report(cs, rows)


if __name__ == "__main__":
    main()
