"""Live eval: a risky change's review - the strong model or the fast one?

The cheap first pass (evals/risk_cases.py) sends risky changes to a review. How bad a mistake
could be decides who reviews: severe (someone gets in who shouldn't, secrets or user data leak,
stored data is lost or corrupted, money is charged wrong) -> the strong model; moderate (it could
break callers, a format, or behaviour, but nothing is lost or exposed) -> the fast one, ~7 s
instead of ~5 min live. A severe change sent to the fast reviewer is the costly error.

Cases: the risky ones of risk_cases.py (same diffs) plus more moderate ones, each labelled.

    python evals/severity_cases.py            # live, tuning cases
    python evals/severity_cases.py --holdout  # live, held-out cases
    python evals/severity_cases.py --offline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import risk_cases  # noqa: E402
from jevh.config import load_config  # noqa: E402
from jevh.decide import SystemOneClient, noul  # noqa: E402
import _out  # noqa: E402  (evals/_out.py: where a run's results go)

diff = risk_cases.diff
RAW = Path(__file__).with_name("severity_cases_raw.json")
RAW_HOLDOUT = Path(__file__).with_name("severity_holdout_raw.json")

# the risky cases of risk_cases.py that are moderate (everything else risky there is severe)
MODERATE = {"Rename the id parameter to user_id for clarity.", "Use camelCase in the orders API response.",
            "The counter is slow; simplify it.", "Change the date format in the CSV export."}

EXTRA = [  # more moderate risky changes (request, file, change, severe?)
    ("Return a list of names instead of a dict.", "lib/people.py",
     diff("    return {p.id: p.name for p in people}", "    return [p.name for p in people]"), False),
    ("Rename the --out flag to --output.", "cli/main.py",
     diff("    ap.add_argument('--out')", "    ap.add_argument('--output')"), False),
    ("Change the config file to TOML.", "app/settings.py",
     diff("    return yaml.safe_load(open('config.yaml'))", "    return tomllib.load(open('config.toml', 'rb'))"),
     False),
    ("Stop retrying failed webhooks.", "app/webhooks.py",
     diff("@retry(times=3)\ndef deliver(hook, payload):", "def deliver(hook, payload):"), False),
    ("Page the orders API by 50 instead of 20.", "app/routes/orders.py",
     diff("    per_page = 20", "    per_page = 50"), False),
    ("Make timeouts shorter.", "app/http.py",
     diff("    r = requests.get(url, timeout=30)", "    r = requests.get(url, timeout=3)"), False),
]
EXTRA_HOLDOUT = [
    ("Sort the report newest first.", "app/report.py",
     diff("    rows.sort(key=lambda r: r.date)", "    rows.sort(key=lambda r: r.date, reverse=True)"), False),
    ("Drop Python 3.9 support.", "pyproject.toml",
     diff('requires-python = ">=3.9"', 'requires-python = ">=3.11"'), False),
    ("Rename the event from 'order.done' to 'order.completed'.", "app/events.py",
     diff("    emit('order.done', order)", "    emit('order.completed', order)"), False),
]


def risky(cases):
    return [(req, f, ch, req not in MODERATE) for req, f, ch, is_risky in cases if is_risky]


CASES = risky(risk_cases.CASES) + EXTRA
HOLDOUT = risky(risk_cases.HOLDOUT) + EXTRA_HOLDOUT

QUESTIONS = {
    "severe": ("Could a mistake in `change` let someone in who shouldn't be, expose secrets or users' data, "
               "delete or corrupt stored data, or charge or pay the wrong amount?"),
    "harm": ("If `change` has a mistake, would the harm be serious and hard to undo - a security hole, lost "
             "data or money - rather than something that breaks and gets fixed, like a format or a parameter?"),
    "breaks_only": ("Is the worst a mistake in `change` could do to break other code or change how something "
                    "behaves, without exposing anything, losing data or touching money?"),
}


def run_live(cases, raw: Path) -> list[dict]:
    cfg = load_config()
    c = SystemOneClient(cfg.jev["url"], cfg.jev["backend"], timeout=300)
    rows = []
    for req, file, change, _ in cases:
        r = c.ask({"request": req, "file": file, "change": change}, {k: noul(q) for k, q in QUESTIONS.items()})
        rows.append({k: r[k].noul for k in QUESTIONS})
    raw.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return rows


def report(cases, rows) -> None:
    labels = [c[3] for c in cases]
    for (req, file, _, lab), s in zip(cases, rows):
        print(f"  {'SEVERE  ' if lab else 'moderate'}  " + "  ".join(f"{k}={v:.2f}" for k, v in s.items())
              + f"  {file[:26]:26} | {req[:40]}")
    from jevh.verify import SEVERE_RE
    plain = [bool(SEVERE_RE.search("\n".join(ln for ln in c[2].splitlines() if ln[:1] in "+-"))) for c in cases]
    print("\n== chosen rule: severe >= t or plain code (SEVERE_RE on the diff's lines)")
    for t in (0.05, 0.1, 0.15, 0.2, 0.3):
        strong = [s["severe"] >= t or p for s, p in zip(rows, plain)]
        sev = sum(st and l for st, l in zip(strong, labels))
        mod = sum(not st and not l for st, l in zip(strong, labels))
        print(f"  t={t:.2f}: severe -> strong {sev}/{labels.count(True)}, moderate -> fast {mod}/{labels.count(False)}")
    for key in QUESTIONS:
        strong_if_high = key != "breaks_only"
        print(f"\n== {key}")
        for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            strong = [(s[key] >= t) if strong_if_high else (s[key] < t) for s in rows]
            sev = sum(st and l for st, l in zip(strong, labels))
            mod = sum(not st and not l for st, l in zip(strong, labels))
            print(f"  t={t:.1f}: severe -> strong {sev}/{labels.count(True)}, "
                  f"moderate -> fast {mod}/{labels.count(False)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    _out.add_args(ap)
    a = ap.parse_args()
    dest = _out.Out.of(a)
    cases, raw = (HOLDOUT, RAW_HOLDOUT) if a.holdout else (CASES, RAW)
    if a.offline:
        rows = json.loads(dest.read(raw).read_text(encoding="utf-8"))
    else:
        rows = run_live(cases, dest.write(raw))
        print(f"raw scores saved to {dest.where(dest.write(raw))}")
    report(cases, rows)


if __name__ == "__main__":
    main()
