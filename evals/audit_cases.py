"""Live eval: the spot check (jevh/audit.py). Does a model rereading a finished task tell a wrong answer
from a right one?

evals/audit_cases.json: 20 finished tasks, 8 wrong and 12 right, mostly real runs from 2026-09-30 (the
request, the answer as delivered, the files handed over, the work log). The wrong ones are the kind no
check of the answer's text catches: a planets.csv holding only its own name (its answer even pasted the
right CSV), "Ana" turned into "name", a leaked tool call handed over as the answer, a regex summary that
miscounted its own test cases, the wrong row's price, a CSV missing a row, a count off by two.

A wrong verdict on a right answer puts a good run among the failures the harness is changed from, so
false alarms count double here; "unsure" logs nothing.

    python evals/audit_cases.py                                   # the strong model (high_end_dev's)
    python evals/audit_cases.py --model lmstudio:ornith-1.5-35b-a3b
    python evals/audit_cases.py --offline                         # re-print the saved verdicts
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevh.audit import audit  # noqa: E402
from jevh.config import load_config  # noqa: E402
from jevh.llm import make_llm  # noqa: E402
from jevh.personas import load_personas  # noqa: E402
import _out  # noqa: E402  (evals/_out.py: where a run's results go)

CASES = Path(__file__).with_name("audit_cases.json")


def block_for(cfg, model: str | None) -> dict:
    """The model with the settings the harness runs it with (a persona's or models.fast)."""
    personas = load_personas(cfg.personas_dir)
    if not model:
        return dict(personas["high_end_dev"].model)
    blocks = [cfg.models.get("fast"), cfg.models.get("strong"), *(p.model for p in personas.values())]
    for b in blocks:
        if b and f"{b['provider']}:{b['model']}" == model:
            return dict(b)
    provider, _, name = model.partition(":")
    return {"provider": provider, "model": name}


def materialize(case: dict, folder: Path) -> list[str]:
    paths = []
    for name, f in case["files"].items():
        p = folder / name
        if "text" in f:
            p.write_text(f["text"], encoding="utf-8")
        else:
            p.write_bytes(base64.b64decode(f["base64"]))
        paths.append(str(p))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--offline", action="store_true")
    _out.add_args(ap)
    a = ap.parse_args()
    dest = _out.Out.of(a)
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    raw = CASES.with_name(f"audit_cases_raw{'-' + a.model.split(':')[-1] if a.model else ''}.json")
    if a.offline:
        rows = json.loads(dest.read(raw).read_text(encoding="utf-8"))
    else:
        cfg = load_config()
        llm = make_llm(block_for(cfg, a.model), cfg.providers)
        rows = []
        for c in cases:
            started = time.time()
            with tempfile.TemporaryDirectory() as d:
                v = audit(llm, c["request"], c["answer"], materialize(c, Path(d)), c["log"])
            rows.append({"id": c["id"], **v, "seconds": round(time.time() - started, 1)})
            print(f"  {c['label']:<5} -> {v['verdict']:<6} {rows[-1]['seconds']:>6.1f}s  {c['id']}", flush=True)
        dest.write(raw).write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"raw verdicts saved to {dest.where(dest.write(raw))}")
    print()
    for c, r in zip(cases, rows):
        mark = "ok " if (r["verdict"] == "wrong") == (c["label"] == "wrong") and r["verdict"] != "unsure" else "-- "
        print(f"{mark}{c['label']:<5} -> {r['verdict']:<6} {c['id']:<36} {'; '.join(r['problems'])[:110]}")
    wrong = [r for c, r in zip(cases, rows) if c["label"] == "wrong"]
    right = [r for c, r in zip(cases, rows) if c["label"] == "right"]
    caught = sum(r["verdict"] == "wrong" for r in wrong)
    false = sum(r["verdict"] == "wrong" for r in right)
    unsure = sum(r["verdict"] == "unsure" for r in rows)
    secs = sum(r.get("seconds", 0) for r in rows) / max(len(rows), 1)
    print(f"\nwrong answers caught {caught}/{len(wrong)}, right answers flagged {false}/{len(right)}, "
          f"unsure {unsure}, {secs:.0f}s per check")


if __name__ == "__main__":
    main()
