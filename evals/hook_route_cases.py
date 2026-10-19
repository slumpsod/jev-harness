"""Live eval of the skill/hook router's first question (`HOOK_Q`, jevh/learn.py): could a program decide
whether the main requirements of this work were met, without understanding the request?

NOT TUNED, NOT RUN AGAINST A LIVE JEV YET. `HOOK_Q`'s wording and the `route_hook` threshold
(config.yaml, jevh/config.py) are first guesses; the router (docs/architecture/14-skill-hook-router.md)
is not built, so nothing depends on them. Run this, then set `route_hook` from the sweep.

    python evals/hook_route_cases.py            # asks Jev, saves the scores, sweeps the thresholds
    python evals/hook_route_cases.py --offline  # re-sweeps the saved scores (no Jev)
    python evals/hook_route_cases.py --check    # exit 1 unless the configured threshold has no false positive
                                                # on the tuning cases or the held-out ones (offline)

The cases and what their labels mean are in evals/hook_route_data.py. The policy: a false positive
costs a model call and a hook that may flag good work; a false negative falls back to a draft skill,
which is what the harness does today. So pick the threshold with no false positives on the tuning cases,
as much recall as that leaves, and a margin below the lowest checkable score you'd miss; then score the
held-out cases once at that threshold. Moving it to fit the held-out cases turns them into tuning data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hook_route_data import ALL, HOLDOUT, KINDS, TUNING, Case  # noqa: E402

from jevh.config import DEFAULT_THRESHOLDS  # noqa: E402
from jevh.learn import HOOK_Q, hook_state  # noqa: E402
import _out  # noqa: E402  (evals/_out.py: where a run's results go)

RAW = Path(__file__).with_name("hook_route_raw.json")
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99)
MARGIN = 0.05   # the recommended threshold sits at least this far above the highest score of a non-checkable case


# ---------------------------------------------------------------- scoring

def run_live(path: Path | None = None) -> dict:
    from jevh.config import load_config
    from jevh.decide import SystemOneClient, noul
    cfg = load_config()
    c = SystemOneClient(cfg.jev["url"], cfg.jev["backend"], timeout=300)
    scores = {}
    for case in ALL:
        scores[case.id] = c.ask(hook_state(case.task, case.log), {"q": noul(HOOK_Q)})["q"].noul
    raw = {"question": HOOK_Q, "scores": scores}
    (path or RAW).write_text(json.dumps(raw, indent=1), encoding="utf-8")
    return raw


def load_saved(path: Path | None = None) -> dict:
    """The saved scores, or SystemExit saying why they can't be used (missing, or from other cases or
    another wording of the question: sweeping them would tune on something that no longer exists)."""
    path = path or RAW
    if not path.exists():
        raise SystemExit(f"{path.name} not found in {path.parent}: run `python evals/hook_route_cases.py` against a "
                         "live Jev first (--save-committed to make its scores the committed ones).")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("question") != HOOK_Q:
        raise SystemExit(f"{path.name} was scored with a different HOOK_Q: re-run live.")
    missing = sorted({c.id for c in ALL} - set(raw.get("scores", {})))
    if missing:
        raise SystemExit(f"{path.name} has no scores for {', '.join(missing[:5])}"
                         f"{' ...' if len(missing) > 5 else ''}: re-run live.")
    return raw


# ---------------------------------------------------------------- the decision and its sweep

def decide(score: float, threshold: float) -> bool:
    """Route to a hook. Same rule as the learning gates: at or above the threshold."""
    return score >= threshold


def evaluate(scores: list[float], labels: list[bool], threshold: float) -> dict:
    if len(scores) != len(labels):
        raise ValueError(f"{len(scores)} scores for {len(labels)} labels")
    for s in scores:
        if not (0.0 <= s <= 1.0) or s != s:   # also rejects NaN
            raise ValueError(f"a noul score must be in [0, 1], got {s!r}")
    tp = sum(decide(s, threshold) and l for s, l in zip(scores, labels))
    fp = sum(decide(s, threshold) and not l for s, l in zip(scores, labels))
    pos = sum(labels)
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": pos - tp, "tn": len(labels) - pos - fp,
            "positives": pos, "negatives": len(labels) - pos,
            "recall": tp / pos if pos else None}


def recommend(scores: list[float], labels: list[bool], candidates: tuple[float, ...] = THRESHOLDS,
              margin: float = MARGIN) -> dict | None:
    """The candidate threshold with no false positive and the most recall, at least `margin` above
    the highest-scoring non-checkable case, that still routes at least one checkable case (a threshold
    that routes nothing is not a recommendation). None when there is nothing to tune on (no positives or
    no negatives) or no candidate qualifies (a non-checkable case scores about as high as a checkable one)."""
    if len(scores) != len(labels):
        raise ValueError(f"{len(scores)} scores for {len(labels)} labels")
    neg = [s for s, l in zip(scores, labels) if not l]
    if not neg or not any(labels):
        return None
    floor = max(neg) + margin
    for t in sorted(candidates):
        if t >= floor - 1e-9:   # 0.8 + 0.05 is 0.8500000000000001, which must not rule out 0.85
            r = evaluate(scores, labels, t)
            if r["fp"] == 0 and r["tp"] > 0:
                return {**r, "highest_negative": max(neg), "margin": round(t - max(neg), 4)}
    return None


def by_kind(cases: list[Case], scores: list[float], threshold: float) -> dict[str, dict]:
    """Per kind: how many were routed to a hook, against how many should have been."""
    out = {}
    for kind in KINDS:
        idx = [i for i, c in enumerate(cases) if c.kind == kind]
        if not idx:
            continue
        routed = sum(decide(scores[i], threshold) for i in idx)
        should = sum(cases[i].checkable for i in idx)
        right = sum(decide(scores[i], threshold) == cases[i].checkable for i in idx)
        out[kind] = {"cases": len(idx), "should_route": should, "routed": routed, "right": right}
    return out


def wrong(cases: list[Case], scores: list[float], threshold: float) -> list[tuple[str, str, float]]:
    return [(c.id, "false positive" if not c.checkable else "missed", s)
            for c, s in zip(cases, scores) if decide(s, threshold) != c.checkable]


# ---------------------------------------------------------------- report

def sweep(name: str, cases: list[Case], scores: list[float]) -> None:
    labels = [c.checkable for c in cases]
    print(f"\n== {name} ({len(cases)} cases, {sum(labels)} checkable)")
    for t in THRESHOLDS:
        r = evaluate(scores, labels, t)
        print(f"  t={t:.2f}  routed to a hook {r['tp']}/{r['positives']}  false positives {r['fp']}")
    print("  " + "  ".join(f"{'+' if c.checkable else '-'}{s:.2f}" for c, s in zip(cases, scores)))


def report(raw: dict) -> None:
    tune = [raw["scores"][c.id] for c in TUNING]
    held = [raw["scores"][c.id] for c in HOLDOUT]
    sweep("tuning", TUNING, tune)
    sweep("held out", HOLDOUT, held)
    rec = recommend(tune, [c.checkable for c in TUNING])
    cur = float(DEFAULT_THRESHOLDS["route_hook"])
    print("\n== recommendation (tuning cases only)")
    if rec is None:
        print("  none: no threshold has zero false positives with a margin. Look at the highest-scoring")
        print("  non-checkable cases below; reword HOOK_Q or add cases before choosing one.")
    else:
        print(f"  route_hook = {rec['threshold']}  (routes {rec['tp']}/{rec['positives']} checkable, 0 false positives, "
              f"{rec['margin']} above the highest non-checkable score {rec['highest_negative']:.2f})")
    print(f"  configured: {cur}" + ("" if rec and rec["threshold"] == cur else "  <- NOT TUNED to these scores"))
    t = rec["threshold"] if rec else cur
    print(f"\n== per kind at {t}")
    for kind, k in by_kind(TUNING, tune, t).items():
        print(f"  {kind:<10} {k['right']}/{k['cases']} right  (routed {k['routed']}, should route {k['should_route']})")
    for label, cases, sc in (("tuning", TUNING, tune), ("held out", HOLDOUT, held)):
        bad = wrong(cases, sc, t)
        print(f"\n== {label}: wrong at {t}" + ("" if bad else ": none"))
        for cid, what, s in bad:
            print(f"  {what:<14} {cid}  ({s:.2f})")
    hold = evaluate(held, [c.checkable for c in HOLDOUT], t)
    print(f"\n== held out at {t}: routed {hold['tp']}/{hold['positives']} checkable, {hold['fp']} false positives")


def check(raw: dict) -> int:
    """0 when the configured threshold has no false positive on either set of cases."""
    t = float(DEFAULT_THRESHOLDS["route_hook"])
    code = 0
    for label, cases in (("tuning", TUNING), ("held out", HOLDOUT)):
        r = evaluate([raw["scores"][c.id] for c in cases], [c.checkable for c in cases], t)
        print(f"{label}: {r['fp']} false positives, routed {r['tp']}/{r['positives']} at route_hook={t}")
        code |= 1 if r["fp"] else 0
    return code


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="re-sweep the saved scores")
    ap.add_argument("--check", action="store_true", help="offline; exit 1 on a false positive at the configured threshold")
    _out.add_args(ap)
    a = ap.parse_args()
    dest = _out.Out.of(a)
    if a.offline or a.check:
        raw = load_saved(dest.read(RAW))
    else:
        raw = run_live(dest.write(RAW))
        print(f"raw scores saved to {dest.where(dest.write(RAW))}")
    if a.check:
        sys.exit(check(raw))
    report(raw)


if __name__ == "__main__":
    main()
