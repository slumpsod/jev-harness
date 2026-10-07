"""The session-category eval's own code, offline: its scoring, its labels, and the raw Jev scores committed beside it, so
the numbers the docs quote (and the threshold config.yaml ships) stay reproducible."""
import json
import sys

from jevh.config import ROOT, load_config
from jevh.history import MIXED, settings
from jevh.personas import load_personas

sys.path.insert(0, str(ROOT / "evals"))

import session_category_cases as sc  # noqa: E402

PROBS = {"daily_assistant": 0.2, "fitness_coach": 0.7, "high_end_dev": 0.1}


def test_a_best_category_that_is_not_likely_enough_is_listed_as_no_single_field():
    assert sc.shown(PROBS, 0.6) == "fitness_coach" and sc.shown(PROBS, 0.8) == MIXED


def test_scoring_tells_right_from_wrong_from_unsure_and_ambiguous_sessions_accept_either():
    rows = [{"requests": ["a"], "want": ["fitness_coach"], "probs": PROBS},                       # right
            {"requests": ["b"], "want": ["high_end_dev"], "probs": PROBS},                        # confidently wrong
            {"requests": ["c"], "want": ["fitness_coach", "high_end_dev"], "probs": PROBS},       # either is right
            {"requests": ["d"], "want": ["daily_assistant"], "probs": {"daily_assistant": 0.4, "fitness_coach": 0.35}}]   # unsure
    e = sc.evaluate(rows, 0.6)
    assert (e["right"], e["wrong"], e["mixed"], e["lost_clear"]) == (2, 1, 1, 1) and e["confident_accuracy"] == 2 / 3
    assert e["bad"] == [("b", ["high_end_dev"], "fitness_coach", 0.7)]


def test_every_case_is_about_an_assistant_the_harness_ships_and_is_well_formed():
    shipped = {p.stem for p in (ROOT / "personas").glob("*.yaml")}   # the shipped ones, not the learned or home personas on this machine
    cases = sc.TUNING + sc.HELD + sc.HARD
    assert len(sc.TUNING) >= 50 and len(sc.HELD) >= 25 and len(sc.HARD) >= 20
    for reqs, want in cases:
        wanted = [want] if isinstance(want, str) else list(want)
        assert reqs and all(r.strip() for r in reqs) and set(wanted) <= shipped, reqs
    assert len({tuple(r) for r, _ in cases}) == len(cases)                                       # no session twice


def test_the_saved_scores_give_the_numbers_the_docs_quote_at_the_shipped_threshold():
    saved = json.loads(sc.RAW.read_text(encoding="utf-8"))
    at = settings(load_config().raw)["min_confidence"]
    assert {k: len(v) for k, v in saved.items()} == {"tuning": 51, "held": 27, "hard": 23}
    shipped = {p.stem for p in (ROOT / "personas").glob("*.yaml")}   # the shipped ones, not the learned or home personas on this machine
    assert all(set(r["probs"]) == shipped for rows in saved.values() for r in rows)               # scored over today's assistants
    total = [sc.evaluate(rows, at) for rows in saved.values()]
    assert at == 0.6 and sum(e["wrong"] for e in total) == 0 and sum(e["right"] for e in total) == 99
    assert sum(e["mixed"] for e in total) == 2 and sum(e["n"] for e in total) == 101
    assert sc.evaluate(saved["held"], at)["right"] == 27                                          # the held-out set: all right
    assert sc.evaluate(sum(saved.values(), []), 0.5)["wrong"] == 1                                # (one a lower bar would let through)
