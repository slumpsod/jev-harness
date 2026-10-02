"""The skill/hook router's tuning set (evals/hook_route_data.py) and the code that sweeps it
(evals/hook_route_cases.py). All offline: no Jev is asked here. These tests keep the dataset honest (every
case does what its label assumes, nothing leaks between tuning and held-out), pin what the scorer is
shown to what `learn_skill` shows, and check the sweep, the threshold recommendation and the saved-score
handling on edge cases, so that the one live run that tunes `route_hook` can be trusted.

Not covered, because it needs a live Jev: whether HOOK_Q separates the cases at all. That is what
`python evals/hook_route_cases.py` measures; until it has been run, `route_hook` is untuned.
"""
from __future__ import annotations

import json
import math
import random
import re
import sys
from pathlib import Path

import pytest

from jevh.config import DEFAULT_THRESHOLDS, ROOT, load_config
from jevh.decide import SystemOneClient  # noqa: F401  (the eval imports it lazily; fail here if it moves)
from jevh.jev import Jev
from jevh.learn import HOOK_Q, Learner, hook_state, work_log

from conftest import FakeSystemOne, noul_ans

sys.path.insert(0, str(ROOT / "evals"))
import hook_route_cases as ev  # noqa: E402
import hook_route_data as data  # noqa: E402
from hook_route_data import ALL, HOLDOUT, KINDS, TUNING, Case  # noqa: E402

TUNED = ev.RAW.exists()   # a live run has been saved


def case(kind="checkable", checkable=True, cid="x", task="t", log="CODE:\nx"):
    return Case(cid, kind, task, log, checkable, "why")


# ------------------------------------------------------------------------- the dataset

def test_ids_are_unique_and_every_field_is_filled():
    ids = [c.id for c in ALL]
    assert len(ids) == len(set(ids))
    for c in ALL:
        assert c.kind in KINDS, c.id
        assert isinstance(c.checkable, bool), c.id
        assert c.task.strip() and c.log.strip() and c.why.strip(), c.id
        assert c.id == c.id.strip() and " " not in c.id, c.id


def test_the_sets_are_big_enough_and_balanced():
    assert len(TUNING) >= 40 and len(HOLDOUT) >= 14
    for name, cases in (("tuning", TUNING), ("held out", HOLDOUT)):
        share = sum(c.checkable for c in cases) / len(cases)
        assert 0.3 <= share <= 0.6, f"{name}: {share:.0%} checkable"


def test_every_kind_appears_in_both_sets():
    for cases in (TUNING, HOLDOUT):
        assert {c.kind for c in cases} == set(KINDS)


def test_kinds_and_labels_agree():
    for c in ALL:
        if c.kind == "checkable":
            assert c.checkable, c.id
        if c.kind in ("judgment", "lookalike"):
            assert not c.checkable, c.id
    # mixed cases are the hard ones: both answers must exist, or the set can't tell "mostly checkable" apart
    mixed = [c.checkable for c in TUNING if c.kind == "mixed"]
    assert True in mixed and False in mixed
    # edge cases too: a router that says yes to everything odd, or no to everything odd, must be caught
    edge = [c.checkable for c in TUNING if c.kind == "edge"]
    assert True in edge and False in edge


def test_held_out_cases_share_nothing_with_the_tuning_cases():
    assert not {c.id for c in TUNING} & {c.id for c in HOLDOUT}
    assert not {c.task for c in TUNING} & {c.task for c in HOLDOUT}
    assert not {c.log for c in TUNING} & {c.log for c in HOLDOUT}
    tuning_logs = [c.log for c in TUNING]
    for c in HOLDOUT:   # a held-out log that is a piece of a tuning log (or the reverse) isn't held out
        assert not any(c.log in t or t in c.log for t in tuning_logs), c.id


def test_no_two_cases_in_a_set_are_the_same_work():
    for cases in (TUNING, HOLDOUT):
        logs = [c.log for c in cases]
        assert len(logs) == len(set(logs))
        tasks = [c.task for c in cases]
        assert len(tasks) == len(set(tasks))


def test_logs_look_like_the_work_logs_the_learner_produces():
    # `work_log` (learn.py) renders "CODE:" blocks; a log with none is a different thing from what Jev will see
    for c in ALL:
        assert "CODE:" in c.log, c.id
        assert len(c.log) < 60_000, c.id


def test_cues_are_in_the_log_and_the_window_shows_or_hides_them_as_labelled():
    cued = [c for c in ALL if c.cue]
    assert cued, "no edge case tests the log window"
    for c in cued:
        assert c.cue in c.log, c.id
        assert c.cue_in_window is not None, c.id
        seen = hook_state(c.task, c.log)["work_log"]
        assert (c.cue in seen) is c.cue_in_window, f"{c.id}: the window {'hides' if c.cue_in_window else 'shows'} its cue"
    # both directions are covered, in both sets that matter
    assert any(c.cue_in_window is True for c in TUNING) and any(c.cue_in_window is False for c in TUNING)
    assert any(c.cue_in_window is True for c in HOLDOUT)
    # a case whose cue is cut off must be labelled from what is left, i.e. not checkable
    assert all(not c.checkable for c in ALL if c.cue_in_window is False)


def test_a_long_task_is_cut_where_the_scorer_cuts_it():
    long_task = [c for c in ALL if len(c.task) > 1500]
    assert long_task, "no case has a task longer than the window"
    for c in long_task:
        assert len(hook_state(c.task, c.log)["task"]) == 1500
        assert "Final deliverable" not in hook_state(c.task, c.log)["task"]   # the part that is cut is the last sentence
        assert c.checkable   # the log still shows the rule, which is what makes the case an edge


def test_the_odd_input_cases_exist():
    edge = [c for c in ALL if c.kind == "edge"]
    assert any(not c.task.isascii() or not c.log.isascii() for c in edge), "no non-ASCII case"
    assert any(any(ord(ch) > 0xFFFF for ch in c.task + c.log) for c in ALL), "no emoji case"
    assert any(c.task.lower().strip() in ("fix it", "do it", "help") for c in edge), "no vague-task case"
    steer = [c for c in edge if "classifier" in c.log.lower() or "ignore the previous" in c.log.lower()]
    assert steer and all(not c.checkable for c in steer), "the steering-text case must exist and be a no"
    assert any(c.task.lower().startswith(("cr", "écr")) for c in ALL), "no non-English task"
    assert any(len(c.log) < 120 for c in edge), "no very short log"


def test_the_trap_cases_the_design_cares_about_are_present_and_are_no():
    by_id = {c.id: c for c in ALL}
    # the tool lesson that started this ("don't call web.ask"): a grep could find it, but too late
    assert not by_id["web-ask-times-out"].checkable
    # live-server facts and side-effect procedures are not checkable from finished files
    for cid in ("comfy-model-name", "deploy-pages", "login-and-read-bill", "h-service-healthy"):
        assert not by_id[cid].checkable, cid
    # a schema is checkable, the content isn't
    for cid in ("invoice-extraction", "h-json-summary"):
        assert not by_id[cid].checkable, cid
    # and the cases the router exists for
    for cid in ("html-dark-mode", "json-required-keys", "csv-headers", "h-sitemap-xml"):
        assert by_id[cid].checkable, cid


# ------------------------------------------------------------------------- what the scorer is shown

def test_hook_state_uses_the_learn_skill_window():
    t, g = "T" * 2000, "L" * 5000 + "end"
    s = hook_state(t, g)
    assert set(s) == {"task", "work_log"}
    assert s["task"] == "T" * 1500 and len(s["work_log"]) == 3500 and s["work_log"].endswith("end")


@pytest.mark.parametrize("task,log", [("", ""), ("a", ""), ("", "b"), ("short", "short"), ("é" * 1500, "🙂" * 3500),
                                      ("x" * 1500, "y" * 3500)])
def test_hook_state_edges(task, log):
    s = hook_state(task, log)
    assert s == {"task": task[:1500], "work_log": log[-3500:]}
    s["task"].encode("utf-8"), s["work_log"].encode("utf-8")   # slicing never leaves a lone surrogate


def test_hook_state_is_what_learn_skill_sends_jev(cfg, tmp_path):
    """If someone changes the window in `Learner.skill`, the eval would score something the router never sees."""
    from jevh.memory import Memory
    fake = FakeSystemOne(lambda k, q, s: noul_ans(0.0) if k == "skill" else None)
    mem = Memory(tmp_path / "mem")
    L = Learner(Jev(cfg, fake), mem, tmp_path / "skills", cfg.t)
    task = "build a page " + "x" * 2000
    steps = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    for i in range(60):   # a transcript long enough to be cut
        steps += [{"role": "assistant", "content": f"```python\nprint({i} * 'a')\n```"},
                  {"role": "user", "content": "Output:\n" + "ok " * 40}]
    L.skill(task, steps, [], llm=None)
    sent = fake.calls[0][0]
    assert sent == hook_state(task, work_log(steps))
    mem.close()


def test_the_question_is_a_question_about_the_state_it_is_given():
    assert HOOK_Q.endswith("?") and HOOK_Q.count("?") == 1
    assert "`work_log`" in HOOK_Q and "work_log" in hook_state("t", "l")
    assert "program" in HOOK_Q
    assert "{" not in HOOK_Q and "}" not in HOOK_Q   # not a format string waiting for a value


# ------------------------------------------------------------------------- the threshold: config and notes

def test_the_default_threshold_is_valid_and_the_same_in_config_yaml_and_code():
    t = DEFAULT_THRESHOLDS["route_hook"]
    assert 0.0 < t < 1.0
    assert load_config().t("route_hook") == t   # config.yaml (no local config under test) agrees with the code default


def test_it_is_marked_untuned_until_a_live_run_has_been_saved():
    """The notes come off when `python evals/hook_route_cases.py` has been run and its scores saved
    (evals/hook_route_raw.json); until then every place a reader meets the threshold says it isn't tuned."""
    if TUNED:
        pytest.skip("scores saved: the untuned notes are for the time before the first live run")
    def near(path: str, anchor: str, before: int = 0, after: int = 400) -> str:
        """The text around `anchor` in a file: the note has to sit where the threshold or question is,
        not just somewhere in a file that also tunes other things."""
        text = (ROOT / path).read_text(encoding="utf-8")
        assert anchor in text, f"{path} has no {anchor!r}"
        i = text.index(anchor)
        return text[max(0, i - before): i + after]

    untuned = re.compile(r"not tuned|untuned", re.I)
    notes = {
        "config.yaml": near("config.yaml", "  route_hook:", 0, 300),
        "jevh/learn.py": near("jevh/learn.py", "HOOK_Q = (", 450, 0),
        "evals/hook_route_cases.py": near("evals/hook_route_cases.py", "NOT TUNED", 0, 0) +
                                     (ROOT / "evals/hook_route_cases.py").read_text(encoding="utf-8")[:700],
        "docs/architecture/14-skill-hook-router.md":
            (ROOT / "docs/architecture/14-skill-hook-router.md").read_text(encoding="utf-8")[:1600],
    }
    for name, text in notes.items():
        assert untuned.search(text), f"{name} doesn't say next to route_hook / HOOK_Q that it is untuned"
    doc = notes["docs/architecture/14-skill-hook-router.md"]
    assert "never been run against a live Jev" in doc and "evals/hook_route_cases.py" in doc


# ------------------------------------------------------------------------- decide / evaluate

def test_a_score_equal_to_the_threshold_routes():
    assert ev.decide(0.85, 0.85) and not ev.decide(0.8499999, 0.85)
    assert ev.decide(1.0, 1.0) and not ev.decide(0.0, 0.01) and ev.decide(0.0, 0.0)


def test_evaluate_counts():
    r = ev.evaluate([0.9, 0.8, 0.4, 0.95, 0.1], [True, True, True, False, False], 0.85)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 1, 2, 1)
    assert r["positives"] == 3 and r["negatives"] == 2 and r["recall"] == pytest.approx(1 / 3)


def test_evaluate_edges():
    assert ev.evaluate([], [], 0.5) == {"threshold": 0.5, "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                                        "positives": 0, "negatives": 0, "recall": None}
    r = ev.evaluate([0.2, 0.3], [False, False], 0.5)           # nothing checkable: recall is undefined, not 0
    assert r["recall"] is None and r["tn"] == 2
    r = ev.evaluate([0.6, 0.7], [True, True], 0.5)             # nothing to be wrong about on the negative side
    assert r["recall"] == 1.0 and r["fp"] == 0 and r["negatives"] == 0
    assert ev.evaluate([0.5, 0.6], [True, False], 0.0)["tp"] == 1 and ev.evaluate([0.5, 0.6], [True, False], 0.0)["fp"] == 1
    assert ev.evaluate([0.5, 0.6], [True, False], 1.5)["tp"] == 0     # above every possible score: routes nothing
    assert ev.evaluate([1.0], [False], 1.0)["fp"] == 1                # a certain "yes" on a non-checkable case


@pytest.mark.parametrize("scores", [[1.2], [-0.1], [float("nan")], [float("inf")], [0.5, 2]])
def test_evaluate_refuses_scores_that_are_not_probabilities(scores):
    with pytest.raises(ValueError):
        ev.evaluate(scores, [True] * len(scores), 0.5)


def test_evaluate_and_recommend_refuse_mismatched_lengths():
    with pytest.raises(ValueError):
        ev.evaluate([0.5], [True, False], 0.5)
    with pytest.raises(ValueError):
        ev.recommend([0.5], [True, False])


# ------------------------------------------------------------------------- recommend

def test_recommend_takes_the_lowest_threshold_with_no_false_positive_and_a_margin():
    scores = [0.97, 0.92, 0.88, 0.60, 0.35, 0.10]
    labels = [True, True, True, False, False, False]
    r = ev.recommend(scores, labels)
    assert r["threshold"] == 0.7 and r["fp"] == 0 and r["tp"] == 3 and r["highest_negative"] == 0.6
    assert r["margin"] == pytest.approx(0.1)


def test_recommend_keeps_its_distance_from_the_highest_negative():
    # 0.7 would have no false positive, but is only 0.01 above a negative: not a safe place to sit
    r = ev.recommend([0.95, 0.91, 0.69, 0.2], [True, True, False, False])
    assert r["threshold"] == 0.8
    assert ev.recommend([0.95, 0.91, 0.69, 0.2], [True, True, False, False], margin=0.0)["threshold"] == 0.7


def test_recommend_may_give_up_recall_for_zero_false_positives():
    # a checkable case at 0.5 sits below a non-checkable one at 0.8: it is missed, never traded for a false positive
    scores, labels = [0.99, 0.5, 0.8, 0.1], [True, True, False, False]
    r = ev.recommend(scores, labels)
    assert r["fp"] == 0 and r["tp"] == 1 and r["fn"] == 1 and r["threshold"] == 0.85


def test_recommend_says_none_when_nothing_qualifies():
    assert ev.recommend([0.9, 0.88], [False, True]) is None            # a negative above the positive: only a
                                                                        # threshold that routes nothing has no false positive
    assert ev.recommend([0.9, 0.9], [True, False]) is None             # a tie
    assert ev.recommend([0.99, 0.5], [False, True]) is None            # a negative at 0.99: no candidate 0.05 above
    assert ev.recommend([0.5, 0.6], [True, True]) is None              # nothing to tune against
    assert ev.recommend([0.5, 0.6], [False, False]) is None            # nothing to find
    assert ev.recommend([], []) is None


def test_recommend_when_every_negative_scores_zero():
    r = ev.recommend([0.99, 0.0, 0.0], [True, False, False])
    assert r["threshold"] == min(ev.THRESHOLDS) and r["tp"] == 1


def test_recommend_with_a_negative_exactly_on_a_candidate():
    r = ev.recommend([0.99, 0.8], [True, False], margin=0.0)           # 0.8 itself would be a false positive
    assert r["threshold"] == 0.85 and r["fp"] == 0


def test_recommend_ignores_the_order_of_the_cases():
    rng = random.Random(7)
    pairs = [(0.97, True), (0.93, True), (0.9, True), (0.55, False), (0.4, False), (0.05, False), (0.85, True)]
    want = ev.recommend([s for s, _ in pairs], [l for _, l in pairs])
    for _ in range(20):
        rng.shuffle(pairs)
        assert ev.recommend([s for s, _ in pairs], [l for _, l in pairs]) == want


def test_recommend_and_evaluate_properties_on_random_scores():
    rng = random.Random(1234)
    for _ in range(300):
        n = rng.randint(1, 40)
        scores = [round(rng.random(), 3) for _ in range(n)]
        labels = [rng.random() < 0.45 for _ in range(n)]
        recalls = [ev.evaluate(scores, labels, t)["recall"] for t in ev.THRESHOLDS]
        known = [r for r in recalls if r is not None]
        assert known == sorted(known, reverse=True)                     # a higher bar never catches more
        fps = [ev.evaluate(scores, labels, t)["fp"] for t in ev.THRESHOLDS]
        assert fps == sorted(fps, reverse=True)                         # ...nor lets more false positives through
        rec = ev.recommend(scores, labels)
        if rec is not None:
            assert rec["fp"] == 0
            assert rec["threshold"] - max(s for s, l in zip(scores, labels) if not l) >= ev.MARGIN - 1e-9
            lower = [t for t in ev.THRESHOLDS if t < rec["threshold"]]
            for t in lower:    # nothing lower qualifies: either a false positive or inside the margin
                r = ev.evaluate(scores, labels, t)
                assert r["fp"] > 0 or t < max(s for s, l in zip(scores, labels) if not l) + ev.MARGIN


# ------------------------------------------------------------------------- by_kind / wrong

def test_by_kind_counts_what_was_routed_against_what_should_have_been():
    cases = [case("checkable", True, "a"), case("checkable", True, "b"), case("judgment", False, "c"),
             case("lookalike", False, "d"), case("edge", True, "e")]
    scores = [0.9, 0.2, 0.1, 0.9, 0.9]
    k = ev.by_kind(cases, scores, 0.85)
    assert k["checkable"] == {"cases": 2, "should_route": 2, "routed": 1, "right": 1}
    assert k["judgment"] == {"cases": 1, "should_route": 0, "routed": 0, "right": 1}
    assert k["lookalike"] == {"cases": 1, "should_route": 0, "routed": 1, "right": 0}
    assert k["edge"] == {"cases": 1, "should_route": 1, "routed": 1, "right": 1}
    assert "mixed" not in k                                           # a kind with no cases isn't reported
    assert ev.by_kind([], [], 0.5) == {}


def test_wrong_names_the_false_positives_and_the_misses():
    cases = [case(cid="a", checkable=True), case("judgment", False, "b"), case(cid="c", checkable=True),
             case("judgment", False, "d")]
    got = ev.wrong(cases, [0.9, 0.9, 0.1, 0.1], 0.85)
    assert got == [("b", "false positive", 0.9), ("c", "missed", 0.1)]
    assert ev.wrong([], [], 0.5) == []


# ------------------------------------------------------------------------- saved scores

def fake_scores(neg=0.1, pos=0.95, **override):
    scores = {c.id: (pos if c.checkable else neg) for c in ALL}
    scores.update(override)
    return {"question": HOOK_Q, "scores": scores}


@pytest.fixture
def raw_path(tmp_path, monkeypatch):
    p = tmp_path / "hook_route_raw.json"
    monkeypatch.setattr(ev, "RAW", p)
    return p


def test_saved_scores_load_when_they_match_the_cases_and_the_question(raw_path):
    raw_path.write_text(json.dumps(fake_scores()), encoding="utf-8")
    assert ev.load_saved()["scores"].keys() == {c.id for c in ALL}


def test_saved_scores_that_cannot_be_used_stop_the_run_with_a_reason(raw_path):
    with pytest.raises(SystemExit, match="run `python evals/hook_route_cases.py`"):
        ev.load_saved()                                               # nothing saved yet
    stale = fake_scores()
    stale["question"] = HOOK_Q + " (older wording)"
    raw_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(SystemExit, match="different HOOK_Q"):
        ev.load_saved()
    partial = fake_scores()
    del partial["scores"][TUNING[0].id]
    raw_path.write_text(json.dumps(partial), encoding="utf-8")
    with pytest.raises(SystemExit, match=TUNING[0].id):
        ev.load_saved()
    raw_path.write_text(json.dumps({"scores": {}}), encoding="utf-8")   # an old file without the question
    with pytest.raises(SystemExit):
        ev.load_saved()


def test_scores_for_cases_that_were_since_removed_do_not_break_loading(raw_path):
    extra = fake_scores(**{"a-case-we-deleted": 0.5})
    raw_path.write_text(json.dumps(extra), encoding="utf-8")
    assert "a-case-we-deleted" in ev.load_saved()["scores"]            # ignored, not an error


# ------------------------------------------------------------------------- check / report / cli

def test_check_passes_only_without_false_positives_in_either_set():
    assert ev.check(fake_scores()) == 0
    t = float(DEFAULT_THRESHOLDS["route_hook"])
    bad_tuning = next(c.id for c in TUNING if not c.checkable)
    bad_held = next(c.id for c in HOLDOUT if not c.checkable)
    assert ev.check(fake_scores(**{bad_tuning: t})) == 1               # exactly at the threshold counts
    assert ev.check(fake_scores(**{bad_held: 0.99})) == 1
    assert ev.check(fake_scores(**{bad_tuning: t - 0.001})) == 0
    missed = next(c.id for c in TUNING if c.checkable)
    assert ev.check(fake_scores(**{missed: 0.0})) == 0                 # a miss is a fallback to a skill, not a failure


def test_report_recommends_a_threshold_and_says_when_it_is_not_the_configured_one(capsys):
    ev.report(fake_scores(neg=0.1, pos=0.97))
    out = capsys.readouterr().out
    assert "== tuning" in out and "== held out" in out and "== recommendation" in out
    for t in ev.THRESHOLDS:
        assert f"t={t:.2f}" in out
    assert "route_hook = 0.3" in out and "NOT TUNED" in out            # 0.3 is the lowest safe threshold; config says 0.85
    assert "wrong at 0.3: none" in out
    for kind in KINDS:
        assert kind in out


def test_report_when_nothing_separates_the_cases(capsys):
    muddled = fake_scores(neg=0.9, pos=0.9)
    ev.report(muddled)
    out = capsys.readouterr().out
    assert "none: no threshold has zero false positives" in out
    assert "false positive" in out                                    # and it lists which cases were wrong


def test_report_lists_each_wrong_case(capsys):
    neg = next(c.id for c in TUNING if c.kind == "lookalike")
    pos = next(c.id for c in HOLDOUT if c.checkable)
    ev.report(fake_scores(**{neg: 0.6, pos: 0.2}))
    out = capsys.readouterr().out
    assert f"missed         {pos}" in out
    assert "false positive" not in out.split("== per kind")[1] or neg not in out.split("== held out: wrong")[0]


def test_offline_without_saved_scores_says_what_to_do(raw_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["hook_route_cases.py", "--offline"])
    with pytest.raises(SystemExit, match="live Jev"):
        ev.main()
    monkeypatch.setattr(sys, "argv", ["hook_route_cases.py", "--check"])
    with pytest.raises(SystemExit, match="live Jev"):
        ev.main()


def test_check_flag_sets_the_exit_code(raw_path, monkeypatch):
    raw_path.write_text(json.dumps(fake_scores()), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["hook_route_cases.py", "--check"])
    with pytest.raises(SystemExit) as e:
        ev.main()
    assert e.value.code == 0
    bad = next(c.id for c in HOLDOUT if not c.checkable)
    raw_path.write_text(json.dumps(fake_scores(**{bad: 0.99})), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        ev.main()
    assert e.value.code == 1


def test_a_live_run_scores_every_case_with_the_state_the_router_will_send(raw_path, monkeypatch, cfg):
    """run_live with a fake Jev client: every case asked once, with hook_state and HOOK_Q, and saved."""
    asked = []

    class Client:
        def __init__(self, *a, **k):
            pass

        def ask(self, state, questions):
            asked.append((state, questions))

            class A:
                noul = 0.5 if "viewport" in state["work_log"] else 0.1
            return {"q": A()}

    import jevh.decide
    monkeypatch.setattr(jevh.decide, "SystemOneClient", Client)
    raw = ev.run_live()
    assert len(asked) == len(ALL) and set(raw["scores"]) == {c.id for c in ALL}
    assert json.loads(raw_path.read_text(encoding="utf-8")) == raw and raw["question"] == HOOK_Q
    for (state, qs), c in zip(asked, ALL):
        assert state == hook_state(c.task, c.log)
        assert list(qs) == ["q"] and HOOK_Q in json.dumps(qs)


# ------------------------------------------------------------------------- once a live run has been saved

@pytest.mark.skipif(not TUNED, reason="no live run saved yet (evals/hook_route_raw.json): route_hook is untuned")
def test_saved_scores_match_the_current_cases_and_the_configured_threshold_has_no_false_positive():
    raw = ev.load_saved()
    assert ev.check(raw) == 0
    assert all(0.0 <= s <= 1.0 and not math.isnan(s) for s in raw["scores"].values())
    rec = ev.recommend([raw["scores"][c.id] for c in TUNING], [c.checkable for c in TUNING])
    assert rec is not None and float(DEFAULT_THRESHOLDS["route_hook"]) >= rec["threshold"], \
        "route_hook is below the lowest threshold that has no false positive on the tuning cases"
