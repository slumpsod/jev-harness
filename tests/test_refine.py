"""Refinement: checkers, test generation, the matrix, the verdict and report, and the whole loop
(on a throwaway git repo, with scripted models and a fake runner)."""
import json
import subprocess
import time
from pathlib import Path

import pytest

from jevh.failures import FailureLog
from jevh.jev import Jev
from jevh.refine import checks, report, runner
from jevh.refine.cases import Case, controls
from jevh.refine.generate import Generator, parse_json
from jevh.refine.loop import Refiner
from jevh.refine.run_case import run_one

from conftest import FakeSystemOne, ScriptedLLM, choice_ans, py


def judge(q, state):   # a fake Jev judge: says yes to plans that list days
    return 0.9 if "Day 1" in state["answer"] else 0.1


# ---------------------------------------------------------------- checkers

def test_checkers(tmp_path):
    page = tmp_path / "index.html"
    page.write_text("<h1>Rise and Crumb</h1>", encoding="utf-8")
    c = checks.check
    assert c({"type": "contains", "text": "Apple,  Banana"}, "apple, banana")[0]
    assert not c({"type": "not_contains", "text": "sorry"}, "Sorry, no.")[0]
    assert c({"type": "regex", "pattern": r"\$4\d\d"}, "TAP at $412")[0]
    assert c({"type": "number", "value": 1284, "tolerance": 0.5}, "Delta is $1,284.00")[0]
    assert not c({"type": "number", "value": 1284}, "Delta is $1,099")[0]
    assert c({"type": "json_equals", "value": [1, 2]}, "Here: [1, 2] done")[0]
    assert c({"type": "artifact", "name": "*.html", "contains": "crumb"}, "", [str(page)])[0]
    assert not c({"type": "artifact", "name": "*.html"}, "", [])[0]
    assert c({"type": "python", "code": "def check(answer, artifacts):\n    return answer.count('x') == 2"}, "xx")[0]
    assert not c({"type": "python", "code": "def check(answer, artifacts):\n    raise ValueError('no')"}, "x")[0]
    assert c({"type": "jev", "question": "Is it a plan?"}, "Day 1: squats", judge=judge)[0]
    both = {"type": "all", "checks": [{"type": "contains", "text": "a"}, {"type": "contains", "text": "b"}]}
    assert c(both, "ab")[0] and not c(both, "a")[0]
    assert c({"type": "any", "checks": both["checks"]}, "b")[0]


def test_a_checker_must_tell_good_from_bad():
    good = {"type": "number", "value": 45}
    assert checks.discriminates(good, "45 days", "about six weeks")[0]
    lax = {"type": "regex", "pattern": ".*"}
    ok, why = checks.discriminates(lax, "45 days", "about six weeks")
    assert not ok and "accepts a bad answer" in why
    assert not checks.discriminates({"type": "number", "value": 45}, "six weeks", "")[0]   # rejects the good one
    assert not checks.discriminates({"type": "nope"}, "a", "b")[0]
    assert not checks.discriminates({"type": "all", "checks": []}, "a", "b")[0]


def test_the_control_tasks_have_checkers_that_hold_up():
    cs = controls()
    assert len(cs) >= 6 and all(c.split == "control" for c in cs)
    for c in cs:
        assert checks.discriminates(c.checker, c.good_answer, c.bad_answer, c.request, judge) == (True, "ok"), c.request


# ---------------------------------------------------------------- generation

FAILURE = {"id": 7, "kind": "gave_up", "source": "jev", "summary": "said it couldn't compute it",
           "request": "What is 12% of 850?", "answer": "I can't do maths reliably, sorry.", "persona": "daily_assistant",
           "model": "m1", "detail": "", "work_log": ""}


def variant(req, value, bad="I can't work that out."):
    return {"request": req, "checker": {"type": "number", "value": value},
            "good_answer": f"It is {value}.", "bad_answer": bad}


def test_generator_keeps_only_cases_whose_checker_holds_up():
    replay = json.dumps({"checker": {"type": "number", "value": 102}, "good_answer": "102"})
    first = json.dumps([variant("What is 15% of 60?", 9), variant("What is 15% of 60?", 9),   # a duplicate
                        {"request": "What is 20% of 45?", "checker": {"type": "regex", "pattern": "."},
                         "good_answer": "9", "bad_answer": "no"},                          # accepts anything
                        variant("What is 3% of 900?", 27)])
    second = "Sure! ```json\n" + json.dumps([variant(f"What is {p}% of 200?", p * 2) for p in (5, 10, 25)]) + "\n```"
    gen = Generator(ScriptedLLM([replay, first, second]), judge, cases=5, holdout=0.4, seed=1)
    r = gen.replay(FAILURE)
    assert r.origin == "replay" and r.persona == "daily_assistant" and r.bad_answer.startswith("I can't")
    vs = gen.variants(FAILURE, "daily_assistant")
    assert [v.request for v in vs] == ["What is 15% of 60?", "What is 3% of 900?", "What is 5% of 200?",
                                       "What is 10% of 200?", "What is 25% of 200?"]
    assert sum(v.split == "holdout" for v in vs) == 2
    assert {d["why"] for d in gen.dropped} >= {"a duplicate"} and any("accepts" in d["why"] for d in gen.dropped)


def test_replay_needs_a_checker_that_rejects_the_failed_answer():
    lax = json.dumps({"checker": {"type": "contains", "text": "sorry"}, "good_answer": "102, sorry for the wait"})
    gen = Generator(ScriptedLLM([lax, "no json here", lax]), judge, rounds=3)
    assert gen.replay(FAILURE) is None and len(gen.dropped) == 3


def test_a_reply_cut_off_is_recorded_not_skipped_silently():
    replay = json.dumps({"checker": {"type": "number", "value": 102}, "good_answer": "102"})
    # cut off mid-list: the only list that parses is a checker's "checks", which holds no tasks
    cut = ('[{"request": "What is 1% of 5?", "checker": {"type": "all", "checks": [{"type": "number", '
           '"value": 0.05}]}, "good_answer": "0.0\n[output cut off at max_tokens: ~8000 thinking + ~150 answer tokens]')
    gen = Generator(ScriptedLLM([replay, cut, json.dumps([variant("What is 3% of 900?", 27)])]), judge, cases=1)
    assert gen.replay(FAILURE) is not None
    assert [v.request for v in gen.variants(FAILURE)] == ["What is 3% of 900?"]
    assert len(gen.dropped) == 1 and gen.dropped[0]["request"] == "(the whole reply)"
    assert gen.dropped[0]["why"].startswith("no list of tasks: the reply was cut off at max_tokens")
    assert [r["kind"] for r in gen.replies] == ["replay", "variants", "variants"]


def test_parse_json_reads_through_chatter():
    assert parse_json('ok: [1, {"a": 2}] trailing') == [1, {"a": 2}]
    assert parse_json('x {"a": [1]} y', "{") == {"a": [1]}
    assert parse_json("nothing") is None


# ---------------------------------------------------------------- matrix, verdict, report

def fake_exec(passes, calls=None):
    def run(root, case, model, ws):
        if calls is not None:
            calls.append((root.name, model, case["id"]))
        ok = passes(root, case, model)
        return {"answer": case["good_answer"] if ok else case["bad_answer"], "seconds": 10.0, "input": 100,
                "output": 20, "thinking": 5, "failures": []}
    return run


def test_matrix_is_resumable_and_stops_at_the_deadline(tmp_path):
    cs = [Case("a", "q", {"type": "contains", "text": "yes"}, good_answer="yes", bad_answer="no"),
          Case("b", "q", {"type": "contains", "text": "yes"}, split="control", good_answer="yes", bad_answer="no")]
    calls = []
    ex = fake_exec(lambda root, case, model: root.name == "new", calls)
    versions = {"old": tmp_path / "old", "new": tmp_path / "new"}
    res = tmp_path / "results.jsonl"
    assert not runner.run_matrix(cs, versions, ["m1", "m2"], 2, res, tmp_path / "runs", None, ex, time.time() - 1)
    assert calls == []
    assert runner.run_matrix(cs, versions, ["m1"], 1, res, tmp_path / "runs", None, ex)
    assert calls == [("old", "m1", "a"), ("new", "m1", "a"), ("old", "m1", "b"), ("new", "m1", "b")]
    runner.run_matrix(cs, versions, ["m1"], 2, res, tmp_path / "runs", None, ex)   # only rep 1 is new
    assert len(calls) == 8 and len(runner.load_rows(res)) == 8
    rows = runner.load_rows(res)
    assert all(r["passed"] == (r["version"] == "new") for r in rows)


def test_a_run_error_scores_as_not_done():
    case = Case("a", "q", {"type": "contains", "text": "yes"})
    assert runner.score(case, {"error": "timed out after 3600s", "answer": "yes"}, None) == (
        False, "run error: timed out after 3600s")
    assert runner.score(Case("b", "q", {"type": "jev", "question": "?"}), {"answer": "x"}, None)[1].startswith("checker error")


def test_only_local_models_are_tested(cfg):
    cfg.providers = {**cfg.providers, "anthropic": {"type": "anthropic"}}
    cfg.models = {"strong": {"provider": "lmstudio", "model": "qwen"}, "fast": {"provider": "anthropic", "model": "claude"}}
    assert runner.local_models(cfg) == ["lmstudio:qwen"]
    cfg.raw = {**cfg.raw, "refine": {"models": ["lmstudio:a", "openai:gpt", "anthropic:claude", "lmstudio:a"]}}
    assert runner.local_models(cfg) == ["lmstudio:a"]     # openai here needs an API key: not local


def rows_for(version, model, split, passed, n, seconds=10.0, tokens=100):
    return [{"version": version, "model": model, "case": f"{split}{i}", "rep": 0, "split": split,
             "passed": i < passed, "seconds": seconds, "input": tokens, "output": 0, "thinking": 0} for i in range(n)]


def test_verdicts():
    base = rows_for("old", "m1", "dev", 2, 8) + rows_for("old", "m1", "holdout", 1, 4) + rows_for("old", "m1", "control", 5, 5)
    good = rows_for("new", "m1", "dev", 7, 8) + rows_for("new", "m1", "holdout", 3, 4) + rows_for("new", "m1", "control", 5, 5)
    v = report.verdict(base + good, "old", "new")
    assert v["verdict"] == "improves" and v["gain"] == 0.583 and v["p"] < 0.05
    broke = good[:-3] + rows_for("new", "m1", "control", 2, 5)[2:]   # 2/5 controls instead of 5/5
    v = report.verdict(base + broke, "old", "new")
    assert v["verdict"] == "mixed" and any("controls got worse" in c for c in v["caveats"])
    same = [dict(r, version="new") for r in base]
    assert report.verdict(base + same, "old", "new")["verdict"] == "no_gain"
    slow = rows_for("new", "m1", "dev", 7, 8, seconds=30) + rows_for("new", "m1", "holdout", 3, 4, seconds=30) + \
        rows_for("new", "m1", "control", 5, 5, seconds=30)
    v = report.verdict(base + slow, "old", "new")
    assert v["verdict"] == "improves" and any("time per run up 200%" in c for c in v["caveats"])


def test_p_better():
    assert report.p_better(5, 10, 5, 10) > 0.5
    assert report.p_better(0, 10, 10, 10) < 0.001
    assert report.p_better(0, 0, 3, 3) == 1.0


# ---------------------------------------------------------------- one run, in process

def test_run_one_reports_answer_time_tokens_and_failures(cfg, tmp_path):
    cfg.workspace = tmp_path / "ws"
    rules = lambda k, q, s: choice_ans("daily_assistant", 0.9, q["criteria"]) if k == "persona" else None  # noqa: E731
    llm = ScriptedLLM([py("print(report_failure('flaky'))"), py("final('It is 102.')")])
    out = run_one({"request": "What is 12% of 850?", "persona": "daily_assistant"}, cfg, "",
                  jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: llm)
    assert out["answer"] == "It is 102." and out["status"] == "done" and out["failures"] == ["reported"]
    assert out["input"] == 20 and out["output"] == 10 and out["seconds"] >= 0
    assert out["jevh_file"].endswith("__init__.py")


def test_execute_runs_the_code_of_the_tree_it_is_given(tmp_path):
    """The real subprocess path: no model server here, so the run fails - but it ran, in that tree's code."""
    root = Path(__file__).resolve().parent.parent
    res = runner.execute(root, {"id": "x", "request": "hi", "persona": "daily_assistant"}, "nowhere:none",
                         tmp_path / "run", timeout=120)
    assert res.get("jevh_file", "").startswith(str(root))
    assert "error" in res or res.get("status") == "failed"


# ---------------------------------------------------------------- the whole loop

def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for k in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(k, "Test")
    for k in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(k, "test@example.com")
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    (r / "harness.txt").write_text("broken", encoding="utf-8")
    (r / ".gitignore").write_text("workspace/\n", encoding="utf-8")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "init")
    return r


def make_refiner(cfg, repo, fix_text="fixed", passes=None, n_variants=6):
    cfg.root, cfg.workspace = repo, repo / "workspace"
    cfg.raw = {**cfg.raw, "refine": {"model": "lmstudio:m1", "models": ["lmstudio:m1", "lmstudio:m2"], "runs": 1,
                                     "repro_runs": 2, "cases": 6, "min_cases": 5, "fix_tries": 2,
                                     "test_command": "python -c \"print('tests ok')\""}}
    log = FailureLog(cfg.workspace / "failures.db")
    fid = log.record(source="jev", kind="gave_up", summary="said it couldn't compute it", persona="daily_assistant",
                     model="m1", request=FAILURE["request"], answer=FAILURE["answer"])["id"]
    log.record(source="jev", kind="gave_up", summary="gave up on a sum", persona="daily_assistant",
               request="What is 7% of 300?", answer="Sorry.")
    replay = json.dumps({"checker": {"type": "number", "value": 102}, "good_answer": "It is 102."})
    variants = json.dumps([variant(f"What is {p}% of 400?", p * 4) for p in range(1, n_variants + 1)])
    llm = ScriptedLLM([replay, variants])
    fixed = lambda root: (root / "harness.txt").read_text(encoding="utf-8") == "fixed"  # noqa: E731
    passes = passes or (lambda root, case, model: case["split"] == "control" or fixed(root))
    turns = []

    def run_turn(tree, task):
        turns.append(task)
        (tree / "harness.txt").write_text(fix_text, encoding="utf-8")
        (tree / "tests").mkdir(exist_ok=True)
        (tree / "tests" / "test_rigged.py").write_text("# a fix may not touch the tests", encoding="utf-8")
        return "Made the assistant compute percentages in code instead of giving up."
    ref = Refiner(cfg, llm=llm, judge=judge, execute_fn=fake_exec(passes), run_turn=run_turn, log=lambda m: None)
    return ref, fid, turns


def test_the_loop_proposes_a_fix_that_works_and_approval_merges_it(cfg, repo):
    ref, fid, turns = make_refiner(cfg, repo)
    assert [g["count"] for g in ref.groups()] == [2]
    aid = ref.start(ref.pick()["latest"]["id"])
    st = ref.run(aid)
    assert st["stage"] == "proposed" and st["verdict"] == "improves"
    assert st["repro"] == {"model": "lmstudio:m1", "runs": 2, "failed": 2}
    t = st["tries"][0]
    assert t["files"] == ["harness.txt"] and t["reverted"] == ["tests/test_rigged.py"] and t["tests_ok"]
    # the fixer saw the dev cases and the old results, never the held-out ones
    cases = ref.cases(st)
    held = [c for c in cases if c.split == "holdout"]
    assert held and not any(c.request in turns[0] for c in held)
    assert "What is 12% of 850?" in turns[0] and "old: 0/2 passed" in turns[0]
    md = (ref.dir(aid) / "report.md").read_text(encoding="utf-8")
    for part in ("Verdict: ✅ Improves", "The failure that caught it", "What changed", "How much it improved",
                 "Caveats", "thrown away (protected)", "Reproduced 2/2 times", "compute percentages in code"):
        assert part in md, part
    assert ref.pick() is None   # both failures of the group are being handled
    commit = ref.approve(aid)
    assert (repo / "harness.txt").read_text(encoding="utf-8") == "fixed"
    assert not (repo / "tests" / "test_rigged.py").exists()
    assert "Refine" in git(repo, "log", "-1", "--format=%s") and commit == git(repo, "rev-parse", "HEAD").strip()
    assert {f["status"] for f in ref.failures.list()} == {"fixed"}
    assert st["branch"] not in git(repo, "branch")
    assert ref.state(aid)["stage"] == "approved"


def test_a_rejected_fix_is_dropped_and_its_reason_reaches_the_next_fixer(cfg, repo):
    ref, fid, turns = make_refiner(cfg, repo)
    aid = ref.start(fid)
    ref.run(aid)
    ref.reject(aid, "hard-codes percentages instead of fixing the cause")
    assert ref.state(aid)["stage"] == "rejected" and (repo / "harness.txt").read_text(encoding="utf-8") == "broken"
    assert ref.rejections()[0]["reason"].startswith("hard-codes")
    ref._llm = ScriptedLLM([json.dumps({"checker": {"type": "number", "value": 102}, "good_answer": "It is 102."}),
                            json.dumps([variant(f"What is {p}% of 400?", p * 4) for p in range(1, 7)])])
    ref.run(ref.start(fid))
    assert "hard-codes percentages" in turns[-1]


def test_a_fix_that_changes_nothing_useful_gets_another_try_then_is_dropped(cfg, repo):
    ref, fid, turns = make_refiner(cfg, repo, fix_text="still broken")
    st = ref.run(ref.start(fid))
    assert st["stage"] == "dropped" and "no change improved it in 2 tries" in st["reason"]
    assert len(turns) == 2 and "try 1 changed harness.txt" in turns[1] and "no gain" in turns[1]
    assert not Path(st["tree"]).exists()


def test_a_failure_that_does_not_reproduce_is_dropped_before_any_fix(cfg, repo):
    ref, fid, turns = make_refiner(cfg, repo, passes=lambda root, case, model: True)
    st = ref.run(ref.start(fid))
    assert st["stage"] == "dropped" and st["reason"].startswith("doesn't reproduce") and turns == []


def test_too_few_good_test_cases_drops_the_attempt(cfg, repo):
    ref, fid, _ = make_refiner(cfg, repo, n_variants=2)
    st = ref.run(ref.start(fid))
    assert st["stage"] == "dropped" and "only 3 test cases" in st["reason"]


def test_a_night_cut_short_carries_on(cfg, repo):
    ref, fid, _ = make_refiner(cfg, repo)
    aid = ref.start(fid)
    n = {"runs": 0}
    real = ref.execute_fn

    def count(*a):
        n["runs"] += 1
        return real(*a)
    ref.execute_fn = count
    st = ref.run(aid, deadline=time.time() - 1)
    assert st["stage"] == "new" and n["runs"] == 0
    done = ref.nightly(time.time() + 600)
    assert [s["stage"] for s in done] == ["proposed"]


def test_the_writer_keeps_the_settings_the_harness_runs_its_model_with(cfg, repo, monkeypatch):
    from types import SimpleNamespace
    ref, _, _ = make_refiner(cfg, repo)
    fixer = {"provider": "lmstudio", "model": "q@q6", "think": "unlimited", "temperature": 0.3}
    monkeypatch.setattr("jevh.personas.load_personas", lambda d: {ref.s["fixer_persona"]: SimpleNamespace(model=fixer)})
    ref.cfg.models = {k: v for k, v in ref.cfg.models.items() if k != "strong"}
    want = {"provider": "lmstudio", "model": "q@q6", "temperature": 0.3, "max_tokens": 65536}   # a budget, not unlimited
    ref.s["model"] = None
    assert ref.model_block() == want                                   # the fixer persona's block
    ref.s["model"] = "lmstudio:q@q6"
    assert ref.model_block() == want                                   # named: the harness's block for it
    ref.s["model"], ref.s["max_tokens"] = "lmstudio:m1", 16000
    assert ref.model_block() == {"provider": "lmstudio", "model": "m1", "max_tokens": 16000}


def test_gate_bounces_are_not_picked(cfg, repo):
    ref, _, _ = make_refiner(cfg, repo)
    ref.failures.record(source="gate", kind="gate_bounce", summary="final() sent back", persona="daily_assistant",
                        request="What is 2+2?", answer="4")
    assert {g["kind"] for g in ref.groups()} == {"gave_up"}


def test_the_refine_model_must_be_local(cfg, repo):
    ref, _, _ = make_refiner(cfg, repo)
    ref.cfg.providers = {**ref.cfg.providers, "anthropic": {"type": "anthropic"}}
    ref.s["model"] = "anthropic:claude-opus-5"
    with pytest.raises(RuntimeError, match="local models only"):
        ref.model()
