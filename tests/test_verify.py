"""The final() gate: files, pages, project tests and the answer check."""
import json
from pathlib import Path

import pytest

from jevh.jev import Jev
from jevh.orchestrator import Harness
from jevh.verify import Verifier, answer_rules, detect_test_command

from conftest import FakeSystemOne, ScriptedLLM, choice_ans, noul_ans, py


def answer_scores(claims=0.05, delivers=0.95, honest=0.1, accurate=0.95):
    def rules(k, q, s):
        if k == "persona":
            return choice_ans("daily_assistant", 0.9, q["criteria"])
        if k == "claims":
            return noul_ans(claims)
        if k == "delivers":
            return noul_ans(delivers)
        if k == "honest":
            return noul_ans(honest)
        if k == "accurate":
            return noul_ans(accurate)
    return rules


def gate_on(cfg, render=False):
    cfg.raw = {**cfg.raw, "verify": {"enabled": True, "max_bounces": 2, "render_html": render}}
    return cfg


def verifier(cfg, **scores):
    return Verifier(Jev(gate_on(cfg), FakeSystemOne(answer_scores(**scores))), cfg)


def test_files_are_checked(cfg, tmp_path):
    v = verifier(cfg)
    (tmp_path / "page.html").write_text("<html><body><h1>[Your Company]</h1><p>Lorem ipsum dolor</p>"
                                        "<img src='hero.png'><input placeholder='Email'></body></html>", encoding="utf-8")
    (tmp_path / "bad.py").write_text("def f(:\n  pass\n", encoding="utf-8")
    (tmp_path / "data.json").write_text("{broken", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    (tmp_path / "fake.png").write_text("not a png", encoding="utf-8")
    arts = [str(tmp_path / n) for n in ("page.html", "bad.py", "data.json", "empty.txt", "fake.png", "missing.md")]
    problems = " | ".join(v.check("build it", "Built it.", arts, "CODE:\nsave_artifact(...)").problems)
    for expected in ("[Your Company]", "Lorem ipsum", "hero.png", "syntax error on line 1", "isn't valid JSON",
                     "empty.txt is empty", "fake.png isn't a valid png", "missing.md is listed"):
        assert expected in problems, expected
    assert "Email" not in problems  # a placeholder= attribute on an input is fine


def test_clean_files_and_a_good_answer_pass(cfg, tmp_path):
    (tmp_path / "app.py").write_text("print('hi')\n", encoding="utf-8")
    v = verifier(cfg).check("write a script", "Here's app.py; it prints hi.", [str(tmp_path / "app.py")],
                            "CODE:\nsave_artifact('app.py', ...)")
    assert v.ok, v.problems


def test_answer_rules():
    t = {"verify_claims": 0.8, "verify_delivers": 0.8, "verify_honest": 0.8, "verify_accurate": 0.4}.__getitem__
    ok = {"claims": 0.05, "delivers": 0.95, "honest": 0.1}
    assert answer_rules("fix it", "Fixed; tests pass.", [], "", {**ok, "claims": 0.9}, t)
    assert answer_rules("build it", "Here's my plan...", [], "", {**ok, "delivers": 0.1}, t)
    assert not answer_rules("build it", "Couldn't: ComfyUI is down.", [], "", {**ok, "delivers": 0.1, "honest": 0.95}, t)
    assert answer_rules("Make three logo variations", "Here they are.", ["a.png"], "", ok, t)
    assert not answer_rules("Make three logo variations", "Here they are.", ["a.png", "b.png", "c.png"], "", ok, t)
    assert answer_rules("check the page", "I checked it in the browser: it renders.", [], "CODE:\nprint(1)", ok, t)
    assert not answer_rules("check the page", "I checked it in the browser.", [], "browser.goto('file:///x')", ok, t)
    assert answer_rules("build the site", "Header done. Next I'll add the footer.", [], "", ok, t)
    assert answer_rules("total the invoice", "It's $1,012.", [], "OUTPUT: 912.0", {**ok, "accurate": 0.01}, t)
    assert not answer_rules("sign in", "Sign-in failed: check the password?", [], "error", {**ok, "accurate": 0.1,
                                                                                         "honest": 0.97}, t)
    assert not answer_rules("build the site", "Header done. Want me to add the footer?", [], "", ok, t)


def test_project_tests_gate(cfg, tmp_path):
    proj = tmp_path / "proj"
    (proj / "tests").mkdir(parents=True)
    (proj / "tests" / "test_x.py").write_text("def test_x():\n    assert 1 + 1 == 3\n", encoding="utf-8")
    assert detect_test_command(proj) == "python -m pytest -q -x"
    v = verifier(cfg).check("fix the tests", "Fixed.", [], "CODE:\nproject.edit(...)", str(proj), ["tests/test_x.py"])
    assert not v.ok and any("tests fail after your changes" in p for p in v.problems)
    (proj / "tests" / "test_x.py").write_text("def test_x():\n    assert 1 + 1 == 2\n", encoding="utf-8")
    assert verifier(cfg).check("fix", "Fixed.", [], "log", str(proj), ["tests/test_x.py"]).ok


def test_render_catches_script_errors_and_blank_pages(cfg, tmp_path):
    pytest.importorskip("playwright.sync_api")
    (tmp_path / "broken.html").write_text("<html><body><h1>Tracker</h1><script>undefinedFn()</script></body></html>",
                                          encoding="utf-8")
    (tmp_path / "blank.html").write_text("<html><body><div></div></body></html>", encoding="utf-8")
    v = Verifier(Jev(gate_on(cfg, render=True), FakeSystemOne(answer_scores())), cfg)
    probs = " | ".join(v.check("build", "Built.", [str(tmp_path / "broken.html"), str(tmp_path / "blank.html")],
                               "log").problems)
    if "throws errors" not in probs and "skipped" in str(v.checks):
        pytest.skip("no chromium here")
    assert "broken.html throws errors" in probs and "undefinedFn" in probs and "blank.html renders as a blank" in probs


def test_worker_is_sent_back_then_fixes_it(cfg):
    events = []
    llm = ScriptedLLM([py("p = save_artifact('page.html', '<h1>[Your Company]</h1>')\nfinal('Built it.', [p])"),
                       py("p = save_artifact('page.html', '<h1>Kettle and Co</h1><p>Warm mugs.</p>')\n"
                          "final('Built it.', [p])")])
    h = Harness(gate_on(cfg), jev=Jev(cfg, FakeSystemOne(answer_scores())), llm_factory=lambda p: llm,
                on_event=lambda k, d: events.append((k, d)))
    h.learning = False
    r = h.turn("build a landing page")
    assert r.answer == "Built it." and Path(r.artifacts[0]).read_text(encoding="utf-8").startswith("<h1>Kettle")
    verdicts = [d for k, d in events if k == "verify"]
    assert [v["ok"] for v in verdicts] == [False, True] and verdicts[0]["bounced"]
    assert "[Your Company]" in llm.seen[1][-1]["content"] and "Not accepted yet" in llm.seen[1][-1]["content"]


def test_unfixed_problems_are_stated_not_hidden(cfg):
    bad = py("p = save_artifact('page.html', '<h1>Lorem ipsum</h1>')\nfinal('Built it.', [p])")
    llm = ScriptedLLM([bad, bad, bad])
    h = Harness(gate_on(cfg), jev=Jev(cfg, FakeSystemOne(answer_scores())), llm_factory=lambda p: llm)
    h.learning = False
    r = h.turn("build a landing page")
    assert r.answer.startswith("Built it.") and "Not verified" in r.answer and "Lorem ipsum" in r.answer
    assert len(llm.seen) == 3  # two bounces, then accepted with the warning


def test_hallucinated_plain_answer_is_sent_back(cfg):
    llm = ScriptedLLM(["I searched and the population is 38 million.",
                       py("print('Canada: 41,828,000 (Statistics Canada, July 2026)')\nfinal('About 41.8 million.')")])
    rules = answer_scores()

    def seq(k, q, s):  # first answer claims a search the log doesn't show; the second is fine
        if k == "claims":
            return noul_ans(0.9 if "38 million" in str(s) else 0.05)
        return rules(k, q, s)
    h = Harness(gate_on(cfg), jev=Jev(cfg, FakeSystemOne(seq)), llm_factory=lambda p: llm)
    h.learning = False
    assert h.turn("what's the population of Canada?").answer == "About 41.8 million."


def test_bare_artifact_names_resolve_before_the_gate(cfg):
    events = []
    llm = ScriptedLLM([py("save_artifact('page.html', '<h1>Rise and Crumb</h1><p>Fresh bread daily.</p>')\n"
                          "open('notes.md', 'w').write('# Notes')\n"
                          "final('Built it.', ['page.html', 'notes.md'])")])
    h = Harness(gate_on(cfg), jev=Jev(cfg, FakeSystemOne(answer_scores())), llm_factory=lambda p: llm,
                on_event=lambda k, d: events.append((k, d)))
    h.learning = False
    r = h.turn("build a page")
    assert [d["ok"] for k, d in events if k == "verify"] == [True]
    assert all(Path(a).is_absolute() and Path(a).exists() for a in r.artifacts)
    assert {Path(a).name for a in r.artifacts} == {"page.html", "notes.md"}


def test_made_up_number_is_sent_back_and_a_bounced_fast_answer_escalates(cfg):
    """Live 2026-09-28 every turn ran on the fast tier. Now a fast-tier answer the gate sends
    back is retried on the persona's strong model; the number rule catches a fare no output showed."""
    fast = ScriptedLLM([py("print('United $1,317  Delta $1,284')"), "The cheapest is United at $1,099."])
    strong = ScriptedLLM(["The cheapest is Delta at $1,284."])
    base = answer_scores()

    def rules(k, q, s):
        if k == "tier":
            return {"choice": "fast", "confidence": 0.95, "probabilities": {"strong": 0.05, "fast": 0.95}}
        return base(k, q, s)
    events = []
    h = Harness(gate_on(cfg), jev=Jev(cfg, FakeSystemOne(rules)),
                llm_factory=lambda p: fast if p.model["model"] == cfg.models["fast"]["model"] else strong,
                on_event=lambda k, d: events.append((k, d)))
    h.learning = False
    r = h.turn("cheapest flight SLC to Tokyo?", force="daily_assistant")
    assert r.answer == "The cheapest is Delta at $1,284."
    bounced = [d for k, d in events if k == "verify" and not d["ok"]]
    assert bounced and "1,099" in bounced[0]["problems"][0]
    assert [k for k, _ in events].count("escalate") == 1


def test_grounding_tolerates_rounding_but_not_new_numbers():
    from jevh.verify import grounding
    assert grounding("About 41.8 million", "estimated at 41,828,000") == ({"41.8"}, set())
    assert grounding("It costs $1,999 in 2026", "Regular price $1,999.00") == ({"1,999"}, set())
    assert grounding("United at $1,099", "United $1,317") == (set(), {"1,099"})
    # a printed dict shows a page's line breaks as "\n": "\n110 points" is still 110 (live, Hacker News)
    assert grounding("110 points, 57 comments", r"{'page': '1.\nStory\n110 points by x | 57 comments'}") == \
        ({"110", "57"}, set())


def test_follow_up_to_an_escalated_turn_starts_strong(cfg):
    from jevh.bus import Task
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(answer_scores())), llm_factory=lambda p: ScriptedLLM(["ok"]))
    p = h.personas["daily_assistant"]
    task = Task(id=1, parent=None, requester="user", to=p.name, text="where did you get that?")
    task.history = [("price?", "$2,999.99", "")]
    h._escalated = True
    assert h._pick_tier(task, p).tier == "strong"


# -- the cheap first pass: Jev's risk check, then a full review of risky changes -----------------
def review_setup(cfg, tmp_path, risky_files):
    """A project with auth.py and README.md changed by a worker (originals in `backups`)."""
    base = answer_scores()

    def rules(k, q, s):
        if k == "risky":
            return noul_ans(0.95 if s["file"] in risky_files else 0.05)
        return base(k, q, s)
    proj, backups = tmp_path / "proj", tmp_path / "work" / ".project_backups"
    for root, auth, readme in ((backups, "def login(u, pw):\n    return check(u, pw)\n", "# App\n"),
                               (proj, "def login(u, pw):\n    return True\n", "# App\n\nRun it.\n")):
        root.mkdir(parents=True, exist_ok=True)
        (root / "auth.py").write_text(auth, encoding="utf-8")
        (root / "README.md").write_text(readme, encoding="utf-8")
    return Verifier(Jev(gate_on(cfg), FakeSystemOne(rules)), cfg), proj, backups


def test_safe_changes_are_not_reviewed(cfg, tmp_path):
    v, proj, backups = review_setup(cfg, tmp_path, risky_files=set())
    reviews = []
    verdict = v.check("tidy up", "done", [], "ran code", str(proj), ["auth.py", "README.md"],
                      backups=backups, review=lambda p, tier: reviews.append(p) or "NO PROBLEMS")
    assert verdict.ok and not reviews and verdict.checks["review"].startswith("skipped: no risky changes")


def test_a_risky_change_is_reviewed_and_problems_go_back(cfg, tmp_path):
    v, proj, backups = review_setup(cfg, tmp_path, risky_files={"auth.py"})
    reviews = []
    found = "auth.py: login() now returns True for any password - restore the check."
    verdict = v.check("speed up login", "done", [], "ran code", str(proj), ["auth.py", "README.md"],
                      backups=backups, review=lambda p, tier: reviews.append(p) or found)
    assert not verdict.ok and found in verdict.problems[0] and "auth.py" in verdict.checks["review"]
    prompt = reviews[0]
    assert "-    return check(u, pw)" in prompt and "+    return True" in prompt   # the real diff
    assert "=== auth.py (flagged risky)" in prompt and "=== README.md\n" in prompt
    # the same diff again (the worker disputes the finding): not reviewed twice
    again = v.check("speed up login", "done", [], "ran code", str(proj), ["auth.py", "README.md"],
                    backups=backups, review=lambda p, tier: reviews.append(p) or found)
    assert again.ok and len(reviews) == 1


def test_a_clean_review_passes(cfg, tmp_path):
    v, proj, backups = review_setup(cfg, tmp_path, risky_files={"auth.py"})
    verdict = v.check("speed up login", "done", [], "ran code", str(proj), ["auth.py"],
                      backups=backups, review=lambda p, tier: "NO PROBLEMS")
    assert verdict.ok and verdict.checks["review"] == "pass: reviewed auth.py (fast model)"


def test_a_new_file_is_reviewed_as_all_added_lines(cfg, tmp_path):
    v, proj, backups = review_setup(cfg, tmp_path, risky_files={"jobs.py"})
    (proj / "jobs.py").write_text("db.execute('DELETE FROM orders')\n", encoding="utf-8")
    reviews = []
    v.check("cleanup job", "done", [], "ran code", str(proj), ["jobs.py"], backups=backups,
            review=lambda p, tier: reviews.append(p) or "NO PROBLEMS")
    assert "+db.execute('DELETE FROM orders')" in reviews[0]


def test_first_pass_end_to_end_through_the_harness(cfg, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "auth.py").write_text("def login(u, pw):\n    return check(u, pw)\n", encoding="utf-8")
    base = answer_scores()

    def rules(k, q, s):
        if k == "persona":
            return choice_ans("high_end_dev", 0.9, q["criteria"])
        if k == "risky":
            return noul_ans(0.95)
        return base(k, q, s)
    llm = ScriptedLLM([py("print(project.edit('auth.py', 'return check(u, pw)', 'return True'))"),
                       py("final('login is faster now')"),
                       "auth.py: login() accepts any password now - restore check(u, pw).",   # the review
                       py("print(project.undo('auth.py')); final('reverted: that change skipped the password check')")])
    h = Harness(gate_on(cfg), jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: llm)
    h.learning = False
    h.set_project(str(proj))
    r = h.turn("speed up login", force="high_end_dev")
    review = next(m for m in llm.seen if "reviewing a code change" in m[-1]["content"])[-1]["content"]
    assert "-    return check(u, pw)" in review and "+    return True" in review
    bounced = llm.seen[-1][-1]["content"]
    assert "A review of your changes to auth.py" in bounced and "accepts any password" in bounced
    assert r.answer.startswith("reverted") and "check(u, pw)" in (proj / "auth.py").read_text(encoding="utf-8")


def test_the_reviewer_is_picked_by_how_severe_a_mistake_could_be(cfg, tmp_path):
    def run(files, severe=0.05, forced=None):
        v, proj, backups = review_setup(cfg, tmp_path, risky_files=set(files))
        if forced:
            v.settings["review_tier"] = forced
        base = v.jev.client.rules
        v.jev.client.rules = lambda k, q, s: noul_ans(severe) if k == "severe" else base(k, q, s)
        for rel, text in files.items():
            (proj / rel).write_text(text, encoding="utf-8")
        tiers = []
        v.check("change it", "done", [], "ran code", str(proj), list(files), backups=backups,
                review=lambda p, tier: tiers.append(tier) or "NO PROBLEMS")
        return tiers[0]
    assert run({"api.py": "def get_user(user_id):\n"}) == "fast"                  # could only break callers
    assert run({"api.py": "def get_user(user_id):\n"}, severe=0.9) == "strong"    # Jev: could be severe
    assert run({"jobs.py": "db.execute('DELETE FROM orders')\n"}) == "strong"      # plain code: deletes data
    assert run({"pay.py": "stripe.Refund.create(amount=amount)\n"}) == "strong"    # plain code: money
    assert run({"jobs.py": "db.execute('DELETE FROM orders')\n"}, forced="fast") == "fast"
    assert run({"api.py": "def get_user(user_id):\n"}, forced="strong") == "strong"


def test_a_json_grid_is_not_read_as_one_big_number():
    """Live: an ARC answer's grid row '3,3,0,0,0,...' was flagged as a made-up number."""
    from jevh.verify import grounding
    answer = "[[0, 0, 3, 3, 0, 0], [3,3,0,0,0,0,0,0,0,3,3,0,0,0,0,0]]"
    assert grounding(answer, "ran predict(test_in)") == (set(), set())
    assert grounding("It costs $1,999.", "price: $1,999")[0] == {"1,999"}      # thousands still one number
    assert grounding("It costs $1,999.", "price: $2,499")[1] == {"1,999"}


def test_a_json_data_answer_is_not_judged_as_prose(cfg):
    """Live (Gemma 4, ARC): a correct grid was bounced twice by the 'accurate' question; data makes no claims."""
    v = verifier(cfg, accurate=0.05, claims=0.95)     # Jev would call it inaccurate and making claims
    grid = json.dumps([[0, 2, 0], [3, 3, 0]])
    verdict = v.check("solve the puzzle", grid, [], "CODE: print(json.dumps(pred))\nOUTPUT: " + grid)
    assert verdict.ok and verdict.checks["answer"] == "skipped: the answer is JSON data"
    prose = v.check("solve the puzzle", "Done - all tests pass.", [], "CODE: print(1)\nOUTPUT: 1")
    assert not prose.ok                                 # prose is still checked
