from jevh.jev import Jev
from jevh.personas import load_personas
from jevh.skills import load_roster

from conftest import FakeSystemOne, choice_ans, noul_ans


def personas(cfg):
    return load_personas(cfg.personas_dir)


def test_route_confident_choice(cfg):
    ps = personas(cfg)
    fake = FakeSystemOne(lambda k, q, s: choice_ans("high_end_dev", 0.8, q["criteria"]) if k == "persona" else None)
    d = Jev(cfg, fake).route_persona("fix my python bug", ps)
    assert d.persona == "high_end_dev" and d.reason == "choice"


def test_route_low_confidence_sticks_when_conversation_continues(cfg):
    ps = personas(cfg)

    def rules(k, q, s):
        if k == "persona":
            return choice_ans("daily_assistant", 0.3, q["criteria"])
        if k == "continues":
            return noul_ans(0.9)

    d = Jev(cfg, FakeSystemOne(rules)).route_persona("and on fridays?", ps, current="fitness_coach",
                                                     recent="[fitness_coach] User: plan my week")
    assert d.persona == "fitness_coach"


def test_route_low_confidence_falls_back_to_default(cfg):
    ps = personas(cfg)
    fake = FakeSystemOne(lambda k, q, s: choice_ans("comfyui_manager", 0.3, q["criteria"]) if k == "persona" else None)
    assert Jev(cfg, fake).route_persona("hmm", ps).persona == cfg.default_persona


def skill_rules(choice_pick, choice_conf, none_p, fits_by_name):
    """Fake answers for the skill Choice (key 'pick') and fits nouls (keys f0..)."""
    def rules(k, q, s):
        if k == "pick":
            opts = list(q["criteria"])
            rest = (1 - choice_conf - none_p) / max(len(opts) - 2, 1)
            probs = {o: (choice_conf if o == choice_pick else none_p if o == "none_of_these" else rest)
                     for o in opts}
            return {"choice": max(probs, key=probs.get), "probabilities": probs, "confidence": max(probs.values())}
        if k.startswith("f"):
            name = q["instructions"]["skill"].split(":")[0]
            return noul_ans(fits_by_name.get(name, 0.01))
    return rules


def test_skills_scoped_to_persona(cfg):
    ps, roster = personas(cfg), load_roster(cfg.skills_dirs)
    fake = FakeSystemOne(skill_rules("workout-programming", 0.6, 0.1, {"workout-programming": 0.5}))
    d = Jev(cfg, fake).pick_skills("plan my lifts", ps["fitness_coach"], roster)
    assert d.skills == ["workout-programming"]
    offered = set(fake.calls[0][1]["pick"]["criteria"])
    assert "frontend-design" not in offered and "none_of_these" in offered


def test_skills_none_veto(cfg):
    ps, roster = personas(cfg), load_roster(cfg.skills_dirs)
    fake = FakeSystemOne(skill_rules("workout-programming", 0.02, 0.95, {"workout-programming": 0.99}))
    d = Jev(cfg, fake).pick_skills("what's 2+2", ps["fitness_coach"], roster)
    assert d.skills == [] and len(fake.calls) == 1  # vetoed before the fits call


def test_skills_confident_choice_overrides_low_fits(cfg):
    ps, roster = personas(cfg), load_roster(cfg.skills_dirs)
    fake = FakeSystemOne(skill_rules("workout-programming", 0.95, 0.02, {"workout-programming": 0.02}))
    assert Jev(cfg, fake).pick_skills("x", ps["fitness_coach"], roster).skills == ["workout-programming"]


def test_skills_low_fits_and_unsure_choice_loads_nothing(cfg):
    ps, roster = personas(cfg), load_roster(cfg.skills_dirs)
    fake = FakeSystemOne(skill_rules("workout-programming", 0.5, 0.3, {"workout-programming": 0.03}))
    assert Jev(cfg, fake).pick_skills("x", ps["fitness_coach"], roster).skills == []


def test_big_choice_runs_in_rounds_under_the_cap(cfg):
    fake = FakeSystemOne()
    opts = {f"s{i}": f"skill {i}" for i in range(40)}
    probs = Jev(cfg, fake).choice_probs("t", "state", "which?", opts, {"none_of_these": "none"})
    sizes = [len(q["pick"]["criteria"]) for _, q in fake.calls]
    assert max(sizes) <= 16 and all(n >= 2 for n in sizes) and len(fake.calls) >= 4
    assert "none_of_these" in fake.calls[-1][1]["pick"]["criteria"]  # none joins the final only
    assert set(probs) <= set(opts) | {"none_of_these"}


def test_memory_gate_keeps_only_relevant(cfg):
    chunks = ["[fitness_coach] squat 100kg", "[high_end_dev] fastapi route", "[fitness_coach] bench 80kg"]
    fake = FakeSystemOne(lambda k, q, s: noul_ans(0.9 if "fastapi" in q["instructions"]["snippet"] else 0.1))
    d = Jev(cfg, fake).gate_context("refactor my fastapi route", chunks)
    assert d.kept == ["[high_end_dev] fastapi route"]


def test_collab_needs_complexity(cfg):
    ps = personas(cfg)

    def rules(complexity):
        def r(k, q, s):
            if k == "complexity":
                return {"score": complexity, "confidence": 0.9}
            if k in ("needs_high_end_dev", "needs_front_end_designer"):
                return noul_ans(0.9)
        return r

    assert Jev(cfg, FakeSystemOne(rules(0.1))).collab("x", ps["fitness_coach"], ps).suggested == []
    d = Jev(cfg, FakeSystemOne(rules(1.8))).collab("build an app", ps["fitness_coach"], ps)
    assert set(d.suggested) == {"high_end_dev", "front_end_designer"}


def test_approve_request(cfg):
    ps = personas(cfg)
    fake = FakeSystemOne(lambda k, q, s: noul_ans(0.9 if k == "in_scope" else 0.2))
    a = Jev(cfg, fake).approve_request(ps["fitness_coach"], ps["high_end_dev"], "build it", "goal")
    assert not a.approved and "necessary" in a.reason


def test_memory_long_history_ranks_then_checks_leaders(cfg):
    chunks = [f"[daily_assistant] chat about topic {i}" for i in range(10)] + ["[high_end_dev] fastapi route"]

    def rules(k, q, s):
        if k == "pick":
            opts = list(q["criteria"])
            probs = {o: (0.9 if "fastapi" in str(q["criteria"][o]) else 0.1 / len(opts)) for o in opts}
            return {"choice": max(probs, key=probs.get), "probabilities": probs, "confidence": 0.9}
        if k.startswith("m"):
            return noul_ans(0.8 if "fastapi" in q["instructions"]["snippet"] else 0.05)

    fake = FakeSystemOne(rules)
    d = Jev(cfg, fake).gate_context("refactor my fastapi route", chunks)
    assert d.kept == ["[high_end_dev] fastapi route"]
    last_questions = fake.calls[-1][1]
    assert len(last_questions) == 4  # only the Choice leaders get the per-snippet check
