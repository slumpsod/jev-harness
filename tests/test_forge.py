"""Persona creation (jevh/forge.py): the no-fit and orphaned-skill triggers, the routing gate, the fences
a draft lives in, and review (approve / delete)."""
import json
from pathlib import Path

import yaml

from jevh.decide import SystemOneError
from jevh.forge import Forge, set_owner
from jevh.jev import ROUTE_Q, SKILL_HOME_Q, Jev
from jevh.memory import Memory
from jevh.orchestrator import Harness
from jevh.personas import Persona, load_learned, load_personas
from jevh.skills import Skill, load_roster, parse_skill

from conftest import FakeSystemOne, ScriptedLLM, choice_ans, noul_ans

SPEC = {"name": "Tax Helper", "description": "Handles personal and small-business tax questions: deductions, "
        "filing deadlines, quarterly estimates.", "system_prompt": "You are a careful tax helper.\nShow your "
        "working and say what you are unsure of.\nYou do not give legal advice."}


def spec_reply(**over) -> str:
    return "Here you go:\n" + json.dumps({**SPEC, **over})


def pick_all(options, name, conf=0.9):
    return choice_ans(name, conf, options)


class Rules:
    """Fake Jev answers for everything the forge asks. Knobs are attributes, so a test can change them."""

    def __init__(self):
        self.route_pick, self.route_conf = "daily_assistant", 0.3      # a low-confidence route...
        self.no_fit = 0.95                                              # ...and Jev says nothing fits
        self.similar = 0.9                                              # unhandled requests that suit one specialist
        self.twin = False                                               # the new description matches an existing persona
        self.home = None                                                # skill_home's pick: None = "none of these"
        self.probe_to = lambda text: "tax_helper" if "tax" in text.lower() else "daily_assistant"

    def __call__(self, k, q, s):
        instr = q.get("instructions")
        if k == "persona":
            return choice_ans(self.route_pick, self.route_conf, q["criteria"])
        if k == "no_fit":
            return noul_ans(self.no_fit)
        if isinstance(instr, dict) and "request_a" in instr:             # same specialist
            if instr["request_a"] == SPEC["description"]:                # the twin check on a new description
                return noul_ans(0.99 if self.twin else 0.0)
            return noul_ans(self.similar)                                # two unhandled requests
        if k == "pick" and instr == SKILL_HOME_Q:
            opts = q["criteria"]
            name = self.home or "none_of_these"
            return choice_ans(name, 0.9, opts) if name in opts else None
        if k == "pick" and instr == ROUTE_Q:
            want = self.probe_to(s["latest_request"])
            return choice_ans(want if want in q["criteria"] else "daily_assistant", 0.9, q["criteria"])
        if k in ("fact", "skill"):
            return noul_ans(0.0)


def make_forge(cfg, tmp_path, rules=None, llm_replies=None):
    skills = tmp_path / "skills"
    (skills / "learned").mkdir(parents=True)
    cfg.skills_dirs = [skills]
    rules = rules or Rules()
    jev = Jev(cfg, FakeSystemOne(rules))
    memory = Memory(cfg.workspace / "memory")
    events = []
    llm = ScriptedLLM(llm_replies if llm_replies is not None else [spec_reply()])
    forge = Forge(cfg, jev, memory, skills, lambda l, prompt: l.chat([{"role": "user", "content": prompt}]).text,
                  lambda kind, data: events.append((kind, data)))
    return forge, llm, events, rules, memory


def learned_skill(skills_home, name="learned-quarterly-tax", desc="Estimate quarterly tax payments for a sole trader."):
    d = skills_home / "learned" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndraft: true\ndescription: {desc}\n---\n# Quarterly tax\n1. Add income.\n",
                                encoding="utf-8")
    return parse_skill(d / "SKILL.md")


# -- loading -----------------------------------------------------------------------------

def test_created_personas_load_beside_hand_written_ones(cfg):
    learned = cfg.personas_dir / "learned"
    learned.mkdir()
    base = {"description": "Does taxes for people.", "system_prompt": "You help.", "model": {"provider": "x"}}
    (learned / "tax_helper.yaml").write_text(yaml.safe_dump({"name": "tax_helper", **base, "draft": True}))
    (learned / "daily_assistant.yaml").write_text(yaml.safe_dump({"name": "daily_assistant", **base}))  # clash
    (learned / "off.yaml").write_text(yaml.safe_dump({"name": "off", **base, "draft": True, "active": False}))
    (learned / "broken.yaml").write_text("name: [unterminated")
    (learned / "weird.yaml").write_text(yaml.safe_dump({"name": "weird", "nonsense_field": 1}))
    ps = load_personas(cfg.personas_dir)
    assert ps["tax_helper"].draft and "off" not in ps and "broken" not in ps and "weird" not in ps
    assert ps["daily_assistant"].description != "Does taxes for people."   # hand-written wins a name clash
    assert "off" in load_personas(cfg.personas_dir, include_inactive=True)
    assert {p.name for p in load_learned(cfg.personas_dir)} == {"tax_helper", "daily_assistant", "off"}


def test_created_personas_go_to_the_first_personas_folder(cfg, tmp_path):
    """With a home folder (personas_dirs = [home/personas, built-in]) created personas are written beside
    your own, and load from every folder's learned/ without replacing a hand-written persona."""
    home = tmp_path / "home" / "personas"
    home.mkdir(parents=True)
    cfg.personas_dirs = [home, cfg.personas_dir]
    forge = Forge(cfg, None, None, tmp_path / "skills", None)
    assert forge.learned_dir == home / "learned"
    base = {"description": "Does taxes for people.", "system_prompt": "You help.", "model": {"provider": "x"}}
    path = forge._write(Persona(name="tax_helper", **base, draft=True))
    assert path.parent == home / "learned" and not (cfg.personas_dir / "learned").exists()
    (cfg.personas_dir / "learned").mkdir()
    (cfg.personas_dir / "learned" / "old.yaml").write_text(yaml.safe_dump({"name": "old", **base, "draft": True}))
    ps = load_personas(cfg.personas_dirs)
    assert "tax_helper" in ps and "old" in ps and "daily_assistant" in ps
    assert {p.name for p in load_learned(cfg.personas_dirs)} == {"tax_helper", "old"}


def test_an_owned_skill_is_offered_only_to_its_owner(tmp_path):
    f = tmp_path / "learned-x" / "SKILL.md"
    f.parent.mkdir()
    f.write_text("---\nname: learned-x\ndescription: Does x.\n---\nbody\n", encoding="utf-8")
    anyone = Persona("a", "d", "p", {}, skills_allowed=["learned-*"])
    owner = Persona("tax_helper", "d", "p", {}, skills_allowed=[])
    assert anyone.offered_skill(parse_skill(f)) and not owner.offered_skill(parse_skill(f))
    set_owner(f, "tax_helper")
    s = parse_skill(f)
    assert s.owner == "tax_helper" and owner.offered_skill(s) and not anyone.offered_skill(s)
    set_owner(f, None)
    assert parse_skill(f).owner == "" and parse_skill(f).body == "body"


# -- routing: the no-fit check -----------------------------------------------------------

def test_no_fit_is_asked_only_when_routing_is_unsure(cfg):
    ps = load_personas(cfg.personas_dir)
    rules = Rules()
    fake = FakeSystemOne(rules)
    d = Jev(cfg, fake).route_persona("what's the capital gains rate?", ps)
    assert d.persona == cfg.default_persona and d.no_fit == 0.95 and d.reason.startswith("no assistant fits")
    rules.route_conf = 0.9                      # a confident route adds no call
    fake.calls.clear()
    d = Jev(cfg, fake).route_persona("fix my python bug", ps)
    assert d.no_fit is None and len(fake.calls) == 1


def test_no_fit_off_and_a_jev_failure_do_not_break_routing(cfg):
    ps = load_personas(cfg.personas_dir)
    cfg.raw = {**cfg.raw, "persona_creation": {"enabled": False}}
    fake = FakeSystemOne(Rules())
    assert Jev(cfg, fake).route_persona("x", ps).no_fit is None and len(fake.calls) == 1
    cfg.raw = {**cfg.raw, "persona_creation": {"enabled": True}}
    jev = Jev(cfg, fake)
    real = jev._ask

    def flaky(step, state, questions):
        if step == "route_fit":
            raise SystemOneError("down")
        return real(step, state, questions)
    jev._ask = flaky
    d = jev.route_persona("x", ps)
    assert d.persona == cfg.default_persona and d.no_fit is None


def test_skill_home_reports_none_of_them(cfg):
    ps = load_personas(cfg.personas_dir)
    rules = Rules()
    sk = Skill("learned-q", "Estimate quarterly tax.", "body", "/x")
    best, none, probs = Jev(cfg, FakeSystemOne(rules)).skill_home(sk, ps)
    assert best is None and none == 0.9
    rules.home = "daily_assistant"
    best, none, _ = Jev(cfg, FakeSystemOne(rules)).skill_home(sk, ps)
    assert best == "daily_assistant" and none < 0.5


# -- trigger 1: unhandled requests ---------------------------------------------------------

def test_similar_unhandled_requests_gather_then_make_a_persona(cfg, tmp_path):
    forge, llm, events, rules, memory = make_forge(cfg, tmp_path)
    assert forge.note_no_fit("how do I claim my home office on tax?", llm) is None
    assert forge.note_no_fit("when are quarterly tax estimates due?", llm) is None
    assert len(forge.pending()) == 2 and llm.seen == []        # nothing drafted from two
    made = forge.note_no_fit("is my tax refund taxable?", llm)
    assert made and made.name == "tax_helper" and made.active and made.trigger == "no_fit"
    assert forge.pending() == []                               # the three left the pool
    data = yaml.safe_load((cfg.personas_dir / "learned" / "tax_helper.yaml").read_text())
    # the fences: no MCP, no collaboration, no tools, no skills, a draft
    assert data["draft"] is True and data["mcp"] == [] and data["can_request"] == [] and data["tools"] == []
    assert data["skills_allowed"] == [] and data["model"] == load_personas(cfg.personas_dir)["daily_assistant"].model
    assert "tax_helper" in load_personas(cfg.personas_dir)
    assert any(k == "persona_drafted" for k, _ in events)


def test_dissimilar_requests_never_make_a_persona(cfg, tmp_path):
    forge, llm, _, rules, _ = make_forge(cfg, tmp_path)
    rules.similar = 0.1
    for r in ("how do I claim my home office on tax?", "what's a good name for a goldfish?",
              "translate 'thank you' to Welsh", "plan a three-day trip to Lisbon"):
        assert forge.note_no_fit(r, llm) is None
    assert len(forge.pending()) == 4 and llm.seen == [] and not (cfg.personas_dir / "learned").exists()


def test_the_pool_is_capped(cfg, tmp_path):
    cfg.raw = {**cfg.raw, "persona_creation": {"max_pending": 3}}
    forge, llm, _, rules, _ = make_forge(cfg, tmp_path)
    rules.similar = 0.0
    for i in range(6):
        forge.note_no_fit(f"request {i}", llm)
    assert [p["text"] for p in forge.pending()] == ["request 3", "request 4", "request 5"]


def test_an_unusable_reply_keeps_the_requests_pooled(cfg, tmp_path):
    forge, llm, events, *_ = make_forge(cfg, tmp_path, llm_replies=["I cannot do that.", "{}"])
    for r in ("tax a", "tax b", "tax c"):
        forge.note_no_fit(r, llm)
    assert len(forge.pending()) == 3 and any(k == "persona_skipped" for k, _ in events)
    assert not (cfg.personas_dir / "learned").exists()


def test_a_near_duplicate_of_an_existing_persona_is_not_created(cfg, tmp_path):
    forge, llm, events, rules, _ = make_forge(cfg, tmp_path)
    rules.twin = True     # an existing persona "suits both" the new description: it is a twin
    for r in ("tax a", "tax b"):
        forge.note_no_fit(r, llm)
    assert forge.note_no_fit("tax c", llm) is None
    assert any(k == "persona_skipped" and "too close" in d["reason"] for k, d in events)


def test_drafts_are_capped(cfg, tmp_path):
    cfg.raw = {**cfg.raw, "persona_creation": {"max_live_drafts": 1}}
    forge, llm, events, rules, _ = make_forge(cfg, tmp_path, llm_replies=[spec_reply(), spec_reply(name="Other")])
    for r in ("tax a", "tax b", "tax c"):
        forge.note_no_fit(r, llm)
    assert len(forge.drafts()) == 1
    for r in ("tax d", "tax e", "tax f"):
        forge.note_no_fit(r, llm)
    assert len(forge.drafts()) == 1 and any("waiting for approval" in d.get("reason", "") for _, d in events)


# -- the routing gate ------------------------------------------------------------------------

def test_a_draft_that_takes_other_personas_requests_stays_inactive(cfg, tmp_path):
    forge, llm, events, rules, memory = make_forge(cfg, tmp_path)
    memory.append("fitness_coach", "what tax do I pay on gym membership?", "...", "s")   # "tax" -> would be stolen
    memory.append("daily_assistant", "plan my day", "...", "s")
    for r in ("tax a", "tax b"):
        forge.note_no_fit(r, llm)
    made = forge.note_no_fit("tax c", llm)
    assert made and not made.active and "would take 1" in made.gate.note
    assert made.gate.steals == ["what tax do I pay on gym membership?"]
    assert "tax_helper" not in load_personas(cfg.personas_dir)                  # not routable
    assert [p.name for p in forge.drafts()] == ["tax_helper"]                    # but there to review
    assert forge.activate("tax_helper") and "tax_helper" in load_personas(cfg.personas_dir)


def test_a_draft_that_does_not_win_its_own_requests_stays_inactive(cfg, tmp_path):
    forge, llm, _, rules, _ = make_forge(cfg, tmp_path)
    rules.probe_to = lambda text: "daily_assistant"
    for r in ("tax a", "tax b"):
        forge.note_no_fit(r, llm)
    made = forge.note_no_fit("tax c", llm)
    assert made and not made.active and "too few" in made.gate.note


def test_the_requests_that_made_a_draft_are_not_counted_as_stolen(cfg, tmp_path):
    forge, llm, _, rules, memory = make_forge(cfg, tmp_path)
    for r in ("tax a", "tax b", "tax c"):     # all handled earlier by the default persona
        memory.append("daily_assistant", r, "ok", "s")
    for r in ("tax a", "tax b"):
        forge.note_no_fit(r, llm)
    assert forge.note_no_fit("tax c", llm).active


# -- trigger 2: orphaned skills --------------------------------------------------------------

def test_an_orphaned_skill_makes_a_persona_that_owns_it(cfg, tmp_path):
    forge, llm, events, rules, _ = make_forge(cfg, tmp_path)
    skill = learned_skill(forge.skills_home)
    made = forge.note_skill(skill, llm)
    assert made and made.trigger == "orphan_skill" and made.skill == "learned-quarterly-tax" and made.active
    roster = load_roster([forge.skills_home])
    owned = roster["learned-quarterly-tax"]
    ps = load_personas(cfg.personas_dir)
    assert owned.owner == "tax_helper"
    assert ps["tax_helper"].offered_skill(owned)
    assert not ps["daily_assistant"].offered_skill(owned) and not ps["high_end_dev"].offered_skill(owned)


def test_a_skill_with_a_home_creates_nothing(cfg, tmp_path):
    forge, llm, _, rules, _ = make_forge(cfg, tmp_path)
    rules.home = "high_end_dev"
    assert forge.note_skill(learned_skill(forge.skills_home), llm) is None
    assert llm.seen == [] and not (cfg.personas_dir / "learned").exists()


def test_only_learned_unowned_skills_are_considered(cfg, tmp_path):
    forge, llm, _, rules, _ = make_forge(cfg, tmp_path)
    skill = learned_skill(forge.skills_home)
    set_owner(Path(skill.path), "someone")
    assert forge.note_skill(parse_skill(Path(skill.path)), llm) is None   # owned
    hand = tmp_path / "hand" / "SKILL.md"
    hand.parent.mkdir()
    hand.write_text("---\nname: hand\ndescription: A hand-written skill of mine.\n---\nx\n", encoding="utf-8")
    assert forge.note_skill(parse_skill(hand), llm) is None                                      # not ours to edit


def test_a_skill_is_not_owned_by_a_persona_that_failed_the_gate_until_it_is_switched_on(cfg, tmp_path):
    forge, llm, _, rules, _ = make_forge(cfg, tmp_path)
    rules.probe_to = lambda text: "daily_assistant"     # the draft wins nothing
    skill = learned_skill(forge.skills_home)
    made = forge.note_skill(skill, llm)
    assert made and not made.active
    assert parse_skill(Path(skill.path)).owner == ""      # still offered as before
    assert forge.approve("tax_helper")
    s = parse_skill(Path(skill.path))
    assert s.owner == "tax_helper"
    p = load_personas(cfg.personas_dir)["tax_helper"]
    assert not p.draft and p.active


def test_delete_gives_the_skill_back(cfg, tmp_path):
    forge, llm, _, rules, _ = make_forge(cfg, tmp_path)
    skill = learned_skill(forge.skills_home)
    forge.note_skill(skill, llm)
    assert forge.delete("tax_helper") and not forge.delete("tax_helper")
    assert parse_skill(Path(skill.path)).owner == ""
    assert "tax_helper" not in load_personas(cfg.personas_dir)


def test_sweep_checks_every_unowned_learned_skill(cfg, tmp_path):
    forge, llm, _, rules, _ = make_forge(cfg, tmp_path, llm_replies=[spec_reply()])
    learned_skill(forge.skills_home)
    learned_skill(forge.skills_home, "learned-other", "Something about chess openings.")
    rules.home = None
    made = forge.sweep(load_roster([forge.skills_home]), llm)
    assert [m.name for m in made] == ["tax_helper"]    # the second skill gets no persona: the model has no reply left


# -- through the harness --------------------------------------------------------------------

def test_the_harness_grows_a_persona_from_unhandled_requests_and_routes_to_it(cfg, tmp_path):
    rules = Rules()
    replies = ["Done a.", "Done b.", "Done c.", spec_reply(), "Here is your tax answer."]
    llm = ScriptedLLM(replies)
    events = []
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: llm,
                on_event=lambda k, d: events.append((k, d)))
    h.learn_sync = True
    h.learning = False                            # only persona creation runs after a turn
    for text in ("tax question a", "tax question b"):
        assert h.turn(text).persona == "daily_assistant"
    assert "tax_helper" not in h.personas
    h.turn("tax question c")
    assert "tax_helper" in h.personas and h.personas["tax_helper"].draft
    assert any(k == "persona_drafted" for k, _ in events)
    # now routing can pick it (a confident route to the new persona)
    rules.route_pick, rules.route_conf = "tax_helper", 0.9
    assert h.turn("tax question d").persona == "tax_helper"


def test_a_confident_route_never_feeds_the_pool(cfg, tmp_path):
    rules = Rules()
    rules.route_conf = 0.9
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: ScriptedLLM(["ok"]))
    h.learn_sync = True
    h.learning = False
    h.turn("anything")
    assert h.forge.pending() == []


def test_a_pinned_persona_never_feeds_the_pool(cfg, tmp_path):
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(Rules())), llm_factory=lambda p: ScriptedLLM(["ok"]))
    h.learn_sync = True
    h.learning = False
    h.turn("anything", force="daily_assistant")
    assert h.forge.pending() == []


def test_persona_creation_can_be_switched_off(cfg, tmp_path):
    cfg.raw = {**cfg.raw, "persona_creation": {"enabled": False}}
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(Rules())), llm_factory=lambda p: ScriptedLLM(["ok"]))
    h.learn_sync = True
    h.learning = False
    h.turn("anything")
    assert h.forge.pending() == [] and not (cfg.personas_dir / "learned").exists()
