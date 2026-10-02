"""Memory search/facts and after-turn learning (facts + skills)."""
import json
import time
from pathlib import Path

from jevh.jev import Jev
from jevh.learn import Learner, work_log
from jevh.memory import Memory, Turn, fts_query
from jevh.orchestrator import Harness
from jevh.skills import Skill, load_roster, parse_skill

from conftest import FakeSystemOne, ScriptedLLM, choice_ans, noul_ans, py


def test_fts_query_is_safe_and_drops_stopwords():
    assert fts_query('What is "the" radon; test? OR DROP') == '"radon" OR "test" OR "drop"'
    assert fts_query("is it a?") is None


def test_search_reaches_turns_outside_the_recent_window(tmp_path):
    m = Memory(tmp_path)
    m.append("daily_assistant", "Dave Miller's radon test is Thursday at 430 Cleveland St.", "Added.", "s1")
    for i in range(50):
        m.append("daily_assistant", f"filler question {i}", "ok", "s1")
    assert not any("radon" in c for c in m.candidates("daily_assistant", 20, 6))
    cands = m.candidates("daily_assistant", 20, 6, "When is the radon appointment?", 10, 0)
    assert any("Cleveland" in c for c in cands)
    m.close()


def test_imports_existing_jsonl_history(tmp_path):
    t = Turn("fitness_coach", "My squat max is 140kg", "Noted", "old", time.time())
    (tmp_path / "timeline.jsonl").write_text(json.dumps(t.__dict__) + "\n", encoding="utf-8")
    m = Memory(tmp_path)
    assert m.count() == 1 and m.search("squat max")[0].user == "My squat max is 140kg"
    m.close()


def test_facts_match_then_top_up_with_newest(tmp_path):
    m = Memory(tmp_path)
    a = m.add_fact("Jed is vegetarian.")
    m.add_fact("Jed's gym is closed on Sundays.")
    cands = m.candidates("daily_assistant", 0, 0, "plan my meals", 0, 5)
    assert any("vegetarian" in c for c in cands) and any("Sundays" in c for c in cands)
    m.retire_fact(a)
    assert not any("vegetarian" in c for c in m.candidates("daily_assistant", 0, 0, "vegetarian food", 0, 5))
    m.close()


def learn_rules(fact=0.95, same=0.9, skill=0.9):
    def rules(k, q, s):
        instr = str(q.get("instructions", ""))
        if k == "fact":
            return noul_ans(fact)
        if k == "skill":
            return noul_ans(skill)
        if "redundant or outdated" in instr:
            return noul_ans(same)
    return rules


def learner(cfg, tmp_path, rules):
    mem = Memory(tmp_path / "mem")
    return Learner(Jev(cfg, FakeSystemOne(rules)), mem, tmp_path / "skills", cfg.t), mem


def test_fact_is_extracted_and_replaces_the_outdated_one(cfg, tmp_path):
    L, mem = learner(cfg, tmp_path, learn_rules())
    mem.add_fact("Jed trains at Gold's Gym.")
    llm = ScriptedLLM(['["Jed now trains at Anytime Fitness."]'])
    facts, retired, p = L.facts("I switched gyms, I'm at Anytime Fitness now", "Noted!", llm)
    assert facts == ["Jed now trains at Anytime Fitness."] and retired == ["Jed trains at Gold's Gym."]
    assert [f.text for f in mem.facts(None)] == ["Jed now trains at Anytime Fitness."]
    mem.close()


def test_no_fact_no_model_call(cfg, tmp_path):
    L, mem = learner(cfg, tmp_path, learn_rules(fact=0.1))
    llm = ScriptedLLM([])
    assert L.facts("what's 2+2?", "4", llm)[0] == [] and llm.seen == []
    mem.close()


TRANSCRIPT = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"},
              {"role": "assistant", "content": py("comfy.run(wf)")},
              {"role": "user", "content": "Output:\nRuntimeError: value_not_in_list ckpt_name"},
              {"role": "assistant", "content": py("comfy.set(wf, {'4.ckpt_name': 'Qwen/x.safetensors'})")},
              {"role": "user", "content": "Output:\n['a.png']"}]


def test_work_log_keeps_code_and_outputs():
    log = work_log(TRANSCRIPT)
    assert "CODE:\ncomfy.run(wf)" in log and "OUTPUT: RuntimeError" in log and "sys" not in log


def llm_reply_again():
    return "---\nname: qwen image checkpoint\ndescription: Same thing again.\n---\n# again"


def test_draft_skills_are_labelled_in_the_prompt(cfg):
    from jevh.personas import load_personas
    from jevh.worker import build_system
    p = load_personas(cfg.personas_dir)["daily_assistant"]
    draft = Skill("learned-x", "d", "Use the Algolia API.", "x", draft=True)
    assert "not yet reviewed" in build_system(p, [draft], {}, [])
    assert "not yet reviewed" not in build_system(p, [Skill("y", "d", "b", "y")], {}, [])


def test_new_skill_is_written_and_loadable(cfg, tmp_path):
    L, mem = learner(cfg, tmp_path, learn_rules())
    llm = ScriptedLLM(["```markdown\n---\nname: qwen image checkpoint\ndescription: Use when a Qwen image "
                       "workflow fails on ckpt_name.\n---\n# Qwen checkpoints\n## Steps\n1. list models\n```"])
    path, action, p = L.skill("make a qwen image", TRANSCRIPT, [], llm)
    assert action == "created" and Path(path).as_posix().endswith("learned/learned-qwen-image-checkpoint/SKILL.md")
    s = parse_skill(Path(path))
    assert s.name == "learned-qwen-image-checkpoint" and "list models" in s.body and s.draft
    again, _, _ = L.skill("make a qwen image", TRANSCRIPT, [], ScriptedLLM([llm_reply_again()]))
    assert parse_skill(Path(again)).name == "learned-qwen-image-checkpoint-2"
    mem.close()


def test_lessons_for_a_foreign_skill_go_to_an_overlay(cfg, tmp_path):
    L, mem = learner(cfg, tmp_path, learn_rules())
    foreign = tmp_path / "hermes" / "comfyui" / "SKILL.md"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("---\nname: comfyui\ndescription: run comfy\n---\nbody", encoding="utf-8")
    skill = parse_skill(foreign)
    llm = ScriptedLLM(["- Checkpoint names need the Qwen/ folder prefix."])
    path, action, _ = L.skill("make a qwen image", TRANSCRIPT, [skill], llm)
    assert action == "lessons" and foreign.read_text(encoding="utf-8").endswith("body")  # untouched
    roster = load_roster([tmp_path / "hermes"], tmp_path / "skills" / "_lessons")
    assert "Qwen/ folder prefix" in roster["comfyui"].body
    mem.close()


def test_turn_learns_a_fact_through_the_harness(cfg):
    def rules(k, q, s):
        if k == "persona":
            return choice_ans("daily_assistant", 0.9, q["criteria"])
        return learn_rules(skill=0.0)(k, q, s)
    llm = ScriptedLLM(["Noted, no Sunday sessions.", '["Jed\'s gym is closed on Sundays."]'])
    events = []
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: llm,
                on_event=lambda k, d: events.append((k, d)))
    h.learn_sync = True
    h.turn("my gym is closed on sundays")
    assert [f.text for f in h.memory.facts(None)] == ["Jed's gym is closed on Sundays."]
    assert any(k == "learned" and d["learned"].facts for k, d in events)


def test_recall_and_remember_from_the_repl(cfg):
    def rules(k, q, s):
        if k == "persona":
            return choice_ans("daily_assistant", 0.9, q["criteria"])
        if k.startswith("m") and q["type"] == "noul":
            return noul_ans(0.9 if "Cleveland" in str(q["instructions"]) else 0.0)
    llm = ScriptedLLM([py("remember('Dave Miller is a radon client.')\nfinal(str(recall('radon appointment')))")])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: llm)
    h.learning = False
    h.memory.append("daily_assistant", "Dave's radon test is Thursday at 430 Cleveland St.", "Added.", "old")
    r = h.turn("when is the radon appointment?", force="daily_assistant")
    assert "Cleveland" in r.answer
    assert [f.text for f in h.memory.facts(None)] == ["Dave Miller is a radon client."]


def test_a_new_turn_drops_background_learning(cfg):
    import time as _time

    class SlowLearnLLM(ScriptedLLM):
        def chat(self, messages, cancel=None):
            if "lasting facts" in messages[-1]["content"]:  # the learning call: slow unless stopped
                for _ in range(100):
                    if cancel is not None and cancel.is_set():
                        from jevh.errors import Cancelled
                        raise Cancelled("dropped")
                    _time.sleep(0.05)
                return super().chat(messages)
            return super().chat(messages)

    def rules(k, q, s):
        if k == "persona":
            return choice_ans("daily_assistant", 0.9, q["criteria"])
        return learn_rules(skill=0.0)(k, q, s)

    events = []
    llm = SlowLearnLLM(["Noted.", '["Jed trains at 6am."]', "Sure, here's a joke."])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: llm,
                on_event=lambda k, d: events.append((k, d)))
    h.turn("I always train at 6am")                   # learning starts on a thread (slow)
    _time.sleep(0.3)
    started = _time.time()
    h.learning = False                                # (so only the first turn's learning is in play)
    h.turn("tell me a joke")                          # the new turn drops it
    h.wait_learning(10)
    assert _time.time() - started < 4
    assert any(k == "learn_error" and "new turn" in d["error"] for k, d in events)
    assert h.memory.facts(None) == []


def test_a_long_output_line_keeps_the_part_with_the_answers_number():
    from jevh.learn import evidence_lines, evidence_terms
    para = ("On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in "
            "Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, "
            "establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is listed in "
            "the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs.")
    kept = evidence_lines("Model\nBLEU\n" + para, evidence_terms("28.4"))
    assert "score of 28.4" in kept and kept.startswith("...") and len(kept) < 220   # live: cut at char 200 before
    assert evidence_lines("short line with 28.4 in it", evidence_terms("28.4")) == "short line with 28.4 in it"
