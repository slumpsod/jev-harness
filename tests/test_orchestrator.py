import time

from jevh.decide import SystemOneError
from jevh.jev import Jev
from jevh.orchestrator import Harness

from conftest import FakeSystemOne, ScriptedLLM, choice_ans, noul_ans, py


def team_rules(route_to):
    """Route to `route_to`, suggest dev+designer for app builds, approve requests."""

    def rules(k, q, s):
        if k == "persona":
            return choice_ans(route_to, 0.9, q["criteria"])
        if k == "complexity":
            return {"score": 1.8, "confidence": 0.9}
        if k in ("needs_high_end_dev", "needs_front_end_designer", "in_scope", "necessary"):
            return noul_ans(0.9)
    return rules


def harness(cfg, rules, scripts, events=None):
    llms = {name: ScriptedLLM(replies, name) for name, replies in scripts.items()}
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: llms[p.name],
                on_event=(lambda k, d: events.append((k, d))) if events is not None else None)
    return h, llms


def test_plain_answer_turn_writes_memory(cfg):
    h, llms = harness(cfg, team_rules("daily_assistant"), {"daily_assistant": ["Hello Jed!"]})
    r = h.turn("hi")
    assert r.persona == "daily_assistant" and r.answer == "Hello Jed!"
    assert h.memory.own("daily_assistant", 5)[0].assistant == "Hello Jed!"


def test_fitness_app_pulls_in_dev_and_designer(cfg):
    events = []
    scripts = {
        "fitness_coach": [py(
            "spec = save_artifact('spec.md', '# Workout tracker\\n- log sets\\n- chart e1RM')\n"
            "r = request('high_end_dev', 'Build a workout tracker per the spec', inputs=[spec])\n"
            "final('Your tracker is ready: ' + r['answer'], r['artifacts'])")],
        "high_end_dev": [py(
            "spec = load_artifact(ctx['inputs'][0])\n"
            "d = request('front_end_designer', 'Style the tracker UI')\n"
            "p = save_artifact('index.html', '<html>' + d['answer'] + '</html>')\n"
            "final('built index.html', [p])")],
        "front_end_designer": [py(
            "img = request('comfyui_manager', 'hero image of a barbell')\n"
            "final('dark theme, hero=' + str(img['artifacts']))")],
        "comfyui_manager": [py("p = save_artifact('hero.png', b'PNG')\nfinal('hero done', [p])")],
    }
    h, llms = harness(cfg, team_rules("fitness_coach"), scripts, events)
    r = h.turn("I want an app to look at my workouts and track progress")

    assert r.persona == "fitness_coach"
    assert [t.to for t in r.tasks] == ["fitness_coach", "high_end_dev", "front_end_designer", "comfyui_manager"]
    assert all(t.status == "done" for t in r.tasks)
    assert r.artifacts and r.artifacts[0].endswith("index.html")
    assert "hero" in open(r.artifacts[0]).read()
    collab = [d for k, d in events if k == "collab"][0]["decision"]
    assert set(collab.suggested) == {"high_end_dev", "front_end_designer"}
    # the dev saw only its task, never the user's original message
    dev_prompt = llms["high_end_dev"].seen[0]
    assert "Build a workout tracker per the spec" in dev_prompt[1]["content"]
    assert "track progress" not in dev_prompt[0]["content"] + dev_prompt[1]["content"]


def test_request_outside_can_request_is_refused(cfg):
    scripts = {"comfyui_manager": [py("r = request('high_end_dev', 'write code')\nfinal(r['status'] + ': ' + r['answer'])")]}
    def rules(k, q, s):  # simple request: Jev suggests no collaborators
        return {"score": 0.0} if k == "complexity" else team_rules("comfyui_manager")(k, q, s)

    h, _ = harness(cfg, rules, scripts)
    r = h.turn("make an image")
    assert r.answer.startswith("rejected") and "may not call" in r.answer


def test_cycle_is_refused(cfg):
    scripts = {
        "high_end_dev": [py("r = request('front_end_designer', 'design')\nfinal(r['answer'])")],
        "front_end_designer": [py("r = request('high_end_dev', 'code it')\nfinal(r['status'] + ' ' + r['answer'])")],
    }
    h, _ = harness(cfg, team_rules("high_end_dev"), scripts)
    h.personas["front_end_designer"].can_request = ["*"]
    r = h.turn("build a page")
    assert "rejected" in r.answer and "cycle" in r.answer


def test_jev_declines_unnecessary_request(cfg):
    def rules(k, q, s):
        base = team_rules("high_end_dev")(k, q, s)
        return noul_ans(0.1) if k == "necessary" else base

    scripts = {"high_end_dev": [py("r = request('comfyui_manager', 'random cat picture')\nfinal(r['answer'])")]}
    h, _ = harness(cfg, rules, scripts)
    assert "Jev declined" in h.turn("fix the login bug").answer


def test_memory_gating_keeps_other_personas_out(cfg):
    def rules(k, q, s):
        if k == "persona":
            return choice_ans("fitness_coach" if "squat" in s["latest_request"] else "high_end_dev", 0.9, q["criteria"])
        if k.startswith("m"):
            return noul_ans(0.9 if "fastapi" in q["instructions"]["snippet"] else 0.05)

    scripts = {"fitness_coach": ["Squat 3x5 at 100kg."] * 3, "high_end_dev": ["Use a dependency.", "ok"]}
    h, llms = harness(cfg, rules, scripts)
    h.turn("what should I squat today")
    h.turn("how do I structure a fastapi route?")
    h.turn("more squat advice")
    r = h.turn("and fastapi auth?")
    last_dev_prompt = llms["high_end_dev"].seen[-1][1]["content"]
    assert r.persona == "high_end_dev"
    assert "memory" in last_dev_prompt and "list of 1" in last_dev_prompt  # only the fastapi turn passed


def test_jev_down_degrades_to_default(cfg):
    class Down:
        def ask(self, *a, **k):
            raise SystemOneError("unreachable")

    events = []
    h = Harness(cfg, jev=Jev(cfg, Down()), llm_factory=lambda p: ScriptedLLM(["fine"]),
                on_event=lambda k, d: events.append(k))
    r = h.turn("hello")
    assert r.persona == cfg.default_persona and r.answer == "fine"
    assert "jev_error" in events


def test_worker_crash_is_contained(cfg):
    class Broken:
        name = "broken"

        def chat(self, messages):
            raise ConnectionError("LM Studio is down")

    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(team_rules("daily_assistant"))), llm_factory=lambda p: Broken())
    r = h.turn("hi")
    assert "failed" in r.answer and r.tasks[0].status == "failed"


def test_cut_off_code_block_asks_for_smaller_steps(cfg):
    scripts = {"high_end_dev": ["```python\nhtml = '<div>' * 99999", py("final('ok in parts')")]}
    h, llms = harness(cfg, team_rules("high_end_dev"), scripts)
    r = h.turn("build a page")
    assert r.answer == "ok in parts"
    assert "cut off" in llms["high_end_dev"].seen[1][-1]["content"]


def test_failed_turn_is_not_remembered(cfg):
    class Broken:
        name = "broken"

        def chat(self, messages):
            raise ConnectionError("down")

    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(team_rules("daily_assistant"))), llm_factory=lambda p: Broken())
    h.turn("hi")
    assert h.memory.own("daily_assistant", 5) == []


def test_cli_imports_and_runs_one_shot(cfg, monkeypatch, capsys):
    import jevh.cli as cli

    monkeypatch.setattr(cli, "load_config", lambda path=None: cfg)
    monkeypatch.setattr(cli, "Harness", lambda c, on_event=None, **kw: Harness(
        c, jev=Jev(c, FakeSystemOne(team_rules("daily_assistant"))),
        llm_factory=lambda p: ScriptedLLM(["pong"]), on_event=on_event))
    cli.main(["-c", "ping"])
    assert "pong" in capsys.readouterr().out


def test_step_limit_warns_and_harvests_files(cfg):
    step = py("open('app.html', 'w').write('<html>hi</html>')")
    scripts = {"high_end_dev": [step] * 30}
    h, llms = harness(cfg, team_rules("high_end_dev"), scripts)
    h.personas["high_end_dev"].max_steps = 4
    r = h.turn("build a page")
    assert "step limit" in r.answer
    assert r.artifacts and r.artifacts[0].endswith("high_end_dev_app.html")
    prompts = [m["content"] for m in llms["high_end_dev"].seen[-1] if m["role"] == "user"]
    assert any("2 steps left" in c for c in prompts)


class SlowLLM(ScriptedLLM):
    def __init__(self, replies, delay):
        super().__init__(replies)
        self.delay = delay

    def chat(self, messages):
        time.sleep(self.delay)
        return super().chat(messages)


def slow_harness(cfg, scripts, delay=1.5, events=None):
    llms = {n: (SlowLLM(r, delay) if n != "fitness_coach" else ScriptedLLM(r)) for n, r in scripts.items()}
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(team_rules("fitness_coach"))), llm_factory=lambda p: llms[p.name],
                on_event=(lambda k, d: events.append((k, d))) if events is not None else None)
    return h, llms


def test_parallel_handoffs_overlap_in_time(cfg):
    scripts = {
        "fitness_coach": [py("rs = request_all([('high_end_dev', 'build it'), ('front_end_designer', 'style it')])\n"
                             "final(' + '.join(r['answer'] for r in rs), [a for r in rs for a in r['artifacts']])")],
        "high_end_dev": [py("final('code', [save_artifact('app.js', 'x')])")],
        "front_end_designer": [py("final('css', [save_artifact('app.css', 'y')])")],
    }
    h, _ = slow_harness(cfg, scripts, delay=1.5)
    t = time.time()
    r = h.turn("app please")
    elapsed = time.time() - t
    assert r.answer == "code + css"
    assert sorted(a.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for a in r.artifacts) == ["app.css", "app.js"]
    assert elapsed < 2.9, f"hand-offs ran sequentially ({elapsed:.1f}s)"  # 2 x 1.5s if sequential


def test_request_async_wait_timeout_then_result(cfg):
    scripts = {
        "fitness_coach": [py("h = request_async('high_end_dev', 'slow job')\n"
                             "first = wait(h, timeout=0.2)['status']\n"
                             "final(first + ' then ' + wait(h)['answer'])")],
        "high_end_dev": [py("final('done')")],
    }
    h, _ = slow_harness(cfg, scripts, delay=1.0)
    assert h.turn("go").answer == "running then done"


def test_unwaited_handoff_is_finished_and_artifacts_kept(cfg):
    events = []
    scripts = {
        "fitness_coach": [py("request_async('high_end_dev', 'fire and forget')\nfinal('started it')")],
        "high_end_dev": [py("final('ok', [save_artifact('late.txt', 'z')])")],
    }
    h, _ = slow_harness(cfg, scripts, delay=0.5, events=events)
    r = h.turn("go")
    assert r.answer == "started it" and any(a.endswith("late.txt") for a in r.artifacts)
    assert any(k == "unwaited" for k, _ in events)


def test_task_limit_holds_under_parallel_requests(cfg):
    cfg.limits["max_tasks_per_turn"] = 3  # root + 2 hand-offs
    scripts = {
        "fitness_coach": [py("rs = request_all([('high_end_dev', 'a'), ('front_end_designer', 'b'), ('comfyui_manager', 'c')])\n"
                             "final(','.join(sorted(r['status'] for r in rs)))")],
        "high_end_dev": [py("final('a')")], "front_end_designer": [py("final('b')")], "comfyui_manager": [py("final('c')")],
    }
    h, _ = slow_harness(cfg, scripts, delay=0.3)
    h.personas["fitness_coach"].can_request = ["*"]
    r = h.turn("go")
    assert r.answer == "done,done,rejected"


def test_unknown_handle_is_a_clear_error(cfg):
    scripts = {"fitness_coach": [py("try:\n    wait('nope')\nexcept RuntimeError as e:\n    final(str(e))")]}
    h, _ = slow_harness(cfg, scripts)
    assert "unknown handle" in h.turn("go").answer


def test_model_override_puts_every_persona_on_one_model(cfg):
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne()), model_override="lmstudio:tiny-model")
    m = h._model_cfg(h.personas["high_end_dev"])
    assert m["provider"] == "lmstudio" and m["model"] == "tiny-model" and "fallback" not in m
    assert m.get("max_tokens") is None or m["max_tokens"] > 0


def test_repeated_error_triggers_stuck_note(cfg):
    scripts = {"high_end_dev": [py("1/0")] * 4 + [py("final('gave up')")]}
    h, llms = harness(cfg, team_rules("high_end_dev"), scripts)
    assert h.turn("x").answer == "gave up"
    fed_back = [m["content"] for m in llms["high_end_dev"].seen[-1] if m["role"] == "user"]
    assert sum("same error has now happened 3 times" in c for c in fed_back) >= 1
    assert "same error" not in fed_back[2]  # not before the third repeat


def test_tool_alias_importable_and_comfy_down_is_clear(cfg, monkeypatch):
    monkeypatch.setenv("JEVH_COMFY_URL", "http://127.0.0.1:9")  # nothing listens there
    scripts = {"comfyui_manager": [py("import comfy\ntry:\n    comfy.status()\nexcept Exception as e:\n"
                                      "    final(type(e).__name__ + ': ' + str(e)[:60])")]}
    h, _ = harness(cfg, team_rules("comfyui_manager"), scripts)
    r = h.turn("make an image")
    assert r.answer.startswith("ComfyDown: ComfyUI is not reachable")


def test_final_without_artifacts_ships_written_files(cfg):
    scripts = {
        "fitness_coach": [py("r = request('high_end_dev', 'write index.html')\nfinal(r['answer'], r['artifacts'])")],
        "high_end_dev": [py("open('index.html', 'w').write('<html>ok</html>')\nfinal('wrote index.html')")],
    }
    h, _ = harness(cfg, team_rules("fitness_coach"), scripts)
    r = h.turn("x")
    assert any(a.endswith("high_end_dev_index.html") for a in r.artifacts)


def test_handoff_overlap_math():
    from types import SimpleNamespace as NS
    from jevh.cli import handoff_overlap
    ts = [NS(depth=1, started=0, finished=10), NS(depth=1, started=2, finished=8),
          NS(depth=1, started=12, finished=15), NS(depth=2, started=1, finished=9), NS(depth=0, started=0, finished=20)]
    assert handoff_overlap(ts) == (19, 13)


def tier_harness(cfg, p_strong, persona="daily_assistant"):
    """Jev answers the tier Choice with P(strong)=p_strong; records each model block used."""
    def rules(k, q, s):
        if k == "tier":
            return {"choice": "strong" if p_strong >= 0.5 else "fast", "confidence": max(p_strong, 1 - p_strong),
                    "probabilities": {"strong": p_strong, "fast": 1 - p_strong}}
        return team_rules(persona)(k, q, s)
    used = []

    def factory(p):
        used.append((p.name, p.model["model"]))
        return ScriptedLLM(["ok"])
    fake = FakeSystemOne(rules)
    return Harness(cfg, jev=Jev(cfg, fake), llm_factory=factory), used, fake


def test_easy_task_goes_to_fast_model(cfg):
    h, used, _ = tier_harness(cfg, p_strong=0.1)
    h.turn("what's 2+2")
    assert used == [("daily_assistant", cfg.models["fast"]["model"])]


def test_hard_task_keeps_persona_model(cfg):
    h, used, _ = tier_harness(cfg, p_strong=0.9)
    h.turn("reconcile a year of ledgers")
    assert used == [("daily_assistant", h.personas["daily_assistant"].model["model"])]


def test_fast_model_keeps_persona_sampling_but_drops_fallback(cfg):
    h, _, _ = tier_harness(cfg, p_strong=0.1)
    dev = h._model_cfg(h.personas["high_end_dev"], "fast")
    assert dev["model"] == cfg.models["fast"]["model"] and "fallback" not in dev and "effort" not in dev
    fe = h._model_cfg(h.personas["front_end_designer"], "fast")
    assert fe["model"] == cfg.models["fast"]["model"]
    assert "temperature" not in fe   # no persona temperature: the model's own recommended sampling applies
    assert "think" not in fe and "max_tokens" not in fe  # and not Qwen's "think: unlimited" / caps
    h.personas["front_end_designer"].model["temperature"] = 0.6
    assert h._model_cfg(h.personas["front_end_designer"], "fast")["temperature"] == 0.6   # an explicit one is kept


def test_no_tier_call_when_persona_opts_out(cfg):
    h, used, fake = tier_harness(cfg, p_strong=0.1)
    h.personas["daily_assistant"].fast_ok = False
    h.turn("hi")
    assert not any("tier" in q for _, q in fake.calls)
    assert used == [("daily_assistant", h.personas["daily_assistant"].model["model"])]


def test_forced_tier_skips_jev(cfg):
    h, used, fake = tier_harness(cfg, p_strong=0.99)
    h.tier_mode = "fast"
    h.turn("reconcile a year of ledgers")
    assert not any("tier" in q for _, q in fake.calls)
    assert used == [("daily_assistant", cfg.models["fast"]["model"])]


def test_tiering_off_without_fast_model(cfg):
    cfg.models = {}
    h, used, fake = tier_harness(cfg, p_strong=0.1)
    h.turn("hi")
    assert used == [("daily_assistant", h.personas["daily_assistant"].model["model"])]


def test_jev_down_during_tier_uses_strong(cfg):
    h, used, _ = tier_harness(cfg, p_strong=0.1)

    def boom(*a, **k):
        raise SystemOneError("down")
    h.jev.pick_tier = boom
    h.turn("hi")
    assert used == [("daily_assistant", h.personas["daily_assistant"].model["model"])]


def test_handoffs_are_tiered_independently(cfg):
    def rules(k, q, s):
        if k == "tier":
            hard = "hard" in s["task"]
            p = 0.95 if hard else 0.05
            return {"choice": "strong" if hard else "fast", "confidence": 0.95,
                    "probabilities": {"strong": p, "fast": 1 - p}}
        return team_rules("fitness_coach")(k, q, s)
    used = []
    replies = {"fitness_coach": [py("request_all([('high_end_dev', 'hard build'), "
                                    "('front_end_designer', 'easy tweak')])\nfinal('ok')")]}

    def factory(p):
        used.append((p.name, p.model["model"]))
        return ScriptedLLM(replies.get(p.name, [py("final('done')")]))
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=factory)
    h.turn("easy app request")
    got = dict(used)
    assert got["fitness_coach"] == cfg.models["fast"]["model"]
    assert got["front_end_designer"] == cfg.models["fast"]["model"]
    assert got["high_end_dev"] == h.personas["high_end_dev"].model["model"]


def followup_rules(continues, replies, persona="daily_assistant"):
    def rules(k, q, s):
        if k == "persona":
            return choice_ans(persona, 0.9, q["criteria"])
        if k == "continues":
            return noul_ans(continues)
        if k == "replies":
            return noul_ans(replies)
    return rules


def test_answer_to_a_clarifying_question_gets_the_exchange(cfg):
    llm = ScriptedLLM(["Which state, and mail forwarding or DMV?", "USPS mail forwarding in Utah: ..."])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(followup_rules(0.3, 0.1))), llm_factory=lambda p: llm)
    h.learning = False
    h.turn("how do I make a change of address?")
    h.turn("Utah, for my mail")
    second = llm.seen[1]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]
    assert second[1]["content"] == "how do I make a change of address?"
    assert second[2]["content"].startswith("Which state, and mail forwarding or DMV?")
    assert "no code or tools were run" in second[2]["content"]   # the work notes ride along
    assert second[3]["content"].startswith("Utah, for my mail")


def test_follow_up_sees_what_the_last_answer_was_based_on(cfg):
    """Live 2026-09-28: 'where did you pull that from?' saw only the reply and confessed to
    making up a price the worker had read off the store's page."""
    llm = ScriptedLLM([py("print('Fairview 4 Piece Lounge Set - Dark / Sale price $2,999.99 / Only 2 units left')"),
                       "It's $2,999.99, and the Dark finish has 2 left.",
                       "From the store's product listing."])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(followup_rules(0.1, 0.9))), llm_factory=lambda p: llm)
    h.learning = False
    h.turn("how much is the Fairview lounge set at Porch & Patio?")
    h.turn("where did you pull that information from?")
    earlier = llm.seen[-1][2]["content"]
    assert earlier.startswith("It's $2,999.99") and "Sale price $2,999.99" in earlier


def test_new_topic_starts_clean(cfg):
    llm = ScriptedLLM(["plan...", "Salt Lake City."])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(followup_rules(0.05, 0.1))), llm_factory=lambda p: llm)
    h.learning = False
    h.turn("plan my week")
    h.turn("what's the capital of Utah?")
    assert [m["role"] for m in llm.seen[1]] == ["system", "user"]


def test_follow_up_also_works_with_a_pinned_persona_and_informs_jev(cfg):
    fake = FakeSystemOne(followup_rules(0.1, 0.8))
    llm = ScriptedLLM(["How many days can you train?", "4-day plan..."])
    h = Harness(cfg, jev=Jev(cfg, fake), llm_factory=lambda p: llm)
    h.learning = False
    h.turn("plan my workouts", force="fitness_coach")
    h.turn("4 days, full gym", force="fitness_coach")
    assert [m["role"] for m in llm.seen[1]][:3] == ["system", "user", "assistant"]
    tier_states = [s for s, q in fake.calls if "tier" in q]
    assert "plan my workouts" in tier_states[-1]["task"] and "Latest message: 4 days, full gym" in tier_states[-1]["task"]


def test_a_broken_event_handler_cannot_stop_the_work(cfg):
    def explode(kind, data):
        raise KeyError("stuck")
    llm = ScriptedLLM([py("print('working')"), py("final('done anyway')")])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(team_rules("daily_assistant"))), llm_factory=lambda p: llm,
                on_event=explode)
    h.learning = False
    assert h.turn("do it").answer == "done anyway"


def test_empty_reply_is_continued_not_judged(cfg):
    """ARC bench: Qwen once thought for 10 minutes and replied with nothing; it went to the
    answer check as a 'plain answer' and was bounced as undelivered."""
    llm = ScriptedLLM([py("print(6 * 7)"), "", py("final('42')")])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne()), llm_factory=lambda p: llm)
    h.learning = False
    assert h.turn("what is 6 times 7? use code", force="high_end_dev").answer == "42"
    assert "reply was empty" in llm.seen[2][-1]["content"]


def test_reasoning_tail_carries_to_the_next_step(cfg):
    """ARC bench: the next step never saw what the last one worked out, so it re-derived it."""
    from jevh.llm import LLMResult
    llm = ScriptedLLM([LLMResult(text="", reasoning="... so the rule is: mirror each row left to right."),
                       py("final('mirrored')")])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne()), llm_factory=lambda p: llm)
    h.learning = False
    assert h.turn("solve the puzzle", force="high_end_dev").answer == "mirrored"
    carried = llm.seen[1][-2]
    assert "mirror each row" in carried["reasoning_content"] and "mirror" not in carried["content"]
