"""Fixes for the sessions flagged 2026-10-21/22 (live bench: evals/bench/flags_12_15_17.py):
#12 what the worker wrote beside its code (steps and links for the user) showed only in the thinking view;
#15 ask_user options in a tool call written as text became a button per character;
#17 light Jev read "go ahead and continue" as a new topic, and the turn started without the conversation."""
import json

from jevh.jev import Jev
from jevh.llm import LLMResult
from jevh.orchestrator import Harness, _option_list
from jevh.worker import spoken, translate_tool_calls

from conftest import FakeSystemOne, ScriptedLLM, choice_ans, noul_ans, py

LEAKED = """<tool_call>
<function=ask_user>
<parameter=question>
That Docker Desktop error is a stale lock file. Want me to handle it?
</parameter>
<parameter=options>
["Yes, go ahead", "Just quit Docker, I'll restart it myself", "No - skip email today"]
</parameter>
</function>
</tool_call>"""


def rules(continues=0.05, replies=0.05, persona="daily_assistant"):
    def r(k, q, s):
        if k == "persona":
            return choice_ans(persona, 0.9, q["criteria"])
        if k == "continues":
            return noul_ans(continues)
        if k == "replies":
            return noul_ans(replies)
    return r


def call(name, args, i="c1"):
    return {"id": i, "name": name, "arguments": json.dumps(args)}


# ------------------------------------------------------------------------------------------------------------ #15
def test_a_tool_call_written_as_text_keeps_its_options_a_list():
    codes, failed = translate_tool_calls(LEAKED, {})
    assert not failed
    assert "'options': ['Yes, go ahead', \"Just quit Docker, I'll restart it myself\", 'No - skip email today']" in codes[0]


def test_text_parameters_stay_text_even_when_they_look_like_json():
    codes, _ = translate_tool_calls("<function=final><parameter=answer>[[1, 2], [3, 4]]</parameter></function>", {})
    assert "'[[1, 2], [3, 4]]'" in codes[0]          # an ARC grid is the answer's text
    codes, _ = translate_tool_calls("<function=ask_user><parameter=question>[urgent] go on?</parameter>"
                                    "<parameter=options>['yes', 'no']</parameter></function>", {})
    assert "'[urgent] go on?'" in codes[0] and "'options': ['yes', 'no']" in codes[0]   # Python quotes too
    codes, _ = translate_tool_calls("<function=ask_user><parameter=question>ok?</parameter>"
                                    "<parameter=options>[not json</parameter></function>", {})
    assert "'options': '[not json'" in codes[0]       # unreadable: left as it was (and _option_list makes it one)


def test_ask_user_options_given_as_one_string_are_a_list():
    assert _option_list('["Yes, go ahead", "No"]') == ["Yes, go ahead", "No"]
    assert _option_list("Yes") == ["Yes"] and _option_list("  ") == [] and _option_list(None) == []
    assert _option_list(("a", "b")) == ["a", "b"]


def test_the_question_box_gets_the_options_from_a_leaked_call(cfg):
    events = []
    llm = ScriptedLLM([LEAKED, py("final('ok')")])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules())), llm_factory=lambda p: llm,
                on_event=lambda k, d: events.append((k, d)))
    h.learning = False

    def answer_when_asked(kind, data):
        events.append((kind, data))
        if kind == "question":
            h.answer(data["question"]["id"], "Yes, go ahead")
    h.on_event = answer_when_asked
    h.turn("check my email")
    q = next(d["question"] for k, d in events if k == "question")
    assert q["options"] == ["Yes, go ahead", "Just quit Docker, I'll restart it myself", "No - skip email today"]


# ------------------------------------------------------------------------------------------------------------ #12
def test_what_a_step_says_beside_its_code():
    assert spoken("Let me check.\n```python\nprint(1)\n```") == "Let me check."
    assert spoken("<tool_call><function=python><parameter=code>print(1)</parameter></function></tool_call>") == ""
    assert spoken("Moves so far:\n```\n1. e4 e5\n```", native=True).startswith("Moves so far:\n```")
    assert spoken("<think>hmm</think>Opening the page now.") == "Opening the page now."


def test_text_beside_the_code_reaches_the_chat_but_not_beside_the_answer(cfg):
    """Live 2026-10-21 (flagged): the click-by-click steps were written beside an ask_user call whose question said only
    "Paste the Client ID"; they showed only in the thinking view."""
    events = []
    steps = "Open https://console.cloud.google.com/auth/clients and click your Nango client."
    llm = ScriptedLLM([LLMResult(text=steps, tool_calls=[call("python", {"code": "x = ask_user('Done?', ['Done'])"})]),
                       LLMResult(text=steps, tool_calls=[call("python", {"code": "print(x)"}, "c2")]),
                       LLMResult(text="Here it is.", tool_calls=[call("final", {"answer": "Connected."}, "c3")])])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules())), llm_factory=lambda p: llm,
                on_event=lambda k, d: events.append((k, d)))
    h.learning = False
    h._ask_user = lambda *a, **k: "Done"
    assert h.turn("connect my calendar").answer == "Connected."
    said = [d for k, d in events if k == "say"]
    assert [d["text"] for d in said] == [steps]          # once (not again word for word), and not beside final()
    assert said[0]["narration"] is True
    assert "Said while working" in h.memory.session_turns(h.session, 1)[0].notes


def test_narration_can_be_turned_off(cfg):
    cfg.limits["narrate"] = 0
    events = []
    llm = ScriptedLLM([LLMResult(text="Checking.", tool_calls=[call("python", {"code": "print(1)"})]), "done"])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules())), llm_factory=lambda p: llm,
                on_event=lambda k, d: events.append((k, d)))
    h.learning = False
    h.turn("go")
    assert not [d for k, d in events if k == "say"]


# ------------------------------------------------------------------------------------------------------------ #17
def test_every_turn_carries_the_last_ones_even_when_jev_reads_a_new_topic(cfg):
    """Live 2026-10-22 (light Jev): "could you also give me unlimited coins? but yeah go ahead and continue" was read as
    a new topic; the worker had nothing of the game it had been modding and went off to start Docker."""
    llm = ScriptedLLM([py("save = {'coins': 120}\nfinal('Patched the save: coins 120.')"),
                       py("save['coins'] = 10**9\nfinal(str(save['coins']))")])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules())), llm_factory=lambda p: llm)
    h.learning = False
    h.turn("mod my game's save file")
    assert h.turn("could you also give me unlimited coins? but yeah go ahead and continue").answer == "1000000000"
    assert [m["role"] for m in llm.seen[1]][:3] == ["system", "user", "assistant"]
    assert llm.seen[1][1]["content"] == "mod my game's save file"


def test_the_first_turn_carries_nothing(cfg):
    llm = ScriptedLLM(["Hello!"])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules())), llm_factory=lambda p: llm)
    h.learning = False
    h.turn("hi")
    assert [m["role"] for m in llm.seen[0]] == ["system", "user"]


def test_off_jev_decides_again(cfg):
    cfg.limits["follow_always"] = 0
    llm = ScriptedLLM(["plan...", "Salt Lake City."])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules())), llm_factory=lambda p: llm)
    h.learning = False
    h.turn("plan my week")
    h.turn("what's the capital of Utah?")
    assert [m["role"] for m in llm.seen[1]] == ["system", "user"]
