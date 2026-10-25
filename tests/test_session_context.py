"""Session-long context (limits.session_context): a follow-up by the same assistant continues its own conversation - every
step, the code it ran and what came back - until the assistant changes; earlier requests condense first at the compaction
limit. Flag #18, 2026-10-24. Live: evals/bench/session_context.py."""
import json

from jevh.jev import Jev, ToolDecision
from jevh.orchestrator import Harness
from jevh.worker import EARLIER_HEAD, Worker

from conftest import FakeSystemOne, ScriptedLLM, choice_ans, noul_ans, py


def rules(persona="daily_assistant"):
    def r(k, q, s):
        if k == "persona":
            return choice_ans(persona, 0.9, q["criteria"])
        if k in ("continues", "replies"):
            return noul_ans(0.05)
    return r


def harness(cfg, llm, persona="daily_assistant"):
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules(persona))), llm_factory=lambda p: llm)
    h.learning = False
    return h


def test_a_follow_up_continues_the_conversation_with_every_step(cfg):
    llm = ScriptedLLM([py("print('Fairview 4 Piece Lounge Set - Dark / Sale price $2,999.99 / Only 2 units left')"),
                       "It's $2,999.99, and the Dark finish has 2 left.",
                       "From the store's product listing."])
    h = harness(cfg, llm)
    h.turn("how much is the Fairview lounge set at Porch & Patio?")
    h.turn("where did you pull that information from?")
    seen = llm.seen[-1]
    assert seen[1]["content"].startswith("how much is the Fairview lounge set")   # the first request, as the model had it
    # what the code printed is there itself, not a summary of it
    assert any("Sale price $2,999.99" in str(m.get("content")) and m["role"] != "system" and "Harness note" not in
               str(m.get("content")) for m in seen[2:-1])
    assert seen[-1]["role"] == "user" and seen[-1]["content"].startswith("where did you pull that information from?")
    # this turn's own record (learning, the failure log, the turn's notes) is only this turn's work
    notes = h.memory.session_turns(h.session, 1)[-1].notes
    assert "Sale price" not in notes


def test_it_goes_on_across_several_turns(cfg):
    llm = ScriptedLLM([py("x = 41\nprint('x set')"), "Set x.",
                       py("x += 1\nprint(x)"), "x is 42.",
                       "You asked me to set x, then to add one: 42."])
    h = harness(cfg, llm)
    h.turn("set x to 41")
    h.turn("add one to it")
    h.turn("what have we done so far?")
    text = json.dumps(llm.seen[-1])
    assert "set x to 41" in text and "add one to it" in text and "x set" in text and "42" in text


def test_another_assistant_starts_its_own_conversation(cfg):
    llm = ScriptedLLM([py("print('steps for the gym')"), "Leg day.", "Here's a meal plan."])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules())), llm_factory=lambda p: llm)
    h.learning = False
    h.turn("plan my workout", force="fitness_coach")
    h.turn("and what should I eat?", force="daily_assistant")
    seen = llm.seen[-1]
    # the other assistant's exchange as a short summary (the old way), none of its steps
    assert [m["role"] for m in seen] == ["system", "user", "assistant", "user"]
    assert "steps for the gym" not in json.dumps(seen[:2]) and "Leg day." in seen[2]["content"]


def test_off_gives_the_old_summary(cfg):
    cfg.limits["session_context"] = 0
    llm = ScriptedLLM([py("print('the raw output')"), "Done.", "Sure."])
    h = harness(cfg, llm)
    h.turn("do the thing")
    h.turn("and again")
    seen = llm.seen[-1]
    assert [m["role"] for m in seen] == ["system", "user", "assistant", "user"]


def test_earlier_requests_condense_first_at_the_limit(cfg):
    cfg.limits["compact_tokens"] = 900       # (no model window known here: this is the compaction limit)
    cfg.limits["keep_steps"] = 1
    long = "\n".join(f"row {i}: " + "data " * 30 for i in range(40))
    llm = ScriptedLLM([py(f"print({long!r})"), py("print('second look')"), "The table has 40 rows.",
                       py("print('now the sum')"), "The sum is 7."])
    h = harness(cfg, llm)
    h.turn("read the table")
    h.turn("now sum it")
    task = next(m for m in llm.seen[-1] if m["role"] == "user" and "now sum it" in str(m["content"]))
    assert task["content"].startswith(EARLIER_HEAD)
    assert "## Request: read the table" in task["content"] and "You answered: The table has 40 rows." in task["content"]
    # this turn's request and steps are left as they were
    assert llm.seen[-1][1] is not None and "now sum it" in str(llm.seen[-1][1]["content"])
    cp = json.loads((h.session_dir / "u2t1-daily_assistant" / "checkpoint.json").read_text(encoding="utf-8"))
    assert cp["earlier"].startswith(EARLIER_HEAD) and cp["turns"][0]["start"] == 1 and cp["turns"][0]["answer"] == "The sum is 7."


def test_a_cut_off_turn_ends_on_no_unanswered_call(tmp_path):
    w = Worker.__new__(Worker)
    w.continue_from = {"task_idx": 1, "task_text": "mod the save\n\n(In your REPL:\n- ctx['skills']: dict)", "messages": [
        {"role": "user", "content": "mod the save"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "type": "function",
                                                              "function": {"name": "python", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "ok"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "b", "type": "function",
                                                              "function": {"name": "python", "arguments": "{}"}}]}]}
    msgs = w._carried()
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool"]
    assert w._turns == [{"start": 1, "request": "mod the save", "answer": ""}]   # (a checkpoint from before the record)


def test_the_conversation_keeps_its_system_prompt(cfg):
    """The model server reuses what it has read of a prompt's start: a continued conversation keeps the system prompt
    it began with, and what this request's picks add (a tool guide) goes in the request's message."""
    llm = ScriptedLLM(["one", "two"])
    h = harness(cfg, llm)
    picks = iter([ToolDecision([]), ToolDecision(["web"])])
    h._pick_tools = lambda task, persona: next(picks)
    h.turn("first")
    h.turn("second")
    first, second = llm.seen
    assert second[0]["content"] == first[0]["content"]   # word for word (its working folder too)
    task = second[-1]["content"]
    assert "[Your REPL now starts in this request's own working folder:" in task and "u2t1-daily_assistant" in task
    if "web" in h.personas["daily_assistant"].tools:   # the guide picked for this request only
        assert "web.ask(" in task and "web.ask(" not in first[0]["content"]


def test_what_changed_since_the_conversation_began_reaches_the_request(cfg, tmp_path):
    """A project folder opened between two requests (or the user now writing from another computer) changes the
    prompt: the kept one can't say it, so the request does, and says it wins over the start of the conversation."""
    from test_tools import tool_rules
    (tmp_path / "repo").mkdir()
    llm = ScriptedLLM(["one", "two"])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(tool_rules(tools_yes={"project"}))), llm_factory=lambda p: llm)
    h.learning = False
    h.turn("hello", force="high_end_dev")
    h.set_project(str(tmp_path / "repo"))
    h.turn("what's in my repo?", force="high_end_dev")
    first, second = llm.seen
    assert second[0]["content"] == first[0]["content"]
    task = second[-1]["content"]
    assert "project.find(question)" in task and "follow these" in task
