"""Workers acting through a native `python` tool (Prime Agent style) instead of fenced blocks."""
import json

from jevh.jev import Jev
from jevh.learn import work_log
from jevh.llm import LLMResult
from jevh.orchestrator import Harness

from conftest import FakeSystemOne, choice_ans, noul_ans


def call(code: str | None = None, name: str = "python", cid: str = "c1", **args) -> LLMResult:
    args = {"code": code} if code is not None else args
    return LLMResult(text="", tool_calls=[{"id": cid, "name": name, "arguments": json.dumps(args)}])


class NativeLLM:
    """Scripted replies; a model whose provider has native_tools on."""
    native_tools = True

    def __init__(self, replies):
        self.replies, self.seen, self.tools_seen, self.name = list(replies), [], [], "native"

    def chat(self, messages, tools=None):
        self.seen.append([dict(m) for m in messages])
        self.tools_seen.append(tools)
        r = self.replies.pop(0) if self.replies else "done"
        return r if isinstance(r, LLMResult) else LLMResult(text=r)


def harness(cfg, llm, rules=None):
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: llm)
    h.learning = False
    return h


def test_python_tool_call_runs_and_its_result_comes_back_as_a_tool_message(cfg):
    llm = NativeLLM([call("x = 6 * 7\nprint(x)"), call("final(str(x))")])
    r = harness(cfg, llm).turn("what is 6*7? use code", force="high_end_dev")
    assert r.answer == "42"
    assert llm.tools_seen[0][0]["function"]["name"] == "python"
    assert "the `python` tool" in llm.seen[0][0]["content"] and "```python block" not in llm.seen[0][0]["content"]
    second = llm.seen[1]
    assert second[-2]["tool_calls"][0]["function"]["name"] == "python"
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "c1" and second[-1]["content"].strip() == "42"


def test_a_module_called_as_a_tool_is_run_and_an_unknown_one_is_explained(cfg):
    llm = NativeLLM([call(name="weather", city="SLC"), call(name="final", answer="Sunny.")])
    r = harness(cfg, llm).turn("weather in Salt Lake?", force="high_end_dev")
    assert r.answer == "Sunny."
    reply = llm.seen[1][-1]
    assert reply["role"] == "tool" and "no `weather` tool" in reply["content"]


def test_several_calls_in_one_reply_each_get_a_result(cfg):
    two = LLMResult(text="", tool_calls=[{"id": "a", "name": "python", "arguments": '{"code": "y = 2"}'},
                                         {"id": "b", "name": "python", "arguments": '{"code": "print(y + 1)"}'}])
    llm = NativeLLM([two, call("final(str(y))", cid="c")])
    assert harness(cfg, llm).turn("go", force="high_end_dev").answer == "2"
    ids = [m["tool_call_id"] for m in llm.seen[1] if m["role"] == "tool"]
    assert ids == ["a", "b"] and llm.seen[1][-2]["content"].strip() == "3"


def test_bad_json_arguments_still_run():
    from jevh.worker import call_args
    assert call_args('{"code": "print(1)"}') == {"code": "print(1)"}
    assert call_args('{"code": "print("a\\qb")"}')["code"].startswith("print(")
    assert call_args("print(2)") == {"code": "print(2)"}


def test_work_log_reads_native_transcripts():
    t = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"},
         {"role": "assistant", "content": "", "tool_calls": [
             {"id": "c1", "type": "function", "function": {"name": "python", "arguments": '{"code": "print(41 + 1)"}'}}]},
         {"role": "tool", "tool_call_id": "c1", "content": "42"}]
    log = work_log(t)
    assert "CODE:\nprint(41 + 1)" in log and "OUTPUT: 42" in log


def test_a_bounced_final_gets_its_tool_result_then_the_problems(cfg):
    cfg.raw = {**cfg.raw, "verify": {"enabled": True, "max_bounces": 1, "render_html": False}}
    scores = {"claims": 0.05, "delivers": 0.95, "honest": 0.1, "accurate": 0.95}

    def rules(k, q, s):
        if k == "persona":
            return choice_ans("high_end_dev", 0.9, q["criteria"])
        if k in scores:
            return noul_ans(scores[k])
    llm = NativeLLM([call("print('Delta $1,284')\nfinal('Cheapest: $1,099')"),
                     call("final('Cheapest: Delta at $1,284')", cid="c2")])
    r = harness(cfg, llm, rules).turn("cheapest fare?", force="high_end_dev")
    assert r.answer == "Cheapest: Delta at $1,284"
    after = llm.seen[1]
    assert after[-2]["role"] == "tool" and after[-2]["tool_call_id"] == "c1" and "Delta $1,284" in after[-2]["content"]
    assert after[-1]["role"] == "user" and "1,099" in after[-1]["content"]
