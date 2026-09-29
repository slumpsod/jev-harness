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


def test_final_is_offered_as_a_tool_and_finishes_the_task(cfg):
    llm = NativeLLM([call("rows = 600\nprint(rows)"),
                     call(name="final", answer="600 rows", artifacts=[])])
    r = harness(cfg, llm).turn("how many rows? use code", force="high_end_dev")
    assert r.answer == "600 rows" and len(llm.seen) == 2
    assert [t["function"]["name"] for t in llm.tools_seen[0]] == ["python", "final"]
    assert "the `final` tool" in llm.seen[0][0]["content"]


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


def test_final_tool_hands_over_a_variable_exactly(cfg):
    """Live (ornith, ARC): a computed grid passed as `answer` was retyped as strings; answer_var sends it as is."""
    llm = NativeLLM([call("pred = [[0, 2], [3, 0]]"), call(name="final", answer_var="pred")])
    r = harness(cfg, llm).turn("solve it with code", force="high_end_dev")
    assert r.answer == "[[0, 2], [3, 0]]"
    assert "answer_var" in json.dumps(llm.tools_seen[0][1])
    assert "never retype data" in llm.seen[0][0]["content"]


def test_final_in_code_sends_a_list_as_json(cfg):
    llm = NativeLLM([call("final({'total': 3, 'items': ['a', 'b']})")])
    assert harness(cfg, llm).turn("go", force="high_end_dev").answer == '{"total": 3, "items": ["a", "b"]}'


def test_a_silent_server_is_retried_twice(cfg):
    """Live (Gemma via LM Studio): no tokens for 180 s on one step failed a task already solved."""
    from jevh.llm import LLMError

    class Stalls(NativeLLM):
        def __init__(self, replies, stalls):
            super().__init__(replies)
            self.stalls = stalls

        def chat(self, messages, tools=None):
            if self.stalls:
                self.stalls -= 1
                raise LLMError("gemma: no tokens for 180s")
            return super().chat(messages, tools)
    ok = Stalls([call(name="final", answer="done")], stalls=1)
    assert harness(cfg, ok).turn("go", force="high_end_dev").answer == "done"
    twice = Stalls([call(name="final", answer="done")], stalls=2)
    assert harness(cfg, twice).turn("go", force="high_end_dev").answer == "done"
    thrice = Stalls([call(name="final", answer="done")], stalls=3)
    assert "no tokens for 180s" in harness(cfg, thrice).turn("go", force="high_end_dev").answer


def test_a_malformed_final_call_is_read_leniently():
    """Live (Gemma 4 in LM Studio): arguments packed into a key, or left as raw text."""
    from jevh.worker import final_params
    assert final_params({"answer_var='test_output',artifacts=[],answer": "[[8, 8, 9]]"}) == {"answer_var": "test_output"}
    assert final_params({"code": "answer='[[8, 8, 9], [8, 2, 9]]',artifacts=[]"}) == {"answer": "[[8, 8, 9], [8, 2, 9]]"}
    assert final_params({"answer": "ok", "artifacts": []}) == {"answer": "ok", "artifacts": []}


def test_template_tokens_are_cut_from_a_call_and_a_runaway_call_still_runs(cfg):
    """Live: '<tool_call|><|channel>thought\n<channel|>' repeated for 437 KB inside a final call."""
    from jevh.llm import LLMResult, clean_call_args
    junk = "<tool_call|>" + "<|channel>thought\n<channel|>" * 30
    assert clean_call_args('{"answer": "[[1, 2]]"}' + junk) == '{"answer": "[[1, 2]]"}'
    stopped = LLMResult(text="", interrupted=True,
                        tool_calls=[{"id": "c", "name": "final", "arguments": '{"answer": "[[1, 2]]"}'}])
    llm = NativeLLM([stopped])
    assert harness(cfg, llm).turn("go", force="high_end_dev").answer == "[[1, 2]]"


def test_a_call_that_degenerates_into_one_repeated_unit_counts_as_runaway():
    """Live (Gemma 4): '..._thought_content_content_content' for 200 KB inside a final call."""
    from jevh.llm import RUNAWAY_MIN_CHARS, RUNAWAY_RATIO, _packs
    runaway = '{"answer": "[[7, 6, 5]]" songwriter=True_thought' + "_content" * 3000
    code = "\n".join(f"row_{i} = [{', '.join(str((i * j) % 10) for j in range(30))}]" for i in range(200))
    assert len(runaway) > RUNAWAY_MIN_CHARS and _packs(runaway[-6000:]) < RUNAWAY_RATIO
    assert len(code) > RUNAWAY_MIN_CHARS and _packs(code[-6000:]) > RUNAWAY_RATIO


def test_a_broken_call_is_stored_as_valid_json(cfg):
    """Live (Gemma 4 in LM Studio): a call cut off mid-string was sent back as is and every later
    request got HTTP 500."""
    from jevh.worker import valid_args
    broken = '{"code":"print(\\"hi\\")\n# Let\'s check the 3s at (3,2), (9,'
    fixed = valid_args(broken)
    assert json.loads(fixed)["code"].startswith('print("hi")')
    assert valid_args('{"code": "x = 1"}') == '{"code": "x = 1"}'
    llm = NativeLLM([LLMResult(text="", tool_calls=[{"id": "c1", "name": "python", "arguments": '{"code":"x = 2\nprint(x)'}]),
                     call(name="final", answer="done", cid="c2")])
    assert harness(cfg, llm).turn("go", force="high_end_dev").answer == "done"
    json.loads(llm.seen[1][-2]["tool_calls"][0]["function"]["arguments"])     # what went back is valid
