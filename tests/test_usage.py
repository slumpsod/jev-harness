"""Usage tracking: model calls, Jev calls, tools' Jev calls, per turn / task / persona."""
from jevh.jev import Jev
from jevh.llm import LLMResult
from jevh.orchestrator import Harness
from jevh.usage import Usage, brief

from conftest import FakeSystemOne, ScriptedLLM, choice_ans, noul_ans, py


class CountingLLM(ScriptedLLM):
    """Reports token counts like LM Studio does."""

    def chat(self, messages):
        r = super().chat(messages)
        return LLMResult(text=r.text, input_tokens=1000, output_tokens=300, model="qwen-test", thinking_tokens=250)


def rules(k, q, s):
    if k == "persona":
        return choice_ans("daily_assistant", 0.9, q["criteria"])
    if k in ("needs_high_end_dev", "in_scope", "necessary"):
        return noul_ans(0.9)
    if k == "complexity":
        return {"score": 1.8, "confidence": 0.9}


def test_turn_usage_counts_models_and_jev(cfg):
    llms = {"daily_assistant": CountingLLM([py("r = request('high_end_dev', 'build it')\nfinal('ok ' + r['answer'])")]),
            "high_end_dev": CountingLLM([py("final('built')")])}
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: llms[p.name])
    h.learning = False
    r = h.turn("build me an app")
    k = r.usage["by_kind"]
    assert k["llm"]["calls"] == 2 and k["llm"]["input"] == 2000 and k["llm"]["thinking"] == 500
    assert k["jev"]["calls"] >= 5 and set(r.usage["by_persona"]) >= {"daily_assistant", "high_end_dev"}
    line = brief(r.usage)
    assert line.startswith("2.0k in / 600 out (500 thinking)") and "Jev" in line
    h.turn("again")  # a second turn is counted separately
    assert h.usage.summary(h.session, 1)["by_kind"]["llm"]["calls"] == 2
    assert h.usage.summary(h.session)["turns"] == 2


def test_tools_jev_calls_inside_the_repl_are_reported(cfg):
    # the worker's own code calls Jev through a SystemOneClient inside the REPL (like the browser tool)
    code = ("from jevh.decide import SystemOneClient, TOTALS\n"
            "TOTALS['calls'] += 3; TOTALS['seconds'] += 1.5; TOTALS['tokens'] += 900\n"
            "final('done')")
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: CountingLLM([py(code)]))
    h.learning = False
    r = h.turn("browse something")
    t = r.usage["by_kind"]["jev_tools"]
    assert t["calls"] == 3 and t["input"] == 900 and t["seconds"] == 1.5


def test_prices_turn_into_cost(tmp_path):
    u = Usage(tmp_path / "u.db", {"claude-opus-5": {"input": 5.0, "output": 25.0}})
    u.record(session="s", turn=1, kind="llm", model="claude-opus-5", input_tokens=10_000, output_tokens=2_000)
    u.record(session="s", turn=1, kind="llm", model="qwen3.8-27b@q6_k", input_tokens=10_000, output_tokens=2_000)
    s = u.summary("s", 1)
    assert s["total"]["cost"] == 0.1 and s["by_model"]["qwen3.8-27b@q6_k"]["cost"] == 0
    assert u.turns("s")[0]["cost"] == 0.1
    u.close()
