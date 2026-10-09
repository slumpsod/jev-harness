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


# -- tokens per second --------------------------------------------------------------------------------------------

def test_a_rate_counts_only_calls_whose_decode_time_is_known_and_leaves_the_prompt_reading_out(tmp_path):
    u = Usage(tmp_path / "u.db")
    u.record(session="s", turn=1, kind="llm", model="local", input_tokens=9000, output_tokens=600, seconds=15.0, decode_seconds=10.0)
    u.record(session="s", turn=1, kind="llm", model="hosted", input_tokens=500, output_tokens=2000, seconds=20.0)   # reports no decode time
    s = u.summary("s", 1)
    assert s["by_kind"]["llm"]["decode_tokens"] == 600 and s["by_kind"]["llm"]["decode_seconds"] == 10.0
    line = brief(s)
    assert "60 tok/s" in line and line.index("model 35s") < line.index("60 tok/s")       # 600 / 10, not 2600 / 35
    u.record(session="s", turn=2, kind="llm", model="hosted", output_tokens=100, seconds=2.0)
    assert "tok/s" not in brief(u.summary("s", 2))                                         # nothing known: no number made up


def test_a_usage_db_from_before_rates_were_measured_is_upgraded_and_keeps_its_rows(tmp_path):
    import sqlite3
    db = sqlite3.connect(tmp_path / "old.db")
    db.execute("CREATE TABLE usage(ts REAL, session TEXT, turn INTEGER, task INTEGER, persona TEXT, kind TEXT, model TEXT, "
               "tier TEXT, calls INTEGER, input INTEGER, output INTEGER, thinking INTEGER, seconds REAL, cost REAL)")
    db.execute("INSERT INTO usage VALUES (1,'s',1,1,'p','llm','m','',1,100,50,0,2.0,0)")
    db.commit()
    db.close()
    u = Usage(tmp_path / "old.db")
    assert u.summary("s", 1)["total"]["output"] == 50 and "tok/s" not in brief(u.summary("s", 1))
    u.record(session="s", turn=2, kind="llm", output_tokens=40, seconds=1.0, decode_seconds=0.5)
    assert "80 tok/s" in brief(u.summary("s", 2))


def test_a_turn_reports_its_rate_from_the_model_calls_decode_time(cfg):
    class Timed(ScriptedLLM):
        def chat(self, messages):
            r = super().chat(messages)
            return LLMResult(text=r.text, input_tokens=2000, output_tokens=300, model="qwen-test", decode_seconds=5.0)
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: Timed([py("final('ok')")]))
    h.learning = False
    assert "60 tok/s" in brief(h.turn("hello").usage)


def test_the_live_rate_a_client_reports_goes_out_as_a_progress_event(cfg):
    class Streaming(ScriptedLLM):
        on_progress = None

        def chat(self, messages):
            self.on_progress(50, 3, 42.04)
            self.on_progress(50, 3, 0.0)
            return super().chat(messages)
    seen = []
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: Streaming([py("final('ok')")]),
                on_event=lambda kind, data: seen.append((kind, data)))
    h.learning = False
    h.turn("hello")
    assert [d["tps"] for k, d in seen if k == "progress"][:2] == [42.0, 0.0]
    assert {"persona": "daily_assistant", "chunks": 50, "thinking": 3, "tps": 42.0} in [d for k, d in seen if k == "progress"]
