"""Light Jev (jevh/light.py): which decisions the small model keeps, which go to the task's model as one quick question,
which are skipped; the answers the task's model gives in the shape Jev's callers read; the Settings tab turning it on."""
import json
import shutil

import pytest

from jevh import light
from jevh import settings as st
from jevh.config import ROOT, load_config
from jevh.decide import SystemOneClient, SystemOneError, choice, noul, score
from jevh.jev import Jev
from jevh.llm import LLMResult

from conftest import FakeSystemOne

QS = {"claims": noul("Does `answer` claim something was done?"),
      "team": choice("Which team?", {"billing": "money", "tech": None}),
      "level": score("How hard?", ["easy", "routine", "hard"])}


class Talker:
    """A model that answers Jev's questions with `reply`; keeps what it was sent."""

    def __init__(self, reply):
        self.reply, self.seen, self.model = reply, [], "qwen-test"

    def chat(self, messages, cancel=None, body_extra=None):
        self.seen.append({"messages": messages, "body_extra": body_extra})
        return LLMResult(text=self.reply if isinstance(self.reply, str) else json.dumps(self.reply), model=self.model)


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.setenv("JEVH_NO_LOCAL_CONFIG", "1")
    return load_config()


def lit(cfg, checks="worker", **light_cfg):
    cfg.jev = {**cfg.jev, "mode": "light", "light": {**(cfg.jev.get("light") or {}), "checks": checks, **light_cfg}}
    return cfg


def test_full_mode_sends_everything_to_jev_and_light_mode_splits_it():
    full = light.settings({"mode": "full"})
    assert {light.route(s, full) for s in ("verify", "skills", "compact")} == {"jev"}
    s = light.settings({"mode": "light"})
    assert light.route("skills", s) == "jev" and light.route("tools", s) == "jev" and light.route("memory", s) == "jev"
    assert light.route("verify", s) == "worker" and light.route("failure_check", s) == "worker"
    assert light.route("compact", s) == "off" and light.route("thinking_watch", s) == "off"
    off = light.settings({"mode": "light", "light": {"checks": "off"}})
    assert light.route("verify", off) == "off" and light.route("skills", off) == "jev"
    assert light.settings({"mode": "nonsense", "light": {"checks": "?"}}) | {} == {**light.settings({}), "backend": "laya"}


def test_a_handed_off_decision_is_answered_by_the_tasks_model_in_jevs_shape(cfg):
    fake = FakeSystemOne()
    jev = Jev(lit(cfg), fake)
    talker = Talker({"claims": {"answer": "yes", "p": 0.93}, "team": {"answer": "Billing", "p": 0.7},
                     "level": {"answer": "2", "p": 0.6}})
    jev.check_llm = lambda: talker
    r = jev._ask("verify", {"request": "refund me", "answer": "Refunded."}, QS)
    assert not fake.calls                                       # the small model was not asked
    assert r["claims"].noul == 0.93 and r["team"].choice == "billing"
    assert r["team"].probabilities == {"billing": 0.7, "tech": 0.3}
    assert r["level"].score == pytest.approx(0 * 0.2 + 1 * 0.2 + 2 * 0.6) and r.model == "worker:qwen-test"
    sent = talker.seen[0]
    assert "refund me" in sent["messages"][0]["content"] and "option billing: money" in sent["messages"][0]["content"]
    assert sent["body_extra"] == {"max_tokens": light.MAX_TOKENS}   # a quick side question
    assert jev.trace[-1]["step"] == "verify" and jev.trace[-1]["model"] == "worker:qwen-test"
    jev._ask("skills", {"latest_request": "x"}, {"pick": choice("Which?", {"a": None, "b": None})})
    assert len(fake.calls) == 1                                 # a kept decision still goes to the small model


def test_off_or_no_model_skips_the_decision_the_way_jev_down_does(cfg):
    fake = FakeSystemOne()
    jev = Jev(lit(cfg, checks="off"), fake)
    jev.check_llm = lambda: Talker("{}")
    with pytest.raises(SystemOneError, match="not asked"):
        jev._ask("failure_check", {"request": "x"}, {"gave_up": noul("Gave up?")})
    jev = Jev(lit(cfg), fake)                                   # worker, but no model to ask
    with pytest.raises(SystemOneError):
        jev._ask("verify", {"request": "x"}, {"claims": noul("?")})
    with pytest.raises(SystemOneError):                          # skipped even in worker mode: one call per old step
        Jev(lit(cfg), fake)._ask("compact", {"step": "x"}, {"info": noul("?")})
    assert not fake.calls


@pytest.mark.parametrize("reply", ["I think yes", '{"claims": {"answer": "maybe"}, "team": {"answer": "sales"}}',
                                   '{"claims": {"answer": "yes"}}'])
def test_an_unusable_reply_is_a_jev_error_so_the_caller_falls_back(reply):
    with pytest.raises(SystemOneError):
        light.parse(reply, {"claims": QS["claims"], "team": QS["team"]})


def test_a_thinking_reply_and_a_missing_p_still_parse():
    out = light.parse('<think>hmm {"x": 1}</think>\n```json\n{"claims": {"answer": "no"}}\n```', {"claims": QS["claims"]})
    assert out["claims"]["noul"] == 0.1 and out["claims"]["probabilities"]["no"] == 0.9


def test_the_settings_tab_turns_light_jev_on_and_the_harness_jev_follows(tmp_path, monkeypatch):
    # loaded the way a real start loads it (the saved settings on), from a copy, so this machine's own files stay out
    monkeypatch.delenv("JEVH_NO_LOCAL_CONFIG", raising=False)
    monkeypatch.setenv("JEVH_HOME", str(tmp_path / "no-home"))
    shutil.copy(ROOT / "config.yaml", tmp_path / "config.yaml")
    cfg = load_config(tmp_path / "config.yaml")
    jev = Jev(cfg, SystemOneClient("http://127.0.0.1:9", backend="semif"))
    assert st.view(cfg)["jev"] == {"mode": "full", "checks": "worker"}
    st.save(cfg, {"jev": {"mode": "light", "checks": "off"}})
    assert st.view(cfg)["jev"] == {"mode": "light", "checks": "off"}
    assert cfg.jev["url"] and cfg.jev["light"]["backend"] == "laya"     # the files' addresses and backends stay
    jev.reload()
    assert jev.client.backend == "laya" and light.route("verify", jev.light) == "off"
    st.save(cfg, {"jev": {"mode": "full"}})
    jev.reload()
    assert jev.client.backend == cfg.jev["backend"] and jev.light["mode"] == "full"
    with pytest.raises(st.SettingsError):
        st.save(cfg, {"jev": {"mode": "turbo"}})
    with pytest.raises(st.SettingsError):
        st.save(cfg, {"jev": {"checks": "maybe"}})


def test_a_skipped_decision_falls_back_without_reporting_an_outage(cfg, tmp_path):
    from jevh.orchestrator import Harness
    from conftest import ScriptedLLM
    cfg.workspace = tmp_path / "ws"
    events = []
    h = Harness(lit(cfg, checks="off"), jev=Jev(cfg, FakeSystemOne()), llm_factory=lambda p: ScriptedLLM(["ok"]),
                on_event=lambda k, d: events.append(k))
    assert h._safe("follow_up", lambda: h.jev.follow_up("and tomorrow?", "user: weather today?"), "fb") == "fb"
    assert "jev_error" not in events and not h.failures.list()
    h.jev.client = type("Down", (), {"ask": lambda self, s, q: (_ for _ in ()).throw(SystemOneError("down"))})()
    assert h._safe("skills", lambda: h.jev._ask("skills", {}, {"q": noul("?")}), "fb") == "fb"
    assert "jev_error" in events                                   # a real outage is still reported
    assert h._check_llm() is not None                              # the model handed-off decisions would go to


def test_the_quick_question_gets_its_budget_through_the_usage_wrapper_and_no_thinking_on_a_local_server(cfg):
    from jevh.usage import Metered

    class Local(Talker):
        base_url = "http://127.0.0.1:8080/v1"

    seen = {}
    for name, inner in (("local", Local({"claims": {"answer": "no"}})), ("hosted", Talker({"claims": {"answer": "no"}}))):
        inner.base_url = getattr(inner, "base_url", "https://api.example.com/v1")
        llm = Metered(inner, lambda **k: None, lambda: {}, "strong", "m")
        light.ask(llm, {"request": "x"}, {"claims": QS["claims"]}, light.settings({"mode": "light"}))
        seen[name] = inner.seen[0]["body_extra"]
    assert seen["local"] == {"max_tokens": light.MAX_TOKENS, "chat_template_kwargs": {"enable_thinking": False}}
    assert seen["hosted"] == {"max_tokens": light.MAX_TOKENS}


def test_light_mode_has_thresholds_of_its_own_and_full_mode_ignores_them(cfg):
    cfg.jev = {**cfg.jev, "light": {"thresholds": {"tier_strong": 0.5}}}
    base = cfg.thresholds.get("tier_strong", 0.55)
    assert cfg.t("tier_strong") == base                       # full mode: semif's
    cfg.jev["mode"] = "light"
    assert cfg.t("tier_strong") == 0.5 and cfg.t("tool") == cfg.thresholds.get("tool", 0.65)


def test_a_yes_no_answer_and_its_p_always_agree():
    q = {"claims": QS["claims"]}
    assert light.parse('{"claims": {"answer": "no", "p": 0.9}}', q)["claims"]["noul"] == pytest.approx(0.1)   # confidence
    assert light.parse('{"claims": {"answer": "no", "p": 0.2}}', q)["claims"]["noul"] == pytest.approx(0.2)   # P(yes)
    assert light.parse('{"claims": {"answer": "yes", "p": 0.3}}', q)["claims"]["noul"] == pytest.approx(0.7)


def test_a_handed_off_check_whose_model_is_down_falls_back_like_jev_down(cfg, tmp_path):
    """The task's model unreachable or failing during a quick question: a Jev error the caller falls back on (reported
    as an outage), not a model error that ends the turn; a Stop still stops."""
    from jevh.errors import Cancelled
    from jevh.llm import LLMError
    from jevh.orchestrator import Harness
    from conftest import ScriptedLLM

    class Down(Talker):
        def chat(self, messages, cancel=None, body_extra=None):
            raise self.reply

    jev = Jev(lit(cfg), FakeSystemOne())
    jev.check_llm = lambda: Down(LLMError("qwen@http://127.0.0.1:8083/v1: unreachable"))
    with pytest.raises(SystemOneError, match="could not answer"):
        jev._ask("failure_check", {"request": "x"}, {"gave_up": noul("Gave up?")})
    jev.check_llm = lambda: Down(Cancelled())
    with pytest.raises(Cancelled):
        jev._ask("failure_check", {"request": "x"}, {"gave_up": noul("Gave up?")})
    cfg.workspace = tmp_path / "ws"
    events = []
    h = Harness(lit(cfg), jev=Jev(cfg, FakeSystemOne()), llm_factory=lambda p: ScriptedLLM(["ok"]),
                on_event=lambda k, d: events.append(k))
    h.jev.check_llm = lambda: Down(LLMError("unreachable"))
    assert h._safe("verify", lambda: h.jev._ask("verify", {"request": "x"}, {"c": noul("?")}), "fb") == "fb"
    assert "jev_error" in events
