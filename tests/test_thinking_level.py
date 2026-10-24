"""The thinking level (config `thinking.level`, the Settings tab): what it puts in a model's request, that a model's own
setting wins, that the quick side questions still turn thinking off, and the Settings tab saving it. Live:
evals/bench/thinking_level.py."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from jevh import settings as st
from jevh.config import SETTINGS_FILE
from jevh.jev import Jev
from jevh.llm import OpenAICompat, thinking_level, with_thinking
from jevh.orchestrator import Harness

from conftest import FakeSystemOne, ScriptedLLM
from test_settings import conf, fakes  # noqa: F401

PROVIDERS = {"strata": {"type": "openai", "base_url": "http://127.0.0.1:8083/v1"},
             "hosted": {"type": "openai", "base_url": "https://api.example.com/v1"},
             "claude": {"type": "anthropic"}}


@pytest.mark.parametrize("level,extra", [
    ("auto", None),
    ("none", {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}),
    ("low", {"reasoning_effort": "low"}),
    ("medium", {"reasoning_effort": "medium"}),
    ("high", {"reasoning_effort": "high"})])
def test_a_level_becomes_the_requests_reasoning_effort(level, extra):
    out = with_thinking({"provider": "strata", "model": "q"}, level, PROVIDERS)
    assert out.get("extra") == extra


def test_a_hosted_server_gets_no_local_only_field_and_a_claude_model_gets_its_effort():
    assert with_thinking({"provider": "hosted", "model": "o"}, "none", PROVIDERS)["extra"] == {"reasoning_effort": "none"}
    assert with_thinking({"provider": "claude", "model": "c"}, "low", PROVIDERS)["effort"] == "low"
    assert with_thinking({"provider": "claude", "model": "c"}, "none", PROVIDERS)["thinking"] is False


def test_a_models_own_setting_wins():
    own = {"provider": "strata", "model": "q", "extra": {"reasoning_effort": "low"}}
    assert with_thinking(own, "high", PROVIDERS) == own
    off = {"provider": "strata", "model": "q", "extra": {"chat_template_kwargs": {"enable_thinking": False}}}
    assert with_thinking(off, "high", PROVIDERS) == off
    assert with_thinking({"provider": "claude", "model": "c", "effort": "max"}, "low", PROVIDERS)["effort"] == "max"


def test_an_unknown_level_is_auto():
    assert thinking_level({"thinking": {"level": "MEDIUM"}}) == "medium"
    assert thinking_level({}) == "auto" and thinking_level({"thinking": "bogus"}) == "auto"


class Server:
    """An OpenAI-style server on localhost that records each request body and answers "ok"."""

    def __init__(self):
        self.bodies = []
        me = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):  # noqa: N802
                me.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                chunk = {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
                data = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"


def test_a_turn_sends_the_level_and_a_quick_side_question_still_turns_thinking_off(cfg):
    srv = Server()
    try:
        cfg.providers = {**cfg.providers, "local": {"type": "openai", "base_url": srv.url}}
        cfg.models = {"strong": {"provider": "local", "model": "q"}}
        cfg.raw = {**cfg.raw, "thinking": {"level": "medium"}}
        h = Harness(cfg, jev=Jev(cfg, FakeSystemOne()))
        h.learning = False
        assert h.turn("hi", force="daily_assistant").answer == "ok"
        assert srv.bodies[-1]["reasoning_effort"] == "medium"
        # a quick side question (light Jev's checks, the Newsroom's summaries) lays thinking-off over it
        h._check_llm().chat([{"role": "user", "content": "yes or no?"}],
                            body_extra={"max_tokens": 100, "chat_template_kwargs": {"enable_thinking": False}})
        assert srv.bodies[-1]["chat_template_kwargs"] == {"enable_thinking": False}
        # changed in Settings: the next client is built with the new level
        cfg.raw = {**cfg.raw, "thinking": {"level": "auto"}}
        h.reload_settings()
        h.turn("hi again", force="daily_assistant")
        assert "reasoning_effort" not in srv.bodies[-1]
    finally:
        srv.srv.shutdown()


def test_the_settings_tab_saves_the_level_and_reset_puts_auto_back(conf):
    assert st.view(conf)["thinking"] == "auto"
    v = st.save(conf, {"thinking": "low"})
    assert v["thinking"] == "low" and thinking_level(conf.raw) == "low"
    assert "level: low" in (conf.workspace / SETTINGS_FILE).read_text(encoding="utf-8")
    with pytest.raises(st.SettingsError):
        st.save(conf, {"thinking": "extreme"})
    assert st.view(conf)["thinking"] == "low"
    assert st.reset(conf)["thinking"] == "auto"


# ---------------------------------------------------------------- the selector, in a real browser
pytest.importorskip("playwright.sync_api")
pytest.importorskip("uvicorn")
from test_dashboard_ui import REPLIES, browse, serve  # noqa: E402,F401


@pytest.fixture
def dashboard(conf):
    yield from serve(conf, REPLIES)


def test_the_settings_tab_has_a_thinking_selector_that_saves(dashboard, conf, browse):
    def flow(page):
        page.goto(dashboard)
        page.wait_for_selector(".hello")
        page.click('.rail [data-page="settings"]')
        page.wait_for_selector("#p-settings select[data-set=thinking]")
        assert page.input_value("select[data-set=thinking]") == "auto"
        assert page.eval_on_selector_all("select[data-set=thinking] option", "els => els.map(e => e.value)") == \
            ["auto", "none", "low", "medium", "high"]
        sent = []
        page.on("request", lambda r: sent.append(r.post_data) if r.method == "POST" and r.url.endswith("/api/config") else None)
        page.select_option("select[data-set=thinking]", "medium")
        page.click('[data-set-act="save"]')
        page.wait_for_function("document.getElementById('set-msg').textContent.startsWith('Saved')")
        assert json.loads(sent[-1]) == {"thinking": "medium"}                     # only what changed
        assert "level: medium" in (conf.workspace / SETTINGS_FILE).read_text(encoding="utf-8")
        page.reload()
        page.click('.rail [data-page="settings"]')
        page.wait_for_selector("#p-settings select[data-set=thinking]")
        assert page.input_value("select[data-set=thinking]") == "medium"
    browse(flow)
