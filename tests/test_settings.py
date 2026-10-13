"""The Settings tab's engine (jevh/settings.py): the layer it saves into, what it validates, the models it lists, the web search
choices, and a harness picking the new choices up. No browser here (tests/test_settings_ui.py has one); the servers are real
local ones, so the requests that leave are the requests a real server would get."""
import json
import os
import shutil
import stat
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest
import yaml

from jevh import settings as st
from jevh.config import ROOT, SETTINGS_FILE, load_config, read_settings
from jevh.jev import Jev
from jevh.orchestrator import Harness
from jevh.tools import web

from conftest import FakeSystemOne, ScriptedLLM


class Fake:
    """A local HTTP server: `routes` maps a path to (status, body); `seen` lists what it was asked, headers included."""

    def __init__(self, routes):
        self.routes, self.seen = routes, []
        me = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):  # noqa: N802
                u = urlparse(self.path)
                me.seen.append({"path": u.path, "query": parse_qs(u.query), "headers": dict(self.headers)})
                status, body = me.routes.get(u.path, (404, {}))
                raw = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.srv.server_address[1]}"

    def stop(self):
        self.srv.shutdown()


@pytest.fixture
def fakes():
    made = []

    def make(routes):
        f = Fake(routes)
        made.append(f)
        return f
    yield make
    for f in made:
        f.stop()


SEARX = {"/search": (200, {"results": [{"title": "Llama.cpp", "url": "https://ex.test/a", "content": "an <b>engine</b>"},
                                         {"title": "no link", "content": "dropped"},
                                         {"title": "Second", "url": "https://ex.test/b", "content": "more"}]})}
BRAVE = {"/res/v1/web/search": (200, {"web": {"results": [{"title": "Brave <b>one</b>", "url": "https://ex.test/1",
                                                          "description": "about <strong>things</strong>"}]}})}


@pytest.fixture
def conf(tmp_path, monkeypatch, fakes):
    """A config of its own (providers pointing at fake local servers), loaded the way a real start loads it: the layers on,
    no home folder, this machine's real config.local.yaml out of it."""
    monkeypatch.delenv("JEVH_NO_LOCAL_CONFIG", raising=False)
    monkeypatch.setenv("JEVH_HOME", str(tmp_path / "no-home"))
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    lms = fakes({"/api/v1/models": (200, {"models": [
        {"key": "qwen3.8-27b", "type": "llm", "loaded_instances": [{"id": "qwen3.8-27b"}]},
        {"key": "ornith", "type": "llm", "loaded_instances": []},
        {"key": "nomic-embed", "type": "embedding", "loaded_instances": []}]})})
    llama = fakes({"/v1/models": (200, {"data": [{"id": "qwen3.8-27b-q5"}]})})
    base = tmp_path / "config.yaml"
    real = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))          # the shipped limits, thresholds, tools...
    real.update({
        "workspace": "ws", "personas_dir": "personas", "personas_dirs": [], "mcp_servers": {},
        "verify": {**(real.get("verify") or {}), "enabled": False}, "failures": {**(real.get("failures") or {}), "audit": 0},
        "models": {"fast": {"provider": "lmstudio", "model": "ornith"}},
        "providers": {"lmstudio": {"type": "openai", "base_url": lms.url + "/v1", "exclusive": True},
                      "llamacpp": {"type": "openai", "base_url": llama.url + "/v1"},
                      "down": {"type": "openai", "base_url": "http://127.0.0.1:9/v1"},
                      "openai": {"type": "openai", "base_url": "https://api.openai.com/v1"}}})
    real.pop("search", None)
    base.write_text(yaml.safe_dump(real), encoding="utf-8")
    st._discovered.clear()
    shutil.copytree(ROOT / "personas", tmp_path / "personas", ignore=shutil.ignore_patterns("learned"))
    cfg = load_config(base)
    cfg.skills_dirs = [ROOT / "skills"]
    cfg.llm_fakes = (lms, llama)
    return cfg


# ---------------------------------------------------------------- the layer

def test_the_saved_file_wins_over_every_config_file_and_null_removes_what_they_set(conf, tmp_path):
    (tmp_path / "config.local.yaml").write_text(yaml.safe_dump({"models": {"strong": {"provider": "llamacpp", "model": "local-q",
                                                                                      "temperature": 0.2}}}), encoding="utf-8")
    cfg = load_config(conf.source)
    assert cfg.models["strong"]["model"] == "local-q" and cfg.models["fast"]["model"] == "ornith"
    st.save(cfg, {"model": "lmstudio:qwen3.8-27b", "fast": ""})
    again = load_config(conf.source)
    assert again.models["strong"] == {"provider": "lmstudio", "model": "qwen3.8-27b", "temperature": 0.2}   # merged over, not replaced
    assert "fast" not in again.models                                                                      # "" saved as null: off
    st.save(cfg, {"model": ""})                                                                            # each assistant's own
    assert "strong" not in load_config(conf.source).models


def test_the_environment_switch_that_keeps_this_machines_files_out_also_keeps_the_saved_file_out(conf, monkeypatch):
    st.save(conf, {"fast": ""})
    assert "fast" not in load_config(conf.source).models
    monkeypatch.setenv("JEVH_NO_LOCAL_CONFIG", "1")
    assert load_config(conf.source).models["fast"]["model"] == "ornith"


def test_a_broken_or_foreign_settings_file_never_stops_the_harness_and_only_its_sections_count(conf, tmp_path, capsys):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    (ws / SETTINGS_FILE).write_text("models: [unclosed\n  - {", encoding="utf-8")
    assert load_config(conf.source).models["fast"]["model"] == "ornith"
    assert "ignoring" in capsys.readouterr().err
    (ws / SETTINGS_FILE).write_text(yaml.safe_dump({"workspace": "/elsewhere", "thresholds": {"tool": 0.01},
                                                    "search": {"provider": "searxng", "searxng_url": "http://x:1"}}), encoding="utf-8")
    cfg = load_config(conf.source)
    assert cfg.workspace == tmp_path / "ws" and cfg.thresholds["tool"] != 0.01          # workspace and thresholds are not its to set
    assert cfg.raw["search"]["provider"] == "searxng"                                   # search is
    assert read_settings(tmp_path / "does-not-exist") == {}


# ---------------------------------------------------------------- saving

def test_saving_writes_a_private_file_loads_it_into_the_config_and_never_hands_the_key_back(conf, tmp_path):
    v = st.save(conf, {"model": "llamacpp:qwen3.8-27b-q5", "search": {"provider": "brave", "brave_api_key": "BSA-secret"}})
    assert v["model"] == "llamacpp:qwen3.8-27b-q5" and v["fast"] == "lmstudio:ornith"
    assert v["search"] == {"provider": "brave", "searxng_url": "", "brave_key": True} and v["saved"]
    assert "BSA-secret" not in json.dumps(v)
    f = tmp_path / "ws" / SETTINGS_FILE
    assert conf.models["strong"] == {"provider": "llamacpp", "model": "qwen3.8-27b-q5"}      # the live config has it already
    assert yaml.safe_load(f.read_text(encoding="utf-8"))["search"]["brave_api_key"] == "BSA-secret"
    if os.name != "nt":
        assert stat.S_IMODE(f.stat().st_mode) == 0o600
    assert not [p for p in f.parent.iterdir() if p.name.startswith(".settings-")]               # no temp file left


def test_a_saved_brave_key_is_kept_unless_replaced_or_cleared(conf):
    st.save(conf, {"search": {"provider": "brave", "brave_api_key": "k1"}})
    st.save(conf, {"search": {"provider": "brave"}})                                          # the page left the key box empty
    assert conf.raw["search"]["brave_api_key"] == "k1"
    st.save(conf, {"search": {"provider": "duckduckgo"}})                                     # switching engine keeps it for later
    assert conf.raw["search"]["brave_api_key"] == "k1" and conf.raw["search"]["provider"] == "duckduckgo"
    st.save(conf, {"search": {"provider": "duckduckgo", "brave_api_key": "k2"}})
    assert conf.raw["search"]["brave_api_key"] == "k2"
    st.save(conf, {"search": {"provider": "duckduckgo", "brave_api_key": ""}})
    assert "brave_api_key" not in conf.raw["search"]


@pytest.mark.parametrize("change, message", [
    ({"model": "nowhere:some-model"}, "provider:model"),
    ({"model": "lmstudio"}, "provider:model"),
    ({"model": "lmstudio:two words"}, "not a model name"),
    ({"fast": "just-a-name"}, "Quick model"),
    ({"search": {"provider": "askjeeves"}}, "choose one of"),
    ({"search": {"provider": "searxng"}}, "needs its server address"),
    ({"search": {"provider": "searxng", "searxng_url": "ftp://x"}}, "http://host:port"),
    ({"search": {"provider": "brave"}}, "needs an API key"),
    ({"search": {"provider": "brave", "brave_api_key": "two words"}}, "does not look like a key"),
    ({"theme": "dark"}, "unknown setting"),
])
def test_a_choice_that_cannot_work_is_refused_with_what_to_change_and_nothing_is_written(conf, tmp_path, change, message):
    with pytest.raises(st.SettingsError, match=message):
        st.save(conf, change)
    assert not (tmp_path / "ws" / SETTINGS_FILE).exists()


def test_reset_goes_back_to_what_the_config_files_say(conf, tmp_path):
    st.save(conf, {"model": "llamacpp:qwen3.8-27b-q5", "fast": "", "search": {"provider": "searxng", "searxng_url": "http://h:1/"}})
    assert conf.raw["search"]["searxng_url"] == "http://h:1"
    v = st.reset(conf)
    assert not (tmp_path / "ws" / SETTINGS_FILE).exists() and not v["saved"]
    assert v["model"] == "" and v["fast"] == "lmstudio:ornith" and v["search"]["provider"] == "duckduckgo"
    st.reset(conf)                                                                          # nothing to reset is not an error


# ---------------------------------------------------------------- the models on offer

def test_the_models_on_offer_are_what_the_local_servers_say_and_a_cloud_provider_is_not_asked(conf):
    o = st.model_options(conf, fresh=True)
    by = {m["value"]: m for m in o["models"]}
    assert set(by) == {"lmstudio:qwen3.8-27b", "lmstudio:ornith", "llamacpp:qwen3.8-27b-q5"}      # no embedding model
    assert by["lmstudio:qwen3.8-27b"]["loaded"] is True and by["lmstudio:ornith"]["loaded"] is False
    assert by["llamacpp:qwen3.8-27b-q5"]["loaded"] is None
    assert o["down"] == ["down"]                                                               # asked, did not answer; openai: not asked
    lms, llama = conf.llm_fakes
    n = len(lms.seen)
    st.model_options(conf)                                                                     # within the cache time: no new request
    assert len(lms.seen) == n


# ---------------------------------------------------------------- web search

def test_a_searxng_search_asks_for_json_and_reads_the_results(conf, fakes):
    srv = fakes(SEARX)
    hits = web.search_with({"JEVH_SEARCH_PROVIDER": "searxng", "JEVH_SEARXNG_URL": srv.url + "/"}, "llama cpp", 5)
    assert hits == [{"title": "Llama.cpp", "url": "https://ex.test/a", "snippet": "an engine"},
                    {"title": "Second", "url": "https://ex.test/b", "snippet": "more"}]
    assert srv.seen[0]["query"] == {"q": ["llama cpp"], "format": ["json"]}
    assert len(web.search_with({"JEVH_SEARCH_PROVIDER": "searxng", "JEVH_SEARXNG_URL": srv.url}, "x", 1)) == 1


def test_searxng_failures_say_what_to_do(fakes):
    def go(url):
        return web.search_with({"JEVH_SEARCH_PROVIDER": "searxng", "JEVH_SEARXNG_URL": url}, "x")
    with pytest.raises(web.SearchError, match="allow the json format"):
        go(fakes({"/search": (403, {})}).url)
    with pytest.raises(web.SearchError, match="not reachable"):
        go("http://127.0.0.1:9")
    with pytest.raises(web.SearchError, match="not with JSON"):
        go(fakes({"/search": (200, b"<html>hello</html>")}).url)
    with pytest.raises(web.SearchError, match="address is not set"):
        web.search_with({"JEVH_SEARCH_PROVIDER": "searxng"}, "x")


def test_a_brave_search_sends_the_key_and_reads_the_results(fakes, monkeypatch):
    srv = fakes(BRAVE)
    monkeypatch.setattr(web, "BRAVE_URL", srv.url + "/res/v1/web/search")
    hits = web.search_with({"JEVH_SEARCH_PROVIDER": "brave", "JEVH_BRAVE_KEY": "BSA-1"}, "rust", 3)
    assert hits == [{"title": "Brave one", "url": "https://ex.test/1", "snippet": "about things"}]
    seen = srv.seen[0]
    assert seen["query"] == {"q": ["rust"], "count": ["3"]} and seen["headers"]["X-Subscription-Token"] == "BSA-1"
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BSA-env")                                    # no key saved: the environment's
    web.search_with({"JEVH_SEARCH_PROVIDER": "brave"}, "rust")
    assert srv.seen[1]["headers"]["X-Subscription-Token"] == "BSA-env"


def test_brave_failures_say_what_to_do(fakes, monkeypatch):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    with pytest.raises(web.SearchError, match="no API key"):
        web.search_with({"JEVH_SEARCH_PROVIDER": "brave"}, "x")
    for status, words in ((401, "refused"), (429, "limit")):
        monkeypatch.setattr(web, "BRAVE_URL", fakes({"/res/v1/web/search": (status, {})}).url + "/res/v1/web/search")
        with pytest.raises(web.SearchError, match=words):
            web.search_with({"JEVH_SEARCH_PROVIDER": "brave", "JEVH_BRAVE_KEY": "k"}, "x")


def test_the_engine_is_duckduckgo_until_one_is_chosen_and_an_unknown_one_is_an_error(monkeypatch):
    called = []
    monkeypatch.setattr(web, "_search_ddg", lambda q, n: called.append(q) or [])
    monkeypatch.delenv("JEVH_SEARCH_PROVIDER", raising=False)
    web.search("anything")
    monkeypatch.setenv("JEVH_SEARCH_PROVIDER", "")          # empty is "not set" too
    web.search("again")
    assert called == ["anything", "again"]
    with pytest.raises(web.SearchError, match="unknown web search provider"):
        web.search_with({"JEVH_SEARCH_PROVIDER": "askjeeves"}, "x")
    assert web.search_env(None) == {"JEVH_SEARCH_PROVIDER": "duckduckgo"}


def test_web_ask_says_why_it_is_empty_when_the_chosen_engine_cannot_search():
    with web.using({"JEVH_SEARCH_PROVIDER": "searxng"}):                                       # no address set
        out = web.ask("who made it?")
    assert out["passages"] == [] and "address is not set" in out["note"]


def test_the_test_button_runs_one_real_search_with_the_choice_on_the_page(conf, fakes):
    srv = fakes(SEARX)
    ok = st.test_search(conf, {"provider": "searxng", "searxng_url": srv.url})
    assert ok["ok"] and ok["count"] == 2 and ok["first"] == "Llama.cpp" and srv.seen[0]["query"]["format"] == ["json"]
    assert not (conf.workspace / SETTINGS_FILE).exists()                                       # trying is not saving
    bad = st.test_search(conf, {"provider": "searxng", "searxng_url": "http://127.0.0.1:9"})
    assert not bad["ok"] and "not reachable" in bad["error"]
    assert "needs its server address" in st.test_search(conf, {"provider": "searxng"})["error"]
    st.save(conf, {"search": {"provider": "brave", "brave_api_key": "saved-key"}})
    brave = fakes(BRAVE)
    web.BRAVE_URL, was = brave.url + "/res/v1/web/search", web.BRAVE_URL
    try:
        assert st.test_search(conf, {"provider": "brave"})["ok"]                               # a blank key box means the saved one
        assert brave.seen[0]["headers"]["X-Subscription-Token"] == "saved-key"
    finally:
        web.BRAVE_URL = was


# ---------------------------------------------------------------- a harness using what was saved

def test_a_harness_builds_its_clients_and_workers_from_the_new_choices_after_reload_settings(conf, tmp_path):
    built = []
    h = Harness(conf, jev=Jev(conf, FakeSystemOne()), llm_factory=lambda p: built.append(p.model) or ScriptedLLM(["ok"]))
    persona = h.personas[conf.default_persona]
    h._llm(persona, "strong")
    assert built[-1].get("model") != "qwen3.8-27b-q5" and h.tool_env["JEVH_SEARCH_PROVIDER"] == "duckduckgo"
    st.save(conf, {"model": "llamacpp:qwen3.8-27b-q5", "search": {"provider": "searxng", "searxng_url": "http://h:8888"}})
    h.reload_settings()
    h._llm(persona, "strong")
    assert built[-1]["provider"] == "llamacpp" and built[-1]["model"] == "qwen3.8-27b-q5"       # a new client, on the new model
    assert h._model_cfg(persona, "fast") == {**{k: v for k, v in persona.model.items() if k not in
                                                ("fallback", "effort", "thinking", "think", "max_tokens")},
                                             "provider": "lmstudio", "model": "ornith"}
    assert h.tool_env["JEVH_SEARCH_PROVIDER"] == "searxng" and h.tool_env["JEVH_SEARXNG_URL"] == "http://h:8888"
    st.save(conf, {"fast": ""})                                                                  # quick model off: easy tasks use the main one
    h.reload_settings()
    assert not conf.models.get("fast") and h._model_cfg(persona, "fast")["model"] == "qwen3.8-27b-q5"


# ---------------------------------------------------------------- through the dashboard's API

def api(conf, delay=0.0, model_override=None):
    from fastapi.testclient import TestClient

    from jevh.web.server import create_app
    from test_web import BASE, TOKEN, auth, make_client  # noqa: F401
    client, app = make_client(conf, delay=delay)
    if model_override:
        app.state.hub.model_override = model_override
    return client, app, auth


def test_the_api_shows_the_choices_and_what_can_be_picked_and_never_the_key(conf):
    client, app, auth = api(conf)
    r = client.post("/api/config", json={"search": {"provider": "brave", "brave_api_key": "BSA-secret"}}, headers=auth())
    assert r.status_code == 200
    got = client.get("/api/config", headers=auth()).json()
    assert got["search"] == {"provider": "brave", "searxng_url": "", "brave_key": True}
    assert "BSA-secret" not in json.dumps(got) and "BSA-secret" not in r.text
    assert {"lmstudio:qwen3.8-27b", "llamacpp:qwen3.8-27b-q5"} <= {m["value"] for m in got["options"]["models"]}
    assert [p["id"] for p in got["providers"]] == ["duckduckgo", "searxng", "brave"]
    assert got["fast"] == "lmstudio:ornith" and got["saved"] and got["busy"] is False and got["model_override"] is None


def test_saving_through_the_api_changes_what_the_running_harness_does(conf):
    client, app, auth = api(conf)
    h = app.state.hub.harness
    r = client.post("/api/config", json={"model": "llamacpp:qwen3.8-27b-q5", "fast": "",
                                         "search": {"provider": "searxng", "searxng_url": "http://127.0.0.1:8888"}}, headers=auth())
    assert r.status_code == 200 and r.json()["model"] == "llamacpp:qwen3.8-27b-q5" and r.json()["fast"] == ""
    persona = h.personas[conf.default_persona]
    assert h._model_cfg(persona, "strong")["model"] == "qwen3.8-27b-q5" and not conf.models.get("fast")
    assert h.tool_env["JEVH_SEARCH_PROVIDER"] == "searxng"
    reset = client.post("/api/config/reset", headers=auth()).json()
    assert reset["model"] == "" and reset["fast"] == "lmstudio:ornith" and reset["saved"] is False
    assert h.tool_env["JEVH_SEARCH_PROVIDER"] == "duckduckgo"


def test_the_api_refuses_a_bad_choice_in_plain_words_and_a_change_while_a_turn_runs(conf):
    client, app, auth = api(conf, delay=1.0)
    bad = client.post("/api/config", json={"model": "nowhere:x"}, headers=auth())
    assert bad.status_code == 400 and "provider:model" in bad.json()["detail"]
    assert client.post("/api/config", json=[1], headers=auth()).status_code == 400
    client.post("/api/chat", json={"message": "build a page"}, headers=auth())              # a turn is now running
    busy = client.post("/api/config", json={"fast": ""}, headers=auth())
    assert busy.status_code == 409 and "running turn" in busy.json()["detail"]
    assert client.post("/api/config/reset", headers=auth()).status_code == 409
    assert conf.models.get("fast")                                                          # nothing was changed
    from test_web import wait_done
    wait_done(client)


def test_a_dashboard_started_with_a_model_flag_says_the_flag_wins_but_still_takes_a_search_choice(conf):
    client, app, auth = api(conf, model_override="lmstudio:qwen3.8-27b")
    no = client.post("/api/config", json={"model": "llamacpp:qwen3.8-27b-q5"}, headers=auth())
    assert no.status_code == 409 and "--model lmstudio:qwen3.8-27b" in no.json()["detail"]
    ok = client.post("/api/config", json={"search": {"provider": "duckduckgo"}}, headers=auth())
    assert ok.status_code == 200 and ok.json()["model_override"] == "lmstudio:qwen3.8-27b"


def test_the_test_button_endpoint_tries_a_choice_without_saving_it(conf, fakes):
    client, app, auth = api(conf)
    srv = fakes(SEARX)
    got = client.post("/api/config/test-search", json={"provider": "searxng", "searxng_url": srv.url}, headers=auth()).json()
    assert got["ok"] and got["count"] == 2
    assert not (conf.workspace / SETTINGS_FILE).exists()
    assert client.post("/api/config/test-search", json={"provider": "searxng"}, headers=auth()).json()["ok"] is False


def test_a_newsroom_web_widget_looks_with_the_engine_chosen_in_settings(conf, fakes):
    """Its looks run in the dashboard's own process, not in a worker's REPL: they asked DuckDuckGo whatever was chosen."""
    client, app, auth = api(conf)
    hub = app.state.hub
    srv = fakes(SEARX)
    client.post("/api/config", json={"search": {"provider": "searxng", "searxng_url": srv.url}}, headers=auth())
    w = hub.newsroom.add_widget("web", {"query": "llama cpp"})
    first = hub.newsroom.refresh(w["id"])
    assert first["first"] and not first["error"] and len(hub.newsroom.items(w["id"])) == 2
    assert srv.seen and srv.seen[0]["query"]["q"] == ["llama cpp"]
    client.post("/api/config", json={"search": {"provider": "searxng", "searxng_url": "http://127.0.0.1:9"}}, headers=auth())
    broken = hub.newsroom.refresh(w["id"])
    assert "not reachable" in broken["error"]                                              # the engine's own words, on the widget


def test_a_save_takes_effect_even_when_the_workspace_was_changed_after_the_config_was_loaded(conf, tmp_path):
    """Found live: save() wrote to cfg.workspace but the reload read the folder the config files name, so for a config whose
    workspace was moved after loading (every bench, every test fixture) a saved choice silently never took effect."""
    conf.workspace = tmp_path / "moved"
    st.save(conf, {"model": "llamacpp:qwen3.8-27b-q5"})
    assert (tmp_path / "moved" / SETTINGS_FILE).is_file() and not (tmp_path / "ws" / SETTINGS_FILE).exists()
    assert st.view(conf)["model"] == "llamacpp:qwen3.8-27b-q5" and conf.models["strong"]["model"] == "qwen3.8-27b-q5"
    st.reset(conf)
    assert st.view(conf)["model"] == ""


# ---------------------------------------------------------------- the desktop app's chat glass

def test_the_chat_glass_is_saved_at_once_kept_beside_the_other_choices_and_checked(conf, tmp_path):
    assert st.view(conf)["appearance"] == {"orb_glass": st.ORB_GLASS}                      # nothing saved: the look it was made with
    st.save(conf, {"search": {"provider": "searxng", "searxng_url": "http://h:1"}})
    assert st.save_appearance(conf, {"orb_glass": 72.4}) == {"orb_glass": 72}
    saved = read_settings(tmp_path / "ws")
    assert saved["appearance"] == {"orb_glass": 72} and saved["search"]["provider"] == "searxng"   # the other choice stays
    assert load_config(conf.source, settings_dir=conf.workspace).raw["appearance"]["orb_glass"] == 72
    for bad in ({"orb_glass": 101}, {"orb_glass": -1}, {"orb_glass": "lots"}, {"orb_glass": True}, {}, {"glow": 3}):
        with pytest.raises(st.SettingsError):
            st.save_appearance(conf, bad)
    assert st.view(conf)["appearance"] == {"orb_glass": 72}
    conf.raw["appearance"] = {"orb_glass": "nonsense"}                                       # a hand-edited file: the default
    assert st.appearance(conf) == {"orb_glass": st.ORB_GLASS}
    st.reset(conf)
    assert st.view(conf)["appearance"] == {"orb_glass": st.ORB_GLASS}


def test_the_chat_glass_saves_even_while_a_turn_runs_and_every_orb_hears_of_it(conf):
    client, app, auth = api(conf, delay=1.0)
    client.post("/api/chat", json={"message": "build a page"}, headers=auth())              # a turn is now running
    r = client.post("/api/config/appearance", json={"orb_glass": 60}, headers=auth())
    assert r.status_code == 200 and r.json() == {"appearance": {"orb_glass": 60}}
    assert client.get("/api/state", headers=auth()).json()["appearance"] == {"orb_glass": 60}   # the orb's chat page reads this
    assert any(e["kind"] == "appearance" and e["data"] == {"orb_glass": 60} for e in app.state.hub.events)
    bad = client.post("/api/config/appearance", json={"orb_glass": 500}, headers=auth())
    assert bad.status_code == 400 and "0 to 100" in bad.json()["detail"]
    assert client.get("/api/config", headers=auth()).json()["appearance"] == {"orb_glass": 60}
    from test_web import wait_done
    wait_done(client)
