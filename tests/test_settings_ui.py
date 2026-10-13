"""The Settings tab in a real browser: what it shows, that it only sends what changed, that an engine's own fields appear, that
Test and Save and Reset work, and that the key never comes back to the page. Skipped when Playwright's Chromium isn't installed."""
import pytest
import yaml

pytest.importorskip("playwright.sync_api")
pytest.importorskip("uvicorn")

from test_dashboard_ui import REPLIES, browse, serve  # noqa: E402,F401
from test_settings import SEARX, conf, fakes  # noqa: E402,F401
from jevh.config import SETTINGS_FILE  # noqa: E402


@pytest.fixture
def dashboard(conf):
    yield from serve(conf, REPLIES)


def test_the_settings_tab_picks_models_tries_a_search_and_saves_only_what_changed(dashboard, conf, fakes, browse):
    srv = fakes(SEARX)
    browse(lambda page: _flow(page, dashboard, conf, srv))


def _flow(page, url, conf, srv):
    page.on("dialog", lambda d: d.accept())
    saved = conf.workspace / SETTINGS_FILE
    posts = []
    page.on("request", lambda r: posts.append(r.post_data) if r.method == "POST" and "/api/config" in r.url and "test" not in r.url else None)
    page.goto(url)
    page.wait_for_selector(".hello")
    page.click('#tabs [data-tab="settings"]')
    assert page.inner_text("#ptitle") == "Settings"
    page.wait_for_selector("#p-settings select[data-set=model]")
    # what is in force now: no main model (each assistant's own), the quick model from the config, DuckDuckGo
    assert page.input_value("select[data-set=model]") == "" and page.input_value("select[data-set=fast]") == "lmstudio:ornith"
    assert page.input_value("select[data-set=search]") == "duckduckgo"
    models = page.eval_on_selector_all("select[data-set=model] option", "els => els.map(e => e.value)")
    assert "llamacpp:qwen3.8-27b-q5" in models and "lmstudio:qwen3.8-27b" in models and "__other__" in models
    assert page.inner_text("select[data-set=model] option[value='lmstudio:qwen3.8-27b']") == "qwen3.8-27b · lmstudio · loaded"
    assert page.inner_text("select[data-set=model] option[value='llamacpp:qwen3.8-27b-q5']") == "qwen3.8-27b-q5 · llamacpp"   # the provider is in the name
    assert page.locator("[data-set=url]").count() == 0 and page.locator("[data-set=key]").count() == 0   # no fields DuckDuckGo needs

    page.click('[data-set-act="save"]')                                                        # nothing changed: nothing is sent
    page.wait_for_function("document.getElementById('set-msg').textContent === 'Nothing changed.'")
    assert posts == [] and not saved.exists()

    page.select_option("select[data-set=search]", "searxng")                                   # its own field appears
    page.fill("[data-set=url]", srv.url)
    page.click('[data-set-act="test"]')
    page.wait_for_selector("#set-test .badge.done")
    assert "2 results" in page.inner_text("#set-test") and "Llama.cpp" in page.inner_text("#set-test")
    assert not saved.exists()                                                                  # trying is not saving
    page.fill("[data-set=url]", "http://127.0.0.1:9")
    page.click('[data-set-act="test"]')
    page.wait_for_selector("#set-test .badge.failed")
    assert "not reachable" in page.inner_text("#set-test")

    page.fill("[data-set=url]", srv.url)
    page.select_option("select[data-set=model]", "llamacpp:qwen3.8-27b-q5")
    page.select_option("select[data-set=fast]", "")
    page.click('[data-set-act="save"]')
    page.wait_for_function("document.getElementById('set-msg').textContent.startsWith('Saved.')")
    assert len(posts) == 1 and '"model":"llamacpp:qwen3.8-27b-q5"' in posts[0] and '"fast":""' in posts[0]
    layer = yaml.safe_load(saved.read_text(encoding="utf-8"))
    assert layer["models"] == {"strong": {"provider": "llamacpp", "model": "qwen3.8-27b-q5"}, "fast": None}
    assert layer["search"] == {"provider": "searxng", "searxng_url": srv.url}
    assert conf.models["strong"]["model"] == "qwen3.8-27b-q5" and not conf.models.get("fast")   # in force already

    page.reload()                                                                              # it is what the page shows next time
    page.click('#tabs [data-tab="settings"]')
    page.wait_for_selector("select[data-set=model]")
    assert page.input_value("select[data-set=model]") == "llamacpp:qwen3.8-27b-q5" and page.input_value("select[data-set=fast]") == ""
    assert page.input_value("[data-set=url]") == srv.url

    page.select_option("select[data-set=search]", "brave")                                     # a key: typed, saved, never shown again
    page.fill("[data-set=key]", "BSA-secret-key")
    page.click('[data-set-act="save"]')
    page.wait_for_function("document.getElementById('set-msg').textContent.startsWith('Saved.')")
    assert "BSA-secret-key" not in page.content() and "BSA-secret-key" not in page.input_value("[data-set=key]")
    assert page.get_attribute("[data-set=key]", "placeholder") == "saved"
    assert "One is saved" in page.inner_text("#p-settings")

    page.select_option("select[data-set=model]", "__other__")                                  # any model, typed as provider:model
    page.fill("[data-set=modelOther]", "nowhere:x")
    page.click('[data-set-act="save"]')
    page.wait_for_selector("#set-msg.bad")
    assert "provider:model" in page.inner_text("#set-msg")                                     # refused in plain words, nothing lost
    assert conf.models["strong"]["model"] == "qwen3.8-27b-q5"

    page.click('[data-set-act="reset"]')                                                       # back to the config files
    page.wait_for_function("document.getElementById('set-msg').textContent.startsWith('Back to')")
    assert not saved.exists() and page.input_value("select[data-set=model]") == "" and page.input_value("select[data-set=fast]") == "lmstudio:ornith"
    assert page.input_value("select[data-set=search]") == "duckduckgo"
    assert page.errors == []


def test_the_chat_glass_slider_previews_as_it_moves_and_saves_when_let_go(dashboard, conf, browse):
    import os
    shots = os.environ.get("JEVH_UI_SHOTS")   # (a folder: screenshots of the card, to look at)

    def flow(page):
        page.goto(dashboard)
        page.wait_for_selector(".hello")
        page.click('#tabs [data-tab="settings"]')
        slider = page.locator("input[data-glass]")
        slider.wait_for()
        assert slider.input_value() == "10" and page.inner_text("#glass-val") == "10"
        solid = page.eval_on_selector("#glass-preview", "e => e.style.getPropertyValue('--tint')")
        if shots:
            page.locator("#p-settings .card", has=slider).screenshot(path=os.path.join(shots, "glass-10.png"))
        slider.fill("85")                                                                       # moving it: the preview follows
        assert page.inner_text("#glass-val") == "85"
        assert float(page.eval_on_selector("#glass-preview", "e => e.style.getPropertyValue('--tint')")) < float(solid)
        page.wait_for_function("document.getElementById('glass-msg').textContent === 'Saved.'")
        assert yaml.safe_load((conf.workspace / SETTINGS_FILE).read_text(encoding="utf-8"))["appearance"] == {"orb_glass": 85}
        if shots:
            page.locator("#p-settings .card", has=slider).screenshot(path=os.path.join(shots, "glass-85.png"))
        page.reload()
        page.click('#tabs [data-tab="settings"]')
        page.locator("input[data-glass]").wait_for()
        assert page.input_value("input[data-glass]") == "85"
    browse(flow)
