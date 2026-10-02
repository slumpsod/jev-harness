"""The dashboard in a real browser, against the real server with scripted models (no LLM needed).
Skipped when Playwright's Chromium isn't installed."""
import glob
import os
import socket
import threading
import time

import pytest

pytest.importorskip("playwright.sync_api")
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from jevh.jev import Jev  # noqa: E402
from jevh.orchestrator import Harness  # noqa: E402
from jevh.web.server import create_app  # noqa: E402

from conftest import FakeSystemOne, ScriptedLLM, choice_ans, noul_ans, py  # noqa: E402

REPLIES = [
    py("print(sorted([3, 1, 2]))\nx = 6 * 7\nprint(x)"),
    py("p = save_artifact('hello.html', '<h1>Hello from an artifact</h1>')\nfinal('Here is **your page**.', [p])"),
]


def rules(k, q, s):
    if k == "persona":
        return choice_ans("daily_assistant", 0.9, q["criteria"])
    if k in ("needs_high_end_dev", "in_scope", "necessary"):
        return noul_ans(0.9)


@pytest.fixture
def dashboard(cfg):
    def factory(on_event):
        return Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: ScriptedLLM(list(REPLIES)), on_event=on_event)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(cfg, port=port, harness_factory=factory, token="t"),
                                           host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/"
    server.should_exit = True


def _launch(p):
    for path in [None, os.environ.get("CHROMIUM_PATH"), *glob.glob("/opt/pw-browsers/chromium-*/chrome-linux*/chrome")]:
        try:
            return p.chromium.launch(**({"executable_path": path} if path else {}))
        except Exception:  # noqa: BLE001 - try the next candidate
            continue
    pytest.skip("chromium unavailable")


@pytest.fixture
def browse():
    """Run `body(page)` in a fresh thread: Playwright's sync API refuses to start in a thread that has (or had)
    an event loop, which other test modules leave behind in the main one."""
    def run(body):
        out = {}

        def work():
            try:
                with sync_playwright() as p:
                    browser = _launch(p)
                    try:
                        page = browser.new_page(viewport={"width": 1400, "height": 900})
                        page.errors = []
                        page.on("pageerror", lambda e: page.errors.append(str(e)))
                        body(page)
                    finally:
                        browser.close()
            except BaseException as e:  # noqa: BLE001 - re-raised in the test's own thread (assertions, skips)
                out["err"] = e
        t = threading.Thread(target=work)
        t.start()
        t.join(180)
        assert not t.is_alive(), "the browser test hung"
        if "err" in out:
            raise out["err"]
    return run


def attr(page, sel, name):
    return page.get_attribute(sel, name)


def test_chat_is_the_focus_until_something_needs_the_panel(dashboard, browse):
    browse(lambda page: _focus(page, dashboard))


def _focus(page, dashboard):
    page.goto(dashboard)
    page.wait_for_selector(".hello")
    assert attr(page, "#app", "data-panel") == "closed"
    assert page.evaluate("document.getElementById('side').getBoundingClientRect().width") < 2
    page.click('#tabs [data-tab="memory"]')                                  # a clicked tab opens it ...
    page.wait_for_function("document.getElementById('side').getBoundingClientRect().width > 300")
    assert page.inner_text("#ptitle") == "Memory"
    page.click('#tabs [data-tab="memory"]')                                  # ... and the same tab closes it
    page.wait_for_function("document.getElementById('side').getBoundingClientRect().width < 2")

    page.fill("#input", "make me a hello page")
    page.keyboard.press("Enter")
    page.wait_for_selector(".turn .answer", timeout=30000)
    chips = page.inner_text(".turn .wchips")                                 # the tool calls, as chips, not as code
    assert "Artifacts" in chips and "save_artifact" not in page.inner_text(".turn")
    page.wait_for_function("document.getElementById('app').dataset.panel === 'open'", timeout=10000)   # a file was made
    assert page.inner_text("#ptitle") == "Artifacts"
    assert page.locator("#p-artifacts iframe[sandbox]").count() == 1
    assert not page.errors


def test_a_step_opens_to_its_verbatim_code(dashboard, browse):
    browse(lambda page: _step(page, dashboard))


def _step(page, dashboard):
    page.goto(dashboard)
    page.fill("#input", "go")
    page.keyboard.press("Enter")
    page.wait_for_selector(".turn .answer", timeout=30000)
    page.click(".turn .workhead")
    rows = page.locator(".turn .stline:not(.stnote)")
    assert rows.count() == 2
    assert page.locator(".turn .stdetail").count() == 0                      # one line each, no code block yet
    rows.first.click()
    assert "sorted([3, 1, 2])" in page.inner_text(".turn .stdetail")
    assert "[1, 2, 3]" in page.inner_text(".turn .stdetail")                 # and its output


def test_appearance_choices_apply_and_survive_a_reload(dashboard, browse):
    browse(lambda page: _appearance(page, dashboard))


def _appearance(page, dashboard):
    page.goto(dashboard)
    page.wait_for_selector(".hello")
    assert "--grid" not in page.evaluate("getComputedStyle(document.body).backgroundImage")
    page.click("#appearbtn")
    for theme in ("sumi", "paper", "clay", "obsidian", "nocturne", "phosphor", "aurora"):
        page.click(f'[data-theme-pick="{theme}"]')
        assert page.evaluate("document.documentElement.dataset.theme") == theme
        assert page.is_visible("#apppop")                                    # picking one keeps the picker open
    page.click('[data-theme-pick="paper"]')
    page.click('[data-accent="#3e63dd"]')
    page.click('[data-pref="width"][data-val="wide"]')
    page.reload()
    page.wait_for_selector(".hello")
    root = "document.documentElement"
    assert page.evaluate(f"{root}.dataset.theme + ',' + {root}.dataset.width") == "paper,wide"
    assert page.evaluate(f"getComputedStyle({root}).getPropertyValue('--accent').trim()") == "#3e63dd"
    page.click("#appearbtn")
    page.click("[data-areset]")
    assert page.evaluate(f"{root}.dataset.theme") == "aurora"
    assert not page.errors
