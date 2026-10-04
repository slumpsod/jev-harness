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


def serve(cfg, replies):
    """The dashboard on a free port with a scripted model; yields its address."""
    def factory(on_event):
        return Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: ScriptedLLM(list(replies)), on_event=on_event)
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


@pytest.fixture
def dashboard(cfg):
    yield from serve(cfg, REPLIES)


@pytest.fixture
def slow_dashboard(cfg):
    """A task whose first step takes a few seconds: time to talk to it."""
    yield from serve(cfg, [py("import time\ntime.sleep(4)\nprint('step one')"), py("final('calendar done')")])


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


def test_the_notebook_takes_ideas_and_problems(dashboard, browse):
    browse(lambda page: _notebook(page, dashboard))


def _notebook(page, dashboard):
    page.goto(dashboard)
    page.wait_for_selector(".hello")
    page.click("#jotbtn")                                                    # the Jot button opens the notebook, ready to type
    page.wait_for_function("document.getElementById('side').getBoundingClientRect().width > 300")
    assert page.inner_text("#ptitle") == "Notebook"
    page.wait_for_function("document.activeElement && document.activeElement.id === 'nbtext'")
    page.fill("#nbtext", "Try a **second** model as the checker")
    page.keyboard.press("Control+Enter")
    page.wait_for_selector("#nblist .nbitem")
    assert page.inner_text("#nblist .nbtext b") == "second" and "from session" in page.inner_text("#nblist .nbsess")
    page.click('[data-nbkind="problem"]')
    page.uncheck("#nbtie")
    page.fill("#nbtext", "The chart axis had no units")
    page.click("#nbadd")
    page.wait_for_function("document.querySelectorAll('#nblist .nbitem').length === 2")
    assert page.inner_text("#nblist .nbitem:first-child .badge") == "Problem"          # newest first
    assert page.locator("#nblist .nbsess").count() == 1                                # only the idea is tied to a session
    assert page.inner_text("#nbcount") == "2"

    page.click('[data-nbfilter="problem"]')
    assert page.locator("#nblist .nbitem").count() == 1
    page.click('[data-nbfilter="all"]')
    page.fill("#nbsearch", "second")
    assert page.locator("#nblist .nbitem").count() == 1
    page.fill("#nbsearch", "")
    page.click('[data-nbfilter="open"]')

    page.click('#nblist .nbitem:first-child [data-nbact="toggle"]')                    # the problem is solved: off the open list
    page.wait_for_function("document.querySelectorAll('#nblist .nbitem').length === 1")
    assert page.inner_text("#nbcount") == "1"
    page.click('[data-nbfilter="done"]')
    assert page.inner_text("#nblist .nbitem .badge.done") == "solved"
    page.click('[data-nbfilter="open"]')

    page.click('#nblist .nbitem [data-nbact="edit"]')
    page.fill("#nbedit", "Try a second model as the checker, on hard cases")
    page.click('#nblist [data-nbact="save"]')
    page.wait_for_selector("#nblist .nbitem .nbtext")
    assert "on hard cases" in page.inner_text("#nblist .nbitem .nbtext")
    page.click('#nblist .nbitem [data-nbact="pin"]')
    page.wait_for_selector("#nblist .nbitem.pin")

    page.reload()                                                            # it is kept: the rail shows what is still open
    page.wait_for_function("document.getElementById('nbcount').textContent === '1'")
    page.click('#tabs [data-tab="notebook"]')
    page.once("dialog", lambda d: d.accept())
    page.click('#nblist .nbitem [data-nbact="delete"]')
    page.wait_for_function("document.querySelectorAll('#nblist .nbitem').length === 0")
    assert "Nothing" in page.inner_text("#nblist")
    assert not page.errors


def test_the_phone_layout_fits_and_every_tab_stays_reachable(dashboard, browse):
    browse(lambda page: _phone(page, dashboard))


def _phone(page, dashboard):
    page.set_viewport_size({"width": 360, "height": 640})
    page.goto(dashboard)
    page.wait_for_selector(".hello")
    too_wide = "document.documentElement.scrollWidth > innerWidth"
    assert not page.evaluate(too_wide)
    assert page.evaluate("document.getElementById('send').getBoundingClientRect().right <= innerWidth")     # the composer fits
    assert page.evaluate("getComputedStyle(document.getElementById('input')).fontSize") == "16px"          # no zoom-on-tap on iOS
    for tab in ("timeline", "inspector", "artifacts", "memory", "browser", "review", "notebook", "status"):
        page.click(f'#tabs [data-tab="{tab}"]')                              # a panel never covers the tab bar, so the next click works
        page.wait_for_function("document.getElementById('app').dataset.panel === 'open'")
        assert not page.evaluate(too_wide), tab
        colour = page.evaluate("getComputedStyle(document.getElementById('side')).backgroundColor")
        assert not colour.startswith("rgba") or colour.endswith(", 1)"), f"{tab}: the panel is see-through ({colour})"
        assert page.evaluate("document.getElementById('side').getBoundingClientRect().bottom"
                             " <= document.querySelector('.rail').getBoundingClientRect().top + 1"), tab
    page.click('#tabs [data-tab="notebook"]')
    page.fill("#nbtext", "an idea from my phone")                            # the notebook works with a thumb
    assert page.evaluate("getComputedStyle(document.getElementById('nbtext')).fontSize") == "16px"
    page.click("#nbadd")
    page.wait_for_selector("#nblist .nbitem")
    assert not page.evaluate(too_wide) and not page.errors


def test_a_message_sent_while_the_task_runs_is_heard(slow_dashboard, browse):
    browse(lambda page: _interject(page, slow_dashboard))


def _interject(page, url):
    page.goto(url)
    page.fill("#input", "check my email")
    page.keyboard.press("Enter")
    page.wait_for_selector(".turn .work.running", timeout=20000)             # it is working
    assert page.is_enabled("#send") and "running task" in page.get_attribute("#input", "placeholder")
    page.fill("#input", "/queue and then the weather")                       # another request: waits its turn
    page.keyboard.press("Enter")
    page.wait_for_selector("#queuebar .qchip", timeout=10000)
    assert "and then the weather" in page.inner_text("#queuebar")
    page.fill("#input", "actually check my calendar")                        # plain: added to the prompt
    page.keyboard.press("Enter")
    page.wait_for_selector(".say .saytag.corrected", timeout=15000)
    assert "actually check my calendar" in page.inner_text(".say:has(.saytag.corrected)")
    page.wait_for_function("document.querySelectorAll('.turn').length === 2 && "
                           "document.querySelectorAll('.turn .answer').length === 2", timeout=60000)
    assert page.inner_text(".turn:nth-child(1) .answer") == "calendar done"     # the correction reached the task
    assert page.is_hidden("#queuebar")                                       # the queued request started by itself
    assert not page.errors
