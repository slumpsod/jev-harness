"""The dashboard in a real browser, against the real server with scripted models (no LLM needed).
Skipped when Playwright's Chromium isn't installed."""
import glob
import json
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

from jevh.context import Window  # noqa: E402
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


class Windowed(ScriptedLLM):
    """A scripted model that knows its context window, as a model client asked of LM Studio does."""
    max_tokens = 8192

    def window(self):
        return Window(100096, "lmstudio", True, 262144, 2)


def serve(cfg, replies, model=ScriptedLLM, jev_rules=rules, setup=None):
    """The dashboard on a free port with a scripted model; yields its address. `setup(harness)` runs on the new harness
    (past sessions to remember)."""
    def factory(on_event, session=None):
        h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(jev_rules)), llm_factory=lambda p: model(list(replies)), on_event=on_event,
                    session=session)
        if setup and not session:   # (a session picked up again already has its past)
            setup(h)
        return h
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
def windowed_dashboard(cfg):
    yield from serve(cfg, REPLIES, Windowed)


@pytest.fixture
def history_dashboard(cfg):
    """Four past sessions in memory, and a Jev that knows which field each is about (tests/test_history.py)."""
    from test_history import SESSIONS, populate, rules as history_rules
    yield from serve(cfg, REPLIES, jev_rules=history_rules, setup=lambda h: populate(h, SESSIONS))


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
    said = page.inner_text(".turn .workhead")                                # the tool calls, as a sentence, not as code
    assert "saved a file" in said.lower() and "save_artifact" not in page.inner_text(".turn")
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
    assert rows.count() == 2 and "Saved hello.html" in page.inner_text(".turn .steps")
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
    for tab in ("inspector", "artifacts", "memory", "history", "browser", "review", "notebook"):
        page.click(f'#tabs [data-tab="{tab}"]')                              # a panel never covers the tab bar, so the next click works
        page.wait_for_function("document.getElementById('app').dataset.panel === 'open'")
        assert not page.evaluate(too_wide), tab
        colour = page.evaluate("getComputedStyle(document.getElementById('side')).backgroundColor")
        assert not colour.startswith("rgba") or colour.endswith(", 1)"), f"{tab}: the panel is see-through ({colour})"
        assert page.evaluate("document.getElementById('side').getBoundingClientRect().bottom"
                             " <= document.querySelector('.rail').getBoundingClientRect().top + 1"), tab
    for name in ("newsroom", "settings"):                                    # the full pages: above the tab bar too, with a way back
        page.click(f'.rail [data-page="{name}"]')
        page.wait_for_function(f"document.getElementById('app').dataset.view === '{name}'")
        assert not page.evaluate(too_wide), name
        assert page.evaluate(f"document.getElementById('page-{name}').getBoundingClientRect().bottom"
                             " <= document.querySelector('.rail').getBoundingClientRect().top + 1"), name
        assert page.is_visible(f"#page-{name} .back") and not page.is_visible("#chat")
    for sec in ("general", "devices", "status", "updates"):                  # every section of Settings, at phone width
        page.click(f'#setnav [data-sec="{sec}"]')
        page.wait_for_selector(f"#sec-{sec} .card, #sec-{sec} .arow, #sec-{sec} .kv")
        assert not page.evaluate(too_wide), sec
    page.click("#page-settings .back")
    assert page.is_visible("#chat") and page.inner_text("#ptitle") == "Notebook"   # back to the panel that was open
    page.fill("#nbtext", "an idea from my phone")                            # the notebook works with a thumb
    assert page.evaluate("getComputedStyle(document.getElementById('nbtext')).fontSize") == "16px"
    page.click("#nbadd")
    page.wait_for_selector("#nblist .nbitem")
    assert not page.evaluate(too_wide) and not page.errors


VERSION = {"branch": "main", "channel": "stable", "channels": ["stable", "main"], "upstream": "origin/stable",
           "current": {"sha": "aaaaaaa", "date": "2026-10-10", "subject": "what is on disk"},
           "latest": {"sha": "bbbbbbb", "date": "2026-10-12", "subject": "the newest"}, "behind": 2, "ahead": 0,
           "incoming": [{"sha": "bbbbbbb", "date": "2026-10-12", "subject": "the newest"},
                        {"sha": "ccccccc", "date": "2026-10-11", "subject": "the one before"}],
           "dirty": [], "conflicts": [], "in_way": [], "deps_changed": False, "fetched": True, "can_update": True,
           "previous": None, "snapshot": None, "error": None}


def test_the_newsroom_and_settings_are_pages_of_their_own_and_the_chat_comes_back(dashboard, browse):
    browse(lambda page: _pages(page, dashboard))


def _pages(page, dashboard):
    page.route("**/api/version*", lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps(VERSION)))
    page.goto(dashboard)
    page.wait_for_selector(".hello")
    view = "document.getElementById('app').dataset.view"
    page.click('#tabs [data-tab="memory"]')                                  # a panel is open beside the chat ...
    page.click('.rail [data-page="newsroom"]')                               # ... and the Newsroom takes the place of both
    assert page.evaluate(view) == "newsroom"
    assert not page.is_visible("#chat") and not page.is_visible("#side") and page.is_visible("#page-newsroom")
    assert page.evaluate("document.querySelector('#page-newsroom').getBoundingClientRect().width"
                         " >= innerWidth - document.querySelector('.rail').offsetWidth - 1")   # the whole width
    assert page.evaluate("document.activeElement.id") == "nrtitle"          # a screen reader lands on the page
    assert page.get_attribute('.rail [data-page="newsroom"]', "aria-pressed") == "true"
    page.keyboard.press("Escape")                                            # Esc goes back, to the panel that was open
    assert page.evaluate(view) == "chat" and page.is_visible("#chat")
    assert page.inner_text("#ptitle") == "Memory" and attr(page, "#app", "data-panel") == "open"
    assert page.evaluate("document.activeElement.dataset.page") == "newsroom"
    page.click('.rail [data-page="newsroom"]')
    page.click('.rail [data-page="newsroom"]')                               # its rail button again goes back too
    assert page.evaluate(view) == "chat"
    page.click('.rail [data-page="newsroom"]')
    page.click("#page-newsroom .back")                                       # and so does "Chat"
    assert page.evaluate(view) == "chat"

    # Settings: one page, four sections (what were the Settings, Devices and Status tabs)
    page.wait_for_function("document.getElementById('updcount').textContent === '2' && !document.getElementById('updcount').hidden")
    page.click('.rail [data-page="settings"]')                               # the first time: models and search
    assert page.get_attribute("#st-general", "aria-selected") == "true" and page.is_visible("#sec-general")
    assert not page.is_visible("#sec-devices") and not page.is_visible("#sec-status") and not page.is_visible("#sec-updates")
    assert page.is_visible("#updcount2") and page.inner_text("#updcount2") == "2"   # the count is on the Updates section too
    page.wait_for_selector("#p-settings select[data-set=model]")
    page.click('#setnav [data-sec="devices"]')
    page.wait_for_selector("#p-devices .card h3:has-text('Signing in')")
    assert "is the hub" in page.inner_text("#p-devices")
    page.focus("#st-devices")
    page.keyboard.press("ArrowDown")                                         # a tab list: arrows move between the sections
    assert page.get_attribute("#st-status", "aria-selected") == "true" and page.evaluate("document.activeElement.id") == "st-status"
    page.wait_for_selector("#p-status .card")
    page.keyboard.press("End")
    assert page.is_visible("#sec-updates") and page.get_attribute("#st-devices", "tabindex") == "-1"
    card = page.inner_text("#p-updates")
    assert "2 new commits" in card and "the one before" in card and page.is_visible('#p-updates [data-upd="apply"]')
    page.click('#tabs [data-tab="inspector"]')                               # a panel's tab leaves the page for the chat
    assert page.evaluate(view) == "chat" and page.inner_text("#ptitle") == "Inspector"
    page.click('.rail [data-page="settings"]')                               # Settings opens where it was left
    assert page.is_visible("#sec-updates")
    page.click("#page-settings .back")
    page.click("#stat")                                                      # the header's status dots open Status
    assert page.evaluate(view) == "settings" and page.is_visible("#sec-status")
    assert not page.errors


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


def test_the_context_ring_says_what_window_the_model_was_loaded_with(windowed_dashboard, browse):
    browse(lambda page: _window(page, windowed_dashboard))


def _window(page, dashboard):
    page.goto(dashboard)
    page.fill("#input", "make me a hello page")
    page.keyboard.press("Enter")
    page.wait_for_selector(".turn .answer", timeout=30000)
    page.hover("#ctxring")                                                   # the breakdown opens over the ring
    page.wait_for_selector("#ctxpop:not([hidden])")
    said = page.inner_text("#ctxpop")
    assert "Model window: 100,096 tokens (what LM Studio loaded it with)" in said
    assert "the model supports up to 262,144" in said
    assert "Past 60.1k, older steps get condensed" in said                   # 60% of that window, not limits.compact_tokens
    assert "of 100.1k tokens" in said                                        # the wheel is out of the window, not out of the 60k trigger
    assert "of 100.1k tokens" in page.get_attribute("#ctxring", "aria-label")
    assert not page.errors


def test_a_live_list_you_scrolled_up_in_stays_where_you_left_it(dashboard, browse):
    browse(lambda page: _scrolled(page, dashboard))


def _scrolled(page, dashboard):
    page.goto(dashboard)
    page.wait_for_selector("#logi")
    make = """(key) => {   // what renderChat draws: a fresh element for a live list each time its turn is redrawn
        const old = document.querySelector('[data-follow="' + key + '"]'); if (old) old.remove();
        const p = document.createElement('div'); p.dataset.follow = key;
        p.style.cssText = 'height:60px;overflow:auto;position:fixed;top:0;left:0;width:200px;z-index:99';
        p.innerHTML = '<div style="height:1000px"></div>'; document.body.appendChild(p);
        followLists(document.body); return p.scrollTop; }"""
    assert page.evaluate(make, "t:x") > 500                                  # a new list starts at its end
    page.evaluate("document.querySelector('[data-follow=\"t:x\"]').scrollTop = 100")   # you scroll up to read
    page.wait_for_function("FOLLOW.get('t:x') && FOLLOW.get('t:x').top === 100")
    assert page.evaluate(make, "t:x") == 100                                 # the redraw leaves it where you put it
    page.evaluate("(() => { const p = document.querySelector('[data-follow=\"t:x\"]'); p.scrollTop = p.scrollHeight; })()")
    page.wait_for_function("FOLLOW.get('t:x').end === true")
    assert page.evaluate(make, "t:x") > 500                                  # back at the end, it follows again
    assert not page.errors


def test_the_inspector_has_the_timeline_and_the_decisions_in_one_panel(dashboard, browse):
    browse(lambda page: _inspector(page, dashboard))


def _inspector(page, dashboard):
    page.goto(dashboard)
    assert page.locator('#tabs [data-tab="timeline"]').count() == 0          # one tab where there were two
    page.fill("#input", "make me a hello page")
    page.keyboard.press("Enter")
    page.wait_for_selector(".turn .answer", timeout=30000)
    page.click('#tabs [data-tab="inspector"]')
    page.wait_for_selector("#p-inspector .sechead")
    assert [h.lower() for h in page.locator("#p-inspector .sechead").all_inner_texts()] == ["timeline", "decisions"]
    assert page.locator("#p-inspector .tl .bar").count() >= 1                # the task's lane, positioned in time
    assert page.locator("#p-inspector .card h3", has_text="Routing").count() == 1
    page.click("#p-inspector .tl .label[data-jump]")                         # a lane's name jumps to that task's decisions
    page.wait_for_selector("#p-inspector [data-card].flash")
    assert page.inner_text("#ptitle") == "Inspector" and not page.errors


def test_the_history_tab_lists_past_sessions_by_the_field_jev_put_them_in(history_dashboard, browse):
    browse(lambda page: _history(page, history_dashboard))


def _history(page, dashboard):
    page.goto(dashboard)
    page.click('#tabs [data-tab="history"]')
    page.wait_for_function("document.querySelectorAll('#p-history .hrow').length === 4 && !document.querySelector('[data-cat=__waiting]')",
                           timeout=20000)                                    # Jev sorts them in the background
    assert page.locator("#p-history .hgroup summary b").all_inner_texts() == ["Daily assistant", "Fitness coach", "High end dev", "No single field"]
    assert "flask app" in page.locator("#p-history .hgroup", has=page.locator("summary b", has_text="High end dev")).inner_text()
    page.click('[data-h="recat"]')                                           # "Sort again": Jev is asked about every session again
    page.wait_for_function("document.querySelectorAll('#p-history .hgroup').length === 4 && !document.querySelector('[data-cat=__waiting]')"
                           " && !document.querySelector('[data-h=recat]').disabled", timeout=20000)   # ...and the button is usable again
    page.click('[data-h="recat"]')
    page.wait_for_function("!document.querySelector('[data-h=recat]').disabled", timeout=20000)
    page.wait_for_function("document.querySelectorAll('#p-history .hrow').length === 4 && !document.querySelector('[data-cat=__waiting]')", timeout=20000)
    page.fill("#hq", "workout")                                              # the search narrows the list as you type
    assert page.locator("#p-history .hrow").count() == 1
    page.fill("#hq", "")
    page.locator("#p-history .hrow", has_text="workouts").click()           # a session opens read-only, with Jev's odds
    page.wait_for_selector("#p-history .hturn")
    assert page.locator("#p-history .hturn").count() == 2
    card = page.inner_text("#p-history .card h3")
    assert "Fitness coach" in card and "97% sure" in card
    assert "swap the squats" in page.inner_text("#p-history")
    page.click('[data-h="back"]')
    page.wait_for_selector("#p-history .hgroup")
    assert page.inner_text("#ptitle") == "History" and not page.errors


def test_a_turns_footer_gives_tokens_per_second_only_when_the_decode_time_is_known(dashboard, browse):
    def body(page):
        page.goto(dashboard)
        page.wait_for_function("typeof usageBrief === 'function'")
        llm = lambda **k: {"by_kind": {"llm": {"input": 9000, "output": 2600, "thinking": 0, "seconds": 35, **k}}, "total": {}}
        assert " 60 tok/s" in page.evaluate("(u) => usageBrief(u)", llm(decode_seconds=10, decode_tokens=600))
        assert "tok/s" not in page.evaluate("(u) => usageBrief(u)", llm())
        assert not page.errors
    browse(body)


def test_the_message_box_shows_the_speed_the_model_is_writing_at_while_it_writes(dashboard, browse):
    def body(page):
        page.goto(dashboard)
        page.wait_for_function("typeof handle === 'function'")
        page.evaluate("""() => {
          handle({seq: 1, kind: "turn_start", turn: 1, data: {turn: {id: 1, message: "hi", status: "running", started: Date.now() / 1000}}});
          render();
        }""")
        assert page.locator("#tps").is_hidden()                                   # nothing written yet
        page.evaluate("""() => { handle({seq: 2, kind: "progress", turn: 1, data: {persona: "daily_assistant", chunks: 50, thinking: 0, tps: 61.4}}); render(); }""")
        assert page.inner_text("#tps") == "61 tok/s" and page.locator("#tps").is_visible()
        page.evaluate("""() => { handle({seq: 3, kind: "progress", turn: 1, data: {persona: "high_end_dev", chunks: 50, thinking: 0, tps: 20}}); render(); }""")
        assert page.inner_text("#tps") == "81 tok/s (2 streams)"                  # hand-offs in parallel: the sum
        page.evaluate("""() => { for (const p of ["daily_assistant", "high_end_dev"]) handle({seq: 4, kind: "progress", turn: 1, data: {persona: p, chunks: 90, thinking: 0, tps: 0}}); render(); }""")
        assert page.locator("#tps").is_hidden()                                   # the calls ended: nothing is being written
        page.evaluate("""() => { handle({seq: 5, kind: "progress", turn: 1, data: {persona: "daily_assistant", chunks: 50, thinking: 0, tps: 40}}); S.live.daily_assistant.at -= 20000; render(); }""")
        assert page.locator("#tps").is_hidden()                                   # a report that stopped coming is not kept on show
        assert not page.errors
    browse(body)


def test_a_past_session_can_be_resumed_from_the_history_tab(history_dashboard, browse):
    browse(lambda page: _resume(page, history_dashboard))


def _resume(page, dashboard):
    page.goto(dashboard)
    page.click('#tabs [data-tab="history"]')
    page.wait_for_function("document.querySelectorAll('#p-history .hrow').length === 4 && !document.querySelector('[data-cat=__waiting]')",
                           timeout=20000)
    page.locator("#p-history .hrow", has_text="workouts").locator('[data-h="load"]').click()   # the row's own Resume button
    page.wait_for_function("document.querySelectorAll('#logi .turn').length === 2")           # the conversation is back in the chat
    chat = page.inner_text("#logi")
    assert "plan my workouts" in chat and "swap the squats" in chat and "Mon push, Tue pull" in chat
    assert "20261002-100000-bbbb" in page.inner_text("#sess")
    assert page.inner_text("#ptitle") == "History" and page.locator("#side.on, #app[data-panel=open]").count() == 1   # (a wide screen keeps the panel)
    page.wait_for_function("!document.querySelector('#p-history .hrow [data-h=load]') || "
                           "document.querySelectorAll('#p-history .hrow [data-h=load]').length === 3")   # this session no longer offers Resume
    assert "this session" in page.inner_text("#p-history")
    page.locator("#p-history .hrow", has_text="flask app").click()                              # the detail view has its own button
    page.wait_for_selector("#p-history .hhead .primary")
    assert page.inner_text("#p-history .hhead .primary") == "Resume this session"
    page.click("#p-history .hhead .primary")
    page.wait_for_function("document.querySelectorAll('#logi .turn').length === 2 && document.querySelector('#logi').innerText.includes('flask app')")
    assert "20261001-100000-aaaa" in page.inner_text("#sess") and not page.errors


@pytest.fixture
def llamacpp_dashboard(cfg, monkeypatch):
    """The config puts every model on a llama.cpp server (as config.local.yaml does on Jed's machine)."""
    from jevh import context as ctxwin
    monkeypatch.setitem(cfg.providers, "llamacpp", {"type": "openai", "base_url": "http://127.0.0.1:8080/v1"})
    monkeypatch.setitem(cfg.models, "strong", {"provider": "llamacpp", "model": "qwen3.8-27b"})
    monkeypatch.setitem(cfg.models, "fast", None)
    monkeypatch.setattr(ctxwin, "server_props", lambda url, **k: {"context": 76032, "slots": 2, "model": "qwen3.8-27b",
                                                                    "file": "Qwen3.8-27B-UD-Q5_K_M.gguf", "build": "b11451"}
                        if ":8080" in url else None)
    yield from serve(cfg, REPLIES)


def test_status_shows_the_llama_cpp_server_the_harness_runs_on(llamacpp_dashboard, browse):
    browse(lambda page: _llamacpp(page, llamacpp_dashboard))


def _llamacpp(page, dashboard):
    page.goto(dashboard)
    page.wait_for_function("document.getElementById('statmodel').textContent.includes('qwen3.8-27b')")
    assert page.inner_text("#statmodel") == "qwen3.8-27b · 76.0k"             # the header names the server the harness uses
    page.click("#stat")                                                      # the header's dots open Status, in Settings
    assert page.get_attribute("#st-status", "aria-selected") == "true" and page.is_visible("#sec-status")
    page.wait_for_selector("#p-status .card h3:has-text('llama.cpp server')")
    card = page.inner_text("#p-status .card:has(h3:has-text('llama.cpp server'))")
    assert "used by the harness" in card.lower() and "76,032 context, 2 slots" in card
    assert not page.errors


def test_the_browser_panel_sends_a_press_and_its_release(dashboard, browse):
    browse(lambda page: _hold_panel(page, dashboard))


def _hold_panel(page, dashboard):
    sent = []
    page.route("**/api/browser/input**", lambda route: (sent.append(json.loads(route.request.post_data)["kind"]),
                                                         route.fulfill(status=200, body='{"ok": true}')))
    page.route("**/api/browser/frame**", lambda route: route.fulfill(     # as if a docked browser were up
        status=200, body=b"", headers={"Content-Type": "image/jpeg", "X-Browser-Id": "t-1", "X-Page-Url": "about%3Ablank"}))
    page.goto(dashboard)
    page.click('#tabs [data-tab="browser"]')
    page.wait_for_function("B.live === true")
    box = page.locator("#screen").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.wait_for_timeout(400)
    page.mouse.move(box["x"] + box["width"] / 2 + 20, box["y"] + box["height"] / 2)
    page.mouse.up()
    page.wait_for_timeout(300)
    assert sent[0] == "down" and sent[-1] == "up" and "click" not in sent, sent
    assert not page.errors


# ---- the orb's chat bubble (jevh/web/static/orb.html; the orb itself: tests/test_orb.py)
@pytest.fixture
def asking_hub(cfg):
    """A dashboard whose assistant asks the user a question, then answers with what they said."""
    import uvicorn
    replies = [py("a = ask_user('Go ahead with the plan?', ['Yes', 'No'])\nfinal('You said ' + a)")]

    def factory(on_event, session=None):
        return Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: ScriptedLLM(list(replies)),
                       on_event=on_event)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(cfg, port=port, harness_factory=factory, token="t"),
                                           host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/orb"
    server.should_exit = True


def test_the_bubble_chats_answers_questions_and_tells_the_orb_its_mood(asking_hub, browse):
    def body(page):
        # the app's bridge, as pywebview gives it: the moods it is told are recorded
        page.add_init_script("window.__moods = []; window.pywebview = {api: {mood: (m) => window.__moods.push(m), "
                             "close: () => {}, open_app: () => {}}};")
        page.goto(asking_hub)
        page.wait_for_selector(".empty")
        page.fill("#input", "make the plan")
        page.keyboard.press("Enter")
        page.wait_for_selector(".ask button", timeout=20000)             # the question, with its options
        assert "Go ahead with the plan?" in page.inner_text(".ask")
        page.wait_for_function("window.__moods.includes('attention')")
        page.click(".ask button:has-text('Yes')")
        page.wait_for_selector(".msg.bot", timeout=20000)
        assert "You said Yes" in page.inner_text(".msg.bot")
        page.wait_for_function("window.__moods[window.__moods.length - 1] === 'unread'")   # (not shown: an unread answer)
        moods = page.evaluate("window.__moods")
        assert "working" in moods and moods.index("working") < moods.index("attention")
        page.evaluate("window.orbShown()")                                # the bubble opened: it has been seen
        page.wait_for_function("window.__moods[window.__moods.length - 1] === 'idle'")
        assert not page.errors
    browse(body)


def test_a_page_with_a_stale_token_reloads_itself_once(dashboard, browse):
    """The dashboard restarted (a new API token each launch): the open page reloads to get it, instead of showing
    'missing or bad token' - and only once, if the refusal goes on."""
    def body(page):
        page.goto(dashboard)
        page.wait_for_selector(".hello")
        refusals = {"n": 0}

        def stale(route):
            refusals["n"] += 1
            route.fulfill(status=403, content_type="application/json", body='{"detail": "missing or bad token"}')
        page.route("**/api/status*", stale)
        loads = []
        page.on("load", lambda: loads.append(1))
        page.evaluate("api('/api/status').catch(() => {})")
        page.wait_for_timeout(1500)
        assert len(loads) == 1                                   # reloaded, picking up the page's new token
        page.evaluate("api('/api/status').catch(() => {})")      # still refused right after: no loop
        page.wait_for_timeout(1500)
        assert len(loads) == 1 and refusals["n"] >= 2
    browse(body)
