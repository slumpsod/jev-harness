"""The floating orb (jevh/orb_art.py, jevh/orb.py, the hub's /orb page, jevh/app.py's bubble), offline.

The window itself is Windows-only and drawn by Windows; it was checked live on the screen (2026-10-12): see
docs/devices.md. These test what can be tested without a desktop: the pictures, where it settles, the chat page and the
moods it reports, and the app's click logic.
"""
from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from jevh import orb_art
from jevh.jev import Jev
from jevh.orchestrator import Harness
from jevh.web.server import create_app

from conftest import FakeSystemOne, ScriptedLLM, choice_ans, py
from test_dashboard_ui import browse  # noqa: F401 - the fixture

R = 30


def test_frames_are_premultiplied_water_with_click_through_corners():
    f = orb_art.frame(0.1, R, "idle")
    w, h = orb_art.size_for(R)
    assert f.shape == (h, w, 4) and f.dtype == np.uint8
    a = f[..., 3].astype(int)
    assert a[0, 0] == 0 and a[-1, -1] == 0 and a[0, -1] == 0          # see-through corners: clicks pass to the desktop
    assert a[h // 2, w // 2] > 150                                     # the water catches them
    assert (f[..., :3].astype(int) <= a[..., None] + 1).all()           # premultiplied, as UpdateLayeredWindow wants
    assert len(orb_art.frames(R, "idle")) == int(orb_art.LOOP * orb_art.FPS)


def test_each_mood_glows_its_own_colour():
    w, h = orb_art.size_for(R)
    gx, gy = w // 2, int(h / 2 + 0.08 * R + 1.25 * R)                   # just below the water: glow only
    idle, attention = orb_art.frame(0.0, R, "idle"), orb_art.frame(0.0, R, "attention")
    b, g, r = (int(v) for v in idle[gy, gx, :3])
    assert b > r                                                        # cyan
    b, g, r = (int(v) for v in attention[gy, gx, :3])
    assert r > b                                                        # amber: it needs you
    working = orb_art.frame(0.0, R, "working")
    assert working[gy, gx, 3] > idle[gy, gx, 3]                         # brighter while it works


def test_it_settles_against_the_nearer_side():
    from jevh.orb_art import MARGIN, settle
    area = (0, 0, 1920, 1040)
    x, y = settle(1500, 300, 120, 100, area)
    assert x > 1800 - 120 and y == 300                                  # right half: the right edge
    x, y = settle(200, 2000, 120, 100, area)
    assert x < MARGIN and y == 1040 - 100                               # left half, and back above the taskbar


def rules(k, q, s):
    if k == "persona":
        return choice_ans("daily_assistant", 0.9, q["criteria"])


def test_the_bubble_page_is_behind_the_same_sign_in(cfg):
    def factory(on_event, session=None):
        return Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: ScriptedLLM([]), on_event=on_event)
    c = TestClient(create_app(cfg, port=8740, harness_factory=factory, token="tok", allow_hosts=["box.ts.net:10000"],
                              access_key="k3y"), base_url="http://127.0.0.1:8740")
    page = c.get("/orb")
    assert page.status_code == 200 and 'TOKEN = "tok"' in page.text
    assert c.get("/orb", headers={"Host": "box.ts.net:10000"}).status_code == 403
    assert c.get("/orb?key=k3y", headers={"Host": "box.ts.net:10000"}).status_code == 200


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


class _Fake:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append(name)


def test_a_click_on_the_orb_toggles_the_bubble(monkeypatch):
    from jevh.app import App
    app = App({"role": "device", "hub": "https://hub:10000"})
    app.bubble, app.orb = _Fake(), _Fake()
    monkeypatch.setattr(app, "place_bubble", lambda: None)
    app.orb_clicked()
    assert app.bubble_visible and "show" in app.bubble.calls
    app.orb_clicked()
    assert not app.bubble_visible and "hide" in app.bubble.calls
    app.bubble_visible = True
    app.hide_bubble()                        # it lost the focus to a click on the orb ...
    app.orb_clicked()                        # ... and that click is the one that arrives now: it stays closed
    assert not app.bubble_visible
    app.main_shown()                         # the main window came back: the orb goes, and the bubble with it
    assert "hide" in app.orb.calls
