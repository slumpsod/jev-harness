"""The floating orb (jevh/orb_art.py, jevh/orb.py, the hub's /orb page, jevh/app.py's bubble), offline.

The window itself is Windows-only and drawn by Windows; it was checked live on the screen (2026-10-12): see
docs/devices.md. These test what can be tested without a desktop: the pictures, where it settles, the chat page behind
sign-in, and the app's click logic. The bubble driven in Chromium (moods through a question) is in test_dashboard_ui.py.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from jevh import orb_art
from jevh.jev import Jev
from jevh.orchestrator import Harness
from jevh.web.server import create_app

from conftest import FakeSystemOne, ScriptedLLM, choice_ans

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


def test_the_app_title_bar_script_is_served_to_every_app_page(cfg):
    """jevh/web/static/app-chrome.js: open (the sign-in page needs it), included by the dashboard, the sign-in page and
    the app's first-run page; it does nothing outside the app (it waits for pywebview's frameless window)."""
    def factory(on_event, session=None):
        return Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: ScriptedLLM([]), on_event=on_event)
    app = create_app(cfg, port=8740, harness_factory=factory, token="tok", allow_hosts=["box.ts.net:10000"])
    app.state.hub.accounts.add_user("jed", "correct horse")
    c = TestClient(app, base_url="http://127.0.0.1:8740")
    js = c.get("/app-chrome.js", headers={"Host": "box.ts.net:10000"})              # before signing in, from elsewhere
    assert js.status_code == 200 and "javascript" in js.headers["content-type"] and "app-region: drag" in js.text
    assert "win_state" in js.text and "frameless" in js.text                          # inert unless the app says so
    assert '<script src="/app-chrome.js">' in c.get("/").text
    assert '<script src="/app-chrome.js">' in c.get("/login", headers={"Host": "box.ts.net:10000"}).text
    from jevh.app import setup_html
    page = setup_html()
    assert "app-region: drag" in page and 'class="app-drag"' in page


def test_the_orb_is_the_apps_icon(tmp_path):
    from PIL import Image
    img = orb_art.icon_image(64)
    assert img.size == (64, 64) and img.mode == "RGBA"
    assert img.getpixel((0, 0))[3] < 40 and img.getpixel((32, 32))[3] > 150    # a drop on nothing
    orb_art.write_icon(tmp_path / "orb.ico")
    assert (16, 16) in Image.open(tmp_path / "orb.ico").info["sizes"]


# ---------------------------------------------------------------------- the orb's own chat (bar + speech bubbles)
def test_the_bar_takes_typing_and_sends_on_enter():
    from jevh.orb_chat import ChatState
    c = ChatState()
    for ch in "hello wrld":
        c.char(ord(ch))
    c.key(0x25); c.key(0x25); c.key(0x25)          # left x3: before "rld"
    c.char(ord("o"))
    assert c.text == "hello world" and c.caret == 8
    c.char(127)                                     # ctrl+backspace: a word
    assert c.text == "hello rld"
    c.char(1); c.char(ord("x"))                     # ctrl+a, then typing replaces it
    assert c.text == "x"
    assert c.char(13) == ("send", "x") and c.text == ""
    assert c.char(13) is None                       # nothing to send
    assert c.char(27) == ("close",)


def test_the_bubble_follows_the_conversation():
    from jevh.orb_chat import ChatState
    c = ChatState()
    old = {"id": 4, "message": "hi", "answer": "hello", "status": "done"}
    assert c.update({"busy": False, "last": old}) is False and c.bubble is None     # what was already answered is not news
    c.sent("17 times 23?")
    assert c.bubble["kind"] == "waiting" and c.bubble_open
    c.update({"busy": True, "last": {"id": 5, "message": "17 times 23?", "status": "running", "persona": "daily_assistant"}})
    assert c.bubble["status"] == "Working (daily assistant)"
    q = {"id": "q1", "question": "Go ahead?", "options": ["Yes", "No"], "secret": False}
    assert c.update({"busy": True, "last": {"id": 5, "status": "running"}, "question": q}) is True
    assert c.bubble["kind"] == "question" and c.bubble["options"] == ["Yes", "No"]
    assert c.char(ord("Y")) is None and c.char(13) == ("answer", "q1", "Y")       # typed answers go to the question
    c.answered()
    assert c.update({"busy": False, "last": {"id": 5, "message": "17 times 23?", "answer": "391", "status": "done",
                                             "persona": "daily_assistant"}}) is True
    assert c.bubble["kind"] == "reply" and c.bubble["text"] == "391" and c.awaiting is None
    assert c.update({"busy": False, "last": {"id": 5, "answer": "391", "status": "done"}}) is False   # shown once
    c.update({"busy": True, "question": {"id": "pw", "question": "Password?", "options": [], "secret": True}})
    assert c.secret                                                                 # the bar hides what is typed


def test_the_bar_and_bubble_are_drawn_with_their_clickable_parts():
    from jevh import orb_chat_art as a
    bar, regions = a.draw_bar("hello", 420, 1.5)
    assert bar.mode == "RGBA" and bar.getpixel((0, 0))[3] == 0 and "send" in regions        # see-through around the pill
    narrow, regions = a.draw_bar("hello", 50, 1.5)                                            # still sliding out: no text yet
    assert "send" not in regions
    img, regions, overflow = a.draw_bubble({"kind": "question", "text": "Go ahead?", "options": ["Yes", "No", "Later"]}, 1.0)
    assert {"close", "open", "option:0", "option:2"} <= set(regions) and overflow == 0
    long = "\n\n".join(f"Paragraph {i}: " + "words " * 40 for i in range(12))
    img, regions, overflow = a.draw_bubble({"kind": "reply", "text": long}, 1.0)
    assert overflow > 0 and img.height < 420                                                   # a long reply scrolls
    _, _, after = a.draw_bubble({"kind": "reply", "text": long}, 1.0, scroll=10 ** 6)
    assert after == overflow
    assert len(a.premultiplied(img)) == img.width * img.height * 4
