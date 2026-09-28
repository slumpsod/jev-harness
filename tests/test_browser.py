"""The browser loop against the eval site, with scripted Jev answers (no model needed).
Skipped when Playwright's Chromium isn't installed."""
import functools
import http.server
import threading
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
from jevh.tools.browser import IRREVERSIBLE, REVERSIBLE, Session, field_matches  # noqa: E402

from conftest import FakeSystemOne, choice_ans, noul_ans  # noqa: E402

SITE = Path(__file__).resolve().parent.parent / "evals" / "browser_site"


@pytest.fixture(scope="module")
def site():
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(SITE)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.shutdown()


def scripted(click_label: str, done_when: str, unwanted: str | None = None):
    """Jev that clicks the element whose description contains `click_label` and says done
    once the page URL contains `done_when`. The goal "asks for" every button but `unwanted`."""
    def rules(k, q, s):
        page = (s or {}).get("page", {}) if isinstance(s, dict) else {}
        if k == "done":
            return noul_ans(0.97 if done_when in page.get("url", "") else 0.02)
        if k == "wanted":
            return noul_ans(0.02 if unwanted and unwanted in s.get("element", "") else 0.95)
        if q["type"] == "choice":
            hit = next((key for key, d in q["criteria"].items() if d and click_label in d), None)
            return choice_ans(hit or "none", 0.9, q["criteria"])
        if q["type"] == "noul":
            return noul_ans(0.02)
    return FakeSystemOne(rules)


@pytest.fixture(scope="module")
def session():
    try:
        s = Session(headed=False, jev=FakeSystemOne(), cap=16)
    except Exception as e:  # noqa: BLE001 - no Chromium in this environment
        pytest.skip(f"chromium unavailable: {e}")
    yield s
    s.close()


def test_values_fill_matching_fields_then_jev_clicks_submit(session, site):
    session.jev = scripted("button 'Sign in'", "account.html")
    session.goto(site + "login.html")
    r = session.do("sign in", {"email": "user@example.com", "password": "hunter22"})
    assert r["status"] == "done", r
    assert r["actions"][0].startswith("filled text field 'Email'") and "password 'Password'" in r["actions"][0]
    assert "account.html" in r["url"]


def test_irreversible_click_needs_confirmation(session, site):
    session.jev = scripted("button 'Place order'", "never")
    session.goto(site + "checkout.html")
    r = session.do("place the order", {"name": "Sam", "address": "12 Elm St"})
    assert r["status"] == "needs_confirmation" and "Place order" in r["note"]
    assert session.page.evaluate("localStorage.getItem('order')") is None


def test_field_matching_is_conservative():
    assert field_matches("email", {"kind": "text field", "label": "Work email"})
    assert field_matches("password", {"kind": "password", "label": "Choose a password"})
    assert field_matches("query", {"kind": "search box", "label": "Search products"})
    assert not field_matches("name", {"kind": "text field", "label": "Street address"})
    assert not field_matches("email", {"kind": "password", "label": "Password"})


def test_big_store_page_is_read_like_a_person_sees_it(session, site):
    """Amazon patterns (evals/browser_site/buybox.html): the real buy buttons are transparent
    inputs over styled text, a shortcut panel parked off-page repeats their names, and a
    60-option department dropdown used to fill the text Jev decides from."""
    session.goto(site + "buybox.html")
    obs = session.observe()
    labels = [e["label"] for e in obs["elements"]]
    assert "Add to cart" in labels and "Buy Now" in labels                  # overlay inputs count
    assert not any("shift, alt" in lab for lab in labels)                 # off-page panel doesn't
    assert "Alexa Skills" not in obs["text"] and "Luggage" not in obs["text"]  # option lists stay out
    assert "Only 16 left in stock" in session._view(obs)


def test_view_follows_the_scroll(session, site):
    session.goto(site + "buybox.html")
    session.page.set_viewport_size({"width": 1280, "height": 300})
    try:
        session.page.mouse.wheel(0, 400)
        obs = session.observe()
        assert obs["view_start"] > 0 and not session._view(obs, 40).startswith("MegaMart")
    finally:
        session.page.set_viewport_size({"width": 1280, "height": 900})


def test_unwanted_button_is_skipped_for_the_next_best(session, site):
    session.jev = scripted("button 'Add to cart'", "never", unwanted="Add to cart")
    session.goto(site + "buybox.html")
    session.page.evaluate("localStorage.clear()")
    session.do("open the shopping cart", max_actions=2)
    assert session.page.evaluate("localStorage.getItem('cart')") is None
    assert any(b["element"] == "button 'Add to cart'" and b["wanted"] < 0.4 for b in session.button_log)


def test_add_to_cart_never_needs_confirmation():
    for label in ("Add to cart", "Add to Cart", "Add all 3 to Cart, Red Bull", "Add to List", "Save for later",
                  "add to your basket"):
        assert REVERSIBLE.match(label) and not IRREVERSIBLE.search(label), label
    for label in ("Buy Now", "Place order", "Add payment method", "Proceed to checkout"):
        assert not REVERSIBLE.match(label), label
