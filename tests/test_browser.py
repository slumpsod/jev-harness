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


# -- browser.test: steps, the expectation check, script errors --------------------------------------
def scripted_check(click_label: str, done_text: str, shows_text: str | None):
    """Jev that clicks `click_label`, says done once the page text has `done_text`, and says the
    expectation holds when the page text has `shows_text`."""
    def rules(k, q, s):
        page = (s or {}).get("page", {}) if isinstance(s, dict) else {}
        text = page if isinstance(page, str) else page.get("text", "")
        if k == "done":
            return noul_ans(0.97 if done_text in text else 0.02)
        if k in ("shows", "holds"):
            return noul_ans(0.95 if shows_text and shows_text in text else 0.05)
        if k == "wanted":
            return noul_ans(0.95)
        if q["type"] == "choice":
            hit = next((key for key, d in q["criteria"].items() if d and click_label in d), None)
            return choice_ans(hit or "none", 0.9, q["criteria"])
        if q["type"] == "noul":
            return noul_ans(0.02)
    return FakeSystemOne(rules)


@pytest.fixture
def shots(tmp_path, monkeypatch):
    import jevh.tools.browser as browser
    monkeypatch.setattr(browser, "ARTIFACTS", tmp_path)
    return tmp_path


def test_a_passing_flow_on_a_local_app_may_place_an_order(session, site, shots):
    session.jev = scripted_check("button 'Place order'", "Thank you", "Order #4471")
    t = session.test(site + "checkout.html", "place the order", "the order is placed",
                     values={"name": "Sam", "address": "12 Elm St"})
    assert t["passed"], t
    assert t["steps"][0]["status"] == "done" and t["expect"]["shows"] >= 0.9
    assert session.page.evaluate("localStorage.getItem('order')") is not None   # local app: no confirmation
    assert t["screenshot"] is None and not t["js_errors"]


def test_an_unmet_expectation_fails_with_a_screenshot(session, site, shots):
    session.jev = scripted_check("button 'Place order'", "Thank you", "Order #9999")
    t = session.test(site + "checkout.html", "place the order", "order #9999 is placed",
                     values={"name": "Sam", "address": "12 Elm St"})
    assert not t["passed"] and t["why"] == "the page doesn't show that order #9999 is placed"
    assert t["screenshot"] and Path(t["screenshot"]).exists()


def test_a_step_that_cannot_be_done_stops_the_test(session, site, shots):
    session.jev = scripted_check("button 'Nothing like this'", "never", "anything")
    t = session.test(site + "checkout.html", ["open the admin panel", "delete everything"], "all gone")
    assert not t["passed"] and t["why"].startswith("step 'open the admin panel' ended")
    assert len(t["steps"]) == 1 and "shows" not in t["expect"]


def test_a_script_error_fails_an_otherwise_passing_test(session, tmp_path, shots):
    page = tmp_path / "broken.html"
    page.write_text("<!doctype html><title>Broken</title><button onclick=\"document.body.insertAdjacentHTML("
                    "'beforeend', '<p>Saved!</p>'); null.x\">Save</button>", encoding="utf-8")
    session.jev = scripted_check("button 'Save'", "Saved!", "Saved!")
    t = session.test(page.as_uri(), "save", "it says Saved!")
    assert not t["passed"] and "script error" in t["why"] and "null" in t["js_errors"][0].lower()


def test_irreversible_steps_still_need_confirmation_off_the_local_machine():
    from jevh.tools.browser import LOCAL_URL
    assert LOCAL_URL.match("http://localhost:3000/cart") and LOCAL_URL.match("http://127.0.0.1:8000")
    assert LOCAL_URL.match("file:///C:/site/index.html") and LOCAL_URL.match("https://shop.test/")
    assert not LOCAL_URL.match("https://www.amazon.com/") and not LOCAL_URL.match("https://localhost.evil.com/")
