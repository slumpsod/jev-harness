"""The browser loop against the eval site, with scripted Jev answers (no model needed).
Skipped when Playwright's Chromium isn't installed."""
import functools
import http.server
import threading
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
from jevh.tools.browser import Session, field_matches  # noqa: E402

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


def scripted(click_label: str, done_when: str):
    """Jev that clicks the element whose description contains `click_label` and says done
    once the page URL contains `done_when`."""
    def rules(k, q, s):
        page = (s or {}).get("page", {}) if isinstance(s, dict) else {}
        if k == "done":
            return noul_ans(0.97 if done_when in page.get("url", "") else 0.02)
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
