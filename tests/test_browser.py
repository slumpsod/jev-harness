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


def test_docked_browser_is_mirrored_and_takes_the_users_input(tmp_path, monkeypatch):
    """Watch browser in the dashboard: no window; the dashboard attaches over CDP, shows the page
    and passes clicks and typing on to it."""
    from concurrent.futures import ThreadPoolExecutor
    from jevh.web.server import BrowserDock
    monkeypatch.setenv("JEVH_BROWSER_DOCK", str(tmp_path))
    own = ThreadPoolExecutor(1)   # the session's own thread (the module's session holds this one's Playwright)
    run = lambda fn: own.submit(fn).result(60)  # noqa: E731
    try:
        s = run(lambda: Session(headed=True, jev=FakeSystemOne(), cap=16))
    except Exception as e:  # noqa: BLE001 - no Chromium in this environment
        pytest.skip(f"chromium unavailable: {e}")
    dock = BrowserDock(lambda: tmp_path)
    try:
        assert s.headed and len(list(tmp_path.glob("*.json"))) == 1
        run(lambda: s.page.set_content("<input id=q style='position:fixed;left:0;top:0;width:100%;height:100%'>"))
        f = dock.frame()
        assert f["jpeg"].startswith(bytes([0xFF, 0xD8])) and f["url"] == "about:blank"   # a JPEG
        assert dock.send({"kind": "click", "x": 0.5, "y": 0.5}) and dock.send({"kind": "type", "text": "hi there"})
        assert dock.send({"kind": "key", "key": "Backspace"})
        assert run(lambda: s.page.eval_on_selector("#q", "e => e.value")) == "hi ther"
        with pytest.raises(ValueError):
            dock.send({"kind": "key", "key": "F12"})   # only plain editing/navigation keys
    finally:
        run(s.close)
        own.shutdown()
    assert not list(tmp_path.glob("*.json")) and dock.frame() is None


# -- live failures on real sites (evals/bench/browser_tasks.py), replayed on local pages -----------
def picks(*labels: str, done_when: str = "never"):
    """Jev that acts on the first element whose description contains one of `labels` (in order),
    and says done once the page text has `done_when`."""
    def rules(k, q, s):
        if k == "done":
            return noul_ans(0.97 if done_when in s["page"]["text"] else 0.02)
        if k == "wanted":
            return noul_ans(0.95)
        if q["type"] == "choice":
            for lab in labels:
                hit = next((key for key, d in q["criteria"].items() if d and lab in d), None)
                if hit:
                    return choice_ans(hit, 0.9, q["criteria"])
            return choice_ans("none", 0.9, q["criteria"])
        if q["type"] == "noul":
            return noul_ans(0.02)
    return FakeSystemOne(rules)


def test_a_new_tab_link_is_followed_once_and_its_bot_check_reported(session, site, monkeypatch):
    """Red Rocks: "Buy Tickets" opens AXS in a new tab. The tab registers after the click returned, so
    the old page looked unchanged, the click was banned as "no effect" and the next concert's link
    was clicked; "Buy" in the label also stopped it as if it were a payment."""
    import jevh.tools.browser as browser
    monkeypatch.setattr(browser, "WALL_WAIT", 0)
    session.jev = picks("glass-harbor")
    session.goto(site + "events.html")
    assert "link 'Buy Tickets' (in: Tue, Sep 29, 7:00 pm - Glass Harbor) (opens ticket.html?e=glass-harbor)" in \
        [browser.describe(e) for e in session.observe()["elements"]]
    try:
        r = session.do("open the tickets for Glass Harbor")
        assert r["status"] == "blocked" and "Just a moment" in r["note"], r
        # the link would open a new tab; it opens in this one (one page for the agent and the dock)
        assert r["actions"] == ["clicked link 'Buy Tickets'"], r["actions"]
        assert session.page.url.endswith("ticket.html?e=glass-harbor")
        assert len(session.context.pages) == 1
    finally:
        for p in session.context.pages[1:]:
            p.close()
        session.page = session.context.pages[0]


def test_labels_beside_react_fields_and_autocomplete_suggestions(session, site):
    """demoqa: the label is in the next column (so the field was named 'subjectsInput'), and a
    suggestion must be clicked - "done" with 'Maths' only typed selected nothing."""
    session.goto(site + "reactform.html")
    session.page.evaluate("localStorage.clear()")
    labels = {e["label"] for e in session.observe()["elements"] if e["kind"] == "text field"}
    assert {"First Name", "Date of Birth", "Subjects"} <= labels, labels
    session.jev = picks("option 'Maths'", "text field 'Subjects'", done_when="Maths")   # semif: done once it shows
    r = session.do("choose Maths as a subject", {"subject": "Maths"})
    # the one suggestion matching what was typed is picked right away, no Jev decision needed
    assert r["actions"][0] == "typed into text field 'Subjects'; picked suggestion 'Maths' (value 'subject')", r
    assert session.page.evaluate("localStorage.getItem('subjects')") == '["Maths"]'


def test_enter_is_never_pressed_in_a_filled_form_field(session, site):
    session.goto(site + "reactform.html")
    session.page.evaluate("localStorage.clear()")
    session.page.fill("#firstName", "Sam")
    session.jev = picks("text field 'First Name'")
    session.do("put Sam in the first name")
    assert session.page.evaluate("localStorage.getItem('submitted')") is None   # Enter would submit the form


def test_a_quoted_value_in_the_goal_is_typed(session, site):
    session.goto(site + "reactform.html")
    session.jev = picks("text field 'First Name'")
    r = session.do("type 'Rivera' into the first name field", max_actions=1)
    assert session.page.input_value("#firstName") == "Rivera", r


def test_a_mistyped_host_becomes_the_site_the_request_names(monkeypatch):
    from jevh.tools.browser import corrected_url
    monkeypatch.setenv("JEVH_TASK_HOSTS", "blazedemo.com www.target.com")
    assert corrected_url("https://blavedemo.com/reserve.php") == "https://blazedemo.com/reserve.php"
    assert corrected_url("https://blazemo.com") == "https://blazedemo.com"
    assert corrected_url("https://www.targt.com/s?x=1") == "https://www.target.com/s?x=1"
    assert corrected_url("https://example-nothing.com") is None


def test_identical_buttons_are_told_apart_by_their_row(session, site):
    """BlazeDemo: one 'Choose This Flight' per row, and Jev couldn't tell which was Aer Lingus's."""
    session.goto(site + "flights.html")
    chosen = [e for e in session.observe()["elements"] if e["label"] == "Choose This Flight"]
    assert [e["context"] for e in chosen] == ["43 Virgin America $472.56", "9696 Aer Lingus $200.98"]
    session.jev = picks("Aer Lingus", done_when="Flight 9696")
    r = session.do("choose the Aer Lingus flight")
    assert r["status"] == "done" and "Flight 9696" in r["page"], r


def test_refused_data_requests_are_reported(session, site):
    session.goto(site + "index.html")
    session.refused = [("127.0.0.1", 435)]
    assert session._refused_note() == "" and not session._refusing()     # one refusal can be anything
    session.refused += [("127.0.0.1", 435), ("127.0.0.1", 403), ("ads.example", 403)]
    assert "refused 3" in session._refused_note() and "HTTP 403, 435" in session._refused_note()
    assert session._refusing()
    session.refused = [("other.com", 435)] * 5                            # another site's refusals
    assert not session._refusing()
    session.refused = []


def test_the_module_works_after_the_worker_started_playwright_itself(site, tmp_path):
    """A worker that runs sync_playwright().start() in the REPL leaves an asyncio loop running there;
    every browser.goto() then raised "Sync API inside the asyncio loop" (live, Google Flights)."""
    import subprocess
    import sys
    code = ("from playwright.sync_api import sync_playwright\n"
            "pw = sync_playwright().start()\n"
            "import jevh.tools.browser as browser\n"
            f"print(browser.goto({site + 'events.html'!r})['title'])\n"
            "browser.close()\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                         cwd=str(Path(__file__).resolve().parent.parent))
    assert out.stdout.strip() == "Venue events", out.stderr[-800:]


def test_a_form_that_appears_after_load_is_waited_for(session, site):
    """Booking.com: the search form rendered 1.5 s after load; the destination value had no field yet
    and Jev clicked the 'Denver hotels' promo link instead."""
    session.jev = picks("link 'Denver hotels'")          # what Jev did when the field wasn't there
    session.goto(site + "late.html")
    r = session.do("choose Denver as the destination", {"destination": "Denver"}, max_actions=1)
    assert r["actions"] == ["filled text field 'Enter destination' (value 'destination')"], r


def test_press_keys_for_games(session, site):
    session.goto(site + "late.html")
    r = session.press(["ArrowUp", "ArrowLeft"], times=3, delay=0)
    assert r["changed"] and "Score: 24" in session.read()


def test_read_shows_what_the_fields_hold_and_a_prefilled_field_is_replaced(session, site):
    session.goto(site + "reactform.html")
    session.jev = picks("nothing")
    r = session.do("fill in the date of birth", {"Date of Birth": "12 Mar 1990", "First Name": "Sam"})
    assert r["status"] == "done", r          # both values went in; nothing asks to submit
    text = session.read()
    assert "text field 'Date of Birth' = '12 Mar 1990'" in text and "text field 'First Name' = 'Sam'" in text


def test_set_x_to_y_types_y():
    from jevh.tools.browser import SET_TO
    assert SET_TO.match("Set the Date of Birth to 12 March 1990").group(1) == "12 March 1990"
    assert SET_TO.match("change my phone number to 217-555-0142").group(1) == "217-555-0142"
    assert SET_TO.match("add this to the cart") is None


def test_a_date_the_picker_rejects_is_retried_in_another_format(session, site):
    """demoqa's picker turned "12 March 1990" back into today's date on blur, and do() said done."""
    session.goto(site + "datepicker.html")
    session.jev = picks("nothing")
    r = session.do("set the date of birth", {"Date of Birth": "12 March 1990"})
    assert session.page.input_value("#dob") == "12 Mar 1990", r
    assert "[as '12 Mar 1990']" in r["actions"][0] and r["status"] == "done"


def test_a_missing_page_is_an_error_and_html_is_there_for_scraping(session, site):
    r = session.goto(site + "no-such-page.html")
    assert r["status"] == "error" and "HTTP 404" in r["note"]
    session.goto(site + "flights.html")
    rows = session.html("tr")
    assert rows.count("<tr>") == 3 and "Aer Lingus" in rows


def test_a_lone_value_goes_into_the_only_empty_field(session, site):
    session.goto(site + "late.html")
    session.jev = picks("link 'Denver hotels'")
    r = session.do("search for Denver", {"From": "Denver"}, max_actions=1)
    assert r["actions"][0].startswith("filled text field 'Enter destination' (value 'From', the only empty field)"), r


def test_filling_a_form_never_submits_it_and_choices_are_clicked(session, site):
    """demoqa: "fill name, email, gender and mobile" filled them and then clicked Submit; "Hobbies":
    "Reading" was typed into a text box instead of ticking the checkbox."""
    session.goto(site + "reactform.html")
    session.page.evaluate("localStorage.clear()")
    session.jev = picks("button 'Submit'")               # Jev reaching for the form's button
    r = session.do("fill in the first name and hobby", {"First Name": "Sam", "Hobbies": "Reading"})
    assert "clicked checkbox 'Reading' (value 'Hobbies')" in r["actions"][0], r
    assert session.page.is_checked("#h2") and not session.page.is_checked("#h1")
    assert session.page.evaluate("localStorage.getItem('submitted')") is None, r


# -- the worker acting itself when Jev is stuck ------------------------------------------------------
def _num(listing: list[str], text: str) -> int:
    return int(next(s for s in listing if text in s).split("]")[0][1:])


def test_when_jev_is_stuck_the_worker_gets_candidates_and_clicks_one(session, site):
    session.jev = picks("nothing matches")               # Jev finds nothing to do
    session.goto(site + "flights.html")
    r = session.do("choose the Aer Lingus flight")
    assert r["status"] == "stuck" and "browser.click(i)" in r["note"], r
    assert any("9696 Aer Lingus" in c for c in r["candidates"]), r["candidates"]
    c = session.click(_num(r["candidates"], "9696 Aer Lingus"))
    assert c["status"] == "done" and c["changed"] and "Flight 9696" in c["page"], c


def test_a_direct_click_on_an_order_needs_confirmation(session, site):
    session.goto(site + "index.html")
    session.page.evaluate("localStorage.clear(); " + "localStorage.setItem('cart', JSON.stringify([{id:'blue-mug',"
                          "name:'Blue ceramic mug',size:'Large',qty:1}]))")
    session.goto(site + "checkout.html")
    r = session.click(_num(session.elements("Place order"), "Place order"))
    assert r["status"] == "needs_confirmation" and "allow_irreversible=True" in r["note"]
    assert session.page.evaluate("localStorage.getItem('order')") is None


def test_direct_fill_reads_back_refuses_passwords_and_offers_suggestions(session, site):
    session.goto(site + "login.html")
    assert session.fill(_num(session.elements("password"), "password"), "hunter22")["status"] == "refused"
    session.goto(site + "datepicker.html")
    r = session.fill(_num(session.elements("Date of Birth"), "Date of Birth"), "12 March 1990")
    assert r["status"] == "done" and "[as '12 Mar 1990']" in r["action"], r
    session.goto(site + "reactform.html")
    session.page.evaluate("localStorage.clear()")
    r = session.fill(_num(session.elements("Subjects"), "Subjects"), "s")   # several match: the list comes back
    assert len(r["suggestions"]) == 4 and "pick one with browser.click(i)" in r["note"], r
    assert session.click(_num(r["suggestions"], "Physics"))["status"] == "done"
    assert session.page.evaluate("localStorage.getItem('subjects')") == '["Physics"]'


def test_numbers_are_checked_and_kinds_are_explained(session, site):
    session.goto(site + "flights.html")
    with pytest.raises(ValueError, match=r"elements\(\) again"):
        session.click(999)
    button = _num(session.elements("Choose This Flight"), "Choose This Flight")
    assert "not a text field" in session.fill(button, "x")["note"]
    assert "not a dropdown" in session.select(button, "x")["note"]


def test_over_the_cap_what_is_on_screen_is_kept(session, tmp_path):
    """Booking.com: the calendar opens at the end of the page, and the first 300 controls in page order
    (hotel cards far below) left its days out."""
    links = "".join(f'<p><a href="#h{n}">Hotel {n}</a></p>' for n in range(400))
    page = tmp_path / "busy.html"
    page.write_text(f'<!doctype html><title>Busy</title><div style="margin-top:900px">{links}</div>'
                    '<div style="position:fixed;top:20px;left:20px;background:#fff"><button>Wednesday, October 14, 2026'
                    '</button><button>Friday, October 16, 2026</button></div>', encoding="utf-8")
    session.goto(page.as_uri())
    days = session.elements("October 1")
    assert len(days) == 2 and len(session.elements()) == 300, days
    assert session.click(_num(days, "October 14"))["status"] == "done"


def test_a_canvas_page_says_so_and_press_sees_the_board_change(session, site):
    """play2048.co: the worker searched the HTML for the board for 30 steps, and a move that scored
    nothing came back changed=False, so it decided the keys did nothing."""
    r = session.goto(site + "canvasgame.html")
    assert "drawn on a canvas" in r["note"] and "browser.look" in r["note"]
    assert "drawn on a canvas" in session.read() and session.html().startswith("<!-- jevh note:")
    p = session.press("ArrowRight", delay=0)
    assert p["changed"] and "Score: 0" in p["page"]      # the board moved; no text did


def test_look_asks_the_workers_model_about_a_screenshot(site, monkeypatch, shots):
    import jevh.tools.browser as browser
    asked = []
    monkeypatch.setattr(browser, "_host", lambda name, path, q: asked.append((name, path, q)) or "a red tile, 2nd column")
    try:
        browser.goto(site + "canvasgame.html")
        r = browser.look("where is the red tile?")
    finally:
        browser.close()
    assert r["answer"] == "a red tile, 2nd column" and Path(r["screenshot"]).is_file()
    name, path, q = asked[0]
    assert name == "look" and path == r["screenshot"] and q.endswith("where is the red tile?") and "right away" in q


def test_see_image_sends_the_picture_the_way_each_provider_takes_it(tmp_path):
    from jevh.llm import AnthropicLLM, LLMResult, see_image
    img = tmp_path / "s.png"
    img.write_bytes(b"\x89PNG fake")

    class Local:
        def chat(self, messages):
            self.content = messages[0]["content"]
            return LLMResult(text="4 tiles")
    local = Local()
    assert see_image(local, str(img), "how many?") == "4 tiles"
    assert local.content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    class Claude(AnthropicLLM):
        def __init__(self):
            pass

        def chat(self, messages):
            self.content = messages[0]["content"]
            return LLMResult(text="ok")
    claude = Claude()
    see_image(claude, str(img), "how many?")
    assert claude.content[0]["type"] == "image" and claude.content[0]["source"]["media_type"] == "image/png"


def test_a_search_the_site_throws_away_is_reported_as_blocked(session, site):
    """Booking.com toward a headless browser: searchresults.html?ss=Denver&checkin=... went on to a page with
    no query - no dates, no hotels, no error text - and the worker retried for 30 steps."""
    r = session.goto(site + "stays.html?ss=Denver&checkin=2026-10-14&checkout=2026-10-16")
    assert r["status"] == "blocked" and "checkin, checkout, ss" in r["note"] and "own browser" in r["note"], r
    session.goto(site + "stays.html")
    session.jev = picks("button 'Search'")
    r = session.do("search for stays in Denver", {"destination": "Denver"})
    assert r["status"] == "blocked" and "threw the search away" in r["note"], r
    assert session.goto(site + "search.html?q=mug&page=2")["status"] == "done"     # a kept search is fine
