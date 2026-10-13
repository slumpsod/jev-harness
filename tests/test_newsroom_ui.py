"""The Newsroom page in a real browser: add a widget by hand, see it fill, get a toast and a bell count when a new
item matches, change it, remove it. Skipped when Playwright's Chromium isn't installed."""
import pytest

pytest.importorskip("playwright.sync_api")
pytest.importorskip("uvicorn")

from test_dashboard_ui import REPLIES, browse, serve  # noqa: E402,F401
from test_newsroom import A, B, C, Feed, feed  # noqa: E402,F401


@pytest.fixture
def dashboard(cfg, feed):
    feed.set(A, B)
    yield from serve(cfg, REPLIES)


def test_the_newsroom_page_from_an_empty_board_to_an_alert(dashboard, feed, browse):
    browse(lambda page: _flow(page, dashboard, feed))


def _flow(page, url, feed):
    page.on("dialog", lambda d: d.accept())
    page.goto(url)
    page.wait_for_selector(".hello")
    page.click('.rail [data-page="newsroom"]')
    assert page.is_visible("#nrtitle") and not page.is_visible("#chat")
    page.wait_for_selector("#nr .empty")
    assert "Ask in the chat" in page.inner_text("#nr")
    page.click("#nr .nrex button >> nth=0")                                   # an example goes back to the chat and fills the box, it does not send
    assert page.is_visible("#chat") and "Nvidia" in page.input_value("#input") and not page.is_visible("#page-newsroom")
    assert page.locator(".turn").count() == 0

    page.click('.rail [data-page="newsroom"]')
    page.click('#nracts [data-nract="add"]')                                      # add by hand: the form comes from the kind
    page.click('#nrkinds [data-nrkind="rss"]')
    page.fill('#nrfields [name="p_url"]', feed.url)
    page.check('#nrfields [name="mode"][value="keywords"]')
    page.fill('#nrfields [name="keywords"]', "earnings")
    page.click("#nrok")
    page.wait_for_selector("#nr .nrw .nri")
    assert page.locator("#nr .nrw .nri").count() == 2
    assert "notifies on earnings" in page.inner_text("#nr .nrw")
    assert page.locator("#nr .nrw .nri a").first.get_attribute("rel") == "noopener noreferrer"

    feed.set(("Nvidia earnings call moved to Thursday", "https://ex.test/d", "Date change"), A, B)
    page.click('#nr .nrw [data-nract="refresh"]')                             # a new item that mentions earnings
    page.wait_for_selector(".toast")
    assert "Nvidia earnings call moved" in page.inner_text(".toast")
    page.wait_for_function("!document.getElementById('nrcount').hidden && document.getElementById('nrcount').textContent === '1'")
    assert "mentions earnings" in page.inner_text("#nrnotes")
    page.click('#nrnotes [data-nract="notesread"]')
    page.wait_for_function("document.getElementById('nrcount').hidden")

    page.click('#nr .nrw [data-nract="edit"]')
    page.fill('#nr .nredit [name="title"]', "Wire watch")
    page.select_option('#nr .nredit [name="size"]', "l")
    page.click('#nr .nredit button[type="submit"]')
    page.wait_for_selector('#nr .nrw[data-size="l"] h3')
    assert page.inner_text("#nr .nrw h3") == "Wire watch"

    page.click('#nr .nrw [data-nract="edit"]')                                # nothing was typed wrong: remove it
    page.click('#nr .nredit [data-nract="delete"]')
    page.wait_for_selector("#nr .empty")
    assert not page.errors


def test_the_newsroom_fits_a_phone(dashboard, feed, browse):
    def body(page):
        page.set_viewport_size({"width": 390, "height": 800})
        page.goto(dashboard)
        page.wait_for_selector(".hello")
        page.click('.rail [data-page="newsroom"]')
        page.click('#nracts [data-nract="add"]')
        page.click('#nrkinds [data-nrkind="rss"]')
        page.fill('#nrfields [name="p_url"]', feed.url)
        page.click("#nrok")
        page.wait_for_selector("#nr .nrw .nri")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")   # no sideways scroll
        assert page.evaluate("document.querySelector('#page-newsroom .back').getBoundingClientRect().top >= 0")   # the way back is on screen
        assert page.evaluate("document.getElementById('page-newsroom').getBoundingClientRect().bottom"
                             " <= document.querySelector('.rail').getBoundingClientRect().top + 1")       # and so is the tab bar
        assert not page.errors
    browse(body)


def test_a_turn_that_adds_a_widget_opens_the_newsroom_page_on_it(dashboard, feed, browse):
    def body(page):
        page.goto(dashboard)
        page.wait_for_selector(".hello")
        page.wait_for_function("NR.data !== null")
        page.evaluate("""async (url) => {
            const before = NR.data;    // the board as it was when the turn started; the turn's tool adds a widget
            await api('/api/newsroom/add', {method: 'POST', body: JSON.stringify({kind: 'rss', params: {url}, title: 'Wire'})});
            NR.data = before; NR.rev = -1;
            afterTurn({artifacts: []}, {ts: Date.now() / 1000});
        }""", feed.url)
        page.wait_for_function("document.getElementById('app').dataset.view === 'newsroom'")
        assert not page.is_visible("#chat")
        page.wait_for_selector("#nr .nrw.flash h3:has-text('Wire')")             # the new tile, marked
        page.click("#page-newsroom .back")
        assert page.is_visible("#chat")
        assert not page.errors
    browse(body)


def test_what_a_feed_says_is_shown_as_text_never_run(dashboard, feed, browse, monkeypatch):
    from jevh import newsroom as nr
    # (an X post is kept as written: a "<" in it must reach the page as a "<")
    monkeypatch.setattr(nr.KINDS["rss"], "fetch", lambda p: [
        {"title": '<img src=x onerror="window.pwned=1">Breaking', "url": "javascript:window.pwned=2", "uid": "1",
         "summary": "<script>window.pwned=3</script>", "source": "<b>src</b>"}])

    def body(page):
        page.goto(dashboard)
        page.wait_for_selector(".hello")
        page.click('.rail [data-page="newsroom"]')
        page.click('#nracts [data-nract="add"]')
        page.click('#nrkinds [data-nrkind="rss"]')
        page.fill('#nrfields [name="p_url"]', feed.url)
        page.click("#nrok")
        page.wait_for_selector("#nr .nrw .nri")
        page.wait_for_timeout(300)
        text = page.inner_text("#nr .nrw")
        assert '<img src=x onerror="window.pwned=1">Breaking' in text and "<script>window.pwned=3</script>" in text
        assert page.locator("#nr .nrw img").count() == 0 and page.locator("#nr .nrw script").count() == 0
        assert page.locator("#nr .nrw a").count() == 0                       # a javascript: address is not a link
        assert page.evaluate("window.pwned") is None
    browse(body)
