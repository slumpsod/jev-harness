"""The Cron page in a real browser: add a job (with the schedule explained as it is typed), run it, read what the run did,
pause, edit and remove it. Skipped when Playwright's Chromium isn't installed."""
import pytest

pytest.importorskip("playwright.sync_api")
pytest.importorskip("uvicorn")

from test_dashboard_ui import browse, serve  # noqa: E402,F401
from conftest import py  # noqa: E402


@pytest.fixture
def dashboard(cfg):
    yield from serve(cfg, [py("final('Mail summarized: nothing urgent.')")])


def test_the_cron_page_from_an_empty_list_to_a_finished_run(dashboard, browse):
    browse(lambda page: _flow(page, dashboard))


def _flow(page, url):
    page.on("dialog", lambda d: d.accept())
    page.goto(url)
    page.wait_for_selector(".hello")
    page.click('.rail [data-page="cron"]')
    assert page.is_visible("#crtitle") and not page.is_visible("#chat")
    page.wait_for_selector("#cr .empty")
    page.click("#cr .crex button >> nth=0")                                    # an example opens the form with its request filled in
    assert "unread mail" in page.input_value('#crform [name="prompt"]')
    page.fill('#crform [name="name"]', "Morning mail")
    page.fill('#crform [name="schedule"]', "whenever")
    page.wait_for_selector("#crsay.err")
    assert "could not read" in page.inner_text("#crsay")
    page.fill('#crform [name="schedule"]', "weekdays at 8:30")
    page.wait_for_function("document.getElementById('crsay').textContent.startsWith('weekdays at 8:30 AM')")
    page.fill('#crform [name="schedule"]', "* * * * *")
    page.wait_for_selector("#crsay.err")
    page.click("#crok")                                                         # (too often: refused, the form stays)
    page.wait_for_selector("#crerr:not([hidden])")
    assert "more often" in page.inner_text("#crerr") and page.is_visible("#crdlg")
    page.fill('#crform [name="schedule"]', "weekdays at 8:30")
    page.click("#crok")
    page.wait_for_selector("#cr .crc")
    card = page.inner_text("#cr .crc")
    assert "Morning mail" in card and "weekdays at 8:30 AM" in card and "never run" in card

    page.click('#cr [data-cract="run"]')                                        # run it now: the answer and the session are kept
    page.wait_for_selector("#cr .crc .badge.ok")
    page.wait_for_selector(".toast")
    assert "Morning mail: ok" in page.inner_text(".toast")
    page.wait_for_function("document.getElementById('crcount').textContent === '1'")
    page.click('#cr [data-cract="runs"]')
    page.wait_for_selector("#cr .crrun")
    assert "Mail summarized: nothing urgent." in page.inner_text("#cr .crrun") and "run by hand" in page.inner_text("#cr .crrun")
    page.wait_for_function("document.getElementById('crcount').hidden")      # (looked at)
    assert page.locator('#cr [data-cract="session"]').count() == 1

    page.click('#cr [data-cract="toggle"]')                                     # pause, edit, remove
    page.wait_for_selector("#cr .crc.paused")
    assert "paused" in page.inner_text("#cr .crstat")
    page.click('#cr [data-cract="edit"]')
    page.fill('#crform [name="name"]', "Mail check")
    page.click("#crok")
    page.wait_for_selector("#cr .crc h3:has-text('Mail check')")
    page.click('#cr [data-cract="delete"]')
    page.wait_for_selector("#cr .empty")
