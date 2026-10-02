"""A stand-in for the user's own browser, for evals/bench/nango_setup.py.

webbrowser.open() runs the command in the BROWSER environment variable with the page's URL, so the bench
points it here. A page of this machine's Nango Connect UI (localhost:3009) whose text contains $JEVH_FAKE_CLICK
gets its Connect button clicked in a headless browser (what a user does on a sign-in that needs no login);
anything else is only logged to $JEVH_FAKE_LOG, as a user who "opened" it but didn't sign in.

    python fake_browser.py <url>
"""
import os
import sys
import time
from urllib.parse import urlparse

url = sys.argv[1] if len(sys.argv) > 1 else ""
log = os.environ.get("JEVH_FAKE_LOG")
click = os.environ.get("JEVH_FAKE_CLICK", "")


def note(msg: str) -> None:
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


note(f"open {url.split('session_token=')[0]}...")
host = urlparse(url)
if click and host.hostname in ("localhost", "127.0.0.1") and host.port == 3009:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True)
        page = b.new_page()
        page.goto(url)
        try:
            page.wait_for_selector(f"text={click}", timeout=15000)
            page.get_by_role("button", name="Connect").click(timeout=15000)
            page.wait_for_selector("text=Success", timeout=20000)
            note("clicked Connect: success")
        except Exception as e:  # noqa: BLE001
            note(f"could not click Connect: {type(e).__name__}")
        b.close()
else:
    note("not signed in (nothing to click)")
