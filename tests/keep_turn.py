"""One turn of a task, for tests/test_browser.py: a docked, kept browser in its own process (a worker's REPL).
    python tests/keep_turn.py <dock dir> 1 <site>   # opens consent.html, types into App name, then waits to be killed
    python tests/keep_turn.py <dock dir> 2          # attaches, prints where it is and what the field holds, ends normally"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.update({"JEVH_BROWSER_DOCK": sys.argv[1], "JEVH_BROWSER_KEEP": "1"})
from jevh.tools.browser import Session  # noqa: E402

s = Session(headed=True, jev=object())
if sys.argv[2] == "1":
    s.page.goto(sys.argv[3] + "consent.html")
    s.page.fill("#appname", "kept across turns")
    print("TURN1", s.real, flush=True)
    time.sleep(120)
else:
    print("TURN2", s.real, s.page.url.rsplit("/", 1)[-1], s.page.input_value("#appname"), flush=True)
    s.close()
