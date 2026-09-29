"""Live eval: browser.test's final check - does the page show the expected result?

After a test's steps, Jev reads the page and decides whether `expect` holds. A false pass hides
a broken app (the costly error); a false fail sends the worker to fix something that works.
Cases are what an app page's text looks like after a flow: carts, logins, forms, lists, totals -
with near misses (a wrong count or total, an error next to the expected words).

    python evals/expect_cases.py            # live, tuning cases
    python evals/expect_cases.py --holdout
    python evals/expect_cases.py --offline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevh.config import load_config  # noqa: E402
from jevh.decide import SystemOneClient, noul  # noqa: E402

RAW = Path(__file__).with_name("expect_cases_raw.json")
RAW_HOLDOUT = Path(__file__).with_name("expect_holdout_raw.json")

CART2 = ("Shop  Search  Cart (2)\nYour cart\nTrail Runner 2  Size 10  Qty 2  $178.00  Remove\n"
         "Subtotal (2 items): $178.00\nProceed to checkout")
CART_EMPTY = "Shop  Search  Cart (0)\nYour cart is empty.\nContinue shopping"
LOGIN_FAIL = "Sign in\nEmail  sam@example.com\nPassword\nInvalid email or password.\nSign in  Forgot password?"
DASHBOARD = "Dashboard  Projects  Settings  Sign out\nWelcome back, Sam\nYou have 3 open projects."
SENT = "Contact us\nThanks! Your message was sent. We'll reply within 2 days.\nBack to home"
FORM_ERR = "Contact us\nName  Sam\nEmail\nEmail is required.\nMessage  Hello there\nSend"
TODOS = "Todos\n[ ] Buy milk\n[ ] Call the dentist\n[x] Book flights\n2 items left  All  Active  Completed"
NO_RESULTS = "Search: xylophone\nNo results for \"xylophone\". Try a different search."
RESULTS = "Search: boots\n12 results\nTrail Boot GTX  $149\nWinter Boot  $119\nHiking Boot Lite  $99"
SAVED = "Settings\nDisplay name  Sam Rivera\nTheme  Dark\nSettings saved.\nSave"
SAVE_FAIL = "Settings\nDisplay name  Sam Rivera\nTheme  Dark\nCould not save settings: server error (500). Try again.\nSave"
ORDER = "Order confirmed\nOrder #10432\nTrail Runner 2 x 2  $178.00\nShipping  $0.00\nTotal  $178.00\nA receipt was emailed to sam@example.com."

CASES = [  # (expect, page, holds?)
    ("the cart has 2 items", CART2, True),
    ("the cart has 3 items", CART2, False),
    ("the cart subtotal is $178.00", CART2, True),
    ("the cart subtotal is $168.00", CART2, False),
    ("the cart is empty", CART_EMPTY, True),
    ("the cart is empty", CART2, False),
    ("the user is signed in", DASHBOARD, True),
    ("the user is signed in", LOGIN_FAIL, False),
    ("an error says the login details are wrong", LOGIN_FAIL, True),
    ("the dashboard shows 3 open projects", DASHBOARD, True),
    ("the message was sent", SENT, True),
    ("the message was sent", FORM_ERR, False),
    ("the form asks for an email address", FORM_ERR, True),
    ("'Buy milk' is on the list and not done", TODOS, True),
    ("'Book flights' is not done yet", TODOS, False),
    ("2 items are left to do", TODOS, True),
    ("search results for boots are listed", RESULTS, True),
    ("search results are listed", NO_RESULTS, False),
    ("the settings were saved", SAVED, True),
    ("the settings were saved", SAVE_FAIL, False),
    ("the theme is set to Dark", SAVE_FAIL, True),
    ("the order is confirmed with a total of $178.00", ORDER, True),
    ("the order total includes a shipping charge", ORDER, False),
    ("the page shows the user's order history", ORDER, False),
]

HOLDOUT = [
    ("the cart shows Trail Runner 2 in size 10", CART2, True),
    ("the cart shows Trail Runner 2 in size 9", CART2, False),
    ("the user is signed out", DASHBOARD, False),
    ("a password reset link is offered", LOGIN_FAIL, True),
    ("the reply time is promised within 2 days", SENT, True),
    ("the contact form was submitted successfully", FORM_ERR, False),
    ("all todos are done", TODOS, False),
    ("'Book flights' is marked done", TODOS, True),
    ("the cheapest boot costs $99", RESULTS, True),
    ("more than 20 results were found", RESULTS, False),
    ("an error says saving failed", SAVE_FAIL, True),
    ("a receipt was sent by email", ORDER, True),
    ("the order number is 10432", ORDER, True),
    ("the order number is 10423", ORDER, False),
]

QUESTIONS = {
    "shows": "Does `page` show that `expect` is true?",
    "holds": ("Going only by what `page` shows, is `expect` true - with every detail in `expect` (numbers, names, "
              "states) matching the page, and nothing on the page saying otherwise?"),
    "contradicts": "Does `page` show something that makes `expect` false, or leave out what `expect` needs?",
}


def run_live(cases, raw: Path) -> list[dict]:
    cfg = load_config()
    c = SystemOneClient(cfg.jev["url"], cfg.jev["backend"], timeout=300)
    rows = []
    for expect, page, _ in cases:
        r = c.ask({"expect": expect, "page": page}, {k: noul(q) for k, q in QUESTIONS.items()})
        rows.append({k: r[k].noul for k in QUESTIONS})
    raw.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return rows


def report(cases, rows) -> None:
    labels = [c[2] for c in cases]
    for (expect, _, lab), s in zip(cases, rows):
        print(f"  {'HOLDS' if lab else 'no   '}  " + "  ".join(f"{k}={v:.2f}" for k, v in s.items()) + f"  {expect[:60]}")
    for key in QUESTIONS:
        pass_if_high = key != "contradicts"
        print(f"\n== {key}")
        for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            passed = [(s[key] >= t) if pass_if_high else (s[key] < 1 - t) for s in rows]
            ok = sum(p and l for p, l in zip(passed, labels))
            false_pass = sum(p and not l for p, l in zip(passed, labels))
            print(f"  t={t:.1f}: true results passed {ok}/{labels.count(True)}, false results passed "
                  f"{false_pass}/{labels.count(False)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    cases, raw = (HOLDOUT, RAW_HOLDOUT) if a.holdout else (CASES, RAW)
    rows = json.loads(raw.read_text(encoding="utf-8")) if a.offline else run_live(cases, raw)
    report(cases, rows)


if __name__ == "__main__":
    main()
