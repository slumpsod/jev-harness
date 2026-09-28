"""Browser agent eval: Jev drives Chromium through a local test shop (evals/browser_site).

Each case gives a start page, one goal for do(), and what counts as right: the statuses that
are acceptable and a JS check of the page/app state afterwards. A false "done" (status done
but the check fails) is the worst outcome and is counted separately.

    python evals/browser_cases.py                 # local Jev (JEVH_JEV_URL / semif)
    python evals/browser_cases.py --cloud         # hosted Jev (TYPESAFE_API_KEY)
    python evals/browser_cases.py --live          # also run the public-web cases (Wikipedia)
    python evals/browser_cases.py --holdout       # only the cases written after tuning
    python evals/browser_cases.py -k login        # only cases whose name contains "login"
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

CART = "localStorage.setItem('cart', JSON.stringify([{id:'blue-mug',name:'Blue ceramic mug',size:'Large',qty:1}]))"
DONE = {"done", "likely_done"}
NOT_DONE = {"stuck", "ambiguous", "max_actions", "needs_value", "error", "blocked"}

# name, start page, goal, values, allowed statuses, check (JS -> bool), setup JS, allow_irreversible
CASES = [
    ("search", "index.html", "search the shop for mugs", {"query": "mug"}, DONE,
     "location.href.includes('search.html') && /mug/i.test(new URLSearchParams(location.search).get('q'))", None, False),
    ("open-result", "search.html?q=mug", "open the blue ceramic mug product page", None, DONE,
     "location.href.includes('id=blue-mug')", None, False),
    ("select-size", "product.html?id=blue-mug", "choose size Large", None, DONE,
     "document.getElementById('size').value === 'Large'", None, False),
    ("add-to-cart", "product.html?id=blue-mug", "add this mug to the cart in size Large", None, DONE,
     "JSON.parse(localStorage.getItem('cart')||'[]').some(i => i.size === 'Large')", None, False),
    ("login", "login.html", "sign in", {"email": "user@example.com", "password": "hunter22"}, DONE,
     "location.href.includes('account.html')", None, False),
    ("login-wrong", "login.html", "sign in", {"email": "user@example.com", "password": "wrongpass"}, {"error"},
     "!location.href.includes('account.html')", None, False),
    ("login-needed", "login.html", "sign in to the account", None, {"needs_login", "needs_value", "stuck"},
     "!location.href.includes('account.html')", None, False),
    ("settings-toggle", "settings.html", "turn off email notifications and save the settings", None, DONE,
     "JSON.parse(localStorage.getItem('settings')||'{}').email === false", None, False),
    ("settings-theme", "settings.html", "switch the theme to Dark and save", None, DONE,
     "JSON.parse(localStorage.getItem('settings')||'{}').theme === 'Dark'", None, False),
    ("checkout-guard", "checkout.html", "fill in the delivery details and place the order",
     {"name": "Sam Rivera", "address": "12 Elm St"}, {"needs_confirmation"},
     "!localStorage.getItem('order')", CART, False),
    ("checkout-allowed", "checkout.html", "fill in the delivery details and place the order",
     {"name": "Sam Rivera", "address": "12 Elm St"}, DONE,
     "JSON.parse(localStorage.getItem('order')||'null')?.address === '12 Elm St'", CART, True),
    ("dense-links", "articles.html", "open the article about tidal locking", None, DONE,
     "location.href.includes('Tidal+locking') || location.href.includes('Tidal%20locking')", None, False),
    ("already-done", "account.html", "sign in", None, DONE, "location.href.includes('account.html')",
     "localStorage.setItem('user', JSON.stringify('user@example.com'))", False),
    ("no-such-page", "index.html", "open the pricing page", None, NOT_DONE,
     "true", None, False),
    ("search-no-value", "index.html", "search the shop", None, {"needs_value", "stuck", "ambiguous"},
     "true", None, False),
    ("go-to-cart", "product.html?id=red-mug", "open the shopping cart", None, DONE,
     "location.href.includes('cart.html')", CART, False),
    # big-store product page (Amazon's patterns; see buybox.html), added after a live Amazon miss
    ("bb-add", "buybox.html", "add the coconut berry energy drink 12 pack to the cart", None, DONE,
     "store.get('cart', []).length === 1 && !localStorage.getItem('order')", None, False),
    ("bb-add-qty", "buybox.html", "add 3 of these to my cart", None, DONE,
     "store.get('cart', []).some(i => i.qty === 3) && !localStorage.getItem('order')", None, False),
    ("bb-cart", "buybox.html", "open the shopping cart", None, DONE,
     "location.href.includes('cart.html') && store.get('cart', []).length === 0", None, False),
]
# Written after the loop was tuned on CASES and run once without changes (--holdout).
HOLDOUT = [
    ("h-todo-add", "todo.html", "add a todo to water the plants", {"todo": "Water the plants"}, DONE,
     "JSON.parse(localStorage.getItem('todos')||'[]').some(t => t.t === 'Water the plants')", None, False),
    ("h-todo-check", "todo.html", "mark 'Call mom' as done", None, DONE,
     "JSON.parse(localStorage.getItem('todos')||'[]').find(t => t.t === 'Call mom')?.d === true", None, False),
    ("h-todo-delete", "todo.html", "delete the 'Book dentist' todo", None, DONE | {"needs_confirmation"},
     "true", None, False),
    ("h-tab", "tabs.html", "show the reviews", None, DONE,
     "!document.getElementById('rv').hidden", None, False),
    ("h-tab-read", "tabs.html", "open the specifications", None, DONE,
     "!document.getElementById('sp').hidden", None, False),
    ("h-cookie", "cookie.html", "dismiss the cookie banner without accepting tracking", None, DONE,
     "localStorage.getItem('consent') === 'essential'", None, False),
    ("h-radio", "shipping.html", "choose overnight shipping and continue", None, DONE,
     "localStorage.getItem('ship') === 'ovn'", None, False),
    ("h-gift", "shipping.html", "mark the order as a gift and continue to payment", None, DONE,
     "localStorage.getItem('gift') === 'true'", None, False),
    ("h-page3", "results.html", "go to page 3 of the job listings", None, DONE,
     "location.search.includes('page=3')", None, False),
    ("h-next", "results.html?page=2", "show the next page of jobs", None, DONE,
     "location.search.includes('page=3')", None, False),
    ("h-signup-short", "signup.html", "create an account", {"email": "sam@work.com", "password": "abc"},
     {"error"}, "!localStorage.getItem('account')", None, False),
    ("h-signup", "signup.html", "create an account and accept the terms",
     {"email": "sam@work.com", "password": "correct-horse-9"}, DONE,
     "localStorage.getItem('account') === 'sam@work.com'", None, False),
]
LIVE = [
    ("wiki-link", "https://en.wikipedia.org/wiki/Coffee", "open the Wikipedia article about espresso", None, DONE,
     "location.pathname === '/wiki/Espresso'", None, False),
    ("wiki-search", "https://en.wikipedia.org/wiki/Main_Page", "search Wikipedia for tidal locking and open the article",
     {"query": "tidal locking"}, DONE, "location.pathname === '/wiki/Tidal_locking'", None, False),
]


def serve() -> str:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    handler = functools.partial(Quiet, directory=str(ROOT / "browser_site"))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}/"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--holdout", action="store_true", help="run only the held-out cases")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("-k", default="")
    ap.add_argument("--buttons", action="store_true", help="print Jev's scores for every button it checks")
    args = ap.parse_args()
    if args.cloud:
        os.environ["JEVH_TOOLS_JEV"] = "cloud"
    from jevh.tools.browser import Session

    base = serve()
    s = Session(headed=args.headed)
    rows, ok, false_done, t_all = [], 0, 0, 0.0
    cases = HOLDOUT if args.holdout else CASES + HOLDOUT + (LIVE if args.live else [])
    cases = [c for c in cases if args.k in c[0]]
    for name, start, goal, values, allowed, check, setup, irrev in cases:
        url = start if start.startswith("http") else base + start
        if not start.startswith("http"):
            s.page.goto(base + "index.html")
            s.page.evaluate("localStorage.clear()")
            if setup:
                s.page.evaluate(setup)
        s.goto(url)
        t = time.monotonic()
        try:
            r = s.do(goal, values, allow_irreversible=irrev)
        except Exception as e:  # noqa: BLE001
            r = {"status": f"crash: {e}", "actions": [], "jev_calls": 0, "note": ""}
        dt = time.monotonic() - t
        t_all += dt
        passed_check = bool(s.page.evaluate(f"() => {{ try {{ return !!({check}) }} catch (e) {{ return false }} }}"))
        good = r["status"] in allowed and passed_check
        if r["status"] == "done" and not passed_check:
            false_done += 1
        ok += good
        rows.append({"case": name, "ok": good, "status": r["status"], "check": passed_check, "secs": round(dt, 1),
                     "calls": r["jev_calls"], "actions": r["actions"], "note": r.get("note", "")})
        mark = "PASS" if good else "FAIL"
        print(f"{mark} {name:17} {r['status']:18} check={passed_check!s:5} {dt:5.1f}s (jev {r.get('jev_seconds', 0):4.1f}s, "
              f"{r['jev_calls']:2} calls)  "
              f"{' > '.join(r['actions'])[:110]}", flush=True)
        if not good and r.get("note"):
            print(f"      note: {r['note'][:200]}")
        if args.buttons:
            for b in s.button_log:
                print(f"      button {b['element'][:50]:50} " + " ".join(f"{k}={v}" for k, v in b.items()
                                                                     if k not in ("goal", "element")))
        rows[-1]["buttons"] = list(s.button_log)
        s.button_log.clear()
    s.close()
    n = len(cases)
    print(f"\n{ok}/{n} correct, {false_done} false 'done', {t_all:.0f}s total, {t_all / max(n, 1):.1f}s per goal "
          f"({'cloud' if args.cloud else 'local ' + os.environ.get('JEVH_JEV_BACKEND', 'semif')})")
    out = ROOT / f"browser_{'cloud' if args.cloud else 'local'}_results.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
