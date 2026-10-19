"""Docked browser bench: does the dashboard's Browser panel show, and pass the user's clicks and typing to, the page the
user is acting on - through sign-in pop-ups, new tabs, redirects and a pop-up that closes itself - and does the worker's
`browser` follow the same page?

Live case (session 20261011-203023-3e47): "when I signed in on facebook the browser went back to its previous tab".
The panel showed whichever tab its Playwright connection listed last, and that order is not the order the tabs were
opened in: a fresh connection lists the first tab last. The panel reconnects after any failed frame (a screenshot taken
mid-navigation fails), so a sign-in in a second tab flipped the panel back to the first tab, and the user's typing went
there too. A kept browser's next turn picked its tab the same way.

Real Chrome (or Edge), started the way the dashboard starts it (docked: no window, a kept session browser on a
persistent profile, in a temporary folder), the real BrowserDock from jevh/web/server.py and the real browser.Session.
The pages are local (http://127.0.0.1, a sign-in form that takes any made-up address); nothing leaves this machine and no
model is called (`--e2e` adds full harness turns, below). The "user" acts only through the dock, as the panel does.

    python evals/bench/browser_dock.py                       # this checkout, every case, 3 times each
    python evals/bench/browser_dock.py --arm both            # also the code before the fix (BEFORE_REF, exported with git archive)
    python evals/bench/browser_dock.py --holdout             # only the held-out cases (written after the fix)
    python evals/bench/browser_dock.py -k popup --repeat 5   # cases whose name contains any of these
    python evals/bench/browser_dock.py --offline             # the checks again on the saved raw results (no browser)
    python evals/bench/browser_dock.py --e2e                 # + full turns: the worker (live models) opens the sign-in,
                                                             #   the bench signs in through the dock, the worker reports

Per case: whether the panel showed the sign-in page while the user acted, frames that jumped to another tab (each frame
the panel would have shown), whether the user's typing reached the form (the server records what was posted), the page
the panel and the worker end on, and errors. Raw results: browser_dock_raw.json; summary: browser_dock_summary.txt
(a live run writes to workspace/evals/<backend>/; --offline re-sweeps the committed ones; see evals/_out.py).
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
import _out  # noqa: E402  (evals/_out.py: where a run's results go)

RAW = HERE / "browser_dock_raw.json"
SUMMARY = HERE / "browser_dock_summary.txt"
BEFORE_REF = "dcef277"   # the code before the fix
EMAIL = "pat@example.test"   # made up: the form takes anything
SECRET = "not-a-real-password"

# ------------------------------------------------------------------ the site
FIELD = "position:fixed;left:5vw;width:90vw;height:12vh;font-size:5vh;box-sizing:border-box"
PAGES = {
    "start": """<title>Start</title><h1 style="margin-top:70vh">Start page</h1>
<a id=same href="/login?mode=same">Sign in here</a>
<a id=tab href="#" onclick="window.open('/login?mode=popup'); return false">Sign in (new tab)</a>
<a id=win href="#" onclick="window.open('/login?mode=popup', 'signin', 'width=520,height=700'); return false">Sign in (pop-up window)</a>
<a id=help href="#" onclick="window.open('/help'); return false">Open help (new tab)</a>
<a id=blank href="/help" target="_blank">Help (link to a new tab)</a>""",
    "login": f"""<title>Log in</title><form action="/auth" method="post">
<input type=hidden name=mode value="{{mode}}">
<input name=email placeholder=Email style="{FIELD};top:4vh">
<input name=pw type=password placeholder=Password style="{FIELD};top:24vh">
<button style="{FIELD};top:44vh">Log in</button></form>""",
    # what an OAuth pop-up does at the end: tell the opener where to go, then close itself
    "done": """<title>Signed in</title><p>Signed in, closing...</p><script>
const mode = new URLSearchParams(location.search).get('mode');
if (mode === 'popup' && window.opener) { window.opener.location = '/home?signed=1'; setTimeout(() => window.close(), 400); }
else location.replace('/home?signed=1');
</script>""",
    "home": """<title>Home</title><h1>Home: you are signed in</h1>""",
    "help": f"""<title>Help</title><h1>Help, page 1</h1><button onclick="location='/help2'" style="{FIELD};top:44vh">Next</button>
<button onclick="window.close()" style="{FIELD};top:64vh">Close this tab</button>""",
    "help2": """<title>Help 2</title><h1>Help, page 2</h1>""",
}


class Site:
    """The local pages. /auth answers slowly and redirects twice, as sign-ins do."""

    def __init__(self):
        self.posted: list[dict] = []
        site = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _page(self, name: str, **kw):
                body = PAGES[name].replace("{mode}", kw.get("mode", "")).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _go(self, where: str):
                self.send_response(302)
                self.send_header("Location", where)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self):
                u = urllib.parse.urlparse(self.path)
                q = dict(urllib.parse.parse_qsl(u.query))
                name = u.path.strip("/") or "start"
                if name == "auth2":
                    time.sleep(0.8)
                    return self._go(f"/done?mode={q.get('mode', '')}")
                if name in PAGES:
                    return self._page(name, mode=q.get("mode", ""))
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                form = dict(urllib.parse.parse_qsl(self.rfile.read(n).decode()))
                site.posted.append({"email": form.get("email", ""), "pw_ok": form.get("pw") == SECRET,
                                    "mode": form.get("mode", "")})
                time.sleep(1.2)   # (Facebook's sign-in takes a moment: the panel keeps taking pictures meanwhile)
                self._go(f"/auth2?mode={form.get('mode', '')}")

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def close(self):
        self.srv.shutdown()


# ------------------------------------------------------------------ the cases
# opener: the link the worker clicks on the start page; flow: what the user does in the panel; final: where both the
# panel and the worker should end up. reattach: the panel connects afresh while the sign-in is open (what a failed frame
# or a dashboard restart does). turn2: the task ends and the next turn attaches to the kept browser.
CASES = [
    {"name": "popup_tab_signin", "opener": "Sign in (new tab)", "flow": "signin", "final": "/home?signed=1"},
    {"name": "popup_tab_signin_reattach", "opener": "Sign in (new tab)", "flow": "signin", "final": "/home?signed=1",
     "reattach": True},
    {"name": "popup_window_signin", "opener": "Sign in (pop-up window)", "flow": "signin", "final": "/home?signed=1"},
    {"name": "same_tab_redirects", "opener": "Sign in here", "flow": "signin", "final": "/home?signed=1"},
    # regression: what already worked must keep working
    {"name": "same_tab_redirects_reattach", "opener": "Sign in here", "flow": "signin", "final": "/home?signed=1",
     "reattach": True},
    {"name": "new_tab_then_next_turn", "opener": "Open help (new tab)", "flow": "help_next", "final": "/help2",
     "turn2": True},
    # held out: written after the fix, not used to shape it
    {"name": "popup_window_signin_reattach", "opener": "Sign in (pop-up window)", "flow": "signin",
     "final": "/home?signed=1", "reattach": True, "holdout": True},
    {"name": "user_closes_new_tab", "opener": "Open help (new tab)", "flow": "help_close", "final": "/start",
     "holdout": True},
    {"name": "target_blank_link_turn2", "opener": "Help (link to a new tab)", "flow": "help_next", "final": "/help2",
     "turn2": True, "holdout": True},
]
INPUT_AT = {"email": 0.10, "pw": 0.30, "submit": 0.50, "next": 0.50, "close": 0.70}   # fraction of the panel's height


def pq(url: str | None) -> str | None:
    """A local page's path and query ("/home?signed=1"), the way the checks compare them."""
    if not url:
        return None
    u = urllib.parse.urlparse(url)
    return u.path + (f"?{u.query}" if u.query else "")


class NoJev:
    def ask(self, state, questions):
        raise RuntimeError("this bench makes no Jev decisions")


def child(case: dict, root: str, repeat: int) -> dict:
    """One case, in its own process, on the code in `root`."""
    sys.path.insert(0, root)
    from concurrent.futures import ThreadPoolExecutor

    from jevh.tools.browser import Session, close_kept
    from jevh.web.server import BrowserDock
    tmp = Path(tempfile.mkdtemp(prefix="jevh-dock-bench-"))
    sess_dir = tmp / "session"
    dock_dir = sess_dir / "browsers"
    dock_dir.mkdir(parents=True)
    # the dashboard's way: docked, kept between turns, on a profile that stays signed in (nango.browser_mode())
    os.environ.update(JEVH_BROWSER_DOCK=str(dock_dir), JEVH_BROWSER_KEEP="1", JEVH_BROWSER_PROFILE=str(tmp / "profile"),
                      JEVH_BROWSER_PAUSE="0")
    site = Site()
    own = ThreadPoolExecutor(1)
    run = lambda fn: own.submit(fn).result(90)  # noqa: E731
    out = {"case": case["name"], "rep": repeat, "root": root, "frames": [], "errors": []}
    phase = ["start"]
    t0 = time.monotonic()
    docks = [BrowserDock(lambda: dock_dir)]
    stop = threading.Event()

    def poll():   # the dashboard's Browser tab asks for a picture every 250 ms
        while not stop.is_set():
            try:
                f = docks[-1].frame()
                out["frames"].append({"t": round(time.monotonic() - t0, 2), "phase": phase[0],
                                      "url": pq(f["url"]) if f else None})
            except Exception as e:  # noqa: BLE001 - what the endpoint does: no picture this time
                out["frames"].append({"t": round(time.monotonic() - t0, 2), "phase": phase[0], "url": None,
                                      "error": type(e).__name__})
            time.sleep(0.25)

    def shown(want: str, secs: float = 10) -> float | None:
        """Seconds until the panel shows a page whose path+query starts with `want` (None: it never did)."""
        start = time.monotonic()
        while time.monotonic() - start < secs:
            fr = [f for f in out["frames"] if f["t"] >= start - t0 and f.get("url")]
            if fr and fr[-1]["url"].startswith(want):
                return round(time.monotonic() - start, 2)
            time.sleep(0.1)
        return None

    def user(kind: str, at: str | None = None, **kw):
        ev = {"kind": kind, "x": 0.5, "y": INPUT_AT[at] if at else 0.5, **kw}
        try:
            return bool(docks[-1].send(ev))
        except Exception as e:  # noqa: BLE001
            out["errors"].append(f"input {kind}: {type(e).__name__}: {str(e)[:120]}")
            return False

    s = None
    poller = threading.Thread(target=poll, daemon=True)
    try:
        s = run(lambda: Session(headed=True, jev=NoJev(), cap=16))
        out["real_chrome"] = bool(getattr(s, "real", False))
        poller.start()
        run(lambda: s.goto(site.url + "/start"))
        if shown("/start") is None:
            out["errors"].append("the panel never showed the start page")
        phase[0] = "worker_opens"
        listing = run(lambda: s.elements(case["opener"]))
        i = int(listing[0].split("]")[0].strip("["))
        r = run(lambda: s.click(i, allow_irreversible=True))
        out["worker_click"] = r.get("action")
        first = "/login" if case["flow"] == "signin" else "/help"
        out["panel_showed_first"] = shown(first)
        phase[0] = "user_acts"
        if case.get("reattach"):   # the panel connects again (a frame that failed mid-navigation, a dashboard restart)
            at = time.monotonic() - t0
            docks.append(BrowserDock(lambda: dock_dir))
            for _ in range(40):   # the new connection's first picture
                got = [f["url"] for f in out["frames"] if f["t"] > at + 0.3 and f.get("url")]
                if got:
                    break
                time.sleep(0.25)
            out["after_reattach"] = got[0] if got else None
        if case["flow"] == "signin":
            user("click", "email")
            user("type", text=EMAIL)
            user("click", "pw")
            user("type", text=SECRET)
            user("click", "submit")
        elif case["flow"] == "help_next":
            user("click", "next")
        elif case["flow"] == "help_close":
            user("click", "close")
        phase[0] = "settling"
        out["panel_final_secs"] = shown(case["final"], 15)
        # what browser.url() says half a second after the panel shows the end, with no other tool call in between (live:
        # asked after the user's "Done", url() gave the closed pop-up's address; Playwright's page.url is only as new as
        # the last event the tool's connection handled)
        time.sleep(0.5)
        try:
            # browser.url(): Session.address, or the page's own url before the fix
            out["worker_url_at_once"] = pq(run(lambda: s.address() if hasattr(s, "address") else s.page.url))
        except Exception as e:  # noqa: BLE001
            out["errors"].append(f"worker url: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
        time.sleep(1.0)
        phase[0] = "worker_reads"
        try:
            out["worker_url"] = pq(run(lambda: s.observe()["url"]))
        except Exception as e:  # noqa: BLE001
            out["worker_url"] = None
            out["errors"].append(f"worker observe: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
        if case.get("turn2"):   # the task ends; the next turn of the session attaches to the kept browser
            run(s.close)
            s = run(lambda: Session(headed=True, jev=NoJev(), cap=16))
            docks.append(BrowserDock(lambda: dock_dir))   # (the panel attaches again: a new info file)
            phase[0] = "turn2"
            time.sleep(1.5)
            try:
                out["turn2_worker_url"] = pq(run(lambda: s.observe()["url"]))
            except Exception as e:  # noqa: BLE001
                out["turn2_worker_url"] = None
                out["errors"].append(f"turn2 observe: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
            out["turn2_panel_secs"] = shown(case["final"], 6)
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")
    finally:
        stop.set()
        poller.join(5)
        out["posted"] = site.posted
        if s is not None:
            try:
                run(s.close)
            except Exception:  # noqa: BLE001
                pass
        close_kept(sess_dir)
        own.shutdown(wait=False)
        site.close()
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return out


# ------------------------------------------------------------------ checks
def grade(case: dict, r: dict) -> dict:
    """The checks, from the raw result alone (so --offline can run them again)."""
    final = case["final"]
    frames = r.get("frames", [])
    acting = [f for f in frames if f["phase"] in ("user_acts", "settling") and f.get("url")]
    first = "/login" if case["flow"] == "signin" else "/help"
    # a jump: while the user acts in the page that opened, a picture of another page that is neither on the way (the
    # sign-in's own redirects) nor the end (the opener at its new address)
    ok_paths = (first, "/auth", "/done", final) if case["flow"] == "signin" else (first, "/help2", final)
    jumps = [f for f in acting if not f["url"].startswith(ok_paths)]
    checks = {
        "panel_showed_page": r.get("panel_showed_first") is not None,
        "no_jumps": not jumps,
        "panel_ends_right": r.get("panel_final_secs") is not None,
        "worker_follows": (r.get("worker_url") or "").startswith(final),
        "worker_url_at_once": (r.get("worker_url_at_once") or "").startswith(final) if "worker_url_at_once" in r else True,
        "no_errors": not r.get("errors"),
    }
    if case["flow"] == "signin":
        p = r.get("posted") or []
        checks["typing_reached_form"] = bool(p) and p[-1]["email"] == EMAIL and p[-1]["pw_ok"]
    if case.get("reattach"):
        checks["reattach_shows_page"] = (r.get("after_reattach") or "").startswith(first)
    if case.get("turn2"):
        checks["turn2_worker_on_page"] = (r.get("turn2_worker_url") or "").startswith(final)
        checks["turn2_panel_on_page"] = r.get("turn2_panel_secs") is not None
    return {"checks": checks, "pass": all(checks.values()), "jumps": len(jumps),
            "jumped_to": sorted({f["url"] for f in jumps})[:4]}


def summarize(results: list[dict], title: str) -> str:
    by = {c["name"]: c for c in CASES}
    lines = [title, ""]
    arms = sorted({r["arm"] for r in results})
    for arm in arms:
        rs = [r for r in results if r["arm"] == arm]
        lines.append(f"== {arm} ({rs[0]['root'] if rs else ''})")
        total = ok = 0
        for name in [c["name"] for c in CASES]:
            mine = [r for r in rs if r["case"] == name]
            if not mine:
                continue
            g = [grade(by[name], r) for r in mine]
            passed = sum(x["pass"] for x in g)
            total += len(g)
            ok += passed
            failing = sorted({k for x in g for k, v in x["checks"].items() if not v})
            jumps = sum(x["jumps"] for x in g)
            to = sorted({u for x in g for u in x["jumped_to"]})
            lines.append(f"  {name:32s} {passed}/{len(g)} pass" + (" (held out)" if by[name].get("holdout") else "")
                         + (f"  failing: {', '.join(failing)}" if failing else "")
                         + (f"  jump frames: {jumps} (to {', '.join(to)})" if jumps else ""))
            lines.append("      " + ", ".join(f"{k} {sum(x['checks'].get(k, False) for x in g)}/{len(g)}"
                                         for k in g[0]["checks"]))
            errs = sorted({e for r in mine for e in r.get("errors", [])})
            for e in errs[:3]:
                lines.append(f"      error: {e}")
        lines.append(f"  total: {ok}/{total} ({100 * ok / max(total, 1):.0f}%)")
        lines.append("")
    return "\n".join(lines)


def export_before(ref: str) -> str:
    """The code at `ref`, unpacked into a temporary folder (git archive: no worktree, nothing checked out here)."""
    dest = Path(tempfile.gettempdir()) / f"jevh-dock-before-{ref}"
    if not (dest / "jevh").is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        tar = dest / "src.tar"
        subprocess.run(["git", "-C", str(REPO), "archive", "-o", str(tar), ref, "jevh"], check=True)
        with tarfile.open(tar) as t:
            t.extractall(dest, filter="data")
        tar.unlink()
    return str(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", default="after", choices=["before", "after", "both"])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("-k", default="", help="comma-separated: cases whose name contains any of these")
    ap.add_argument("--holdout", action="store_true", help="only the held-out cases")
    ap.add_argument("--offline", action="store_true", help="grade the saved raw results again")
    ap.add_argument("--e2e", action="store_true", help="also full harness turns with the live models (evals/bench/browser_dock_e2e.py)")
    ap.add_argument("--child")
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--rep", type=int, default=0)
    _out.add_args(ap)
    a = ap.parse_args()
    dest = _out.Out.of(a)
    if a.child:
        case = next(c for c in CASES if c["name"] == a.child)
        print("RESULT " + json.dumps(child(case, a.root, a.rep)))
        return
    if a.offline:
        raw = dest.read(RAW)
        results = json.loads(raw.read_text(encoding="utf-8"))
        print(summarize(results, f"browser_dock (offline re-grade of {dest.where(raw)})"))
        return
    cases = [c for c in CASES if bool(c.get("holdout")) == a.holdout or (not a.holdout and a.k)]
    if a.k:
        cases = [c for c in CASES if any(k in c["name"] for k in a.k.split(","))]
    arms = {"before": [("before", export_before(BEFORE_REF))], "after": [("after", str(REPO))],
            "both": [("before", export_before(BEFORE_REF)), ("after", str(REPO))]}[a.arm]
    results = []
    started = time.time()
    for rep in range(a.repeat):
        for case in cases:
            for arm, root in arms:
                t = time.time()
                p = subprocess.run([sys.executable, __file__, "--child", case["name"], "--root", root, "--rep", str(rep)],
                                   capture_output=True, text=True, timeout=240, cwd=str(REPO), encoding="utf-8",
                                   errors="replace")
                line = next((x for x in p.stdout.splitlines() if x.startswith("RESULT ")), None)
                r = json.loads(line[7:]) if line else {"case": case["name"], "rep": rep, "root": root, "frames": [],
                                                       "errors": [f"child failed: {p.stderr.strip()[-300:]}"]}
                r.update(arm=arm, seconds=round(time.time() - t, 1))
                results.append(r)
                g = grade(case, r)
                print(f"{arm:6s} {case['name']:32s} rep {rep}: {'pass' if g['pass'] else 'FAIL'} "
                      f"{[k for k, v in g['checks'].items() if not v] or ''} ({r['seconds']}s)", flush=True)
    raw = dest.write(RAW)
    old = json.loads(raw.read_text(encoding="utf-8")) if raw.is_file() else []   # (a later run of a case replaces it)
    keep = {(r["arm"], r["case"], r["rep"]) for r in results}
    raw.write_text(json.dumps([r for r in old if (r["arm"], r["case"], r["rep"]) not in keep] + results, indent=1),
                   encoding="utf-8")
    text = summarize(results, f"browser_dock: {len(results)} runs in {time.time() - started:.0f}s, "
                              f"{time.strftime('%Y-%m-%d %H:%M')}, Chrome via {sys.executable}")
    summary = dest.write(SUMMARY)
    summary.write_text(text + "\n", encoding="utf-8")
    print("\n" + text + f"\n\nraw: {dest.where(raw)}   summary: {dest.where(summary)}")
    if a.e2e:
        where = ["--save-committed"] if a.save_committed else ["--out", str(dest.dir)]
        subprocess.run([sys.executable, str(HERE / "browser_dock_e2e.py"), "--repeat", str(a.repeat), *where], cwd=str(REPO))


if __name__ == "__main__":
    main()
