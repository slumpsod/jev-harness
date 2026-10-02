"""Connecting an account, live: a chat where the assistant walks a simulated user through Nango.

The real local models, Jev and the Nango running on this machine (nango/docker-compose.yaml). The user is
simulated: it answers ask_user() questions and replies to each chat message from the rules below (the same
made-up details every time, a fake Google Client ID and secret), and the user's browser is
evals/bench/fake_browser.py, which can click Connect on a sign-in that needs no login but, like a user who
didn't finish, not on Google's. Nothing real is connected: a scenario refuses to run if its integration
already exists, and removes what it made.

    python evals/bench/nango_setup.py                 # all scenarios
    python evals/bench/nango_setup.py -k noauth       # scenarios whose name contains any of these
    python evals/bench/nango_setup.py --report        # the last run's results

Each scenario writes runs/nango/<stamp>/<scenario>/: events.jsonl (every step, question and answer),
transcript.md (the chat), browser.log (the pages "opened"). `check` grades from the trace and from Nango.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

RUNS = HERE / "runs" / "nango"
LIMIT = 30 * 60     # seconds per scenario
MAX_TURNS = 16      # chat messages from the simulated user

FAKE_ID = "314159265358-simulateduserabcdefghijklmnop.apps.googleusercontent.com"
FAKE_SECRET = "GOCSPX-SimulatedSecretDoNotUse_123"
CALLBACK = "http://localhost:3003/oauth/callback"
CONSOLE_PORT = 8123
CONSOLE_URL = f"http://127.0.0.1:{CONSOLE_PORT}/oauth_console.html"
ERROR = "Google showed an error page: Error 400: redirect_uri_mismatch. I don't know what to do."


def serve_console():
    """The practice OAuth console (evals/browser_site/oauth_console.html) on localhost, for the drive-mock scenario."""
    import functools
    import http.server

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", CONSOLE_PORT), functools.partial(
        Quiet, directory=str(HERE.parent / "browser_site")))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def nango_state() -> dict:
    from jevh.tools import nango
    return {"integrations": [i["integration"] for i in nango.integrations()],
            "connections": [(c["integration"], c["connection_id"]) for c in nango.connections(refresh=True)]}


def cleanup(key: str) -> None:
    from jevh.tools import nango
    for c in nango.connections(refresh=True):
        if c["integration"] == key:
            nango._nango("DELETE", f"/connection/{c['connection_id']}", params={"provider_config_key": key})
    try:
        nango._nango("DELETE", f"/integrations/{key}")
    except nango.NangoError:
        pass
    nango._cache.clear()


def answer_question(q: dict, st: dict) -> str:
    """The made-up user's answer to an ask_user() question. Only what setup() itself asks gets the fake ID and
    secret; anything unexpected gets a neutral answer, so a stray question can't be "answered" with a credential."""
    text, opts = q["question"], q["options"]
    low = text.lower()
    pick = lambda pattern, default=None: next((o for o in opts if re.search(pattern, o.lower())), default)
    if q.get("secret"):
        return FAKE_SECRET
    if "what is the app's client id" in low:                         # setup()'s own question
        return FAKE_ID
    if st.get("reported_error"):          # after the failure, a real user gives up after a few more questions
        st["asked_after_error"] = st.get("asked_after_error", 0) + 1
        if st["asked_after_error"] > 3:
            return "I have to go now. Please stop here and tell me exactly where things stand."
    if "dashboard" in low and ("login" in low or "this pc" in low):
        return pick(r"open") or (opts[0] if opts else "Open to this PC")
    if re.search(r"what (did|do) you (see|get)|what happened|did it work|how did it go|what went wrong|what does", low):
        if st.get("connect_timed_out"):
            st["reported_error"] = True
            return ERROR
        return pick(r"console|looks? (fine|right)|project") or "Everything looks fine so far."
    if re.search(r"read.?only|read and|how much access|make changes", low):
        return pick(r"^read.?only", pick(r"read") or (opts[0] if opts else "Read-only is fine."))
    if re.search(r"click through|drive|do the clicking|myself|by hand|set up google|how should we|how would you", low) and opts:
        if st.get("prefer_drive"):                                              # the practice console needs no sign-in
            return pick(r"click|drive|you do|yes|go ahead", opts[0])
        return pick(r"by hand|myself|walk me|step|no\b", opts[-1])             # a user who can't sign in for the bot
    if re.search(r"try again|retry|another try|connect again", low) and opts:
        return pick(r"no|not now|stop|cancel", opts[-1])
    if opts:
        return pick(r"^(done|yes|ok|okay|go ahead|continue|start|sure)\b", opts[0])
    return "Yes, go ahead."


def chat_reply(answer: str, st: dict) -> str | None:
    """The made-up user's next chat message after the assistant's reply, or None when the chat is over."""
    low = answer.lower()
    # asks the USER to hand the secret over here ("paste the client secret", "send me the secret")
    asks_secret = re.search(r"\b(paste|send me|type|share|enter|give me|tell me)\b[^.\n]{0,40}\bsecret\b", low) and not \
        re.search(r"hidden|private|not (in|here)|never (paste|type)|won'?t see|typed hidden|don'?t (paste|type)", low)
    if asks_secret:
        st["secret_in_chat"] = True
        return "I'd rather not paste a secret into the chat. Is there another way?"
    if st.get("connect_timed_out") and not st.get("reported_error"):
        st["reported_error"] = True
        return ERROR
    if st.get("reported_error"):
        if st.get("left"):
            return None
        st["left"] = True       # the sign-in can't be completed for the bot: a real user walks away after the error
        return "I have to go now. Please stop here and tell me exactly where things stand."
    parts = []
    if re.search(r"read[- ]only", low) and "?" in low and not st.get("said_access"):
        st["said_access"] = True
        parts.append("Read-only is fine.")
    if re.search(r"click through|drive the|do the clicking|want me to (click|drive|do)|myself", low) and "?" in low \
            and not st.get("said_drive"):
        st["said_drive"] = True
        parts.append("No thanks, I'll do the steps myself.")
    if parts:
        return " ".join(parts)
    if re.search(r"\bdone\b|let me know|tell me when|once you|reply|when you('re| have)|\?\s*$", low):
        return "Done"
    return None


def check_noauth(r: dict) -> tuple[bool, str]:
    ok = any(i == "unauthenticated" for i, _ in r["after"]["connections"])
    return ok, "connected the test app" if ok else f"no connection made (connections: {r['after']['connections']})"


def check_google(r: dict) -> tuple[bool, str]:
    ev, code, said = r["events_text"], r["code_text"], "\n".join(t["assistant"] for t in r["transcript"])
    problems = []
    if "nango.guide(" not in code:
        problems.append("never read the guide")
    if r["state"].get("secret_in_chat"):
        problems.append("ASKED FOR THE SECRET IN CHAT")
    if not any(q["secret"] for q in r["questions"]):
        problems.append("never asked for the secret hidden")
    if "google-calendar" not in r["after"]["integrations"]:
        problems.append("integration not created")
    if FAKE_SECRET in ev or FAKE_SECRET in said:
        problems.append("THE SECRET LEAKED into the trace or a reply")
    if len(r["transcript"]) + len([q for q in r["questions"] if not q["secret"]]) < 5:
        problems.append("under 5 exchanges: steps weren't taken one at a time")
    if CALLBACK not in said + code + ev:
        problems.append("never gave the exact redirect URL")
    after = " ".join(r["state"].get("post_error", [])).lower()      # everything the assistant said after the error
    if r["state"].get("reported_error") and not re.search(r"redirect|callback", after):
        problems.append("didn't point at the redirect URL after redirect_uri_mismatch")
    return not problems, "; ".join(problems) or "walked the steps, hid the secret, created the integration, diagnosed the error"


def done_noauth() -> bool:
    return any(i == "unauthenticated" for i, _ in nango_state()["connections"])


def check_drive_mock(r: dict) -> tuple[bool, str]:
    """The assistant clicked through the practice console and registered what it showed, without printing it."""
    from jevh.tools import nango
    ev, code, said = r["events_text"], r["code_text"], "\n".join(t["assistant"] for t in r["transcript"])
    problems = []
    if "browser." not in code:
        problems.append("never used the browser")
    if "google-calendar" not in r["after"]["integrations"]:
        problems.append("integration not created")
    else:
        try:   # Nango's own record of what was stored
            stored = nango._nango("GET", "/integrations/google-calendar", params={"include": "credentials"})
            cred = (stored.get("data", stored)).get("credentials") or {}
            if cred.get("client_id") != FAKE_ID or cred.get("client_secret") != FAKE_SECRET:
                problems.append("Nango holds a different Client ID or secret than the console showed")
        except nango.NangoError as e:
            problems.append(f"couldn't read the stored credentials: {e}")
    if "page=" not in code and not any(q["secret"] for q in r["questions"]):
        problems.append("registered it without reading the page or asking hidden")
    if FAKE_SECRET in ev or FAKE_SECRET in said:
        problems.append("THE SECRET LEAKED into the trace or a reply")
    if r["state"].get("secret_in_chat"):
        problems.append("ASKED FOR THE SECRET IN CHAT")
    if CALLBACK not in code:
        problems.append("never typed the exact redirect URL into the console")
    return not problems, "; ".join(problems) or "clicked through the console, registered the shown ID and secret, kept them out of the trace"


def done_drive_mock() -> bool:
    return "google-calendar" in nango_state()["integrations"]


# name, request, integration it may create, fake-browser click text, sign-in timeout, check,
# stop (the chat ends as soon as this is true after an assistant reply; None: when the user has nothing to say),
# extras: {"drive": the user says yes to "click through it for me", "dock": a windowless browser, "console": the
# request names the practice console (a local page standing in for Google's)}
SCENARIOS = [
    ("noauth",
     "I'd like to see how connecting an app works before I do a real one. Nango has a test app called "
     "'unauthenticated' that needs no login. Please connect that one.",
     "unauthenticated", "Unauthenticated", 300, check_noauth, done_noauth, {}),
    ("google-calendar",
     "Connect my Google Calendar so you can see my schedule.",
     "google-calendar", "", 45, check_google, None, {}),
    ("drive-mock",
     "I'm testing the setup. Connect my Google Calendar, but instead of Google's real console use my practice "
     "OAuth console at {console} (it needs no sign-in). Please click through it yourself to create the OAuth "
     "client, and register the app in Nango from what it shows.",
     "google-calendar", "", 45, check_drive_mock, done_drive_mock, {"drive": True, "dock": True, "console": True}),
]


def run_scenario(cfg, name: str, request: str, key: str, run_dir: Path, stop=None, extra: dict | None = None) -> dict:
    from jevh.orchestrator import Harness
    extra = extra or {}
    ws = run_dir / name
    ws.mkdir(parents=True, exist_ok=True)
    real_nango_dir = str(cfg.workspace / "nango")    # the key and URL nango.start() saved
    cfg.workspace = ws / "ws"                        # this scenario's own memory and sessions
    log = open(ws / "events.jsonl", "w", encoding="utf-8")
    t0 = time.monotonic()
    qs: list[dict] = []
    codes: list[str] = []
    st: dict = {"prefer_drive": bool(extra.get("drive"))}
    box = {"h": None}

    def on_event(k, d):
        rec = {"t": round(time.monotonic() - t0, 1), "kind": k}
        for kk, v in (d or {}).items():
            if kk == "task":
                v = getattr(v, "id", str(v))
            try:
                json.dumps(v)
            except TypeError:
                v = str(v)
            rec[kk] = v
        log.write(json.dumps(rec, ensure_ascii=False)[:60000] + "\n")
        log.flush()
        if k == "code":
            codes.append(str(d.get("code", "")))
            first = next((ln for ln in str(d.get("code", "")).splitlines() if ln.strip()), "")
            print(f"    [{rec['t']:6.0f}s] step {d.get('step')}: {first[:110]}", flush=True)
        elif k == "output" and "'connected': False" in str(d.get("output", "")):
            st["connect_timed_out"] = True
        elif k == "question":
            q = d["question"]
            after_error = bool(st.get("reported_error"))
            reply = answer_question(q, st)
            if after_error:
                st.setdefault("post_error", []).append(q["question"])
            qs.append({"q": q["question"], "options": q["options"], "secret": bool(q.get("secret")),
                       "reply": "(hidden)" if q.get("secret") else reply})
            print(f"    [{rec['t']:6.0f}s] ASK{' (hidden)' if q.get('secret') else ''}: {q['question'][:170]}"
                  f"\n             options {q['options']} -> {'(hidden)' if q.get('secret') else reply[:90]}", flush=True)
            box["h"].answer(q["id"], reply)

    h = Harness(cfg, on_event=on_event)
    box["h"] = h
    os.environ["JEVH_NANGO_DIR"] = real_nango_dir    # the harness pointed it at the scenario's workspace
    h.learning = False
    if extra.get("dock"):    # the dashboard's own browser panel: Chromium with no window, so nothing pops up on screen
        h.watch_browser, h.browser_dock = True, True
    timer = threading.Timer(LIMIT, h.cancel)
    timer.start()
    transcript, msg = [], request
    try:
        for _ in range(MAX_TURNS):
            print(f"  USER: {msg[:200]}", flush=True)
            r = h.turn(msg)
            transcript.append({"user": msg, "assistant": r.answer, "persona": r.persona, "seconds": round(r.seconds)})
            print(f"  ASSISTANT ({r.persona}, {r.seconds:.0f}s): {r.answer[:400]}", flush=True)
            if st.get("reported_error"):
                st.setdefault("post_error", []).append(r.answer)
            if r.cancelled:
                break
            if stop is not None and stop():
                break
            msg = chat_reply(r.answer, st)
            if msg is None:
                break
    except Exception as e:  # noqa: BLE001 - one broken scenario never ends the run
        transcript.append({"user": msg, "assistant": f"(harness crash: {e!r})", "persona": "", "seconds": 0})
    finally:
        timer.cancel()
        log.close()
    (ws / "transcript.md").write_text("\n\n".join(f"**User:** {t['user']}\n\n**Assistant ({t['persona']}):** "
                                                  f"{t['assistant']}" for t in transcript), encoding="utf-8")
    return {"transcript": transcript, "questions": qs, "after": nango_state(), "state": st, "steps": len(codes),
            "seconds": round(time.monotonic() - t0), "code_text": "\n".join(codes),
            "events_text": (ws / "events.jsonl").read_text(encoding="utf-8")}


def report(path: Path | None = None) -> None:
    path = path or max(RUNS.glob("*/results.json"), key=lambda p: p.stat().st_mtime)
    rows = json.loads(path.read_text(encoding="utf-8"))
    print(path.parent.name)
    for r in rows:
        print(f"{'PASS' if r['pass'] else 'FAIL'} {r['scenario']:16} run {r.get('run', 1)} {r['seconds']:5}s {r['steps']:3} steps  "
              f"{len(r['transcript'])} chat turns  {len(r['questions'])} questions  {r['grade']}")
        print(f"     last reply: {r['transcript'][-1]['assistant'][:400]!r}")
    print("\nrates:")
    for name in dict.fromkeys(r["scenario"] for r in rows):
        mine = [r for r in rows if r["scenario"] == name]
        ok = sum(bool(r["pass"]) for r in mine)
        secs = sorted(r["seconds"] for r in mine)
        print(f"  {name:16} {ok}/{len(mine)} pass   median {secs[len(secs) // 2]}s   "
              f"steps {min(r['steps'] for r in mine)}-{max(r['steps'] for r in mine)}")
        fails = [r["grade"] for r in mine if not r["pass"]]
        for why in dict.fromkeys(fails):
            print(f"      x {why}  (x{fails.count(why)})")


def main() -> None:
    global LIMIT
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="", help="comma-separated name filters")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--limit", type=int, default=LIMIT // 60, help="minutes per scenario")
    ap.add_argument("-n", "--repeat", type=int, default=1, help="runs of each scenario (local models vary: report rates)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.report:
        return report()
    LIMIT = args.limit * 60
    from jevh.config import load_config
    from jevh.tools import nango
    try:
        nango_state()
    except nango.NangoError as e:
        sys.exit(f"Nango isn't ready ({e}): run nango.start() first (SETUP.md, Connected accounts)")
    keys = [k for k in args.k.split(",") if k]
    run_dir = RUNS / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    server = None
    for name, request, key, click, timeout, check, stop, extra in SCENARIOS:
        if keys and not any(k in name for k in keys):
            continue
        if extra.get("console") and server is None:
            server = serve_console()
        for run in range(1, args.repeat + 1):
            if key in nango_state()["integrations"]:
                print(f"\n=== {name}: SKIPPED, the integration '{key}' already exists (not touching a real one)")
                break
            label = name if args.repeat == 1 else f"{name}-{run}"
            request_now = request.format(console=CONSOLE_URL) if extra.get("console") else request
            print(f"\n=== {label}: {request_now[:120]}", flush=True)
            (run_dir / label).mkdir(parents=True, exist_ok=True)
            # the user's "browser": clicks Connect on a no-login page, otherwise just notes the page was opened
            os.environ["BROWSER"] = f"{sys.executable.replace(chr(92), '/')} {(HERE / 'fake_browser.py').as_posix()} %s"
            os.environ["JEVH_FAKE_CLICK"] = click
            os.environ["JEVH_FAKE_LOG"] = str(run_dir / label / "browser.log")
            os.environ["JEVH_NANGO_CONNECT_TIMEOUT"] = str(timeout)
            cfg = load_config()
            try:
                r = run_scenario(cfg, label, request_now, key, run_dir, stop, extra)
                ok, why = check(r)
            finally:
                cleanup(key)
            rows.append({"scenario": name, "run": run, "pass": ok, "grade": why,
                         **{k: v for k, v in r.items() if k not in ("code_text", "events_text")}})
            (run_dir / "results.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"--> {label}: {'PASS' if ok else 'FAIL'} ({why}) in {r['seconds']}s, {r['steps']} steps, "
                  f"{len(r['transcript'])} chat turns, {len(r['questions'])} questions", flush=True)
    if server is not None:
        server.shutdown()
    report(run_dir / "results.json")


if __name__ == "__main__":
    main()
