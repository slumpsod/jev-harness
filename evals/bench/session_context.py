"""Session-long context, live: does a follow-up do better when the same assistant's conversation goes on (every step, the
code it ran and what came back) than with the last two replies as a summary? Flag #18, 2026-10-24.

    python evals/bench/session_context.py                  # every scenario, both arms, 2 repeats (~2-2.5 h on Strata)
    python evals/bench/session_context.py --reps 1 -k data # one scenario once
    python evals/bench/session_context.py --holdout        # only the held-out scenarios
    python evals/bench/session_context.py --offline        # summarize the saved results again

Two arms on the same multi-turn conversations, through the harness's own code (Harness.turn, the configured worker model
and Jev): `old` is limits.session_context 0 (a follow-up gets the last follow_turns replies, 3000 chars each, plus a
2500-char note of the last code), `new` is 1. Both keep follow_always on (the default), so the carried context is the only
difference. Designed for Qwen3.8-Flash-Next on Strata with light Jev (Laya); any configured worker works.

Scenarios. Nothing real is touched: each talks to local stand-ins this script serves on 127.0.0.1, which also count the
requests each turn makes - re-reading what an earlier turn already had is the "redo" the flag was about.
  sonos     the flagged session itself (20261024-141900-8b4f): four Sonos speakers on the local UPnP API (device
            description, RenderingControl volume, AVTransport with a Spotify stream stalled at 0:00:00 until it is paused
            and played again). Six turns, as the user wrote them. Graded on the speakers' state after each turn.
  data      a shop's orders API (1500 orders): the total, then by region, the best month of the top region, its worst
            product, and "what was the total again, and where did it come from?". Graded on exact numbers.
  topic     (held out) the same kind of API, an unrelated question in the middle ("17% of 2350"), then back to the orders.
  build     (held out) a todo script grown over four requests, then used. Graded by running the script.
  persona   (regression, new arm only) a workout plan by the fitness coach, then the daily assistant saves and edits it:
            another assistant starts its own conversation. The old arm takes that path by construction, so it is not run.
The held-out scenarios were written before any run and not tuned on.

Every question a worker asks is answered "Yes, go ahead." - except a request to change this computer, which is refused
(the bench runs on a real PC). Results: runs/session_context/<stamp>/ (each run's workspace), session_context_raw.json and
session_context_summary.txt (a live run writes to workspace/evals/<backend>/; --offline re-sweeps the committed ones; see
evals/_out.py).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))
import _out  # noqa: E402  (evals/_out.py: where a run's results go)

RUNS = HERE / "runs" / "session_context"
RAW = HERE / "session_context_raw.json"
SUMMARY = HERE / "session_context_summary.txt"
LIMIT = 12 * 60   # seconds a turn may take before it is stopped (and fails)
SYSTEM_CHANGE = re.compile(r"firewall|portproxy|port proxy|netsh|registry|admin|elevat|UAC|install|uninstall|delete|"
                           r"remove|run as|on this PC|this computer|reset|clean|tidy|move|rename|archive|kill|"
                           r"restart|docker", re.I)
REFUSE = "No - please don't change anything on this computer. Just tell me what you would do."
YES = "Yes, go ahead."


# ------------------------------------------------------------------------------------------------- stand-in services
class Service:
    """A local HTTP server that logs each request under the bench's current turn."""

    def __init__(self, handler_cls, n: int = 1):
        self.turn = 0
        self.log: list[tuple[int, str, str]] = []   # (turn, kind, detail)
        self.servers = []
        for _ in range(n):
            srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
            srv.service = self   # type: ignore[attr-defined]
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            self.servers.append(srv)
        self.ports = [s.server_address[1] for s in self.servers]

    def hit(self, kind: str, detail: str = "") -> None:
        self.log.append((self.turn, kind, detail))

    def hits(self, turn: int, kind: str) -> int:
        return sum(1 for t, k, _ in self.log if t == turn and k == kind)

    def close(self) -> None:
        for s in self.servers:
            s.shutdown()
            s.server_close()


class Quiet(BaseHTTPRequestHandler):
    def log_message(self, *a):   # noqa: D401 - no request log on stderr
        pass

    def send(self, code: int, body: str, ctype: str = "text/xml; charset=utf-8") -> None:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


ROOMS = ["Front Showroom L", "Front Showroom R", "2nd Showroom L", "2nd Showroom R"]


class Sonos:
    """Four speakers grouped into one zone, playing a Spotify stream that is stalled (position frozen at 0:00:00 while
    the transport says PLAYING) until it is paused and played again - what the flagged session found."""

    def __init__(self):
        self.volume = [24, 24, 26, 26]
        self.muted = [False] * 4
        self.state = "PLAYING"
        self.stalled = True
        self.paused_once = False
        self.started = time.time()
        self.lock = threading.Lock()
        owner = self

        class H(Quiet):
            def do_GET(self):
                i = owner.svc.ports.index(self.server.server_address[1])
                if self.path.startswith("/xml/device_description.xml"):
                    owner.svc.hit("discover", str(i))
                    return self.send(200, f"""<?xml version="1.0"?><root xmlns="urn:schemas-upnp-org:device-1-0"><device>
<deviceType>urn:schemas-upnp-org:device:ZonePlayer:1</deviceType><friendlyName>127.0.0.1 - Sonos One</friendlyName>
<manufacturer>Sonos, Inc.</manufacturer><modelName>Sonos One</modelName><modelNumber>S18</modelNumber>
<serialNum>48-A6-B8-5C-B8-7{i}:{i}</serialNum><softwareVersion>79.1-56030</softwareVersion><roomName>{ROOMS[i]}</roomName>
<UDN>uuid:RINCON_48A6B85CB87{i}01400</UDN><serviceList>
<service><serviceType>urn:schemas-upnp-org:service:AVTransport:1</serviceType><controlURL>/MediaRenderer/AVTransport/Control</controlURL></service>
<service><serviceType>urn:schemas-upnp-org:service:RenderingControl:1</serviceType><controlURL>/MediaRenderer/RenderingControl/Control</controlURL></service>
</serviceList></device></root>""")
                owner.svc.hit("other", self.path)
                self.send(404, "not found", "text/plain")

            def do_POST(self):
                i = owner.svc.ports.index(self.server.server_address[1])
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode("utf-8", "replace")
                action = (self.headers.get("SOAPACTION") or self.headers.get("SOAPAction") or "").strip('"').split("#")[-1]
                if not action:
                    m = re.search(r"<u:(\w+)", body)
                    action = m[1] if m else ""
                owner.svc.hit("soap", f"{i}:{action}")
                out = owner.act(i, action, body)
                if out is None:
                    return self.send(500, "<s:Envelope><s:Body><s:Fault><faultstring>UPnPError</faultstring>"
                                          "<detail><UPnPError><errorCode>401</errorCode></UPnPError></detail></s:Fault>"
                                          "</s:Body></s:Envelope>")
                ns = "RenderingControl" if "Volume" in action or "Mute" in action else "AVTransport"
                self.send(200, f'<?xml version="1.0"?><s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">'
                               f'<s:Body><u:{action}Response xmlns:u="urn:schemas-upnp-org:service:{ns}:1">{out}'
                               f'</u:{action}Response></s:Body></s:Envelope>')
        self.svc = Service(H, 4)

    def position(self) -> str:
        if self.stalled or self.state != "PLAYING":
            return "0:00:00"
        s = int(time.time() - self.started) % 158
        return f"0:{s // 60:02d}:{s % 60:02d}"

    def act(self, i: int, action: str, body: str) -> str | None:
        with self.lock:
            if action == "GetVolume":
                return f"<CurrentVolume>{self.volume[i]}</CurrentVolume>"
            if action == "SetVolume":
                m = re.search(r"<DesiredVolume>\s*(\d+)", body)
                if not m:
                    return None
                self.volume[i] = max(0, min(100, int(m[1])))
                return ""
            if action == "GetMute":
                return f"<CurrentMute>{int(self.muted[i])}</CurrentMute>"
            if action == "SetMute":
                m = re.search(r"<DesiredMute>\s*(\d)", body)
                self.muted[i] = bool(m and m[1] == "1")
                return ""
            if action == "GetTransportInfo":
                return (f"<CurrentTransportState>{self.state}</CurrentTransportState>"
                        "<CurrentTransportStatus>OK</CurrentTransportStatus><CurrentSpeed>1</CurrentSpeed>")
            if action == "GetPositionInfo":
                return ("<Track>1</Track><TrackDuration>0:02:38</TrackDuration><TrackMetaData>&lt;DIDL-Lite&gt;&lt;item&gt;"
                        "&lt;dc:title&gt;Song About You&lt;/dc:title&gt;&lt;dc:creator&gt;The Band CAMINO&lt;/dc:creator&gt;"
                        "&lt;/item&gt;&lt;/DIDL-Lite&gt;</TrackMetaData><TrackURI>x-sonos-vli:RINCON_48A6B85CB87201400:1,"
                        f"spotify:track:4uLU6hMCjMI75M1A2tKUQC</TrackURI><RelTime>{self.position()}</RelTime>")
            if action == "GetMediaInfo":
                return ("<NrTracks>1</NrTracks><CurrentURI>x-sonos-vli:RINCON_48A6B85CB87201400:1,spotify</CurrentURI>"
                        "<PlayMedium>NETWORK</PlayMedium>")
            if action == "Pause":
                self.state, self.paused_once = "PAUSED_PLAYBACK", True
                return ""
            if action == "Stop":
                self.state, self.paused_once = "STOPPED", True
                return ""
            if action == "Play":
                if self.paused_once:   # the stalled stream only comes back after it is paused (or stopped) and played
                    self.stalled = False
                    self.started = time.time()
                self.state = "PLAYING"
                return ""
            return None

    def snapshot(self) -> dict:
        return {"volume": list(self.volume), "muted": list(self.muted), "state": self.state, "stalled": self.stalled}


REGIONS = ["North", "South", "East", "West"]
PRODUCTS = ["Lamp", "Chair", "Desk", "Shelf", "Rug", "Mirror", "Sofa", "Clock"]


class Shop:
    """GET /sales: a shop's orders, deterministic for a seed."""

    def __init__(self, seed: int, n: int = 1500):
        rnd = random.Random(seed)
        weight = {r: 0.7 + 0.2 * k for k, r in enumerate(rnd.sample(REGIONS, 4))}
        self.orders = []
        for k in range(n):
            region = rnd.choices(REGIONS, [weight[r] for r in REGIONS])[0]
            month = rnd.randint(1, 9)
            self.orders.append({"id": 10000 + k, "date": f"2026-{month:02d}-{rnd.randint(1, 28):02d}", "region": region,
                                "product": rnd.choice(PRODUCTS), "amount": round(rnd.uniform(8, 400), 2)})
        owner = self

        class H(Quiet):
            def do_GET(self):
                if self.path.split("?")[0].rstrip("/") == "/sales":
                    owner.svc.hit("fetch")
                    return self.send(200, json.dumps({"orders": owner.orders}), "application/json")
                owner.svc.hit("other", self.path)
                self.send(404, json.dumps({"error": "not found; try /sales"}), "application/json")
        self.svc = Service(H, 1)

    def facts(self) -> dict:
        o = self.orders
        total = round(sum(x["amount"] for x in o), 2)
        by_region = {r: round(sum(x["amount"] for x in o if x["region"] == r), 2) for r in REGIONS}
        top = max(by_region, key=by_region.get)
        months = {}
        for x in o:
            if x["region"] == top:
                months[x["date"][:7]] = months.get(x["date"][:7], 0) + x["amount"]
        prod = {}
        for x in o:
            if x["region"] == top:
                prod[x["product"]] = prod.get(x["product"], 0) + x["amount"]
        counts = {r: sum(1 for x in o if x["region"] == r) for r in REGIONS}
        return {"total": total, "by_region": by_region, "top": top, "best_month": max(months, key=months.get),
                "worst_product": min(prod, key=prod.get), "count": len(o), "avg": round(total / len(o), 2),
                "fewest": min(counts, key=counts.get)}


# ------------------------------------------------------------------------------------------------------- scenarios
@dataclass
class Scenario:
    name: str
    holdout: bool
    note: str
    both_arms: bool = True   # False: a regression check of the new arm only (the old arm has nothing to compare)


SCENARIOS = [
    Scenario("sonos", False, "the flagged session, replayed against four stand-in speakers"),
    Scenario("data", False, "follow-up questions on a shop's orders API"),
    Scenario("persona", False, "regression: another assistant starts its own conversation", both_arms=False),
    Scenario("topic", True, "an unrelated question in the middle, then back"),
    Scenario("build", True, "a script grown over four requests, then used"),
]


def has_num(text: str, value: float, tol: float = 0.011) -> bool:
    """`value` appears in `text` (commas, a $ and rounding to whole units allowed)."""
    for m in re.finditer(r"-?\d[\d,]*(?:\.\d+)?", text or ""):
        try:
            v = float(m[0].replace(",", ""))
        except ValueError:
            continue
        if abs(v - value) <= tol or (abs(value) >= 100 and abs(v - round(value)) < 0.5 and abs(v - value) <= 0.51):
            return True
    return False


def todo_check(script: Path, level: int) -> tuple[bool, str]:
    """Run the todo script in a scratch copy: level 1 add/list, 2 done with [x], 3 --high first."""
    if not script.is_file():
        return False, "no todo.py"
    tmp = script.parent / "_check"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    shutil.copy(script, tmp / "todo.py")

    def run(*args):
        r = subprocess.run([sys.executable, str(tmp / "todo.py"), *args], cwd=tmp, capture_output=True, text=True,
                           timeout=30)
        return r.returncode, r.stdout + r.stderr
    try:
        run("add", "water plants")
        run("add", "pay rent")
        code, out = run("list")
        if code or "water plants" not in out or "pay rent" not in out:
            return False, f"add/list: {out[-200:]}"
        if level >= 2:
            code, _ = run("done", "1")
            code2, out = run("list")
            line = next((ln for ln in out.splitlines() if "water plants" in ln), "")
            if code or code2 or "[x]" not in line.lower():
                return False, f"done: {out[-200:]}"
        if level >= 3:
            code, _ = run("add", "--high", "fix the door")
            code2, out = run("list")
            lines = [ln for ln in out.splitlines() if any(w in ln for w in ("water plants", "pay rent", "fix the door"))]
            if code or code2 or not lines or "fix the door" not in lines[0]:
                return False, f"--high first: {out[-200:]}"
        return True, "ok"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def turns_for(sc: str, env: dict) -> list[dict]:
    """The scenario's turns: {prompt, persona, grade(answer, env) -> (passed, why), redo: service kind or None}."""
    if sc == "sonos":
        p = env["sonos"].svc.ports
        s = env["sonos"]

        def vols(v):
            return lambda a: (all(x == v for x in s.volume), f"volumes {s.volume}")
        return [
            {"prompt": f"can you check all of the smart speakers on my network? They show up for this computer at "
                       f"127.0.0.1 on ports {p[0]}, {p[1]}, {p[2]} and {p[3]}.",
             "grade": sonos_found},
            {"prompt": "yeah so what you can do with the sonos speakers",
             "grade": lambda a: (bool(re.search(r"volume", a, re.I)) and bool(re.search(r"play|pause", a, re.I)),
                                 "says it can change the volume and play/pause")},
            {"prompt": "turn the volume up to 25%", "grade": vols(25)},
            {"prompt": "turn the volume up to 75%", "grade": vols(75)},
            {"prompt": "i dont hear anything, are you sure the music playing?",
             "grade": lambda a: (not s.stalled or bool(re.search(r"stall|stuck|frozen|0:00:00|not (actually )?playing|"
                                                                  r"isn't (actually )?playing", a, re.I)),
                                 f"found or fixed the stalled stream (stalled={s.stalled})")},
            {"prompt": "turn the volume down to 25% and play the music",
             "grade": lambda a: (all(x == 25 for x in s.volume) and not s.stalled and s.state == "PLAYING",
                                 f"volumes {s.volume}, state {s.state}, stalled {s.stalled}")},
        ]
    if sc in ("data", "topic"):
        f = env["shop"].facts()
        url = f"http://127.0.0.1:{env['shop'].svc.ports[0]}/sales"
        if sc == "data":
            return [
                {"prompt": f"pull the orders from my shop's API at {url} and tell me the total revenue",
                 "grade": lambda a: (has_num(a, f["total"]), f"total {f['total']}")},
                {"prompt": "break that down by region",
                 "grade": lambda a: (all(has_num(a, v) for v in f["by_region"].values()), f"regions {f['by_region']}")},
                {"prompt": "which month was best for the top region?",
                 "grade": lambda a: (f["top"] in a and (f["best_month"] in a or _month_name(f["best_month"]) in a.lower()),
                                     f"{f['top']} best in {f['best_month']}")},
                {"prompt": "and what's the worst selling product in that region overall?",
                 "grade": lambda a: (f["worst_product"] in a, f"worst product {f['worst_product']}")},
                {"prompt": "what was the total again, and where did the data come from?",
                 "grade": lambda a: (has_num(a, f["total"]) and ("/sales" in a or str(env["shop"].svc.ports[0]) in a),
                                     f"total {f['total']} and the API")},
            ]
        return [
            {"prompt": f"get the orders from {url} and tell me how many there are and the average order value",
             "grade": lambda a: (has_num(a, f["count"], 0.5) and has_num(a, f["avg"]), f"{f['count']} orders, avg {f['avg']}")},
            {"prompt": "unrelated - what's 17% of 2350?", "grade": lambda a: (has_num(a, 399.5), "399.5")},
            {"prompt": "ok back to the orders: which region had the fewest?",
             "grade": lambda a: (f["fewest"] in a, f"fewest: {f['fewest']}")},
            {"prompt": "and what was the average order value you found earlier?",
             "grade": lambda a: (has_num(a, f["avg"]), f"avg {f['avg']}")},
        ]
    if sc == "build":
        d = env["dir"]
        script = d / "todo.py"

        def lvl(n):
            return lambda a: todo_check(script, n)

        def used(a):
            try:
                data = json.loads((d / "todos.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return False, "no todos.json"
            text = json.dumps(data).lower()
            ok = "buy milk" in text and "call mom" in text and "buy milk" in a.lower()
            return ok, "buy milk and call mom in todos.json and the list shown"
        return [
            {"prompt": f"write a python script at {script} that keeps a todo list in a todos.json next to it, with "
                       "`add <text>` and `list` commands (list numbers the items from 1)", "grade": lvl(1)},
            {"prompt": "add a `done <n>` command that marks item n as done, and show done items with [x] in the list",
             "grade": lvl(2)},
            {"prompt": "add priorities: `add --high <text>` makes a high-priority item, and list shows high-priority "
                       "items first", "grade": lvl(3)},
            {"prompt": "now use it: add 'buy milk' as high priority and 'call mom' normally, mark 'call mom' done, then "
                       "show me the list", "grade": used},
        ]
    if sc == "persona":
        d = env["dir"]
        plan = d / "plan.txt"

        def saved(a):
            text = plan.read_text(encoding="utf-8", errors="replace") if plan.is_file() else ""
            days = len(set(re.findall(r"day\s*([1-3])", text, re.I)))
            return days >= 3, f"plan.txt with 3 days ({days} found)"

        def edited(a):
            text = plan.read_text(encoding="utf-8", errors="replace").strip() if plan.is_file() else ""
            last = text.splitlines()[-1] if text else ""
            return "start monday" in last.lower() and len(text.splitlines()) > 3, f"last line {last[:60]!r}"
        return [
            {"prompt": "give me a 3-day beginner workout plan, keep it short (Day 1, Day 2, Day 3)",
             "persona": "fitness_coach", "grade": lambda a: (len(re.findall(r"day\s*[1-3]", a, re.I)) >= 3, "3 days")},
            {"prompt": f"save that plan as {plan}", "persona": "daily_assistant", "grade": saved},
            {"prompt": "now add a line at the end saying 'start Monday'", "persona": "daily_assistant", "grade": edited},
        ]
    raise ValueError(sc)


def sonos_found(answer: str) -> tuple[bool, str]:
    """Turn 1 of the Sonos replay: it says they are Sonos speakers and accounts for all four (by room or by port). The
    first run also wanted a room name, which correct answers that listed the four by port left out (3 of 4 "fails"
    there were right answers); regraded offline with this rule."""
    ports = set(re.findall(r"\b\d{4,5}\b", answer or "")) - {"1400"}
    four = len(ports) >= 4 or sum(r in (answer or "") for r in ROOMS) >= 4 or re.search(r"\b(all )?(four|4)\b", answer or "", re.I)
    return bool(re.search(r"sonos", answer or "", re.I) and four), "named them as Sonos speakers, all four"


def regrade(records: list[dict]) -> list[dict]:
    """The checks that read only the answer, again (state checks were taken live and stay as they were)."""
    for r in records:
        if r["scenario"] == "sonos" and r["turn"] == 1:
            r["passed"], r["why"] = sonos_found(r.get("answer", ""))
    return records


def _month_name(ym: str) -> str:
    return ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
            "november", "december"][int(ym[5:7]) - 1]


# ------------------------------------------------------------------------------------------------------ the harness
def make_harness(where: Path, events: list, which: str, compact_at: float | None = None):
    from jevh.config import load_config
    from jevh.orchestrator import Harness
    cfg = load_config()
    cfg.workspace = where / "ws"
    cfg.personas_dir = where / "personas"
    cfg.personas_dirs = [cfg.personas_dir]
    if not cfg.personas_dir.exists():
        shutil.copytree(ROOT / "personas", cfg.personas_dir, ignore=shutil.ignore_patterns("learned"))
    cfg.raw = {**cfg.raw, "learning": False, "failures": {**(cfg.raw.get("failures") or {}), "audit": 0}}
    cfg.limits = {**cfg.limits, "follow_always": 1, "session_context": int(which == "new")}
    if compact_at:   # (--compact-at: compaction early, so condensing earlier requests runs on short sessions)
        cfg.raw = {**cfg.raw, "context": {**(cfg.raw.get("context") or {}), "compact_at": compact_at}}
    t0 = time.time()
    h = Harness(cfg, on_event=lambda k, d: events.append((time.time() - t0, k, d)))
    h.learning, h.tier_mode = False, "strong"

    def ask(task, persona, question, options=None, secret=False):
        events.append((time.time() - t0, "asked", {"question": str(question)[:500]}))
        # (the stand-in speakers are not this computer: "restart the stream?" is about them)
        about_speakers = re.search(r"speaker|sonos|music|volume|play|stream|track|song", str(question), re.I)
        return REFUSE if SYSTEM_CHANGE.search(str(question)) and not about_speakers else YES
    h._ask_user = ask
    return h


def timed_turn(h, prompt: str, persona: str | None):
    watchdog = threading.Timer(LIMIT, h.cancel)
    watchdog.daemon = True
    watchdog.start()
    try:
        return h.turn(prompt, force=persona), None
    except Exception as e:  # noqa: BLE001 - a crash is a result
        return None, f"{type(e).__name__}: {e}"
    finally:
        watchdog.cancel()


def lead_checkpoint(h, turn_no: int) -> dict:
    for d in h.session_dir.iterdir() if h.session_dir.is_dir() else []:
        if d.name.startswith(f"u{turn_no}t1-"):
            try:
                return json.loads((d / "checkpoint.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}
    return {}


def run_scenario(sc: Scenario, which: str, where: Path, rep: int, compact_at: float | None = None) -> list[dict]:
    env: dict = {"dir": where / "files"}
    env["dir"].mkdir(parents=True, exist_ok=True)
    services = []
    if sc.name == "sonos":
        env["sonos"] = Sonos()
        services.append(("sonos", env["sonos"].svc, "discover"))
    if sc.name in ("data", "topic"):
        env["shop"] = Shop(seed=7 if sc.name == "data" else 23)
        services.append(("shop", env["shop"].svc, "fetch"))
    events: list = []
    h = make_harness(where, events, which, compact_at)
    records = []
    prev_sha = None
    try:
        for n, t in enumerate(turns_for(sc.name, env), 1):
            for _, svc, _ in services:
                svc.turn = n
            events.clear()
            started = time.time()
            r, err = timed_turn(h, t["prompt"], t.get("persona", "daily_assistant"))
            secs = round(time.time() - started, 1)
            answer = r.answer if r else ""
            try:
                passed, why = t["grade"](answer)
            except Exception as e:  # noqa: BLE001
                passed, why = False, f"grading: {type(e).__name__}: {e}"
            usage = (r.usage or {}) if r else {}
            llm = (usage.get("by_kind") or {}).get("llm") or {}
            cp = lead_checkpoint(h, n)
            sha = cp.get("system_sha")
            code = [d for _, k, d in events if k == "code"]
            redo = {kind: svc.hits(n, kind) for _, svc, kind in services}
            records.append({
                "scenario": sc.name, "holdout": sc.holdout, "arm": which, "rep": rep, "turn": n, "prompt": t["prompt"],
                "persona": r.persona if r else None, "answer": answer[:3000], "error": err, "passed": bool(passed),
                "why": why, "seconds": secs, "steps": len(code),
                "input_tokens": llm.get("input", 0), "output_tokens": llm.get("output", 0),
                "model_calls": llm.get("calls", 0), "redo": redo,
                # code written this turn: rebuilding what an earlier turn had (a helper, a parser) shows up here
                "code_chars": sum(len(str(d.get("code", ""))) for d in code),
                "service_calls": {name: sum(1 for tt, _, _ in svc.log if tt == n) for name, svc, _ in services},
                "read_checkpoint": any("checkpoint.json" in json.dumps(c) for c in code),
                "continued": len(cp.get("turns") or []) > 1 or bool(cp.get("earlier")),
                "earlier_condensed": bool(cp.get("earlier")),
                "same_system": (sha == prev_sha) if (sha and prev_sha) else None,
                "messages": len(cp.get("messages") or []),
                "state": env["sonos"].snapshot() if "sonos" in env else None})
            prev_sha = sha or prev_sha
            print(f"    turn {n}: {'PASS' if passed else 'FAIL'} {secs}s steps {len(code)} redo {redo} - {why}",
                  flush=True)
    finally:
        for _, svc, _ in services:
            svc.close()
    return records


# ----------------------------------------------------------------------------------------------------- the summary
def summarize(records: list[dict]) -> str:
    out = ["Session-long context, live (evals/bench/session_context.py): old = the last 2 replies as a summary "
           "(session_context 0), new = the conversation goes on (1)."]
    if not records:
        return out[0] + "\n(no records)"
    reps = sorted({r["rep"] for r in records})
    out.append(f"{len(records)} turns, {len(reps)} repeat(s).")

    def block(title, recs):
        out.append(f"\n== {title}")
        for arm in ("old", "new"):
            rs = [r for r in recs if r["arm"] == arm]
            if not rs:
                continue
            fu = [r for r in rs if r["turn"] > 1]
            redo = [sum(r["redo"].values()) for r in fu]
            out.append(f"  {arm}: {sum(r['passed'] for r in rs)}/{len(rs)} turns passed "
                       f"(follow-ups {sum(r['passed'] for r in fu)}/{len(fu)}); "
                       f"median {statistics.median([r['seconds'] for r in rs]):.0f} s/turn, "
                       f"follow-up median {statistics.median([r['seconds'] for r in fu]) if fu else 0:.0f} s; "
                       f"steps {sum(r['steps'] for r in rs)}, model calls {sum(r['model_calls'] for r in rs)}, "
                       f"tokens in {sum(r['input_tokens'] for r in rs) / 1000:.0f}k / out "
                       f"{sum(r['output_tokens'] for r in rs) / 1000:.0f}k; re-reads in follow-ups {sum(redo)}; "
                       f"code written in follow-ups {sum(r.get('code_chars', 0) for r in fu) / 1000:.0f}k chars; "
                       f"read an old checkpoint {sum(r['read_checkpoint'] for r in rs)}x; errors "
                       f"{sum(bool(r['error']) for r in rs)}")
    block("all scenarios", records)
    block("tuned scenarios (sonos, data, persona)", [r for r in records if not r["holdout"]])
    block("held out (topic, build)", [r for r in records if r["holdout"]])
    for sc in [s.name for s in SCENARIOS]:
        recs = [r for r in records if r["scenario"] == sc]
        if not recs:
            continue
        out.append(f"\n-- {sc}: pass rate per turn (old -> new), median seconds, re-reads")
        for n in sorted({r["turn"] for r in recs}):
            cells = []
            for arm in ("old", "new"):
                rs = [r for r in recs if r["turn"] == n and r["arm"] == arm]
                if rs:
                    cells.append(f"{arm} {sum(r['passed'] for r in rs)}/{len(rs)} "
                                 f"{statistics.median([r['seconds'] for r in rs]):.0f}s "
                                 f"redo {sum(sum(r['redo'].values()) for r in rs)} "
                                 f"code {sum(r.get('code_chars', 0) for r in rs) / 1000:.1f}k")
            prompt = recs[[r["turn"] for r in recs].index(n)]["prompt"]
            out.append(f"  turn {n} ({prompt[:50]}...): " + " | ".join(cells))
    out.append("\nfollow-up turns, total seconds per run (old -> new), scenarios run on both arms:")
    for sc in [s.name for s in SCENARIOS if s.both_arms]:
        per = {}
        for arm in ("old", "new"):
            runs = {}
            for r in records:
                if r["scenario"] == sc and r["arm"] == arm and r["turn"] > 1:
                    runs[r["rep"]] = runs.get(r["rep"], 0) + r["seconds"]
            per[arm] = runs
        if per["old"] and per["new"]:
            o, n = statistics.mean(per["old"].values()), statistics.mean(per["new"].values())
            out.append(f"  {sc}: {o:.0f} s -> {n:.0f} s ({(n - o) / o * 100:+.0f}%)   runs old "
                       f"{[round(v) for v in per['old'].values()]}, new {[round(v) for v in per['new'].values()]}")
    new = [r for r in records if r["arm"] == "new" and r["turn"] > 1]
    if new:
        same = [r["same_system"] for r in new if r["same_system"] is not None]
        out.append(f"\nnew arm: {sum(r['continued'] for r in new)}/{len(new)} follow-ups continued a conversation; "
                   f"{sum(r['earlier_condensed'] for r in new)} condensed earlier requests; system prompt unchanged "
                   f"from the turn before in {sum(same)}/{len(same)} (the model server's cached prompt start holds)")
    fails = [r for r in records if not r["passed"]]
    if fails:
        out.append("\nFailed turns:")
        for r in fails:
            out.append(f"  {r['arm']} {r['scenario']} rep {r['rep'] + 1} turn {r['turn']}: {r['why']}"
                       + (f" [{r['error'][:120]}]" if r["error"] else ""))
    return "\n".join(out)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", default="both", choices=["old", "new", "both"])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--holdout", action="store_true", help="only the held-out scenarios")
    ap.add_argument("-k", default="", help="only scenarios whose name contains one of these (comma-separated)")
    ap.add_argument("--compact-at", type=float, default=None,
                    help="compaction at this share of the window instead of the config's (0.1: earlier requests get "
                         "condensed within a short session)")
    ap.add_argument("--offline", nargs="?", const="", metavar="RAW.json",
                    help="summarize saved results again (default: the committed raw file, or --out's)")
    _out.add_args(ap)
    a = ap.parse_args()
    dest = _out.Out.of(a)
    if a.offline is not None:
        data = json.loads((Path(a.offline) if a.offline else dest.read(RAW)).read_text(encoding="utf-8"))
        records = regrade(data["records"])
        text = summarize(records)
        print(text)
        if a.offline and getattr(a, "save_committed", False):   # (a run's results made the reference)
            dest.write(RAW).write_text(json.dumps({"records": records}, indent=1, ensure_ascii=False, default=str),
                                       encoding="utf-8")
            dest.write(SUMMARY).write_text(text + "\n", encoding="utf-8")
        return 0
    scenarios = [s for s in SCENARIOS if (s.holdout or not a.holdout) and (not a.k or any(x in s.name for x in a.k.split(",")))]
    arms = ["old", "new"] if a.arm == "both" else [a.arm]
    out_dir = RUNS / time.strftime("%Y%m%d-%H%M%S")
    records: list[dict] = []
    for rep in range(a.reps):
        for sc in scenarios:
            for which in (arms if sc.both_arms else [a for a in arms if a == "new"]):   # (alternating: a slow spell
                # of the model server hits both arms)
                print(f"[{time.strftime('%H:%M:%S')}] {sc.name} {which} rep {rep + 1}", flush=True)
                try:
                    records += run_scenario(sc, which, out_dir / f"{sc.name}-{which}-{rep}", rep, a.compact_at)
                except Exception as e:  # noqa: BLE001
                    print(f"    bench error: {type(e).__name__}: {e}", flush=True)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "raw.json").write_text(json.dumps({"records": records}, indent=1, ensure_ascii=False,
                                                             default=str), encoding="utf-8")
    text = summarize(regrade(records))
    print(text)
    dest.write(RAW).write_text(json.dumps({"records": records}, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    dest.write(SUMMARY).write_text(text + "\n", encoding="utf-8")
    print(f"raw: {dest.where(dest.write(RAW))}   summary: {dest.where(dest.write(SUMMARY))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
