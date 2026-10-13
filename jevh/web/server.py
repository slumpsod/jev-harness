"""Local dashboard: chat + live view of Jev's decisions, hand-offs, swaps and artifacts.

    jevh dashboard [--port 8740] [--model provider:model] [--no-browser]

Security (the agent runs code, so the browser must not become a remote control):
- binds 127.0.0.1 only, and rejects any Host header but 127.0.0.1/localhost:<port>
  (blocks DNS-rebinding). `--allow-host NAME` (for `tailscale serve`, which proxies to 127.0.0.1) also
  accepts NAME, but only from a browser that holds this launch's access key (a link `?key=...`, then a cookie):
  the page itself carries the API token, so a bare reachable page would hand it to anyone who can reach it,
- every /api call needs the per-launch token baked into the served page; POSTs from a
  foreign Origin are refused (blocks CSRF from other websites),
- /api/file serves only files under the workspace; HTML artifacts get a CSP sandbox so a
  generated page can never reach the dashboard's origin.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import mimetypes
import platform
import re
import secrets
import shutil
import subprocess
import queue
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import Future
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse

from .. import context as ctxwin
from .. import settings as usersettings
from .. import newsroom as nr
from .. import update as updater
from ..bus import jsonable, task_json
from ..config import Config, load_config
from ..interject import KINDS
from ..lmstudio import gate_for
from ..orchestrator import Harness
from ..history import History
from ..accounts import SESSION_DAYS, AccountError, Accounts
from ..devices import wire
from ..devices.hub import DeviceHub
from ..review import list_sessions, session_record
from ..skills import load_roster

STATIC = Path(__file__).with_name("static")
MAX_EVENTS = 5000
# Events waiting to be sent on one open stream. A stream that falls this far behind (a phone asleep with the page open:
# the connection stays up, nothing is read) stops getting more; when it is read again it ends, and the page reconnects
# from the last event it saw, the rest coming from `events`. Unbounded, it held every event of the session in memory.
SUB_QUEUE = 2000
SESSION_COOKIE = "jevh_session"
HERE = "here"   # a message's origin when it was written on the hub itself
OPEN_PATHS = {"/login", "/auth/login", "/auth/logout", "/auth/pair", "/setup", "/auth/setup"}   # reachable from another device before signing in


def forwarded_ip(header: str | None, peer: str) -> str:
    """The client's address: `peer`, or behind a proxy on this machine (tailscale serve) the last X-Forwarded-For entry."""
    if peer in ("127.0.0.1", "::1") and header:
        return header.split(",")[-1].strip()
    return peer


def this_machine_ips() -> set[str]:
    """This computer's own addresses (its tailnet one included): a request from one of them was written on the hub."""
    import socket
    ips = {"127.0.0.1", "::1"}
    try:
        ips |= {a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None)}
    except OSError:
        pass
    return ips


class _Stream(asyncio.Queue):
    """One open event stream's queue (at most SUB_QUEUE events)."""
    behind = False   # an event was dropped because the stream fell too far behind: it ends when next read


# ---------------------------------------------------------------- app state

class Hub:
    """Owns the Harness, runs one turn at a time on a thread, fans events out to SSE clients."""

    def __init__(self, cfg: Config, harness_factory=None, model_override: str | None = None):
        self.cfg = cfg
        self.model_override = model_override
        # factory(on_event) -> a Harness for a new session; factory(on_event, session=id) -> one that carries on session `id`
        self._factory = harness_factory or (lambda on_event, session=None: Harness(
            cfg, on_event=on_event, model_override=model_override, session=session))
        # sign-in and the user's other computers (jevh/accounts.py, jevh/devices/hub.py); every harness the Hub makes uses them
        self.accounts = Accounts(cfg.workspace / "accounts.db")
        self.devices = DeviceHub(self.accounts, jev_url=lambda: self.harness.cfg.jev.get("url", "http://127.0.0.1:33001"),
                                 env=self._device_env, publish=self.publish)
        self.devices.here = platform.node()
        self.harness = self._factory(self.publish)
        self._adopt(self.harness)
        self.dock = BrowserDock(lambda: Path(self.harness.session_dir) / "browsers")
        self.events: list[dict] = []
        self.turns: list[dict] = []
        self.swaps: list[dict] = []
        self.seq = 0
        self.busy = threading.Lock()
        self._drain = threading.Lock()   # held to decide: run the next queued request, or let go of `busy`
        self._lock = threading.Lock()
        self._subs: list[_Stream] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self.pin: str | None = None
        self.updated: dict | None = None   # set once the code on disk was updated: a restart loads it
        # past sessions, categorized by Jev (jevh/history.py); asks the Hub's harness of the moment: a new session replaces it
        self.history = History(lambda: self.harness, self.publish)
        # the Newsroom board (jevh/newsroom.py): its tiles are looked at on a timer by the runner, started with the app
        self.newsroom_cfg = {**nr.settings(cfg.raw), "about_threshold": cfg.t("newsroom_about")}
        nr.export_env(cfg)
        self.newsroom = nr.Newsroom(cfg.workspace / "newsroom.db", self.newsroom_cfg, judge=self._judge_about,
                                    tool_env=lambda: self.harness.tool_env)   # (its web looks use the engine chosen in Settings)
        self.newsroom_runner = nr.Runner(
            self.newsroom, on_alerts=lambda notes: self.publish("newsroom_alert", {"notes": notes}),
            on_change=lambda: self.publish("newsroom", {"rev": self.newsroom.rev()}))

    def _adopt(self, harness) -> None:
        harness.browser_dock = True   # a watched browser shows in the Browser panel, not a window
        harness.devices = self.devices

    def _device_env(self) -> dict:
        """What a device's browser and desktop tools need to know of this hub's settings (never a key: a device's Jev
        questions come back here, see DeviceHub._answer_jev)."""
        env = getattr(self.harness, "tool_env", {}) or {}
        return {k: env[k] for k in ("JEVH_JEV_BACKEND",) if env.get(k)}

    # events from worker threads -> history + every open SSE stream
    def publish(self, kind: str, data: dict | None = None) -> None:
        with self._lock:
            self.seq += 1
            evt = {"seq": self.seq, "ts": time.time(), "kind": kind,
                   "turn": self.turns[-1]["id"] if self.turns else None, "data": jsonable(data or {})}
            self.events.append(evt)
            del self.events[:-MAX_EVENTS]
            if kind == "swap":
                self.swaps.append({"ts": evt["ts"], **evt["data"]})
                del self.swaps[:-50]
            subs = list(self._subs)
        if self.loop:
            for q in subs:
                try:
                    self.loop.call_soon_threadsafe(self._offer, q, evt)
                except RuntimeError:   # the event loop is closed: the server is shutting down
                    return

    def _judge_about(self, text: str, about: str) -> float:
        """P(this new item is about what a widget's `about` says): Jev's answer, asked of the harness of the moment."""
        from ..decide import noul
        # (the client directly: a look made by the timer is no part of the turn Jev's trace and usage belong to)
        r = self.harness.jev.client.ask({"item": text[:700], "interest": about}, {"match": noul(nr.ABOUT_Q)})
        return r.answers["match"].noul

    @staticmethod
    def _offer(q: _Stream, evt: dict) -> None:
        """(on the event loop) Queue an event for one stream, unless it has fallen SUB_QUEUE events behind."""
        try:
            q.put_nowait(evt)
        except asyncio.QueueFull:
            q.behind = True   # the stream ends when it is read again; the page reconnects and catches up

    def subscribe(self) -> _Stream:
        q = _Stream(maxsize=SUB_QUEUE)
        with self._lock:
            self._subs.append(q)
        return q

    def since(self, seq: int) -> list[dict]:
        """The kept events after `seq`. (Under the lock: a worker thread trimming the list meanwhile made a copy taken
        without it skip events.)"""
        with self._lock:
            return [e for e in self.events if e["seq"] > seq]

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def _open_turn(self, message: str) -> dict:
        # (numbered as the harness numbers its turns, which go on from where a resumed session stopped)
        turn = {"id": max(len(self.turns), getattr(self.harness, "turn_no", 0)) + 1, "message": message, "status": "running", "started": time.time(),
                "pin": self.pin, "tier_mode": self.harness.tier_mode,
                # which computer it was written on, shown under the message - only while there are others to tell apart
                "origin": (self.devices.here if self.harness.origin == HERE else self.harness.origin)
                if self.devices.online() else None}
        self.turns.append(turn)
        self.publish("turn_start", {"turn": dict(turn)})
        return turn

    def _run_turn(self, turn: dict, message: str, resume: str | None) -> None:
        try:
            r = self.harness.resume(resume) if resume else self.harness.turn(message, force=self.pin)
            turn.update(usage=r.usage, checkpoint=r.checkpoint)
            turn.update(status="cancelled" if r.cancelled else "done", persona=r.persona, answer=r.answer, artifacts=r.artifacts,
                        seconds=r.seconds, trace=jsonable(r.trace), tasks=[task_json(t) for t in r.tasks])
            self.publish("turn_done", {"turn": dict(turn)})
        except Exception as e:  # noqa: BLE001 - report, never kill the server
            turn.update(status="error", answer=f"{type(e).__name__}: {e}", seconds=time.time() - turn["started"])
            self.publish("turn_error", {"turn": dict(turn)})

    def start_turn(self, message: str, resume: str | None = None, origin: str | None = None) -> dict:
        """`origin`: the computer the message was written on (HERE, a device's name, or None): "this computer" in it means
        that one. (Requests queued while a turn ran keep the origin of the turn they follow.)"""
        if not self.busy.acquire(blocking=False):
            raise HTTPException(409, "a turn is already running")
        if not resume:
            self.harness.origin = origin
        turn = self._open_turn(message)

        def run() -> None:
            released = False
            try:
                self._run_turn(turn, message, resume)
                while not released:   # requests sent while it ran (jevh/interject.py): the next turns, in order
                    with self._drain:
                        queued = self.harness.take_queued()
                        if queued is None:
                            self.busy.release()
                            released = True
                    if queued is not None:
                        self._run_turn(self._open_turn(queued), queued, None)
            finally:
                if not released:
                    self.busy.release()

        threading.Thread(target=run, name=f"turn-{turn['id']}", daemon=True).start()
        return turn

    def reset(self) -> None:
        # (held, not just looked at: a message arriving between a look and the swap started its turn on the harness
        # being replaced - that turn's Stop then went to the new one, and clearing `turns` dropped it from the page)
        if not self.busy.acquire(blocking=False):
            raise HTTPException(409, "wait for the running turn to finish")
        try:
            self.close_browser()   # the session is over: so is the browser it kept open between turns
            mode, project, watch = self.harness.tier_mode, self.harness.project, self.harness.watch_browser
            self.harness = self._factory(self.publish)
            self.harness.tier_mode, self.harness.project, self.harness.watch_browser = mode, project, watch
            self._adopt(self.harness)
            self.turns.clear()
            self.publish("reset", {"session": self.harness.session})
        finally:
            self.busy.release()

    def load(self, session: str) -> dict:
        """Carry on a past session: a harness for it (its memory, so Jev's follow-up check and the context see the
        earlier turns; its folder, so new work lands beside the old) and its conversation shown again in the chat."""
        if not self.busy.acquire(blocking=False):   # (held meanwhile: no message can start a turn on the harness we swap out)
            raise HTTPException(409, "wait for the running turn to finish")
        try:
            if session == self.harness.session:
                return {"session": session, "turns": len(self.turns), "already": True}
            rec = session_record(self.harness.cfg.workspace, self.harness.memory, self.harness.failures, session)
            self.close_browser()   # the session we leave is over for now: so is the browser it kept open
            mode, project, watch = self.harness.tier_mode, self.harness.project, self.harness.watch_browser
            self.harness = self._factory(self.publish, session=session)
            self.harness.tier_mode, self.harness.project, self.harness.watch_browser = mode, project, watch
            self._adopt(self.harness)
            self.turns.clear()
            self.publish("reset", {"session": session, "loaded": True})
            for t in rec["turns"]:   # (published one by one: an event carries the id of the newest turn)
                turn = {"id": t["n"], "message": t["user"], "status": "done", "started": t["ts"], "persona": t["persona"],
                        "answer": t["assistant"], "artifacts": [], "tasks": [], "restored": True}
                self.turns.append(turn)
                self.publish("turn_done", {"turn": dict(turn)})
            if self.turns:
                self.harness.turn_no = max(self.harness.turn_no, self.turns[-1]["id"])
            return {"session": session, "turns": len(self.turns)}
        finally:
            self.busy.release()

    def close_browser(self) -> None:
        """Close the docked Chrome this session kept open between turns (jevh/tools/browser.py close_kept), in the background."""
        from ..tools.browser import close_kept
        folder = Path(self.harness.session_dir)
        threading.Thread(target=close_kept, args=(folder,), name="close-browser", daemon=True).start()


# ---------------------------------------------------------------- docked browser

KEYS = {"Enter", "Backspace", "Tab", "Escape", "Delete", "Home", "End", "PageUp", "PageDown",
        "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"}


class BrowserDock:
    """Mirrors the browser tool's docked Chromium (JEVH_BROWSER_DOCK) into the dashboard: frames of
    the page it is on, and the user's clicks and typing passed on to it. The tool writes
    <session>/browsers/<pid>.json with its CDP port; we attach over CDP and never close it (it
    belongs to the worker). Playwright's sync API is bound to one thread: every call runs on ours."""

    def __init__(self, folder):
        self.folder = folder
        self._jobs: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def call(self, fn, timeout: float = 10.0):
        """fn(page) on the dock thread -> its result, or None when no docked browser is up."""
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="browser-dock", daemon=True)
                self._thread.start()
        fut: Future = Future()
        self._jobs.put((fn, fut))
        return fut.result(timeout)

    def _newest(self, dead: set) -> dict | None:
        try:
            files = sorted(Path(self.folder()).glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        except OSError:
            return None
        for f in files:
            try:
                key = (str(f), f.stat().st_mtime)
                if key not in dead:
                    return {**json.loads(f.read_text(encoding="utf-8")), "key": key}
            except (OSError, ValueError):
                continue   # being written or just removed
        return None

    def _run(self) -> None:
        from playwright.sync_api import sync_playwright
        pw = sync_playwright().start()
        browser, at, dead = None, None, set()   # dead: info files whose browser is gone (a killed worker)
        failed: dict = {}                       # attach attempts that failed, per info file: a slow Chrome gets a few tries
        try:
            while True:
                fn, fut = self._jobs.get()
                try:
                    info = self._newest(dead)
                    if info and (browser is None or at != info["key"] or not browser.is_connected()):
                        browser = None
                        try:
                            browser = pw.chromium.connect_over_cdp(f"http://127.0.0.1:{info['port']}", timeout=8000)
                            at = info["key"]
                        except Exception:  # noqa: BLE001
                            failed[info["key"]] = failed.get(info["key"], 0) + 1
                            if failed[info["key"]] >= 4:
                                dead.add(info["key"])
                    pages = [p for c in browser.contexts for p in c.pages if not p.is_closed()] if info and browser else []
                    fut.set_result(fn(pages[-1], info) if pages else None)   # the tool follows the newest tab too
                except Exception as e:  # noqa: BLE001
                    browser = None
                    fut.set_exception(e)
        finally:
            pw.stop()

    def frame(self) -> dict | None:
        def shot(page, info):
            try:
                title = page.title()
            except Exception:  # noqa: BLE001 - mid-navigation
                title = ""
            return {"jpeg": page.screenshot(type="jpeg", quality=70, timeout=5000), "url": page.url, "title": title,
                    "id": f"{info['pid']}-{info['started']}"}
        return self.call(shot)

    def send(self, ev: dict) -> bool:
        """The user's input: click/scroll at (x, y) as fractions of the frame, type text, press a key."""
        kind = ev.get("kind")

        def act(page, _info):
            # (a real Chrome has no viewport set by Playwright: ask the page how big it is)
            vp = page.viewport_size or page.evaluate("({width: innerWidth, height: innerHeight})")
            x = min(max(float(ev.get("x", 0)), 0.0), 1.0) * vp["width"]
            y = min(max(float(ev.get("y", 0)), 0.0), 1.0) * vp["height"]
            if kind == "click":
                page.mouse.click(x, y)
            elif kind in ("down", "up", "move"):   # the user's own press-and-hold or drag, as it happens
                page.mouse.move(x, y)
                if kind == "down":
                    page.mouse.down()
                elif kind == "up":
                    page.mouse.up()
            elif kind == "scroll":
                page.mouse.move(x, y)
                page.mouse.wheel(0, max(-2000.0, min(2000.0, float(ev.get("dy", 0)))))
            elif kind == "type":
                page.keyboard.type(str(ev.get("text", ""))[:2000])
            elif kind == "key" and ev.get("key") in KEYS:
                page.keyboard.press(ev["key"])
            else:
                raise ValueError("unknown input")
            return True
        return bool(self.call(act))


# ---------------------------------------------------------------- status probes

_gpu_cache: dict[str, Any] = {"ts": 0.0, "data": []}


def gpus() -> list[dict]:
    if time.time() - _gpu_cache["ts"] < 3:
        return _gpu_cache["data"]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        data = []
        for line in out.strip().splitlines():
            i, name, used, total, util = [x.strip() for x in line.split(",")]
            data.append({"index": int(i), "name": name, "used_mb": int(used), "total_mb": int(total), "util": int(util)})
    except Exception:  # noqa: BLE001
        data = []
    _gpu_cache.update(ts=time.time(), data=data)
    return data


def probe(url: str, timeout: float = 2.0) -> dict:
    t = time.time()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read(4000).decode("utf-8", "replace")
        try:
            body = json.loads(body)
        except ValueError:
            pass
        return {"up": True, "ms": round((time.time() - t) * 1000), "info": body}
    except Exception as e:  # noqa: BLE001
        return {"up": False, "error": str(getattr(e, "reason", e))[:200]}


# ---------------------------------------------------------------- FastAPI app

def create_app(cfg: Config | None = None, port: int = 8740, harness_factory=None,
               model_override: str | None = None, token: str | None = None,
               allow_hosts: list[str] | None = None, access_key: str | None = None) -> FastAPI:
    cfg = cfg or load_config()
    hub = Hub(cfg, harness_factory, model_override)
    token = token or secrets.token_urlsafe(24)
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    allowed_origins = {f"http://{h}" for h in allowed_hosts}
    # each entry is a name or name:port (a `tailscale serve --https=10000` front is name:10000); the bare name is
    # allowed too, since the proxy's Host header omits the port on 443
    remote_hosts = {h for e in (allow_hosts or []) for h in (e.lower(), e.lower().partition(":")[0]) if h}
    remote_origins = {f"{scheme}://{h}" for h in remote_hosts for scheme in ("http", "https")}
    allowed_hosts |= remote_hosts
    allowed_origins |= remote_origins
    access_key = access_key or secrets.token_urlsafe(24)
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        hub.loop = asyncio.get_running_loop()  # worker threads hand events to SSE via this loop
        hub.devices.loop = hub.loop            # and calls to devices to their connections
        if hub.newsroom_cfg["enabled"]:
            hub.newsroom_runner.start()
        yield
        hub.newsroom_runner.stop()
        try:   # the dashboard is going away: so is the browser the session kept open
            from ..tools.browser import close_kept
            await asyncio.to_thread(close_kept, Path(hub.harness.session_dir))
        except Exception:  # noqa: BLE001
            pass

    app = FastAPI(title="jev-harness dashboard", docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=lifespan)
    app.state.hub, app.state.token, app.state.access_key = hub, token, access_key

    accounts = hub.accounts
    local_ips = this_machine_ips()
    hub_urls = [f"https://{e.lower()}" for e in (allow_hosts or [])]   # how other devices reach this dashboard

    def is_access_key(value: str | None) -> bool:
        # bytes: compare_digest refuses a str with non-ASCII characters, which a ?key=%C3%A9 would be
        return bool(value) and secrets.compare_digest(value.encode(), access_key.encode())

    def client_of(request) -> str:
        """Who is asking (for throttling, and which computer a message comes from): `tailscale serve` connects from
        127.0.0.1 and appends the tailnet address of the device it proxies for to X-Forwarded-For. The last entry is the
        one it wrote; an earlier one could be the client's own (measured 2026-10-12: tailscale sends the client's
        100.x address there)."""
        return forwarded_ip(request.headers.get("x-forwarded-for"), request.client.host if request.client else "")

    def origin_of(request) -> str | None:
        """Which computer the user is writing from: HERE (this one, the hub), a connected device's name, or None (a
        phone, or a browser on a computer that is not paired)."""
        if not request.state.remote:
            return HERE
        ip = client_of(request)
        if ip in local_ips:
            return HERE
        return hub.devices.device_at(ip)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = request.headers.get("host", "").lower()
        if host not in allowed_hosts:
            return JSONResponse({"detail": "bad host"}, status_code=403)
        remote = host in remote_hosts
        path = request.url.path
        key_sent = request.query_params.get("key")
        user = accounts.session_user(request.cookies.get(SESSION_COOKIE))
        request.state.remote, request.state.user = remote, user
        # (holds this launch's access key: on another device before any account exists, the only way in)
        request.state.keyed = remote and (is_access_key(key_sent) or is_access_key(request.cookies.get("jevh_key")))
        if remote and path not in OPEN_PATHS:
            if accounts.has_users():   # accounts exist: signing in is the only way in from another device
                if user is None:
                    if path.startswith("/api/"):
                        return JSONResponse({"detail": "sign in first"}, status_code=401)
                    return RedirectResponse("/login", status_code=303)
            # no accounts yet: only a browser holding this launch's access key gets in. The link's ?key= counts beside
            # the cookie: the key changes each launch, so a phone still holding the last launch's cookie must not be
            # locked out of the new link by it
            elif not (is_access_key(key_sent) or is_access_key(request.cookies.get("jevh_key"))):
                return JSONResponse({"detail": "missing or bad access key"}, status_code=403)
        origin = request.headers.get("origin")
        if request.method != "GET" and origin is not None and origin not in allowed_origins:
            return JSONResponse({"detail": "foreign origin"}, status_code=403)
        if path.startswith("/api/"):
            sent = request.headers.get("x-jevh-token") or request.query_params.get("token")
            if not sent or not secrets.compare_digest(sent, token):
                return JSONResponse({"detail": "missing or bad token"}, status_code=403)
        response = await call_next(request)
        if remote and is_access_key(key_sent) and not accounts.has_users():   # (this also replaces a stale cookie)
            response.set_cookie("jevh_key", access_key, httponly=True, secure=True, samesite="strict")
        if remote and user and request.method == "GET":
            # renew the browser's copy too (a page load, or the open page's own requests: the app can stay open for weeks):
            # the hub slides the sign-in's 30 days with each use, and a cookie still set to end 30 days after signing in
            # would sign the device out anyway
            response.set_cookie(SESSION_COOKIE, request.cookies.get(SESSION_COOKIE, ""), max_age=SESSION_DAYS * 86400,
                                httponly=True, secure=True, samesite="lax")
        return response

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8").replace("__JEVH_TOKEN__", token)

    @app.get("/orb", response_class=HTMLResponse)
    def orb_page() -> str:
        """The desktop app's orb chat bubble (jevh/app.py): the same conversation, small; same sign-in and token."""
        return (STATIC / "orb.html").read_text(encoding="utf-8").replace("__JEVH_TOKEN__", token)

    # ------------------------------------------------------------ signing in (jevh/accounts.py)
    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request):
        if request.state.user or (not request.state.remote and not accounts.has_users()):
            return RedirectResponse("/", status_code=303)
        return (STATIC / "login.html").read_text(encoding="utf-8").replace(
            "__JEVH_NO_ACCOUNTS__", "" if accounts.has_users() else "1").replace("__JEVH_INVITE__", "")

    @app.post("/auth/login")
    async def login(request: Request):
        body = await request.json()
        who = client_of(request)
        if accounts.throttled(f"login:{who}"):
            raise HTTPException(429, "Too many tries: wait a few minutes.")
        name = await asyncio.to_thread(accounts.verify, str(body.get("name", "")), str(body.get("password", "")), who)
        if name is None:
            raise HTTPException(401, "That name and password don't match.")
        resp = JSONResponse({"user": name})
        resp.set_cookie(SESSION_COOKIE, accounts.start_session(name, request.headers.get("user-agent", "")),
                        max_age=SESSION_DAYS * 86400, httponly=True, secure=request.state.remote, samesite="lax")
        return resp

    @app.post("/auth/logout")
    def logout(request: Request):
        accounts.end_session(request.cookies.get(SESSION_COOKIE))
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(SESSION_COOKIE)
        return resp

    @app.get("/setup", response_class=HTMLResponse)
    def setup_page(invite: str = ""):
        """An invite link (jevh/accounts.py invites): make your account, from any device."""
        page = (STATIC / "login.html").read_text(encoding="utf-8").replace("__JEVH_NO_ACCOUNTS__", "")
        return page.replace("__JEVH_INVITE__", "1" if accounts.invite_ok(invite) else "0")

    @app.post("/auth/setup")
    async def setup(request: Request):
        body = await request.json()
        try:
            name = await asyncio.to_thread(accounts.accept_invite, str(body.get("invite", "")), str(body.get("name", "")),
                                           str(body.get("password", "")), client_of(request))
        except AccountError as e:
            raise HTTPException(400, str(e)) from None
        resp = JSONResponse({"user": name})
        resp.set_cookie(SESSION_COOKIE, accounts.start_session(name, request.headers.get("user-agent", "")),
                        max_age=SESSION_DAYS * 86400, httponly=True, secure=request.state.remote, samesite="lax")
        return resp

    @app.post("/api/accounts/invite")
    def make_invite(request: Request) -> dict:
        """From the hub computer, or by someone signed in: a one-time link to make an account from any device."""
        if request.state.remote and not request.state.user:
            raise HTTPException(403, "Sign in first.")
        token, expires = accounts.invite(request.state.user or "hub computer")
        base = hub_urls[0] if hub_urls else f"http://127.0.0.1:{port}"
        return {"link": f"{base}/setup?invite={token}", "expires": expires,
                "remote": bool(hub_urls), "note": "" if hub_urls else "This hub is not reachable from other devices "
                "(no --allow-host): the link only works on this computer."}

    @app.post("/auth/pair")
    async def pair(request: Request) -> dict:
        """A device trades a pairing code for its key (jevh/devices/agent.py pair). No sign-in: the code is the proof."""
        body = await request.json()
        try:
            dev, key = accounts.redeem(str(body.get("code", "")), str(body.get("name", "")), str(body.get("platform", "")),
                                       who=client_of(request))
        except AccountError as e:
            raise HTTPException(400, str(e)) from None
        hub.publish("devices", {"name": dev.name, "paired": True})
        return {"device": {"id": dev.id, "name": dev.name}, "key": key, "hub": hub.devices.here}

    @app.get("/api/me")
    def me(request: Request) -> dict:
        return {"user": request.state.user, "local": not request.state.remote, "accounts": accounts.has_users(),
                "keyed": bool(request.state.keyed) and not accounts.has_users(),
                "users": [u["name"] for u in accounts.users()] if (request.state.user or not request.state.remote) else [],
                "here": hub.devices.here, "hubs": hub_urls}

    @app.post("/api/accounts")
    async def add_account(request: Request):
        """A new account: from this computer, by someone already signed in, or - the first one only - from another device
        that came in with this launch's key link (whoever holds that link already has the whole assistant, so letting them
        make the first account gives them nothing more; it is how the app on another computer gets started). That first
        account is signed in at once on the device that made it: from then on, the key link no longer opens the dashboard."""
        first_by_key = request.state.remote and request.state.keyed and not accounts.has_users()
        if request.state.remote and not request.state.user and not first_by_key:
            raise HTTPException(403, "Sign in first.")
        body = await request.json()
        try:
            name = await asyncio.to_thread(accounts.add_user, str(body.get("name", "")), str(body.get("password", "")))
        except AccountError as e:
            raise HTTPException(400, str(e)) from None
        resp = JSONResponse({"user": name, "users": [u["name"] for u in accounts.users()], "signed_in": first_by_key})
        if first_by_key:
            resp.set_cookie(SESSION_COOKIE, accounts.start_session(name, request.headers.get("user-agent", "")),
                            max_age=SESSION_DAYS * 86400, httponly=True, secure=True, samesite="lax")
        return resp

    @app.post("/api/accounts/password")
    async def change_password(request: Request) -> dict:
        body = await request.json()
        name = request.state.user or (str(body.get("name", "")) if not request.state.remote else "")
        if not name:
            raise HTTPException(403, "Sign in first.")
        if request.state.user and not await asyncio.to_thread(accounts.verify, name, str(body.get("old", "")),
                                                               client_of(request)):
            raise HTTPException(401, "The current password is wrong.")
        try:
            await asyncio.to_thread(accounts.set_password, name, str(body.get("password", "")))
        except AccountError as e:
            raise HTTPException(400, str(e)) from None
        return {"ok": True, "note": "Every browser signed in to this account was signed out."}

    # ------------------------------------------------------------ devices (jevh/devices)
    @app.get("/api/devices")
    def devices_list() -> dict:
        return {"devices": hub.devices.list(), "here": hub.devices.here, "hubs": hub_urls}

    @app.post("/api/devices/pair-code")
    def devices_pair_code(request: Request) -> dict:
        if not hub_urls:
            raise HTTPException(400, "Other computers reach this dashboard through `tailscale serve`: restart it with "
                                     "--allow-host <this machine>.<tailnet>.ts.net:<port> first (see the README).")
        code, expires = accounts.pair_code(request.state.user or "local")
        return {"code": code, "expires": expires, "hubs": hub_urls,
                "command": f"jevh device pair {hub_urls[0]} {code}"}

    @app.post("/api/devices/rename")
    async def devices_rename(request: Request) -> dict:
        body = await request.json()
        dev = accounts.device(str(body.get("id", "")))
        if dev is None:
            raise HTTPException(404, "No such device.")
        try:
            accounts.rename(dev.id, str(body.get("name", "")))
        except AccountError as e:
            raise HTTPException(400, str(e)) from None
        return devices_list()   # (the device hears its new name when it next connects)

    @app.post("/api/devices/revoke")
    async def devices_revoke(request: Request) -> dict:
        body = await request.json()
        dev = accounts.device(str(body.get("id", "")))
        if dev is None:
            raise HTTPException(404, "No such device.")
        accounts.revoke(dev.id)
        await hub.devices.drop(dev.id)
        hub.publish("devices", {"name": dev.name, "revoked": True})
        return devices_list()

    @app.websocket(wire.PATH)
    async def device_socket(ws: WebSocket):
        """A device's one connection (jevh/devices/agent.py). The http guard does not see WebSockets: the Host is checked
        here, and the device's key (sent as a bearer token, which no web page can set on a WebSocket) is the sign-in."""
        if ws.headers.get("host", "").lower() not in allowed_hosts:
            await ws.close(code=1008)
            return
        auth = ws.headers.get("authorization", "")
        dev = accounts.device_for_key(auth[7:].strip() if auth.lower().startswith("bearer ") else None)
        await ws.accept()
        if dev is None:
            await ws.close(code=wire.CLOSE_REVOKED)
            return
        try:
            await hub.devices.serve(ws, dev, ip=forwarded_ip(ws.headers.get("x-forwarded-for"),
                                                              ws.client.host if ws.client else ""))
        except WebSocketDisconnect:
            pass

    @app.get("/api/state")
    def state() -> dict:
        h = hub.harness
        return {"session": h.session, "tier_mode": h.tier_mode, "pin": hub.pin, "busy": hub.busy.locked(),
                "fast_model": (h.cfg.models.get("fast") or {}).get("model"), "model_override": hub.model_override,
                "personas": {n: {"description": p.description, "model": p.model.get("model"),
                                 "fast_ok": p.fast_ok, "draft": p.draft} for n, p in h.personas.items()},
                "skills": len(h.roster), "turns": hub.turns, "last_seq": hub.seq, "project": h.project,
                "watch_browser": h.watch_browser, "questions": h.pending_questions(), "queued": h.queued(),
                "thresholds": h.cfg.thresholds, "problems": h.problems()}

    @app.post("/api/chat")
    async def chat(request: Request) -> dict:
        body = await request.json()
        message = str(body.get("message", "")).strip()
        if not message:
            raise HTTPException(400, "empty message")
        if hub.busy.locked():
            # sent while a turn runs: it is added to the running task's prompt, unless it is a question about
            # progress (Jev tells) or a /command says otherwise: /queue, /stop, /btw, /steer (jevh/interject.py).
            # Jev is asked off the event loop.
            kind = body.get("kind") if body.get("kind") in KINDS else None
            res = await asyncio.to_thread(hub.harness.interject, message, kind)
            if res["outcome"] != "idle":
                return {"interjected": res}
            message = res.get("text") or message
            with hub._drain:   # the turn ended meanwhile, or the runner is between two queued ones
                if hub.busy.locked():
                    hub.harness.enqueue(message)
                    return {"interjected": {"outcome": "queued", "kind": "separate", "text": message,
                                            "queue": hub.harness.queued()}}
        return {"turn": hub.start_turn(message, origin=origin_of(request))}

    @app.get("/api/usage")
    def usage(range: str = "session") -> dict:
        h = hub.harness
        since = {"today": time.time() - 86400, "week": time.time() - 7 * 86400}.get(range)
        session = h.session if range == "session" else None
        return {"range": range, "summary": h.usage.summary(session=session, since=since),
                "turns": h.usage.turns(session=session, limit=60)}

    @app.get("/api/failures")
    def failures(range: str = "week", kind: str | None = None, source: str | None = None,
                 status: str | None = None, limit: int = 100) -> dict:
        h = hub.harness
        since = {"today": time.time() - 86400, "week": time.time() - 7 * 86400}.get(range)
        session = h.session if range == "session" else None
        return {"range": range, "counts": h.failures.counts(session=session, since=since),
                "failures": [{k: v for k, v in f.items() if k != "work_log"} for f in h.failures.list(
                    session=session, kind=kind, source=source, status=status, since=since, limit=min(limit, 500))]}

    @app.post("/api/failures/status")
    async def failure_status(request: Request) -> dict:
        body = await request.json()
        try:
            ok = hub.harness.failures.set_status(int(body.get("id", 0)), str(body.get("status", "")), body.get("note"))
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        if not ok:
            raise HTTPException(404, "no such failure")
        return {"ok": True}

    # -- session review: flag a session, add notes, have a model judge it (jevh/review.py) ------------
    def flag_or_404(fid) -> dict:
        try:
            f = hub.harness.review.reviews.get(int(fid))
        except (TypeError, ValueError):
            f = None
        if f is None:
            raise HTTPException(404, "no such flagged session")
        return f

    def flag_out(f: dict) -> dict:
        return {**f, "running": f["id"] in hub.harness.review.running}

    @app.get("/api/reviews")
    def reviews(status: str | None = None) -> dict:
        rv = hub.harness.review
        current = rv.reviews.find(hub.harness.session)
        return {"session": hub.harness.session, "counts": rv.reviews.counts(),
                "current": flag_out(current) if current else None,
                "flags": [flag_out(f) for f in rv.reviews.list(status=status or None, limit=200)]}

    @app.get("/api/reviews/{fid}")
    def review_detail(fid: int) -> dict:
        f = flag_or_404(fid)
        rec = hub.harness.review.record(f["session"])
        for t in rec["tasks"]:
            t["work_log"] = t["work_log"][-6000:]
        return {"flag": flag_out(f), "record": rec}

    @app.get("/api/sessions")
    def sessions(limit: int = 30) -> list:
        h = hub.harness
        flagged = {f["session"]: f["id"] for f in h.review.reviews.list(limit=10 ** 6)}
        return [{**s, "flag": flagged.get(s["session"]), "current": s["session"] == h.session}
                for s in list_sessions(h.cfg.workspace, h.memory, min(limit, 200))]

    # -- past sessions, categorized by Jev (jevh/history.py) ---------------------------------------------
    @app.get("/api/history")
    def history(limit: int = 500) -> dict:
        """Every session with Jev's category for it (null while Jev has not been asked: `pending` says how many wait)."""
        return hub.history.overview(min(limit, 2000))

    @app.get("/api/history/{session}")
    def history_session(session: str) -> dict:
        out = hub.history.transcript(session)
        if out is None:
            raise HTTPException(404, "no such session")
        return out

    @app.post("/api/history/{session}/load")
    async def history_load(session: str) -> dict:
        """Carry on a past session: new messages join its conversation and its folder. Refused while a turn runs."""
        if not hub.history.exists(session):
            raise HTTPException(404, "no such session")
        return await asyncio.to_thread(hub.load, session)

    @app.post("/api/history/recategorize")
    async def history_recategorize(request: Request) -> dict:
        """Ask Jev again about one session (`session`) or, without it, all of them."""
        body = await request.json()
        session = str(body.get("session") or "") or None
        if session and not hub.history.exists(session):
            raise HTTPException(404, "no such session")
        return {"pending": await asyncio.to_thread(hub.history.recategorize, session)}

    @app.post("/api/reviews/flag")
    async def review_flag(request: Request) -> dict:
        """Flag a session (default: this one) with an optional note, or add a note to its flag."""
        body = await request.json()
        h = hub.harness
        session = str(body.get("session") or h.session)
        if session != h.session and session not in {s["session"] for s in list_sessions(h.cfg.workspace, h.memory, 10 ** 6)}:
            raise HTTPException(404, "no such session")
        turn = body.get("turn")
        f = h.review.flag(session, str(body.get("note") or ""), int(turn) if turn else None)
        hub.publish("review_flag", {"flag": {"id": f["id"], "session": f["session"]}})
        return {"flag": flag_out(f)}

    @app.post("/api/reviews/note")
    async def review_note(request: Request) -> dict:
        body = await request.json()
        f = flag_or_404(body.get("id"))
        turn = body.get("turn")
        try:
            hub.harness.review.reviews.add_note(f["id"], str(body.get("text") or ""), int(turn) if turn else None)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        return {"flag": flag_out(flag_or_404(f["id"]))}

    @app.post("/api/reviews/run")
    async def review_run(request: Request) -> dict:
        """Start a model review of a flagged session; review_done / review_error events report the end."""
        body = await request.json()
        f = flag_or_404(body.get("id"))
        rv = hub.harness.review
        if f["id"] in rv.running:
            raise HTTPException(409, "a review of this session is already running")
        model = str(body.get("model") or "").strip() or None
        hub.publish("review_start", {"flag": {"id": f["id"], "session": f["session"]}})

        def run() -> None:
            try:
                out = rv.run(f["id"], model)
                hub.publish("review_done", {"flag": {"id": out["id"], "session": out["session"],
                                                     "verdict": out["verdict"], "verdict_by": out["verdict_by"]}})
            except Exception as e:  # noqa: BLE001 - reported to the page, never kills the server
                hub.publish("review_error", {"flag": {"id": f["id"], "session": f["session"]},
                                             "error": f"{type(e).__name__}: {e}"[:600]})

        threading.Thread(target=run, name=f"review-{f['id']}", daemon=True).start()
        return {"running": True}

    @app.post("/api/reviews/verdict")
    async def review_verdict(request: Request) -> dict:
        body = await request.json()
        f = flag_or_404(body.get("id"))
        try:
            out = hub.harness.review.decide(f["id"], str(body.get("verdict", "")), str(body.get("reason") or ""))
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        return {"flag": flag_out(out)}

    @app.post("/api/reviews/status")
    async def review_status(request: Request) -> dict:
        body = await request.json()
        f = flag_or_404(body.get("id"))
        status = str(body.get("status", ""))
        if status not in ("open", "dismissed"):
            raise HTTPException(400, "status must be open|dismissed (a verdict decides a flag)")
        # a dismissed flag that already had a verdict comes back decided
        hub.harness.review.reviews.set_status(f["id"], "decided" if status == "open" and f["verdict"] else status)
        return {"flag": flag_out(flag_or_404(f["id"]))}

    # -- notebook: ideas and problems to come back to (jevh/notebook.py) -------------------------------
    def entry_or_404(eid) -> dict:
        try:
            e = hub.harness.notebook.get(int(eid))
        except (TypeError, ValueError):
            e = None
        if e is None:
            raise HTTPException(404, "no such notebook entry")
        return e

    @app.get("/api/notebook")
    def notebook_list(kind: str | None = None, status: str | None = None, q: str | None = None,
                      limit: int = 500) -> dict:
        nb = hub.harness.notebook
        try:
            entries = nb.list(kind=kind or None, status=status or None, query=q, limit=max(1, min(limit, 2000)))
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        return {"counts": nb.counts(), "entries": entries}

    @app.post("/api/notebook")
    async def notebook_add(request: Request) -> dict:
        """Jot something down. `session: true` ties it to the session you are in."""
        body = await request.json()
        h = hub.harness
        try:
            e = h.notebook.add(str(body.get("text") or ""), str(body.get("kind") or "note"),
                               h.session if body.get("session") else "", bool(body.get("pinned")))
        except ValueError as err:
            raise HTTPException(400, str(err)) from None
        return {"entry": e}

    @app.post("/api/notebook/update")
    async def notebook_update(request: Request) -> dict:
        body = await request.json()
        e = entry_or_404(body.get("id"))
        change = {k: body[k] for k in ("text", "kind", "status", "pinned") if k in body}
        try:
            out = hub.harness.notebook.update(e["id"], **change)
        except ValueError as err:
            raise HTTPException(400, str(err)) from None
        if out is None:
            raise HTTPException(404, "no such notebook entry")
        return {"entry": out}

    @app.post("/api/notebook/delete")
    async def notebook_delete(request: Request) -> dict:
        body = await request.json()
        e = entry_or_404(body.get("id"))
        hub.harness.notebook.delete(e["id"])
        return {"ok": True}

    # -- the Newsroom: tiles that follow news and posts, and the notifications they make (jevh/newsroom.py) ------
    def board(items: int = 20) -> dict:
        room = hub.newsroom
        ws = room.widgets()
        for w in ws:
            w["items"] = room.items(w["id"], items)
        return {"rev": room.rev(), "widgets": ws, "notes": room.notes(limit=40), "unread_notes": room.unread_notes(),
                "kinds": nr.kinds(), "running": hub.newsroom_cfg["enabled"], "min_minutes": hub.newsroom_cfg["min_minutes"],
                "default_minutes": hub.newsroom_cfg["default_minutes"]}

    def newsroom_call(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except nr.NewsroomError as e:
            raise HTTPException(400, str(e)) from None

    def num(body: dict, key: str, default: int = 0) -> int:
        try:
            return int(body.get(key, default))
        except (TypeError, ValueError):
            raise HTTPException(400, f"{key} must be a number") from None

    @app.get("/api/newsroom")
    def newsroom_board(rev: int = -1, items: int = 20) -> dict:
        """The board. With `rev` equal to the current one, only says nothing changed (a page polls with it)."""
        if rev == hub.newsroom.rev():
            return {"rev": rev, "unchanged": True, "unread_notes": hub.newsroom.unread_notes()}
        return board(max(1, min(items, 100)))

    @app.post("/api/newsroom/add")
    async def newsroom_add(request: Request) -> dict:
        """Add a tile by hand (the Add widget form) and look at it once. The chat does the same through the worker's tool."""
        body = await request.json()
        room = hub.newsroom
        w = newsroom_call(room.add_widget, str(body.get("kind") or ""), body.get("params") or {}, title=body.get("title") or None,
                          every=body.get("every"), alert=body.get("alert"), size=str(body.get("size") or "m"),
                          session=hub.harness.session)
        if not w.get("existing"):
            r = await asyncio.to_thread(room.refresh, w["id"])
            if r["alerts"]:
                hub.publish("newsroom_alert", {"notes": r["alerts"]})
        return {"widget": room.widget(w["id"]), "existing": bool(w.get("existing"))}

    @app.post("/api/newsroom/update")
    async def newsroom_update(request: Request) -> dict:
        body = await request.json()
        change = {k: body[k] for k in ("title", "every", "alert", "paused", "size", "params") if k in body}
        w = newsroom_call(hub.newsroom.update_widget, num(body, "id"), **change)
        if "params" in change:   # follows something else: look at once
            await asyncio.to_thread(hub.newsroom.refresh, w["id"])
        return {"widget": hub.newsroom.widget(w["id"])}

    @app.post("/api/newsroom/delete")
    async def newsroom_delete(request: Request) -> dict:
        body = await request.json()
        if not hub.newsroom.delete_widget(num(body, "id")):
            raise HTTPException(404, "no such widget")
        return {"ok": True}

    @app.post("/api/newsroom/move")
    async def newsroom_move(request: Request) -> dict:
        body = await request.json()
        newsroom_call(hub.newsroom.move, num(body, "id"), num(body, "step"))
        return {"ok": True}

    @app.post("/api/newsroom/refresh")
    async def newsroom_refresh(request: Request) -> dict:
        """Look now: one tile (`id`) or every tile that isn't paused."""
        body = await request.json() if await request.body() else {}
        room = hub.newsroom
        ids = [num(body, "id")] if body.get("id") else [w["id"] for w in room.widgets() if not w["paused"]]
        results = await asyncio.to_thread(lambda: [newsroom_call(room.refresh, i) for i in ids])
        notes = [n for r in results for n in r["alerts"]]
        if notes:
            hub.publish("newsroom_alert", {"notes": notes})
        return {"results": [{k: v for k, v in r.items() if k != "alerts"} | {"alerts": len(r["alerts"])} for r in results]}

    @app.post("/api/newsroom/read")
    async def newsroom_read(request: Request) -> dict:
        body = await request.json()
        hub.newsroom.mark_read(item=num(body, "item") if body.get("item") else None,
                               widget=num(body, "widget") if body.get("widget") else None)
        return {"ok": True}

    @app.post("/api/newsroom/notes/read")
    async def newsroom_notes_read(request: Request) -> dict:
        body = await request.json() if await request.body() else {}
        hub.newsroom.read_notes(num(body, "id") if body.get("id") else None)
        return {"ok": True}

    @app.post("/api/newsroom/notes/clear")
    async def newsroom_notes_clear() -> dict:
        hub.newsroom.clear_notes()
        return {"ok": True}

    # -- Settings page: the model, the quick model and the web search engine (jevh/settings.py) ---------------------------
    def settings_payload(fresh: bool = False) -> dict:
        return {**usersettings.view(hub.cfg), "options": usersettings.model_options(hub.cfg, fresh=fresh),
                "providers": usersettings.providers_view(), "model_override": hub.model_override,
                "busy": hub.busy.locked()}

    @app.get("/api/config")
    def config_get(fresh: bool = False) -> dict:
        return settings_payload(fresh)

    @app.post("/api/config")
    async def config_save(request: Request) -> dict:
        """Save choices (any of model, fast, search) and use them from the next message. Refused while a turn runs: the
        turn keeps the model it has, and a model that changes under a worker is not a thing to do to it."""
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(400, "send an object")
        if hub.model_override and ("model" in body or "fast" in body):
            raise HTTPException(409, f"This dashboard was started with --model {hub.model_override}, which overrides the "
                                     "models. Restart it without that flag to choose here.")
        if not hub.busy.acquire(blocking=False):   # (held: no turn can start meanwhile)
            raise HTTPException(409, "Wait for the running turn to finish, then save again.")
        try:
            try:
                await asyncio.to_thread(usersettings.save, hub.cfg, body)
            except usersettings.SettingsError as e:
                raise HTTPException(400, str(e)) from None
            hub.harness.reload_settings()
        finally:
            hub.busy.release()
        return await asyncio.to_thread(settings_payload, True)

    @app.post("/api/config/reset")
    async def config_reset() -> dict:
        if not hub.busy.acquire(blocking=False):
            raise HTTPException(409, "Wait for the running turn to finish, then reset.")
        try:
            await asyncio.to_thread(usersettings.reset, hub.cfg)
            hub.harness.reload_settings()
        finally:
            hub.busy.release()
        return await asyncio.to_thread(settings_payload, True)

    @app.post("/api/config/test-search")
    async def config_test_search(request: Request) -> dict:
        """One real search with the choice as it is on the page (nothing is saved)."""
        body = await request.json()
        return await asyncio.to_thread(usersettings.test_search, hub.cfg, body if isinstance(body, dict) else {})

    @app.get("/api/thinking")
    def thinking() -> dict:
        """The running turn's newest live thinking file (its end): workers stream each step's
        reasoning and reply to <task folder>/thinking/step-NN.txt as they are generated."""
        h = hub.harness
        try:
            files = sorted(Path(h.session_dir).glob("*/thinking/step-*.txt"), key=lambda f: f.stat().st_mtime)
        except OSError:
            files = []
        if not files or not hub.busy.locked():
            return {"file": None}
        f = files[-1]
        size = f.stat().st_size
        with open(f, "rb") as fh:
            fh.seek(max(0, size - 4000))
            text = fh.read().decode("utf-8", "replace")
        return {"file": f"{f.parent.parent.name}/{f.name}", "bytes": size, "text": text,
                "idle": round(time.time() - f.stat().st_mtime, 1)}

    @app.get("/api/browser/frame")
    def browser_frame():
        """The docked browser's page as a JPEG (204: none is open). Only while Watch browser is on."""
        try:
            f = hub.dock.frame()
        except Exception:  # noqa: BLE001 - closing, navigating: the next poll gets one
            f = None
        if not f:
            return Response(status_code=204)
        q = urllib.parse.quote
        return Response(f["jpeg"], media_type="image/jpeg", headers={
            "Cache-Control": "no-store", "X-Browser-Id": f["id"], "X-Page-Url": q(f["url"], safe=":/?&=#%"),
            "X-Page-Title": q(f["title"][:200])})

    @app.post("/api/browser/input")
    async def browser_input(request: Request) -> dict:
        body = await request.json()
        try:
            ok = await asyncio.to_thread(hub.dock.send, body)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        except Exception as e:  # noqa: BLE001
            raise HTTPException(409, f"browser busy: {e}") from None
        if not ok:
            raise HTTPException(404, "no docked browser is open")
        return {"ok": True}

    @app.get("/api/resumable")
    def resumable() -> list:
        return hub.harness.resumable(10)

    @app.post("/api/resume")
    async def resume(request: Request) -> dict:
        body = await request.json()
        path = str(body.get("path", ""))
        if path not in {x["path"] for x in hub.harness.resumable(50)}:
            raise HTTPException(404, "nothing to resume there")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        task = data.get("task_text", "").split("\n\n(In your REPL:")[0]
        return {"turn": hub.start_turn(f"(resume) {task[:300]}", resume=path)}

    @app.post("/api/cancel")
    def cancel() -> dict:
        return {"stopped": hub.harness.cancel()}

    @app.post("/api/answer")
    async def answer(request: Request) -> dict:
        body = await request.json()
        text = str(body.get("answer", "")).strip()
        if not text:
            raise HTTPException(400, "empty answer")
        if not hub.harness.answer(str(body.get("id", "")), text):
            raise HTTPException(404, "that question is no longer waiting")
        return {"ok": True}

    @app.post("/api/settings")
    async def settings(request: Request) -> dict:
        body = await request.json()
        if "tier_mode" in body:
            if body["tier_mode"] not in ("auto", "fast", "strong"):
                raise HTTPException(400, "tier_mode must be auto|fast|strong")
            hub.harness.tier_mode = body["tier_mode"]
        if "pin" in body:
            if body["pin"] not in (None, "", *hub.harness.personas):
                raise HTTPException(400, "unknown persona")
            hub.pin = body["pin"] or None
        if "project" in body:
            try:
                hub.harness.set_project(str(body["project"]).strip() if body["project"] else None)
            except (NotADirectoryError, OSError) as e:
                raise HTTPException(400, f"not a folder: {e}") from None
        if "watch_browser" in body:
            hub.harness.watch_browser = bool(body["watch_browser"])  # applies from the next task's browser
        return {"tier_mode": hub.harness.tier_mode, "pin": hub.pin, "project": hub.harness.project,
                "watch_browser": hub.harness.watch_browser}

    @app.get("/api/memory")
    def memory() -> dict:
        h = hub.harness
        home = Path(h.skills_home).resolve()
        learned = [{"name": s.name, "description": s.description, "path": s.path, "draft": s.draft}
                   for s in h.roster.values() if (home / "learned") in Path(s.path).resolve().parents]
        lessons = []
        for note in sorted((home / "_lessons").glob("*.md")) if (home / "_lessons").is_dir() else []:
            text = note.read_text(encoding="utf-8", errors="replace")
            lessons.append({"name": note.stem, "entries": text.count("<!--")})
        for s in h.roster.values():  # our own skills improve in place
            if home in Path(s.path).resolve().parents and "## Lessons from past runs" in Path(s.path).read_text(
                    encoding="utf-8", errors="replace"):
                lessons.append({"name": s.name, "entries": s.body.count("<!--")})
        return {"facts": [dataclasses.asdict(f) for f in h.memory.facts(None, 200)], "turns": h.memory.count(),
                "learned": learned, "lessons": lessons,
                "persona_drafts": [{"name": p.name, "description": p.description, "active": p.active,
                                    "draft": p.draft, "source": p.source, "skills": p.skills_allowed,
                                    "tools": p.tools, "live": p.name in h.personas}
                                   for p in h.forge.drafts()],
                "persona_pending": [{"text": p["text"], "ts": p.get("ts")} for p in h.forge.pending()]}

    @app.post("/api/personas/review")
    async def review_persona(request: Request) -> dict:
        """Approve, activate or delete a persona the harness created. Only personas/learned/ is touched."""
        body = await request.json()
        h = hub.harness
        name, action = str(body.get("name", "")), body.get("action")
        if name not in {p.name for p in h.forge.drafts()}:
            raise HTTPException(404, "no such created persona waiting for review")
        if action == "approve":
            h.forge.approve(name)
        elif action == "activate":
            h.forge.activate(name)
        elif action == "delete":
            if hub.pin == name:
                hub.pin = None
            h.forge.delete(name)
        else:
            raise HTTPException(400, "action must be approve|activate|delete")
        h.reload_personas()
        return {"ok": True, "personas": list(h.personas)}

    @app.post("/api/personas/forge")
    def forge_sweep() -> dict:
        """Look at the learned skills that have no owner yet for one that fits no persona."""
        if hub.busy.locked():
            raise HTTPException(409, "a turn is running; try again when it is done")
        made = hub.harness.forge_sweep()
        return {"created": [m.name for m in made]}

    @app.post("/api/skills/review")
    async def review_skill(request: Request) -> dict:
        """Approve (drop the draft flag) or delete a learned skill. Only skills/learned/ is touched."""
        body = await request.json()
        h = hub.harness
        s = h.roster.get(str(body.get("name", "")))
        learned_root = (Path(h.skills_home) / "learned").resolve()
        if not s or learned_root not in Path(s.path).resolve().parents:
            raise HTTPException(404, "no such learned skill")
        path = Path(s.path)
        if body.get("action") == "approve":
            path.write_text(re.sub(r"^draft:\s*true\s*\n", "", path.read_text(encoding="utf-8"), count=1, flags=re.M),
                            encoding="utf-8")
        elif body.get("action") == "delete":
            # a skill's own folder only: a SKILL.md sitting straight in learned/ (the roster finds it: rglob) would
            # otherwise take every learned skill with it
            if learned_root not in path.parent.resolve().parents:
                raise HTTPException(409, f"{s.name} is not in a folder of its own under learned/: delete it by hand")
            shutil.rmtree(path.parent)
        else:
            raise HTTPException(400, "action must be approve|delete")
        h.roster = load_roster(h.cfg.skills_dirs, h.lesson_dirs)
        return {"ok": True}

    @app.post("/api/memory/forget")
    async def forget(request: Request) -> dict:
        body = await request.json()
        try:
            fact = int(body.get("id", -1))
        except (TypeError, ValueError):
            raise HTTPException(400, "id must be a number") from None
        hub.harness.memory.retire_fact(fact)
        return {"ok": True}

    @app.post("/api/reset")
    def reset() -> dict:
        hub.reset()
        return {"session": hub.harness.session}

    @app.get("/api/events")
    async def events(request: Request, since: int = 0):
        q = hub.subscribe()
        backlog = hub.since(since)

        async def stream():
            try:
                for e in backlog:
                    yield f"id: {e['seq']}\ndata: {json.dumps(e)}\n\n"
                last = backlog[-1]["seq"] if backlog else since
                while not await request.is_disconnected():
                    if q.behind:
                        break   # it fell SUB_QUEUE events behind: the page reconnects with `last` and catches up
                    try:
                        e = await asyncio.wait_for(q.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    if e["seq"] > last:
                        last = e["seq"]
                        yield f"id: {e['seq']}\ndata: {json.dumps(e)}\n\n"
            finally:
                hub.unsubscribe(q)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/status")
    def status() -> dict:
        h = hub.harness
        out: dict[str, Any] = {"gpus": gpus(), "swaps": hub.swaps[-20:], "problems": h.problems()}
        lm = next((p for p in h.cfg.providers.values() if p.get("exclusive")), None)
        if lm:
            try:
                gate = gate_for(lm["base_url"])
                loaded = [m for m in gate.llms_info() if m["loaded"]]
                out["lmstudio"] = {"up": True, "loaded": [m["key"] for m in loaded],
                                   "contexts": {m["key"]: m["context"] for m in loaded},   # what each is loaded with
                                   "gate": {"current": gate.current, "inflight": gate.inflight}}
            except Exception as e:  # noqa: BLE001
                out["lmstudio"] = {"up": False, "error": str(e)[:200]}
        # which servers the harness's models run on: models.strong/fast put every persona on one; else each persona's own
        tiers = {k: v for k, v in (h.cfg.models or {}).items() if v}
        used = {b.get("provider") for b in tiers.values()} if tiers.get("strong") else \
            {p.model.get("provider") for p in h.personas.values()}
        if getattr(h, "model_override", None):
            used = {h.model_override.partition(":")[0]}
        if "lmstudio" in out and lm:
            out["lmstudio"]["in_use"] = any(h.cfg.providers.get(u) is lm for u in used)
        servers = []
        for name, prov in h.cfg.providers.items():   # a llama.cpp (or other OpenAI-style) server on this machine
            url = str(prov.get("base_url") or "")
            if prov.get("exclusive") or prov.get("type") != "openai" or not re.match(r"https?://(127\.0\.0\.1|localhost)[:/]", url):
                continue
            props = ctxwin.server_props(url)
            row = {"name": name, "url": url, "in_use": name in used, "up": props is not None, **(props or {})}
            if props is None:
                row["up"] = bool(probe(url.rstrip("/") + "/models", 2.0).get("up"))   # up, but not llama.cpp
            if row["up"] or row["in_use"]:   # (a server in the config that nothing uses and is not running: not news)
                servers.append(row)
        out["servers"] = servers
        out["jev_adapter"] = probe(h.cfg.jev.get("url", "http://127.0.0.1:33001").rstrip("/") + "/docs", 2.0)
        if h.cfg.jev.get("backend_health"):
            out["jev_backend"] = {"name": h.cfg.jev.get("backend"), **probe(h.cfg.jev["backend_health"], 2.0)}
        if h.cfg.tools.get("comfy_url"):
            out["comfyui"] = probe(str(h.cfg.tools["comfy_url"]).rstrip("/") + "/system_stats", 2.0)
        return out

    # -- updating the harness from GitHub (jevh/update.py) --------------------------------------------
    version_cache: dict[str, Any] = {"ts": 0.0, "data": None}
    running = (updater.check(fetch=False)["current"] or {}).get("sha")   # the code this server loaded

    @app.get("/api/version")
    def version(refresh: bool = False) -> dict:
        """This checkout against its GitHub branch. Asks GitHub at most every 10 minutes unless refresh."""
        if refresh or version_cache["data"] is None or time.time() - version_cache["ts"] > 600:
            version_cache.update(ts=time.time(), data=updater.check(fetch=True))
        return {**version_cache["data"], "restart_needed": hub.updated}

    def updating(fn):
        """Run an update/undo with no turn running (and none starting); the server keeps its loaded code."""
        if not hub.busy.acquire(blocking=False):
            raise HTTPException(409, "wait for the running turn to finish")
        try:
            r = fn()
        except updater.UpdateError as e:
            raise HTTPException(409, str(e)) from None
        finally:
            hub.busy.release()
        if r["from"] != r["to"]:   # an undo back to the version this server runs needs no restart
            hub.updated = None if r["to"] == running else {
                "from": running, "to": r["to"], "undone": bool(r.get("undone")),
                "deps_changed": r["deps_changed"], "deps_note": r["deps_note"]}
        version_cache.update(ts=time.time(), data=updater.check(fetch=False))
        return {**r, "message": updater.done(r), "restart_needed": hub.updated}

    @app.post("/api/update")
    async def update(request: Request) -> dict:
        """Update to the latest version, keeping local edits; with test, the tests must pass or it's undone."""
        body = await request.json() if await request.body() else {}
        return await asyncio.to_thread(updating, lambda: updater.apply(test=bool(body.get("test"))))

    @app.post("/api/update/channel")
    async def update_channel(request: Request) -> dict:
        """Follow another channel (stable, main, ...) from now on; "default" goes back to stable."""
        name = str((await request.json()).get("channel") or "default").strip()
        was = updater.get_channel()
        updater.set_channel(name)
        st = await asyncio.to_thread(updater.check, None, True)
        if st["error"] and name != "default":
            updater.set_channel(was)
            raise HTTPException(400, st["error"])
        version_cache.update(ts=time.time(), data=st)
        return {**st, "restart_needed": hub.updated}

    @app.post("/api/update/undo")
    async def update_undo() -> dict:
        return await asyncio.to_thread(updating, updater.undo)

    @app.get("/api/artifacts")
    def artifacts() -> list[dict]:
        d = hub.harness.artifacts
        if not d.exists():
            return []
        files = sorted((p for p in d.iterdir() if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)
        return [{"path": str(p), "name": p.name, "bytes": p.stat().st_size, "mtime": p.stat().st_mtime,
                 "type": mimetypes.guess_type(p.name)[0] or "application/octet-stream"} for p in files]

    @app.get("/api/file")
    def file(path: str):
        # what the page opens: a session's files (artifacts, task folders). The rest of the workspace holds every
        # conversation (memory.db), the usage and failure logs and the Nango secret key (nango/): not for a link
        root = (hub.harness.cfg.workspace / "sessions").resolve()
        p = Path(path).resolve()
        if not p.is_file() or not p.is_relative_to(root):
            raise HTTPException(404, "not found")
        mt = mimetypes.guess_type(p.name)[0] or "text/plain"
        headers = {"X-Content-Type-Options": "nosniff"}
        if mt in ("text/html", "image/svg+xml"):
            # opaque origin: a generated page can run its own scripts but never touch the dashboard
            headers["Content-Security-Policy"] = "sandbox allow-scripts allow-forms"
        return FileResponse(p, media_type=mt, headers=headers)

    return app
