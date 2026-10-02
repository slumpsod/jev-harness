"""Local dashboard: chat + live view of Jev's decisions, hand-offs, swaps and artifacts.

    python -m jevh.web [--port 8740] [--model provider:model] [--no-browser]

Security (the agent runs code, so the browser must not become a remote control):
- binds 127.0.0.1 only, and rejects any Host header but 127.0.0.1/localhost:<port>
  (blocks DNS-rebinding),
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

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from .. import update as updater
from ..bus import jsonable, task_json
from ..config import Config, load_config
from ..lmstudio import gate_for
from ..orchestrator import Harness
from ..review import ReviewError, list_sessions
from ..skills import load_roster

STATIC = Path(__file__).with_name("static")
MAX_EVENTS = 5000


# ---------------------------------------------------------------- app state

class Hub:
    """Owns the Harness, runs one turn at a time on a thread, fans events out to SSE clients."""

    def __init__(self, cfg: Config, harness_factory=None, model_override: str | None = None):
        self.cfg = cfg
        self.model_override = model_override
        self._factory = harness_factory or (lambda on_event: Harness(cfg, on_event=on_event,
                                                                     model_override=model_override))
        self.harness = self._factory(self.publish)
        self.harness.browser_dock = True   # a watched browser shows in the Browser panel, not a window
        self.dock = BrowserDock(lambda: Path(self.harness.session_dir) / "browsers")
        self.events: list[dict] = []
        self.turns: list[dict] = []
        self.swaps: list[dict] = []
        self.seq = 0
        self.busy = threading.Lock()
        self._lock = threading.Lock()
        self._subs: list[asyncio.Queue] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self.pin: str | None = None
        self.updated: dict | None = None   # set once the code on disk was updated: a restart loads it

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
                self.loop.call_soon_threadsafe(q.put_nowait, evt)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def start_turn(self, message: str, resume: str | None = None) -> dict:
        if not self.busy.acquire(blocking=False):
            raise HTTPException(409, "a turn is already running")
        turn = {"id": len(self.turns) + 1, "message": message, "status": "running", "started": time.time(),
                "pin": self.pin, "tier_mode": self.harness.tier_mode}
        self.turns.append(turn)
        self.publish("turn_start", {"turn": dict(turn)})

        def run() -> None:
            try:
                r = self.harness.resume(resume) if resume else self.harness.turn(message, force=self.pin)
                turn.update(usage=r.usage, checkpoint=r.checkpoint)
                turn.update(status="cancelled" if r.cancelled else "done", persona=r.persona, answer=r.answer, artifacts=r.artifacts,
                            seconds=r.seconds, trace=jsonable(r.trace), tasks=[task_json(t) for t in r.tasks])
                self.publish("turn_done", {"turn": dict(turn)})
            except Exception as e:  # noqa: BLE001 - report, never kill the server
                turn.update(status="error", answer=f"{type(e).__name__}: {e}", seconds=time.time() - turn["started"])
                self.publish("turn_error", {"turn": dict(turn)})
            finally:
                self.busy.release()

        threading.Thread(target=run, name=f"turn-{turn['id']}", daemon=True).start()
        return turn

    def reset(self) -> None:
        if self.busy.locked():
            raise HTTPException(409, "wait for the running turn to finish")
        mode, project, watch = self.harness.tier_mode, self.harness.project, self.harness.watch_browser
        self.harness = self._factory(self.publish)
        self.harness.tier_mode, self.harness.project, self.harness.watch_browser = mode, project, watch
        self.harness.browser_dock = True
        self.turns.clear()
        self.publish("reset", {"session": self.harness.session})


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
        try:
            while True:
                fn, fut = self._jobs.get()
                try:
                    info = self._newest(dead)
                    if info and (browser is None or at != info["key"] or not browser.is_connected()):
                        browser = None
                        try:
                            browser = pw.chromium.connect_over_cdp(f"http://127.0.0.1:{info['port']}", timeout=3000)
                            at = info["key"]
                        except Exception:  # noqa: BLE001
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
            vp = page.viewport_size or {"width": 1280, "height": 900}
            x = min(max(float(ev.get("x", 0)), 0.0), 1.0) * vp["width"]
            y = min(max(float(ev.get("y", 0)), 0.0), 1.0) * vp["height"]
            if kind == "click":
                page.mouse.click(x, y)
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
               model_override: str | None = None, token: str | None = None) -> FastAPI:
    cfg = cfg or load_config()
    hub = Hub(cfg, harness_factory, model_override)
    token = token or secrets.token_urlsafe(24)
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    allowed_origins = {f"http://{h}" for h in allowed_hosts}
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        hub.loop = asyncio.get_running_loop()  # worker threads hand events to SSE via this loop
        yield

    app = FastAPI(title="jev-harness dashboard", docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=lifespan)
    app.state.hub, app.state.token = hub, token

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.headers.get("host", "") not in allowed_hosts:
            return JSONResponse({"detail": "bad host"}, status_code=403)
        if request.url.path.startswith("/api/"):
            sent = request.headers.get("x-jevh-token") or request.query_params.get("token")
            if not sent or not secrets.compare_digest(sent, token):
                return JSONResponse({"detail": "missing or bad token"}, status_code=403)
            origin = request.headers.get("origin")
            if request.method != "GET" and origin is not None and origin not in allowed_origins:
                return JSONResponse({"detail": "foreign origin"}, status_code=403)
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8").replace("__JEVH_TOKEN__", token)

    @app.get("/api/state")
    def state() -> dict:
        h = hub.harness
        return {"session": h.session, "tier_mode": h.tier_mode, "pin": hub.pin, "busy": hub.busy.locked(),
                "fast_model": (h.cfg.models.get("fast") or {}).get("model"), "model_override": hub.model_override,
                "personas": {n: {"description": p.description, "model": p.model.get("model"),
                                 "fast_ok": p.fast_ok} for n, p in h.personas.items()},
                "skills": len(h.roster), "turns": hub.turns, "last_seq": hub.seq, "project": h.project,
                "watch_browser": h.watch_browser, "questions": h.pending_questions(),
                "thresholds": h.cfg.thresholds}

    @app.post("/api/chat")
    async def chat(request: Request) -> dict:
        body = await request.json()
        message = str(body.get("message", "")).strip()
        if not message:
            raise HTTPException(400, "empty message")
        return {"turn": hub.start_turn(message)}

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
                "learned": learned, "lessons": lessons}

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
            shutil.rmtree(path.parent)
        else:
            raise HTTPException(400, "action must be approve|delete")
        h.roster = load_roster(h.cfg.skills_dirs, Path(h.skills_home) / "_lessons")
        return {"ok": True}

    @app.post("/api/memory/forget")
    async def forget(request: Request) -> dict:
        body = await request.json()
        hub.harness.memory.retire_fact(int(body.get("id", -1)))
        return {"ok": True}

    @app.post("/api/reset")
    def reset() -> dict:
        hub.reset()
        return {"session": hub.harness.session}

    @app.get("/api/events")
    async def events(request: Request, since: int = 0):
        q = hub.subscribe()
        backlog = [e for e in hub.events if e["seq"] > since]

        async def stream():
            try:
                for e in backlog:
                    yield f"id: {e['seq']}\ndata: {json.dumps(e)}\n\n"
                last = backlog[-1]["seq"] if backlog else since
                while not await request.is_disconnected():
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
        out: dict[str, Any] = {"gpus": gpus(), "swaps": hub.swaps[-20:]}
        lm = next((p for p in h.cfg.providers.values() if p.get("exclusive")), None)
        if lm:
            try:
                out["lmstudio"] = {"up": True, "loaded": [k for k, _ in gate_for(lm["base_url"])._loaded()],
                                   "gate": {"current": gate_for(lm["base_url"]).current,
                                            "inflight": gate_for(lm["base_url"]).inflight}}
            except Exception as e:  # noqa: BLE001
                out["lmstudio"] = {"up": False, "error": str(e)[:200]}
        out["jev_adapter"] = probe(h.cfg.jev.get("url", "http://127.0.0.1:33001").rstrip("/") + "/docs", 2.0)
        if h.cfg.jev.get("backend_health"):
            out["jev_backend"] = {"name": h.cfg.jev.get("backend"), **probe(h.cfg.jev["backend_health"], 2.0)}
        if h.cfg.tools.get("comfy_url"):
            out["comfyui"] = probe(str(h.cfg.tools["comfy_url"]).rstrip("/") + "/system_stats", 2.0)
        return out

    # -- updating the harness from GitHub (jevh/update.py) --------------------------------------------
    version_cache: dict[str, Any] = {"ts": 0.0, "data": None}

    @app.get("/api/version")
    def version(refresh: bool = False) -> dict:
        """This checkout against its GitHub branch. Asks GitHub at most every 10 minutes unless refresh."""
        if refresh or version_cache["data"] is None or time.time() - version_cache["ts"] > 600:
            version_cache.update(ts=time.time(), data=updater.check(fetch=True))
        return {**version_cache["data"], "restart_needed": hub.updated}

    @app.post("/api/update")
    def update() -> dict:
        """Fast-forward to the latest version. The running server keeps its loaded code until restarted."""
        if not hub.busy.acquire(blocking=False):
            raise HTTPException(409, "wait for the running turn to finish")
        try:
            r = updater.apply()
        except updater.UpdateError as e:
            raise HTTPException(409, str(e)) from None
        finally:
            hub.busy.release()
        if r["updated"]:
            hub.updated = {"from": r["from"], "to": r["to"], "commits": r["commits"], "deps_changed": r["deps_changed"]}
        version_cache.update(ts=time.time(), data=updater.check(fetch=False))
        return {**r, "message": updater.done(r), "restart_needed": hub.updated}

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
        root = hub.harness.cfg.workspace.resolve()
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
