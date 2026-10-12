"""The hub's side of devices: who is connected, and a worker's call made on one of them.

Runs inside the dashboard (jevh/web/server.py): each device holds one WebSocket open to `/api/devices/ws`, served on the
dashboard's event loop by `serve()`. A worker's call (`call()`) comes from a harness thread: it is sent on the loop and
waited for on that thread, with the turn's Stop and a time limit.

While a call runs, the device's tools may ask back (wire.py `host`): only HOST_NAMES are passed to the harness (the worker's
model looking at a screenshot, a question to the user). A device cannot hand out tasks, read memory or call MCP tools.
A device's Jev questions (`jev`) are forwarded to the hub's own Jev adapter, on JEV_PATHS only.
"""
from __future__ import annotations

import asyncio
import json
import secrets
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable

from ..accounts import Accounts, Device
from . import wire

HOST_NAMES = ("look", "ask_user")
JEV_PATHS = ("/v1/systemone", "/health")
DEFAULT_TIMEOUT = 600.0


class DeviceError(RuntimeError):
    """The call could not be made, or the device said no: told to the worker in words."""


@dataclass
class Link:
    device: Device
    ws: Any
    platform: str = ""
    tools: dict = field(default_factory=dict)
    paused: bool = False
    since: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    calls: int = 0


@dataclass
class _Pending:
    future: Future
    link: Link
    on_host: Callable[..., Any] | None


class DeviceHub:
    def __init__(self, accounts: Accounts, jev_url: Callable[[], str], env: Callable[[], dict] | None = None,
                 publish: Callable[[str, dict], None] | None = None):
        self.accounts = accounts
        self.jev_url = jev_url
        self.env = env or (lambda: {})
        self.publish = publish or (lambda kind, data: None)
        self.loop: asyncio.AbstractEventLoop | None = None
        self.here = ""                         # the hub's own computer name: its files are the worker's own
        self._links: dict[str, Link] = {}      # device id -> its live connection
        self._pending: dict[str, _Pending] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ connections (event loop)
    async def serve(self, ws, device: Device) -> None:
        """One device's connection, from its hello until it goes."""
        link = Link(device, ws)
        old = self._links.get(device.id)
        self._links[device.id] = link
        if old is not None:   # the same device again (it lost its network and came back): the old socket is dead
            try:
                await old.ws.close(code=wire.CLOSE_REPLACED)
            except Exception:  # noqa: BLE001
                pass
        self.accounts.seen(device.id)
        try:
            await self._send(link, {"t": "welcome", "device": {"id": device.id, "name": device.name}, "env": self.env()})
            while True:
                text = await ws.receive_text()
                try:
                    msg = json.loads(text)
                except ValueError:
                    continue
                await self._on_message(link, msg)
        finally:
            if self._links.get(device.id) is link:
                del self._links[device.id]
                self.accounts.seen(device.id)
            with self._lock:
                gone = [k for k, p in self._pending.items() if p.link is link]
                for k in gone:
                    p = self._pending.pop(k)
                    if not p.future.done():
                        p.future.set_exception(DeviceError(f"{device.name} went offline during the call"))
            self.publish("devices", {"name": device.name, "online": self._links.get(device.id) is not None})

    async def _send(self, link: Link, msg: dict) -> None:
        text = json.dumps(msg, ensure_ascii=False, default=str)
        async with link.lock:
            await link.ws.send_text(text)

    async def _on_message(self, link: Link, msg: dict) -> None:
        t = msg.get("t")
        if t == "hello":
            link.platform = str(msg.get("platform") or "")[:60]
            link.tools = msg.get("tools") if isinstance(msg.get("tools"), dict) else {}
            link.paused = bool(msg.get("paused"))
            self.publish("devices", {"name": link.device.name, "online": True, "paused": link.paused})
        elif t == "state":
            link.paused = bool(msg.get("paused"))
            self.publish("devices", {"name": link.device.name, "online": True, "paused": link.paused})
        elif t == "result":
            with self._lock:
                p = self._pending.pop(str(msg.get("id")), None)
            if p is not None and not p.future.done():
                p.future.set_result(msg)
        elif t == "host":
            threading.Thread(target=self._answer_host, args=(link, msg), name="device-host", daemon=True).start()
        elif t == "jev":
            threading.Thread(target=self._answer_jev, args=(link, msg), name="device-jev", daemon=True).start()

    def _reply(self, link: Link, msg: dict) -> None:
        """(any thread) send to a device on the event loop."""
        if self.loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._send(link, msg), self.loop).result(30)
        except Exception:  # noqa: BLE001 - it went offline: its call fails on its own
            pass

    def _answer_host(self, link: Link, msg: dict) -> None:
        with self._lock:
            p = self._pending.get(str(msg.get("id")))
        name = str(msg.get("name"))
        value, error = None, None
        if p is None or p.link is not link or p.on_host is None:
            error = "no call of this device is waiting for that"
        elif name not in HOST_NAMES:
            error = f"{name}() cannot be asked from a device"
        else:
            try:
                value = p.on_host(name, *(msg.get("args") or []), **(msg.get("kwargs") or {}))
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
        self._reply(link, {"t": "host_reply", "hid": msg.get("hid"), "value": value, "error": error})

    def _answer_jev(self, link: Link, msg: dict) -> None:
        path = str(msg.get("path") or "/v1/systemone")
        if path not in JEV_PATHS:
            self._reply(link, {"t": "jev_reply", "rid": msg.get("rid"), "status": 404, "body": "{}"})
            return
        url = self.jev_url().rstrip("/") + path
        body = msg.get("body")
        req = urllib.request.Request(url, data=body.encode("utf-8") if body else None,
                                     method="POST" if body else "GET", headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=290) as r:
                status, text = r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            status, text = e.code, e.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            status, text = 502, json.dumps({"detail": f"Jev is unreachable from the hub: {e}"})
        self._reply(link, {"t": "jev_reply", "rid": msg.get("rid"), "status": status, "body": text})

    async def drop(self, device_id: str) -> None:
        """Close a device's connection (it was revoked)."""
        link = self._links.pop(device_id, None)
        if link is not None:
            try:
                await link.ws.close(code=wire.CLOSE_REVOKED)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ for workers (any thread)
    def online(self) -> list[str]:
        return [l.device.name for l in self._links.values()]

    def list(self) -> list[dict]:
        live = {l.device.id: l for l in self._links.values()}
        out = []
        for d in self.accounts.devices():
            l = live.get(d.id)
            out.append({"id": d.id, "name": d.name, "owner": d.owner, "platform": (l.platform if l else "") or d.platform,
                        "online": l is not None, "paused": bool(l and l.paused), "tools": l.tools if l else {},
                        "connected_since": l.since if l else None, "last_seen": d.seen, "busy": l.calls if l else 0,
                        "paired": d.created})
        return out

    def find(self, ref: str) -> Link:
        """A connected device by name (any case), id, or the start of one name."""
        ref = str(ref or "").strip()
        links = list(self._links.values())
        for test in (lambda l: l.device.id == ref, lambda l: l.device.name.lower() == ref.lower(),
                     lambda l: l.device.name.lower().startswith(ref.lower())):
            hits = [l for l in links if test(l)]
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                raise DeviceError(f"{ref!r} could be {', '.join(l.device.name for l in hits)}: use the full name")
        known = self.accounts.device(ref)
        if known is not None:
            ago = f"last seen {time.strftime('%Y-%m-%d %H:%M', time.localtime(known.seen))}" if known.seen else "never connected"
            raise DeviceError(f"{known.name} is offline ({ago}): the device app or `jevh device run` must be running on it")
        if self.here and ref.lower() == self.here.lower():
            raise DeviceError(f"{ref} is the hub itself: use the ordinary tools (project, browser, desktop, start_job) "
                              f"for this computer")
        names = ", ".join(sorted(self.online())) or "none"
        raise DeviceError(f"no device called {ref!r}; online: {names}")

    def call(self, ref: str, op: str, args: list | None = None, kwargs: dict | None = None,
             timeout: float = DEFAULT_TIMEOUT, cancel: threading.Event | None = None,
             on_host: Callable[..., Any] | None = None) -> Any:
        if self.loop is None:
            raise DeviceError("devices need the dashboard (jevh dashboard / jevh app): it holds their connections")
        link = self.find(ref)
        if link.paused:
            raise DeviceError(f"{link.device.name} is paused by its user: ask them to resume it (tray icon) and try again")
        cid = secrets.token_hex(8)
        fut: Future = Future()
        with self._lock:
            self._pending[cid] = _Pending(fut, link, on_host)
        link.calls += 1
        try:
            asyncio.run_coroutine_threadsafe(self._send(link, {"t": "call", "id": cid, "op": op, "args": args or [],
                                                               "kwargs": kwargs or {}, "timeout": timeout}),
                                             self.loop).result(60)
            deadline = time.time() + timeout
            while True:
                try:
                    msg = fut.result(timeout=0.2)
                    break
                except TimeoutError:
                    pass
                if cancel is not None and cancel.is_set():
                    self._reply(link, {"t": "cancel", "id": cid})
                    from ..errors import Cancelled
                    raise Cancelled("stopped by the user")
                if time.time() > deadline:
                    self._reply(link, {"t": "cancel", "id": cid})
                    raise DeviceError(f"{link.device.name} did not finish {op} within {timeout:.0f}s")
        finally:
            link.calls -= 1
            with self._lock:
                self._pending.pop(cid, None)
        if not msg.get("ok"):
            raise DeviceError(f"{link.device.name}: {msg.get('error') or 'the call failed'}")
        return msg.get("value")
