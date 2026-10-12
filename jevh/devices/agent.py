"""The device agent: this computer, signed in to a hub, doing what the hub's assistant asks (jevh/devices/ops.py).

    jevh device pair https://<hub>.<tailnet>.ts.net:10000 ABCD-EFGH [--name laptop]
    jevh device run                      # stays connected (the desktop app runs this for you)
    jevh device status | pause | resume | forget

One connection, made by the device, out to the hub (wss through `tailscale serve`): nothing listens on this computer
except a proxy on 127.0.0.1 that hands the browser and desktop tools' Jev questions to the hub (Jev runs beside the hub's
models, not here). The device's key is in <device home>/device.json, readable by this user only; `forget` deletes it,
and revoking the device in the dashboard ends it on the hub's side.

Every call is written to <device home>/activity.jsonl (what was asked, never a password), and `on_activity` lets the
desktop app show it. Paused (tray icon, `jevh device pause`, or `paused` in device.json), every call is refused.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import secrets
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from . import wire
from .ops import Call, Ops, Refused, device_home, summary

CONF = "device.json"
HOST_WAIT = 30 * 60      # a question to the user can wait this long
JEV_WAIT = 300


def conf_path() -> Path:
    return device_home() / CONF


def load_conf() -> dict:
    try:
        return json.loads(conf_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_conf(conf: dict) -> None:
    p = conf_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(conf, indent=2), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, p)


def hub_base(url: str) -> str:
    url = url.strip().rstrip("/")
    if "://" not in url:
        url = "https://" + url
    return url


def ws_url(hub: str) -> str:
    base = hub_base(hub)
    return ("wss://" + base[len("https://"):] if base.startswith("https://") else "ws://" + base.split("://", 1)[1]) + wire.PATH


def pair(hub: str, code: str, name: str | None = None, timeout: float = 20) -> dict:
    """Trade a pairing code (made in the hub's dashboard) for this device's key, and keep it."""
    body = json.dumps({"code": code, "name": name or platform.node() or "device",
                       "platform": f"{platform.system()} {platform.release()}"}).encode()
    req = urllib.request.Request(hub_base(hub) + "/auth/pair", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            got = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode("utf-8")).get("detail")
        except (ValueError, OSError):
            detail = None
        raise RuntimeError(detail or f"the hub said {e.code}") from None
    conf = {**load_conf(), "hub": hub_base(hub), "key": got["key"], "id": got["device"]["id"],
            "name": got["device"]["name"], "paired": time.time(), "paused": False}
    conf.pop("revoked", None)
    save_conf(conf)
    return conf


class _JevProxy(BaseHTTPRequestHandler):
    """127.0.0.1 only: the tools' SystemOneClient posts here; the agent forwards it to the hub."""
    agent: "Agent"

    def log_message(self, *a):  # noqa: D401 - quiet
        pass

    def _forward(self, body: bytes | None) -> None:
        if self.path not in ("/v1/systemone", "/health"):
            self.send_error(404)
            return
        try:
            status, text = self.agent.jev(self.path, (body or b"").decode("utf-8", "replace"))
        except Exception as e:  # noqa: BLE001
            status, text = 502, json.dumps({"detail": f"the hub did not answer: {e}"})
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):  # noqa: N802
        self._forward(self.rfile.read(int(self.headers.get("Content-Length") or 0)))

    def do_GET(self):  # noqa: N802
        self._forward(None)


class Agent:
    def __init__(self, conf: dict | None = None, on_activity: Callable[[dict], None] | None = None,
                 ops: Ops | None = None, hook_main: bool = True):
        self.hook_main = hook_main   # route the desktop tool's ask_user through __main__._call (False in tests)
        self.conf = conf or load_conf()
        if not self.conf.get("key"):
            raise RuntimeError("this device is not paired: run `jevh device pair <hub> <code>` (the code comes from "
                               "the dashboard's Devices panel)")
        self.ops = ops or Ops()
        self.ops.paused = bool(self.conf.get("paused"))
        self.on_activity = on_activity or (lambda e: None)
        self.ws = None
        self.state = "starting"      # starting | connected | offline | revoked | stopped
        self.error = ""
        self._stop = threading.Event()
        self._send_lock = threading.Lock()
        self._calls: dict[str, Call] = {}
        self._waiting: dict[str, Future] = {}   # host and jev replies
        self._proxy: ThreadingHTTPServer | None = None
        self._conf_mtime = 0.0
        self.log = device_home() / "activity.jsonl"

    # ------------------------------------------------------------------ running
    def start(self) -> threading.Thread:
        t = threading.Thread(target=self.run, name="jevh-device", daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._stop.set()
        for c in list(self._calls.values()):
            c.cancelled.set()
        ws = self.ws
        if ws is not None:
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass
        if self._proxy:
            self._proxy.shutdown()
        self._unhook_main()
        self.ops.close()
        self.state = "stopped"

    def run(self) -> None:
        self._start_proxy()
        self._install_ask_user()
        threading.Thread(target=self._housekeeping, name="jevh-device-idle", daemon=True).start()
        delay = 2.0
        while not self._stop.is_set():
            try:
                self._session()
                delay = 2.0
            except _Revoked:
                self.state, self.error = "revoked", "the hub no longer accepts this device's key: pair it again"
                self.conf["revoked"] = time.time()
                save_conf(self.conf)
                self._activity({"kind": "revoked"})
                return
            except Exception as e:  # noqa: BLE001 - the hub is down or unreachable: try again
                self.state, self.error = "offline", f"{type(e).__name__}: {e}"[:300]
                self._activity({"kind": "offline", "error": self.error})
            if self._stop.wait(delay):
                break
            delay = min(delay * 2, 60.0)
        self.state = "stopped"

    def _session(self) -> None:
        from websockets.exceptions import InvalidStatus
        from websockets.sync.client import connect
        try:
            opened = connect(ws_url(self.conf["hub"]), additional_headers={"Authorization": f"Bearer {self.conf['key']}"},
                             max_size=wire.MAX_MESSAGE, open_timeout=20, ping_interval=20, ping_timeout=30,
                             close_timeout=5)
        except InvalidStatus as e:   # (a wrong key is a close with CLOSE_REVOKED after the handshake, not this)
            raise ConnectionError(f"the hub refused the connection (HTTP {e.response.status_code})") from None
        with opened as ws:
            self._serve_connection(ws)

    def _serve_connection(self, ws) -> None:
        from websockets.exceptions import ConnectionClosed
        self.ws = ws
        try:
            self._send({"t": "hello", "name": self.conf.get("name"), "platform": f"{platform.system()} {platform.release()}",
                        "version": wire.PROTOCOL, "paused": self.ops.paused, "tools": self.ops.tools_available(),
                        "info": {"host": platform.node(), "user": os.environ.get("USERNAME") or os.environ.get("USER")}})
            self.state, self.error = "connected", ""
            self._activity({"kind": "connected", "hub": self.conf["hub"]})
            while not self._stop.is_set():
                try:
                    raw = ws.recv(timeout=1.0)
                except TimeoutError:
                    continue
                self._handle(json.loads(raw))
        except ConnectionClosed as e:
            code = e.rcvd.code if e.rcvd else None
            if code == wire.CLOSE_REVOKED:
                raise _Revoked from None
            raise ConnectionError(f"the hub closed the connection ({code})") from None
        finally:
            self.ws = None
            for f in list(self._waiting.values()):   # a reply that can no longer come
                if not f.done():
                    f.set_exception(ConnectionError("lost the connection to the hub"))
            self._waiting.clear()

    def _send(self, msg: dict) -> None:
        ws = self.ws
        if ws is None:
            raise ConnectionError("not connected to the hub")
        text = json.dumps(msg, ensure_ascii=False, default=str)
        with self._send_lock:
            ws.send(text)

    # ------------------------------------------------------------------ messages from the hub
    def _handle(self, msg: dict) -> None:
        t = msg.get("t")
        if t == "welcome":
            env = msg.get("env") or {}
            for k in ("JEVH_JEV_BACKEND", "JEVH_TOOLS_JEV", "JEVH_CLOUD_MODEL"):
                if env.get(k):
                    os.environ[k] = str(env[k])
            name = (msg.get("device") or {}).get("name")
            if name and name != self.conf.get("name"):
                self.conf["name"] = name
                save_conf(self.conf)
        elif t == "call":
            call = Call(str(msg["id"]), str(msg.get("op")))
            call.host = lambda name, *a, _id=call.id, **k: self._host_call(_id, name, a, k)
            self._calls[call.id] = call
            threading.Thread(target=self._run_call, args=(call, msg), name=f"device-call-{call.id}", daemon=True).start()
        elif t == "cancel":
            call = self._calls.get(str(msg.get("id")))
            if call is not None:
                call.cancelled.set()
                if call.proc is not None and call.proc.poll() is None:
                    from ..procs import kill_tree
                    kill_tree(call.proc)
        elif t in ("host_reply", "jev_reply"):
            f = self._waiting.pop(str(msg.get("hid") or msg.get("rid")), None)
            if f is not None and not f.done():
                f.set_result(msg)

    def _run_call(self, call: Call, msg: dict) -> None:
        args, kwargs = msg.get("args") or [], msg.get("kwargs") or {}
        line = summary(call.op, args, kwargs)
        started = time.time()
        self._activity({"kind": "call", "op": call.op, "what": line})
        try:
            value = self.ops.run(call.op, args, kwargs, call)
            reply = {"t": "result", "id": call.id, "ok": True, "value": value}
            status = value.get("status") if isinstance(value, dict) else "done"
        except Refused as e:
            reply, status = {"t": "result", "id": call.id, "ok": False, "error": str(e), "refused": True}, "refused"
        except Exception as e:  # noqa: BLE001 - reported to the worker as an error
            reply, status = {"t": "result", "id": call.id, "ok": False, "error": f"{type(e).__name__}: {e}"}, "error"
        finally:
            self._calls.pop(call.id, None)
        self._activity({"kind": "done", "op": call.op, "what": line, "status": status,
                        "seconds": round(time.time() - started, 1)})
        try:
            self._send(reply)
        except Exception as e:  # noqa: BLE001 - too big, or the connection went: say so if we still can
            try:
                self._send({"t": "result", "id": call.id, "ok": False, "error": f"the answer could not be sent: {e}"})
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ asking the hub
    def _wait_reply(self, key: str, msg: dict, timeout: float, call: Call | None = None) -> dict:
        f: Future = Future()
        self._waiting[key] = f
        self._send(msg)
        deadline = time.time() + timeout
        while True:
            try:
                return f.result(timeout=0.5)
            except TimeoutError:   # (concurrent.futures' TimeoutError is this one since 3.11)
                pass
            if call is not None and call.cancelled.is_set():
                self._waiting.pop(key, None)
                raise RuntimeError("stopped by the user")
            if time.time() > deadline:
                self._waiting.pop(key, None)
                raise TimeoutError("the hub did not answer in time")

    def _host_call(self, call_id: str, name: str, args, kwargs):
        hid = secrets.token_hex(6)
        reply = self._wait_reply(hid, {"t": "host", "id": call_id, "hid": hid, "name": name, "args": list(args),
                                       "kwargs": kwargs}, HOST_WAIT, self._calls.get(call_id))
        if reply.get("error"):
            raise RuntimeError(reply["error"])
        return reply.get("value")

    def jev(self, path: str, body: str) -> tuple[int, str]:
        rid = secrets.token_hex(6)
        reply = self._wait_reply(rid, {"t": "jev", "rid": rid, "path": path, "body": body}, JEV_WAIT)
        return int(reply.get("status") or 502), str(reply.get("body") or "")

    # ------------------------------------------------------------------ local plumbing
    def _start_proxy(self) -> None:
        handler = type("JevProxy", (_JevProxy,), {"agent": self})
        self._proxy = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._proxy.daemon_threads = True
        threading.Thread(target=self._proxy.serve_forever, name="jevh-device-jev", daemon=True).start()
        os.environ["JEVH_JEV_URL"] = f"http://127.0.0.1:{self._proxy.server_address[1]}"

    def _install_ask_user(self) -> None:
        """The desktop tool asks the user (may it type a password?) through `__main__._call`, as it does in a worker's
        REPL: here that goes to the hub, which asks in the dashboard."""
        main = sys.modules.get("__main__")
        if main is None or not self.hook_main:
            return

        def _call(name, *args, **kwargs):
            if name == "ask_user":
                return self.ops.ask_user(*args, **kwargs)
            call = self.ops.call_now()
            if call is None:
                raise RuntimeError(f"{name}() needs a call in progress")
            return call.host(name, *args, **kwargs)
        self._main_before = {k: getattr(main, k) for k in ("_call", "ask_user") if hasattr(main, k)}
        main._call = _call
        main.ask_user = lambda q, options=None, secret=False: _call("ask_user", q, options, secret=secret)
        self._hooked = (main, _call)

    def _unhook_main(self) -> None:
        hooked = getattr(self, "_hooked", None)
        if not hooked:
            return
        main, ours = hooked
        if getattr(main, "_call", None) is ours:
            for k in ("_call", "ask_user"):
                if k in self._main_before:
                    setattr(main, k, self._main_before[k])
                elif hasattr(main, k):
                    delattr(main, k)
        self._hooked = None

    def _housekeeping(self) -> None:
        while not self._stop.wait(2.0):
            self._reread_conf()
            try:
                if self.ops.close_idle():
                    self._activity({"kind": "browser_closed", "why": "idle"})
            except Exception:  # noqa: BLE001
                pass

    def _reread_conf(self) -> None:
        """`jevh device pause` from another terminal edits device.json: follow it."""
        try:
            m = conf_path().stat().st_mtime
        except OSError:
            return
        if m == self._conf_mtime:
            return
        self._conf_mtime = m
        paused = bool(load_conf().get("paused"))
        if paused != self.ops.paused:
            self.set_paused(paused, save=False)

    def set_paused(self, paused: bool, save: bool = True) -> None:
        self.ops.paused = paused
        if paused:   # what is running stops where it can (a shell command is killed)
            for c in list(self._calls.values()):
                c.cancelled.set()
                if c.proc is not None and c.proc.poll() is None:
                    from ..procs import kill_tree
                    kill_tree(c.proc)
        if save:
            self.conf["paused"] = paused
            save_conf(self.conf)
            try:
                self._conf_mtime = conf_path().stat().st_mtime
            except OSError:
                pass
        try:
            self._send({"t": "state", "paused": paused})
        except Exception:  # noqa: BLE001 - offline: the hello says it when we are back
            pass
        self._activity({"kind": "paused" if paused else "resumed"})

    def _activity(self, event: dict) -> None:
        event = {"ts": time.time(), **event}
        try:
            self.log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except OSError:
            pass
        try:
            self.on_activity(event)
        except Exception:  # noqa: BLE001
            pass


class _Revoked(Exception):
    pass


def recent_activity(n: int = 200) -> list[dict]:
    try:
        lines = (device_home() / "activity.jsonl").read_text(encoding="utf-8").splitlines()[-n:]
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


# ---------------------------------------------------------------------- jevh device ...

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="jevh device", description="Let a hub's assistant use this computer.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pair", help="pair this computer with a hub, using a code from its dashboard (Devices panel)")
    p.add_argument("hub", help="the hub's address, e.g. https://desktop.tailnet.ts.net:10000")
    p.add_argument("code", help="the pairing code, e.g. ABCD-EFGH")
    p.add_argument("--name", help="what the assistant calls this computer (default: its host name)")
    sub.add_parser("run", help="connect and do what the assistant asks, until Ctrl+C")
    sub.add_parser("status", help="what this device is paired with, and whether it is paused")
    sub.add_parser("pause", help="refuse every call until resumed (also stops what is running)")
    sub.add_parser("resume", help="accept calls again")
    sub.add_parser("forget", help="delete this device's key (also revoke it in the dashboard)")
    sub.add_parser("activity", help="the last calls the assistant made here")
    a = ap.parse_args(argv)
    conf = load_conf()
    if a.cmd == "pair":
        conf = pair(a.hub, a.code, a.name)
        print(f"Paired as {conf['name']} with {conf['hub']}. Run `jevh device run` (or `jevh app`) to connect.")
    elif a.cmd == "status":
        if not conf.get("key"):
            print("Not paired.")
        else:
            print(f"{conf.get('name')} -> {conf.get('hub')}  "
                  f"{'PAUSED' if conf.get('paused') else 'active'}{'  (revoked by the hub)' if conf.get('revoked') else ''}")
            print(f"tools here: {json.dumps(Ops.tools_available())}")
    elif a.cmd in ("pause", "resume"):
        if not conf.get("key"):
            sys.exit("Not paired.")
        conf["paused"] = a.cmd == "pause"
        save_conf(conf)
        print("Paused: the assistant cannot use this computer." if conf["paused"] else "Resumed.")
    elif a.cmd == "forget":
        try:
            conf_path().unlink()
        except FileNotFoundError:
            pass
        print("This device's key is deleted. Revoke it in the dashboard too (Devices panel).")
    elif a.cmd == "activity":
        for e in recent_activity(40):
            if e.get("kind") == "done":
                print(time.strftime("%m-%d %H:%M:%S", time.localtime(e["ts"])), e.get("status", ""), e.get("what", ""))
    elif a.cmd == "run":
        agent = Agent(conf, on_activity=lambda e: print(json.dumps(e, default=str)[:300], flush=True))
        print(f"{conf.get('name')}: connecting to {conf.get('hub')} (Ctrl+C to stop)", flush=True)
        try:
            agent.run()
        except KeyboardInterrupt:
            agent.stop()
        if agent.state == "revoked":
            sys.exit(agent.error)


if __name__ == "__main__":
    main()
