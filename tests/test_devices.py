"""Sign-in and devices (jevh/accounts.py, jevh/devices, jevh/tools/devices.py), offline.

The end-to-end tests run a real dashboard (uvicorn on a free port) and a real device agent in this process, talking over a
real WebSocket on 127.0.0.1, with a fake harness and a fake Jev. Live models are in evals/bench/devices.py.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jevh.accounts import AccountError, Accounts
from jevh.devices import wire
from jevh.devices.ops import RISKY, Call, Ops, Refused
from jevh.jev import Jev
from jevh.orchestrator import Harness
from jevh.web.server import create_app

from conftest import FakeSystemOne, ScriptedLLM, py

TOKEN = "test-token"
REMOTE = "box.tail1234.ts.net"


# ---------------------------------------------------------------------- accounts
def test_accounts_passwords_sessions_and_throttling(tmp_path):
    acc = Accounts(tmp_path / "accounts.db")
    assert not acc.has_users()
    with pytest.raises(AccountError):
        acc.add_user("jed", "short")
    assert acc.add_user("Jed", "correct horse") == "Jed"
    with pytest.raises(AccountError):
        acc.add_user("jed", "another one")              # names are one per person, any case
    assert acc.verify("jed", "correct horse") == "Jed"
    assert acc.verify("jed", "wrong password", who="1.2.3.4") is None
    assert acc.verify("nobody", "correct horse") is None
    assert "correct horse" not in (tmp_path / "accounts.db").read_bytes().decode("latin-1")
    token = acc.start_session("Jed")
    assert acc.session_user(token) == "Jed" and acc.session_user("forged") is None
    acc.set_password("Jed", "a new password")            # a new password signs every browser out
    assert acc.session_user(token) is None
    for _ in range(8):
        acc.verify("jed", "nope", who="9.9.9.9")
    assert acc.verify("jed", "a new password", who="9.9.9.9") is None   # throttled, even with the right one
    assert acc.verify("jed", "a new password", who="8.8.8.8") == "Jed"


def test_pairing_codes_work_once_and_device_keys_can_be_revoked(tmp_path):
    acc = Accounts(tmp_path / "accounts.db")
    code, expires = acc.pair_code("jed")
    assert len(code) == 9 and code[4] == "-" and expires > time.time()
    dev, key = acc.redeem(code.lower().replace("-", " "), "Laptop", "Windows 11")   # typed loosely
    assert dev.name == "Laptop" and acc.device_for_key(key).id == dev.id
    assert key not in (tmp_path / "accounts.db").read_bytes().decode("latin-1")
    with pytest.raises(AccountError):
        acc.redeem(code, "again")                          # used
    dev2, _ = acc.redeem(acc.pair_code("jed")[0], "laptop")
    assert dev2.name == "laptop-2"                         # two live devices never share a name
    with pytest.raises(AccountError):
        acc.rename(dev2.id, "LAPTOP")
    acc.revoke(dev.id)
    assert acc.device_for_key(key) is None
    assert [d.name for d in acc.devices()] == ["laptop-2"]
    with pytest.raises(AccountError):
        acc.redeem("ZZZZ-ZZZZ", "x")


# ---------------------------------------------------------------------- what a device does
@pytest.fixture
def ops(tmp_path, monkeypatch):
    monkeypatch.setenv("JEVH_DEVICE_HOME", str(tmp_path / "devhome"))
    return Ops(tmp_path / "devhome")


def test_files_are_listed_read_written_and_deleted_undoably(ops, tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    (d / "invoice march.pdf").write_bytes(b"%PDF\0binary")
    listed = ops.run("files.list", [str(d)])
    assert [e["name"] for e in listed["entries"]] == ["a.txt", "invoice march.pdf"]
    assert ops.run("files.read", [str(d / "a.txt")], {"start": 2, "end": 3}) == "2: two\n3: three"
    with pytest.raises(ValueError):
        ops.run("files.read", [str(d / "invoice march.pdf")])   # binary: fetch it instead
    hits = ops.run("files.find", ["invoice march"], {"root": str(d)})["hits"]
    assert [h["name"] for h in hits] == ["invoice march.pdf"]
    w = ops.run("files.write", [str(d / "a.txt"), "replaced"])
    assert Path(w["backup"]).read_text(encoding="utf-8") == "one\ntwo\nthree\n"   # the old one is kept
    gone = ops.run("files.delete", [str(d / "a.txt")])
    assert not (d / "a.txt").exists() and Path(gone["moved_to"]).read_text(encoding="utf-8") == "replaced"
    with pytest.raises(Refused):
        ops.run("files.delete", [str(Path.home())])
    got = ops.run("files.fetch", [str(d / "invoice march.pdf")])
    assert wire.file_bytes(got["file"]) == b"%PDF\0binary"
    put = ops.run("files.put", [str(d / "copy.pdf"), got["file"]])
    assert put["status"] == "done" and (d / "copy.pdf").read_bytes() == b"%PDF\0binary"
    assert ops.run("files.put", [str(d / "copy.pdf"), got["file"]])["status"] == "exists"


def test_risky_commands_ask_first_and_a_paused_device_refuses_everything(ops):
    for cmd in ("Remove-Item -Recurse C:/x", "rm -rf ~/x", "del /s /q C:\\x", "shutdown /s /t 0", "git push --force",
                "format D:", "Stop-Process -Name chrome"):
        assert ops.run("sh", [cmd])["status"] == "needs_confirmation", cmd
    for cmd in ("Get-ChildItem ~", "git status", "python --version", "echo model deleted"):
        assert not RISKY.search(cmd), cmd
    out = ops.run("sh", ["echo hello-device"], {"shell": "powershell" if os.name == "nt" else "sh"})
    assert out["status"] == "done" and out["code"] == 0 and "hello-device" in out["out"]
    ops.paused = True
    with pytest.raises(Refused, match="paused"):
        ops.run("files.list", ["~"])
    ops.paused = False
    with pytest.raises(Refused):
        ops.run("files.rmtree", ["~"])
    with pytest.raises(Refused):
        ops.run("browser._session", [])


def test_a_long_command_stops_when_the_call_is_cancelled(ops):
    call = Call("c1", "sh")
    cmd = "Start-Sleep -Seconds 30" if os.name == "nt" else "sleep 30"
    threading.Timer(1.0, call.cancelled.set).start()
    started = time.time()
    out = ops.run("sh", [cmd], {"shell": "powershell" if os.name == "nt" else "sh"}, call)
    assert out["status"] == "stopped" and time.time() - started < 15


def test_results_carry_the_files_a_tool_made_on_the_device(ops, tmp_path):
    ops.artifacts.mkdir(parents=True)
    shot = ops.artifacts / "page.png"
    shot.write_bytes(b"\x89PNG fake")
    packed = ops.attach({"screenshot": str(shot), "other": str(tmp_path)})
    assert wire.is_file(packed["screenshot"]) and packed["other"] == str(tmp_path)   # only what the device made
    back = wire.unpack_files(packed, tmp_path / "here")
    assert Path(back["screenshot"]).read_bytes() == b"\x89PNG fake"
    assert wire.safe_name("..\\..\\evil/../x.txt") == "x.txt" and wire.safe_name("..") == "file"


# ---------------------------------------------------------------------- sign-in on the dashboard
def rules(k, q, s):
    return None


def make_app(cfg, allow=True, replies=None):
    def factory(on_event, session=None):
        return Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: ScriptedLLM(list(replies or [])),
                       on_event=on_event)
    return create_app(cfg, port=8740, harness_factory=factory, token=TOKEN,
                      allow_hosts=[f"{REMOTE}:10000"] if allow else None, access_key="k3y")


def test_accounts_replace_the_key_link_for_other_devices(cfg):
    app = make_app(cfg)
    c = TestClient(app, base_url="http://127.0.0.1:8740")
    remote = {"Host": f"{REMOTE}:10000"}
    assert c.get("/?key=k3y", headers=remote).status_code == 200          # no accounts: the key link, as before
    r = c.post("/api/accounts", json={"name": "jed", "password": "correct horse"}, headers={"X-Jevh-Token": TOKEN})
    assert r.status_code == 200                                           # made on the hub computer itself
    r = c.get("/?key=k3y", headers=remote, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"     # now: sign in, the key no longer opens it
    assert c.get("/api/state", headers={**remote, "X-Jevh-Token": TOKEN}).status_code == 401
    assert "Sign in" in c.get("/login", headers=remote).text
    assert c.post("/auth/login", json={"name": "jed", "password": "nope"}, headers=remote).status_code == 401
    r = c.post("/auth/login", json={"name": "jed", "password": "correct horse"}, headers=remote)
    assert r.status_code == 200
    cookie = r.headers["set-cookie"].lower()
    assert "jevh_session=" in cookie and "httponly" in cookie and "secure" in cookie and "samesite=lax" in cookie
    # (a Secure cookie: the test client is plain http, while tailscale serve is https, so it is sent by hand)
    signed = {**remote, "Cookie": "jevh_session=" + r.cookies["jevh_session"]}
    page = c.get("/", headers=signed)
    assert page.status_code == 200 and TOKEN in page.text
    assert c.get("/api/me", headers={**signed, "X-Jevh-Token": TOKEN}).json()["user"] == "jed"
    # someone on another device cannot make accounts without signing in
    c2 = TestClient(app, base_url="http://127.0.0.1:8740")
    assert c2.post("/api/accounts", json={"name": "eve", "password": "12345678"},
                   headers={**remote, "X-Jevh-Token": TOKEN}).status_code == 401
    c.post("/auth/logout", headers=signed)
    assert c.get("/", headers=signed, follow_redirects=False).status_code == 303   # that sign-in is over
    # a cross-site form can't sign anyone in or pair a device
    assert c.post("/auth/login", json={"name": "jed", "password": "correct horse"},
                  headers={**remote, "Origin": "https://evil.example"}).status_code == 403


def test_the_key_link_can_make_the_first_account_from_another_device_and_only_the_first(cfg):
    """The app on another computer opens the dashboard's key link; there it may make the first account (whoever holds the
    link already has the whole assistant) and is signed in at once. A second account needs a signed-in user."""
    from jevh.devices.agent import hub_base
    c = TestClient(make_app(cfg), base_url="http://127.0.0.1:8740")
    remote = {"Host": f"{REMOTE}:10000", "X-Jevh-Token": TOKEN}
    assert c.post("/api/accounts", json={"name": "eve", "password": "12345678"}, headers=remote).status_code == 403
    keyed = {**remote, "Cookie": "jevh_key=k3y"}
    assert c.get("/api/me", headers=keyed).json()["keyed"] is True
    r = c.post("/api/accounts", json={"name": "jed", "password": "correct horse"}, headers=keyed)
    assert r.status_code == 200 and r.json()["signed_in"] is True
    session = r.cookies["jevh_session"]
    signed = {**remote, "Cookie": f"jevh_session={session}"}
    assert c.get("/api/me", headers=signed).json()["user"] == "jed"
    assert c.post("/api/accounts", json={"name": "eve", "password": "12345678"}, headers=keyed).status_code == 401
    assert c.get("/api/me", headers=keyed).status_code == 401          # the key link no longer opens it
    assert hub_base(f"https://{REMOTE}:10000/?key=k3y") == f"https://{REMOTE}:10000"
    assert hub_base(f"{REMOTE}:10000") == f"https://{REMOTE}:10000" and hub_base("http://127.0.0.1:8740/") == "http://127.0.0.1:8740"


def test_pairing_needs_a_way_in_for_other_devices(cfg):
    c = TestClient(make_app(cfg, allow=False), base_url="http://127.0.0.1:8740")
    r = c.post("/api/devices/pair-code", headers={"X-Jevh-Token": TOKEN})
    assert r.status_code == 400 and "allow-host" in r.json()["detail"]
    c = TestClient(make_app(cfg), base_url="http://127.0.0.1:8740")
    r = c.post("/api/devices/pair-code", headers={"X-Jevh-Token": TOKEN}).json()
    assert r["command"] == f"jevh device pair https://{REMOTE}:10000 {r['code']}"
    p = c.post("/auth/pair", json={"code": r["code"], "name": "bilbo", "platform": "Windows 11"},
               headers={"Host": f"{REMOTE}:10000"})
    assert p.status_code == 200 and p.json()["key"].startswith("jevd_")
    assert c.post("/auth/pair", json={"code": r["code"], "name": "x"}, headers={"Host": f"{REMOTE}:10000"}).status_code == 400
    listed = c.get("/api/devices", headers={"X-Jevh-Token": TOKEN}).json()["devices"]
    assert [(d["name"], d["online"]) for d in listed] == [("bilbo", False)]


# ---------------------------------------------------------------------- hub <-> agent, for real
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeJevHTTP(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeJevHTTP.seen.append(body)
        data = json.dumps({"answers": {k: {"type": "noul", "noul": 0.77} for k in body["questions"]}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def live(cfg, tmp_path, monkeypatch):
    """A dashboard on a free port, a fake Jev behind it, and a paired device agent connected to it."""
    import uvicorn
    jev_srv = HTTPServer(("127.0.0.1", 0), FakeJevHTTP)
    threading.Thread(target=jev_srv.serve_forever, daemon=True).start()
    cfg.jev = {**cfg.jev, "url": f"http://127.0.0.1:{jev_srv.server_address[1]}"}
    port = free_port()
    monkeypatch.setenv("JEVH_DEVICE_HOME", str(tmp_path / "devhome"))
    monkeypatch.setenv("JEVH_JEV_URL", "http://unused")   # the agent points it at its proxy; restored after
    app = create_app(cfg, port=port, token=TOKEN, allow_hosts=[f"{REMOTE}:10000"], harness_factory=lambda on_event, session=None: Harness(
        cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: ScriptedLLM([]), on_event=on_event))
    from jevh.web.__main__ import server_config
    server = uvicorn.Server(server_config(app, port))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    hub = app.state.hub
    from jevh.devices import agent as ag
    code, _ = hub.accounts.pair_code("jed")
    conf = ag.pair(f"http://127.0.0.1:{port}", code, "testbox")
    agent = ag.Agent(conf, hook_main=False)
    agent.start()
    for _ in range(100):
        if hub.devices.online() == ["testbox"]:
            break
        time.sleep(0.05)
    assert hub.devices.online() == ["testbox"], agent.error
    yield hub, agent, port
    agent.stop()
    server.should_exit = True
    jev_srv.shutdown()


def test_a_call_goes_to_the_device_and_its_answer_comes_back(live, tmp_path):
    hub, agent, _ = live
    (tmp_path / "note.txt").write_text("from the device", encoding="utf-8")
    assert hub.devices.call("TESTBOX", "files.read", [str(tmp_path / "note.txt")]) == "from the device"
    assert hub.devices.call("test", "info")["tools"]["files"] == "ok"          # (a unique start of the name works too)
    from jevh.devices.hub import DeviceError
    with pytest.raises(DeviceError, match="no device called"):
        hub.devices.call("elsewhere", "info")
    with pytest.raises(DeviceError, match="does not exist"):
        hub.devices.call("testbox", "files.read", [str(tmp_path / "missing.txt")])
    listed = hub.devices.list()
    assert listed[0]["name"] == "testbox" and listed[0]["online"] and listed[0]["tools"]["files"] == "ok"


def test_the_device_asks_back_only_what_it_may(live):
    hub, agent, _ = live
    asked = []

    def run(op, args=None, kwargs=None, call=None):   # a tool on the device that asks the user, then the hub's memory
        if op == "test.ask":
            answer = call.host("ask_user", "May I type the password?", ["Allow", "Don't allow"])
            try:
                call.host("remember", "steal this")
            except RuntimeError as e:
                return {"answer": answer, "refused": str(e)}
        return Ops.run(agent.ops, op, args, kwargs, call)
    agent.ops.run = run
    got = hub.devices.call("testbox", "test.ask", on_host=lambda name, *a, **k: asked.append((name, a)) or "Allow")
    assert got["answer"] == "Allow" and "cannot be asked from a device" in got["refused"]
    assert asked == [("ask_user", ("May I type the password?", ["Allow", "Don't allow"]))]


def test_the_devices_jev_questions_go_through_the_hub(live):
    _, agent, _ = live
    from jevh.decide import SystemOneClient, noul
    url = os.environ["JEVH_JEV_URL"]
    assert url.startswith("http://127.0.0.1:")                                 # the agent's own proxy
    r = SystemOneClient(url, "semif", timeout=20).ask({"page": "x"}, {"done": noul("Is it done?")})
    assert r["done"].noul == pytest.approx(0.77) and FakeJevHTTP.seen[-1]["state"] == {"page": "x"}
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(url + "/v1/anything-else", data=b"{}", timeout=10)


def test_pausing_the_device_refuses_calls_and_a_stop_cancels_one(live, cfg):
    hub, agent, _ = live
    from jevh.devices.hub import DeviceError
    from jevh.errors import Cancelled
    agent.set_paused(True)
    for _ in range(50):
        if hub.devices.list()[0]["paused"]:
            break
        time.sleep(0.05)
    with pytest.raises(DeviceError, match="paused"):
        hub.devices.call("testbox", "info")
    agent.set_paused(False)
    time.sleep(0.3)
    stop = threading.Event()
    threading.Timer(1.0, stop.set).start()
    cmd = "Start-Sleep -Seconds 30" if os.name == "nt" else "sleep 30"
    started = time.time()
    with pytest.raises(Cancelled):
        hub.devices.call("testbox", "sh", [cmd], {"shell": "powershell" if os.name == "nt" else "sh"}, cancel=stop)
    assert time.time() - started < 10


def test_revoking_a_device_disconnects_it_for_good(live):
    hub, agent, port = live
    import asyncio
    dev = hub.accounts.device("testbox")
    hub.accounts.revoke(dev.id)
    asyncio.run_coroutine_threadsafe(hub.devices.drop(dev.id), hub.devices.loop).result(5)
    for _ in range(100):
        if agent.state == "revoked":
            break
        time.sleep(0.05)
    assert agent.state == "revoked" and hub.devices.online() == []


def test_a_wrong_key_is_turned_away(live, tmp_path):
    hub, agent, port = live
    from jevh.devices import agent as ag
    bad = ag.Agent({"hub": f"http://127.0.0.1:{port}", "key": "jevd_forged", "name": "x"}, hook_main=False)
    t = bad.start()
    t.join(10)
    assert bad.state == "revoked" and hub.devices.online() == ["testbox"]


# ---------------------------------------------------------------------- the worker's side
class FakeDeviceHub:
    def __init__(self, names):
        self.names, self.calls, self.here = names, [], "hub-pc"

    def online(self):
        return list(self.names)

    def list(self):
        return [{"name": n, "online": True} for n in self.names]

    def call(self, ref, op, args, kwargs, timeout, cancel, on_host):
        self.calls.append((ref, op, args, kwargs))
        if op == "screenshot":
            return {"__file__": {"name": "screen.png", "b64": "iVBORw==", "bytes": 4}}
        if op == "desktop.do":
            return {"status": "done", "asked": on_host("ask_user", "Type the password?", ["Allow"])}
        return {"status": "done", "op": op}


def test_the_worker_gets_the_tool_only_while_a_device_is_online(cfg):
    seen = {}

    class Spy(ScriptedLLM):
        def chat(self, messages):
            seen.setdefault("system", messages[0]["content"])
            return super().chat(messages)

    def tools_rule(k, q, s):
        if k == "_pick" and "devices" in (q.get("criteria") or {}):
            seen["offered"] = q["criteria"]["devices"]
        return None
    code = ("pc = devices.get('bilbo')\nshot = pc.screenshot()\nr = pc.desktop.do('log in', window='App')\n"
            "final(f\"{open(shot, 'rb').read()!r} {r['asked']} {devices.list()[-1]['name']}\")")
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(tools_rule)), llm_factory=lambda p: Spy([py(code)]))
    r = h.turn("on bilbo, log in to the app", force="daily_assistant")
    assert "devices" not in seen["system"]                       # no hub: no tool, no guide, no pointer
    seen.clear()
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(tools_rule)), llm_factory=lambda p: Spy([py(code)]))
    h.devices = FakeDeviceHub(["bilbo"])
    asked = []
    h._ask_user = lambda task, persona, q, options=None, secret=False: asked.append(q) or "Allow"
    r = h.turn("on bilbo, log in to the app", force="daily_assistant")
    assert "devices" in seen["system"]
    # picked or not, the prompt names the computers and says the other tools act on the hub (live: a worker given the browser
    # guide for "on benchbox, open example.com" opened it in the hub's own browser)
    assert "Connected now: bilbo." in seen["system"] and "(the hub) only" in seen["system"]
    assert r.answer == "b'\\x89PNG' Allow hub-pc", r.answer       # the screenshot came back as a file here
    assert asked == ["[on bilbo] Type the password?"]             # the user is told which computer asks
    assert [c[1] for c in h.devices.calls] == ["screenshot", "desktop.do"]
    if "offered" in seen:
        assert "connected now: bilbo" in seen["offered"]
