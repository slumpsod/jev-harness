"""The desktop app: the dashboard in its own window, a tray icon, and (on a device) the device agent.

    jevh app                 # first run asks: is this computer the hub, or does it connect to one?
    jevh app --hidden        # start in the tray (what the "Start with Windows" entry runs)
    jevh app install         # a Start menu entry (Windows)
    jevh app reset           # forget the choice made on the first run

- The hub (the computer with the models): the app runs the dashboard (jevh/web) in this process, unless one already
  answers on the port, and shows it. Other devices reach it through `tailscale serve`; the first run can set that up.
- A device: the window shows the hub's dashboard (sign in there). Its Devices panel has a "This computer" card, backed by
  `AppApi` below: pair this computer, pause it, stop letting the assistant use it. The agent (jevh/devices/agent.py) runs
  in this process while the app does; the tray icon says what the assistant is doing here and holds Pause.

Closing the window keeps the app in the tray; Quit (tray menu) ends it and the agent. One app runs per user: starting it
again shows the window that is already open. Its settings are in <device home>/app.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from .devices import agent as device_agent
from .devices.ops import device_home

APP_CONF = "app.json"
SINGLE_PORT = 47431          # 127.0.0.1 only: a second launch asks the first to show its window, then exits
NOTIFY_EVERY = 60.0          # at most one "the assistant is using this computer" notice a minute


# ---------------------------------------------------------------------- settings
def conf_path() -> Path:
    return device_home() / APP_CONF


def load_app_conf() -> dict:
    try:
        return json.loads(conf_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_app_conf(conf: dict) -> None:
    conf_path().parent.mkdir(parents=True, exist_ok=True)
    conf_path().write_text(json.dumps(conf, indent=2), encoding="utf-8")


def tailscale_name() -> str | None:
    """This computer's name on the tailnet (desktop.tailnet.ts.net), or None without Tailscale."""
    exe = shutil.which("tailscale")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "status", "--json"], capture_output=True, text=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        return (json.loads(out).get("Self") or {}).get("DNSName", "").rstrip(".") or None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def tailscale_serve(port: int, https_port: int) -> str:
    """Point https://<this computer>:<https_port> at the dashboard, for this tailnet only (never `funnel`)."""
    exe = shutil.which("tailscale")
    if not exe:
        raise RuntimeError("Tailscale is not installed")
    r = subprocess.run([exe, "serve", "--bg", f"--https={https_port}", f"http://127.0.0.1:{port}"], capture_output=True,
                       text=True, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:300] or "tailscale serve failed")
    return f"{tailscale_name()}:{https_port}"


def dashboard_up(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/login", timeout=2) as r:
            return r.status in (200, 303)
    except urllib.error.HTTPError as e:
        return e.code in (303, 403)
    except OSError:
        return False


# ---------------------------------------------------------------------- one app per user
def claim_single(on_show) -> bool:
    """True: this is the only app (a later launch's "show" calls on_show). False: one runs already and was asked to show."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        srv.bind(("127.0.0.1", SINGLE_PORT))
    except OSError:
        srv.close()
        try:
            with socket.create_connection(("127.0.0.1", SINGLE_PORT), timeout=2) as c:
                c.sendall(b"show\n")
        except OSError:
            pass
        return False
    srv.listen(2)

    def serve():
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                return
            with c:
                if c.recv(16).startswith(b"show"):
                    on_show()
    threading.Thread(target=serve, name="jevh-app-single", daemon=True).start()
    return True


# ---------------------------------------------------------------------- the page's bridge to this computer
class AppApi:
    """window.pywebview.api: the first run's choice, then (on a device) this computer's agent for the dashboard's "This
    computer" card. Only the hub's own pages may use the agent: a page the window wandered to (a link the dashboard
    showed) gets nothing."""

    def __init__(self, app: "App"):
        self._app = app

    def _allowed(self) -> bool:
        w = self._app.window
        url = (w.get_current_url() if w else "") or ""
        return bool(self._app.home_url) and url.startswith(self._app.home_url.rstrip("/"))

    def status(self) -> dict:
        if not self._allowed():
            return {"error": "not available on this page"}
        conf = device_agent.load_conf()
        a = self._app.agent
        same_hub = conf.get("hub") and conf.get("hub").rstrip("/") == (self._app.conf.get("hub") or "").rstrip("/")
        return {"paired": bool(conf.get("key")) and bool(same_hub) and not conf.get("revoked"), "name": conf.get("name"),
                "id": conf.get("id"), "hub": conf.get("hub"), "paused": bool(conf.get("paused")),
                "state": a.state if a else "not running", "last_error": a.error if a else ""}

    def pair(self, code: str) -> dict:
        if not self._allowed():
            return {"error": "not available on this page"}
        try:
            device_agent.pair(self._app.conf["hub"], code)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}
        self._app.start_agent(restart=True)
        return self.status()

    def pause(self, paused: bool) -> dict:
        if not self._allowed():
            return {"error": "not available on this page"}
        self._app.set_paused(bool(paused))
        return self.status()

    def forget(self) -> dict:
        if not self._allowed():
            return {"error": "not available on this page"}
        self._app.stop_agent()
        try:
            device_agent.conf_path().unlink()
        except FileNotFoundError:
            pass
        self._app.refresh_tray()
        return {"ok": True}


    # -- the first run's page (only until a role is chosen)
    def suggest(self) -> dict:
        if self._app.conf.get("role"):
            return {"error": "already set up"}
        return {"tailscale": tailscale_name(), "has_models": (Path(__file__).resolve().parents[1] / "config.local.yaml").is_file()}

    def choose(self, role: str, hub: str = "", serve: bool = False, port: int = 8740) -> dict:
        if self._app.conf.get("role"):
            return {"error": "already set up (jevh app reset asks again)"}
        if role not in ("hub", "device"):
            return {"error": "choose one"}
        conf = {"role": role, "port": int(port)}
        try:
            if role == "device":
                if not hub.strip():
                    return {"error": "Type the hub's address."}
                conf["hub"] = device_agent.hub_base(hub)
                if "key=" in hub:   # the dashboard's key link: opened once, so the first account can be made here
                    conf["first_url"] = hub.strip()
            elif serve:
                conf["allow_hosts"] = [tailscale_serve(int(port), 10000)]
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}
        save_app_conf(conf)
        threading.Thread(target=self._app.begin, args=(conf,), daemon=True).start()
        return {"ok": True}


SETUP_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
:root{color-scheme:light dark;--bg:#f4f5f8;--card:#fff;--text:#16181d;--muted:#646a76;--line:#dde0e6;--accent:#4b5bd6}
@media (prefers-color-scheme:dark){:root{--bg:#111318;--card:#1a1d24;--text:#e8eaef;--muted:#9aa0ad;--line:#2b2f39;--accent:#8c98ff}}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,"Segoe UI",sans-serif;display:grid;place-items:center;min-height:100vh}
main{max-width:520px;width:100%;padding:16px}h1{font-size:22px;margin:0 0 6px}p{color:var(--muted);margin:0 0 18px}
label.opt{display:block;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-bottom:10px;cursor:pointer}
label.opt b{display:block}label.opt span{color:var(--muted);font-size:14px}input[type=text]{width:100%;box-sizing:border-box;font:inherit;padding:8px 10px;
border-radius:8px;border:1px solid var(--line);background:transparent;color:inherit;margin-top:8px}button{font:inherit;font-weight:600;padding:9px 20px;border:0;
border-radius:9px;background:var(--accent);color:#fff;cursor:pointer;margin-top:8px}#msg{color:#d33;margin-top:10px;min-height:1.4em}small{color:var(--muted)}
</style></head><body><main><h1>Set up jev-harness</h1><p>Your assistant runs on one computer (the hub, with the models). Your other computers connect to it.</p>
<label class="opt"><input type="radio" name="role" value="hub"> <b>This computer is the hub</b><span>The assistant and its models run here.</span>
<div><label><input type="checkbox" id="serve"> Let my other devices reach it through Tailscale (<span id="ts">checking...</span>)</label></div></label>
<label class="opt"><input type="radio" name="role" value="device"> <b>Connect to my hub</b><span>Sign in to the assistant on another computer, and let it use this one.</span>
<input type="text" id="hub" placeholder="https://my-desktop.tailnet.ts.net:10000 (or the link the dashboard printed)"></label>
<button id="go">Continue</button><div id="msg"></div></main><script>
window.addEventListener("pywebviewready", async () => { const s = await pywebview.api.suggest();
  document.getElementById("ts").textContent = s.tailscale ? "https://" + s.tailscale + ":10000" : "Tailscale not found";
  document.getElementById("serve").disabled = !s.tailscale;
  document.querySelector(`input[value=${s.has_models ? "hub" : "device"}]`).checked = true; });
document.getElementById("go").onclick = async () => { const role = (document.querySelector("input[name=role]:checked") || {}).value;
  if (!role) return; const r = await pywebview.api.choose(role, document.getElementById("hub").value, document.getElementById("serve").checked, 8740);
  document.getElementById("msg").textContent = r.error || "Starting..."; };
</script></body></html>"""


# ---------------------------------------------------------------------- the app
class App:
    def __init__(self, conf: dict, hidden: bool = False):
        self.conf = conf
        self.hidden = hidden
        self.window = None
        self.icon = None
        self.agent: device_agent.Agent | None = None
        self.server = None
        self.home_url = ""
        self._last_notice = 0.0
        self._quitting = False

    # -- what the window shows
    def begin(self, conf: dict) -> None:
        """(any thread) Start what the chosen role needs and point the window at it."""
        self.conf = conf
        if conf.get("role") == "hub":
            port = int(conf.get("port") or 8740)
            if not dashboard_up(port):
                self._serve(port, conf.get("allow_hosts") or [])
            self.home_url = f"http://127.0.0.1:{port}/"
        else:
            self.home_url = device_agent.hub_base(conf["hub"]) + "/"
            self.start_agent()
        first = conf.pop("first_url", None)
        if first:   # (once: the key in it changes with every launch of the hub)
            save_app_conf(conf)
        if self.window is not None:
            self.window.load_url(first or self.home_url)
        self.refresh_tray()

    def _serve(self, port: int, allow_hosts: list[str]) -> None:
        import uvicorn

        from .config import load_config
        from .web.__main__ import server_config
        from .web.server import create_app
        app = create_app(load_config(), port=port, allow_hosts=allow_hosts)
        self.server = uvicorn.Server(server_config(app, port))
        threading.Thread(target=self.server.run, name="jevh-dashboard", daemon=True).start()
        for _ in range(100):
            if self.server.started:
                return
            time.sleep(0.1)

    # -- the agent (a device only)
    def start_agent(self, restart: bool = False) -> None:
        conf = device_agent.load_conf()
        if not conf.get("key") or conf.get("revoked"):
            return
        if conf.get("hub", "").rstrip("/") != (self.conf.get("hub") or "").rstrip("/"):
            return   # paired with another hub than the one this app shows
        if self.agent is not None:
            if not restart:
                return
            self.stop_agent()
        self.agent = device_agent.Agent(conf, on_activity=self._on_activity)
        self.agent.start()

    def stop_agent(self) -> None:
        if self.agent is not None:
            self.agent.stop()
            self.agent = None

    def set_paused(self, paused: bool) -> None:
        if self.agent is not None:
            self.agent.set_paused(paused)
        else:
            conf = device_agent.load_conf()
            if conf.get("key"):
                conf["paused"] = paused
                device_agent.save_conf(conf)
        self.refresh_tray()

    def _on_activity(self, e: dict) -> None:
        kind = e.get("kind")
        if kind == "call" and self.icon is not None and time.time() - self._last_notice > NOTIFY_EVERY:
            self._last_notice = time.time()
            try:
                self.icon.notify(f"{e.get('what', '')[:180]}\n(Pause it from this icon.)", "The assistant is using this computer")
            except Exception:  # noqa: BLE001
                pass
        if kind in ("connected", "offline", "revoked", "paused", "resumed", "call", "done"):
            self.refresh_tray()

    # -- the tray
    def _image(self, paused: bool = False, busy: bool = False):
        from PIL import Image, ImageDraw
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        color = (150, 150, 160) if paused else (225, 140, 40) if busy else (75, 91, 214)
        d.ellipse((4, 4, 60, 60), fill=color)
        d.text((24, 16), "J", fill=(255, 255, 255))
        if paused:
            d.rectangle((22, 22, 29, 42), fill=(255, 255, 255))
            d.rectangle((35, 22, 42, 42), fill=(255, 255, 255))
        return img

    def _state_line(self) -> str:
        if self.conf.get("role") == "hub":
            return "jev-harness: this computer is the hub"
        a, conf = self.agent, device_agent.load_conf()
        if not conf.get("key"):
            return "jev-harness: this computer is not paired"
        return f"jev-harness: {conf.get('name')} - {'paused' if conf.get('paused') else a.state if a else 'not running'}"

    def refresh_tray(self) -> None:
        if self.icon is None:
            return
        conf = device_agent.load_conf()
        busy = bool(self.agent and self.agent._calls)
        try:
            self.icon.icon = self._image(bool(conf.get("paused")), busy)
            self.icon.title = self._state_line()[:120]
            self.icon.update_menu()
        except Exception:  # noqa: BLE001
            pass

    def _menu(self):
        import pystray
        paired = lambda _i=None: bool(device_agent.load_conf().get("key")) and self.conf.get("role") == "device"  # noqa: E731
        paused = lambda _i=None: bool(device_agent.load_conf().get("paused"))  # noqa: E731
        return pystray.Menu(
            pystray.MenuItem("Open", lambda: self.show(), default=True),
            pystray.MenuItem(lambda _i: self._state_line(), None, enabled=False),
            pystray.MenuItem("Pause the assistant here", lambda: self.set_paused(not paused()), checked=paused,
                             visible=paired),
            pystray.MenuItem("Activity", lambda: self.show_activity(), visible=paired),
            pystray.MenuItem("Start with Windows", lambda: self.toggle_startup(), checked=lambda _i: startup_file().exists(),
                             visible=os.name == "nt"),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", lambda: self.quit()))

    def show(self) -> None:
        if self.window is not None:
            self.window.show()
            self.window.restore()

    def show_activity(self) -> None:
        import html

        import webview
        rows = []
        for e in reversed(device_agent.recent_activity(300)):
            if e.get("kind") in ("done", "connected", "offline", "paused", "resumed", "revoked"):
                when = time.strftime("%m-%d %H:%M:%S", time.localtime(e.get("ts", 0)))
                what = e.get("what") or e.get("hub") or e.get("error") or ""
                rows.append(f"<tr><td>{when}</td><td>{html.escape(e.get('status') or e['kind'])}</td>"
                            f"<td>{html.escape(str(what))}</td></tr>")
        page = ("<!doctype html><meta charset=utf-8><style>body{font:13px system-ui;margin:12px;color-scheme:light dark}"
                "td{padding:3px 8px;border-bottom:1px solid #8883;vertical-align:top}td:nth-child(3){font-family:monospace}</style>"
                "<h3>What the assistant did on this computer</h3><table>" + ("".join(rows) or "<tr><td>Nothing yet.</td></tr>")
                + f"</table><p>Kept in {html.escape(str(device_home() / 'activity.jsonl'))}</p>")
        webview.create_window("Activity - jev-harness", html=page, width=860, height=560)

    def toggle_startup(self) -> None:
        f = startup_file()
        if f.exists():
            f.unlink()
        else:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(f'@start "" "{pythonw()}" -m jevh app --hidden\r\n', encoding="utf-8")

    def quit(self) -> None:
        self._quitting = True
        self.stop_agent()
        if self.server is not None:
            self.server.should_exit = True
        if self.icon is not None:
            self.icon.stop()
        if self.window is not None:
            self.window.destroy()

    # -- running
    def run(self) -> None:
        import pystray
        import webview
        if not claim_single(self.show):
            print("jev-harness is already running: its window was brought up.")
            return
        first = not self.conf.get("role")
        self.window = webview.create_window("jev-harness", html=SETUP_HTML if first else "<p style='font:14px system-ui'>Starting...</p>",
                                            js_api=AppApi(self), width=1280, height=860, min_size=(420, 560),
                                            hidden=self.hidden and not first)

        def closing():
            if self._quitting:
                return True
            self.window.hide()   # the app stays in the tray; Quit ends it
            return False
        self.window.events.closing += closing
        self.icon = pystray.Icon("jev-harness", self._image(), "jev-harness", menu=self._menu())
        self.icon.run_detached()
        if not first:
            threading.Thread(target=self.begin, args=(self.conf,), daemon=True).start()
        webview.start(private_mode=False, storage_path=str(device_home() / "webview"))
        self.quit()


def startup_file() -> Path:
    return Path(os.environ.get("APPDATA", str(Path.home()))) / "Microsoft/Windows/Start Menu/Programs/Startup/jev-harness.cmd"


def pythonw() -> str:
    w = Path(sys.executable).with_name("pythonw.exe")
    return str(w if w.exists() else sys.executable)


def install_shortcut() -> Path:
    """A Start menu entry that opens the app (Windows)."""
    lnk = Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs/jev-harness.lnk"
    ps = ("$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:JEVH_LNK);"
          "$s.TargetPath=$env:JEVH_EXE;$s.Arguments='-m jevh app';$s.WorkingDirectory=$env:USERPROFILE;"
          "$s.Description='jev-harness';$s.Save()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True,
                   env={**os.environ, "JEVH_LNK": str(lnk), "JEVH_EXE": pythonw()},
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return lnk


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="jevh app", description="jev-harness as a desktop app (dashboard window + tray icon).")
    ap.add_argument("action", nargs="?", choices=["run", "install", "reset"], default="run")
    ap.add_argument("--hidden", action="store_true", help="start in the tray, without showing the window")
    ap.add_argument("--hub", help="connect to this hub (skips the first-run question)")
    ap.add_argument("--serve", action="store_true", help="this computer is the hub (skips the first-run question)")
    ap.add_argument("--port", type=int, default=None)
    a = ap.parse_args(argv)
    if a.action == "install":
        if os.name != "nt":
            sys.exit("install makes a Windows Start menu entry; elsewhere run `jevh app` from a launcher of your own")
        print(f"Start menu entry: {install_shortcut()}")
        return
    if a.action == "reset":
        try:
            conf_path().unlink()
        except FileNotFoundError:
            pass
        print("The app will ask again whether this computer is the hub. (A pairing is kept: `jevh device forget` removes it.)")
        return
    conf = load_app_conf()
    if a.hub:
        conf = {**conf, "role": "device", "hub": device_agent.hub_base(a.hub)}
        if "key=" in a.hub:
            conf["first_url"] = a.hub.strip()
    elif a.serve:
        conf = {**conf, "role": "hub"}
    if a.port:
        conf["port"] = a.port
    if conf.get("role"):
        save_app_conf(conf)
    App(conf, hidden=a.hidden).run()


if __name__ == "__main__":
    main()
