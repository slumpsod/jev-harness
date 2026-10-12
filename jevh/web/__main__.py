"""jevh dashboard [--port 8740] [--model provider:model] [--no-browser] [--allow-host NAME]   (or python -m jevh.web)"""
from __future__ import annotations

import argparse
import threading
import webbrowser

import uvicorn

from ..config import load_config
from ..devices import wire
from .server import create_app


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="jevh dashboard")
    ap.add_argument("--port", type=int, default=8740)
    ap.add_argument("--config")
    ap.add_argument("--model", help="put every persona on one model for testing, e.g. lmstudio:ornith-1.5-35b-a3b")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--allow-host", action="append", default=[], metavar="NAME[:PORT]",
                    help="also answer to this host name, e.g. the https name `tailscale serve` gives this "
                         "machine; still bound to 127.0.0.1, and only a signed-in browser (or, with no accounts yet, "
                         "one holding the printed access key) gets in")
    a = ap.parse_args(argv)
    app = create_app(load_config(a.config), port=a.port, model_override=a.model, allow_hosts=a.allow_host)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"jev-harness dashboard: {url}  (local only; Ctrl+C to stop)")
    signin = app.state.hub.accounts.has_users()
    for h in a.allow_host:
        if signin:
            print(f"  also from https://{h}/   (sign in; other computers pair in the Devices panel)")
        else:
            print(f"  also from https://{h}/?key={app.state.access_key}   (keep this link private; it changes each launch)")
    if not a.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    serve(app, a.port)


def server_config(app, port: int) -> uvicorn.Config:
    """How the dashboard is served (also by `jevh app`). Devices send files inside one WebSocket message: uvicorn's 16 MB
    default is less than wire.MAX_FILE in base64. websockets-sansio: the default implementation uses websockets' legacy
    server, deprecated since websockets 14."""
    return uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", ws="websockets-sansio",
                          ws_max_size=wire.MAX_MESSAGE)


def serve(app, port: int) -> None:
    uvicorn.Server(server_config(app, port)).run()


if __name__ == "__main__":
    main()
