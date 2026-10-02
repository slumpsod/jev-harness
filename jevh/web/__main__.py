"""jevh dashboard [--port 8740] [--model provider:model] [--no-browser] [--allow-host NAME]   (or python -m jevh.web)"""
from __future__ import annotations

import argparse
import threading
import webbrowser

import uvicorn

from ..config import load_config
from .server import create_app


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="jevh dashboard")
    ap.add_argument("--port", type=int, default=8740)
    ap.add_argument("--config")
    ap.add_argument("--model", help="put every persona on one model for testing, e.g. lmstudio:ornith-1.5-35b-a3b")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--allow-host", action="append", default=[], metavar="NAME[:PORT]",
                    help="also answer to this host name, e.g. the https name `tailscale serve` gives this "
                         "machine; still bound to 127.0.0.1, and only a browser holding the printed access key gets in")
    a = ap.parse_args(argv)
    app = create_app(load_config(a.config), port=a.port, model_override=a.model, allow_hosts=a.allow_host)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"jev-harness dashboard: {url}  (local only; Ctrl+C to stop)")
    for h in a.allow_host:
        print(f"  also from https://{h}/?key={app.state.access_key}   (keep this link private; it changes each launch)")
    if not a.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
