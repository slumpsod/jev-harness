"""python -m jevh.web [--port 8740] [--model provider:model] [--no-browser]"""
from __future__ import annotations

import argparse
import threading
import webbrowser

import uvicorn

from ..config import load_config
from .server import create_app


def main() -> None:
    ap = argparse.ArgumentParser(prog="jevh.web")
    ap.add_argument("--port", type=int, default=8740)
    ap.add_argument("--config")
    ap.add_argument("--model", help="put every persona on one model for testing, e.g. lmstudio:ornith-1.5-35b-a3b")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    app = create_app(load_config(a.config), port=a.port, model_override=a.model)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"jev-harness dashboard: {url}  (local only; Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
