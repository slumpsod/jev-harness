"""Files sent from the dashboard's chat (issue #5), end to end with the live models: the real dashboard (python -m jevh.web
from this checkout, a free port, a throwaway workspace and home folder), driven by headless Chromium the way a person
would: the attach button, a question about the file, the answer. Once at desktop width with the receipt photo, once at
phone width (390 px, touch) with the scanned form.

Checks per run: the file uploaded (a tray item that finished), the turn shows its thumbnail, the worker's code used
look_at() on the uploaded path, and the answer holds what the picture shows. Screenshots of each stage are saved.

    python evals/bench/look_at_dashboard.py               # both widths, once each
    python evals/bench/look_at_dashboard.py --repeat 3

Needs: the worker model (with vision) and Jev your config names (designed on Qwen3.8 27B on llama-server, semif Jev),
Playwright's Chromium, Pillow and PyMuPDF. Raw: look_at_dashboard_raw.json; summary: look_at_dashboard_summary.txt;
screenshots: evals/bench/runs/look_at_dashboard/<stamp>/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import _out  # noqa: E402  (evals/_out.py: where a run's results go)
from look_at_files import FORM, RECEIPT, make  # noqa: E402

RAW = HERE / "look_at_dashboard_raw.json"
SUMMARY = HERE / "look_at_dashboard_summary.txt"
LIMIT = 8 * 60
RUNS = [
    {"name": "desktop_receipt", "viewport": {"width": 1400, "height": 900}, "mobile": False, "file": "IMG_4471.jpg",
     "text": "which shop is this receipt from and what was the total?", "expect": [r"corner crow", re.escape(RECEIPT["total"])]},
    {"name": "phone_form", "viewport": {"width": 390, "height": 844}, "mobile": True, "file": "intake_form_scan.png",
     "text": "what's the policy number on my form?", "expect": [r"HX\W?44721\W?B"]},
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start(tmp: Path) -> tuple[subprocess.Popen, str]:
    home = tmp / "home"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(f"workspace: {json.dumps(str(tmp / 'ws'))}\n", encoding="utf-8")
    port = free_port()
    env = dict(os.environ, JEVH_HOME=str(home))
    log = open(tmp / "dashboard.log", "w", encoding="utf-8")  # noqa: SIM115 - closed with the process
    proc = subprocess.Popen([sys.executable, "-u", "-m", "jevh.web", "--port", str(port), "--no-browser"], cwd=str(REPO),
                            env=env, stdout=log, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
    url = f"http://127.0.0.1:{port}/"
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2).read()
            return proc, url
        except OSError:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError(f"the dashboard did not start: {(tmp / 'dashboard.log').read_text(encoding='utf-8')[-600:]}")


def one(p, url: str, run: dict, path: str, shots: Path, rep: int) -> dict:
    browser = p.chromium.launch()
    out = {"run": run["name"], "rep": rep, "errors": []}
    try:
        ctx = browser.new_context(viewport=run["viewport"], is_mobile=run["mobile"], has_touch=run["mobile"],
                                  device_scale_factor=2 if run["mobile"] else 1)
        page = ctx.new_page()
        page.on("pageerror", lambda e: out["errors"].append(str(e)[:200]))
        page.goto(url)
        page.wait_for_selector("#attach")
        page.evaluate("api('/api/reset', {method: 'POST'})")   # a fresh session for each run
        page.wait_for_timeout(500)
        with page.expect_file_chooser() as fc:   # the attach button, as a person clicks it
            page.click("#attach")
        fc.value.set_files(path)
        page.wait_for_selector("#atray .aitem.done", timeout=30000)
        out["uploaded"] = True
        page.screenshot(path=str(shots / f"{run['name']}-{rep}-1-attached.png"))
        page.fill("#input", run["text"])
        page.keyboard.press("Enter")
        page.wait_for_selector(".turn .attach .athumb img", timeout=20000)
        out["thumbnail"] = page.evaluate("document.querySelector('.turn .athumb img').naturalWidth > 0")
        page.screenshot(path=str(shots / f"{run['name']}-{rep}-2-sent.png"))
        t0 = time.monotonic()
        page.wait_for_selector(".turn .answer", timeout=LIMIT * 1000)
        out["seconds"] = round(time.monotonic() - t0, 1)
        out["answer"] = page.inner_text(".turn .answer")
        if page.locator("#side").bounding_box() and run["mobile"]:
            page.evaluate("typeof closePanel === 'function' && closePanel()")   # (a file it made opens the panel)
        page.wait_for_timeout(500)
        page.screenshot(path=str(shots / f"{run['name']}-{rep}-3-answered.png"), full_page=False)
        out["fits"] = not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        token = page.evaluate("TOKEN")
        st = json.loads(urllib.request.urlopen(urllib.request.Request(url + "api/state", headers={"X-Jevh-Token": token})).read())
        out["message"] = st["turns"][-1]["message"]
        out["session"] = st["session"]
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")
    finally:
        browser.close()
    return out


def grade(run: dict, r: dict, ws: Path | None) -> dict:
    code = ""
    if ws and r.get("session"):
        for f in (ws / "sessions" / r["session"]).rglob("checkpoint.json"):
            code += f.read_text(encoding="utf-8", errors="replace")
    r["used_look_at"] = r.get("used_look_at", "look_at(" in code)
    checks = {"uploaded": bool(r.get("uploaded")), "thumbnail": bool(r.get("thumbnail")),
              "file_in_request": "[Attached by the user" in (r.get("message") or ""),
              "used_look_at": bool(r["used_look_at"]),
              "correct": all(re.search(x, r.get("answer") or "", re.I) for x in run["expect"]),
              "fits": bool(r.get("fits")), "no_page_errors": not r.get("errors")}
    return {"checks": checks, "pass": all(checks.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    _out.add_args(ap)
    a = ap.parse_args()
    dest = _out.Out.of(a)
    from playwright.sync_api import sync_playwright
    stamp = time.strftime("%Y%m%d-%H%M%S")
    shots = HERE / "runs" / "look_at_dashboard" / stamp
    shots.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="jevh-lookat-dash-"))
    files = make(tmp / "files")
    proc, url = start(tmp)
    results = []
    try:
        with sync_playwright() as p:
            for rep in range(a.repeat):
                for run in RUNS:
                    print(f"{run['name']} rep {rep} ...", flush=True)
                    r = one(p, url, run, files[run["file"]], shots, rep)
                    g = grade(run, r, tmp / "ws")
                    r["grade"] = g
                    results.append(r)
                    print(f"   -> {'pass' if g['pass'] else 'FAIL'} {[k for k, v in g['checks'].items() if not v] or ''} "
                          f"({r.get('seconds')}s) {str(r.get('answer'))[:200]!r}", flush=True)
    finally:
        proc.kill()
    ok = sum(r["grade"]["pass"] for r in results)
    lines = [f"look_at_dashboard: {len(results)} runs, {time.strftime('%Y-%m-%d %H:%M')}, screenshots in {shots}", ""]
    for r in results:
        g = r["grade"]
        lines.append(f"  {r['run']:16s} rep {r['rep']}  {'pass' if g['pass'] else 'FAIL'}  "
                     + ", ".join(f"{k} {'ok' if v else 'NO'}" for k, v in g["checks"].items()) + f"  ({r.get('seconds')}s)")
        lines.append(f"      answer: {str(r.get('answer'))[:300]!r}")
    lines.append(f"  total: {ok}/{len(results)}")
    text = "\n".join(lines)
    dest.write(RAW).write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    dest.write(SUMMARY).write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
