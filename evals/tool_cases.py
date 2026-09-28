"""Live eval of Jev's tool pick: which REPL tools' guides go into the worker's prompt.

Each case lists the tools the request needs and the ones that are fine either way. A case is
right when every needed tool is picked and nothing outside needed+either is. A missing tool
costs capability; a needless one costs context (and is what we're trying to avoid).

    python evals/tool_cases.py            # live, both phrasings, sweep
    python evals/tool_cases.py --offline  # re-sweep the cached raw answers
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevh.config import load_config  # noqa: E402
from jevh.decide import SystemOneClient, noul  # noqa: E402
from jevh.tools import TOOLS  # noqa: E402

RAW = Path(__file__).with_name("tool_cases_raw.json")
W, B, P, C = "web", "browser", "project", "comfy"

CASES = [  # (request, needed, either)
    ("What's the weather in Seattle this weekend?", {W}, {B}),
    ("Who won the NBA finals this year?", {W}, {B}),
    ("What's 17% of 2,340?", set(), set()),
    ("Write a haiku about coffee.", set(), set()),
    ("Find the latest release notes for Playwright.", {W}, {B}),
    ("Log into my Hydro One account and download my latest bill.", {B}, {W}),
    ("Fill in the contact form on example-shop.com asking about bulk pricing.", {B}, {W}),
    ("Check that the signup flow on my site at localhost:3000 works.", {B}, {P}),
    ("Book a table at Luigi's for Friday 7pm on their website.", {B}, {W}),
    ("Add a dark mode toggle to my React app.", {P}, {B}),
    ("Why are the tests failing in my repo?", {P}, set()),
    ("Rename the function parse_cfg to load_config across the codebase.", {P}, set()),
    ("Summarize what this project does.", {P}, set()),
    ("Generate a hero image of a sunrise over mountains.", {C}, set()),
    ("Make a 5 second video of waves crashing on rocks.", {C}, set()),
    ("Upscale this photo to 4K.", {C}, set()),
    ("Compare the price of the RTX 5090 at three retailers.", {W}, {B}),
    ("What does HTTP status 418 mean?", set(), {W}),
    ("Plan my workouts for this week.", set(), set()),
    ("Draft an email to my client about the delayed quote.", set(), set()),
    ("Look up the ComfyUI docs for the KSampler node and fix my workflow.", {W, C}, {B}),
    ("Research best practices for FastAPI auth and apply them to my project.", {W, P}, {B}),
    ("Build a landing page for my bakery and check it renders correctly in a browser.", {B}, {P}),
    ("Scrape the product titles from https://books.toscrape.com", {W}, {B}),
    ("Order more printer ink from Amazon.", {B}, {W}),
    ("Sign me up for the newsletter on that site.", {B}, set()),
    ("Read the README in my project and explain the setup steps.", {P}, set()),
    ("Run the linter on my code and fix the warnings.", {P}, set()),
    ("What's the current population of Canada?", {W}, set()),
    ("Translate 'good morning' into Spanish.", set(), set()),
    ("Tell me a joke.", set(), set()),
    ("How do I center a div in CSS?", set(), {W}),
    ("Create an image of a cat astronaut for my blog post.", {C}, set()),
    ("Check the status of my GitHub Actions run on the website.", {B}, {W}),
    ("What's the price of Bitcoin right now?", {W}, {B}),
    ("Write unit tests for utils.py in my project.", {P}, set()),
    ("Is ComfyUI running right now?", {C}, set()),
    ("Remind me to stretch every hour.", set(), set()),
    ("Go to en.wikipedia.org and tell me today's featured article.", {W}, {B}),
    ("Add a CSV export button to the dashboard in my project and test it in the browser.", {P, B}, set()),
    ("Take a screenshot of example.com.", {B}, set()),
    ("What are good macros for cutting at 80kg bodyweight?", set(), {W}),
]

PHRASINGS = {
    "need": "Would the assistant need this tool to handle the latest request?",
    "help": "Would this tool help the assistant handle the latest request?",
}


def run_live() -> dict:
    cfg = load_config()
    client = SystemOneClient(cfg.jev["url"], cfg.jev["backend"], timeout=300)
    raw: dict = {}
    for name, q in PHRASINGS.items():
        t = time.time()
        rows = []
        for req, _, _ in CASES:
            r = client.ask({"latest_request": req}, {f"t_{n}": noul({"question": q, "tool": TOOLS[n].description})
                                                     for n in (W, B, P, C)})
            rows.append({n: r[f"t_{n}"].noul for n in (W, B, P, C)})
        raw[name] = rows
        print(f"{name}: {(time.time() - t) / len(CASES):.2f}s per request")
    RAW.write_text(json.dumps(raw, indent=1), encoding="utf-8")
    return raw


def score(rows: list[dict], t: float) -> tuple[int, int, int]:
    ok = missing = needless = 0
    for (req, need, either), s in zip(CASES, rows):
        picked = {n for n, p in s.items() if p >= t}
        missing += len(need - picked)
        needless += len(picked - need - either)
        ok += need <= picked <= need | either
    return ok, missing, needless


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    raw = json.loads(RAW.read_text(encoding="utf-8")) if args.offline else run_live()
    for name, rows in raw.items():
        print(f"\n== {name}: {PHRASINGS[name]}")
        for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            ok, miss, extra = score(rows, t)
            print(f"  t={t:.1f}  {ok}/{len(CASES)} right  missing {miss}  needless {extra}")
    best = max(((n, t) for n in raw for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)),
               key=lambda nt: (score(raw[nt[0]], nt[1])[0], -score(raw[nt[0]], nt[1])[2]))
    n, t = best
    print(f"\nbest: {n} at {t}. Misses:")
    for (req, need, either), s in zip(CASES, raw[n]):
        picked = {k for k, p in s.items() if p >= t}
        if not need <= picked <= need | either:
            print(f"  {req[:70]:70} need={sorted(need)} got={sorted(picked)} "
                  + " ".join(f"{k}={v:.2f}" for k, v in s.items()))


if __name__ == "__main__":
    main()
