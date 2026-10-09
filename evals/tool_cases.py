"""Live eval of Jev's tool pick: which REPL tools' guides go into the worker's prompt.

Each case lists the tools the request needs and the ones that are fine either way. A case is
right when every needed tool is picked and nothing outside needed+either is. A missing tool
costs capability; a needless one costs context (and is what we're trying to avoid).

    python evals/tool_cases.py            # live, both phrasings, sweep
    python evals/tool_cases.py --offline  # re-sweep the cached raw answers
    python evals/tool_cases.py --holdout  # the desktop tool's held-out requests (written after its description was fixed)
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
RAW_HOLDOUT = Path(__file__).with_name("tool_cases_holdout_raw.json")
W, B, P, C, N, D = "web", "browser", "project", "comfy", "nango", "desktop"
ALL = (W, B, P, C, N, D)

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
    ("Draft an email to my client about the delayed quote.", set(), {N}),
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
    ("Check the status of my GitHub Actions run on the website.", {B}, {W, N}),
    ("What's the price of Bitcoin right now?", {W}, {B}),
    ("Write unit tests for utils.py in my project.", {P}, set()),
    ("Is ComfyUI running right now?", {C}, set()),
    ("Remind me to stretch every hour.", set(), {N}),
    ("Go to en.wikipedia.org and tell me today's featured article.", {W}, {B}),
    ("Add a CSV export button to the dashboard in my project and test it in the browser.", {P, B}, set()),
    ("Take a screenshot of example.com.", {B}, set()),
    ("What are good macros for cutting at 80kg bodyweight?", set(), {W}),
    # nango: the user's own connected accounts (added 2026-09-29)
    ("Any unread emails from my accountant this week?", {N}, set()),
    ("What's on my calendar tomorrow?", {N}, set()),
    ("Send Sam an email saying I'll be 10 minutes late.", {N}, set()),
    ("Open a GitHub issue in my repo about the login bug.", {N}, {P, B}),
    ("Post in the #general Slack channel that the deploy is done.", {N}, set()),
    ("Find the Q3 budget doc in my Google Drive and summarize it.", {N}, set()),
    ("Add a meeting with Alex on Friday at 2pm to my calendar.", {N}, set()),
    ("List my open pull requests on GitHub.", {N}, {B, W}),
    ("Move the Acme deal to closed-won in HubSpot.", {N}, {B}),
    ("What did Maria say in my last Slack DM?", {N}, set()),
    ("How do I use the Gmail API to send an email from Python?", set(), {W}),
    ("What is OAuth and how does token refresh work?", set(), {W}),
    # setting up and connecting accounts (added 2026-10-02)
    ("Set up Nango so you can use my accounts.", {N}, {W}),
    ("Walk me through connecting my Google Calendar to the assistant.", {N}, {W, B}),
    ("My Gmail connection expired, reconnect it.", {N}, {B}),
    ("I'd like you to be able to read my Todoist tasks. How do I connect it?", {N}, {W, B}),
    ("Link my GitHub account so you can open issues for me.", {N}, {W, B}),
    # the desktop tool: apps on the user's own computer (added 2026-10-08; "send /rc in JevLab" scored 0.18 on the first wording)
    ("Open Notepad and write down my shopping list.", {D}, set()),
    ("Start a new session in the Claude app.", {D}, set()),
    ("Which apps do I have open on my computer right now?", {D}, set()),
    ("Press Ctrl+N in the Claude app.", {D}, set()),
    ("Type hello into the Telegram app and send it to the family group.", {D}, set()),
    ("There's a pop-up dialog on my screen, close it.", {D}, set()),
    ("In JevLab's Composer tab, send the message /rc.", {D}, set()),
    ("Use the Windows Calculator to work out 17 times 23.", {D}, set()),
    ("Open Settings and tell me which Windows version I have.", {D}, set()),
    ("Is Spotify open? If so, pause the music.", {D}, set()),
    ("Open File Explorer to my Downloads folder and tell me which file is the biggest.", {D}, {P}),
    ("In the installer window that is open, click Next until the license page.", {D}, set()),
    ("Open https://news.ycombinator.com and tell me the top story.", {B}, {W}),
    ("Add a 12-pack of paper towels to my Walmart cart.", {B}, {W}),
    ("Click the Buy button on the shop page and stop before paying.", {B}, set()),
    ("Rename the class Parser to Lexer everywhere in my project.", {P}, set()),
]
# written after the desktop description was fixed; run once with --holdout
HOLDOUT = [
    ("Switch to the Brave window and tell me which tabs are open.", {D}, set()),
    ("Rename draft.txt on my Desktop to final.txt using File Explorer.", {D}, {P}),
    ("In the Docker Desktop window, which containers are running?", {D}, set()),
    ("Change the volume to 30% in the Windows sound settings.", {D}, set()),
    ("Hit Enter in the PowerShell window that is open.", {D}, set()),
    ("Dismiss the update notification in the Claude app.", {D}, set()),
    ("In Telegram, open the Saved Messages chat.", {D}, set()),
    ("Maximize the LM Studio window.", {D}, set()),
    ("What does the dialog box on my screen say?", {D}, set()),
    ("Click through the setup wizard in the window that is open, but stop before it installs anything.", {D}, set()),
    ("Find hotels in Denver for Friday on booking.com.", {B}, {W}),
    ("What's the weather going to be like this weekend?", {W}, {B}),
    ("Rename the function foo to bar in my project.", {P}, set()),
    ("Summarize my unread emails.", {N}, set()),
    ("Explain what a TCP handshake is.", set(), {W}),
    ("Fix the failing test in the repo.", {P}, set()),
    ("Scrape the first page of quotes.toscrape.com.", {W}, {B}),
    ("Write a short poem about autumn.", set(), set()),
]

PHRASINGS = {
    "need": "Would the assistant need this tool to handle the latest request?",
    "help": "Would this tool help the assistant handle the latest request?",
}


def run_live(cases=None, out=None) -> dict:
    cases, out = cases or CASES, out or RAW
    cfg = load_config()
    client = SystemOneClient(cfg.jev["url"], cfg.jev["backend"], timeout=300)
    raw: dict = {}
    for name, q in PHRASINGS.items():
        t = time.time()
        rows = []
        for req, _, _ in cases:
            r = client.ask({"latest_request": req}, {f"t_{n}": noul({"question": q, "tool": TOOLS[n].description})
                                                     for n in ALL})
            rows.append({n: r[f"t_{n}"].noul for n in ALL})
        raw[name] = rows
        print(f"{name}: {(time.time() - t) / len(cases):.2f}s per request")
    out.write_text(json.dumps(raw, indent=1), encoding="utf-8")
    return raw


def score(rows: list[dict], t: float, cases=None) -> tuple[int, int, int]:
    ok = missing = needless = 0
    for (req, need, either), s in zip(cases or CASES, rows):
        picked = {n for n, p in s.items() if p >= t}
        missing += len(need - picked)
        needless += len(picked - need - either)
        ok += need <= picked <= need | either
    return ok, missing, needless


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    args = ap.parse_args()
    cases, path = (HOLDOUT, RAW_HOLDOUT) if args.holdout else (CASES, RAW)
    raw = json.loads(path.read_text(encoding="utf-8")) if args.offline else run_live(cases, path)
    for name, rows in raw.items():
        print(f"\n== {name}: {PHRASINGS[name]}")
        for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            ok, miss, extra = score(rows, t, cases)
            print(f"  t={t:.1f}  {ok}/{len(cases)} right  missing {miss}  needless {extra}")
    best = max(((n, t) for n in raw for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)),
               key=lambda nt: (score(raw[nt[0]], nt[1], cases)[0], -score(raw[nt[0]], nt[1], cases)[2]))
    n, t = best
    print(f"\nbest: {n} at {t}. Misses:")
    for (req, need, either), s in zip(cases, raw[n]):
        picked = {k for k, p in s.items() if p >= t}
        if not need <= picked <= need | either:
            print(f"  {req[:70]:70} need={sorted(need)} got={sorted(picked)} "
                  + " ".join(f"{k}={v:.2f}" for k, v in s.items()))


if __name__ == "__main__":
    main()
