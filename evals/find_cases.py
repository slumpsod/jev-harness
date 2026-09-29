"""Live eval: project.find - does it point at where a question is answered?

The corpus is this repo (the harness's own code). Each question describes a piece of code in
words, the way a worker asks, without its name; the answer is a file plus the line a regex finds
there. A hit: the right file, within 25 lines of that line. Compared:
    grep  - the keyword search alone (its own ranking), no Jev
    files - the old find: Jev ranks whole files by path + first line (file hit only)
    find  - keyword search, then Jev ranks the matching snippets

    python evals/find_cases.py            # tuning questions
    python evals/find_cases.py --holdout
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["JEVH_PROJECT"] = str(ROOT)

from jevh.config import load_config  # noqa: E402
from jevh.tools import project  # noqa: E402

_all_files = project._files
# this file holds every question word for word: searching it would find the questions, not the code
project._files = lambda glob="**/*": (f for f in _all_files(glob) if f != "evals/find_cases.py")

CASES = [  # (question, file, regex for the answer's line)
    ("where does the browser tool decide whether clicking a button can't be undone?", "jevh/tools/browser.py",
     r"def _check_button"),
    ("where is the REPL's state saved to disk so a resumed task gets its variables back?", "jevh/repl/server.py",
     r"^def snapshot"),
    ("how are the project rules files parsed into individual rules?", "jevh/tools/rules.py", r"^def parse"),
    ("how does the gate work out which command runs the project's tests?", "jevh/verify.py",
     r"def detect_test_command"),
    ("where are past conversations searched by their words?", "jevh/memory.py", r"def search"),
    ("where is it decided that a message is a follow-up to the previous one?", "jevh/jev.py", r"def _follows"),
    ("where does LM Studio unload one model before loading another?", "jevh/lmstudio.py", r"def _swap_to"),
    ("where are numbers in an answer checked against what the tools returned?", "jevh/verify.py",
     r"def grounding"),
    ("which endpoint serves a worker's live thinking to the dashboard?", "jevh/web/server.py", r"/api/thinking"),
    ("where are XML-style tool calls written as text turned into python?", "jevh/worker.py",
     r"def translate_tool_calls"),
    ("how long does ask_user wait for the user to answer?", "jevh/orchestrator.py", r"def _ask_user"),
    ("where is a ComfyUI workflow submitted and its images collected?", "jevh/tools/comfy.py", r"^def run"),
    ("where are persona yaml files loaded?", "jevh/personas.py", r"def load_personas"),
    ("where are the default thresholds for Jev's decisions set?", "jevh/config.py", r"DEFAULT_THRESHOLDS ="),
    ("how is the price of a paid model call calculated?", "jevh/usage.py", r"def cost"),
    ("where does a worker notice it is repeating itself?", "jevh/worker.py", r"def _stalled"),
]
HOLDOUT = [
    ("where are skills chosen for a persona?", "jevh/jev.py", r"def pick_skills"),
    ("where does the web tool search the internet?", "jevh/tools/web.py", r"^def search"),
    ("how is a lasting fact about the user stored?", "jevh/learn.py", r"def remember"),
    ("where is the diff of an edit built for the rule check?", "jevh/tools/rules.py", r"def change_text"),
    ("where is an old step turned into the full text kept after compaction?", "jevh/worker.py", r"def _full"),
    ("where is the page's state fingerprinted to spot that a click changed nothing?", "jevh/tools/browser.py",
     r"def _signature"),
    ("where is the config file read?", "jevh/config.py", r"def load_config"),
    ("where are REPL variables brought back after a restart?", "jevh/repl/server.py", r"^def restore"),
]
# Written after HOLDOUT had been looked at (it steered the two-step design), so checked once, last.
HOLDOUT2 = [
    ("where is a skill's markdown file turned into a skill the router can pick?", "jevh/skills.py",
     r"def parse_skill"),
    ("where does a PDF download get turned into plain text?", "jevh/tools/web.py", r"def _pdf_text"),
    ("where is the browser told to fill in form fields from the values it was given?", "jevh/tools/browser.py",
     r"def _autofill"),
    ("where does the web tool download a page?", "jevh/tools/web.py", r"^def fetch"),
    ("where is an old fact replaced when the user tells us something new?", "jevh/learn.py", r"def _store_facts"),
    ("how is a user's message turned into a full-text search query?", "jevh/memory.py", r"def fts_query"),
    ("where does the harness write a new skill after a successful task?", "jevh/learn.py", r"def _create"),
    ("where is the list of skills loaded from the skill folders?", "jevh/skills.py", r"def load_roster"),
    ("where does the browser pick which option to choose in a dropdown?", "jevh/tools/browser.py",
     r"def _pick_option"),
    ("where is a worker's work log condensed for the answer check?", "jevh/learn.py", r"def work_log"),
]


def target_line(file: str, rx: str) -> int:
    for n, line in enumerate((ROOT / file).read_text(encoding="utf-8").splitlines(), 1):
        if re.search(rx, line):
            return n
    raise ValueError(f"{rx} not in {file}")


def hit(result: str, file: str, line: int, file_only: bool = False) -> bool:
    m = re.match(r"^(.+?):(\d+):", result)
    path, n = (m.group(1), int(m.group(2))) if m else (result, None)
    return path == file and (file_only or (n is not None and abs(n - line) <= 25))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--holdout2", action="store_true")
    a = ap.parse_args()
    cases = HOLDOUT2 if a.holdout2 else HOLDOUT if a.holdout else CASES
    cfg = load_config()
    os.environ.setdefault("JEVH_JEV_URL", cfg.jev["url"])
    os.environ.setdefault("JEVH_JEV_BACKEND", cfg.jev["backend"])
    totals = {m: [0, 0, 0.0] for m in ("grep", "lines", "files", "find")}   # hit@1, hit@3, seconds
    for q, file, rx in cases:
        line = target_line(file, rx)
        row = []
        for method in totals:
            t = time.time()
            if method == "grep":
                res = [f"{r}:{n}:" for r, n, _, _ in project._candidates(q)[:3]]
            elif method == "lines":   # the one-step version: Jev over 48 keyword lines from anywhere
                from jevh.tools._pick import choose, client_from_env
                c = project._candidates(q)
                jev, cap = client_from_env()
                probs, _ = choose(jev, {"question": q}, project.FIND_Q,
                                  {f"s{i}": f"{r}:{n}: {s}" for i, (r, n, _, s) in enumerate(c)}, cap)
                res = [f"{c[int(k[1:])][0]}:{c[int(k[1:])][1]}:" for k in sorted(probs, key=probs.get, reverse=True)[:3]]
            elif method == "files":
                res = project.find_files(q, 3)
            else:
                res = project.find(q, 3)
            dt = time.time() - t
            h1 = bool(res) and hit(res[0], file, line, method == "files")
            h3 = any(hit(r, file, line, method == "files") for r in res[:3])
            totals[method][0] += h1
            totals[method][1] += h3
            totals[method][2] += dt
            row.append(f"{method} {'1' if h1 else ('3' if h3 else '-')}")
        print(f"  {'  '.join(row)}   {q[:70]}")
    n = len(cases)
    print()
    for method, (h1, h3, secs) in totals.items():
        print(f"  {method:5}: hit@1 {h1}/{n}, hit@3 {h3}/{n}, {secs / n:.1f} s per question"
              + ("   (file only)" if method == "files" else ""))


if __name__ == "__main__":
    main()
