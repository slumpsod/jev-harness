"""Live eval of Jev's model-tier decision: does this task go to the fast model (ornith)
or the persona's own strong model? Compares question phrasings, then sweeps thresholds.

Labels: "fast" (quick answers, lookups, small edits, short scripts, single tool runs),
"strong" (deep reasoning, hard debugging, substantial multi-part builds, long careful
writing/planning), "either" (reasonable both ways; not scored). Sending a strong task to
the fast model costs quality, so it is penalized more than the reverse (which only costs speed).

    python evals/tier_cases.py
    python evals/tier_cases.py --offline
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevh.config import load_config  # noqa: E402
from jevh.decide import SystemOneClient, choice, noul, score  # noqa: E402
from jevh.personas import load_personas  # noqa: E402
import _out  # noqa: E402  (evals/_out.py: where a run's results go)

D, DEV, FE, FIT, COMFY = "daily_assistant", "high_end_dev", "front_end_designer", "fitness_coach", "comfyui_manager"

CASES = [
    # fast
    (D, "What's 17% of 2,340?", "fast"),
    (D, "Remind me to call Dave at 3pm tomorrow.", "fast"),
    (D, "Draft a two-line text telling my crew we start at 8 tomorrow.", "fast"),
    (D, "What's a good sourdough starter feeding ratio?", "fast"),
    (D, "Turn off the living room lights.", "fast"),
    (D, "Convert 2,400 square feet to square meters.", "fast"),
    (D, "Suggest three names for a hiking playlist.", "fast"),
    (D, "What's on my calendar tomorrow?", "fast"),
    (FIT, "How long should I rest between heavy squat sets?", "fast"),
    (FIT, "Log bench press 100kg for 5 reps today.", "fast"),
    (FIT, "Give me a quick 20-minute bodyweight workout.", "fast"),
    (FIT, "Is creatine safe to take every day?", "fast"),
    (DEV, "What's the difference between a list and a tuple in Python?", "fast"),
    (DEV, "Write a Python function that reverses a string.", "fast"),
    (DEV, "What does HTTP status 404 mean?", "fast"),
    (DEV, "Give me the git command to undo my last commit but keep the changes.", "fast"),
    (DEV, "Write a small JavaScript widget: square feet in, estimated insulation cost out.", "fast"),
    (FE, "Suggest a hex color that pairs well with navy #1B2A4A.", "fast"),
    (FE, "Make this button's corners rounded and add a hover state.", "fast"),
    (COMFY, "Generate a hero image of mountains at sunset.", "fast"),
    (COMFY, "Upscale this photo 2x.", "fast"),
    # strong
    (DEV, "Design the database schema and API for a multi-tenant payroll system with audit logs, then implement the core endpoints.", "strong"),
    (DEV, "My async FastAPI app deadlocks under load only in production. Here are the logs - find the root cause and fix it.", "strong"),
    (DEV, "Refactor this 2,000-line module into a clean package with tests, keeping behavior identical.", "strong"),
    (DEV, "Build a full-stack web app for booking appointments with login, a calendar view, and email reminders.", "strong"),
    (DEV, "Write a parser for our custom config language with error recovery and helpful error messages.", "strong"),
    (DEV, "Port our payroll calculations from Excel formulas to Python and verify they match last year's data.", "strong"),
    (DEV, "Figure out why our model's training loss diverges after epoch 3 and propose fixes.", "strong"),
    (DEV, "Prove this sorting algorithm is correct and analyze its worst-case complexity.", "strong"),
    (FE, "Create a complete design system - tokens, type scale, and 20 components with all states - and document it.", "strong"),
    (FE, "Redesign our whole 12-page website for WCAG AA accessibility, with a migration plan.", "strong"),
    (D, "Plan a 10-day Japan itinerary for a family of four with a budget breakdown, rail passes, and rainy-day alternatives.", "strong"),
    (D, "Compare these three insurance quotes line by line and tell me which gives the best coverage for the price, with reasoning.", "strong"),
    (D, "Reconcile last month's payroll ledger against the bank statement and explain every discrepancy.", "strong"),
    (D, "Write a detailed, persuasive two-page proposal for a commercial insulation contract.", "strong"),
    (D, "Research the history of radon mitigation standards and build a sourced timeline.", "strong"),
    (FIT, "Build me a 16-week periodized powerlifting program peaking for a meet, with deloads and attempt selection.", "strong"),
    (FIT, "I have a herniated disc history, knee pain and a 6-day schedule - design a program that works around all of it and explain why.", "strong"),
    (COMFY, "Build a new ComfyUI workflow from scratch that chains image generation, upscaling and video, and debug it until it runs.", "strong"),
    # either
    (DEV, "Build a single-file HTML workout tracker with local storage and progress charts.", "either"),
    (FE, "Design a landing page for my insulation business.", "either"),
    (FIT, "Make me a 4-day upper/lower split for strength.", "either"),
    (D, "Put together a quote for 2,400 sqft of R-30 attic insulation.", "either"),
]

LEVELS = ["trivial: a quick answer, lookup, or one simple action",
          "routine: a standard task done in a few straightforward steps",
          "hard: deep reasoning, tricky debugging, or building something substantial"]
QUESTIONS = {
    "score": lambda: score("How demanding is this task for an AI assistant?", LEVELS),
    "can_fast": lambda: noul("Can a small, fast AI model handle this task well?"),
    "needs_deep": lambda: noul("Does this task require deep reasoning, careful debugging, or building substantial software?"),
    "choice": lambda: choice("Which model should handle this task?", {
        "fast": "a small fast model - good for quick answers, lookups, short scripts and simple edits",
        "strong": "a large careful model - needed for deep reasoning, hard debugging, long or complex builds"}),
}


def run(cfg, personas) -> list[dict]:
    cl = SystemOneClient(cfg.jev["url"], cfg.jev["backend"])
    rows = []
    for pname, text, label in CASES:
        state = {"task": text}  # same as jev.pick_tier: persona context biased simple dev/design tasks up
        t = time.time()
        r = cl.ask(state, {k: q() for k, q in QUESTIONS.items()})
        rows.append({"persona": pname, "text": text, "label": label, "secs": round(time.time() - t, 2),
                     "score": r["score"].score, "can_fast": r["can_fast"].noul,
                     "needs_deep": r["needs_deep"].noul, "p_strong": r["choice"].probabilities.get("strong", 0.0)})
    return rows


def evaluate(rows, signal: str, thr: float) -> dict:
    """Route to strong when signal says so. can_fast is inverted (low = strong)."""
    ok = hard_to_fast = easy_to_strong = n = 0
    for r in rows:
        if r["label"] == "either":
            continue
        n += 1
        v = r[signal]
        strong = (v < thr) if signal == "can_fast" else (v >= thr)
        want_strong = r["label"] == "strong"
        ok += strong == want_strong
        hard_to_fast += want_strong and not strong
        easy_to_strong += strong and not want_strong
    return {"acc": ok / n, "hard_to_fast": hard_to_fast, "easy_to_strong": easy_to_strong,
            "objective": (ok - 0.5 * hard_to_fast) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    _out.add_args(ap)
    a = ap.parse_args()
    dest = _out.Out.of(a)
    cache = Path(__file__).with_name("tier_raw.json")
    cfg = load_config()
    if a.offline:
        rows = json.loads(dest.read(cache).read_text(encoding="utf-8"))
    else:
        t = time.time()
        rows = run(cfg, load_personas(cfg.personas_dir))
        dest.write(cache).write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"raw answers saved to {dest.where(dest.write(cache))}")
        print(f"{len(rows)} cases in {time.time() - t:.0f}s")
    grids = {"score": [x / 10 for x in range(2, 20)], "can_fast": [x / 20 for x in range(1, 20)],
             "needs_deep": [x / 20 for x in range(1, 20)], "p_strong": [x / 20 for x in range(1, 20)]}
    print("signal      best thr | acc   hard->fast easy->strong")
    for sig, grid in grids.items():
        best = max(((evaluate(rows, sig, t), t) for t in grid), key=lambda x: x[0]["objective"])
        s, t = best
        print(f"{sig:10s}  {t:6.2f}   | {s['acc']:.0%}   {s['hard_to_fast']:3d}        {s['easy_to_strong']:3d}")
    for sig in grids:
        print(f"\n{sig}: " + "  ".join(f"{r['label'][0]}{r[sig]:.2f}" for r in sorted(rows, key=lambda r: r[sig])))


if __name__ == "__main__":
    main()
