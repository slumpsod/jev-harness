"""Live eval of Jev's persona routing + collaborator suggestions, with an offline
threshold sweep (one model pass, then every threshold is re-decided from cached
answers - the same "cache-only" method as openjev/router_tune_semif.py).

    python evals/routing_cases.py            # run + report at config thresholds
    python evals/routing_cases.py --sweep    # also sweep collab threshold / complexity guard
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevh.config import load_config  # noqa: E402
from jevh.jev import Jev  # noqa: E402
from jevh.personas import load_personas  # noqa: E402
import _out  # noqa: E402  (evals/_out.py: where a run's results go)

D, DEV, FE, FIT, COMFY = "daily_assistant", "high_end_dev", "front_end_designer", "fitness_coach", "comfyui_manager"

# (request, expected persona, expected collaborators for that persona as lead)
CASES = [
    ("What's on my calendar tomorrow?", D, set()),
    ("Remind me to call Dave Miller about the radon kit on Thursday.", D, set()),
    ("Search my email for the invoice from Centerpoint last month.", D, set()),
    ("Log today's job: 6 hours labor at 75/hr plus 300 in materials for Talon.", D, set()),
    ("What's a good gift for my mom's 60th birthday?", D, set()),
    ("Summarize this PDF rebate form for me.", D, set()),
    ("Turn the living room lights to 30 percent.", D, set()),
    ("How far is it from Spanish Fork to Provo?", D, set()),
    ("Put together a quote for 2,400 sqft of R-30 attic insulation.", D, set()),
    # connecting the user's own accounts (nango-setup), added 2026-10-02
    ("Connect my Gmail so you can read my email.", D, set()),
    ("Help me link my Google Calendar to the assistant.", D, set()),
    ("Set up Nango so you can use my accounts.", D, set()),
    ("My Slack connection stopped working, can you reconnect it?", D, set()),
    ("Refactor this FastAPI route to use dependency injection.", DEV, set()),
    ("Why does my Python script throw KeyError: 'id' when the JSON is empty?", DEV, set()),
    ("Write a PowerShell script that backs up my Documents folder nightly.", DEV, set()),
    ("Set up a SQLite schema for storing customer jobs and invoices.", DEV, set()),
    ("Review this pull request diff for bugs.", DEV, set()),
    ("My vLLM server OOMs on startup with a 27B model, how do I fix it?", DEV, set()),
    ("Build a small web app where customers can book appointments, with a nice modern UI.", DEV, {FE}),
    ("Build a landing page for my insulation business with a custom hero illustration.", FE, {COMFY}),
    ("Design a dark-mode dashboard layout for my payroll numbers.", FE, set()),
    ("Pick a color palette and typography for my company website.", FE, set()),
    ("Make the buttons on this page look more modern and accessible.", FE, set()),
    ("Mock up the home screen for a habit tracking app.", FE, set()),
    ("Make me a 4 day upper/lower split for strength.", FIT, set()),
    ("How should I progress my bench press, I've stalled at 80kg?", FIT, set()),
    ("My knees hurt when I squat, what should I change?", FIT, set()),
    ("How much protein should I eat to build muscle at 85kg?", FIT, set()),
    ("Plan a deload week for me.", FIT, set()),
    ("I want an app to look at my workouts and track my progress over time.", FIT, {DEV, FE}),
    ("Build me a web dashboard that charts my lifts and bodyweight.", FIT, {DEV, FE}),
    ("Generate a hero image of mountains at sunset, 16:9.", COMFY, set()),
    ("Upscale this photo to 4K.", COMFY, set()),
    ("Make a 5 second video of waves crashing on rocks.", COMFY, set()),
    ("Turn this product photo into a 3D model.", COMFY, set()),
    ("Edit this image to remove the background and make it blue.", COMFY, set()),
]


def run(cfg, jev: Jev, personas) -> list[dict]:
    rows = []
    for text, want, want_collab in CASES:
        t = time.time()
        route = jev.route_persona(text, personas)
        lead = personas[want]  # collab is judged with the *expected* lead so errors don't compound
        collab = jev.collab(text, lead, personas)
        rows.append({"text": text, "want": want, "want_collab": sorted(want_collab),
                     "route": route.persona, "route_conf": route.confidence, "probs": route.probabilities,
                     "collab_scores": collab.scores, "complexity": collab.complexity,
                     "secs": round(time.time() - t, 2)})
    return rows


def decide_collab(row, thr: float, min_complexity: float) -> set[str]:
    if row["complexity"] is None or row["complexity"] < min_complexity:
        return set()
    return {n for n, s in row["collab_scores"].items() if s >= thr}


def report(rows, thr: float, min_cx: float, verbose: bool = True) -> dict:
    route_ok = sum(r["route"] == r["want"] for r in rows)
    exact = sum(decide_collab(r, thr, min_cx) == set(r["want_collab"]) for r in rows)
    needless = sum(len(decide_collab(r, thr, min_cx) - set(r["want_collab"])) for r in rows)
    missed = sum(len(set(r["want_collab"]) - decide_collab(r, thr, min_cx)) for r in rows)
    if verbose:
        for r in rows:
            got = decide_collab(r, thr, min_cx)
            flag = "" if r["route"] == r["want"] else f"  ROUTE->{r['route']}"
            cflag = "" if got == set(r["want_collab"]) else f"  COLLAB {sorted(got)} want {r['want_collab']}"
            if flag or cflag:
                print(f"  x {r['text'][:70]:70s}{flag}{cflag}  (cx {r['complexity']:.2f})")
    return {"route_acc": route_ok / len(rows), "collab_exact": exact / len(rows),
            "needless": needless, "missed": missed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--cache", default=str(Path(__file__).with_name("routing_raw.json")),
                    help="the raw answers file (default: the committed routing_raw.json, written to --out instead)")
    ap.add_argument("--offline", action="store_true", help="re-decide from the cache, no model calls")
    _out.add_args(ap)
    a = ap.parse_args()
    dest = _out.Out.of(a)
    cfg = load_config()
    personas = load_personas(cfg.personas_dir)
    if a.offline:
        rows = json.loads(dest.read(a.cache).read_text(encoding="utf-8"))
    else:
        jev = Jev(cfg)
        t = time.time()
        rows = run(cfg, jev, personas)
        dest.write(a.cache).write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"{len(rows)} cases, {jev.client.calls} jev calls, {time.time() - t:.1f}s "
              f"({cfg.jev.get('backend')}); raw answers saved to {dest.where(dest.write(a.cache))}")
    thr, min_cx = cfg.t("collab"), 0.5
    res = report(rows, thr, min_cx)
    print(f"route top-1 {res['route_acc']:.1%} | collab exact {res['collab_exact']:.1%} | "
          f"needless {res['needless']} | missed {res['missed']}  (collab>={thr}, complexity>={min_cx})")
    if a.sweep:
        print("\ncollab_thr  min_cx  exact   needless missed")
        grid = [(t, c) for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) for c in (0.0, 0.5, 0.8, 1.0, 1.2)]
        for t, c in grid:
            r = report(rows, t, c, verbose=False)
            print(f"   {t:.2f}     {c:.1f}   {r['collab_exact']:6.1%}   {r['needless']:3d}     {r['missed']:3d}")


if __name__ == "__main__":
    main()
