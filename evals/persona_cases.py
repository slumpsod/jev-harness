"""Live eval of the three Jev decisions behind persona creation (jevh/forge.py). NOT TUNED YET: the
thresholds in config.yaml (no_fit, orphan, same_specialist) are first guesses; run this on semif and
set them from the sweeps below, then put the numbers in the config comments like the other gates.

- no_fit: "does the request fall outside what every assistant is for?" on the real personas. A false
  positive pools a request that a persona could have handled (and, with enough similar ones, creates
  a needless persona); a miss only delays a persona.
- orphan: a learned skill's P("none of these assistants"). A false positive creates a persona for a
  skill that has a home (the gate and the twin check are the backstop); a miss leaves the skill
  offered to every persona that allows learned-*.
- same_specialist: two unhandled requests that one specialist would suit. False positives gather
  unrelated requests into one persona; misses make a persona wait for more requests.

    python evals/persona_cases.py            # live, sweep
    python evals/persona_cases.py --offline  # re-sweep the cached raw answers
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevh.config import load_config  # noqa: E402
from jevh.jev import Jev  # noqa: E402
from jevh.personas import load_personas  # noqa: E402
from jevh.skills import Skill  # noqa: E402

RAW = Path(__file__).with_name("persona_cases_raw.json")

NO_FIT = [  # (request, no persona is for it?)
    ("Fix this TypeError in my Flask app", False),
    ("Plan my lifts for the week, I have a bad shoulder", False),
    ("Generate an image of a lighthouse at dusk", False),
    ("Design a landing page for my insulation business", False),
    ("What's on my calendar tomorrow?", False),
    ("Draft an email to a client about the delayed quote", False),
    ("Turn off the living room lights", False),
    ("Write a script that renames my photos by date", False),
    ("Set up a ComfyUI workflow for video upscaling", False),
    ("Remind me to call Dave at 3pm", False),
    ("How do I intonate a classical guitar's bridge saddle?", True),
    ("Explain the Sicilian Defense and which lines to study first", True),
    ("Drill me on Spanish preterite vs imperfect with exercises", True),
    ("Plan a no-dig permaculture bed layout for a shady 4x6m yard", True),
    ("Which slicer settings stop stringing in PETG on a Bambu A1?", True),
    ("Walk me through setting up an equatorial mount for astrophotography", True),
    ("Review this tenancy agreement clause by clause for risks", True),
    ("Teach me to read a Kubelka-Munk colour mixing chart for oil paints", True),
]

SAME = [  # (request a, request b, one specialist suits both?)
    ("Which slicer settings stop stringing in PETG?", "My first layer won't stick on a textured PEI plate", True),
    ("Explain the Sicilian Defense", "How do I punish the London System as black?", True),
    ("Drill me on Spanish preterite", "What's the difference between ser and estar?", True),
    ("How do I polar align an equatorial mount?", "What exposure time for the Orion nebula at f/5?", True),
    ("Review this tenancy clause for risks", "Is a 60-day notice period normal in a lease?", True),
    ("Which slicer settings stop stringing in PETG?", "Explain the Sicilian Defense", False),
    ("Drill me on Spanish preterite", "How do I polar align an equatorial mount?", False),
    ("Review this tenancy clause for risks", "Plan a no-dig permaculture bed", False),
    ("How do I intonate a guitar?", "Which slicer settings stop stringing in PETG?", False),
]

SKILLS = [  # (name, description, belongs to no persona?)
    ("learned-guitar-intonation", "Use when setting the intonation on an acoustic or classical guitar.", True),
    ("learned-chess-opening-study", "Use when building a study plan for a chess opening repertoire.", True),
    ("learned-petg-stringing-fix", "Use when tuning slicer retraction and temperature settings for stringing.", True),
    ("learned-tenancy-clause-review", "Use when reviewing a lease or tenancy agreement for risky clauses.", True),
    ("learned-comfyui-checkpoint-names", "Use when a ComfyUI workflow fails because a checkpoint name is not in the list.", False),
    ("learned-pytest-fixture-fix", "Use when pytest fixtures fail with scope or teardown errors.", False),
    ("learned-cloudflare-pages-deploy", "Use when deploying a Vite frontend to Cloudflare Pages with env variables.", False),
    ("learned-weekly-workout-split", "Use when building a weekly strength training split around an injury.", False),
    ("learned-quote-email-followup", "Use when writing a follow-up email about a delayed customer quote.", False),
]


def run_live() -> dict:
    cfg = load_config()
    jev = Jev(cfg)
    personas = load_personas(cfg.personas_dir)
    raw = {"no_fit": [], "same": [], "orphan": []}
    for request, _ in NO_FIT:
        raw["no_fit"].append(jev.no_fit(request, personas))
    for a, b, _ in SAME:
        raw["same"].append(jev.same_specialist(a, [b])[0])
    for name, desc, _ in SKILLS:
        best, none, _probs = jev.skill_home(Skill(name, desc, "", ""), personas)
        raw["orphan"].append({"none": none, "best": best})
    RAW.write_text(json.dumps(raw, indent=1), encoding="utf-8")
    return raw


def sweep(name: str, scores: list[float], labels: list[bool]) -> None:
    print(f"\n== {name} ({len(labels)} cases, {sum(labels)} positive)")
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        tp = sum(s >= t and l for s, l in zip(scores, labels))
        fp = sum(s >= t and not l for s, l in zip(scores, labels))
        print(f"  t={t:.1f}  caught {tp}/{sum(labels)}  false positives {fp}")
    print("  " + "  ".join(f"{'+' if l else '-'}{s:.2f}" for s, l in zip(scores, labels)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    raw = json.loads(RAW.read_text(encoding="utf-8")) if ap.parse_args().offline else run_live()
    sweep("no_fit: no persona is for the request", raw["no_fit"], [c[1] for c in NO_FIT])
    sweep("same_specialist: one specialist suits both", raw["same"], [c[2] for c in SAME])
    # the skill is an orphan when "none of these" leads (forge.note_skill also needs it >= orphan)
    sweep("orphan: P(none of the personas) for a learned skill", [r["none"] for r in raw["orphan"]],
          [c[2] for c in SKILLS])
    led = [r["best"] is None for r in raw["orphan"]]
    print(f"  none-of-these led on {sum(l and c[2] for l, c in zip(led, SKILLS))}/{sum(c[2] for c in SKILLS)} orphans, "
          f"{sum(l and not c[2] for l, c in zip(led, SKILLS))} skills with a home")


if __name__ == "__main__":
    main()
