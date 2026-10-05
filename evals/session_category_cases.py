"""Live eval: does Jev put a whole session in the assistant whose field it is about? (jevh/history.py,
Jev.session_category: the History tab's categories.)

A session is its requests, in order. Jev gets them numbered (history.conversation: a long session shows its first ones and
its last two) and one Choice over the assistants: "which assistant is best suited to handle this whole conversation?",
each option carrying that assistant's own description (what routing uses). The best category is shown, unless it is less
likely than `history.min_confidence`: then the session is listed under "No single field".

Labelling rule: the assistant whose field most of the conversation is in. A conversation that splits evenly between two
fields, or whose field belongs to two assistants (a website is the designer's and the engineer's), lists both and either
is right. A conversation with no specialist field (small talk, calendar, email, shopping, paperwork) belongs to
daily_assistant, the general one.

What matters: a confident category should be right. A "No single field" for a session that has one clear field costs the
reader only a click; a confidently wrong category hides the session in the wrong group. The cases are written for this eval
(they are not anyone's real sessions): single-field sessions of different lengths and styles (typos, short, long), small
talk, and sessions that drift or sit between two fields.

    python evals/session_category_cases.py                  # the tuning, held-out and hard cases (live: needs Jev)
    python evals/session_category_cases.py --holdout        # only the held-out cases
    python evals/session_category_cases.py --offline        # re-sweep min_confidence on the saved raw scores
    python evals/session_category_cases.py --runs 3         # each case three times (a hosted Jev can vary)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevh.config import load_config  # noqa: E402
from jevh.history import MIXED, conversation, settings  # noqa: E402
from jevh.jev import Jev  # noqa: E402
from jevh.personas import load_personas  # noqa: E402

RAW = Path(__file__).with_name("session_category_cases_raw.json")
DEV_, DAILY, FIT, DESIGN, COMFY = "high_end_dev", "daily_assistant", "fitness_coach", "front_end_designer", "comfyui_manager"

# (the session's requests, the assistant(s) whose field it is)
TUNING = [
    # -- engineering
    (["my flask app returns 500 when i delete a job, traceback below", "ok that fixed it, now add pytest tests for it"], DEV_),
    (["write a python script that renames every jpg in a folder by its exif date"], DEV_),
    (["refactor this class to use dataclasses and add type hints", "also make the fetch methods async"], DEV_),
    (["why is my postgres query so slow, here is the explain analyze output"], DEV_),
    (["build me a desktop photo editor in electron with layers and a few filters", "before that, write the test suite for the layer stack"], DEV_),
    (["set up a github actions workflow that runs the tests on every push", "make it cache pip"], DEV_),
    (["write a rust cli that tails a log file and highlights lines with ERROR"], DEV_),
    (["fix the failing typescript build, error TS2345 in api/client.ts"], DEV_),
    (["make a discord bot that posts the weather every morning", "put it in a docker container"], DEV_),
    (["explain this regex and rewrite it so a human can read it"], DEV_),
    (["are we on the newest version of this repo? https://github.com/example/tool", "pull it and rebuild"], DEV_),
    (["i need a rest api for a todo list", "add auth with jwt", "add pagination to the list endpoint", "write the openapi spec",
      "add rate limiting", "dockerize it", "add a health check endpoint", "now write a readme"], DEV_),
    # -- design
    (["make the landing page for my insulation company look modern, blue and white", "add a pricing section with 3 tiers"], DESIGN),
    (["redesign this dashboard so it is less cluttered, screenshot attached"], DESIGN),
    (["what colors go well with a dark navy brand?", "pick me a font pairing for it"], DESIGN),
    (["build me a one page portfolio site for a wedding photographer"], DESIGN),
    (["make my buttons look nicer, soft shadows and rounded corners"], DESIGN),
    (["design a settings screen for a mobile app", "add a dark mode version"], DESIGN),
    (["the nav bar overlaps the hero on mobile, fix the css"], DESIGN),
    (["create a pricing table component in tailwind with a highlighted middle plan"], DESIGN),
    # -- images and video
    (["generate a poster image of a fox in a forest with comfyui", "now upscale it 2x"], COMFY),
    (["make a short video from this photo with a slow zoom in"], COMFY),
    (["load the flux workflow and render 4 variations of a cyberpunk street at night"], COMFY),
    (["restore this old family photo and colorize it"], COMFY),
    (["turn this image of a chair into a 3d model"], COMFY),
    (["which gpu is free for rendering right now", "then queue a batch of 20 portraits"], COMFY),
    (["remove the background from these 30 product photos"], COMFY),
    # -- fitness
    (["plan my workouts for next week, 4 days, full gym", "swap the squats for leg press, my knees hurt"], FIT),
    (["how much protein should i eat to cut at 190 lbs"], FIT),
    (["my bench has been stuck at 225 for a month what do i change"], FIT),
    (["log today: squat 275x5, bench 205x5, row 155x8"], FIT),
    (["build me a 12 week 5k plan, i can run about 1 mile now"], FIT),
    (["is it ok to train legs two days in a row", "what about stretching after"], FIT),
    (["whats a good meal plan for bulking, no dairy"], FIT),
    # -- everyday
    (["whats on my calendar tomorrow", "email dave that ill be late to the 3pm"], DAILY),
    (["remind me to call the dentist friday morning"], DAILY),
    (["get me three quotes for insulating a 1800 sq ft attic"], DAILY),
    (["write a polite email to my landlord about the leaking faucet"], DAILY),
    (["hello", "thanks", "what can you do?"], DAILY),
    (["plan a weekend trip to denver, flights under 300"], DAILY),
    (["draft an invoice for the smith job, 12 hours at 85 an hour"], DAILY),
    (["help me connect my google calendar"], DAILY),
    (["add milk, eggs and bread to my walmart cart"], DAILY),
    (["whats the capital of australia?", "and its population?"], DAILY),
    # -- between fields (either is right)
    (["build me a website with 3 games on it of your choice"], (DESIGN, DEV_)),
    (["make a landing page and wire it to a flask backend with a contact form"], (DESIGN, DEV_)),
    (["i want an app that tracks my workouts and shows my progress over time"], (FIT, DEV_)),
    (["make a poster for my gym in the style of 80s synthwave"], (COMFY, DESIGN)),
    (["write a python script that generates images through the comfyui api in a loop"], (COMFY, DEV_)),
    (["draw a mountain scene in svg", "actually what is on my calendar"], (DAILY, COMFY, DESIGN)),
    (["help me plan meals for the week", "also write a script that scrapes grocery prices"], (DAILY, FIT, DEV_)),
]

HELD = [
    (["our checkout page throws a null pointer when the cart is empty, stack trace attached", "write a regression test"], DEV_),
    (["convert this bash script to python", "and add argparse"], DEV_),
    (["how do i set up a virtualenv and pin my dependencies"], DEV_),
    (["review my pull request, the diff is long but focus on the sql migration"], DEV_),
    (["write a c# class that parses a csv with quoted commas"], DEV_),
    (["build a command line tool that syncs a folder to s3", "it should skip files that did not change", "add a dry run flag",
      "write tests for the skip logic", "package it for pip"], DEV_),
    (["make my portfolio look more premium, the hero section feels flat"], DESIGN),
    (["pick an icon set and a spacing scale for my app", "show me the tokens as css variables"], DESIGN),
    (["design a logo mark for a plumbing company, simple and bold"], DESIGN),
    (["make this email template responsive and a bit prettier"], DESIGN),
    (["animate the cards so they fade up on scroll, keep it subtle"], DESIGN),
    (["render a character turnaround of a knight with the sdxl workflow"], COMFY),
    (["upscale all the images in my outputs folder to 4k"], COMFY),
    (["generate a 5 second clip of waves on a beach, cinematic"], COMFY),
    (["run the comfyui workflow for inpainting and fix the hands in this picture"], COMFY),
    (["i have shoulder pain on overhead press, what should i swap in"], FIT),
    (["make me a push pull legs split and tell me how to progress each lift"], FIT),
    (["how many calories do i burn on a 45 minute run", "and how should i eat after"], FIT),
    (["track my weight every morning and tell me the weekly trend"], FIT),
    (["whats on my schedule this week", "move the thursday meeting to friday"], DAILY),
    (["order a birthday cake for saturday and find a bakery nearby"], DAILY),
    (["summarize this pdf about my health insurance", "what is my deductible"], DAILY),
    (["thanks!", "ok", "never mind"], DAILY),
    (["write a thank you note to the team for the great quarter"], DAILY),
    (["i want to build a little game and also need art for it"], (DEV_, COMFY, DESIGN)),
    (["design a poster for a 5k fun run and also a landing page for signups"], (DESIGN, FIT, COMFY)),
    (["write a tracker for my lifts in python", "make the output look nice in the terminal"], (DEV_, FIT)),
]

# Written after the two sets above all came out right, to be harder: words that point at the wrong field (hardware shopping,
# a calendar parser, "run"), a field no built-in assistant has (cooking), and sessions that change field half-way.
HARD = [
    (["go to amazon and add a 4080 super and 64gb of ddr5 to my cart"], DAILY),
    (["write unit tests for my calendar parsing function", "it should handle all-day events"], DEV_),
    (["run my tests and tell me which ones failed"], DEV_),
    (["can you put together a 5k running plan, i run twice a week already"], FIT),
    (["email my trainer and ask if we can move thursdays session to friday"], (DAILY, FIT)),
    (["design the ui for a workout tracker app", "what colors for a fitness brand"], (DESIGN, FIT)),
    (["whats the best programming language for a total beginner"], (DEV_, DAILY)),
    (["how long do i cook a 1 inch ribeye", "and what internal temp is medium rare"], DAILY),
    (["plan 3 dinners for the week, vegetarian, under $60"], (DAILY, FIT)),
    (["could you help me connect my facebook account to the harness"], DAILY),
    (["what does this error mean: ModuleNotFoundError: No module named 'requests'"], DEV_),
    (["write a cover letter for a warehouse supervisor job"], DAILY),
    (["i need help with my taxes, where do i put my 1099 income"], DAILY),
    (["make a 30 second video of a dog on a skateboard"], COMFY),
    (["explain how the squat works biomechanically", "and why my lower back rounds at the bottom"], FIT),
    (["find me a good dentist near me and book an appointment"], DAILY),
    (["why does my css grid leave a gap under the last row"], (DESIGN, DEV_)),
    (["draft the readme for my python package", "add install instructions and an example"], DEV_),
    (["make me a thumbnail for my youtube video about home repair"], (COMFY, DESIGN)),
    (["compare two quotes for a roof replacement and tell me which is better"], DAILY),
    (["i want a budget spreadsheet for my small business", "add formulas for monthly totals"], (DAILY, DEV_)),
    (["cook a steak", "find a recipe for chimichurri", "actually help me write a python script to convert units"], (DAILY, DEV_)),
    (["generate a logo with comfyui", "now make a website using it", "make the site responsive", "add a contact form with validation"], (COMFY, DESIGN, DEV_)),
]


def categories() -> dict[str, str]:
    cfg = load_config()
    return {n: p.description for n, p in load_personas(cfg.personas_dirs, []).items()}


def run_live(cases, runs: int = 1) -> list[dict]:
    cfg = load_config()
    jev, cats, s = Jev(cfg), categories(), settings(cfg.raw)
    rows = []
    for reqs, want in cases:
        text = conversation(reqs, int(s["max_requests"]), int(s["request_chars"]))
        for _ in range(runs):
            probs, _resp = jev.session_category(text, cats)
            rows.append({"requests": reqs, "want": [want] if isinstance(want, str) else list(want), "probs": probs})
    return rows


def shown(probs: dict[str, float], min_confidence: float) -> str:
    top = max(probs, key=lambda k: probs[k])
    return MIXED if probs[top] < min_confidence else top


def evaluate(rows: list[dict], min_confidence: float) -> dict:
    right = wrong = mixed = lost = 0
    bad = []
    for r in rows:
        got = shown(r["probs"], min_confidence)
        if got == MIXED:
            mixed += 1
            lost += len(r["want"]) == 1   # a clear session that ended up under "No single field"
        elif got in r["want"]:
            right += 1
        else:
            wrong += 1
            bad.append((r["requests"][0][:60], r["want"], got, round(max(r["probs"].values()), 2)))
    n = len(rows)
    return {"n": n, "right": right, "wrong": wrong, "mixed": mixed, "lost_clear": lost, "bad": bad,
            "confident_accuracy": right / max(right + wrong, 1)}


def report(name: str, rows: list[dict], min_confidence: float) -> None:
    e = evaluate(rows, min_confidence)
    print(f"-- {name}: {e['n']} sessions at min_confidence {min_confidence}")
    print(f"   right {e['right']}   wrong {e['wrong']}   under 'No single field' {e['mixed']} (of them clear sessions: {e['lost_clear']})"
          f"   a confident category is right {e['confident_accuracy']:.0%}")
    for first, want, got, conf in e["bad"]:
        print(f"   WRONG {first!r}: wanted {'/'.join(want)}, put under {got} ({conf})")


def sweep(rows: list[dict]) -> None:
    for t in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        e = evaluate(rows, t)
        print(f"   min_confidence {t:.1f}: right {e['right']:3d}  wrong {e['wrong']:2d}  'No single field' {e['mixed']:3d} "
              f"(clear sessions lost: {e['lost_clear']:2d})  confident accuracy {e['confident_accuracy']:.0%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="re-sweep the saved raw scores")
    ap.add_argument("--holdout", action="store_true", help="only the held-out cases")
    ap.add_argument("--runs", type=int, default=1)
    a = ap.parse_args()
    cfg = load_config()
    at = float(settings(cfg.raw)["min_confidence"])
    if a.offline:
        saved = json.loads(RAW.read_text(encoding="utf-8"))
        sets = {k: v for k, v in saved.items() if not a.holdout or k == "held"}
    else:
        sets = {}
        if not a.holdout:
            sets["tuning"] = run_live(TUNING, a.runs)
        sets["held"] = run_live(HELD, a.runs)
        if not a.holdout:
            sets["hard"] = run_live(HARD, a.runs)
            RAW.write_text(json.dumps(sets, indent=1), encoding="utf-8")
    for name, rows in sets.items():
        report(name, rows, at)
        print("   sweep:")
        sweep(rows)
    both = [r for rows in sets.values() for r in rows]
    if len(sets) > 1:
        print("-- all sessions:")
        sweep(both)


if __name__ == "__main__":
    main()
