"""Live eval of the three Jev decisions behind persona creation (jevh/forge.py), on four sets of cases:
dev (the first look), held (written afterwards), fresh and fresh2 (newer; fresh2 was written before any of it
was scored). Thresholds (config.yaml, 2026-10-02, semif): no_fit 0.60, orphan 0.50, same_specialist 0.60.

- no_fit: P("none of these assistants has the expertise this request needs") for a request on the real
  personas: 41 of 42 specialist requests flagged at 0.60 across the four sets (39 reach the pool: the router
  sends a few to a specialist persona); ordinary requests 6/60 (the false positives are chores no persona
  description lists: naming ideas, a slogan, compound interest, a cover letter). A false positive only pools a request (a persona needs 3 similar ones, then the writer, the
  twin check and the gate, then your approval); a miss only delays a persona.
- orphan: a learned skill's P("none of these assistants"): 8/8 orphans, 0 of 9 skills with a home.
- same_specialist: "would one person who is an expert in A's subject also be an expert in B's?": 28/29
  same-subject pairs, 0 of 24 different ones (and 52/54 and 0/220 on the stream's pairs in the bench). The
  first wording ("one specialist assistant for both") passed unrelated pairs at up to 0.75.

    python evals/persona_cases.py            # live, sweep
    python evals/persona_cases.py --holdout  # the held-out cases (run once the thresholds are set)
    python evals/persona_cases.py --fresh    # newer cases (looked at while choosing the no_fit wording)
    python evals/persona_cases.py --fresh2   # newest cases: written before scoring, the confirmation, run once
    python evals/persona_cases.py --offline  # re-sweep the cached raw answers (with --holdout or --fresh too)

The whole feature (the drafting model, the routing gate, a real turn) is in evals/bench/persona_creation.py.
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
import _out  # noqa: E402  (evals/_out.py: where a run's results go)

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


# The held-out set: written after the thresholds were first looked at, from different domains, so it
# shows whether they only fit the cases above. Its negatives include requests that sit between two
# personas (a meal plan around training, a bash script that logs workouts): those must NOT start a persona.
NO_FIT_HELD = [
    ("Summarize the rebate form I uploaded", False),
    ("Why does pytest say fixture 'db' not found?", False),
    ("Make my site's navbar look cleaner", False),
    ("How many sets per week do I need for back hypertrophy?", False),
    ("Upscale and sharpen this product photo", False),
    ("Text my crew that the job moved to Friday", False),
    ("Build me a meal prep plan around my training week", False),
    ("Write a bash script that logs my workouts to a CSV", False),
    ("Make a logo for my insulation company", False),
    ("Help me write a cover letter for a project manager job", False),
    ("How do I overwinter dahlia tubers in zone 5?", True),
    ("Explain cricket's LBW rule and how DRS works", True),
    ("Teach me ii-V-I chord voicings on piano", True),
    ("What's a good sourdough starter feeding schedule for a 75% hydration loaf?", True),
    ("Walk me through filing quarterly estimated taxes as a sole proprietor", True),
    ("Help me read my blood panel's lipid numbers", True),
]
SAME_HELD = [
    ("How do I overwinter dahlia tubers?", "When should I start tomatoes indoors in zone 5?", True),
    ("Explain cricket's LBW rule", "How does DRS work in test matches?", True),
    ("Teach me ii-V-I voicings on piano", "What scales work over a minor 7b5 chord?", True),
    ("What's a good sourdough feeding schedule?", "Why is my loaf dense with a tight crumb?", True),
    ("How do I overwinter dahlia tubers?", "Explain cricket's LBW rule", False),
    ("Teach me ii-V-I voicings on piano", "Review this tenancy clause for risks", False),
    ("What's a good sourdough feeding schedule?", "Which slicer settings stop stringing in PETG?", False),
    ("Explain cricket's LBW rule", "How do I fix a TypeError in Flask?", False),
]
SKILLS_HELD = [
    ("learned-dahlia-overwintering", "Use when storing and overwintering dahlia tubers in a cold climate.", True),
    ("learned-sourdough-hydration", "Use when adjusting sourdough hydration and feeding schedule for a loaf.", True),
    ("learned-jazz-voicings", "Use when teaching jazz chord voicings and ii-V-I progressions on piano.", True),
    ("learned-cricket-dls-method", "Use when explaining the Duckworth-Lewis-Stern method in a rain-affected cricket match.", True),
    ("learned-fastapi-dependency-override", "Use when a FastAPI test needs a dependency override.", False),
    ("learned-hero-image-aspect-ratios", "Use when generating a hero image at the right aspect ratio in ComfyUI.", False),
    ("learned-calendar-recurring-events", "Use when creating recurring calendar events with exceptions.", False),
    ("learned-progressive-overload-plateau", "Use when a lifter's progress has plateaued on a main lift.", False),
]


# The fresh set: written (2026-10-02) after the no_fit question was changed to a Choice with "none of these",
# and scored only after its threshold was chosen on the tuning cases and the held-out ones had been looked at
# once. Ordinary requests include a few general-writing ones (proofread, translate, an About-page bio): a
# general assistant handles them, though its description lists none of them, so they are the likely false
# positives. The same-specialist pairs are in domains (law, tax, medicine, repair) where the score ran lukewarm.
NO_FIT_FRESH = [
    ("What's the weather going to be like for the job site on Thursday?", False),
    ("Reply to Dana's email and tell her the quote will be ready Monday", False),
    ("Add 40 minutes of travel time to tomorrow's 9am estimate", False),
    ("Calculate payroll for last week: 3 guys, 42 hours each at 28 an hour", False),
    ("Dim the kitchen lights and turn on the porch light", False),
    ("Why is my Python script slower after I added the second loop?", False),
    ("Write a SQL query that totals invoices by customer for September", False),
    ("Convert this CSS to use a flex layout instead of floats", False),
    ("Give me a color palette for a bakery website", False),
    ("I'm sore after deadlifts, should I train legs again tomorrow?", False),
    ("What should I eat before a morning lift?", False),
    ("Turn this sketch into a photorealistic render", False),
    ("Remember that the Hendersons prefer cellulose insulation", False),
    ("What's 15% of 2,340?", False),
    ("Build me a 16-week marathon training plan", False),
    ("Help me word a polite reminder to a customer who hasn't paid", False),
    ("Proofread this paragraph for typos", False),
    ("Translate this note into Spanish for my crew", False),
    ("Write a short bio for my company's About page", False),
    ("What torque should I use on the lug nuts for a 2014 F-150?", True),
    ("My check engine light shows P0420, what does that mean and what should I check?", True),
    ("How do I cut a through dovetail by hand and mark it out?", True),
    ("What lens and settings should I use to shoot a wedding reception indoors?", True),
    ("How do I mix a vocal track so it sits in the mix?", True),
    ("Balance this redox equation in acidic solution: MnO4- + Fe2+ -> Mn2+ + Fe3+", True),
    ("How do I tell a Cooper's hawk from a sharp-shinned hawk?", True),
    ("Explain the grammar of this Japanese sentence: watashi wa kinou gakkou ni ikimashita", True),
    ("What's the best way to restring and tune a banjo?", True),
    ("How do I size the beam for a 12 ft span deck with 2x10 joists?", True),
    ("What should I do in the first year after planting an apple tree?", True),
    ("How do I brew a clean pale ale with extract and steeping grains?", True),
]
SAME_FRESH = [
    ("What does P0420 mean on my truck?", "How do I test a catalytic converter?", True),
    ("What torque for lug nuts on an F-150?", "How often should I rotate my tires?", True),
    ("How do I cut a through dovetail?", "What's the best wood glue for a table top?", True),
    ("How do I mix a vocal so it sits in the mix?", "What EQ cuts help a muddy kick drum?", True),
    ("Is a 30-day eviction notice valid for a month-to-month lease?", "Can a landlord enter without notice?", True),
    ("Can I deduct a home office as a sole proprietor?", "When are quarterly estimated taxes due?", True),
    ("What does a high LDL number mean on a blood test?", "How is a lipid panel different from a metabolic panel?", True),
    ("How do I brew a clean pale ale?", "What temperature should I ferment lager yeast at?", True),
    ("What does P0420 mean on my truck?", "How do I cut a through dovetail?", False),
    ("Is a 30-day eviction notice valid for a month-to-month lease?", "How do I mix a vocal so it sits in the mix?", False),
    ("Can I deduct a home office as a sole proprietor?", "How do I brew a clean pale ale?", False),
    ("What does a high LDL number mean on a blood test?", "How do I test a catalytic converter?", False),
    ("Is a 30-day eviction notice valid for a month-to-month lease?", "Can I deduct a home office as a sole proprietor?", False),
]

# Fresh pairs 2: written before they were scored, after the same-specialist wording was changed to "would one person
# who is an expert in A's subject also be an expert in B's" (chosen on 54 same-domain and 220 cross-area pairs from
# the stream in evals/bench/persona_creation.py). Positives share a narrow subject; negatives are different fields.
SAME_FRESH2 = [
    ("How do I bleed the brakes on a 2012 Civic?", "What brake fluid should I use for a Honda Civic?", True),
    ("How much should I water a fiddle-leaf fig?", "Why are the leaves on my fiddle-leaf fig dropping?", True),
    ("How do I fix a leaking compression fitting under a sink?", "How do I replace a toilet fill valve?", True),
    ("What does a P/E ratio tell me about a stock?", "How do I read a company's balance sheet?", True),
    ("How do I stop my sourdough starter smelling of acetone?", "How often should I feed a sourdough starter in the fridge?", True),
    ("What's the best way to learn the notes on a guitar fretboard?", "How do I practice barre chords without buzzing?", True),
    ("How do I read a topographic map?", "How do I use a compass bearing to navigate off trail?", True),
    ("How do I treat a sprained ankle with RICE?", "When can I return to running after an ankle sprain?", True),
    ("What is the difference between a Roth and a traditional IRA?", "How do required minimum distributions work?", True),
    ("How do I clean a rusty cast iron skillet?", "What oil is best for seasoning cast iron?", True),
    ("How do I grind coffee for a V60?", "What water temperature is best for a pour-over?", True),
    ("How do I identify poison ivy?", "What should I do after touching poison ivy?", True),
    ("How do I bleed the brakes on a 2012 Civic?", "How should I water a fiddle-leaf fig?", False),
    ("How do I fix a leaking compression fitting under a sink?", "What does a P/E ratio tell me about a stock?", False),
    ("How do I stop my sourdough starter smelling of acetone?", "How do I read a topographic map?", False),
    ("What's the best way to learn the notes on a guitar fretboard?", "How do I treat a sprained ankle with RICE?", False),
    ("What is the difference between a Roth and a traditional IRA?", "How do I clean a rusty cast iron skillet?", False),
    ("How do I grind coffee for a V60?", "How do I identify poison ivy?", False),
    ("How do I bleed the brakes on a 2012 Civic?", "How do I replace a toilet fill valve?", False),
    ("What does a P/E ratio tell me about a stock?", "How do I bleed the brakes on a 2012 Civic?", False),
    ("How do I practice barre chords without buzzing?", "What is the difference between a Roth and a traditional IRA?", False),
    ("How do I identify poison ivy?", "How do I use a compass bearing to navigate off trail?", False),
    ("How do I stop my sourdough starter smelling of acetone?", "How do I read a company's balance sheet?", False),
]

# Fresh set 2: written before any of it was scored, after the question became "which assistant has the expertise
# this request needs" and the threshold was set from the three sets above. The confirmation: run it once.
NO_FIT_FRESH2 = [
    ("Can you rewrite this sentence to sound more professional?", False),
    ("Give me five name ideas for my new insulation crew", False),
    ("Explain what R-value means in simple terms", False),
    ("Convert 3.5 inches to millimeters", False),
    ("What's the capital of Portugal?", False),
    ("Thanks, that works", False),
    ("Summarize this email thread in three bullets", False),
    ("Make a checklist for closing up a job site at the end of the day", False),
    ("Draft a text to the homeowner confirming Friday's appointment", False),
    ("What time is sunset in Provo today?", False),
    ("Fix the off-by-one error in this for loop", False),
    ("Set up a cron job to back up my database every night", False),
    ("Make the hero section of my landing page feel more premium", False),
    ("Suggest a 3-day full-body routine for a beginner", False),
    ("How many calories are in a chicken burrito bowl?", False),
    ("Upscale this blurry logo to 4K", False),
    ("Remind me to order blown-in insulation on Monday", False),
    ("Rename these 50 files to include the date", False),
    ("Brainstorm a slogan for a weatherization company", False),
    ("Explain how compound interest works", False),
    ("What are the key differences between Python's asyncio and threading for I/O-bound work?", False),
    ("How do I solder a copper pipe joint without leaving pinholes?", True),
    ("What's the correct way to wire a 3-way switch with a switch loop?", True),
    ("How do I treat powdery mildew on squash without chemicals?", True),
    ("What are the pros and cons of a 4-4-2 versus a 4-3-3 formation in soccer?", True),
    ("How do I read a tide table and plan a kayak crossing?", True),
    ("What flour and hydration should I use for a Neapolitan pizza dough?", True),
    ("Walk me through changing the brake pads on a 2016 Honda Civic", True),
    ("How should I set up a freshwater planted aquarium for shrimp?", True),
    ("What does the Krebs cycle produce and where does it happen?", True),
    ("Explain how to use a sextant for celestial navigation", True),
    ("How do I replace a sash cord in an old double-hung window?", True),
    ("How do I teach a dog reliable recall off leash?", True),
    ("Explain how bidding works in contract bridge for opening hands", True),
    ("How do I develop black-and-white film at home with a stainless tank?", True),
    ("Explain the offside rule in rugby union", True),
    ("What's the fingering for an F major barre chord on ukulele?", True),
]

SETS = {"dev": (NO_FIT, SAME, SKILLS), "held": (NO_FIT_HELD, SAME_HELD, SKILLS_HELD),
        "fresh": (NO_FIT_FRESH, SAME_FRESH, []), "fresh2": (NO_FIT_FRESH2, SAME_FRESH2, [])}
RAW_FILES = {"dev": "persona_cases_raw.json", "held": "persona_cases_holdout_raw.json",
             "fresh": "persona_cases_fresh_raw.json", "fresh2": "persona_cases_fresh2_raw.json"}


def run_live(no_fit_cases=NO_FIT, same_cases=SAME, skill_cases=SKILLS, raw_path: Path | None = None) -> dict:
    raw_path = raw_path or RAW
    cfg = load_config()
    jev = Jev(cfg)
    personas = load_personas(cfg.personas_dirs)
    raw = {"no_fit": [], "same": [], "orphan": []}
    for request, _ in no_fit_cases:
        raw["no_fit"].append(jev.no_fit(request, personas))
    for a, b, _ in same_cases:
        raw["same"].append(jev.same_specialist(a, [b])[0])
    for name, desc, _ in skill_cases:
        best, none, _probs = jev.skill_home(Skill(name, desc, "", ""), personas)
        raw["orphan"].append({"none": none, "best": best})
    raw_path.write_text(json.dumps(raw, indent=1), encoding="utf-8")
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
    ap.add_argument("--holdout", action="store_true", help="the held-out cases instead of the tuning ones")
    ap.add_argument("--fresh", action="store_true", help="the fresh cases (run once, after the thresholds are set)")
    ap.add_argument("--fresh2", action="store_true", help="the second fresh set: the confirmation, run once")
    _out.add_args(ap)
    a = ap.parse_args()
    dest = _out.Out.of(a)
    split = "fresh2" if a.fresh2 else "fresh" if a.fresh else "held" if a.holdout else "dev"
    no_fit, same, skills = SETS[split]
    path = RAW.with_name(RAW_FILES[split])
    if a.offline:
        raw = json.loads(dest.read(path).read_text(encoding="utf-8"))
    else:
        raw = run_live(no_fit, same, skills, dest.write(path))
        print(f"raw scores saved to {dest.where(dest.write(path))}")
    sweep("no_fit: no persona is for the request", raw["no_fit"], [c[1] for c in no_fit])
    sweep("same_specialist: one specialist suits both", raw["same"], [c[2] for c in same])
    # the skill is an orphan when "none of these" leads (forge.note_skill also needs it >= orphan)
    sweep("orphan: P(none of the personas) for a learned skill", [r["none"] for r in raw["orphan"]],
          [c[2] for c in skills])
    led = [r["best"] is None for r in raw["orphan"]]
    print(f"  none-of-these led on {sum(l and c[2] for l, c in zip(led, skills))}/{sum(c[2] for c in skills)} orphans, "
          f"{sum(l and not c[2] for l, c in zip(led, skills))} skills with a home")


if __name__ == "__main__":
    main()
