"""Live eval of Jev's memory gating: which earlier turns reach the worker.

A synthetic 18-turn history across all personas, 18 follow-up requests, each labelled
with the turns it NEEDS (must pass), turns that are harmless to pass (acceptable), and
everything else counts as a leak (context poisoning). Compares:

  nouls-A   one noul per snippet, "contain information needed" wording (the original gate)
  nouls-B   one noul per snippet, "refers to / depends on" wording
  harness   jev.gate_context as shipped: Choice ranks the history, B noul on the leaders

Result (2026-09-27, semif): original gate passed 68% of needed turns; harness design at
threshold 0.20 passed 95% with 0 leaks at ~1.6s (vs ~3.1s for a noul per snippet).

    python evals/memory_cases.py            # live pass, then sweeps
    python evals/memory_cases.py --offline
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jevh.jev as jevmod  # noqa: E402
from jevh.config import load_config  # noqa: E402
from jevh.decide import SystemOneClient, noul  # noqa: E402
from jevh.jev import Jev, decide_memory  # noqa: E402
from jevh.memory import Turn  # noqa: E402
import _out  # noqa: E402  (evals/_out.py: where a run's results go)

HISTORY = [
    ("fitness_coach", "My current maxes: squat 140kg, bench 100kg, deadlift 180kg.", "Noted - I'll base your percentages on those."),
    ("fitness_coach", "I can train Monday, Wednesday and Friday, 60 minutes max.", "Three full-body sessions a week it is."),
    ("daily_assistant", "Remind me that Dave Miller's radon test is Thursday at 2pm at 430 Cleveland St.", "Added to your calendar for Thursday 2pm."),
    ("high_end_dev", "My payroll app is FastAPI + SQLite; the jobs table has id, worker, hours, rate, materials.", "Got it - I'll keep that schema in mind."),
    ("high_end_dev", "The POST /jobs route returns 500 when materials is empty.", "Fixed: materials now defaults to 0 before the insert."),
    ("front_end_designer", "Our brand colors are navy #1B2A4A and safety orange #F28C28, font is Inter.", "Saved as design tokens."),
    ("daily_assistant", "My wife's birthday is October 14; she loves pottery and hiking.", "I'll remind you a week before."),
    ("fitness_coach", "My left knee hurts on deep squats.", "Switch to box squats to parallel for two weeks and see a physio if it persists."),
    ("daily_assistant", "Eco Insulation quote for the Spanish Fork job: 2,400 sqft of R-30, $4,850 total.", "Drafted the quote PDF."),
    ("high_end_dev", "Deploy target is my Windows box; run the payroll app as a service on port 8040.", "I'll use NSSM to register it as a service."),
    ("comfyui_manager", "Generate a hero image of a snowy mountain at sunset.", "Done - saved as hero_mountain.png."),
    ("daily_assistant", "What's a good sourdough starter feeding ratio?", "1:1:1 starter, flour, water by weight."),
    ("fitness_coach", "My goal is to lose 5 kg by March while keeping my strength.", "A 300-400 kcal deficit with high protein fits that."),
    ("front_end_designer", "For the landing page: hero with the mountain image, three service cards, and a quote form.", "Layout locked in."),
    ("daily_assistant", "Kristopher at Insulation Specialists wants payroll by the 5th of each month.", "Noted the deadline."),
    ("high_end_dev", "I prefer no heavy frameworks - plain JS and the Python stdlib where possible.", "Understood, I'll keep dependencies minimal."),
    ("daily_assistant", "What should we watch tonight? Something like Interstellar.", "Try Arrival or Contact."),
    ("fitness_coach", "I'm vegetarian and aiming for 150g of protein a day.", "Lean on eggs, Greek yogurt, tofu, lentils and whey."),
]

# (request, needed turn ids, acceptable extra ids)
CASES = [
    ("Update my squat program - what should my working weight be for 5x5?", {0, 7}, {1, 12}),
    ("Plan my meals for tomorrow.", {12, 17}, {1}),
    ("Add a DELETE endpoint for jobs to the payroll app.", {3}, {4, 9, 15}),
    ("Why would POST /jobs still fail when hours is blank?", {3, 4}, {15}),
    ("How do I make the payroll app start automatically when Windows boots?", {9}, {3}),
    ("Build the landing page now.", {5, 13}, {10, 15}),
    ("Any gift ideas for my wife?", {6}, set()),
    ("When is the radon appointment, and where?", {2}, set()),
    ("Resend the Spanish Fork insulation quote with a 5% discount.", {8}, set()),
    ("When is payroll due to Kristopher?", {14}, set()),
    ("What's the capital of Utah?", set(), set()),
    ("Write a Python function that reverses a string.", set(), {15}),
    ("Make me a 3-day workout split.", {1}, {0, 7, 12}),
    ("Recommend another sci-fi movie for tonight.", {16}, set()),
    ("Put the mountain image in the site header.", {10}, {5, 13}),
    ("How much should I feed my sourdough starter?", {11}, set()),
    ("Tell me a joke.", set(), set()),
    ("Is my knee okay for heavy deadlifts?", {7}, {0}),
]

Q_A = "Does this earlier snippet contain information needed to handle the latest request?"
Q_B = jevmod.MEMORY_Q


def chunks() -> list[str]:
    return [Turn(p, u, a, "eval", 0.0).chunk() for p, u, a in HISTORY]


def run(cfg) -> dict:
    cl = SystemOneClient(cfg.jev["url"], cfg.jev["backend"])
    jev = Jev(cfg, cl)
    ch = chunks()
    out = {"nouls-A": [], "nouls-B": [], "harness": []}
    for text, _, _ in CASES:
        state = {"latest_request": text}
        for key, q in (("nouls-A", Q_A), ("nouls-B", Q_B)):
            t = time.time()
            r = cl.ask(state, {f"m{i}": noul({"question": q, "snippet": c}) for i, c in enumerate(ch)})
            out[key].append({"scores": [r[f"m{i}"].noul for i in range(len(ch))], "secs": time.time() - t})
        t = time.time()
        d = jev.gate_context(text, ch)
        out["harness"].append({"scores": d.scores, "secs": time.time() - t})
    return out


def evaluate(picks: list[set[int]]) -> dict:
    need = got = leaks = empty_ok = n_empty = exact = 0
    for (text, needed, ok), kept in zip(CASES, picks):
        need += len(needed)
        got += len(needed & kept)
        leak = kept - needed - ok
        leaks += len(leak)
        if not needed:
            n_empty += 1
            empty_ok += not leak
        exact += needed <= kept and not leak
    return {"recall": got / need, "leaks": leaks, "empty_ok": empty_ok / n_empty,
            "exact": exact / len(CASES), "objective": (got / need) * 0.5 + exact / len(CASES) * 0.5 - leaks * 0.01}


def pick_nouls(rows, t, cap):
    return [set(decide_memory(r["scores"], t, cap)) for r in rows]


def report(name, results, top=5):
    results.sort(key=lambda x: -x[0]["objective"])
    print(f"\n== {name}  (params | recall exact empty_ok leaks)")
    for s, params in results[:top]:
        print(f"  {params} | {s['recall']:.0%}  {s['exact']:.0%}  {s['empty_ok']:.0%}  {s['leaks']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    _out.add_args(ap)
    a = ap.parse_args()
    dest = _out.Out.of(a)
    cache = Path(__file__).with_name("memory_raw.json")
    cfg = load_config()
    if a.offline:
        data = json.loads(dest.read(cache).read_text(encoding="utf-8"))
    else:
        t = time.time()
        data = run(cfg)
        dest.write(cache).write_text(json.dumps(data, indent=1), encoding="utf-8")
        print(f"raw answers saved to {dest.where(dest.write(cache))}")
        print(f"{len(CASES)} requests x {len(HISTORY)} snippets in {time.time() - t:.0f}s")
    for k, rows in data.items():
        secs = sorted(r["secs"] for r in rows)
        print(f"  {k:8s} median {secs[len(secs) // 2]:.2f}s per request")
    cur = (cfg.t("memory"), cfg.lim("max_memory_chunks"))
    s = evaluate(pick_nouls(data["harness"], *cur))
    print(f"\nshipped config (harness, t={cur[0]}, cap={cur[1]}): recall {s['recall']:.0%} exact {s['exact']:.0%} "
          f"empty_ok {s['empty_ok']:.0%} leaks {s['leaks']}")
    for k in ("nouls-A", "nouls-B", "harness"):
        report(k, [(evaluate(pick_nouls(data[k], t, c)), f"t={t:.2f} cap={c}")
                   for t, c in itertools.product((0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.55, 0.7, 0.9), (2, 3, 4))])


if __name__ == "__main__":
    main()
