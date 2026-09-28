"""Does the final() answer check see the evidence deep inside long tool outputs?

The gate's work log used to keep only the first 350 chars of each output, so a correct price
found 600 chars into a fetched page read as made up (live: Porch & Patio, 2026-09-28). Each
case is a transcript plus an answer, labelled ok (the output supports it) or bad (a number or
fact the output contradicts or never shows). Runs the real Verifier on the old log (no
answer) and the new one (work_log(..., answer=...)).

    python evals/verify_evidence_cases.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from jevh.config import load_config  # noqa: E402
from jevh.jev import Jev  # noqa: E402
from jevh.learn import seen_text, work_log  # noqa: E402
from jevh.verify import Verifier  # noqa: E402

PORCH = json.loads((ROOT / "verify_evidence_raw.json").read_text(encoding="utf-8"))["porch"]


def convo(request: str, steps: list[tuple[str, str]]) -> list[dict]:
    msgs = [{"role": "system", "content": "(system)"}, {"role": "user", "content": request}]
    for code, out in steps:
        msgs += [{"role": "assistant", "content": f"Checking.\n\n```python\n{code}\n```"},
                 {"role": "user", "content": "Output:\n" + out}]
    return msgs


NAV = "\n".join(["Skip to main content", "Sign in", "Deals", "Help", "Accept cookies", "Decline"] +
                [f"Menu item {i}" for i in range(40)])
FLIGHTS = convo("find the cheapest flight from Salt Lake City to Tokyo on October 6", [(
    "print(browser.goto('https://flights.example.com/slc-tyo?d=2026-10-06')['page'])",
    NAV + "\nResults for SLC to TYO, Tue Oct 6\nSort: cheapest\nDelta 1 stop (SEA) 16h 5m\n$1,284\n"
          "United 1 stop (SFO) 14h 54m\n$1,317\nANA 1 stop (LAX) 15h 40m\n$1,402\nPrices include taxes")])
ORDER = convo("when will my headphones order arrive?", [(
    "print(browser.read())",
    NAV + "\nYour orders\nOrder #114-2233\nSony WH-1000XM6 headphones\nShipped\n"
          "Arriving Thursday, October 1\nTrack package\nOrder #114-1990\nPhone case\nDelivered Sep 12")])
RECIPE = convo("how long do I bake the sourdough and at what temperature?", [(
    "print(web.fetch('https://bread.example.com/sourdough')[:3000])",
    NAV + "\nClassic sourdough\nJump to recipe\nMy grandmother's story... " + "Lorem ipsum. " * 40 +
    "\nBake covered at 250C for 20 minutes, then uncovered at 230C for 25 minutes.\nCool 1 hour.")])

PORCH_REQ = "okay now can you go to porch and patio in Lindon Utah and find out how much there Fairview 4pc lounge set is?"
CASES = [  # (name, request, transcript, answer, label)
    ("porch-right", PORCH_REQ, PORCH,
     "The Fairview 4pc Lounge Set at Porch & Patio in Lindon, UT is $2,999.99 (40% off, regular $4,999.99). "
     "It comes in Light and Dark finishes; the Light version is sold out and the Dark version has 2 units left.", "ok"),
    ("porch-wrong", PORCH_REQ, PORCH,
     "The Fairview 4pc Lounge Set at Porch & Patio in Lindon, UT is $2,499.99 (50% off, regular $4,999.99), "
     "in stock in both Light and Dark.", "bad"),
    ("flight-right", FLIGHTS[1]["content"], FLIGHTS,
     "The cheapest is Delta at $1,284 with 1 stop in Seattle (16h 5m), taxes included. United is $1,317.", "ok"),
    ("flight-wrong", FLIGHTS[1]["content"], FLIGHTS,
     "The cheapest is United at $1,099 with 1 stop in San Francisco (14h 54m).", "bad"),
    ("order-right", ORDER[1]["content"], ORDER,
     "Your Sony WH-1000XM6 headphones (order #114-2233) have shipped and arrive Thursday, October 1.", "ok"),
    ("order-wrong", ORDER[1]["content"], ORDER,
     "Your Sony WH-1000XM6 headphones were delivered on September 12.", "bad"),
    ("recipe-right", RECIPE[1]["content"], RECIPE,
     "Bake it covered at 250C for 20 minutes, then uncovered at 230C for another 25 minutes, and let it cool an hour.",
     "ok"),
    ("recipe-wrong", RECIPE[1]["content"], RECIPE,
     "Bake it at 200C for 45 minutes uncovered.", "bad"),
]


def main() -> None:
    cfg = load_config()
    v = Verifier(Jev(cfg), cfg)
    tally = {"old": [0, 0], "new": [0, 0]}   # [correct, false alarms on ok answers]
    for name, req, transcript, answer, label in CASES:
        row = []
        for mode in ("old", "new"):
            log = work_log(transcript) if mode == "old" else work_log(transcript, answer=answer)
            verdict = (v.check(req, answer, [], log) if mode == "old"
                       else v.check(req, answer, [], log, evidence=seen_text(transcript)))
            said = "ok" if verdict.ok else "bad"
            tally[mode][0] += said == label
            tally[mode][1] += label == "ok" and said == "bad"
            s = verdict.scores
            row.append(f"{mode}={said:3} (acc {s.get('accurate', 0):.2f} claims {s.get('claims', 0):.2f})")
        print(f"{name:13} label={label:3}  " + "   ".join(row), flush=True)
    n = len(CASES)
    for mode, (ok, fa) in tally.items():
        print(f"{mode}: {ok}/{n} right, {fa} false alarms on correct answers")


if __name__ == "__main__":
    main()
