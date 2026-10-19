"""Live eval: does the new message continue the recent conversation? If yes, the recent
exchange goes to the worker as real context (not through the memory relevance gate), so an
answer to the assistant's own clarifying question is never read cold.

Missing a follow-up is the bad error (the user gets "your message looks cut off"); calling a
new topic a follow-up only adds one or two short turns of context.

    python evals/followup_cases.py
    python evals/followup_cases.py --offline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevh.config import load_config  # noqa: E402
from jevh.decide import SystemOneClient, noul  # noqa: E402
import _out  # noqa: E402  (evals/_out.py: where a run's results go)

RAW = Path(__file__).with_name("followup_cases_raw.json")
ASK = ("Quick question before I dive in - which country or state, and what kind of change: mail "
       "forwarding, driver's license, or accounts?")

CASES = [  # (previous user message, previous assistant reply, new message, continues?)
    ("could you search to figure out how i can make a change of address?", ASK,
     "I need to for my mailing and to get out of a gym membership", True),
    ("could you search to figure out how i can make a change of address?", ASK, "Utah, USPS mail", True),
    ("Plan my workouts for next week.", "Sure - how many days can you train, and what equipment do you have?",
     "4 days, full gym", True),
    ("Draft an email to my client about the delayed quote.", "Here's a draft: Subject: Quote update...",
     "make it shorter and friendlier", True),
    ("Make a hero image of a mountain at sunset.", "Done - saved hero_mountain.png.", "now one at sunrise", True),
    ("What's a good sourdough starter feeding ratio?", "1:1:1 starter, flour, water by weight.",
     "and how often?", True),
    ("Add a DELETE endpoint for jobs.", "Added DELETE /jobs/{id}. Want tests too?", "yes please", True),
    ("Find me flights to Denver next weekend.", "Which airport are you leaving from?", "SLC", True),
    ("Summarize this article about solar panels.", "It argues rooftop solar pays back in 7 years...",
     "what did it say about batteries?", True),
    ("Book a table at Luigi's on Friday.", "For how many people and what time?", "2 people at 7", True),
    ("What's on my calendar tomorrow?", "Two meetings: 9am standup and 2pm with Dave.",
     "move the Dave one to 3", True),
    ("Recommend a sci-fi movie.", "Try Arrival.", "something older", True),
    ("Write a function that reverses a string.", "def rev(s): return s[::-1]", "in JavaScript", True),
    ("How do I cancel my gym membership?", "Which gym is it?", "Anytime Fitness", True),
    ("Plan my workouts for next week.", "Here's a 4-day upper/lower plan...", "What's the capital of Utah?", False),
    ("Draft an email to my client about the delayed quote.", "Here's a draft: Subject: Quote update...",
     "turn off the living room lights", False),
    ("What's a good sourdough starter feeding ratio?", "1:1:1 by weight.", "fix the failing tests in my repo", False),
    ("Make a hero image of a mountain at sunset.", "Done - saved hero_mountain.png.",
     "what's 17% of 2,340?", False),
    ("Find me flights to Denver next weekend.", "Here are three options...", "tell me a joke", False),
    ("Recommend a sci-fi movie.", "Try Arrival.", "remind me to call Dave at 3pm", False),
    ("What's on my calendar tomorrow?", "Two meetings.", "how do I center a div in CSS?", False),
    ("How do I cancel my gym membership?", "Call them or send a written notice...",
     "write a haiku about coffee", False),
]

PHRASINGS = {
    "topic": "Is the latest request a continuation of the recent conversation's topic?",
    "depends": ("Does the latest message answer, follow up on, or only make sense together with the recent "
                "conversation?"),
    "reply": "Is the latest message a reply to the assistant's last message, or a follow-up to it?",
}


def run_live(path: Path | None = None) -> dict:
    path = path or RAW
    cfg = load_config()
    c = SystemOneClient(cfg.jev["url"], cfg.jev["backend"], timeout=300)
    raw = {k: [] for k in PHRASINGS}
    for pu, pa, new, _ in CASES:
        state = {"latest_request": new, "recent_conversation": f"User: {pu}\nAssistant: {pa}"}
        r = c.ask(state, {k: noul(q) for k, q in PHRASINGS.items()})
        for k in PHRASINGS:
            raw[k].append(r[k].noul)
    path.write_text(json.dumps(raw, indent=1), encoding="utf-8")
    return raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    _out.add_args(ap)
    a = ap.parse_args()
    dest = _out.Out.of(a)
    if a.offline:
        raw = json.loads(dest.read(RAW).read_text(encoding="utf-8"))
    else:
        raw = run_live(dest.write(RAW))
        print(f"raw scores saved to {dest.where(dest.write(RAW))}")
    labels = [c[3] for c in CASES]
    for k, scores in raw.items():
        print(f"\n== {k}: {PHRASINGS[k]}")
        print("  " + "  ".join(f"{'+' if l else '-'}{s:.2f}" for s, l in zip(scores, labels)))
        for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            caught = sum(s >= t and l for s, l in zip(scores, labels))
            false = sum(s >= t and not l for s, l in zip(scores, labels))
            print(f"  t={t:.1f}  follow-ups caught {caught}/{sum(labels)}  new topics joined {false}/{labels.count(False)}")


if __name__ == "__main__":
    main()
