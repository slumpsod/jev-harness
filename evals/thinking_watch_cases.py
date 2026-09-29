"""When should the thinking watch pause a model's long think and have it test its idea in code?

Positives ("pause"): snapshots of real Qwen3.8 thinking from the ARC bench (evals/bench, the
thinking/step-NN.txt files) at points where it held a concrete, checkable idea it was working
through by hand, plus non-ARC equivalents. Negatives ("keep thinking"): reasoning where running
code can't help (writing, advice, intent, design before data) and exploration with no idea yet.

    python evals/thinking_watch_cases.py            # score TESTABLE_Q on every case, sweep thresholds
    python evals/thinking_watch_cases.py --holdout  # the cases written after choosing (thresholds.testable = 0.3)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from jevh.config import load_config  # noqa: E402
from jevh.decide import noul  # noqa: E402
from jevh.jev import Jev  # noqa: E402
from jevh.orchestrator import TESTABLE_Q  # noqa: E402

ARC_TASK = ("Solve this ARC-AGI puzzle... work out the rule, write it as a Python function and check that it "
            "reproduces EVERY train output exactly, then apply it to the test input.")

# (task, thinking tail, should pause?)
CASES = [
    # --- real ARC thinking (Qwen3.8-27B), where code would settle it ---
    (ARC_TASK, """(3,4): to 9: sqrt(2.25+16)≈4.27. To 6: sqrt(1+16)≈4.12. -> 6 ✓
(3,5): to 9: sqrt(2.25+25)≈5.22. To 6: sqrt(1+25)≈5.10. -> 6 ✓
(4,3): to 9: sqrt(6.25+9)≈3.91. To 6: sqrt(0+9)=3. -> 6 ✓
(4,4): to 9: sqrt(6.25+16)≈4.72. To 6: 4. -> 6 ✓
(5,3): to 6: sqrt(1+9)≈3.16. To 4: sqrt(4+9)≈3.61. -> 6 ✓
So far the rule "each 5 takes the colour of the nearest bar by Euclidean distance to the bar's centroid" holds.""", True),
    (ARC_TASK, """In the test, there are gaps: 2 is at row 2, then a gap at row 3, then 3 is at rows 4-6. So the
boundary between 2 and 3 is at row 3 (the gap row). The band rule assigns the gap row to the lower bar (3),
but the centroid rule assigns it to 2. Both rules fit the training data. I need more evidence to disambiguate.""", True),
    (ARC_TASK, """7. 3x1, color 6 (transposed)
8. 3x1, color 6 (transposed)
9. 1x3, color 3 (original)
10. 1x3, color 3 (original)
11. 3x1, color 6 (transposed)
So the pattern is: pairs of transposed blocks (color 6), then pairs of original blocks (color 3), then pairs
of transposed blocks (color 6), etc. Each ring is offset from the previous one by the block's size.""", True),
    (ARC_TASK, """What if the reflection is across the vertical center line of the grid (c -> 15-c)? I get a bar at
row 7, cols 9-11. But the output has no bar at row 7, cols 9-11. What if the reflection is across the horizontal
center (r -> 15-r)? I get a bar at row 8, cols 4-6. That's not in the output either. What about rotating 90°?""", True),
    (ARC_TASK, """So it's one big square spiral loop! Starting from J (ring 3 bottom-left) and going clockwise:
J -> D (ring 2 left, up) -> A (up the left side) -> F (ring 1 top-left) -> C (ring 1 left, down) -> I (ring 1
bottom-right) -> E (ring 2 right, down) -> K (ring 3 bottom-right) -> L (ring 4 bottom-right). Block sizes
alternate between the seed's shape and its transpose, and colours alternate between 6 and 3.""", True),
    (ARC_TASK, """B3 (3.5, 8): dx=-3, dy=-7.5 -> atan2(-3, 7.5) = -21.8° -> 338.2°.
B5 (9.5, 3): dx=-8, dy=-1.5 -> atan2(-8, 1.5) = -79.4° -> 280.6°.
B11 (1, 12.5): dx=1.5, dy=-10 -> atan2(1.5, 10) = 8.5°.
If the blocks are placed in order of angle around the seed, sorting these should give the spiral order.""", True),
    # --- non-ARC: a concrete idea code would check ---
    ("Why did revenue drop in the Q1 sales export (sales.csv)?", """Looking at the columns: date, sku, qty, price,
returned. I think the March dip is from returns rather than fewer orders - the returned column looked high in the
rows I saw. If I group by month and sum qty*price with and without returned rows, the gap should show up in
March only.""", True),
    ("Fix the failing test in tipcalc.py", """The failing assertion is 28.74 == 28.75. That's a rounding issue:
round(with_tip / people, 2) rounds half to even, so 28.745 becomes 28.74. If I use Decimal with ROUND_HALF_UP
the split should give 28.75. I can check that by computing the split for the test's inputs.""", True),
    ("What's the sum of the squares of 1 to 20?", """The formula for the sum of squares is n(n+1)(2n+1)/6. For
n=20: 20*21*41/6 = 17220/6 = 2870. Let me double check 20*21 = 420, 420*41 = 17220, /6 = 2870.""", True),
    # --- where code can't help: keep thinking ---
    ("Write a polite but firm email to my landlord about the heater that's been broken for two weeks.",
     """The user wants an email to their landlord. Tone: polite but firm. Key points: the heater has been broken
two weeks, it's getting cold, they want a repair date. I should open with a friendly line, state the problem
and how long it's been, mention I reported it before, then ask for a specific repair date this week.""", False),
    ("Should I spend my 5 days in Japan in Kyoto or split between Tokyo and Osaka?", """The user likes food and
history, dislikes crowds. Kyoto is strong on history but very crowded at the big temples; Osaka is the food
capital; Tokyo is huge and would eat travel time. With only 5 days, splitting three cities means a lot of
transit. Maybe Kyoto with a day trip to Osaka balances history and food.""", False),
    ("i want you to book it for me", """The user says "book it". The previous turn listed two United options,
$1,100 with one stop and $1,126 with two stops, and I suggested the first. They probably mean the $1,100 one,
but the dates were a 2-night trip which seemed wrong. I should confirm the dates and which flight before
starting any booking.""", False),
    ("Build a landing page for my bakery", """Structure first: a hero section with the bakery name and a photo,
then a menu section with the best sellers, opening hours, a location map, and a contact form. The brand should
feel warm - cream and brown tones, a serif heading font. I'll keep it one page with anchor links.""", False),
    ("Make me a 4-week beginner strength plan, 3 days a week", """For a beginner, full-body three times a week
works best. Compound lifts: squat, bench, row, deadlift, overhead press. Start light and add weight each week.
Sets of 3x8 are reasonable. Day A and Day B alternating, so each lift gets practiced twice a week.""", False),
    ("Summarize this article for me", """The article argues that remote work increased productivity for focused
tasks but hurt mentoring of junior staff. The author's main evidence is a survey of 2,000 workers. I should lead
with the main claim, give the two findings, and note the author's caveat about self-reported data.""", False),
    ("What does this poem mean?", """The poem keeps returning to the image of a closed door. In the first stanza
it's literal, the speaker's childhood home. By the last stanza the door stands for the speaker's father, and
the silence behind it. The turn happens in the third stanza when the key appears.""", False),
    (ARC_TASK, """Let me start by loading the puzzle file and examining it. I don't know the format of the
grids yet or how many training examples there are, so first I'll print the shapes.""", False),
]


def main() -> None:
    cfg = load_config()
    jev = Jev(cfg)
    rows = []
    for task, thinking, want in (HOLDOUT if "--holdout" in sys.argv else CASES):
        p = jev._ask("thinking_watch", {"task": task, "thinking": thinking}, {"testable": noul(TESTABLE_Q)})["testable"].noul
        rows.append((p, want, thinking[:70].replace("\n", " ")))
        print(f"{p:.3f} {'PAUSE' if want else 'keep '}  {rows[-1][2]}", flush=True)
    pos = sorted(p for p, w, _ in rows if w)
    neg = sorted(p for p, w, _ in rows if not w)
    print(f"\npause cases: min {pos[0]:.3f}  {[round(x, 2) for x in pos]}")
    print(f"keep cases:  max {neg[-1]:.3f}  {[round(x, 2) for x in neg]}")
    for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        print(f"  t={t}: pauses caught {sum(p >= t for p in pos)}/{len(pos)}, "
              f"wrong pauses {sum(p >= t for p in neg)}/{len(neg)}")



# Written after TESTABLE_Q and the 0.3 threshold were chosen on CASES; run once (--holdout).
HOLDOUT = [
    (ARC_TASK, """Maybe the output is the input rotated 90 degrees clockwise. Row 0 of the output is 3 1 4. Column 0
of the input read bottom-up is 4 1 3 - no, that's reversed. Read top-down it's 3 1 4, so it might be a transpose
instead. Row 1 of the output is 0 5 9; column 1 of the input top-down is 0 5 9. Yes, transpose so far.""", True),
    ("Import the dates from bookings.csv", """Most dates look like 03/04/2026, which could be DD/MM or MM/DD. A few
rows have 25/12/2025, so the day comes first. If I parse every row as DD/MM/YYYY and count failures, a clean zero
would confirm it; any failure would show the rows that are in a different format.""", True),
    ("Why does the export only show 50 customers?", """The customer count came back as exactly 50 both times
I looked, which smells like pagination: the API probably returns 50 per page and I only read the first. The
response might have a next_page or cursor field. Fetching page 2 would confirm it.""", True),
    ("What's the next number: 2, 5, 14, 41, ...?", """Differences: 3, 9, 27 - powers of 3. So each term is the
previous times 3 minus 1: 2*3-1 = 5, 5*3-1 = 14, 14*3-1 = 41, so next is 41*3-1 = 122.""", True),
    ("Write a birthday message for my mom", """She's turning 60 and loves her garden, especially her roses. The
message should be warm and a little funny - she likes teasing. Mention the garden, thank her for always being
there, and end with a plan to visit next weekend.""", False),
    ("Should I take the startup offer or stay at my current job?", """They value stability because they just
bought a house, but they said they feel bored. The startup pays 15% more but the equity is uncertain. The key
question is their risk tolerance given the mortgage - I should lay out both sides and ask what matters most.""",
     False),
    ("Explain the TCP handshake to a beginner", """I'll use an analogy: a phone call where each side confirms
they can hear the other. SYN is 'can you hear me?', SYN-ACK is 'yes, can you hear me?', ACK is 'yes'. Then
explain why three steps and not two, and keep jargon to a minimum.""", False),
    (ARC_TASK, """I've printed the three training pairs. Before guessing anything I want to understand what's in
them: which colours appear, whether the output is the same size as the input, and whether anything moves.""",
     False),
]


if __name__ == "__main__":
    main()
