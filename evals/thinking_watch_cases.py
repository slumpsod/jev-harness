"""When should the thinking watch pause a model's long think and have it test its idea in code?

Positives ("pause"): snapshots of real Qwen3.8 thinking from the ARC bench (evals/bench, the
thinking/step-NN.txt files) at points where it held a concrete, checkable idea it was working
through by hand, plus non-ARC equivalents. Negatives ("keep thinking"): reasoning where running
code can't help (writing, advice, intent, design before data) and exploration with no idea yet.

    python evals/thinking_watch_cases.py            # score TESTABLE_Q on every case, sweep thresholds
    python evals/thinking_watch_cases.py --holdout  # written after the first wording was chosen (now seen)

In the harness the watch only runs once a task has executed code, so the writing/advice negatives
here are a stricter test than it faces live (they'd never be watched).
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

# Live watch test (2026-09-28, puzzle 3): what Jev saw at 6 KB and 10 KB and let through (0.20, 0.09).
# Both read block positions/sizes off printed grids by eye - and misread a 2x4 block as 2x3, which
# then produced a wrong theory ("rotated and shrunk by 1"). Code would have listed them exactly.
CASES += [
    (ARC_TASK, " outward.\n\nRing 1 (immediately around the original block, rows 5–8, columns 6–7):\n- Top: rows 3–4, columns 8–10 (2x3 of 3s) and columns 2–4 (2x3 of 3s)? Hmm, two blocks on top: one at columns 2–4, one at columns 8–10. The original is at columns 6–7. So top side: a block to the left of the top (columns 2–4), a block to the right of the top (columns 8–10).\n\n- Bottom: rows 9–10, columns 2–4 and columns 8–10 (2x3 of 3s). Same as top.\n- Left: rows 0–2, columns 0–1 (3x2 of 6s) and rows 11–14, columns 0–1 (4x2 of 6s).\n- Right: rows 0–2, columns 12–13 (3x2 of 6s) and rows 11–14, columns 12–13 (4x2 of 6s).\n\nSo ring 1 consists of: 2 top 3-blocks, 2 bottom 3-blocks, 2 left 6-blocks, 2 right 6-blocks. Then ring 2 (further out): rows 15–16, columns 14–16 (2x3 of 3s), rows 17–20, columns 18–19 (4x2 of 6s), rows 21–22, columns 19–21 (2x3 of 3s). This is the bottom-right corner region — a continuation of the spiral.\n\nSo the pattern is a square spiral: starting from the original block, going clockwise (or counterclockwise) around it, with alternating block types: top side is 3-blocks (horizontal), right side is 6-blocks (vertical), bottom side is 3-blocks, left side is 6-blocks... but each side has 2 blocks, one at each corner?\n\nHmm wait. Let me look at ring 1 more carefully:\n\nThe original occupies rows 5–8 (4 rows), columns 6–7 (2 columns).\n\nTop blocks (3s, 2x3): rows 3–4, columns 2–4 and rows 3–4, columns 8–10. Distance from the original: 1 row gap (row 5 is the top edge of the original, so rows 3–4 are 1 row above... no wait, the top edge of the original is row 5, so rows 3–4 are adjacent (row 4 is adjacent to row 5). Columns 2–4: the original is at columns 6–7; gap at column 5. Columns 8–10: adjacent to the right.\n\nRight blocks (6s): rows 0–2, columns 12–13 (3x2) and rows 11–14, columns 12–13 (4x2). Distance: 4-column gap (8,9,10,11 are empty... no wait, the original's right edge is column 7, so columns 8–11 are empty, blocks at columns 12–13). Rows 0–2: 2 rows above the top edge of the original (row 5), with row... hmm, rows 0,1,2 and the top edge of the original is row 5; rows 3–4 are occupied by the top 3-blocks. Rows 11–14: below the original (bottom edge is row 8), gap at rows 9–10 (occupied by the bottom 3-blocks).\n\nBottom blocks (3s): rows 9–10, columns 2–4 and columns 8–10.\n\nLeft blocks (6s): rows 0–2, columns 0–1 (3x2) and rows 11–14, columns 0–1 (4x2).\n\nSo ring 1 forms a square ring around the original block, with gaps: the ring extends from rows 0–14, column", True),
    (ARC_TASK, '- Ring 2 bottom 3-block: (16, 16).\n- Ring 2 right 6-block: (20, 19).\n- Ring 2 bottom 3-block: (22, 21).\n\nSo the ring corners: the corners of ring 1 are at (0,1), (0,13), (14,1), (14,13) as the outer corners of the 6-blocks; the inner corners are the 3-blocks at (3,4)... hmm wait.\n\nActually the ring is: the left/right sides are 6-blocks (vertical), the top/bottom sides are 3-blocks (horizontal). At each corner of the ring there are both a vertical 6-block and a horizontal 3-block.\n\nLet me reconsider: maybe it\'s a "spiral of rectangles" where each arm is one block, and the path goes:\n\nOriginal (5–8, 6–7)\n→ Up: top-right 3-block (3–4, 8–10)? That\'s up and to the right.\n\nHmm, this is tricky. Instead let me compare the relative geometry between training examples.\n\nTraining example 1:\n- Input: 1x3 of 3s at row 7, columns 7–9. Wait: row 7 = "0000000333000000" so columns 7,8,9 are 3s. A 1x3 horizontal block of 3s.\n- Output: in addition to the original, the following appear:\n  - Rows 0–2, columns 2 and 13: 6s (3x1)\n  - Row 3: columns 3–5 and 11–13 are 3s (1x3)\n  - Rows 4–6, columns 6 and 9: 6s (3x1)\n  - Row 7: original 1x3 of 3s at columns 7–9\n  - Rows 8–10, columns 6 and 9: 6s (3x1)\n  - Row 11: columns 3–5 and 11–13 are 3s (1x3)\n  - Rows 12–14, columns 2 and 13: 6s (3x1)\n  - Row 15: columns 0–2 and 15 are 3s (1x3, partially cut off at the edge)\n\nSo the original is a 1x3 of 3s (horizontal). The "other" blocks: 6s (vertical 3x1) and 3s (horizontal 1x3).\n\nInteresting: here the original color is 3, and the new blocks use both 3 and 6.\n\nIn training example 0, the original is 6, and the new blocks use 6 and 3.\nIn training example 2, the original is 3 (a 5x2 of 3s at rows 5–9, columns 10–11), and the new blocks: 3s (2x3 horizontal) and 6s (6x2 vertical... let me check).\n\nOutput 2:\n- Rows 0–2, columns 4–5 and 18–19: 3s (3x2)\n- Rows 3–4, columns 6–10 and 12–16: 6s (2x5)\n- Rows 5–9, columns 10–11: 3s (5x2) — the original\n- Rows 10–11, columns 6–10 and 12–16: 6s (2x5)\n- Rows 12–16, columns 4–5 and 18–19: 3s (5x2)\n- Rows 17–18, columns 0–3 and 20–24: 6s (2x4)\n- Rows 19–23, column 22: 3s (5x1)\n- Rows 24–25: nothing (empty)\n\nLet me re-read the output rows of training 2:\nRow 0: 00003300000000000033000000 → columns 4–5 are 3, columns 18–19 are 3\nRow 1: same\nRow 2: same\nRow 3: 00000066666006666600000000 → columns 6–10 are 6, columns 12–16 are 6\nRow 4: same\nRow 5: 00000000000330000000000000 → columns 10–11 are 3 (original)\nRow 6: same\nRow 7: same\nRow 8: same\nRow 9: same\nRow 10: 000', True),
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
    # relabelled True when the question widened to reading data by hand: listing colours and sizes by eye
    # is exactly what a line of code does (and what went wrong live)
    (ARC_TASK, """I've printed the three training pairs. Before guessing anything I want to understand what's in
them: which colours appear, whether the output is the same size as the input, and whether anything moves.""",
     True),
]


if __name__ == "__main__":
    main()
