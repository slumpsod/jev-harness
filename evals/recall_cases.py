"""Live eval: can memory reach OLD turns? The 18-turn history of memory_cases.py is buried
under 40 newer, unrelated turns, then the same follow-up requests are asked.

  window    candidates = persona's last 20 turns + 6 from other personas (the old design)
  search    window + 10 turns from any time that share words with the request (full-text)

`python evals/recall_cases.py 4` buries it under 160 turns. Both go through the shipped Jev gate (gate_context). Reported: needed turns that were
reachable at all, needed turns that reached the worker, and leaks (filler or unrelated turns).

    python evals/recall_cases.py
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_cases import CASES, HISTORY  # noqa: E402

from jevh.config import load_config  # noqa: E402
from jevh.jev import Jev  # noqa: E402
from jevh.memory import Memory, Turn  # noqa: E402

SCALE = int(sys.argv[1]) if len(sys.argv) > 1 else 1   # repeat the filler: 4 -> 160 newer turns
FILLER = [
    ("daily_assistant", "What's 15% tip on $86?", "$12.90."),
    ("daily_assistant", "Turn on the porch light.", "Done."),
    ("daily_assistant", "Spell 'necessary'.", "N-E-C-E-S-S-A-R-Y."),
    ("daily_assistant", "Convert 30 miles to km.", "48.3 km."),
    ("daily_assistant", "Tell me a fun fact about octopuses.", "They have three hearts."),
    ("daily_assistant", "What day of the week is Christmas this year?", "A Friday."),
    ("daily_assistant", "Suggest a name for a grey cat.", "Smokey, Ash or Pebble."),
    ("daily_assistant", "How long do I boil an egg for a runny yolk?", "About 6 minutes."),
    ("daily_assistant", "What's the capital of Australia?", "Canberra."),
    ("daily_assistant", "Give me a synonym for 'happy'.", "Cheerful."),
    ("high_end_dev", "What does `git stash pop` do?", "Re-applies the latest stash and drops it."),
    ("high_end_dev", "Regex for a US zip code?", r"^\d{5}(-\d{4})?$"),
    ("high_end_dev", "Explain Python's GIL in one line.", "One thread runs Python bytecode at a time."),
    ("high_end_dev", "How do I list open ports on Windows?", "netstat -ano."),
    ("high_end_dev", "What's the difference between let and const?", "const can't be reassigned."),
    ("high_end_dev", "Sort a list of dicts by a key in Python?", "sorted(xs, key=lambda d: d['k'])."),
    ("front_end_designer", "What's a good line height for body text?", "About 1.5."),
    ("front_end_designer", "Serif or sans-serif for a bakery menu?", "A warm serif."),
    ("front_end_designer", "How big should a mobile tap target be?", "At least 44x44 px."),
    ("front_end_designer", "Suggest a pastel palette.", "#FFD1DC, #BDE0FE, #CDEAC0."),
    ("fitness_coach", "How many steps a day is healthy?", "7,000-10,000 is a good target."),
    ("fitness_coach", "Is stretching before running useful?", "Dynamic warm-ups beat static stretching."),
    ("fitness_coach", "Best time of day to train?", "Whenever you can be consistent."),
    ("fitness_coach", "How much water should I drink?", "Around 2-3 litres, more when training."),
    ("comfyui_manager", "Make a cartoon robot sticker.", "Saved robot_sticker.png."),
    ("comfyui_manager", "Generate a watercolor fox.", "Saved fox_watercolor.png."),
    ("daily_assistant", "Weather tomorrow?", "Cloudy, 12C."),
    ("daily_assistant", "Recommend a podcast about history.", "Try 'Hardcore History'."),
    ("daily_assistant", "What rhymes with orange?", "Almost nothing - 'door hinge' if you squint."),
    ("daily_assistant", "Set a timer for 10 minutes.", "Timer set."),
    ("high_end_dev", "What's a good commit message style?", "Imperative mood, short subject line."),
    ("high_end_dev", "How do I check Node's version?", "node --version."),
    ("front_end_designer", "Rounded or square buttons?", "Slightly rounded (6-8px) is friendly and modern."),
    ("fitness_coach", "Is creatine safe?", "Yes, 3-5 g a day is well studied."),
    ("daily_assistant", "How many ounces in a cup?", "8 fluid ounces."),
    ("daily_assistant", "Remind me to water the plants.", "Reminder set."),
    ("daily_assistant", "What's the speed of light?", "About 300,000 km/s."),
    ("high_end_dev", "What's a Python f-string?", "An inline-formatted string literal."),
    ("front_end_designer", "Best free icon set?", "Lucide or Heroicons."),
    ("fitness_coach", "How long to rest between sets?", "2-3 minutes for compounds."),
]


def main() -> None:
    cfg = load_config()
    jev = Jev(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        mem = Memory(Path(tmp))
        now = time.time()
        for i, (p, u, a) in enumerate(HISTORY):          # a month ago
            mem._insert(Turn(p, u, a, "old", now - 30 * 86400 + i))
        for i, (p, u, a) in enumerate(FILLER * SCALE):   # the last few days
            mem._insert(Turn(p, u, a, "new", now - 3 * 86400 + i * 60))
        mem.db.commit()
        hist = {Turn(p, u, a, "", 0).chunk(): i for i, (p, u, a) in enumerate(HISTORY)}
        totals = {}
        for mode, n_search in (("window", 0), ("search", 10)):
            reach = passed = need = leaks = 0
            t0 = time.time()
            for text, needed, ok in CASES:
                persona = HISTORY[min(needed)][0] if needed else "daily_assistant"
                cands = mem.candidates(persona, 20, 6, text, n_search, 0)
                ids = {hist[c] for c in cands if c in hist}
                kept = jev.gate_context(text, cands).kept
                kept_ids = {hist[c] for c in kept if c in hist}
                need += len(needed)
                reach += len(needed & ids)
                passed += len(needed & kept_ids)
                leaks += len(kept_ids - needed - ok) + sum(c not in hist for c in kept)
            totals[mode] = (reach, passed, need, leaks, (time.time() - t0) / len(CASES))
        mem.close()
        for mode, (reach, passed, need, leaks, secs) in totals.items():
            print(f"{mode:7} reachable {reach}/{need}  reached the worker {passed}/{need}  leaks {leaks}  "
                  f"{secs:.1f}s per request")


if __name__ == "__main__":
    main()
