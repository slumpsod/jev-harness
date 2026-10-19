"""Live eval of the learning gates (jevh/learn.py): each is one Jev noul that decides whether
the model gets called at all, so false positives cost an LLM call and pollute memory, and false
negatives lose something worth keeping.

    python evals/learn_cases.py            # live, sweep
    python evals/learn_cases.py --offline  # re-sweep the cached raw answers
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevh.config import load_config  # noqa: E402
from jevh.decide import SystemOneClient, noul  # noqa: E402
from jevh.learn import FACT_Q, SAME_FACT_Q, SKILL_Q  # noqa: E402
import _out  # noqa: E402  (evals/_out.py: where a run's results go)

RAW = Path(__file__).with_name("learn_cases_raw.json")

FACTS = [  # (user message, assistant reply, states a lasting fact?)
    ("My gym is closed on Sundays, so plan my week around that.", "Here's a Mon-Sat plan...", True),
    ("I'm vegetarian - any quick dinner ideas?", "Try a chickpea curry...", True),
    ("My daughter Ava turns 7 on May 3rd, help me plan a party.", "Here are party ideas for a 7-year-old...", True),
    ("I run my insulation business out of Salt Lake City. What should a basic quote include?", "A quote should list...", True),
    ("I've got a 3090 and a 4080 in my PC - which should run the image model?", "Put the image model on the 3090...", True),
    ("Please always use metric units with me.", "Got it, metric from now on.", True),
    ("My dog Max is allergic to chicken. What treats are safe?", "Try salmon or lamb-based treats...", True),
    ("I'm training for a half marathon in November.", "Great - here's a 10-week build...", True),
    ("At work we use FastAPI and Postgres. How do I add a migration?", "Use Alembic: ...", True),
    ("Never book my meetings before 10am.", "Understood, 10am or later.", True),
    ("My wife Sarah handles our payroll, so send payroll questions to her.", "Noted.", True),
    ("I'm allergic to penicillin, what should I tell the dentist?", "Tell them about the allergy...", True),
    ("What's 17% of 2,340?", "397.80", False),
    ("Write a haiku about coffee.", "Dark steam rising slow...", False),
    ("Turn off the living room lights.", "Done.", False),
    ("What's the weather tomorrow?", "Sunny, 21C.", False),
    ("Fix this: TypeError: 'NoneType' object is not subscriptable on line 12.", "The dict lookup returns None...", False),
    ("Translate 'good morning' into Spanish.", "Buenos dias.", False),
    ("Make an image of a cat astronaut.", "Here's your image.", False),
    ("How do I center a div in CSS?", "Use flexbox: ...", False),
    ("Tell me a joke.", "Why did the scarecrow...", False),
    ("Remind me to call Dave at 3pm.", "Reminder set for 3pm.", False),
    ("Draft an email to my client about the delayed quote.", "Subject: Quote update...", False),
    ("What's on my calendar tomorrow?", "You have two meetings...", False),
    ("Summarize this article about solar panels.", "The article argues...", False),
]

SAME = [  # (new fact, old fact, should the old one be retired?)
    ("Jed's gym is closed on Sundays.", "Jed's gym closes on Sundays.", True),
    ("Jed now trains at Anytime Fitness.", "Jed trains at Gold's Gym.", True),
    ("Jed's daughter Ava turns 8 on May 3.", "Jed's daughter Ava turns 7 on May 3.", True),
    ("Jed prefers metric units.", "Jed prefers imperial units.", True),
    ("Jed's ComfyUI runs on port 8190.", "Jed's ComfyUI runs on port 8188.", True),
    ("Jed's dog is named Max.", "Jed's daughter is named Ava.", False),
    ("Jed prefers metric units.", "Jed uses FastAPI at work.", False),
    ("Jed's wife Sarah handles payroll.", "Jed's business is in Salt Lake City.", False),
    ("Jed is allergic to penicillin.", "Jed's dog Max is allergic to chicken.", False),
    ("Jed is training for a half marathon in November.", "Jed's gym is closed on Sundays.", False),
]

LOG_API = """CODE:
wf = comfy.load("Qwen Image 2.1 workflows/t2i.json"); comfy.set(wf, {"4.ckpt_name": "qwen_image.safetensors"})
OUTPUT: RuntimeError: ComfyUI 400: value_not_in_list ckpt_name 'qwen_image.safetensors'
CODE:
print([m for m in comfy.models("checkpoints") if "qwen" in m])
OUTPUT: ['Qwen/qwen_image_2.1_fp8.safetensors']
CODE:
comfy.set(wf, {"4.ckpt_name": "Qwen/qwen_image_2.1_fp8.safetensors", "6.text": prompt}); paths = comfy.run(wf)
OUTPUT: ['artifacts/ComfyUI_00012_.png']"""
LOG_DEPLOY = """CODE:
project.sh("npm run build")
OUTPUT: {'code': 1, 'out': 'Error: VITE_API_URL is not defined'}
CODE:
project.write(".env.production", "VITE_API_URL=https://api.example.com")
project.sh("npm run build && npx wrangler pages deploy dist --project-name shop")
OUTPUT: {'code': 0, 'out': 'Deployment complete: https://shop.pages.dev'}"""
LOG_SCRAPE = """CODE:
r = browser.do("sign in", values={"email": e, "password": pw})
OUTPUT: {'status': 'needs_confirmation', 'note': 'next: click button Send code'}
CODE:
r = browser.do("request the sign-in code by email", allow_irreversible=True); code = read_code_from_email()
OUTPUT: {'status': 'done'}
CODE:
r = browser.do("enter the code and continue", values={"code": code}); bill = browser.read("amount due and due date")
OUTPUT: Amount due: $84.12, due Oct 14"""
LOG_TRIVIAL = """CODE:
print(2340 * 0.17)
OUTPUT: 397.8"""
LOG_CHAT = """CODE:
final("Buenos dias.")
OUTPUT: [final answer recorded]"""
LOG_SIMPLE_FILE = """CODE:
save_artifact("haiku.txt", "Dark steam rising slow")
OUTPUT: artifacts/haiku.txt
CODE:
final("Saved.", ["artifacts/haiku.txt"])
OUTPUT: [final answer recorded]"""
LOG_LOOKUP = """CODE:
r = web.ask("population of Canada 2026")
print(r["passages"][0]["text"][:300])
OUTPUT: Canada's population was estimated at 41.8 million on July 1, 2026...
CODE:
final("About 41.8 million (July 2026).")
OUTPUT: [final answer recorded]"""

SKILLS = [  # (task, work log, worth a skill?)
    ("Make a Qwen image of a lighthouse at dusk", LOG_API, True),
    ("Deploy the shop frontend to Cloudflare Pages", LOG_DEPLOY, True),
    ("Download my latest Hydro One bill", LOG_SCRAPE, True),
    ("What's 17% of 2,340?", LOG_TRIVIAL, False),
    ("Translate good morning to Spanish", LOG_CHAT, False),
    ("Write a haiku about coffee and save it", LOG_SIMPLE_FILE, False),
    ("What's the population of Canada?", LOG_LOOKUP, False),
]


def run_live(path: Path | None = None) -> dict:
    path = path or RAW
    cfg = load_config()
    c = SystemOneClient(cfg.jev["url"], cfg.jev["backend"], timeout=300)
    raw = {"facts": [], "same": [], "skills": []}
    for user, reply, _ in FACTS:
        raw["facts"].append(c.ask({"user_message": user, "assistant_reply": reply}, {"q": noul(FACT_Q)})["q"].noul)
    for new, old, _ in SAME:
        raw["same"].append(c.ask({"new_fact": new, "old_facts": {"f1": old}},
                                 {"q": noul(SAME_FACT_Q.replace("`old_fact`", "`old_facts.f1`"))})["q"].noul)
    for task, log, _ in SKILLS:
        raw["skills"].append(c.ask({"task": task, "work_log": log}, {"q": noul(SKILL_Q)})["q"].noul)
    path.write_text(json.dumps(raw, indent=1), encoding="utf-8")
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
    _out.add_args(ap)
    a = ap.parse_args()
    dest = _out.Out.of(a)
    if a.offline:
        raw = json.loads(dest.read(RAW).read_text(encoding="utf-8"))
    else:
        raw = run_live(dest.write(RAW))
        print(f"raw scores saved to {dest.where(dest.write(RAW))}")
    sweep("fact: states a lasting fact", raw["facts"], [c[2] for c in FACTS])
    sweep("same_fact: retire the old fact", raw["same"], [c[2] for c in SAME])
    sweep("learn_skill: reusable procedure", raw["skills"], [c[2] for c in SKILLS])


if __name__ == "__main__":
    main()
