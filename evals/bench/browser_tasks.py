"""Real-website browser tasks: full harness turns (daily_assistant: worker LLM + Jev-driven browser).

Each task is a plain user request on a live website. Nothing real is bought, booked, sent or signed
into: tasks stop at the sign-in or payment step, and a simulated user answers ask_user() - "stop
here" to any order/payment/booking confirmation, the made-up details below to anything else. The one
submit allowed is a public practice form (demoqa), built for exactly that.

    python evals/bench/browser_tasks.py                 # all tasks
    python evals/bench/browser_tasks.py -k books,hn     # tasks whose name contains any of these
    python evals/bench/browser_tasks.py --tier strong   # every task on the strong model (default: Jev picks)
    python evals/bench/browser_tasks.py --report        # the last run's results

Every task writes runs/browser/<stamp>/<task>/: events.jsonl (every code step, its output, questions
and answers), answer.txt and the worker's artifacts (screenshots). `check` scores the answer where the
truth can be fetched independently; the rest are graded by reading the trace (grade: manual).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

RUNS = HERE / "runs" / "browser"
PERSONA = "daily_assistant"
TIER = "auto"     # --tier: auto (Jev picks per task) | fast | strong
LIMIT = 20 * 60   # seconds per task

ME = ("My name is Sam Rivera, email sam.rivera@example.com, phone 217-555-0142, address 12 Elm St, "
      "Springfield, IL 62704. Born March 12, 1990.")
STOP_HERE = "No - don't pay, book or submit anything. Stop here and tell me exactly where you got to."


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "replace")


def has(answer: str, *needles: str) -> list[str]:
    """The needles missing from the answer (case and punctuation insensitive)."""
    norm = lambda s: re.sub(r"[^a-z0-9.]+", " ", s.lower())
    a = norm(answer)
    return [n for n in needles if norm(n).strip() not in a]


def check_books(answer: str) -> tuple[bool, str]:
    miss = has(answer, "Little Dribbling", "23.21", "Full Moon over Noah", "Year in Provence", "1,000 Places")
    return not miss, "missing: " + ", ".join(miss) if miss else "cheapest + all three 4-5 star books"


def check_quotes(answer: str) -> tuple[bool, str]:
    # the count stated next to Einstein ("Total Albert Einstein quotes: 10", "10 quotes by Einstein"),
    # not the site's total ("100 quotes on 10 pages")
    lines = [ln for ln in answer.splitlines() if re.search(r"einstein", ln, re.I) and re.search(r"quot|total", ln, re.I)]
    nums = [int(x) for ln in lines for x in re.findall(r"\b(\d+)\b", ln)]
    n = 10 if 10 in nums else (nums[0] if nums else None)
    miss = has(answer, "change", "deep thoughts", "thinking", "world")   # tags of Einstein's first quote
    ok = n == 10 and len(miss) <= 1
    return ok, f"count={n} (truth 10), first-quote tags missing: {miss}"


def check_hn(answer: str) -> tuple[bool, str]:
    top = json.loads(fetch("https://hacker-news.firebaseio.com/v0/topstories.json"))[:3]
    items = [json.loads(fetch(f"https://hacker-news.firebaseio.com/v0/item/{i}.json")) for i in top]
    for it in items:   # the #1 spot can change during the run: any of the top 3, with its own numbers
        words = [w for w in re.findall(r"[A-Za-z0-9]{4,}", it.get("title", ""))][:5]
        if words and len(has(answer, *words)) <= 1:
            nums = [int(x.replace(",", "")) for x in re.findall(r"\b\d[\d,]*\b", answer)]
            pts, com = it.get("score", 0), it.get("descendants", 0)
            close = lambda v: any(abs(n - v) <= max(5, v * 0.15) for n in nums)
            return close(pts) and close(com), f"'{it['title']}' now {pts} points, {com} comments; answer numbers {nums[:8]}"
    return False, "answer names none of the current top 3: " + " | ".join(i.get("title", "") for i in items)


# name, request, reply to non-confirmation questions, reply to confirmations, check (None: manual)
TASKS = [
    ("flights-google",
     "Use the browser to find the cheapest nonstop one-way flight from Boston (BOS) to Denver (DEN) on "
     "October 14, 2026 for 1 adult on Google Flights (https://www.google.com/travel/flights). Tell me the "
     "airline, departure and arrival times and the price. Don't book anything.",
     ME, STOP_HERE, None),
    ("flights-book",
     "On https://blazedemo.com, book the cheapest flight from Boston to London. Fill in the passenger "
     "details with my info, but stop before the purchase: don't click Purchase. Tell me the flight number, "
     "airline and price you picked. " + ME,
     ME, STOP_HERE, None),
    ("tickets",
     "I want to see a concert at Red Rocks Amphitheatre. Using the browser, start at "
     "https://www.redrocksonline.com/events/ , find the next upcoming concert, and get as far as choosing "
     "2 of the cheapest tickets. Stop before any payment or sign-in, and tell me the show, date, "
     "price per ticket and where you stopped.",
     ME, STOP_HERE, None),
    ("cart-checkout",
     "On https://www.target.com, add a 12 oz bag of Starbucks Pike Place Roast whole bean coffee to the "
     "cart, then go to checkout. Stop at sign-in or payment - don't sign in or pay. Tell me the price and "
     "the cart total you saw.",
     ME, STOP_HERE, None),
    ("form-fill",
     "Fill out and submit the practice form at https://demoqa.com/automation-practice-form with my "
     "details: first name Sam, last name Rivera, email sam.rivera@example.com, gender Male, mobile "
     "2175550142, date of birth 12 March 1990, subjects Maths and Physics, hobbies Reading, current "
     "address 12 Elm St, Springfield, state NCR, city Delhi. It's a practice form, so submit it. Tell me "
     "what the confirmation shows.",
     ME, "Yes, submit it - it's a practice form.", None),
    ("scrape-books",
     "Go to https://books.toscrape.com and look at the Travel category. Which book is the cheapest (title "
     "and price), and which Travel books have a rating of 4 or 5 stars?",
     ME, STOP_HERE, check_books),
    ("scrape-quotes",
     "Using the browser on https://quotes.toscrape.com/js/ (the JavaScript version), go through every page "
     "and collect all quotes by Albert Einstein. How many are there in total, and what tags does the first "
     "one have?",
     ME, STOP_HERE, check_quotes),
    ("game-2048",
     "Play the game 2048 at https://play2048.co/ in the browser and try to reach a score of at least 1000. "
     "Tell me your final score and the biggest tile, and save a screenshot of the board at the end.",
     ME, STOP_HERE, None),
    ("hotel-search",
     "On https://www.booking.com, find the cheapest hotel in Denver for October 14-16, 2026, for 2 adults, "
     "with a guest review score of 8 or higher. Tell me the hotel's name, its score and the total price. "
     "Don't reserve anything.",
     ME, STOP_HERE, None),
    ("hn-live",
     "Open Hacker News (https://news.ycombinator.com) in the browser, find the current #1 story, open its "
     "comments page, and tell me the story title, how many points it has and how many comments.",
     ME, STOP_HERE, check_hn),
]

CONFIRM = re.compile(r"\b(order|pay|payment|purchase|buy|book|booking|reserve|reservation|submit|place|checkout|"
                     r"confirm|proceed|sign in|log in|login|password|card)\b", re.I)


def run_task(cfg, name: str, request: str, info: str, confirm: str, run_dir: Path) -> dict:
    from jevh.orchestrator import Harness
    ws = run_dir / name
    ws.mkdir(parents=True, exist_ok=True)
    cfg.workspace = ws / "ws"
    log = open(ws / "events.jsonl", "w", encoding="utf-8")
    t0 = time.monotonic()
    h = None
    qs: list[dict] = []

    def on_event(k, d):
        rec = {"t": round(time.monotonic() - t0, 1), "kind": k}
        for key, v in (d or {}).items():
            if key == "task":
                v = getattr(v, "id", str(v))
            try:
                json.dumps(v)
            except TypeError:
                v = str(v)
            rec[key] = v
        log.write(json.dumps(rec, ensure_ascii=False)[:20000] + "\n")
        log.flush()
        if k == "code":
            first = next((ln for ln in str(d.get("code", "")).splitlines() if ln.strip()), "")
            print(f"    [{rec['t']:6.0f}s] step {d.get('step')}: {first[:110]}", flush=True)
        elif k == "question":
            q = d["question"]
            reply = confirm if CONFIRM.search(q["question"]) else info
            qs.append({"q": q["question"], "options": q["options"], "reply": reply})
            print(f"    [{rec['t']:6.0f}s] ASK: {q['question'][:150]}\n             -> {reply[:80]}", flush=True)
            h.answer(q["id"], reply)
        elif k == "verify" and d.get("bounced"):
            print(f"    [{rec['t']:6.0f}s] final() sent back: {str(d.get('problems', ''))[:150]}", flush=True)

    h = Harness(cfg, on_event=on_event)
    h.learning, h.tier_mode = False, TIER
    timer = threading.Timer(LIMIT, h.cancel)
    timer.start()
    try:
        r = h.turn(request, force=PERSONA)
        answer, cancelled, arts = r.answer, r.cancelled, r.artifacts
        usage = (r.usage or {}).get("total", {})
    except Exception as e:  # noqa: BLE001 - one broken task never ends the run
        answer, cancelled, arts, usage = f"(harness crash: {e!r})", False, [], {}
    finally:
        timer.cancel()
        log.close()
        try:
            h.wait_learning(5)
        except Exception:  # noqa: BLE001
            pass
    (ws / "answer.txt").write_text(answer, encoding="utf-8")
    steps = sum(1 for ln in open(ws / "events.jsonl", encoding="utf-8") if '"kind": "code"' in ln)
    return {"answer": answer, "timed_out": cancelled, "artifacts": arts, "questions": qs, "steps": steps,
            "seconds": round(time.monotonic() - t0), "tokens_out": usage.get("output", 0),
            "thinking": usage.get("thinking", 0)}


def report(path: Path | None = None) -> None:
    path = path or max(RUNS.glob("*/results.json"), key=lambda p: p.stat().st_mtime)
    rows = json.loads(path.read_text(encoding="utf-8"))
    print(f"{path.parent.name}")
    for r in rows:
        g = {True: "PASS", False: "FAIL", None: "----"}[r["pass"]]
        print(f"{g} {r['task']:15} {r['seconds']:5}s {r['steps']:3} steps  {r['grade']}")
        print(f"     answer: {r['answer'][:300]!r}")


def main() -> None:
    global LIMIT, TIER
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="", help="comma-separated name filters")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--limit", type=int, default=LIMIT // 60, help="minutes per task")
    ap.add_argument("--tier", default="auto", choices=["auto", "fast", "strong"], help="worker model tier")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # answers carry emoji, pound signs...
    if args.report:
        return report()
    LIMIT, TIER = args.limit * 60, args.tier
    from jevh.config import load_config
    keys = [k for k in args.k.split(",") if k]
    tasks = [t for t in TASKS if not keys or any(k in t[0] for k in keys)]
    run_dir = RUNS / (time.strftime("%Y%m%d-%H%M%S") + ("" if TIER == "auto" else f"-{TIER}"))
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, request, info, confirm, check in tasks:
        print(f"\n=== {name}: {request[:120]}", flush=True)
        cfg = load_config()
        r = run_task(cfg, name, request, info, confirm, run_dir)
        if check:
            try:
                ok, why = check(r["answer"])
            except Exception as e:  # noqa: BLE001
                ok, why = False, f"check crashed: {e!r}"
        else:
            ok, why = None, "manual"
        rows.append({"task": name, "pass": ok, "grade": why, **r})
        (run_dir / "results.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"--> {name}: {'PASS' if ok else 'FAIL' if ok is False else 'manual'} ({why}) in {r['seconds']}s, "
              f"{r['steps']} steps{' TIMED OUT' if r['timed_out'] else ''}\n    answer: {r['answer'][:400]!r}",
              flush=True)
    report(run_dir / "results.json")


if __name__ == "__main__":
    main()
