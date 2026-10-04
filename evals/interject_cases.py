"""Live eval: is a message the user sends while a task runs a question about progress? (jevh/interject.py,
Jev.interject_kind)

A message sent while a task runs is added to the task's prompt, except a question about progress, which the task's
model answers on the side. So the decision is two-way (`status` or `correction`); Jev's Choice has five options
because the others keep a message from being mistaken for a question:

  correction  changes what the task should do or how: a different target, a narrower or wider scope, a
              preference, advice. Added to the prompt.
  status      asks how it is going or what it is doing, and asks for nothing new. Answered on the side.
  separate    another request that does not change the current task and can wait. Added to the prompt (the
              user queues it with /queue).
  stop        stop / cancel / drop it. Added to the prompt (the user stops with /stop or the Stop button).

The costly mistakes: a correction, another request or a stop taken for a question is answered instead of heard, and
never reaches the task. A question taken for a correction is added to the prompt: the task reads it, which does
little harm. The labels stay four-way so each set says what the user meant.

Labelling rule: `separate` if the message could be done on its own after the task without changing what the task
produces ("also check my calendar" while checking email); `correction` if it changes the task's target, scope,
inputs, output or method ("also include spam" while searching the inbox). A question about the task's own
activity ("why that site?") is `status`. "Stop using that site" is a correction, not a stop.

    python evals/interject_cases.py                  # the tuning cases, then the held-out ones (live: needs the Jev adapter)
    python evals/interject_cases.py --holdout        # only the held-out cases
    python evals/interject_cases.py --fresh          # only the fresh cases (written after the last wording change)
    python evals/interject_cases.py --offline        # re-sweep the status bar on the saved raw scores
    python evals/interject_cases.py --runs 3         # each case three times (a hosted Jev can vary)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevh.config import load_config  # noqa: E402
from jevh.jev import Jev, decide_interject  # noqa: E402

RAW = Path(__file__).with_name("interject_cases_raw.json")
KINDS = ("correction", "status", "separate", "stop")

EMAIL = ("Check my email for anything from the landlord about the lease.", "running code: nango.gmail.search(query='from:landlord')")
FLIGHTS = ("Find flights to Denver next weekend.", "running code: web.search('flights to Denver next weekend')")
API = ("Add a DELETE endpoint for jobs to the Flask app.", "running code: project.edit('app/routes.py', ...)")
FIB = ("Write a script that prints the first 20 Fibonacci numbers, save it as fib.py and run it.", "the model is writing its next step")
RECIPE = ("Plan a week of dinners for two, vegetarian, under $80.", "the model is writing its next step")
REPORT = ("Summarize the three PDFs in my downloads folder into one page.", "running code: for f in pdfs: extract_text(f)")
HOUSE = ("Get quotes for insulating a 1,800 sq ft attic from three local companies.", "running code: web.search('attic insulation quote')")
CHART = ("Make a bar chart of monthly sales from sales.csv and save it as a PNG.", "running code: df.groupby('month').sum()")
WORKOUT = ("Plan my workouts for next week: 4 days, full gym.", "the model is writing its next step")
CAL = ("Find a time next week for a 30 minute call with Dave.", "running code: nango.calendar.events(next_week)")

# (task, doing, message, label)
DEV = [
    # -- corrections
    (*EMAIL, "Actually check my calendar instead", "correction"),
    (*EMAIL, "only the last two weeks please", "correction"),
    (*EMAIL, "use the work account, not the personal one", "correction"),
    (*EMAIL, "also include the spam folder in the search", "correction"),
    (*FLIGHTS, "make it Sunday morning only", "correction"),
    (*FLIGHTS, "nonstop flights only, and under $300", "correction"),
    (*FLIGHTS, "stop using that site, try Google Flights", "correction"),
    (*API, "use a soft delete, not a hard one", "correction"),
    (*API, "put it in its own file instead", "correction"),
    (*API, "and add tests for it too", "correction"),
    (*FIB, "make it primes instead, the first 15", "correction"),
    (*FIB, "use recursion with memoization", "correction"),
    (*RECIPE, "no mushrooms, my partner hates them", "correction"),
    (*RECIPE, "make it four people, not two", "correction"),
    (*REPORT, "keep it to half a page", "correction"),
    (*REPORT, "skip the third one, it is an old draft", "correction"),
    (*HOUSE, "only companies that do spray foam", "correction"),
    (*CHART, "make the bars blue and add a title", "correction"),
    (*WORKOUT, "I hurt my shoulder, no overhead pressing", "correction"),
    (*CAL, "mornings only, he is in another timezone", "correction"),
    (*FLIGHTS, "don't book anything yet, just list the options", "correction"),
    (*EMAIL, "what if you searched for the lease number instead of the sender?", "correction"),
    # -- status
    (*EMAIL, "how's it going?", "status"),
    (*EMAIL, "any update?", "status"),
    (*FLIGHTS, "what are you doing right now?", "status"),
    (*FLIGHTS, "have you found anything yet?", "status"),
    (*API, "are you stuck?", "status"),
    (*API, "which file are you editing?", "status"),
    (*FIB, "status?", "status"),
    (*RECIPE, "how much longer will this take?", "status"),
    (*REPORT, "which step are you on?", "status"),
    (*REPORT, "is it still running?", "status"),
    (*HOUSE, "what have you got so far?", "status"),
    (*CHART, "why is this taking so long?", "status"),
    (*WORKOUT, "hello? still there?", "status"),
    (*CAL, "did the calendar lookup work?", "status"),
    (*EMAIL, "why did you pick that search query?", "status"),
    (*FLIGHTS, "what site are you on?", "status"),
    (*API, "can you tell me what you have done so far?", "status"),
    (*RECIPE, "ETA?", "status"),
    # -- separate
    (*EMAIL, "remind me to call Dave at 3pm", "separate"),
    (*EMAIL, "what's the capital of Utah?", "separate"),
    (*FLIGHTS, "after that, turn off the living room lights", "separate"),
    (*FLIGHTS, "also write me a haiku about coffee", "separate"),
    (*API, "when you're done, summarize my unread Slack messages", "separate"),
    (*API, "next I need a quote for the Smith job", "separate"),
    (*FIB, "separately, what's 17% of 2,340?", "separate"),
    (*FIB, "and then check my calendar for tomorrow", "separate"),
    (*RECIPE, "later, order more printer paper", "separate"),
    (*REPORT, "one more thing for after: draft a thank-you note to the client", "separate"),
    (*HOUSE, "also, what time does the post office close today?", "separate"),
    (*CHART, "after this, plan my workouts for the week", "separate"),
    (*WORKOUT, "once that's done, find me a good sci-fi movie", "separate"),
    (*CAL, "I also need to renew my car registration, can you look up how?", "separate"),
    (*EMAIL, "also check my calendar", "separate"),
    (*API, "btw how do I center a div in CSS?", "separate"),
    # -- stop
    (*EMAIL, "stop", "stop"),
    (*EMAIL, "cancel that", "stop"),
    (*FLIGHTS, "never mind, forget it", "stop"),
    (*FLIGHTS, "stop what you're doing", "stop"),
    (*API, "abort", "stop"),
    (*API, "drop it, I'll do it myself", "stop"),
    (*FIB, "that's enough, stop now", "stop"),
    (*RECIPE, "I changed my mind, don't do this at all", "stop"),
    (*REPORT, "please stop", "stop"),
    (*HOUSE, "nvm, cancel", "stop"),
    (*CHART, "kill it", "stop"),
    (*CAL, "forget the call, stop looking", "stop"),
]

HELD = [
    ("Look up the opening hours of the three nearest hardware stores.", "running code: web.search('hardware store hours near me')",
     "just the ones that are open on Sundays", "correction"),
    ("Look up the opening hours of the three nearest hardware stores.", "the model is writing its next step",
     "how's it going?", "status"),
    ("Look up the opening hours of the three nearest hardware stores.", "running code: web.read(url)",
     "put a reminder in my calendar to buy paint on Saturday", "separate"),
    ("Look up the opening hours of the three nearest hardware stores.", "running code: web.read(url)",
     "okay stop, I've got what I need", "stop"),
    ("Refactor utils.py so the date helpers live in their own module.", "running code: project.edit('utils.py', ...)",
     "name the new module dates.py, not date_utils.py", "correction"),
    ("Refactor utils.py so the date helpers live in their own module.", "running code: project.grep('def .*date')",
     "what are you changing?", "status"),
    ("Refactor utils.py so the date helpers live in their own module.", "the model is writing its next step",
     "also, can you check whether the nightly job is still failing?", "separate"),
    ("Refactor utils.py so the date helpers live in their own module.", "running code: project.edit('utils.py', ...)",
     "undo everything, I don't want that refactor anymore", "stop"),
    ("Convert my notes.docx to a markdown file.", "running code: docx2md(path)",
     "keep the headings as they are", "correction"),
    ("Convert my notes.docx to a markdown file.", "running code: docx2md(path)", "done yet?", "status"),
    ("Convert my notes.docx to a markdown file.", "the model is writing its next step",
     "afterwards, email the result to Maria", "separate"),
    ("Convert my notes.docx to a markdown file.", "the model is writing its next step", "hold on, cancel it", "stop"),
    ("Compare the prices of the Pixel 9 and the Galaxy S25 at three stores.", "running code: web.search('Pixel 9 price')",
     "include the iPhone 16 too", "correction"),
    ("Compare the prices of the Pixel 9 and the Galaxy S25 at three stores.", "running code: web.read(url)",
     "which stores have you checked?", "status"),
    ("Compare the prices of the Pixel 9 and the Galaxy S25 at three stores.", "the model is writing its next step",
     "separate question: is it worth upgrading my phone this year?", "separate"),
    ("Compare the prices of the Pixel 9 and the Galaxy S25 at three stores.", "running code: web.search('Galaxy S25 price')",
     "scrap this comparison", "stop"),
    ("Write a cover letter for the barista job at Blue Door Cafe.", "the model is writing its next step",
     "mention that I speak Spanish", "correction"),
    ("Write a cover letter for the barista job at Blue Door Cafe.", "the model is writing its next step",
     "how far along are you?", "status"),
    ("Write a cover letter for the barista job at Blue Door Cafe.", "the model is writing its next step",
     "then find the cafe's address", "separate"),
    ("Write a cover letter for the barista job at Blue Door Cafe.", "the model is writing its next step",
     "I got the job elsewhere, stop", "stop"),
    ("Back up my Documents folder to the external drive.", "running code: shutil.copytree(...)",
     "skip the Downloads subfolder", "correction"),
    ("Back up my Documents folder to the external drive.", "running code: shutil.copytree(...)",
     "how much has been copied?", "status"),
    ("Back up my Documents folder to the external drive.", "running code: shutil.copytree(...)",
     "while it copies, tell me what the weather is tomorrow", "separate"),
    ("Back up my Documents folder to the external drive.", "running code: shutil.copytree(...)",
     "don't copy anything, pull the plug", "stop"),
    ("Find a vegetarian restaurant near the office for Friday lunch.", "running code: web.search('vegetarian restaurant near me')",
     "somewhere with parking", "correction"),
    ("Find a vegetarian restaurant near the office for Friday lunch.", "running code: web.search('vegetarian restaurant near me')",
     "found anything?", "status"),
    ("Find a vegetarian restaurant near the office for Friday lunch.", "the model is writing its next step",
     "and add 'send the invoice' to my to-do list", "separate"),
    ("Find a vegetarian restaurant near the office for Friday lunch.", "the model is writing its next step",
     "forget lunch, stop searching", "stop"),
    ("Draft a polite reply declining the meeting invitation from Pat.", "the model is writing its next step",
     "make it warmer, and suggest next month instead", "correction"),
    ("Draft a polite reply declining the meeting invitation from Pat.", "the model is writing its next step",
     "what are you writing?", "status"),
    ("Draft a polite reply declining the meeting invitation from Pat.", "the model is writing its next step",
     "also draft one to Lee, same thing", "separate"),
    ("Draft a polite reply declining the meeting invitation from Pat.", "the model is writing its next step",
     "actually don't reply at all, stop", "stop"),
    ("Debug why test_parse fails in the parser project.", "running code: project.sh('pytest -x tests/test_parse.py')",
     "it only fails on Windows, look at the path handling", "correction"),
    ("Debug why test_parse fails in the parser project.", "running code: project.sh('pytest -x tests/test_parse.py')",
     "is the test still running?", "status"),
    ("Debug why test_parse fails in the parser project.", "the model is writing its next step",
     "afterwards bump the version number", "separate"),
    ("Debug why test_parse fails in the parser project.", "the model is writing its next step",
     "enough, I'll look at it tomorrow, stop", "stop"),
    ("Find me a recipe for sourdough focaccia.", "running code: web.search('sourdough focaccia recipe')",
     "with rosemary, no olives", "correction"),
    ("Find me a recipe for sourdough focaccia.", "running code: web.search('sourdough focaccia recipe')",
     "anything good so far?", "status"),
    ("Find me a recipe for sourdough focaccia.", "the model is writing its next step",
     "what's a good sourdough starter feeding ratio?", "separate"),
    ("Find me a recipe for sourdough focaccia.", "the model is writing its next step",
     "I'm not baking tonight after all, cancel", "stop"),
]


# Written after the wording was last changed (the "replace" option) and scored once: not tuned on.
CONTRACT = ("Translate the attached contract into Spanish.", "running code: translate(chunk)")
PIZZA = ("Order two pizzas from the usual place for 7pm.", "running code: browser.goto('https://orders.example.com')")
HOTEL = ("Find the cheapest hotel in Portland for Oct 12-14.", "running code: web.search('cheap hotel Portland Oct 12')")
PHOTOS = ("Rename all the photos in the Vacation folder by date.", "running code: os.rename(old, new)")
EMAILFN = ("Write a Python function that validates email addresses, and test it.", "the model is writing its next step")
LEDGER = ("Reconcile the March bank statement with the ledger spreadsheet.", "running code: pd.merge(bank, ledger)")
AGENDA = ("Draft the agenda for Thursday's team meeting.", "the model is writing its next step")
FLIGHT = ("Look up the status of my flight UA 452 tomorrow.", "running code: web.search('UA 452 flight status')")

FRESH = [
    (*CONTRACT, "keep the legal terms in English", "correction"),
    (*CONTRACT, "use Latin American Spanish, not Spain's", "correction"),
    (*PIZZA, "actually just one pizza, we're only two", "correction"),
    (*PIZZA, "add garlic knots", "correction"),
    (*HOTEL, "it must have free parking and breakfast", "correction"),
    (*HOTEL, "stop looking at Booking.com, it's all sold out", "correction"),
    (*PHOTOS, "use YYYY-MM-DD, not DD-MM-YYYY", "correction"),
    (*EMAILFN, "use a regex, not a library", "correction"),
    (*LEDGER, "ignore transactions under $5", "correction"),
    (*AGENDA, "put the budget review first", "correction"),
    (*FLIGHT, "check my other flight too, UA 777", "correction"),
    (*PIZZA, "scrap the pizza, order sushi from Koi instead", "correction"),
    (*CONTRACT, "how's the translation going?", "status"),
    (*PIZZA, "are you done yet?", "status"),
    (*HOTEL, "what hotel are you looking at?", "status"),
    (*PHOTOS, "is it still renaming?", "status"),
    (*EMAILFN, "what's the progress?", "status"),
    (*LEDGER, "how many rows have you matched so far?", "status"),
    (*AGENDA, "give me a quick update", "status"),
    (*FLIGHT, "anything yet?", "status"),
    (*HOTEL, "what are you up to?", "status"),
    (*PHOTOS, "how many files are done?", "status"),
    (*CONTRACT, "are you waiting on something?", "status"),
    (*LEDGER, "what's the status?", "status"),
    (*CONTRACT, "remind me to water the plants tonight", "separate"),
    (*PIZZA, "when's sunset today?", "separate"),
    (*HOTEL, "also, what's the weather in Portland next week?", "separate"),
    (*PHOTOS, "can you add milk to the shopping list?", "separate"),
    (*EMAILFN, "afterwards, book me a table for Friday", "separate"),
    (*LEDGER, "while you do that, what's 15% tip on $84?", "separate"),
    (*AGENDA, "next up: draft a thank-you note to Chris", "separate"),
    (*FLIGHT, "separately: how do I reset my router?", "separate"),
    (*CONTRACT, "I also need the phone number of the DMV", "separate"),
    (*HOTEL, "later I want a summary of today's news", "separate"),
    (*PHOTOS, "whenever you're free: when is the next full moon?", "separate"),
    (*PIZZA, "and my dentist's number, please", "separate"),
    (*CONTRACT, "stop", "stop"),
    (*PIZZA, "cancel the order", "stop"),
    (*HOTEL, "hold on, don't do that, stop", "stop"),
    (*PHOTOS, "forget it", "stop"),
    (*EMAILFN, "enough", "stop"),
    (*LEDGER, "I'll take it from here, stop", "stop"),
    (*AGENDA, "abort mission", "stop"),
    (*FLIGHT, "cancel this", "stop"),
    (*CONTRACT, "no, stop, wrong thing entirely", "stop"),
    (*HOTEL, "shut it down", "stop"),
    (*PIZZA, "stop it right now", "stop"),
    (*PHOTOS, "never mind, I did it by hand", "stop"),
]


def run_live(cases, runs: int = 1) -> list[dict]:
    jev = Jev(load_config())
    rows = []
    for i, (task, doing, message, label) in enumerate(cases):
        for _ in range(runs):
            probs = jev.interject_kind(message, task, doing)
            rows.append({"i": i, "task": task, "doing": doing, "message": message, "label": label,
                         "probs": {k: round(float(probs.get(k, 0.0)), 4) for k in (*KINDS, "replace")}})
    return rows


def want(label: str) -> str:
    """What should happen to a message with this label: a question is answered, anything else is added to the prompt."""
    return "status" if label == "status" else "correction"


def evaluate(rows: list[dict], status_at: float) -> dict:
    right = corrections_answered = others_answered = questions_added = 0
    for r in rows:
        got = decide_interject(r["probs"], status_at)
        right += got == want(r["label"])
        if got == "status" and r["label"] != "status":
            if r["label"] == "correction":
                corrections_answered += 1       # a correction answered instead of heard
            else:
                others_answered += 1            # another request or a stop answered instead of heard
        elif got == "correction" and r["label"] == "status":
            questions_added += 1                # a question added to the prompt: the task reads it
    questions = sum(1 for r in rows if r["label"] == "status")
    total = len(rows)
    return {"accuracy": right / total, "right": right, "total": total, "questions": questions,
            "questions_answered": questions - questions_added, "questions_added": questions_added,
            "corrections_answered": corrections_answered, "others_answered": others_answered}


def report(name: str, rows: list[dict], status_at: float) -> None:
    e = evaluate(rows, status_at)
    print(f"\n== {name}: {e['right']}/{e['total']} right ({e['accuracy']:.0%}) at status bar {status_at}")
    print(f"   questions answered on the side: {e['questions_answered']}/{e['questions']}")
    print(f"   corrections answered instead of heard: {e['corrections_answered']}   "
          f"other requests / stops answered instead of heard: {e['others_answered']}")
    for r in rows:
        got = decide_interject(r["probs"], status_at)
        if got != want(r["label"]):
            p = r["probs"]
            print(f"   x [{r['label']}] {r['message']!r} -> {got}  "
                  + " ".join(f"{k[:4]} {p.get(k, 0):.2f}" for k in (*KINDS, "replace")))


def sweep(rows: list[dict]) -> None:
    for v in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        e = evaluate(rows, v)
        print(f"   status bar {v:.2f}: {e['right']}/{e['total']}  questions answered {e['questions_answered']}/{e['questions']}"
              f"  corrections answered instead of heard {e['corrections_answered']}"
              f"  others {e['others_answered']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="re-sweep the saved raw scores")
    ap.add_argument("--holdout", action="store_true", help="only the held-out cases")
    ap.add_argument("--fresh", action="store_true", help="only the fresh cases (written after the last wording change)")
    ap.add_argument("--runs", type=int, default=1)
    a = ap.parse_args()
    status_at = load_config().t("interject_status")
    if a.offline:
        saved = json.loads(RAW.read_text(encoding="utf-8"))
    else:
        saved = {}
        if a.fresh:
            saved["fresh"] = run_live(FRESH, a.runs)
        else:
            if not a.holdout:
                saved["dev"] = run_live(DEV, a.runs)
            saved["held"] = run_live(HELD, a.runs)
        old = json.loads(RAW.read_text(encoding="utf-8")) if RAW.exists() else {}
        RAW.write_text(json.dumps({**old, **saved}, indent=1), encoding="utf-8")
    for name in ("dev", "held", "fresh"):
        if name in saved:
            report(name, saved[name], status_at)
    for name in ("dev", "held", "fresh"):
        if name in saved:
            print(f"\n-- {name}: sweep")
            sweep(saved[name])


if __name__ == "__main__":
    main()
