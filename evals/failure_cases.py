"""Live eval: the failure log's Jev check. After a task, does the answer give up, or fail to
deliver what was asked?

A missed failure only costs a row in workspace/failures.db; a false one puts a good run among the
failures that will be studied to change the harness, so false alarms matter more here.

    python evals/failure_cases.py            # asks Jev, saves the scores, sweeps the thresholds
    python evals/failure_cases.py --offline  # re-sweeps the saved scores
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevh.config import load_config  # noqa: E402
from jevh.decide import SystemOneClient, noul  # noqa: E402
from jevh.failures import GAVE_UP_Q, MISSED_Q  # noqa: E402

RAW = Path(__file__).with_name("failure_cases_raw.json")


def log(*steps):
    return "\n".join(f"CODE:\n{c}\nOUTPUT: {o}" for c, o in steps)


NO_CODE = "(no code was run)"
PLANETS_LOG = log(("rows = [('Mercury', 1, False), ('Venus', 2, False), ('Earth', 3, False), ('Mars', 4, False), "
                   "('Jupiter', 5, True), ('Saturn', 6, True), ('Uranus', 7, True), ('Neptune', 8, True)]\n"
                   "df = pd.DataFrame(rows, columns=['name', 'order_from_sun', 'has_rings'])\n"
                   "df.to_csv('planets.csv', index=False)\nprint(df.to_string(index=False))",
                   "name  order_from_sun  has_rings\nMercury  1  False\n...\nNeptune  8  True"),
                  ("final(answer, artifacts=['planets.csv'])", "[final answer recorded]"))
CASES = [  # (request, answer, work log, label: ok | gave_up | missed[, files handed over])
    ("hi", "Hello Jed! What can I do for you today?", NO_CODE, "ok"),
    ("Plan my training week: 4 days, full gym.",
     "Mon: upper (bench 4x6, row 4x8, OHP 3x8)\nTue: lower (squat 4x5, RDL 3x8)\nThu: upper (incline DB 4x8, "
     "pull-ups 4x6)\nFri: lower (deadlift 3x5, split squat 3x10)", NO_CODE, "ok"),
    ("What's the population of Canada?", "About 41.8 million (Statistics Canada, July 2026).",
     log(("print(web.ask('population of Canada'))", "Canada: 41,828,000 (Statistics Canada, July 2026)")), "ok"),
    ("Fix the failing tests in my project.", "Fixed: get() took `id` but callers passed `record_id`. All 15 tests pass.",
     log(("print(project.sh('pytest -q')['out'])", "1 failed, 14 passed"),
         ("project.edit('app/db.py', 'def get(id):', 'def get(record_id):')", "edited app/db.py"),
         ("print(project.sh('pytest -q')['out'])", "15 passed")), "ok"),
    ("Build a landing page for my bakery.", "Built index.html: hero, menu, opening hours and a contact form.",
     log(("p = save_artifact('index.html', html)", "artifacts/index.html")), "ok", ["index.html"]),
    ("Find the cheapest flight to Lisbon on Oct 3.", "The cheapest is TAP at $412, departing 09:40 from JFK.",
     log(("print(browser.do('search flights JFK to LIS Oct 3'))", "done"),
         ("print(browser.read('cheapest fare'))", "TAP $412 09:40; United $455 18:05")), "ok"),
    ("Summarize this article in 3 bullet points.", "- Rates held at 4.5%\n- Inflation eased to 2.4%\n- Cuts "
     "expected in spring", NO_CODE, "ok"),
    # gave up / partial, said plainly
    ("Find the cheapest flight to Lisbon on Oct 3.", "I couldn't get any fares: the airline site kept showing a "
     "bot check, so I don't have a price for you.",
     log(("print(browser.do('search flights JFK to LIS Oct 3'))", "{'status': 'stuck', 'note': 'captcha'}")),
     "gave_up"),
    ("Fix the failing tests in my project.", "I fixed 2 of the 3 failures. test_retry still fails; I couldn't work "
     "out why the retry count is off.",
     log(("print(project.sh('pytest -q')['out'])", "3 failed"), ("...", "..."),
         ("print(project.sh('pytest -q')['out'])", "1 failed: test_retry")), "gave_up"),
    ("Make a hero image for the site.", "ComfyUI isn't reachable (connection refused), so no image was made.",
     log(("print(comfy.run(wf))", "ConnectionError: [Errno 111] Connection refused")), "gave_up"),
    ("Scrape all 5 pages of the catalogue.", "Here are the products from pages 1-2 (24 rows); the next-page button "
     "stopped working after that.", log(("rows += extract(browser.read())", "12 rows"),
                                         ("browser.do('next page')", "{'status': 'stuck'}")), "gave_up"),
    # doesn't deliver what was asked, without saying so
    ("Build a landing page for my bakery.", "Here's my plan: a hero with a photo of bread, then a menu section, "
     "then opening hours. Let me know if that works!", NO_CODE, "missed"),
    ("What's the population of Canada?", "Canada is the second-largest country in the world by area, with ten "
     "provinces and three territories.", NO_CODE, "missed"),
    ("Convert this CSV to JSON and save it.", "CSV is a plain-text format where each line is a record and commas "
     "separate the fields.", NO_CODE, "missed"),
    ("Plan my training week: 4 days, full gym.", "Great goal! Consistency is the key to progress. Remember to "
     "sleep well and eat enough protein.", NO_CODE, "missed"),
    # files: the request asks for a file; `artifacts` is what was handed over. Live (2026-09-30): without
    # the artifacts in the check, 2 of 5 correct planets.csv runs were logged as missed_request (0.89, 0.97).
    ("Give me a CSV file of the 8 planets with columns name, order_from_sun, has_rings and hand it over.",
     "planets.csv with 8 planets: name, order_from_sun, has_rings. Mercury(1),Venus(2),Earth(3),Mars(4) have no "
     "rings; Jupiter(5),Saturn(6),Uranus(7),Neptune(8) have rings.", PLANETS_LOG, "ok", ["planets.csv"]),
    ("Give me a CSV file of the 8 planets with columns name, order_from_sun, has_rings and hand it over.",
     "Created planets.csv with 8 rows and columns name, order_from_sun, has_rings.", PLANETS_LOG, "ok",
     ["planets.csv"]),
    ("Make me a PNG bar chart of these monthly sales: Jan 12, Feb 18, Mar 9, Apr 15, and hand it over.",
     "PNG bar chart of the four monthly sales values with labeled axes and per-bar value labels.",
     log(("plt.bar(months, sales); plt.savefig('monthly_sales_chart.png')", "")), "ok", ["monthly_sales_chart.png"]),
    ("Give me a CSV file of the 8 planets with columns name, order_from_sun, has_rings and hand it over.",
     "Here are the planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.", NO_CODE, "missed", []),
    ("Make me a PNG bar chart of these monthly sales: Jan 12, Feb 18, Mar 9, Apr 15, and hand it over.",
     "You can make this in Excel: select the data, then Insert > Bar chart.", NO_CODE, "missed", []),
    ("Write a Python file leap.py with a function is_leap(year) and hand it over.",
     "Sure - I'll write leap.py with is_leap() next.", NO_CODE, "missed", []),
]


def run_live() -> list[dict]:
    cfg = load_config()
    c = SystemOneClient(cfg.jev["url"], cfg.jev["backend"], timeout=300)
    rows = []
    for req, ans, wl, _, *files in CASES:
        state = {"request": req, "answer": ans, "work_log": wl, "artifacts": (files[0] if files else []) or "(none)"}
        r = c.ask(state, {"gave_up": noul(GAVE_UP_Q), "missed": noul(MISSED_Q)})
        rows.append({"gave_up": r["gave_up"].noul, "missed": r["missed"].noul})
    RAW.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    rows = json.loads(RAW.read_text(encoding="utf-8")) if ap.parse_args().offline else run_live()
    labels = [c[3] for c in CASES]
    for (req, ans, _, lab, *_files), s in zip(CASES, rows):
        print(f"  {lab:<8} gave_up={s['gave_up']:.2f} missed={s['missed']:.2f}  {req[:40]} / {ans[:40]}")
    for key in ("gave_up", "missed"):
        print(f"\n== {key}")
        for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
            flagged = [s[key] >= t for s in rows]
            caught = sum(f and lab == key for f, lab in zip(flagged, labels))
            false = sum(f and lab == "ok" for f, lab in zip(flagged, labels))
            print(f"  t={t:.2f}: {key} caught {caught}/{labels.count(key)}, good answers flagged "
                  f"{false}/{labels.count('ok')}")
    print("\n== either (as the harness uses them), same threshold for both")
    for t in (0.6, 0.7, 0.8, 0.85, 0.9):
        flagged = [s["gave_up"] >= t or s["missed"] >= t for s in rows]
        caught = sum(f and lab != "ok" for f, lab in zip(flagged, labels))
        false = sum(f and lab == "ok" for f, lab in zip(flagged, labels))
        print(f"  t={t:.2f}: failures caught {caught}/{len(labels) - labels.count('ok')}, good answers flagged "
              f"{false}/{labels.count('ok')}")


if __name__ == "__main__":
    main()
