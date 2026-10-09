"""Live eval of the three decisions Jev makes for the Newsroom (jevh/newsroom.py, docs/newsroom.md). Written in a cloud
session, first run 2026-10-09 on semif (3 runs per case; results in docs/newsroom.md's status note): the tool pick and the routing
fell below the targets below. Read the numbers against the targets; a target missed is a finding, not a bug in this file.
Raw scores: newsroom_cases_raw.json (dev) and newsroom_cases_holdout_raw.json (--holdout), each re-swept by --offline.

  tool   Does Jev put the `newsroom` guide in the worker's prompt for "keep an eye on / track / alert me" requests, and
         NOT for the requests that look like them (a one-off "what's the news on X?", "follow up with Sam", "monitor my
         server")? Runs the harness's own Jev.pick_tools with the daily assistant's tools, so the score is the pick the
         worker really gets. Three sets: --dev (the cases the description was written against), --holdout (written
         after, never used to change the wording), and the regression set (every request in evals/tool_cases.py, none of
         which wants the newsroom: the tool must never be picked for them). With --regress it also asks the same
         requests WITHOUT the newsroom among the tools and compares the other tools' scores, to show adding it moved
         nothing else.
  route  Does the changed daily_assistant description (it now says it keeps an eye on things for the Newsroom) send the
         newsroom requests to it, and still send every request in evals/routing_cases.py where it went before? Old and new
         description side by side on the same requests.
  about  The `about=` alert: Jev reads a new headline against "what the user wants to hear about". Precision and recall
         of keywords alone, `about` alone and either, on labelled headlines (the hard ones have no keyword, or the right
         company but the wrong subject), and a sweep of the threshold (thresholds.newsroom_about: 0.70 is a guess).

Targets (first guesses, printed beside each result)
  tool    fires on >= 90% of the should-fire requests (dev and held-out), on <= 5% of the lookalikes, on 0 of the regression
          requests; the other tools' scores unchanged (max difference < 0.05)
  route   the newsroom requests reach daily_assistant on >= 90%; no request in routing_cases.py changes persona
  about   at the config threshold: `about` finds >= 70% of the headlines the keywords miss, with <= 1 in 10 of what it
          sends being wrong; `either` loses nothing keywords alone found

If the feature is wrong, expect: the description matching one-off news questions (fires on "what's the news on X?",
which sends a simple lookup into a widget), matching "follow up", "monitor my server" or "notify the team"; or missing
requests that never say "track" ("let me know if...", "I want a board of..."); the new description pulling requests
away from the other personas; an `about` threshold that is far too low (notifies on every story about the company) or
too high (never fires).

    python evals/newsroom_cases.py                       # every stage on the dev sets (live: needs the Jev adapter)
    python evals/newsroom_cases.py --holdout             # the held-out sets
    python evals/newsroom_cases.py --stage tool,about --runs 3
    python evals/newsroom_cases.py --offline             # re-sweep the saved raw scores (newsroom_cases_raw.json)
    python evals/newsroom_cases.py --export items.jsonl  # real items from your board, to label for the about stage
    python evals/newsroom_cases.py --labelled items.jsonl  # ...and score them (set "label": true/false on each line)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from jevh import newsroom as nr  # noqa: E402
from jevh.config import load_config  # noqa: E402
from jevh.decide import noul  # noqa: E402
from jevh.jev import Jev  # noqa: E402
from jevh.personas import load_personas  # noqa: E402
from jevh.tools import TOOLS  # noqa: E402

RAW = HERE / "newsroom_cases_raw.json"                       # one file per set: a live run starts from nothing, so one shared
RAW_HOLDOUT = HERE / "newsroom_cases_holdout_raw.json"       # file lost the dev run when the held-out one was made, and
SUMMARY = HERE / "newsroom_cases_last_summary.txt"           # --offline re-swept whichever set was saved last
SUMMARY_HOLDOUT = HERE / "newsroom_cases_holdout_last_summary.txt"
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9)
DAILY_TOOLS = ("web", "browser", "desktop", "nango", "newsroom")
D = "daily_assistant"

# the daily_assistant description before the Newsroom (so `route` can compare it with today's)
OLD_DESCRIPTION = (
    "General everyday assistant: questions, planning the day, calendar and appointments,\n"
    "email, reminders, notes, quick lookups, business paperwork (quotes, payroll), smart home,\n"
    "and connecting the user's own accounts (Gmail, Google Calendar, GitHub, Slack) to the assistant.")

# --------------------------------------------------------------------------- tool: requests (authored; see --export for real ones)
# (request, should the newsroom guide be in the prompt)
TOOL_DEV = [
    ("Keep an eye on Nvidia news and tell me when they mention earnings.", True),
    ("Follow @OpenAI on X and alert me whenever they post.", True),
    ("Add the llama.cpp releases to my newsroom.", True),
    ("Track news about solid-state batteries for me.", True),
    ("Let me know if Apple announces anything about a foldable iPhone.", True),
    ("Watch r/LocalLLaMA and ping me when someone posts about a new quantization method.", True),
    ("Put the BBC tech feed on my dashboard: https://feeds.bbci.co.uk/news/technology/rss.xml", True),
    ("Notify me whenever Tesla is in the news for a recall.", True),
    ("Set up a widget that follows Hacker News for posts about Rust.", True),
    ("I want to be alerted when the Fed changes interest rates.", True),
    ("Keep me updated on SpaceX launches.", True),
    ("Monitor arXiv for new papers on speculative decoding and tell me about them.", True),
    ("Remove the Nvidia tile from my newsroom.", True),
    ("What's on my newsroom right now?", True),
    ("Make the bitcoin widget check every 10 minutes.", True),
    ("Pause the Tesla feed in my newsroom.", True),
    ("Start tracking everything Anthropic posts on X.", True),
    ("Add a news widget for Microsoft and one for Google.", True),
    ("Can you watch for new videos on the Fireship YouTube channel and alert me?", True),
    ("Tell me if anything big happens with Boeing, I want to know the moment it hits the news.", True),
    ("I'd like a dashboard of news about the housing market that updates itself.", True),
    ("Change the alert words on my Nvidia widget to 'guidance' and 'lawsuit'.", True),
    ("Add Reuters' business news to my newsroom and notify me about mergers.", True),
    # lookalikes: must NOT fire
    ("What's the latest news on Nvidia?", False),
    ("Summarize today's top tech headlines for me.", False),
    ("What did OpenAI post on X yesterday?", False),
    ("Remind me to follow up with Sam at 3pm.", False),
    ("Research solid-state batteries and write me a one-page report.", False),
    ("Tell me when my Amazon order ships.", False),
    ("Email me a summary of the news every morning.", False),
    ("Monitor my server's CPU and tell me if it goes above 90%.", False),
    ("Keep track of my workout log.", False),
    ("What is an RSS feed and how does it work?", False),
    ("Write a news article about a fictional product launch.", False),
    ("Is Tesla stock up today?", False),
    ("Follow up with the client about the quote.", False),
    ("Watch this video and summarize it: https://youtu.be/dQw4w9WgXcQ", False),
    ("Who is the CEO of Nvidia?", False),
    ("Explain how Google News ranks stories.", False),
    ("Check my email for anything from the bank.", False),
    ("Alert me 10 minutes before my next meeting.", False),
    ("Bookmark this article for later: https://example.com/a", False),
    ("Post a tweet saying we're open late Friday.", False),
    ("Show me the Hacker News front page right now.", False),
]
TOOL_HOLDOUT = [
    ("I run a small chip shop: keep tabs on any news about AMD and Intel and flag layoffs.", True),
    ("Follow the Python release feed https://www.python.org/dev/peps/peps.rss and tell me about new PEPs.", True),
    ("Tell me whenever @elonmusk posts about Mars.", True),
    ("Give me a live board of headlines on climate policy.", True),
    ("Add GitHub releases for the vllm project to my newsroom.", True),
    ("Keep an eye out for news about my competitor Acme Corp and ping me if they raise money.", True),
    ("Take the Bitcoin widget off my dashboard.", True),
    ("Rename my Tesla news tile to 'EV watch'.", True),
    ("Which of my newsroom widgets have notifications turned on?", True),
    ("I want a feed of new LocalLLaMA posts, no alerts, just show them.", True),
    ("Watch for hospital-closure news in Utah and tell me about it.", True),
    ("Let me know whenever the Rust blog publishes something.", True),
    ("What happened with Boeing this week?", False),
    ("Draft a tweet about our launch.", False),
    ("Follow me on this: we're moving the meeting to 4.", False),
    ("Summarize the last five posts from @OpenAI.", False),
    ("Set a timer for 20 minutes.", False),
    ("How do I add an RSS feed to Thunderbird?", False),
    ("What's the weather going to be like on Saturday?", False),
    ("Notify the team in Slack that the deploy finished.", False),
    ("Track my expenses for October in a spreadsheet.", False),
    ("Look up the latest llama.cpp release and tell me what changed.", False),
    ("Who won the game last night?", False),
    ("Keep it short.", False),
]

# --------------------------------------------------------------------------- about: headlines (authored)
# (what the user wants to hear about, headline, label, the keywords they also gave)
ABOUT_DEV = [
    ("their quarterly results", "Nvidia posts record revenue of $46.7 billion, topping Wall Street estimates", True, ["earnings"]),
    ("their quarterly results", "Nvidia beats earnings estimates as data center sales surge", True, ["earnings"]),
    ("their quarterly results", "Nvidia shares jump after profit more than doubles", True, ["earnings"]),
    ("their quarterly results", "Nvidia unveils its next GPU architecture at GTC", False, ["earnings"]),
    ("their quarterly results", "Nvidia CEO tells Stanford students to learn to code with AI", False, ["earnings"]),
    ("their quarterly results", "Nvidia sponsors a Formula E team for the new season", False, ["earnings"]),
    ("their quarterly results", "Why Nvidia's stock could fall even if earnings beat", True, ["earnings"]),
    ("a product recall or safety problem", "Tesla recalls 120,000 vehicles over a seat belt warning light", True, ["recall"]),
    ("a product recall or safety problem", "Tesla pulls Cybertruck accessory after NHTSA complaint", True, ["recall"]),
    ("a product recall or safety problem", "Tesla opens a new Supercharger site in Utah", False, ["recall"]),
    ("a product recall or safety problem", "Tesla stock rallies as investors recall last year's lows", False, ["recall"]),
    ("a product recall or safety problem", "Federal regulators open a probe into Tesla's driver assistance crashes", True, ["recall"]),
    ("a new model or major product release", "OpenAI releases GPT-6 with longer context and cheaper tokens", True, ["launch", "release"]),
    ("a new model or major product release", "OpenAI quietly ships a smaller reasoning model to the API", True, ["launch", "release"]),
    ("a new model or major product release", "OpenAI names a new chief financial officer", False, ["launch", "release"]),
    ("a new model or major product release", "OpenAI faces a lawsuit from authors over training data", False, ["launch", "release"]),
    ("a company being bought or merging", "Broadcom agrees to acquire cloud software maker in $61 billion deal", True, ["merger", "acquisition"]),
    ("a company being bought or merging", "Two regional banks to combine, creating a $90 billion lender", True, ["merger", "acquisition"]),
    ("a company being bought or merging", "Broadcom raises its dividend by 10 percent", False, ["merger", "acquisition"]),
    ("a company being bought or merging", "Analysts debate whether the acquisition of talent matters more than chips", False, ["merger", "acquisition"]),
    ("layoffs or job cuts", "Intel to cut 15% of its workforce as it restructures", True, ["layoffs"]),
    ("layoffs or job cuts", "Intel says it will eliminate thousands of roles in a cost-cutting push", True, ["layoffs"]),
    ("layoffs or job cuts", "Intel hires a new head of foundry", False, ["layoffs"]),
    ("layoffs or job cuts", "Intel opens a new fab in Ohio, creating 3,000 jobs", False, ["layoffs"]),
    ("the Federal Reserve changing interest rates", "Fed cuts rates by a quarter point, citing a cooling job market", True, ["rate cut"]),
    ("the Federal Reserve changing interest rates", "Fed holds rates steady but signals a cut is coming in December", True, ["rate cut"]),
    ("the Federal Reserve changing interest rates", "Fed governor gives a speech on community banking", False, ["rate cut"]),
    ("the Federal Reserve changing interest rates", "Mortgage applications fall as home prices climb", False, ["rate cut"]),
]
ABOUT_HOLDOUT = [
    ("their quarterly results", "AMD profit jumps 40% as AI chip demand outpaces forecasts", True, ["earnings"]),
    ("their quarterly results", "AMD names a new head of its graphics division", False, ["earnings"]),
    ("their quarterly results", "AMD revenue guidance for the holiday quarter disappoints investors", True, ["earnings"]),
    ("their quarterly results", "AMD and Sony announce a partnership on handheld chips", False, ["earnings"]),
    ("a product recall or safety problem", "Ford pauses deliveries of its electric pickup over a battery fault", True, ["recall"]),
    ("a product recall or safety problem", "Ford unveils a cheaper electric truck for 2028", False, ["recall"]),
    ("a product recall or safety problem", "Ford dealers brace for a busy fall selling season", False, ["recall"]),
    ("a new model or major product release", "Anthropic releases a new Claude model with a larger context window", True, ["launch"]),
    ("a new model or major product release", "Anthropic opens an office in Tokyo", False, ["launch"]),
    ("a new model or major product release", "Google rolls out Gemini 4 to all Workspace users", True, ["launch"]),
    ("a company being bought or merging", "Salesforce in talks to buy a data analytics startup", True, ["acquisition"]),
    ("a company being bought or merging", "Salesforce reports a steady quarter", False, ["acquisition"]),
    ("layoffs or job cuts", "Meta lets go of hundreds of employees in reorganization", True, ["layoffs"]),
    ("layoffs or job cuts", "Meta's new smart glasses sell out in a day", False, ["layoffs"]),
    ("the Federal Reserve changing interest rates", "Central bank lowers its benchmark rate for the first time in two years", True, ["rate cut"]),
    ("the Federal Reserve changing interest rates", "Treasury yields drift as traders wait for jobs data", False, ["rate cut"]),
]


def round_trip(jev: Jev, state: dict, questions: dict) -> dict:
    return jev.client.ask(state, questions).answers


# --------------------------------------------------------------------------- collecting
def collect_tool(jev: Jev, requests: list[str], with_newsroom: bool, runs: int) -> list[dict]:
    own = {t: TOOLS[t].description for t in DAILY_TOOLS if with_newsroom or t != "newsroom"}
    rows = []
    for req in requests:
        scores = [jev.pick_tools(req, own).scores for _ in range(runs)]
        rows.append({"request": req, "runs": scores})
    return rows


def collect_route(jev: Jev, cfg, requests: list[tuple[str, str]], runs: int) -> dict:
    personas = load_personas(cfg.personas_dir)
    old = {n: p for n, p in personas.items()}
    from dataclasses import replace
    old[D] = replace(personas[D], description=OLD_DESCRIPTION)
    out = {"old": [], "new": []}
    for arm, ps in (("old", old), ("new", personas)):
        for text, want in requests:
            picks = [jev.route_persona(text, ps) for _ in range(runs)]
            out[arm].append({"request": text, "want": want, "runs": [{"persona": p.persona, "conf": p.confidence,
                                                                       "probs": p.probabilities} for p in picks]})
    return out


def collect_about(jev: Jev, cases: list[tuple], runs: int) -> list[dict]:
    rows = []
    for about, headline, label, kws in cases:
        scores = []
        for _ in range(runs):
            a = round_trip(jev, {"item": headline[:700], "interest": about}, {"match": noul(nr.ABOUT_Q)})
            scores.append(a["match"].noul)
        rows.append({"about": about, "headline": headline, "label": label, "keywords": kws, "runs": scores})
    return rows


# --------------------------------------------------------------------------- scoring
def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def tool_rates(rows: list[dict], wants: list[bool], thr: float) -> dict:
    """Over all runs of all rows: how often it fired when it should, and when it should not."""
    fire = [s.get("newsroom", 0.0) >= thr for r, w in zip(rows, wants) if w for s in r["runs"]]
    quiet = [s.get("newsroom", 0.0) >= thr for r, w in zip(rows, wants) if not w for s in r["runs"]]
    return {"recall": mean(fire), "false_rate": mean(quiet), "false_n": sum(quiet), "quiet_n": len(quiet), "fire_n": len(fire)}


def about_decisions(r: dict, thr: float) -> dict:
    kw = nr.alert_matches({"keywords": r["keywords"], "exclude": [], "all": False, "about": ""}, {"title": r["headline"]}) is not None
    ab = [s >= thr for s in r["runs"]]
    return {"kw": [kw] * len(ab), "about": ab, "either": [kw or a for a in ab]}


def prf(rows: list[dict], thr: float) -> dict:
    out = {}
    for arm in ("kw", "about", "either"):
        tp = fp = fn = 0
        for r in rows:
            for got in about_decisions(r, thr)[arm]:
                tp += got and r["label"]
                fp += got and not r["label"]
                fn += (not got) and r["label"]
        out[arm] = {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else 1.0,
                    "recall": tp / (tp + fn) if tp + fn else 1.0}
    missed_by_kw = [r for r in rows if r["label"] and not about_decisions(r, thr)["kw"][0]]
    found = mean(about_decisions(r, thr)["about"][i] for r in missed_by_kw for i in range(len(r["runs"]))) if missed_by_kw else None
    out["found_what_keywords_miss"] = found
    out["missed_by_keywords_n"] = len(missed_by_kw)
    return out


# --------------------------------------------------------------------------- reporting
class Out:
    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)


def report_tool(out: Out, raw: dict, thr: float, which: str) -> None:
    out(f"\n== tool: is the newsroom guide in the prompt?  (threshold {thr}, thresholds.tool; {which} requests)")
    sets = [(f"{which}", raw["tool"][which])]
    for name, rows in sets:
        wants = [r["want"] for r in rows]
        res = tool_rates(rows, wants, thr)
        out(f"  {name}: fires on {res['recall']:.0%} of {res['fire_n'] // max(1, len(rows[0]['runs']))} should-fire requests (target >= 90%); "
            f"fires on {res['false_rate']:.0%} of {res['quiet_n'] // max(1, len(rows[0]['runs']))} lookalikes (target <= 5%, {res['false_n']} wrong picks)")
        out("  sweep:  " + "  ".join(f"{t:.2f}: {tool_rates(rows, wants, t)['recall']:.0%}/{tool_rates(rows, wants, t)['false_rate']:.0%}" for t in THRESHOLDS)
            + "   (fires when it should / fires when it should not)")
        for r in rows:
            hit = [s.get("newsroom", 0.0) >= thr for s in r["runs"]]
            if (r["want"] and not all(hit)) or (not r["want"] and any(hit)):
                out(f"    MISS {'(should fire)' if r['want'] else '(should not fire)'}: {r['request'][:84]}  newsroom={mean(s.get('newsroom', 0) for s in r['runs']):.2f}  "
                    + " ".join(f"{k}={mean(s.get(k, 0) for s in r['runs']):.2f}" for k in DAILY_TOOLS if k != "newsroom"))
    reg = raw["tool"].get("regress")
    if reg:
        res = tool_rates(reg["new"], [False] * len(reg["new"]), thr)
        out(f"  regression ({len(reg['new'])} requests from tool_cases.py, none wants it): picked on {res['false_n']} (target 0)")
        for r in reg["new"]:
            if any(s.get("newsroom", 0.0) >= thr for s in r["runs"]):
                out(f"    PICKED: {r['request'][:90]}  newsroom={mean(s['newsroom'] for s in r['runs']):.2f}")
        if reg.get("old"):
            worst = 0.0
            for a, b in zip(reg["old"], reg["new"]):
                for k in DAILY_TOOLS:
                    if k == "newsroom":
                        continue
                    worst = max(worst, abs(mean(s.get(k, 0) for s in a["runs"]) - mean(s.get(k, 0) for s in b["runs"])))
            out(f"  the other tools' scores with and without the newsroom among the options: largest difference {worst:.3f} (target < 0.05)")


def report_route(out: Out, raw: dict, which: str) -> None:
    r = raw["route"]
    out("\n== route: does the new daily_assistant description keep the routing right?  (old vs new, same requests)")
    for part in ("newsroom", "existing"):
        for arm in ("old", "new"):
            rows = [x for x in r[arm] if x["part"] == part]
            ok = mean(p["persona"] == x["want"] for x in rows for p in x["runs"])
            out(f"  {part:9s} {arm}: {ok:.0%} of {len(rows)} requests reach {'daily_assistant' if part == 'newsroom' else 'the expected persona'}"
                + ("   (target >= 90%)" if part == "newsroom" and arm == "new" else ""))
    for a, b in zip(r["old"], r["new"]):
        pa, pb = [p["persona"] for p in a["runs"]], [p["persona"] for p in b["runs"]]
        if pa != pb:
            out(f"    changed: {a['request'][:80]}  {pa[0]} -> {pb[0]}  (want {a['want']})")
    for b in r["new"]:
        if any(p["persona"] != b["want"] for p in b["runs"]):
            out(f"    WRONG (new): {b['request'][:80]}  -> {b['runs'][0]['persona']}  (want {b['want']})")


def report_about(out: Out, rows: list[dict], thr: float, label: str) -> None:
    out(f"\n== about: Jev's `about` alert on {len(rows)} labelled headlines, {label}  (threshold {thr}, thresholds.newsroom_about; authored, not from real runs)")
    res = prf(rows, thr)
    for arm, name in (("kw", "keywords only"), ("about", "about only"), ("either", "keywords or about")):
        r = res[arm]
        out(f"  {name:18s} precision {r['precision']:.0%}  recall {r['recall']:.0%}  (found {r['tp']}, wrong {r['fp']}, missed {r['fn']})")
    if res["found_what_keywords_miss"] is not None:
        out(f"  of the {res['missed_by_keywords_n']} headlines the keywords miss, `about` finds {res['found_what_keywords_miss']:.0%} (target >= 70%)")
    out("  sweep (threshold: about precision / recall):  " + "  ".join(f"{t:.2f}: {prf(rows, t)['about']['precision']:.0%}/{prf(rows, t)['about']['recall']:.0%}"
                                                                      for t in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)))
    for r in rows:
        d = about_decisions(r, thr)["about"]
        if any(x != r["label"] for x in d):
            out(f"    {'MISSED' if r['label'] else 'FALSE ALARM'}: [{r['about']}] {r['headline'][:78]}  score {mean(r['runs']):.2f}")


# --------------------------------------------------------------------------- real items
def export_items(path: str) -> None:
    cfg = load_config()
    room = nr.Newsroom(cfg.workspace / "newsroom.db", nr.settings(cfg.raw))
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for w in room.widgets():
            a = w["alert"]
            for it in room.items(w["id"], 100):
                f.write(json.dumps({"widget": w["title"], "about": a.get("about") or "", "keywords": a.get("keywords") or [],
                                    "headline": it["title"] + (" - " + it["summary"] if it["summary"] else ""), "label": None},
                                   ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} items from {len(room.widgets())} widgets to {path}: set \"label\" to true (should notify) or false on each line "
          "(and \"about\" on lines whose widget has none), then run --labelled")


def labelled(path: str) -> list[tuple]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(r["about"], r["headline"], bool(r["label"]), r.get("keywords") or []) for r in rows if r.get("label") is not None and r.get("about")]


# --------------------------------------------------------------------------- main
def tool_sets(holdout: bool) -> tuple[str, list[tuple[str, bool]]]:
    return ("holdout", TOOL_HOLDOUT) if holdout else ("dev", TOOL_DEV)


def route_requests(holdout: bool) -> list[tuple[str, str, str]]:
    import routing_cases
    _, tool = tool_sets(holdout)
    return [(t, D, "newsroom") for t, fire in tool if fire] + [(t, w, "existing") for t, w, _ in routing_cases.CASES]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", default="tool,route,about", help="comma separated: tool, route, about")
    ap.add_argument("--holdout", action="store_true", help="the held-out requests and headlines")
    ap.add_argument("--offline", action="store_true", help="re-report the saved raw scores, no model calls")
    ap.add_argument("--runs", type=int, default=1, help="ask each case this many times (a hosted Jev can vary)")
    ap.add_argument("--no-regress", action="store_true", help="skip the tool_cases.py regression requests")
    ap.add_argument("--export", metavar="FILE", help="write the items on your board to FILE for labelling, then stop")
    ap.add_argument("--labelled", metavar="FILE", help="score the about stage on your own labelled items")
    a = ap.parse_args()
    if a.export:
        return export_items(a.export)
    cfg = load_config()
    stages = {s.strip() for s in a.stage.split(",")}
    which, tool = tool_sets(a.holdout)
    raw_file, summary_file = (RAW_HOLDOUT, SUMMARY_HOLDOUT) if a.holdout else (RAW, SUMMARY)
    raw = json.loads(raw_file.read_text(encoding="utf-8")) if a.offline and raw_file.exists() else {}
    out = Out()
    t0 = time.time()
    if not a.offline:
        jev = Jev(cfg)
        if "tool" in stages:
            rows = collect_tool(jev, [r for r, _ in tool], True, a.runs)
            for row, (_, want) in zip(rows, tool):
                row["want"] = want
            raw.setdefault("tool", {})[which] = rows
            if not a.no_regress:
                import tool_cases
                reqs = [r for r, _, _ in tool_cases.CASES + tool_cases.HOLDOUT]
                raw["tool"]["regress"] = {"new": collect_tool(jev, reqs, True, a.runs), "old": collect_tool(jev, reqs, False, a.runs)}
        if "route" in stages:
            reqs = route_requests(a.holdout)
            res = collect_route(jev, cfg, [(t, w) for t, w, _ in reqs], a.runs)
            for arm in res:
                for row, (_, _, part) in zip(res[arm], reqs):
                    row["part"] = part
            raw["route"] = res
        if "about" in stages:
            cases = labelled(a.labelled) if a.labelled else (ABOUT_HOLDOUT if a.holdout else ABOUT_DEV)
            raw.setdefault("about", {})["labelled" if a.labelled else which] = collect_about(jev, cases, a.runs)
        raw["meta"] = {"jev": cfg.jev.get("backend"), "calls": jev.client.calls if hasattr(jev.client, "calls") else None,
                       "seconds": round(time.time() - t0, 1), "runs": a.runs, "when": time.strftime("%Y-%m-%d %H:%M")}
        raw_file.write_text(json.dumps(raw, indent=1), encoding="utf-8")
    meta = raw.get("meta") or {}
    out(f"newsroom decisions on Jev backend {meta.get('jev') or cfg.jev.get('backend')}, {meta.get('runs', a.runs)} run(s) per case, "
        f"{meta.get('when', '')}  [{which}]  NOT TUNED: thresholds.tool is the harness's own, newsroom_about is a guess")
    if "tool" in stages and raw.get("tool", {}).get(which):
        report_tool(out, raw, cfg.t("tool"), which)
    if "route" in stages and raw.get("route"):
        report_route(out, raw, which)
    if "about" in stages:
        key = "labelled" if a.labelled else which
        if raw.get("about", {}).get(key):
            report_about(out, raw["about"][key], cfg.t("newsroom_about"), key)
    summary_file.write_text("\n".join(out.lines) + "\n", encoding="utf-8")
    print(f"\nraw scores: {raw_file.name}   summary: {summary_file.name}")


if __name__ == "__main__":
    main()
