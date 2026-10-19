"""Live bench for the Newsroom (jevh/newsroom.py, docs/newsroom.md): real sources, a real Jev, a real worker model, real turns.

First run 2026-10-09 (Qwen3.8 27B on llama.cpp, semif; results in docs/newsroom.md's status note; the saved summaries are
newsroom_last_summary.txt (turns + dashboard), newsroom_sources_summary.txt and newsroom_recheck_last_summary.txt). It was written
in a cloud session that could reach no news site and no model. The offline tests
(tests/test_newsroom*.py) use a feed served from localhost, a fake Jev and scripted models: they show the code runs. This
shows whether it WORKS: does a plain request in the chat put the right widget on the board, does a lookalike request
(a one-off "what's the news on X?", "follow up with Sam") leave the board alone, do real feeds parse and stay quiet on the
second look, and does a new matching story reach the user as a notification, once.

    python evals/bench/newsroom.py --model lmstudio:qwen3.8-27b@q6_k            # every stage, 2 reps
    python evals/bench/newsroom.py --stage sources                              # the real feeds only (no model needed)
    python evals/bench/newsroom.py --stage turns --arm both --reps 3 --model ...   # old harness vs this one, on every task
    python evals/bench/newsroom.py --stage turns --group track --pin               # only the "set it up" requests, daily_assistant pinned
    python evals/bench/newsroom.py --stage dashboard --model ...                   # the whole path, through the dashboard's API and runner
    python evals/bench/newsroom.py --report                                        # the last run's summary
    python evals/bench/newsroom.py --dry                                           # plumbing check with a fake Jev and scripted worker (numbers mean nothing)

Pin the model with --model (the harness's LM Studio gate unloads every other model when a task needs one). Nothing touches
your real workspace or board: every rep runs in evals/bench/runs/newsroom/<stamp>/<arm>-<task>-<n>/ with its own board, on
a copy of personas/. Feeds served from this machine (127.0.0.1) are allowed for the bench only (JEVH_NEWSROOM_ALLOW_LOCAL).
Needs: the Jev adapter (config jev.url) and a worker model for turns/dashboard; internet for sources and the news/web
tasks; for X, one of the three set-ups in docs/newsroom.md (without one, the X task checks that the worker says what is
missing and keeps the widget).

Stages (pass targets are first guesses, printed beside each result: a miss is a finding, not a bug in the bench)
  sources    every widget kind against its real source (the examples in SOURCES): at least one item with a title and a
             link, a latency, how many carry a date, and a SECOND look right after the first that finds no new item
             (target 0: a feed whose items change identity between looks would notify on every refresh)
  turns      full harness turns, each task several times, on a fresh board. Groups: track (set a widget up from a plain
             request), manage (change/remove/list what is on a pre-filled board), lookalike (requests that sound like
             tracking but are not: the board must stay untouched), ordinary (everyday requests: cost must not grow).
             Arms: `new` is this harness, `old` is the daily_assistant as it was (no newsroom tool, the old description).
             By default `old` runs only the lookalike and ordinary groups (it has no tool to set widgets up with); --arm both
             runs it on every task. Reported per task: pass rate over the reps, and per group model calls, tokens, seconds,
             steps (code blocks), questions asked of the user and the share routed to daily_assistant.
  dashboard  the dashboard's own server with a real harness: chat "keep an eye on this feed and tell me when it mentions
             outage" -> the widget is on the board with the first look done and NO notification yet -> a matching story is
             added to the feed -> the runner (tick 2 s) makes exactly one notification and the page's event stream
             carries it -> nothing repeats on the next look -> "remove that widget" in the chat removes it

If the feature is wrong, expect: the worker writing its own polling loop instead of newsroom.add; a widget made for a
one-off news question; alert words that are not the ones asked for; the same story notifying twice; every refresh
notifying (unstable item identity in a real feed); X widgets never created because the worker stops at "X isn't set up";
a worker that says it set something up when the board is empty (the checks read the board, not the answer); an answer
that doesn't say what will notify the user.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import statistics
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "evals"))

import _out  # noqa: E402  (evals/_out.py: where a run's results go)
import yaml  # noqa: E402

from jevh import newsroom as nr  # noqa: E402
from jevh.config import load_config  # noqa: E402

RUNS = HERE / "runs" / "newsroom"
RAW = HERE / "newsroom_raw.json"
SUMMARY = HERE / "newsroom_last_summary.txt"
OLD_DESCRIPTION = (
    "General everyday assistant: questions, planning the day, calendar and appointments,\n"
    "email, reminders, notes, quick lookups, business paperwork (quotes, payroll), smart home,\n"
    "and connecting the user's own accounts (Gmail, Google Calendar, GitHub, Slack) to the assistant.")

# --------------------------------------------------------------------------- real sources
SOURCES = [   # (kind, settings, what must be true of the first item, if anything)
    ("news", {"topic": "Nvidia"}),
    ("news", {"topic": "\"solid-state battery\"", "site": "reuters.com", "days": 30}),
    ("rss", {"url": "https://feeds.bbci.co.uk/news/technology/rss.xml"}),
    ("hn", {"query": "llama.cpp"}),
    ("github", {"repo": "ggml-org/llama.cpp"}),
    ("reddit", {"subreddit": "LocalLLaMA"}),
    ("arxiv", {"query": "speculative decoding"}),
    ("web", {"query": "llama.cpp release notes"}),
    ("x", {"user": "OpenAI"}),
]


def run_sources(*_a, **_k) -> dict:
    out = {}
    tmp = RUNS / "sources-board"
    shutil.rmtree(tmp, ignore_errors=True)
    for kind, params in SOURCES:
        key = f"{kind} {json.dumps(params, ensure_ascii=False)}"
        room = nr.Newsroom(tmp / f"{abs(hash(key))}.db", {"min_minutes": 1})
        row: dict = {"kind": kind, "params": params}
        needs = nr.KINDS[kind].ready()
        if needs:
            out[key] = {**row, "skipped": needs}
            continue
        t = time.time()
        try:
            w = room.add_widget(kind, params)
            first = room.refresh(w["id"])
            row["seconds"] = round(time.time() - t, 1)
            items = room.items(w["id"], 100)
            row.update(error=first["error"], items=len(items),
                       with_link=sum(1 for i in items if i["url"].startswith("http")),
                       dated=sum(1 for i in items if i["published"]),
                       sample=[i["title"][:90] for i in items[:2]])
            if not first["error"]:
                time.sleep(3)
                second = room.refresh(w["id"])
                row["second_look_new"] = second["new"]
                row["second_look_error"] = second["error"]
        except Exception as e:  # noqa: BLE001 - a source failing is the finding
            row.update(error=f"{type(e).__name__}: {e}"[:300], seconds=round(time.time() - t, 1))
        out[key] = row
    return out


# --------------------------------------------------------------------------- a feed on this machine
STORIES = [("Acme data center outage disrupts payments for hours", "https://ex.test/outage-1", "Customers could not pay"),
           ("Acme opens a new office in Lisbon", "https://ex.test/office", "Growth"),
           ("Acme quarterly revenue up 8 percent", "https://ex.test/revenue", "Results"),
           ("Acme names a new chief technology officer", "https://ex.test/cto", "People")]


def rss(items) -> bytes:
    body = "".join(f"<item><title>{t}</title><link>{u}</link><guid>{u}</guid><description>{d}</description></item>" for t, u, d in items)
    return f'<?xml version="1.0"?><rss version="2.0"><channel><title>Acme Newsroom</title>{body}</channel></rss>'.encode()


class LocalFeed:
    def __init__(self, items=STORIES):
        feed = self
        self.body = rss(items)

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/rss+xml")
                self.end_headers()
                self.wfile.write(feed.body)

            def log_message(self, *a):
                pass
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/feed.xml"

    def set(self, items) -> None:
        self.body = rss(items)

    def close(self) -> None:
        self.srv.shutdown()


# --------------------------------------------------------------------------- the tasks
class Ctx:
    """What a check sees: the board before and after, the answer, and the feed."""

    def __init__(self, room, before, answer, feed, rev0, routed):
        self.room, self.before, self.answer, self.feed, self.rev0, self.routed = room, before, answer, feed, rev0, routed
        self.after = room.widgets()

    def one(self, **want) -> dict | None:
        for w in self.after:
            if all(_has(w, k, v) for k, v in want.items()):
                return w
        return None

    def says(self, *words) -> bool:
        return any(w.lower() in self.answer.lower() for w in words)


def _has(w: dict, key: str, value) -> bool:
    if key == "kind":
        return w["kind"] == value
    if key == "param":   # (name, substring)
        n, sub = value
        return sub.lower() in str(w["params"].get(n, "")).lower()
    if key == "alert_word":
        return any(x.lower().startswith(value.lower()) for x in w["alert"].get("keywords", []))
    if key == "alert_all":
        return bool(w["alert"].get("all")) == value
    if key == "title":
        return value.lower() in w["title"].lower()
    return w.get(key) == value


def alert_covers(w: dict | None, word: str) -> bool:
    """The widget notifies on `word`: as a keyword, as an `about` sentence, or on everything."""
    if not w:
        return False
    a = w["alert"]
    return bool(a.get("all")) or any(k.lower().startswith(word.lower()[:5]) for k in a.get("keywords", [])) or word.lower()[:5] in (a.get("about") or "").lower()


SEED = [("news", {"topic": "Nvidia"}, {"keywords": ["earnings"]}), ("news", {"topic": "Bitcoin"}, None),
        ("reddit", {"subreddit": "LocalLLaMA"}, None), ("news", {"topic": "Tesla"}, None)]


# name -> (group, prompt(feed) , seed?, checks(c) -> {name: bool}, note)
TASKS: dict[str, dict] = {
    "nvidia_earnings": {"group": "track", "prompt": lambda f: "Keep an eye on Nvidia news and tell me when they mention earnings.",
        "checks": lambda c: {"one_news_widget_on_nvidia": len(c.after) == 1 and c.one(kind="news", param=("topic", "nvidia")) is not None,
                             "notifies_on_earnings": alert_covers(c.one(kind="news"), "earnings"),
                             "answer_says_what_notifies": c.says("earnings")}},
    "openai_on_x": {"group": "track", "prompt": lambda f: "Follow @OpenAI on X and alert me to every new post.",
        "checks": lambda c: {"x_widget_for_openai": c.one(kind="x", param=("user", "openai")) is not None,
                             "alerts_on_every_post": bool((c.one(kind="x") or {"alert": {}})["alert"].get("all")),
                             "says_what_x_needs_when_not_set_up": nr.KINDS["x"].ready() is None or c.says("token", "bridge", "nango", "connect", "set up", "set-up")}},
    "own_feed": {"group": "track", "prompt": lambda f: f"Add this feed to my newsroom: {f.url} and notify me about anything that mentions outage.",
        "checks": lambda c: {"rss_widget_for_the_feed": c.one(kind="rss", param=("url", c.feed.url)) is not None,
                             "notifies_on_outage": alert_covers(c.one(kind="rss"), "outage"),
                             "first_look_done": bool((c.one(kind="rss") or {}).get("last_ok")),
                             "no_alert_for_what_was_already_there": len(c.room.notes()) == 0}},
    "two_topics": {"group": "track", "prompt": lambda f: "Track news about solid-state batteries and also about Tesla, as two separate widgets.",
        "checks": lambda c: {"two_widgets": len(c.after) == 2,
                             "one_on_batteries": c.one(param=("topic", "battery")) is not None or c.one(param=("topic", "batteries")) is not None,
                             "one_on_tesla": c.one(param=("topic", "tesla")) is not None}},
    "llama_cpp_releases": {"group": "track", "prompt": lambda f: "Watch new releases of the llama.cpp project on GitHub.",
        "checks": lambda c: {"github_widget": c.one(kind="github", param=("repo", "llama.cpp")) is not None}},
    "fed_rates": {"group": "track", "prompt": lambda f: "Keep me posted on the Fed's interest rate decisions, and ping me if rates change.",
        "checks": lambda c: {"a_news_widget_on_the_fed_or_rates": any(w["kind"] in ("news", "web") and any(x in str(w["params"]).lower() for x in ("fed", "rate")) for w in c.after),
                             "it_notifies_on_something": any(w["alert"].get("all") or w["alert"].get("keywords") or w["alert"].get("about") for w in c.after)}},
    "same_twice": {"group": "track", "seed": [("news", {"topic": "Nvidia"}, None)],
        "prompt": lambda f: "Track Nvidia news for me.",
        "checks": lambda c: {"not_doubled": len(c.after) == 1}},
    "vague": {"group": "track", "report_only": True, "prompt": lambda f: "Keep me posted on AI.",
        "checks": lambda c: {"made_something_sensible_or_asked": len(c.after) >= 1 or c.says("?", "which", "what kind", "specific")}},
    "remove_bitcoin": {"group": "manage", "seed": SEED, "prompt": lambda f: "Remove the bitcoin widget.",
        "checks": lambda c: {"bitcoin_gone": c.one(title="bitcoin") is None, "the_other_three_stay": len(c.after) == 3}},
    "tighten_nvidia": {"group": "manage", "seed": SEED, "prompt": lambda f: "Check the Nvidia one every 10 minutes and also alert me on guidance.",
        "checks": lambda c: {"every_10_minutes": (c.one(title="nvidia") or {}).get("every") == 10,
                             "guidance_added": alert_covers(c.one(title="nvidia"), "guidance"),
                             "earnings_kept": alert_covers(c.one(title="nvidia"), "earnings"), "no_other_change": len(c.after) == 4}},
    "pause_tesla": {"group": "manage", "seed": SEED, "prompt": lambda f: "Pause the Tesla tile.",
        "checks": lambda c: {"tesla_paused": bool((c.one(title="tesla") or {}).get("paused")),
                             "others_running": sum(1 for w in c.after if w["paused"]) == 1}},
    "list_board": {"group": "manage", "seed": SEED, "prompt": lambda f: "What's on my newsroom?",
        "checks": lambda c: {"board_untouched": c.room.rev() == c.rev0, "names_all_four": all(c.says(x) for x in ("nvidia", "bitcoin", "tesla", "localllama"))}},
    "alerts_off": {"group": "manage", "seed": SEED, "prompt": lambda f: "Stop alerting me about Nvidia earnings but keep showing the news.",
        "checks": lambda c: {"nvidia_still_there": c.one(title="nvidia") is not None,
                             "no_alert_left": not any((c.one(title="nvidia") or {"alert": {}})["alert"].get(k) for k in ("keywords", "all", "about"))}},
    "latest_nvidia": {"group": "lookalike", "prompt": lambda f: "What's the latest news on Nvidia?",
        "checks": lambda c: {"no_widget": len(c.after) == 0, "answered": len(c.answer) > 40}},
    "top_headlines": {"group": "lookalike", "prompt": lambda f: "Summarize today's top tech headlines for me.",
        "checks": lambda c: {"no_widget": len(c.after) == 0, "answered": len(c.answer) > 40}},
    "follow_up": {"group": "lookalike", "prompt": lambda f: "Remind me to follow up with Sam about the quote at 3pm tomorrow.",
        "checks": lambda c: {"no_widget": len(c.after) == 0, "answered": len(c.answer) > 10}},
    "one_off_post": {"group": "lookalike", "prompt": lambda f: "What did OpenAI post on X yesterday?",
        "checks": lambda c: {"no_widget": len(c.after) == 0, "answered": len(c.answer) > 20}},
    "research_report": {"group": "lookalike", "prompt": lambda f: "Research solid-state batteries and write me a one-paragraph summary.",
        "checks": lambda c: {"no_widget": len(c.after) == 0, "answered": len(c.answer) > 80}},
    "server_cpu": {"group": "lookalike", "prompt": lambda f: "Tell me how I could monitor my server's CPU and alert me above 90%.",
        "checks": lambda c: {"no_widget": len(c.after) == 0, "answered": len(c.answer) > 80}},
    "percent": {"group": "ordinary", "prompt": lambda f: "What's 17% of 2,340?",
        "checks": lambda c: {"no_widget": len(c.after) == 0, "right": "397.8" in c.answer}},
    "plan_day": {"group": "ordinary", "prompt": lambda f: "Plan my day around a 2pm dentist appointment and an hour of focused work.",
        "checks": lambda c: {"no_widget": len(c.after) == 0, "answered": len(c.answer) > 60}},
    "draft_email": {"group": "ordinary", "prompt": lambda f: "Draft a two-sentence email telling my client the quote will be a day late.",
        "checks": lambda c: {"no_widget": len(c.after) == 0, "answered": len(c.answer) > 60}},
}
GROUPS = ("track", "manage", "lookalike", "ordinary")
TARGETS = {"track": "right widget on >= 80% of reps", "manage": "right change on >= 90%", "lookalike": "board untouched on 100% (a widget for a one-off question is the failure)",
           "ordinary": "100% as before, cost within 15% of old"}


# --------------------------------------------------------------------------- rigs
def make_cfg(where: Path, arm: str, tick: float | None = None):
    cfg = load_config()
    shutil.rmtree(where, ignore_errors=True)
    cfg.workspace = where / "ws"
    cfg.personas_dir = where / "personas"
    cfg.personas_dirs = [cfg.personas_dir]
    shutil.copytree(ROOT / "personas", cfg.personas_dir, ignore=shutil.ignore_patterns("learned"))
    # No persona in this rep's copy has the desktop tool: it operates the real screen (the user's windows, tabs and
    # sign-ins), and a bench worker drove their Chrome for 24 minutes trying to follow an X account (2026-10-09). Only
    # this copy is changed: personas/ itself is never edited, so nothing needs turning back on.
    for f in cfg.personas_dir.glob("*.yaml"):
        data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        if "desktop" in (data.get("tools") or []):
            data["tools"] = [t for t in data["tools"] if t != "desktop"]
            f.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    cfg.skills_dirs = [where / "skills", ROOT / "skills"]
    (where / "skills").mkdir(parents=True, exist_ok=True)
    if arm == "old":   # the daily assistant as it was before the Newsroom
        f = cfg.personas_dir / "daily_assistant.yaml"
        data = yaml.safe_load(f.read_text(encoding="utf-8"))
        data["tools"] = [t for t in data["tools"] if t != "newsroom"]
        data["description"] = OLD_DESCRIPTION
        f.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    if tick is not None:
        cfg.raw = {**cfg.raw, "newsroom": {**(cfg.raw.get("newsroom") or {}), "tick": tick}}
    return cfg


def make_harness(cfg, model: str | None, dry: dict | None, on_event):
    from jevh.orchestrator import Harness
    if not dry:
        h = Harness(cfg, on_event=on_event, model_override=model)
    else:   # plumbing only: a fake Jev and a scripted worker (the final() gate asks Jev, which here knows nothing: off)
        cfg.raw = {**cfg.raw, "verify": {**(cfg.raw.get("verify") or {}), "enabled": False}, "failures": {**(cfg.raw.get("failures") or {}), "audit": 0}}
        from conftest import FakeSystemOne, ScriptedLLM, choice_ans, noul_ans, py
        from jevh.jev import Jev

        def rules(k, q, s):
            if k == "persona":
                return choice_ans("daily_assistant", 0.9, q["criteria"])
            if k == "t_newsroom" and dry["fire"]:
                return noul_ans(0.9)
            if k in ("needs_high_end_dev", "in_scope", "necessary"):
                return noul_ans(0.9)
        h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: ScriptedLLM([py(dry["code"])]), on_event=on_event)
    h.learning = False
    return h


def dry_script(name: str, task: dict, feed) -> dict:
    """What a scripted worker does for each task in --dry (the right thing, so every check can be seen to pass)."""
    code = {"nvidia_earnings": "newsroom.add('news', topic='Nvidia', alert='earnings')\nfinal('Added a news widget for Nvidia; I will notify you about earnings.')",
            "openai_on_x": "newsroom.add('x', user='OpenAI', alert=True)\nfinal('Added @OpenAI. X needs a token or a bridge set up; connect it and it starts working.')",
            "own_feed": f"newsroom.add('rss', url={feed.url!r}, alert='outage')\nfinal('Added the feed; I will notify you about outage.')",
            "two_topics": "newsroom.add('news', topic='solid-state batteries')\nnewsroom.add('news', topic='Tesla')\nfinal('Added two widgets.')",
            "llama_cpp_releases": "newsroom.add('github', repo='ggml-org/llama.cpp', alert=True)\nfinal('Watching the releases.')",
            "fed_rates": "newsroom.add('news', topic='Fed interest rate decision', alert='rate')\nfinal('Added; I will notify you about rate changes.')",
            "same_twice": "newsroom.add('news', topic='Nvidia')\nfinal('That one was already on your newsroom.')",
            "vague": "final('Which part of AI: research papers, a company, or product launches?')",
            "remove_bitcoin": "newsroom.remove('bitcoin')\nfinal('Removed it.')",
            "tighten_nvidia": "newsroom.edit('nvidia', every=10, alert='earnings, guidance')\nfinal('Done.')",
            "pause_tesla": "newsroom.edit('tesla', paused=True)\nfinal('Paused.')",
            "list_board": "ws = newsroom.widgets()\nfinal('On your newsroom: ' + ', '.join(w['title'] for w in ws) + ' (localllama)')",
            "alerts_off": "newsroom.edit('nvidia', alert=None)\nfinal('Alerts off.')"}.get(name)
    if code is None:
        code = {"percent": "final('17% of 2,340 is 397.8')"}.get(name, "final('Here is a plain answer of a good length for the check to pass, said without any widget at all, thanks.')")
    return {"code": code, "fire": task["group"] in ("track", "manage")}


def choose_answer(q: dict) -> str:
    """The bench's reply to a worker's question: an option that skips (the user is not here to log in, connect or confirm
    anything: "Done - I'm logged in" was said for them once, and the worker then went on to act on the claim), else the
    first option; to a question without options, that the worker should carry on without it."""
    options = q.get("options") or []
    skip = next((o for o in options if "skip" in str(o).lower()), None)
    return skip or (options[0] if options else "Please carry on without that, using your best judgement.")


class Answerer(threading.Thread):
    """Answers the worker's questions to the user (choose_answer) and counts them."""

    def __init__(self, h):
        super().__init__(daemon=True)
        self.h, self.asked, self.stop = h, [], False

    def run(self):
        while not self.stop:
            for q in self.h.pending_questions():
                self.asked.append(q["question"][:160])
                self.h.answer(q["id"], choose_answer(q))
            time.sleep(0.3)


def run_turn(task_name: str, arm: str, n: int, where: Path, model, pin: bool, dry: bool, feed) -> dict:
    task = TASKS[task_name]
    cfg = make_cfg(where, arm)
    os.environ["JEVH_NEWSROOM_ALLOW_LOCAL"] = "1"
    events: list = []
    h = make_harness(cfg, model, dry_script(task_name, task, feed) if dry else None, lambda k, d: events.append((k, d)))
    room = nr.Newsroom(cfg.workspace / "newsroom.db", nr.settings(cfg.raw))
    for kind, params, alert in task.get("seed", []):
        room.add_widget(kind, params, alert=alert)
    before, rev0 = room.widgets(), room.rev()
    answerer = Answerer(h)
    answerer.start()
    t0 = time.time()
    row: dict = {"task": task_name, "arm": arm, "group": task["group"], "n": n}
    try:
        r = h.turn(task["prompt"](feed), force="daily_assistant" if pin or dry else None)
        row.update(answer=r.answer[:700], seconds=round(time.time() - t0, 1), routed=r.persona, cancelled=r.cancelled,
                   calls=r.usage.get("total", {}).get("calls"), tokens_in=r.usage.get("total", {}).get("input"),
                   tokens_out=r.usage.get("total", {}).get("output"))
        row["steps"] = sum(1 for k, _ in events if k == "code")
        row["bounces"] = sum(1 for k, d in events if k == "verify" and d.get("bounced"))
        ctx = Ctx(room, before, r.answer, feed, rev0, r.persona)
        row["checks"] = {k: bool(v) for k, v in task["checks"](ctx).items()}
        row["board"] = [{"kind": w["kind"], "title": w["title"], "params": w["params"], "alert": w["alert"], "every": w["every"], "paused": w["paused"]}
                        for w in ctx.after]
    except BaseException as e:  # noqa: BLE001 - shown in the result
        row["error"] = f"{type(e).__name__}: {e}"[:400]
        row["checks"] = {"ran": False}
    answerer.stop = True
    row["questions"] = len(answerer.asked)
    row["questions_text"] = answerer.asked[:3]
    return row


def run_turns(a, stamp: str) -> list[dict]:
    feed = LocalFeed()
    rows = []
    try:
        names = [n for n, t in TASKS.items() if (not a.group or t["group"] in a.group.split(",")) and (not a.task or n in a.task.split(","))]
        for name in names:
            group = TASKS[name]["group"]
            arms = ["new"] if a.arm == "new" else (["old", "new"] if a.arm == "both" or group in ("lookalike", "ordinary") else ["new"])
            if a.arm == "old":
                arms = ["old"]
            for arm in arms:
                for n in range(1, a.reps + 1):
                    print(f"  {arm} {name} #{n} ...", flush=True)
                    rows.append(run_turn(name, arm, n, RUNS / stamp / f"{arm}-{name}-{n}", a.model, a.pin, a.dry, feed))
    finally:
        feed.close()
    return rows


# --------------------------------------------------------------------------- the dashboard path
def run_dashboard(a, stamp: str) -> dict:
    import uvicorn

    from jevh.web.server import create_app
    feed = LocalFeed(STORIES[1:])   # the story that mentions "outage" is not there yet
    os.environ["JEVH_NEWSROOM_ALLOW_LOCAL"] = "1"
    cfg = make_cfg(RUNS / stamp / "dashboard", "new", tick=2)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    dry = dry_script("own_feed", TASKS["own_feed"], feed) if a.dry else None
    app = create_app(cfg, port=port, token="bench", model_override=a.model,
                     harness_factory=(lambda on_event, session=None: make_harness(cfg, a.model, dry, on_event)) if a.dry else None)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    end = time.time() + 15
    while not server.started and time.time() < end:
        time.sleep(0.05)
    base = f"http://127.0.0.1:{port}"
    hub = app.state.hub

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"X-Jevh-Token": "bench", "Content-Type": "application/json", "Origin": base})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    def wait_idle(timeout=900):
        end = time.time() + timeout
        while time.time() < end:
            if not call("/api/state")["busy"]:
                return True
            time.sleep(1)
        return False

    seen: list[dict] = []   # the page's event stream, as the dashboard sends it
    def listen():
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{base}/api/events?token=bench&since=0", headers={"Origin": base}), timeout=600) as r:
                for line in r:
                    if line.startswith(b"data: "):
                        seen.append(json.loads(line[6:]))
        except Exception:  # noqa: BLE001
            pass
    threading.Thread(target=listen, daemon=True).start()
    checks: dict = {}
    t0 = time.time()
    try:
        call("/api/settings", {"pin": "daily_assistant"})
        call("/api/chat", {"message": f"Keep an eye on this feed and tell me when it mentions outage: {feed.url}"})
        time.sleep(1)
        checks["turn_finished"] = wait_idle()
        board = call("/api/newsroom")
        w = (board["widgets"] or [None])[0]
        checks["widget_on_the_board"] = bool(w) and w["kind"] == "rss" and feed.url in str(w["params"])
        checks["notifies_on_outage"] = alert_covers(w, "outage")
        checks["first_look_done_with_items"] = bool(w) and len(w["items"]) >= 3
        checks["no_notification_for_what_was_already_there"] = board["unread_notes"] == 0
        feed.set(STORIES)      # the outage story arrives
        hub.newsroom._run("UPDATE widgets SET last_look=0")   # (its time has come)
        end = time.time() + 30
        while time.time() < end and not call("/api/newsroom")["notes"]:
            time.sleep(0.5)
        notes = call("/api/newsroom")["notes"]
        checks["one_notification_for_the_outage_story"] = len(notes) == 1 and "outage" in notes[0]["body"].lower()
        time.sleep(1)
        checks["the_page_got_the_alert_event"] = any(e["kind"] == "newsroom_alert" for e in seen)
        hub.newsroom._run("UPDATE widgets SET last_look=0")
        time.sleep(5)          # two more ticks
        checks["nothing_repeats_on_the_next_look"] = len(call("/api/newsroom")["notes"]) == 1
        if not a.dry:
            call("/api/chat", {"message": "Remove that widget from my newsroom."})
            time.sleep(1)
            wait_idle()
            checks["remove_in_the_chat_removes_it"] = call("/api/newsroom")["widgets"] == []
    except BaseException as e:  # noqa: BLE001
        checks["error"] = f"{type(e).__name__}: {e}"[:400]
    server.should_exit = True
    feed.close()
    return {"checks": checks, "seconds": round(time.time() - t0, 1)}


# --------------------------------------------------------------------------- report
def median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None


def summarize(res: dict) -> str:
    L = [f"NEWSROOM BENCH  {res['stamp']}  model={res.get('model') or 'the personas own'}  reps={res['reps']}  "
         f"{'DRY RUN (fake Jev, scripted worker): the numbers mean nothing' if res.get('dry') else 'NOT TUNED: every target is a first guess'}", ""]
    if "sources" in res:
        L.append("sources (real feeds)   target: >= 1 item with a link, second look finds 0 new")
        for key, r in res["sources"].items():
            if r.get("skipped"):
                L.append(f"  skip {key}: {r['skipped'][:140]}")
            elif r.get("error"):
                L.append(f"  FAIL {key}: {r['error'][:160]}")
            else:
                ok = r["items"] >= 1 and r["with_link"] >= 1 and not r.get("second_look_error") and r.get("second_look_new", 0) == 0
                L.append(f"  {'ok  ' if ok else 'MISS'} {key}: {r['items']} items, {r['with_link']} linked, {r['dated']} dated, {r['seconds']}s, "
                         f"second look new={r.get('second_look_new')}{' ' + r['second_look_error'] if r.get('second_look_error') else ''}   e.g. {r['sample'][:1]}")
        L.append("")
    rows = res.get("turns") or []
    if rows:
        L.append("turns (full harness turns on a fresh board each; checks read the board, not the answer)")
        for group in GROUPS:
            g = [r for r in rows if r["group"] == group]
            if not g:
                continue
            L.append(f"\n[{group}]  target: {TARGETS[group]}")
            for name in dict.fromkeys(r["task"] for r in g):
                for arm in ("old", "new"):
                    rs = [r for r in g if r["task"] == name and r["arm"] == arm]
                    if not rs:
                        continue
                    passed = sum(1 for r in rs if all(r["checks"].values()))
                    flag = "ok  " if passed == len(rs) else ("MISS" if not TASKS[name].get("report_only") else "info")
                    L.append(f"  {flag} {arm} {name}: {passed}/{len(rs)}  (median {median([r.get('seconds') for r in rs])}s, {median([r.get('steps') for r in rs])} steps, "
                             f"{median([r.get('calls') for r in rs])} model calls)")
                    for c in sorted({c for r in rs for c in r["checks"]}):
                        n = sum(1 for r in rs if r["checks"].get(c))
                        if n != len(rs):
                            L.append(f"        - {c}: {n}/{len(rs)}")
                    for r in rs:
                        if r.get("error"):
                            L.append(f"        error: {r['error']}")
            for arm in ("old", "new"):
                rs = [r for r in g if r["arm"] == arm]
                if rs:
                    L.append(f"  {group} {arm} overall: {sum(1 for r in rs if all(r['checks'].values()))}/{len(rs)} reps pass; per rep median "
                             f"{median([r.get('seconds') for r in rs])}s, {median([r.get('calls') for r in rs])} model calls, {median([r.get('tokens_in') for r in rs])} tokens in, "
                             f"{median([r.get('steps') for r in rs])} steps; questions asked of the user {sum(r['questions'] for r in rs)}; "
                             f"routed to daily_assistant {sum(1 for r in rs if r.get('routed') == 'daily_assistant')}/{len(rs)}; gate bounces {sum(r.get('bounces') or 0 for r in rs)}")
        L.append("")
    if "dashboard" in res:
        d = res["dashboard"]
        L.append(f"dashboard (the whole path, {d['seconds']}s)")
        for k, v in d["checks"].items():
            L.append(f"  {'ok  ' if v is True else 'MISS'} {k}" + ("" if v is True else f": {v}"))
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="sources,turns,dashboard")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--arm", default="auto", choices=["auto", "new", "old", "both"])
    ap.add_argument("--group", help="only these task groups (comma separated)")
    ap.add_argument("--task", help="only these tasks (comma separated)")
    ap.add_argument("--pin", action="store_true", help="pin daily_assistant (skip routing); by default Jev routes, as for a user")
    ap.add_argument("--model", help="pin every persona to one model, e.g. lmstudio:qwen3.8-27b@q6_k (recommended)")
    ap.add_argument("--dry", action="store_true", help="plumbing check: fake Jev, scripted worker, no network for the sources stage")
    ap.add_argument("--report", action="store_true", help="print the last run's summary (the committed one, or --out's)")
    _out.add_args(ap)
    a = ap.parse_args()
    dest = _out.Out.of(a)
    if a.report:
        last = dest.read(SUMMARY)
        print(last.read_text(encoding="utf-8") if last.exists() else "no runs yet")
        return
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stages = a.stage.split(",")
    res: dict = {"stamp": stamp, "model": a.model, "reps": a.reps, "dry": a.dry}
    if "sources" in stages and not a.dry:
        print("sources ...", flush=True)
        res["sources"] = run_sources()
    if "turns" in stages:
        print("turns ...", flush=True)
        res["turns"] = run_turns(a, stamp)
    if "dashboard" in stages:
        print("dashboard ...", flush=True)
        res["dashboard"] = run_dashboard(a, stamp)
    raw, summary = dest.write(RAW), dest.write(SUMMARY)
    raw.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    text = summarize(res)
    summary.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"raw results: {dest.where(raw)}   summary: {dest.where(summary)}")


if __name__ == "__main__":
    main()
