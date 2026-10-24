"""The fixes for the sessions Jed flagged on 2026-10-21/22, live: old against new on the real models.

The old arm is this checkout with the fix switched off (limits.follow_always: 0, limits.narrate: 0, the old option
parsing), so the two arms differ only in the fix. Stages:

  options   #15: a tool call the model wrote as text (<tool_call><function=ask_user>...) gave ask_user its options as one
            string, and the question box showed a button per character. No model: every such call in your own sessions
            (read-only), parsed the old and the new way. Plus, in the live stages below, every question a worker asked
            is checked for options that are single characters.
  decide    #17, the decision: real consecutive turns from the harness's own sessions, hand-labelled follow-up or new
            topic (flags_12_15_17_pairs.json, redacted). Old: Jev's routing call decides whether the last turns go along
            (on light Jev, Laya decides); new: every turn after the first carries them. Counted: follow-ups that lose
            their context (the bad error) and new topics that get it anyway (what the new rule costs).
  follow    #17, end to end: two-turn sessions on the full harness (routing on, nothing pinned). Fires: the second
            message only makes sense with the first ("go ahead and continue", "did you find a company associated with
            that gmail?"). Lookalikes: the second message is a new topic, and carrying the first must not derail it.
  narrate   #12, end to end: tasks where the worker must tell the user what to do (where to click, what to fill in)
            before it asks them something. Counted: the user saw those steps (in the chat or the question) before they
            had to answer. Regressions: tasks with nothing for the user to do, where narration is noise; counted: how
            many messages it adds, and whether it repeats the answer.

    python evals/bench/flags_12_15_17.py                       # every stage, both arms, 3 repeats (about 3 hours)
    python evals/bench/flags_12_15_17.py --stage decide,options  # minutes: no worker model
    python evals/bench/flags_12_15_17.py --stage follow --reps 2 -k coins,company
    python evals/bench/flags_12_15_17.py --holdout             # only the held-out cases and pairs
    python evals/bench/flags_12_15_17.py --offline [raw.json]  # the summary again from saved results

Designed for: worker = Qwen3.8-Flash-Next (IQ3_S) on Strata (config.local.yaml, native tools, one slot), Jev = light
mode (Laya on the CPU, jev.mode: light, checks: worker). It runs on whatever your config says; the decide stage needs a
Jev with a routing call. Needs nothing secret and no connected accounts (no case touches Nango, a browser or the desktop).
Every question a worker asks gets the case's scripted answer, except a request to change this computer (SYSTEM_CHANGE),
which is refused. The first live run (2026-10-24) had no such guard: the docker case moved old folders in
%LOCALAPPDATA%\\Docker aside and restarted Docker, and the router case tried (and failed, without admin) to add a port
proxy and a firewall rule. Its committed results are from that run.

Results: runs/flags_12_15_17/<stamp>/ (each run's workspace), flags_12_15_17_raw.json and
flags_12_15_17_summary.txt (a live run writes to workspace/evals/<backend>/; --offline re-sweeps the committed ones;
see evals/_out.py).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))
import _out  # noqa: E402  (evals/_out.py: where a run's results go)

RUNS = HERE / "runs" / "flags_12_15_17"
RAW = HERE / "flags_12_15_17_raw.json"
SUMMARY = HERE / "flags_12_15_17_summary.txt"
PAIRS = HERE / "flags_12_15_17_pairs.json"
LIMIT = 10 * 60   # seconds a turn may take before it is stopped (and fails)
# a question asking to change this computer (the bench answers every other question with the case's scripted answer)
SYSTEM_CHANGE = re.compile(r"firewall|portproxy|port proxy|netsh|registry|admin|elevat|UAC|install|uninstall|delete|"
                           r"remove|run as|on this PC|this computer|reset|clean|tidy|move|rename|archive|kill|stop |"
                           r"restart|docker", re.I)
REFUSE = "No - please don't change anything on this computer. Just tell me what you would do."

EMAIL = """This email just came in, what do you make of it? Should I reply?

From: Sam Doe <ehiburn.sam1987@gmail.com>
Subject: INSULATION INSTALLATION INQUIRY

Hello,
I'm reaching out regarding an upcoming insulation installation project in Utah. I'm looking for a reliable and
experienced insulation contractor to assist with the project.
Could you please let me know what insulation services you offer, your availability for new projects, and how you
typically handle estimates and scheduling?
Thank you for your time. I look forward to hearing from you.
Best regards,"""

SAVE_TASK = ("Make me a little save file for my racing game so we can mod it: save.json with coins 120, gems 3 and three "
             "vehicles (jeep, bus, moon_lander) each with an engine power number, plus a script edit_save.py that can "
             "set any of those values. Then double the jeep's engine power.")


@dataclass
class Case:
    name: str
    stage: str                   # follow | narrate
    kind: str                    # fires | lookalike | regression
    turns: list[str]             # the user's messages; the last one is graded
    check: Callable[[dict], tuple[bool, str]]
    persona: str | None = None   # pinned persona (narrate); None = routing decides (follow)
    holdout: bool = False
    answer: str = "Done."        # what the user answers when the worker asks
    seen: str = ""               # narrate fires: the steps the user needs, as a regex: shown before they answered?


def _files(rec: dict, name: str) -> list[str]:
    return [t for p, t in rec["files"].items() if Path(p).name == name]


def _says(*pats, bad: tuple = ()) -> Callable[[dict], tuple[bool, str]]:
    def check(rec):
        a = rec["answer"]
        miss = [p for p in pats if not re.search(p, a, re.I)]
        if miss:
            return False, f"the answer lacks {miss[0]!r}"
        hit = [p for p in bad if re.search(p, a, re.I)]
        if hit:
            return False, f"the answer brings up the earlier topic ({hit[0]!r})"
        return True, "answers it"
    return check


def _values(obj, key: str):
    """Every number under `key` anywhere in a parsed JSON value (the model nests it as it likes)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and isinstance(v, (int, float)) and not isinstance(v, bool):
                yield v
            yield from _values(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from _values(v, key)


def _saves(files: dict[str, str]):
    for p, text in files.items():
        if Path(p).name == "save.json":
            try:
                yield json.loads(text)
            except ValueError:
                pass


def coins_check(rec):
    best = max((v for s in _saves(rec["files"]) for v in _values(s, "coins")), default=None)
    if best is not None and best >= 10 ** 6:
        return True, f"save.json coins = {best}"
    return False, f"no save.json with a huge coin count (most: {best})"


def _moon_power(files: dict[str, str]) -> float:
    """The moon lander's engine number in any save.json (the vehicle entry's largest number besides its name)."""
    out = 0.0
    for s in _saves(files):
        for v in s.get("vehicles", []) if isinstance(s.get("vehicles"), list) else \
                [{"id": k, **(x if isinstance(x, dict) else {"power": x})} for k, x in (s.get("vehicles") or {}).items()]:
            if isinstance(v, dict) and "moon" in json.dumps(v).lower():
                out = max([out] + [x for k, x in v.items() if isinstance(x, (int, float)) and not isinstance(x, bool)])
    return out


def moon_check(rec):
    before, after = _moon_power(rec["files_before"]), _moon_power(rec["files"])
    if after > before > 0:
        return True, f"moon lander engine {before:g} -> {after:g}"
    return False, f"moon lander engine not raised ({before:g} -> {after:g})"


def company_check(rec):
    a = rec["answer"]
    if re.search(r"which (gmail|email|address)|what (gmail|email) (are you|do you)|can you (share|paste|forward)", a, re.I):
        return False, "asked which email (lost the conversation)"
    if not re.search(r"ehiburn|sam doe|gmail\.com|personal (gmail|address)|free (gmail|email)", a, re.I):
        return False, "doesn't refer to the sender"
    return True, "talks about the sender"


def firmer_check(rec):
    a = rec["answer"]
    ok = re.search(r"\bJed\b", a) and re.search(r"rent", a, re.I) and re.search(r"\b(3|three)\b.{0,12}days?|late", a, re.I)
    return (True, "the letter, firmer, signed") if ok else (False, "not the landlord letter signed Jed")


def tested_check(rec):
    a = rec["answer"]
    if re.search(r"which (function|file|code)|what (function|file) (are you|do you)", a, re.I):
        return False, "asked which code (lost the conversation)"
    return (True, "answers about its test") if re.search(r"test|assert|ran|run", a, re.I) else \
        (False, "doesn't say how it was checked")


def file_check(name: str, pat: str):
    def check(rec):
        found = _files(rec, name)
        if not any(re.search(pat, t, re.I) for t in found):
            return False, f"no {name} with {pat!r}" if not found else f"{name} lacks {pat!r}"
        return True, f"{name} written"
    return check


def regress_check(*pats):
    def check(rec):
        a = rec["answer"].replace(",", "")
        miss = [p for p in pats if not re.search(p, a)]
        return (False, f"the answer lacks {miss[0]!r}") if miss else (True, "right answer")
    return check


SUM_SQ_20_PRIMES = sum(p * p for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71))

CASES = [
    # ---- follow (#17): the second message continues the first; routing and its follow-up call are live
    Case("coins", "follow", "fires", [SAVE_TASK, "could you also give me unlimited coins? but yeah go ahead and continue"],
         coins_check),
    Case("company", "follow", "fires", [EMAIL, "did you find a company associated with that gmail? either way just ignore it."],
         company_check),
    Case("firmer", "follow", "fires", ["Draft a short polite note to my landlord saying the rent will be 3 days late.",
                                      "make it a bit firmer, and sign it Jed"], firmer_check),
    Case("tested", "follow", "fires", ["Write a small python function that reverses the order of the words in a "
                                       "sentence, save it as rev.py, and check it works.",
                                       "did you actually run it to check?"], tested_check, holdout=True),
    Case("moon", "follow", "fires", [SAVE_TASK, "can you also make the moon lander's engine way stronger too?"],
         moon_check, holdout=True),
    # ---- follow lookalikes: a new topic after the first turn; the carried turn must not take over
    Case("percent", "follow", "lookalike", [SAVE_TASK, "what's 17% of 2,340?"],
         _says(r"397\.8", bad=(r"save\.json", r"coins", r"jeep"))),
    Case("utc", "follow", "lookalike", [EMAIL, "write me a python one-liner that prints the current UTC time"],
         _says(r"datetime|time\.", r"utc", bad=(r"insulation", r"inquir", r"ehiburn"))),
    Case("center", "follow", "lookalike", ["Plan a 3-day upper/lower workout split for me.", "how do I center a div in CSS?"],
         _says(r"flex|grid|margin", bad=(r"squat", r"workout", r"bench press")), holdout=True),
    # ---- narrate (#12): the user has to act before the worker can go on
    Case("oauth", "narrate", "fires",
         ["I'm adding 'Sign in with GitHub' to a little web app of my own (nothing for you to connect). I'll do the "
          "clicking on GitHub: tell me exactly where to go and what to fill in (my callback is "
          "http://localhost:5000/auth/callback), wait until I've made the app, then ask me for its Client ID and save "
          "it to client_id.txt."],
         file_check("client_id.txt", r"Ov23liQ7zK2mN4pX9aB1"), persona="daily_assistant",
         answer="Done - the Client ID is Ov23liQ7zK2mN4pX9aB1", seen=r"github\.com/settings|developer settings|oauth apps"),
    Case("drive_api", "narrate", "fires",
         ["A script of mine needs the Google Drive API turned on in my own Google Cloud project (nothing for you to "
          "connect). I'll do the clicking: tell me exactly where to click, wait until I say it's done, then write a "
          "short checklist of what we did to drive_setup.md."],
         file_check("drive_setup.md", r"drive"), persona="daily_assistant", answer="Done, it says Enabled now.",
         seen=r"console\.cloud\.google\.com|APIs? & Services|API Library"),
    Case("router", "narrate", "fires",
         ["Help me forward port 8443 on my router to this PC so I can reach the dashboard from outside. You can't get "
          "to the router yourself: give me the steps, ask me to confirm when I've done it, then save the settings we "
          "used to port_forward.txt."],
         file_check("port_forward.txt", r"8443"), persona="daily_assistant", answer="Done, it's saved on the router.",
         seen=r"192\.168|router'?s? (admin|settings|login|page)|port forwarding", holdout=True),
    Case("docker", "narrate", "fires",
         ["Docker Desktop just showed me an error about a stale lock file under AppData\\Local\\Docker\\run, with the "
          "buttons Quit, Reset to factory defaults and Restart. Before you touch anything, tell me which button to "
          "click and which not to, and ask me whether you should go ahead and clean it up. Then write what you would "
          "do to docker_fix.txt (don't change anything on this computer)."],
         file_check("docker_fix.txt", r"."), persona="daily_assistant", answer="I clicked Quit. Don't clean anything up, just write down what you'd do.",
         seen=r"factory", holdout=True),
    # ---- narrate regressions: nothing for the user to do; narration is only noise here
    Case("fib_sha", "narrate", "regression",
         ["Using Python, work out the 40th Fibonacci number (F(1)=1, F(2)=1) and the SHA-256 hex digest of the text "
          "'strata on the 3090'. Give me both results."],
         regress_check(r"102334155", r"58cfdd685d3bf2dce0a0b426a79f6e8bf6ba59071f4c3a42753b7f74028c53b4"),
         persona="high_end_dev"),
    Case("primes", "narrate", "regression",
         ["Make a CSV of the first 20 primes and their squares, save it as primes.csv, and tell me the sum of the squares."],
         regress_check(str(SUM_SQ_20_PRIMES)), persona="high_end_dev", holdout=True),
]


# ---------------------------------------------------------------------------------------------------------------- arms
def old_option_list(options):
    return list(options or [])   # before: a string became a button per character


@contextlib.contextmanager
def arm(which: str):
    """The old arm: the option parsing as it was (limits are set on each harness, see make_harness)."""
    from unittest import mock

    from jevh import orchestrator, worker
    with contextlib.ExitStack() as st:
        if which == "old":
            st.enter_context(mock.patch.object(worker, "xml_value", lambda key, value: value))
            st.enter_context(mock.patch.object(orchestrator, "_option_list", old_option_list))
        yield


# ------------------------------------------------------------------------------------------------------------ scaffolding
def make_harness(where: Path, events: list, which: str, stage: str, answer: str):
    from jevh import orchestrator
    from jevh.config import load_config
    from jevh.orchestrator import Harness
    cfg = load_config()
    cfg.workspace = where / "ws"
    cfg.personas_dir = where / "personas"
    cfg.personas_dirs = [cfg.personas_dir]
    if not cfg.personas_dir.exists():
        shutil.copytree(ROOT / "personas", cfg.personas_dir, ignore=shutil.ignore_patterns("learned"))
    cfg.raw = {**cfg.raw, "learning": False, "failures": {**(cfg.raw.get("failures") or {}), "audit": 0}}
    # each stage's old arm switches off its own fix only: the arms differ in that one thing
    cfg.limits = {**cfg.limits, "follow_always": int(which == "new" or stage != "follow"),
                  "narrate": int(which == "new" or stage != "narrate")}
    t0 = time.time()
    h = Harness(cfg, on_event=lambda k, d: events.append((time.time() - t0, k, d)))
    h.learning, h.tier_mode = False, "strong"

    def ask(task, persona, question, options=None, secret=False):
        # what the question box would show (the arm's own option parsing: arm() patches it), then the scripted answer -
        # except to a request to change this computer, which is always refused: the bench runs on a real PC (first live
        # run, 2026-10-24: the router case answered "Done" to "may I add a port proxy and a firewall rule?", and the
        # worker tried to elevate; Windows' prompt went unanswered, so nothing changed)
        shown = orchestrator._option_list(options)
        events.append((time.time() - t0, "asked", {"question": str(question), "options": [str(o) for o in shown][:8]}))
        events.append((time.time() - t0, "answered", {}))
        return REFUSE if SYSTEM_CHANGE.search(str(question)) else answer
    h._ask_user = ask
    return h


def timed_turn(h, prompt: str, persona: str | None):
    watchdog = threading.Timer(LIMIT, h.cancel)
    watchdog.daemon = True
    watchdog.start()
    try:
        return h.turn(prompt, force=persona), None
    except Exception as e:  # noqa: BLE001 - a crash is a result
        return None, f"{type(e).__name__}: {e}"
    finally:
        watchdog.cancel()


def run_case(case: Case, which: str, where: Path) -> dict:
    events: list = []
    h = make_harness(where, events, which, case.stage, case.answer)
    for before in case.turns[:-1]:   # earlier turns of the session; only the last one is graded
        timed_turn(h, before, case.persona)
    files_before = session_files(h)
    events.clear()
    started = time.time()
    r, err = timed_turn(h, case.turns[-1], case.persona)
    route = next((d["decision"] for _, k, d in events if k == "route"), None)
    told = [(t, d["text"], bool(d.get("narration"))) for t, k, d in events if k == "say"]
    asked = [(t, d) for t, k, d in events if k == "asked"]
    files = session_files(h)
    usage = (r.usage or {}) if r else {}
    llm = (usage.get("by_kind") or {}).get("llm") or {}
    return {"case": case.name, "stage": case.stage, "kind": case.kind, "arm": which, "holdout": case.holdout,
            "answer": r.answer if r else "", "error": err, "seconds": round(time.time() - started, 1),
            "persona": r.persona if r else None,
            "follows": bool(route.follows) if route else None,
            "continues": getattr(route, "continues", None), "replies": getattr(route, "replies", None),
            "told": [{"t": round(t, 1), "text": s[:2000], "narration": n} for t, s, n in told],
            "asked": [{"t": round(t, 1), **d} for t, d in asked],
            "steps": sum(1 for _, k, _ in events if k == "code"),
            "input_tokens": llm.get("input", 0), "output_tokens": llm.get("output", 0), "model_calls": llm.get("calls", 0),
            "files": files, "files_before": files_before}


def session_files(h) -> dict[str, str]:
    """The session's small text files (what the worker wrote), not its thinking logs or checkpoints."""
    out = {}
    for p in h.session_dir.rglob("*") if h.session_dir.is_dir() else []:
        if p.is_file() and p.suffix in (".json", ".txt", ".md", ".csv", ".py") and p.name != "checkpoint.json" \
                and "thinking" not in p.parts and p.stat().st_size < 200_000:
            out[str(p.relative_to(h.session_dir))] = p.read_text(encoding="utf-8", errors="replace")
    return out


def seen_before_answer(rec: dict, pattern: str) -> bool:
    """The steps reached the user before they had to answer: in a chat message sent before the first question, or in
    the question itself."""
    first_q = rec["asked"][0]["t"] if rec["asked"] else None
    shown = [m["text"] for m in rec["told"] if first_q is None or m["t"] <= first_q]
    shown += [rec["asked"][0]["question"]] if rec["asked"] else []
    return any(re.search(pattern, s, re.I) for s in shown)


def char_buttons(rec: dict) -> bool:
    return any(q["options"] and all(len(o) == 1 for o in q["options"]) for q in rec.get("asked", []))


def grade(rec: dict) -> dict:
    case = next(c for c in CASES if c.name == rec["case"])
    if rec["error"] and not rec["answer"]:
        rec["passed"], rec["why"] = False, rec["error"]
        return rec
    ok, why = case.check(rec)
    if ok and case.seen:
        if not rec["asked"]:
            ok, why = False, "never asked the user (nothing to show before an answer)"
        elif not seen_before_answer(rec, case.seen):
            ok, why = False, f"done, but the user never saw the steps ({case.seen!r}) before answering"
        else:
            why += "; the user saw the steps first"
    if case.stage == "narrate" and case.kind == "regression" and ok:
        dup = [m for m in rec["told"] if len(m["text"]) > 80 and m["text"][:80] in rec["answer"]]
        if dup:
            ok, why = False, "the chat got the answer twice"
    if char_buttons(rec):
        ok, why = False, "a question's buttons were single characters"
    rec["passed"], rec["why"] = bool(ok), why
    return rec


# ------------------------------------------------------------------------------------------------- the decision stage
def run_decide(holdout_only: bool, reps: int) -> list[dict]:
    """Jev's routing call (follows?) on each real pair; the new rule carries every one. Laya answers in well under a
    second, so this needs no worker model."""
    from jevh.config import load_config
    from jevh.memory import Turn
    from jevh.orchestrator import Harness
    pairs = json.loads(PAIRS.read_text(encoding="utf-8"))["pairs"]
    cfg = load_config()
    where = RUNS / "decide"
    cfg.workspace = where / "ws"
    cfg.raw = {**cfg.raw, "learning": False}
    h = Harness(cfg)
    out = []
    for rep in range(reps):
        for p in pairs:
            if holdout_only and not p["holdout"]:
                continue
            # exactly what a live turn sends (Memory.recent_text: the last two turns, each Turn.chunk())
            recent = "\n\n".join(Turn(t["persona"], t["user"], t["assistant"], "", 0.0).chunk() for t in p["recent"])
            current = p["recent"][-1]["persona"]
            t0 = time.time()
            try:
                d = h.jev.route_persona(p["new"], h.personas, current, recent)
                follows, cont, rep_, err = bool(d.follows), d.continues, d.replies, None
            except Exception as e:  # noqa: BLE001 - Jev down: the harness falls back to "not a follow-up"
                follows, cont, rep_, err = False, None, None, f"{type(e).__name__}: {e}"
            out.append({"id": p["id"], "rep": rep, "label": p["follows"], "holdout": p["holdout"],
                        "flagged": p["flagged"], "old": follows, "new": True, "continues": cont, "replies": rep_,
                        "error": err, "seconds": round(time.time() - t0, 2), "new_text": p["new"][:120]})
    return out


# -------------------------------------------------------------------------------------------- the option-parsing replay
def run_options(sessions: Path) -> list[dict]:
    """Every tool call a worker wrote as text in your sessions (read-only), parsed the old and the new way."""
    from jevh import worker
    out = []
    for ck in sorted(sessions.glob("*/u*/checkpoint.json")):
        try:
            msgs = json.loads(ck.read_text(encoding="utf-8"))["messages"]
        except (OSError, ValueError, KeyError):
            continue
        for i, m in enumerate(msgs):
            text = m.get("content") if m.get("role") == "assistant" else None
            if not isinstance(text, str) or "<function=" not in text:
                continue
            for name, body in worker.XML_CALL_RE.findall(text):
                if name == "python":
                    continue
                params = dict(worker.XML_PARAM_RE.findall(body))
                new = {k: worker.xml_value(k, v) for k, v in params.items()}
                typed = sorted(k for k in params if new[k] is not params[k] and not isinstance(new[k], str))
                out.append({"where": f"{ck.parent.parent.name}/{ck.parent.name}#{i}", "call": name,
                            "params": sorted(params), "typed": typed,
                            "old_options": params.get("options"), "new_options": new.get("options")})
    return out


# ------------------------------------------------------------------------------------------------------------- summary
def _rate(xs) -> str:
    xs = list(xs)
    return f"{sum(xs)}/{len(xs)}" if xs else "-"


def summarize(records: list[dict], decided: list[dict] | None, options: list[dict] | None) -> str:
    lines = [f"flags #12/#15/#17 fixes, live ({time.strftime('%Y-%m-%d %H:%M')})"]
    if options is not None:
        typed = [o for o in options if o["typed"]]
        opt = [o for o in options if o["old_options"] is not None]
        lines.append(f"\n[options] {len(options)} tool calls written as text in the sessions read; {len(opt)} gave "
                     f"ask_user options. Old: each of those {len(opt)} reached ask_user as one string (a button per "
                     f"character). New: {sum(isinstance(o['new_options'], list) for o in opt)}/{len(opt)} as a list. "
                     f"Calls whose arguments now change type: {len(typed)} "
                     f"({', '.join(sorted({k for o in typed for k in o['typed']})) or '-'}); every other call parses "
                     "exactly as before.")
        for o in opt:
            lines.append(f"    {o['where']}: {o['new_options']!r}")
    if decided:
        lines.append("\n[decide] Jev's follow-up call (old) vs carry every turn (new), on real hand-labelled pairs:")
        for split in (False, True):
            d = [x for x in decided if x["holdout"] == split]
            if not d:
                continue
            fol, newt = [x for x in d if x["label"]], [x for x in d if not x["label"]]
            name = "held-out" if split else "dev     "
            lines.append(f"  {name}: follow-ups carried  old {_rate(x['old'] for x in fol)}  new {_rate(x['new'] for x in fol)}"
                         f"   |  new topics carried anyway  old {_rate(x['old'] for x in newt)}  new "
                         f"{_rate(x['new'] for x in newt)}")
        flagged = [x for x in decided if x["flagged"]]
        if flagged:
            lines.append(f"  the flagged sessions' own follow-ups: old {_rate(x['old'] for x in flagged)} carried")
        missed = sorted({(x["id"], x["new_text"]) for x in decided if x["label"] and not x["old"]})
        if missed:
            lines.append("  follow-ups the old rule dropped (any repeat):")
            lines += [f"    #{i}: {t!r}" for i, t in missed]
        errs = [x for x in decided if x["error"]]
        if errs:
            lines.append(f"  Jev errors: {len(errs)} (e.g. {errs[0]['error'][:120]})")
        lines.append(f"  median Jev call {statistics.median(x['seconds'] for x in decided):.2f}s")
    for stage in ("follow", "narrate"):
        rs = [r for r in records if r["stage"] == stage]
        if not rs:
            continue
        lines.append(f"\n[{stage}]")
        for kind in ("fires", "lookalike", "regression"):
            for split in (False, True):
                row = []
                for which in ("old", "new"):
                    s = [r for r in rs if r["kind"] == kind and r["arm"] == which and r["holdout"] == split]
                    if not s:
                        continue
                    extra = ""
                    if stage == "follow":
                        extra = f", carried {sum(bool(r['follows']) for r in s)}/{len(s)}"
                    if stage == "narrate":
                        extra = (f", {statistics.mean(sum(m['narration'] for m in r['told']) for r in s):.1f} "
                                 "narrated msgs/turn")
                    row.append(f"{which} {sum(r['passed'] for r in s)}/{len(s)} (median {statistics.median(r['seconds'] for r in s):.0f}s, "
                               f"{statistics.mean(r['input_tokens'] for r in s) / 1000:.1f}k in, "
                               f"{statistics.mean(r['model_calls'] for r in s):.1f} calls{extra})")
                if row:
                    lines.append(f"  {kind:10} {'held-out' if split else 'dev     '}: " + "  |  ".join(row))
        for name in sorted({r["case"] for r in rs}):
            row = [f"{w}: {sum(r['passed'] for r in rs if r['case'] == name and r['arm'] == w)}/"
                   f"{sum(1 for r in rs if r['case'] == name and r['arm'] == w)}" for w in ("old", "new")]
            lines.append(f"    {name:10} " + "  ".join(row))
    bad = [r for r in records if not r["passed"]]
    if bad:
        lines.append("\nfailures:")
        for r in bad:
            lines.append(f"  {r['case']} {r['arm']} rep {r.get('rep', 0) + 1}: {r['why']}; "
                         f"{(r['error'] or r['answer'])[:200]!r}")
    return "\n".join(lines)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # (model answers carry arrows and dashes)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", default="options,decide,follow,narrate")
    ap.add_argument("--arm", default="both", choices=["old", "new", "both"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--holdout", action="store_true", help="only the held-out cases and pairs")
    ap.add_argument("-k", default="", help="only cases whose name contains one of these (comma-separated)")
    ap.add_argument("--sessions", default=str(ROOT / "workspace" / "sessions"),
                    help="the session folders the options stage reads (read-only)")
    ap.add_argument("--offline", nargs="?", const="", metavar="RAW.json",
                    help="summarize saved results again (default: the committed raw file, or --out's)")
    _out.add_args(ap)
    a = ap.parse_args()
    dest = _out.Out.of(a)
    if a.offline is not None:
        data = json.loads((Path(a.offline) if a.offline else dest.read(RAW)).read_text(encoding="utf-8"))
        print(summarize([grade(r) for r in data["records"]], data.get("decide"), data.get("options")))
        return 0
    stages = a.stage.split(",")
    options = run_options(Path(a.sessions)) if "options" in stages and Path(a.sessions).is_dir() else None
    decided = run_decide(a.holdout, a.reps) if "decide" in stages else None
    cases = [c for c in CASES if c.stage in stages and (c.holdout or not a.holdout)
             and (not a.k or any(x in c.name for x in a.k.split(",")))]
    arms = ["old", "new"] if a.arm == "both" else [a.arm]
    out_dir = RUNS / time.strftime("%Y%m%d-%H%M%S")
    records: list[dict] = []

    def save():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "raw.json").write_text(json.dumps({"records": records, "decide": decided, "options": options},
                                                     indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    for rep in range(a.reps if cases else 0):
        for case in cases:
            for which in arms:
                where = out_dir / f"{case.name}-{which}-{rep}"
                print(f"[{time.strftime('%H:%M:%S')}] {case.name} {which} rep {rep + 1}...", flush=True)
                with arm(which):
                    try:
                        rec = run_case(case, which, where)
                    except Exception as e:  # noqa: BLE001
                        rec = {"case": case.name, "stage": case.stage, "kind": case.kind, "arm": which,
                               "holdout": case.holdout, "answer": "", "error": f"bench: {type(e).__name__}: {e}",
                               "seconds": 0, "persona": None, "follows": None, "told": [], "asked": [], "steps": 0,
                               "input_tokens": 0, "output_tokens": 0, "model_calls": 0, "files": {},
                               "files_before": {}}
                rec["rep"] = rep
                records.append(grade(rec))
                print(f"    {'PASS' if rec['passed'] else 'FAIL'} ({rec['seconds']}s): {rec['why']}", flush=True)
                save()
    if options is not None or decided is not None:
        save()
    text = summarize(records, decided, options)
    print(text)
    dest.write(RAW).write_text(json.dumps({"records": records, "decide": decided, "options": options}, indent=1,
                                          ensure_ascii=False, default=str), encoding="utf-8")
    dest.write(SUMMARY).write_text(text + "\n", encoding="utf-8")
    print(f"raw: {dest.where(dest.write(RAW))}   summary: {dest.where(dest.write(SUMMARY))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
