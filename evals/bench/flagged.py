"""The fixes for the sessions Jed flagged on 2026-10-09..12, live: full harness turns on the real models, old against new.

Each stage is one fix. The old arm is this checkout with that fix switched off (the old code put back for the run), so the
two arms differ only in the fix:

  fence    a plain ``` block in a reply was run as Python (flag 8: a chess move list; its SyntaxError came back as the next
           user turn). Cases that should keep the block as text, and code tasks that must still run (regression).
  gate     the final() gate sent back right answers: "$0.49" when the source prints "$.49", totals the answer works out in
           the open, "a 30-second fix", "~20 languages", "Reply go and I'll ..." (flag 11, flag 9). Live: a real rebate
           table, counted bounces and right answers. Plus `--stage replay`: the number check, old vs new, on every answer
           the gate sent back in your own failure log (read-only), with the flags the new check drops listed for review.
  resume   a turn the harness never finished (a crash, a restart) was not remembered and "continue" started over (flag
           10). The model call after step 2 kills the turn as a dying process would; a new harness on the same session
           then gets "continue" (or "what did you get done?").
  say      a question sent while a task runs could only be answered at the end (flag 9). The question is sent after the
           first step; counted: an answer reached the chat before the task ended (say(), or the quick reply), and when.
           A run with no question counts say() chatter (regression).
  carry    a follow-up turn started with a fresh REPL in a new folder (flag 8: "the helper module was lost - let me
           rebuild it" before every chess move). A tic-tac-toe game over two or three turns: does the graded turn go on
           from the last turn's `board` or rebuild it?
  where    the worker was not told the user was on another computer when it had no devices tool, or wrote from a phone
           (flag 11). A stub device hub stands in for the dashboard's (no device is called); the models are live.

    python evals/bench/flagged.py                      # every stage, both arms, 3 repeats (about 2 hours)
    python evals/bench/flagged.py --stage fence,say --reps 2
    python evals/bench/flagged.py --holdout             # only the held-out cases
    python evals/bench/flagged.py --stage replay        # no models: the number check on your failure log
    python evals/bench/flagged.py --offline [raw.json]  # the summary again from saved results

Designed for: worker = Qwen3.8-27B on llama-server (config.local.yaml, native tools), Jev = the semif adapter. Needs
nothing secret; the browser cases need Playwright's Chromium. Results: runs/flagged/<stamp>/ (each run's workspace),
flagged_raw.json and flagged_last_summary.txt beside this file.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
import sqlite3
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
RUNS = HERE / "runs" / "flagged"
RAW = HERE / "flagged_raw.json"
SUMMARY = HERE / "flagged_last_summary.txt"
LIMIT = 8 * 60   # seconds a turn may take before it is stopped (and fails)

# The envelope table of Black Hills Energy's 2026 Colorado gas rebate application (public; the PDF of flagged session
# 20261012-195309-34cd), as pypdf extracts it: the rates print without a leading zero.
REBATE_TABLE = """ENVELOPE MEASURE RETROFIT*
Equipment Type Efficiency Requirement Rebate Amount
Insulation (Ceiling/Attic) >= R-49 $.49 per sq ft up to $890
>= R-60 $.60 per sq ft up to $1,090
Insulation (Basement Wall) >= R-13 $.48 per sq ft up to $675
>= R-19 $.53 per sq ft up to $740
Insulation (Side-Slab) >= R-10 with R-5 Thermal Break $.25 per sq ft up to $425
Insulation (Crawlspace) >= R-13 (must be in unconditioned space) $.25 per sq ft up to $500
Insulation (Rim and Band Joist) >= R-19 $.25 per sq ft up to $200
Insulation (Wall) >= R-13 or Max Fill $.25 per sq ft up to $680
Air Sealing Whole home air sealing 70% of cost up to $690
Duct Sealing & Insulation Combined Duct sealing and >= R-8 (must be in unconditioned space) 70% of cost up to $340
"""


@dataclass
class Case:
    name: str
    stage: str
    kind: str                    # fires | regression | lookalike
    prompt: str
    check: Callable[[dict], tuple[bool, str]]
    persona: str = "daily_assistant"
    holdout: bool = False
    extra: dict = field(default_factory=dict)


def _has(*words):
    return lambda a: all(re.search(w, a, re.I) for w in words)


def ran(rec: dict, pattern: str) -> bool:
    """Code matching `pattern` ran - not counting a final() call, whose answer text holds the block (the `final` tool
    becomes print(final(...)))."""
    return any(re.search(pattern, c) for c in rec["code"] if "final(" not in c)


def syntax_errors(rec: dict) -> int:
    return sum("SyntaxError" in o for o in rec["outputs"])


# ---------------------------------------------------------------------------------------------------------------- cases
def fence_check(text_re: str, block_re: str):
    def check(rec):
        a = rec["answer"]
        if not re.search(text_re, a):
            return False, "the answer lacks the content"
        if ran(rec, block_re) or syntax_errors(rec):
            return False, "the reply's block was run as code"
        return True, "kept as text"
    return check


def regress_check(value: str):
    def check(rec):
        if not re.search(rf"(?<![\d.]){value}(?!\d)(?!\.\d)", rec["answer"].replace(",", "")):
            return False, f"no {value} in the answer"
        return (True, "computed in code") if rec["code"] else (False, "no code ran")
    return check


CASES = [
    Case("fence_moves", "fence", "fires", "Give me the first 5 moves of the Ruy Lopez as a numbered move list inside a plain "
         "``` code block with no language tag (not python). Don't run any code for this.",
         fence_check(r"Bb5", r"1\.\s*e4"), persona="high_end_dev"),
    Case("fence_table", "fence", "fires", "Make me a small plain-text table (aligned with spaces) of Mercury, Venus, Earth and "
         "Mars with how many moons each has, inside a plain ``` block with no language tag. Answer from what you know, no code.",
         fence_check(r"Mars", r"Mercury\s+\d"), persona="high_end_dev"),
    Case("fence_cmd", "fence", "fires", "What's the PowerShell one-liner to list the 5 biggest files in a folder? Put just the "
         "command in a plain ``` block with no language tag; don't run it.",
         fence_check(r"Sort-Object", r"Sort-Object"), persona="high_end_dev", holdout=True),
    Case("fence_haiku", "fence", "fires", "Write me a haiku about rain inside a plain ``` code block (no language tag).",
         fence_check(r"```[^\n`]*\n(?:[^\n`]*\w[^\n`]*\n){3}```", r"(?im)^\s*\w+[^\n=()]*,?\s*$"), persona="high_end_dev",
         holdout=True),
    Case("regress_fib", "fence", "regression", "Use Python to compute the 30th Fibonacci number (F1 = F2 = 1) and tell me "
         "the value.", regress_check("832040"), persona="high_end_dev"),
    Case("regress_count", "fence", "regression", "With Python, count how many times the lowercase letter s appears in "
         "'Mississippi state senators' and tell me the count.", regress_check("7"), persona="high_end_dev"),
]


def gate_check(*rates):
    def check(rec):
        a = rec["answer"].replace(",", "")
        if not all(re.search(r"(?<!\d)0?\." + r + r"(?!\d)", a) for r in rates):
            return False, "rates missing or wrong"
        if "2520" not in a:
            return False, "the total (2,520) is missing or wrong"
        return True, "right"
    return check


CASES += [
    Case("gate_rebate", "gate", "fires", "The file rebates.txt in your working folder is the envelope table of Black Hills "
         "Energy's 2026 gas rebate application. Read it and tell me: the attic and basement wall insulation rates (per sq ft) "
         "and caps for each R-value tier, and the most I could get back if I did R-60 attic insulation, R-19 basement wall "
         "insulation and whole-home air sealing, all at the cap.", gate_check("49", "60", "48", "53"), extra={"file": REBATE_TABLE}),
    Case("gate_rebate_short", "gate", "fires", "rebates.txt in your working folder is a utility's rebate table. What does it "
         "pay per sq ft for R-49 attic insulation and for R-13 basement wall insulation, and what's the combined cap if I max "
         "out R-60 attic, R-19 basement wall and air sealing?", gate_check("49", "48"), extra={"file": REBATE_TABLE}, holdout=True),
]


def resume_check(rec):
    files = rec.get("files") or {}
    want = {"a.txt": "alpha", "b.txt": "bravo", "c.txt": "charlie"}
    missing = [n for n, w in want.items() if w not in files.get(n, "")]
    if missing:
        return False, f"not written in the task's folder: {', '.join(missing)}"
    return True, "finished the cut-off task"


def recall_check(rec):
    a = rec["answer"]
    return (True, "remembered") if re.search(r"a\.txt|alpha", a, re.I) and not re.search(
        r"\b(?:no record|don't have any|nothing (?:was|got) done)\b", a, re.I) else (False, "did not know what it had done")


RESUME_TASK = ("Make a folder called rea_test in your working folder with three files: a.txt containing 'alpha', b.txt "
               "containing 'bravo' and c.txt containing 'charlie'. Write each file in its own separate step and print after "
               "each one, then tell me when all three are done.")
CASES += [
    Case("cutoff_continue", "resume", "fires", "continue", resume_check, extra={"first": RESUME_TASK}),
    Case("cutoff_recall", "resume", "fires", "what did you get done before you stopped?", recall_check,
         extra={"first": RESUME_TASK}, holdout=True),
]

PRIMES_TASK = ("In Python, find all the primes below 300000 in one step; then, in a separate step, count how many of them end "
               "in 7; then, in another separate step, find the largest gap between two consecutive ones. Report the three "
               "numbers.")


def say_check(rec):
    if rec.get("heard_at") is None:
        return False, "no answer reached the chat before the task ended"
    return True, f"answered {rec['heard_at']:.0f}s after the question ({rec['heard_by']})"


def say_where_check(rec):
    """As say_check, and the answer must be right: the bench runs on this computer, and the worker is not told where the
    user is, so "on your laptop" is a made-up answer (seen in the first run)."""
    ok, why = say_check(rec)
    if not ok:
        return ok, why
    import platform
    said = " ".join(rec.get("told") or []) or rec.get("aside", "")
    if rec["heard_by"] == "say()" and (re.search(r"\b(?:on|from) your (?:own )?laptop\b", said, re.I)
                                       or not re.search(re.escape(platform.node()) + r"|\bhub\b|this (?:computer|machine|PC)",
                                                        said, re.I)):
        return False, f"answered, but wrongly or vaguely: {said[:150]!r}"
    return True, why


def quiet_check(rec):
    a = rec["answer"].replace(",", "")
    if "25997" not in a:
        return False, "wrong prime count"
    return (len(rec["told"]) <= 1, f"{len(rec['told'])} say() messages")


CASES += [
    Case("say_where", "say", "fires", PRIMES_TASK, say_where_check, persona="high_end_dev",
         extra={"ask": "quick question - are you running this on my laptop or on the hub computer?"}),
    Case("say_why", "say", "fires", PRIMES_TASK, say_check, persona="high_end_dev", holdout=True,
         extra={"ask": "why did you decide to do it that way?"}),
    Case("say_quiet", "say", "regression", PRIMES_TASK, quiet_check, persona="high_end_dev"),
]

def carry_check(rec):
    a = rec["answer"]
    if not re.search(r"\bX\b", a) or not re.search(r"\bO\b", a):
        return False, "no board in the answer"
    rebuilt = any(re.search(r"^\s*board\s*=\s*\[", c, re.M) for c in rec["code"])
    return (not rebuilt, "went on from the last turn's board" if not rebuilt else "rebuilt the board from scratch")


TTT_FIRST = ("Let's play tic-tac-toe, managed in Python: keep the board in a variable called `board` (a list of 9 cells) "
             "and write a helper `show()` that prints it as a 3x3 grid. I'm X, you're O. I take the center. Make your "
             "move and show me the board.")
CASES += [
    Case("carry_ttt", "carry", "fires", "I take the top-left corner. Your move - show me the board.", carry_check,
         persona="high_end_dev", extra={"before": [TTT_FIRST]}),
    Case("carry_ttt2", "carry", "fires", "Top-right for me. Your turn, and show the board.", carry_check,
         persona="high_end_dev", extra={"before": [TTT_FIRST, "I take the top-left corner. Your move - show me the board."]},
         holdout=True),
]

NOT_THERE =r"hub|DESKTOP-AUIDSM2|this (?:computer|machine|PC)|not (?:on )?your (?:screen|phone|laptop)|can'?t see|won'?t see|" \
            r"isn'?t (?:on|in front of)|on my (?:end|side)|here,? not"


def where_check(must: bool):
    def check(rec):
        a = rec["answer"]
        told = bool(re.search(NOT_THERE, a, re.I)) and not re.search(r"which one is", a, re.I)
        if must:
            return (told, "said it is not on the user's screen" if told else "did not say where it opened")
        return (not re.search(r"not (?:on )?your screen|can'?t see it|won'?t see it", a, re.I), "no false 'you can't see it'")
    return check


CASES += [
    Case("where_phone", "where", "fires", "Open https://example.com in the browser for me so I can look at it.",
         where_check(True), extra={"origin": "elsewhere"}),
    Case("where_bilbo", "where", "fires", "I'm on Bilbo right now. If you open a web page for me, will I see it on my screen?",
         where_check(True), extra={"origin": "Bilbo", "no_devices": True}),
    Case("where_bilbo_open", "where", "fires", "Open https://example.com for me so I can read it.",
         where_check(True), extra={"origin": "Bilbo", "no_devices": True}, holdout=True),
    Case("where_hub", "where", "lookalike", "Open https://example.com in the browser for me so I can look at it.",
         where_check(False), extra={"origin": "here"}),
]


# --------------------------------------------------------------------------------------------------- the old arm's code
OLD_STEER_NOTE = ('[Message from the user, sent while you were working: "{text}"\n'
                  "Take it into account from here on: if it changes what they want, follow it instead of the earlier "
                  "request. Your REPL variables and files are as you left them.]")
OLD_NUM = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?:\s*(thousand|million|billion|trillion|[kKmMbB]n?)\b)?")
NEVER = re.compile(r"(?!x)x")


def old_grounding(answer: str, evidence: str):
    """jevh.verify.grounding before 2026-10-12."""
    from jevh import verify
    seen = [v for _, v, _ in verify._nums(re.sub(r"\\[ntr]", " ", evidence))]
    supported, unsupported = set(), set()
    for digits, value, slack in verify._nums(verify.ASPECT.sub(" ", answer)):
        year = re.fullmatch(r"(19|20)\d\d", digits) is not None
        if len(digits.replace(",", "").replace(".", "")) < 2 or year:
            continue
        (supported if any(abs(s - value) <= slack for s in seen) else unsupported).add(digits)
    return supported, unsupported


@contextlib.contextmanager
def arm(stage: str, which: str):
    """The old arm: the stage's fix switched off for the run."""
    from unittest import mock

    from jevh import orchestrator, verify, worker
    with contextlib.ExitStack() as st:
        if which == "old":
            if stage == "fence":
                st.enter_context(mock.patch.object(worker, "code_blocks",
                                                   lambda text, native=False: worker.CODE_RE.findall(text or "")))
            elif stage in ("gate", "replay"):
                st.enter_context(mock.patch.object(verify, "NUM", OLD_NUM))
                st.enter_context(mock.patch.object(verify, "grounding", old_grounding))
                st.enter_context(mock.patch.object(verify, "USER_FIRST", NEVER))
            elif stage == "resume":
                st.enter_context(mock.patch.object(orchestrator.Harness, "_note_cut_off", lambda self: None))
            elif stage == "say":
                st.enter_context(mock.patch.object(worker, "STEER_NOTE", OLD_STEER_NOTE))
                st.enter_context(mock.patch.object(worker, "HOW_YOU_WORK", re.sub(
                    r"- say\(text\):.*?\n(?=- final)", "", worker.HOW_YOU_WORK, flags=re.S)))
            elif stage == "carry":
                st.enter_context(mock.patch.object(orchestrator.Harness, "_previous_folder", lambda self, n, p: None))
            elif stage == "where":
                st.enter_context(mock.patch.object(orchestrator.Harness, "_where_note",
                                                   lambda self, can_reach=True: None))
        yield


# ------------------------------------------------------------------------------------------------------------ scaffolding
class StubHub:
    """Stands in for the dashboard's device hub: which computers are connected, and this one's name. No device is ever
    called (the cases that use it don't give the worker the devices tool, or have no device online)."""

    def __init__(self, online: list[str]):
        self._online, self.here = online, "DESKTOP-AUIDSM2"

    def online(self):
        return list(self._online)

    def list(self):
        return [{"name": n, "online": True, "tools": {}} for n in self._online]

    def call(self, *a, **k):
        raise RuntimeError("the bench's stub hub has no devices to call")


class Killed(BaseException):
    """The harness's process dying mid-turn (nothing in the harness catches a BaseException)."""


def make_harness(where: Path, events: list, session: str | None = None):
    from jevh.config import load_config
    from jevh.orchestrator import Harness
    cfg = load_config()
    cfg.workspace = where / "ws"
    cfg.personas_dir = where / "personas"
    cfg.personas_dirs = [cfg.personas_dir]
    if not cfg.personas_dir.exists():
        shutil.copytree(ROOT / "personas", cfg.personas_dir, ignore=shutil.ignore_patterns("learned"))
    cfg.raw = {**cfg.raw, "learning": False, "failures": {**(cfg.raw.get("failures") or {}), "audit": 0}}
    t0 = time.time()
    h = Harness(cfg, on_event=lambda k, d: events.append((time.time() - t0, k, d)), session=session)
    h.learning, h.tier_mode = False, "strong"
    h._ask_user = lambda *a, **k: "Yes, go ahead - your call."
    return h


def record(case: Case, which: str, h, events: list, answer: str, err: str | None, started: float) -> dict:
    code = [d.get("code", "") for _, k, d in events if k == "code"]
    outputs = [str(d.get("output", "")) for _, k, d in events if k == "output"]
    bounces = [d for _, k, d in events if k == "verify" and d.get("bounced")]
    return {"case": case.name, "stage": case.stage, "kind": case.kind, "arm": which, "holdout": case.holdout,
            "answer": answer, "error": err, "seconds": round(time.time() - started, 1), "code": code, "outputs": outputs,
            "steps": len(code), "bounces": [p for b in bounces for p in b.get("problems", [])],
            "told": [d["text"] for _, k, d in events if k == "say"]}


def timed_turn(h, prompt: str, persona: str):
    watchdog = threading.Timer(LIMIT, h.cancel)
    watchdog.daemon = True
    watchdog.start()
    try:
        r = h.turn(prompt, force=persona)
        return r.answer, None
    except Killed:
        raise
    except Exception as e:  # noqa: BLE001 - a crash is a result
        return "", f"{type(e).__name__}: {e}"
    finally:
        watchdog.cancel()


def run_case(case: Case, which: str, where: Path) -> dict:
    events: list = []
    started = time.time()
    if case.stage == "resume":
        return run_resume(case, which, where)
    h = make_harness(where, events)
    if case.extra.get("file"):
        # the task's working folder is made when it starts: the file goes where the worker's REPL starts
        h.cfg.raw.setdefault("bench_files", {})
        orig_run_task = h.run_task

        def run_task(task, goal, allow_collab, resume=None):
            d = h.session_dir / f"u{h.turn_no}t{task.id}-{task.to}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "rebates.txt").write_text(case.extra["file"], encoding="utf-8")
            return orig_run_task(task, goal, allow_collab, resume)
        h.run_task = run_task
    if case.stage == "where":
        h.devices = StubHub(["Bilbo"] if case.extra["origin"] == "Bilbo" else [])
        h.origin = case.extra["origin"]
        if case.extra.get("no_devices"):
            p = h.personas[case.persona]
            p.tools = [t for t in p.tools if t != "devices"]
    for before in case.extra.get("before") or []:   # earlier turns of the same session; only the last one is graded
        timed_turn(h, before, case.persona)
    if case.extra.get("before"):
        events.clear()
        started = time.time()
    asker = None
    if case.extra.get("ask"):
        def ask():
            for _ in range(int(LIMIT / 0.5)):
                if any(k == "code" for _, k, _ in events):
                    break
                time.sleep(0.5)
            else:
                return
            events.append((time.time() - started, "bench_asked", {}))
            h.interject(case.extra["ask"])
        asker = threading.Thread(target=ask, daemon=True)
        asker.start()
    answer, err = timed_turn(h, case.prompt, case.persona)
    rec = record(case, which, h, events, answer, err, started)
    if case.extra.get("ask"):
        asked = next((t for t, k, _ in events if k == "bench_asked"), None)
        heard = [(t, "say()") for t, k, _ in events if k == "say"] + \
                [(t, "quick reply") for t, k, d in events if k == "aside" or (k == "interject" and d.get("answer"))]
        heard = sorted(x for x in heard if asked is not None and x[0] >= asked)
        rec["asked"] = asked
        rec["heard_at"], rec["heard_by"] = (heard[0][0] - asked, heard[0][1]) if heard else (None, None)
        rec["outcome"] = next((d.get("outcome") for _, k, d in events if k == "interject"), None)
    return rec


def run_resume(case: Case, which: str, where: Path) -> dict:
    """Turn 1 (the task) dies at its third model call; a new harness on the same session then gets the case's prompt."""
    from jevh import llm as llmmod
    events: list = []
    started = time.time()
    h = make_harness(where, events)
    real = llmmod.OpenAICompat.chat
    calls = {"n": 0}

    def dying(self, messages, *a, **k):
        calls["n"] += 1
        if calls["n"] > 2:
            raise Killed()
        return real(self, messages, *a, **k)
    llmmod.OpenAICompat.chat = dying
    try:
        timed_turn(h, case.extra["first"], case.persona)
    except Killed:
        pass
    finally:
        llmmod.OpenAICompat.chat = real
    first_dir = next((d for d in h.session_dir.iterdir() if d.name.startswith("u1t1-")), None)
    events2: list = []
    h2 = make_harness(where, events2, session=h.session)
    answer, err = timed_turn(h2, case.prompt, case.persona)
    rec = record(case, which, h2, events2, answer, err, started)
    rec["resumed"] = any(k == "resuming" for _, k, _ in events2)
    rec["files"] = {p.name: p.read_text(encoding="utf-8", errors="replace")[:200]
                    for p in (first_dir / "rea_test").glob("*.txt")} if first_dir and (first_dir / "rea_test").is_dir() else {}
    rec["first_calls"] = calls["n"]
    return rec


# ---------------------------------------------------------------------------------------------- the failure-log replay
def replay(db: Path) -> list[dict]:
    """Every answer the gate sent back over numbers in `db` (read-only), checked by the old and the new number rule.
    Evidence is the logged request + work log (the live gate also saw the full tool outputs: an approximation)."""
    from jevh import verify
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute("SELECT id, request, answer, work_log, detail, workdir FROM failures WHERE kind IN "
                       "('gate_bounce', 'gate_unresolved') AND detail LIKE '%appear in none%'").fetchall()
    out, done = [], set()
    for fid, request, answer, log, detail, workdir in rows:
        # the draft the gate sent back, and what the worker had seen by then, from the task's conversation
        ck = Path(workdir or "") / "checkpoint.json"
        if not workdir or workdir in done or not ck.is_file():
            continue
        done.add(workdir)
        try:
            msgs = json.loads(ck.read_text(encoding="utf-8"))["messages"]
        except (OSError, ValueError, KeyError):
            continue
        for i, m in enumerate(msgs):
            draft = _final_draft(m)
            after = " ".join(str(x.get("content", "")) for x in msgs[i + 1:i + 3])
            if draft is None or "appear in none of your outputs" not in after:
                continue
            seen = "\n".join(str(x.get("content", "")) for x in msgs[:i] if x.get("role") in ("tool", "user"))
            evidence = (request or "") + "\n" + seen
            res = {}
            for which in ("old", "new"):
                with arm("replay", which):
                    res[which] = sorted(verify.grounding(draft, evidence)[1])
            logged = re.search(r"appear in none of your outputs or the request: (.*?)\. Check", after)
            out.append({"id": fid, "workdir": workdir, "step": i, "logged": logged.group(1) if logged else "",
                        "old": res["old"], "new": res["new"], "dropped": sorted(set(res["old"]) - set(res["new"])),
                        "draft": draft[:3000]})
    return out


def _final_draft(m: dict) -> str | None:
    """The answer an assistant message handed to final(): a native `final` call, or final("...") with a literal in code."""
    for call in m.get("tool_calls") or []:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            continue
        if fn.get("name") == "final" and isinstance(args.get("answer"), str):
            return args["answer"]
        code = args.get("code") if isinstance(args, dict) else None
        if isinstance(code, str) and "final(" in code:
            lit = _literal_final(code)
            if lit:
                return lit
    return None


def _literal_final(code: str) -> str | None:
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    strings = {n.targets[0].id: n.value.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)
               and isinstance(n.value.value, str)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "final" and n.args:
            a = n.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                return a.value
            if isinstance(a, ast.Name) and a.id in strings:
                return strings[a.id]
    return None


# ------------------------------------------------------------------------------------------------------------- summary
def summarize(records: list[dict], replayed: list[dict] | None = None) -> str:
    lines = [f"flagged-session fixes, live ({time.strftime('%Y-%m-%d %H:%M')}): {len(records)} turns"]
    for stage in ("fence", "gate", "resume", "say", "carry", "where"):
        rs = [r for r in records if r["stage"] == stage]
        if not rs:
            continue
        lines.append(f"\n[{stage}]")
        for kind in ("fires", "regression", "lookalike"):
            for which in ("old", "new"):
                k = [r for r in rs if r["kind"] == kind and r["arm"] == which]
                if not k:
                    continue
                for split in (False, True):
                    s = [r for r in k if r["holdout"] == split]
                    if s:
                        med = statistics.median(r["seconds"] for r in s)
                        extra = ""
                        if stage == "gate":
                            extra = f", bounces {sum(len(r['bounces']) for r in s)}"
                        if stage == "say" and kind == "fires":
                            lat = [r["heard_at"] for r in s if r.get("heard_at") is not None]
                            extra = f", median answer {statistics.median(lat):.0f}s after the question" if lat else ""
                        lines.append(f"  {kind:10} {which} {'held-out' if split else 'dev     '}: "
                                     f"{sum(r['passed'] for r in s)}/{len(s)} passed, median {med:.0f}s, "
                                     f"{statistics.mean(r['steps'] for r in s):.1f} code steps{extra}")
        for name in sorted({r["case"] for r in rs}):
            row = [f"{w}: {sum(r['passed'] for r in rs if r['case'] == name and r['arm'] == w)}/"
                   f"{sum(1 for r in rs if r['case'] == name and r['arm'] == w)}" for w in ("old", "new")]
            lines.append(f"    {name:18} " + "  ".join(row))
    bad = [r for r in records if not r["passed"] and r["arm"] == "new"]
    if bad:
        lines.append("\nnew-arm failures:")
        for r in bad:
            lines.append(f"  {r['case']}: {r['why']}; {(r['error'] or r['answer'])[:220]!r}")
    if replayed:
        n_old = sum(bool(x["old"]) for x in replayed)
        n_new = sum(bool(x["new"]) for x in replayed)
        lines.append(f"\n[replay] {len(replayed)} answers the gate sent back over numbers: the old rule flags {n_old}, "
                     f"the new {n_new}. Numbers the new rule no longer flags (check each is a fair pass):")
        for x in replayed:
            if x["dropped"]:
                lines.append(f"  #{x['id']}: {', '.join(x['dropped'])}  (still flagged: {', '.join(x['new']) or '-'})")
    return "\n".join(lines)


def grade(rec: dict) -> dict:
    case = next(c for c in CASES if c.name == rec["case"])
    if rec["error"] and not rec["answer"]:
        rec["passed"], rec["why"] = False, rec["error"]
        return rec
    ok, why = case.check(rec)
    rec["passed"], rec["why"] = bool(ok), why
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", default="fence,gate,resume,say,carry,where,replay")
    ap.add_argument("--arm", default="both", choices=["old", "new", "both"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--holdout", action="store_true", help="only the held-out cases")
    ap.add_argument("-k", default="", help="only cases whose name contains one of these (comma-separated)")
    ap.add_argument("--offline", nargs="?", const=str(RAW), help="summarize saved results again")
    ap.add_argument("--failures", default=str(ROOT / "workspace" / "failures.db"), help="the failure log to replay")
    a = ap.parse_args()
    if a.offline:
        data = json.loads(Path(a.offline).read_text(encoding="utf-8"))
        print(summarize([grade(r) for r in data["records"]], data.get("replay")))
        return 0
    stages = a.stage.split(",")
    replayed = replay(Path(a.failures)) if "replay" in stages and Path(a.failures).is_file() else None
    cases = [c for c in CASES if c.stage in stages and (c.holdout or not a.holdout)
             and (not a.k or any(x in c.name for x in a.k.split(",")))]
    arms = ["old", "new"] if a.arm == "both" else [a.arm]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = RUNS / stamp
    records = []
    for rep in range(a.reps if cases else 0):
        for case in cases:
            for which in arms:
                where = out_dir / f"{case.name}-{which}-{rep}"
                print(f"[{time.strftime('%H:%M:%S')}] {case.name} {which} rep {rep + 1}...", flush=True)
                with arm(case.stage, which):
                    try:
                        rec = run_case(case, which, where)
                    except Exception as e:  # noqa: BLE001
                        rec = {"case": case.name, "stage": case.stage, "kind": case.kind, "arm": which,
                               "holdout": case.holdout, "answer": "", "error": f"bench: {type(e).__name__}: {e}",
                               "seconds": 0, "code": [], "outputs": [], "steps": 0, "bounces": [], "told": []}
                rec["rep"] = rep
                records.append(grade(rec))
                print(f"    {'PASS' if rec['passed'] else 'FAIL'} ({rec['seconds']}s): {rec['why']}", flush=True)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "raw.json").write_text(json.dumps({"records": records, "replay": replayed}, indent=1,
                                                             ensure_ascii=False, default=str), encoding="utf-8")
    text = summarize(records, replayed)
    print(text)
    if not records:   # (a replay-only run must not overwrite the last live results)
        return 0
    RAW.write_text(json.dumps({"records": records, "replay": replayed}, indent=1, ensure_ascii=False, default=str),
                   encoding="utf-8")
    SUMMARY.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
