"""Live bench for messages sent while a task runs (jevh/interject.py): real Jev, a real worker model, real turns.

The offline tests (tests/test_interject.py) use a fake Jev and scripted models: they show the code runs. This shows
whether it WORKS: does Jev tell a correction from a question from another request, does a correction change what
the task delivers (and does the final() gate then ask for the new thing, not the old), how fast does a message
cut a model reply that is still being written, does Stop work while the model is still reading its prompt, does
the quick reply answer while the task carries on, and what does each cost the model server.

    python evals/bench/interject.py --model lmstudio:qwen3.8-27b@q6_k       # every stage, 2 reps
    python evals/bench/interject.py --stage inflight,stop --reps 3 --model lmstudio:qwen3.8-27b@q6_k
    python evals/bench/interject.py --report                                 # the last run's summary

Pin the model with --model: the harness's LM Studio gate unloads every other model when a task needs one (the fast
tier is a different model), and another program sharing the server (Hermes) loses the one it was using. Nothing else
should use the model during a run: a second client's requests share its slots and the timings.

Needs: the Jev adapter (config jev.url) and LM Studio with the model (the server-side numbers need LM Studio's
log, read from ~/.lmstudio/server-logs or LMSTUDIO_LOG_DIR; on another server they are left out). Nothing touches
your real workspace: every rep runs in evals/bench/runs/interject/<stamp>/<case>-<n>/ on a copy of personas/.
The classification itself is scored by evals/interject_cases.py (dev, held-out and fresh sets); this bench runs it
as its first stage.

Stages (pass targets are first guesses, printed beside each result: a miss is a finding, not a bug in the bench)
  classify   evals/interject_cases.py on the three sets: a question about progress is answered on the side, and no
             correction, other request or stop is (they are added to the prompt; /queue and /stop are commands)
  add        a plain message that reads like another request is added to the task's prompt, not queued, and the
             task does both
  boundary   a correction sent when step 1's code starts reaches the task at its next step and changes what it
             delivers; the gate does not ask for the old file
  inflight   a correction sent while the model is still reading its prompt (1 s in), and again once it is writing, cuts
             the reply at once (target: within 3 s of the message) and the task follows the correction; the
             server-side cost is reported as the share of the prompt it had to read again for the retry (LM
             Studio: ~100% when the cut came while it read the prompt, a few % when it was writing)
  replace    "actually count the python files instead" swaps the request: the old file is not demanded, the new
             one is delivered
  status     "how is it going?" while the code runs: an instant line, then the task's own model answers in its
             own words (digest and full contexts), and the task still finishes right
  queue      "/queue ..." waits, runs as the next turn by itself
  stop       "/stop" while the model is still reading its prompt, and while code runs: the turn ends in seconds

If the feature is wrong, expect: a correction that never reaches the task or is followed by the old deliverable, a
cut that lands only when the model next speaks (the bug this bench found: closing the response does not wake a read
that is waiting for the server), a gate that bounces the answer for not delivering the file the user replaced, a quick
reply that stalls the task, a Stop that waits for the first token.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))
sys.path.insert(0, str(HERE))

import interject_cases as ic  # noqa: E402
import lmstudio_log  # noqa: E402

from jevh.config import load_config  # noqa: E402
from jevh.orchestrator import Harness  # noqa: E402

RUNS = HERE / "runs" / "interject"
DEFAULT_STAGES = "classify,boundary,inflight,replace,add,status,queue,stop"
SUMMARY = HERE / "interject_last_summary.txt"   # a full run; a run of some stages writes interject_<stages>_summary.txt
FIB = ("Write a Python script that prints the first 20 Fibonacci numbers, save it as fib.py in your working "
       "folder, run it, and tell me its output.")
PRIMES = "actually make it the first 15 prime numbers instead, in primes.py"
ALSO = "also write me a haiku about autumn"
SQUARES = "make it the first 12 square numbers instead, in squares.py"
SLEEP = "Run exactly this Python in your REPL and tell me what it prints: import time; time.sleep(45); print('finished waiting')"
ARCH = ROOT / "docs" / "architecture"
CORE = ROOT / "jevh"
COUNT = (f"How many markdown (.md) files are directly inside {ARCH.as_posix()}? Write the number to count.txt "
         "and tell me.")
RECOUNT = f"Actually count the python (.py) files in {CORE.as_posix()} instead, and write that number to count_py.txt"


# --------------------------------------------------------------------------- scaffolding
def make_cfg(where: Path, interject: dict | None = None):
    cfg = load_config()
    shutil.rmtree(where, ignore_errors=True)
    cfg.workspace = where / "ws"
    cfg.personas_dir = where / "personas"
    cfg.personas_dirs = [cfg.personas_dir]
    shutil.copytree(ROOT / "personas", cfg.personas_dir, ignore=shutil.ignore_patterns("learned"))
    cfg.skills_dirs = [where / "skills", ROOT / "skills"]
    (where / "skills").mkdir(parents=True, exist_ok=True)
    if interject:
        cfg.raw = {**cfg.raw, "interject": {**(cfg.raw.get("interject") or {}), **interject}}
    return cfg


class Rig:
    """One harness on a throwaway folder; every event is kept with the time it happened."""

    def __init__(self, where: Path, model: str | None, interject: dict | None = None):
        self.cfg = make_cfg(where, interject)
        self.events: list[tuple[float, str, dict]] = []
        self.t0 = time.time()
        self.h = Harness(self.cfg, on_event=lambda k, d: self.events.append((time.time() - self.t0, k, d)),
                         model_override=model)
        self.h.learning = False
        self.box: dict = {}
        self.thread: threading.Thread | None = None

    def now(self) -> float:
        return time.time() - self.t0

    def start(self, task: str) -> None:
        self.thread = threading.Thread(target=lambda: self.box.update(r=self._run(task)), daemon=True)
        self.thread.start()

    def _run(self, task):
        try:
            return self.h.turn(task)
        except BaseException as e:  # noqa: BLE001 - shown in the result
            self.box["error"] = f"{type(e).__name__}: {e}"

    def wait_for(self, kind: str, timeout: float = 300, after: float = 0.0) -> tuple[float, dict] | None:
        end = time.time() + timeout
        while time.time() < end:
            for t, k, d in self.events:
                if k == kind and t >= after:
                    return t, d
            if self.thread and not self.thread.is_alive():
                return None
            time.sleep(0.05)
        return None

    def first(self, kind: str, after: float = 0.0) -> tuple[float, dict] | None:
        return next(((t, d) for t, k, d in self.events if k == kind and t >= after), None)

    def finish(self, timeout: float = 900):
        self.thread.join(timeout)
        return self.box.get("r")

    def streamed(self) -> int:
        """Characters of the lead worker's current reply written so far (its live thinking file): 0 while the model
        is still reading its prompt."""
        files = sorted(self.cfg.workspace.glob("sessions/*/u1t1-*/thinking/step-*.txt"))
        return files[-1].stat().st_size if files else 0

    def wait_streaming(self, chars: int = 400, timeout: float = 120) -> bool:
        end = time.time() + timeout
        while time.time() < end and self.thread and self.thread.is_alive():
            if self.streamed() >= chars:
                return True
            time.sleep(0.1)
        return False

    def conversation(self) -> str:
        """Everything the lead worker's model was shown, from the checkpoint it saved when it finished."""
        out = []
        for f in sorted(self.cfg.workspace.glob("sessions/*/u1t1-*/checkpoint.json")):
            try:
                out += [str(m.get("content", "")) for m in json.loads(f.read_text(encoding="utf-8")).get("messages", [])]
            except (OSError, ValueError):
                pass
        return "\n".join(out)

    def problems(self) -> list[str]:
        return [p for _, k, d in self.events if k == "verify" for p in (d.get("problems") or [])]

    def bounces(self) -> int:
        return sum(1 for _, k, d in self.events if k == "verify" and d.get("bounced"))

    def server(self, t_end: float | None = None) -> list[dict]:
        """What LM Studio's llama.cpp did with this rig's requests (empty when its log cannot be read)."""
        try:
            return lmstudio_log.between(self.t0 - 2, (t_end or time.time()) + 3)
        except OSError:
            return []


def compact(tasks: list[dict]) -> list[dict]:
    return [{"slot": t["slot"], "how": t.get("how"), "sim": t.get("sim"), "read": t["processed"],
             "prompt": lmstudio_log.prompt_tokens(t), "gen": t["generated"], "cancelled": t["cancelled"]} for t in tasks]


def result(checks: dict, metrics: dict, rig: Rig, r, **extra) -> dict:
    return {"checks": checks, "metrics": metrics, "answer": (getattr(r, "answer", "") or "")[:600],
            "error": rig.box.get("error"), **extra}


# --------------------------------------------------------------------------- the cases
def case_boundary(where: Path, model: str | None, **_) -> dict:
    rig = Rig(where, model)
    rig.start(FIB)
    rig.wait_for("code", 300)
    sent = rig.now()
    res = rig.h.interject(PRIMES)
    r = rig.finish()
    steer = rig.first("steer", sent)
    arts = [Path(a).name for a in getattr(r, "artifacts", [])]
    ok = {"classified_as_a_correction": res.get("outcome") == "corrected", "delivered_to_the_task": steer is not None,
          "task_follows_it": bool(r) and "primes.py" in arts and bool(re.search(r"prime|2, 3, 5|2 3 5", r.answer, re.I)),
          "gate_does_not_ask_for_the_old_file": not any("fib.py" in p for p in rig.problems()),
          "finished": bool(r) and not r.cancelled}
    return result(ok, {"delivered_s": round(steer[0] - sent, 1) if steer else None, "total_s": round(rig.now(), 1),
                       "gate_bounces": rig.bounces()},
                  rig, r, probs=res.get("probs"), server=compact(rig.server()))


def retry_read(tasks: list[dict]) -> float | None:
    """The share of its prompt the server had to read again for the request after a cut one (None: no log, or no cut)."""
    for i, t in enumerate(tasks):
        if t["cancelled"]:
            for u in tasks[i + 1:]:
                pl = lmstudio_log.prompt_tokens(u)
                if u["processed"] is not None and pl:
                    return round(min(1.0, u["processed"] / pl), 2)
    return None


def case_inflight(where: Path, model: str | None, phase: str = "reading", retry_delay: float | None = None, **_) -> dict:
    """phase "reading": 1 s after the task starts (the model is reading its prompt: a new session is not in its cache);
    "writing": once the reply has streamed 400 characters (the prompt is in the server's cache by then)."""
    rig = Rig(where, model, {"retry_delay": retry_delay} if retry_delay is not None else None)
    rig.start(FIB)
    started = rig.wait_for("start", 120)
    if phase == "writing":
        rig.wait_streaming(400)
    else:
        time.sleep(1)
    sent = rig.now()
    res = rig.h.interject(SQUARES)
    cut = rig.wait_for("redirect", 60, after=sent - 0.1)
    r = rig.finish()
    arts = [Path(a).name for a in getattr(r, "artifacts", [])]
    cut_s = round(cut[0] - sent, 1) if cut else None
    ok = {"classified_as_a_correction": res.get("outcome") == "corrected", "reply_cut": cut is not None,
          "cut_within_3s": cut_s is not None and cut_s <= 3.0,
          "task_follows_it": bool(r) and "squares.py" in arts and bool(re.search(r"square|144|1, 4, 9", r.answer, re.I)),
          "gate_does_not_ask_for_the_old_file": not any("fib.py" in p for p in rig.problems()),
          "finished": bool(r) and not r.cancelled}
    kept = cut[1].get("kept") if cut else None
    tasks = rig.server()
    reread = retry_read(tasks)
    return result(ok, {"cut_s": cut_s, "kept_chars": kept, "retry_reread": reread, "total_s": round(rig.now(), 1),
                       "gate_bounces": rig.bounces()},
                  rig, r, probs=res.get("probs"), server=compact(tasks), started=bool(started), phase=phase)


def case_replace(where: Path, model: str | None, **_) -> dict:
    flat = len(list(CORE.glob("*.py")))
    deep = len(list(CORE.rglob("*.py")))
    rig = Rig(where, model)
    rig.start(COUNT)
    rig.wait_for("code", 300)
    sent = rig.now()
    res = rig.h.interject(RECOUNT)
    r = rig.finish()
    arts = [Path(a).name for a in getattr(r, "artifacts", [])]
    steer = rig.first("steer", sent)
    ok = {"classified_as_a_correction": res.get("outcome") == "corrected",
          "seen_as_a_swap": bool(res.get("replaces")), "delivered_to_the_task": steer is not None,
          "new_file_delivered": "count_py.txt" in arts,
          "right_count": bool(r) and any(re.search(rf"\b{n}\b", r.answer) for n in (flat, deep)),
          "gate_does_not_ask_for_the_old_file": not any("count.txt" in p for p in rig.problems()),
          "finished": bool(r) and not r.cancelled}
    return result(ok, {"total_s": round(rig.now(), 1), "gate_bounces": rig.bounces(), "replace_p": (res.get("probs") or {}).get("replace")},
                  rig, r, probs=res.get("probs"), counts=[flat, deep], server=compact(rig.server()))


def case_status(where: Path, model: str | None, context: str = "digest", **_) -> dict:
    rig = Rig(where, model, {"aside_context": context})
    rig.start(SLEEP)
    rig.wait_for("code", 300)
    time.sleep(8)                 # the model is idle now: the code is sleeping
    sent = rig.now()
    t_call = time.time()
    res = rig.h.interject("how is it going?")
    instant_s = round(time.time() - t_call, 1)
    aside = rig.wait_for("aside", 240, after=sent - 0.1)
    r = rig.finish()
    a = aside[1] if aside else {}
    ok = {"classified_as_a_question": res.get("outcome") == "status", "instant_line": bool(res.get("answer")),
          "instant_within_3s": instant_s <= 3.0, "model_answered": a.get("source") == "model",
          "answer_is_about_the_task": bool(re.search(r"sleep|wait|45|second|running|code|print|step", a.get("answer", ""), re.I)),
          "task_finished_right": bool(r) and "finished waiting" in r.answer.lower(),
          "task_never_saw_the_question": bool(rig.conversation()) and "how is it going" not in rig.conversation().lower()}
    tasks = rig.server()
    return result(ok, {"instant_s": instant_s, "aside_s": round(aside[0] - sent, 1) if aside else None,
                       "aside_in": a.get("input_tokens"), "aside_out": a.get("output_tokens"), "total_s": round(rig.now(), 1)},
                  rig, r, context=context, instant=res.get("answer"), aside=a.get("answer"), server=compact(tasks))


def case_add(where: Path, model: str | None, **_) -> dict:
    """A plain message that reads like another request is added to the task's prompt, not queued: only /queue queues."""
    rig = Rig(where, model)
    rig.start(FIB)
    rig.wait_for("code", 300)
    sent = rig.now()
    res = rig.h.interject(ALSO)
    r = rig.finish()
    steer = rig.first("steer", sent)
    ok = {"added_to_the_prompt": res.get("outcome") == "corrected", "not_queued": rig.h.queued() == [],
          "task_heard_it": steer is not None, "finished": bool(r) and not r.cancelled,
          "task_did_both": bool(r) and ("987" in r.answer or "fib" in r.answer.lower())
          and bool(re.search(r"autumn|leaves|fall", r.answer, re.I))}
    return result(ok, {"delivered_s": round(steer[0] - sent, 1) if steer else None, "total_s": round(rig.now(), 1),
                       "gate_bounces": rig.bounces()}, rig, r, probs=res.get("probs"))


def case_status_control(where: Path, model: str | None, **_) -> dict:
    """The same task with no message: what the main task's next request costs the server when nobody interrupts."""
    rig = Rig(where, model)
    rig.start(SLEEP)
    rig.wait_for("code", 300)
    r = rig.finish()
    return result({"task_finished_right": bool(r) and "finished waiting" in r.answer.lower()},
                  {"total_s": round(rig.now(), 1)}, rig, r, server=compact(rig.server()))


def case_queue(where: Path, model: str | None, **_) -> dict:
    rig = Rig(where, model)
    rig.start(FIB)
    rig.wait_for("code", 300)
    res = rig.h.interject("/queue then write a haiku about autumn")
    r = rig.finish()
    nxt = rig.h.take_queued()
    r2 = rig.h.turn(nxt) if nxt else None
    lines = [ln for ln in (r2.answer if r2 else "").splitlines() if ln.strip()]
    ok = {"queued": res.get("outcome") == "queued", "first_task_untouched": bool(r) and ("fib" in r.answer.lower() or "987" in r.answer),
          "next_turn_was_the_queued_request": nxt == "then write a haiku about autumn",
          "next_turn_answered": r2 is not None and len(lines) >= 3}
    return result(ok, {"total_s": round(rig.now(), 1)}, rig, r2 or r, queued=nxt)


def case_stop(where: Path, model: str | None, phase: str = "reading", **_) -> dict:
    rig = Rig(where, model)
    rig.start(SLEEP if phase == "code" else FIB)
    if phase == "code":
        rig.wait_for("code", 300)
        time.sleep(3)
    else:
        rig.wait_for("start", 120)
        time.sleep(3)             # the model is reading its prompt or thinking
    sent = time.time()
    res = rig.h.interject("/stop")
    rig.thread.join(60)
    stopped_s = round(time.time() - sent, 1)
    r = rig.box.get("r")
    ok = {"stop_accepted": res.get("outcome") == "stopped", "turn_ended": not rig.thread.is_alive(),
          "within_10s": stopped_s <= 10.0, "reported_as_stopped": bool(r) and r.cancelled}
    return result(ok, {"stopped_s": stopped_s}, rig, r, phase=phase)


CASES = {
    "boundary": [("boundary", case_boundary, {})],
    "inflight": [("inflight-reading", case_inflight, {"phase": "reading"}),
                 ("inflight-writing", case_inflight, {"phase": "writing"})],
    "replace": [("replace", case_replace, {})],
    "status": [("status-digest", case_status, {"context": "digest"}), ("status-full", case_status, {"context": "full"}),
               ("status-control", case_status_control, {})],
    "add": [("add", case_add, {})],
    "queue": [("queue", case_queue, {})],
    # does a short wait before the retry keep it on the slot that holds the prompt? (run with --reps 5)
    "retry": [(f"retry-delay-{d}", case_inflight, {"phase": "writing", "retry_delay": d}) for d in (0.0, 0.5, 1.0, 2.0)],
    "stop": [("stop-reading", case_stop, {"phase": "reading"}), ("stop-code", case_stop, {"phase": "code"})],
}


# --------------------------------------------------------------------------- the run
def run_classify() -> dict:
    out = {}
    status_at = load_config().t("interject_status")
    for name, cases in (("dev", ic.DEV), ("held", ic.HELD), ("fresh", ic.FRESH)):
        e = ic.evaluate(ic.run_live(cases), status_at)
        out[name] = {k: e[k] for k in ("right", "total", "questions", "questions_answered", "corrections_answered",
                                       "others_answered")}
    return out


def median(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.median(xs), 1) if xs else None


def summarize(res: dict) -> str:
    lines = [f"INTERJECTION BENCH  {res['stamp']}  model={res.get('model') or 'the personas own'}  reps={res['reps']}", ""]
    if "classify" in res:
        lines.append("classify (Jev, evals/interject_cases.py)   target: no correction, other request or stop answered "
                     "on the side instead of added to the prompt")
        for name, c in res["classify"].items():
            lines.append(f"  {name:6s} {c['right']}/{c['total']} right   questions answered {c['questions_answered']}/{c['questions']}   "
                         f"corrections answered instead of heard {c['corrections_answered']}   "
                         f"other requests / stops answered instead of heard {c['others_answered']}")
        lines.append("")
    for name, reps in res.get("cases", {}).items():
        n = len(reps)
        lines.append(f"{name}  ({n} reps)")
        checks = sorted({c for r in reps for c in r["checks"]})
        for c in checks:
            ok = sum(1 for r in reps if r["checks"].get(c))
            lines.append(f"  {'ok  ' if ok == n else 'MISS'} {c}: {ok}/{n}")
        ms = sorted({m for r in reps for m in r["metrics"]})
        shown = [f"{m} {median([r['metrics'].get(m) for r in reps])}" for m in ms]
        if shown:
            lines.append("  medians: " + ", ".join(shown))
        if any("retry_reread" in r["metrics"] for r in reps):
            lines.append("  retry_reread per rep (share of the prompt the server read again; ~0.1 = from its cache, 1.0 = from scratch): "
                         + str([r["metrics"].get("retry_reread") for r in reps]))
        for r in reps:
            if r.get("error"):
                lines.append(f"  ERROR: {r['error']}")
        srv = [r.get("server") for r in reps if r.get("server")]
        if srv:
            lines.append("  server (llama.cpp, rep 1): " + "; ".join(
                f"slot{t['slot']} {t['how'] or '?'} sim {t['sim']} read {t['read']}/{t['prompt']}"
                + (" CUT" if t["cancelled"] else "") for t in srv[0][:10]))
        for r in reps[:1]:
            if r.get("aside"):
                lines.append(f"  instant: {r.get('instant')}")
                lines.append(f"  quick reply ({r.get('context')}): {r.get('aside')}")
        lines.append("")
    lines.append(f"total {res.get('seconds')}s; raw results in {RUNS / res['stamp']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default=DEFAULT_STAGES, help="also: retry (not in the default run)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--model", help="pin every persona to one model, e.g. lmstudio:qwen3.8-27b@q6_k (recommended)")
    ap.add_argument("--report", action="store_true", help="print the last run's summary")
    a = ap.parse_args()
    if a.report:
        print(SUMMARY.read_text(encoding="utf-8") if SUMMARY.exists() else "no runs yet")
        return
    summary = SUMMARY if a.stage == DEFAULT_STAGES else HERE / f"interject_{a.stage.replace(',', '_')}_summary.txt"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = RUNS / stamp
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {"stamp": stamp, "model": a.model, "reps": a.reps, "cases": {}}
    t0 = time.time()
    stages = a.stage.split(",")
    if "classify" in stages:
        print("classify ...", flush=True)
        res["classify"] = run_classify()
    for stage, cases in CASES.items():
        if stage not in stages:
            continue
        for name, fn, kw in cases:
            reps = []
            for i in range(1, a.reps + 1):
                print(f"{name} rep {i}/{a.reps} ...", flush=True)
                try:
                    reps.append(fn(out / f"{name}-{i}", a.model, **kw))
                except Exception as e:  # noqa: BLE001 - one broken rep must not lose the rest
                    reps.append({"checks": {"ran": False}, "metrics": {}, "error": f"{type(e).__name__}: {e}"})
                (out / "results.json").write_text(json.dumps({**res, "seconds": round(time.time() - t0)}, indent=1,
                                                             default=str), encoding="utf-8")
            res["cases"][name] = reps
    res["seconds"] = round(time.time() - t0)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    text = summarize(res)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    summary.write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
