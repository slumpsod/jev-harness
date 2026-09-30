"""The local model's own reading of a failure, after it was replayed on every local model and before
any variant is written: did the harness cause it, could the harness have prevented it, or was it the
model alone (or no failure at all)? Only the first two are worth a night of test writing and fixing.

Live (2026-09-30), what each category looked like:
  harness_caused       the final() gate sent back four correct answers (2026-10-02, 18:05, 28.4 BLEU, a
                       deck) - the gate misjudged, the worker lost steps redoing right work
  harness_preventable  save_artifact("planets.csv", "planets.csv") saved the name as the content - the
                       model's slip, but the API invited it and no check caught the 11-byte file
  model_only           ornith wrote {"name": "name"} for "Ana" converting a two-row CSV
  not_a_failure        a correct planets.csv run logged as missed_request (the check didn't see the file)

The replays are the evidence the reading has to square with: a failure that comes back on several
models points at the harness, whatever the reading says (see keep())."""
from __future__ import annotations

from typing import Any

from .generate import parse_json

CATEGORIES = ("harness_caused", "harness_preventable", "model_only", "not_a_failure")
KEEP = ("harness_caused", "harness_preventable")

PROMPT = """You are triaging a failure logged by an AI assistant harness (jev-harness: a worker model acts \
in a Python REPL with tools; the harness routes the task, hands the worker helpers like save_artifact() and \
final(), checks the answer with a final() gate before it goes out, and logs failures). Decide what caused \
this one, so that only failures the harness can do something about are worked on.

## The failure
Kind: {kind} ({source}) - {summary}
{detail}
Persona: {persona}, model: {model}
Request: {request}
Its answer: {answer}
End of its work log:
{log}

## The same request replayed since, on each local model
A replay counts as failed when {how}.
{replays}

## Categories
- harness_caused: a harness part did the wrong thing - the final() gate sent back or passed the wrong \
answer, a tool or helper misbehaved, a prompt or limit misled the worker, routing picked the wrong assistant.
- harness_preventable: the model made the mistake, but the harness invited it (an unclear helper or \
instruction) or could have caught it (a check it doesn't do) - a change to the harness would stop it.
- model_only: the model slipped on something the harness gave it everything for; no reasonable harness \
change would prevent it.
- not_a_failure: nothing went wrong - the result was right and nothing had to be redone; the failure was \
logged by mistake.

Reply with ONLY a JSON object: {{"category": "<one of the four>", "component": "<the harness part involved, \
e.g. save_artifact, the final() gate's number check, the web tool - or empty>", "evidence": "<the lines of the \
work log or replays that show it>", "why": "<one or two sentences>"}}"""


CHECKER_PROMPT = """A test case for an AI assistant harness failed on every run, on every model. Before that \
counts against the harness, check the test itself.

Request: {request}
The checker that decided each run failed: {checker}
Its verdict on this run: {why}
The run's answer: {answer}
Files the run handed over:
{files}

Is this result correct for the request - so the checker is wrong or too strict (e.g. it demands one layout \
when the request allows others) - or is the result really wrong?
Reply with ONLY a JSON object: {{"checker_wrong": true or false, "why": "<one sentence>"}}"""


def run_files(workspace: str, cap: int = 1500) -> str:
    """The files a test run handed over (from its result.json), each cut to `cap` characters."""
    import json
    from pathlib import Path
    try:
        res = json.loads((Path(workspace) / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "(none)"
    out = []
    for a in res.get("artifacts") or []:
        p = Path(a)
        try:
            out.append(f"--- {p.name} ({p.stat().st_size} bytes)\n{p.read_text(encoding='utf-8', errors='replace')[:cap]}")
        except OSError:
            out.append(f"--- {p.name}: missing")
    return "\n".join(out) or "(none)"


def review_checker(llm, case, row: dict) -> tuple[bool, str]:
    """(checker_wrong, why) for a case that failed every run. Live: an XML checker accepted only
    <product id="101"> and failed a correct <product><id>101</id> file on every run. Unreadable: (False, ...)."""
    import json
    reply = llm.chat([{"role": "user", "content": CHECKER_PROMPT.format(
        request=case.request[:2000], checker=json.dumps(case.checker)[:2500], why=str(row.get("why", ""))[:300],
        answer=str(row.get("answer", ""))[:1500], files=run_files(row.get("workspace", "")))}]).text
    got = parse_json(reply, "{")
    if isinstance(got, dict) and isinstance(got.get("checker_wrong"), bool):
        return got["checker_wrong"], str(got.get("why") or "")[:400]
    return False, "no reading could be parsed; kept"


def replay_text(rows: list[dict], models: list[str]) -> str:
    """Per model: how many replays failed, with why and the answer of up to two failed ones."""
    out = []
    for m in models:
        mine = [r for r in rows if r["model"] == m]
        failed = [r for r in mine if r.get("reproduced", not r["passed"])]
        out.append(f"- {m}: {len(failed)} of {len(mine)} failed")
        for r in failed[:2]:
            out.append(f"    run {r['rep']}: {str(r.get('why', ''))[:200]} | answer: {str(r.get('answer', ''))[:300]!r}")
    return "\n".join(out) or "(not replayed)"


def triage(llm, failure: dict, rows: list[dict], models: list[str], how: str) -> dict[str, Any]:
    """-> {category, component, evidence, why, reply}; category "unknown" if no reading could be parsed."""
    reply = ""
    for _ in range(2):
        reply = llm.chat([{"role": "user", "content": PROMPT.format(
            kind=failure["kind"], source=failure.get("source", ""), summary=failure["summary"],
            detail=f"Details: {failure['detail'][:1500]}" if failure.get("detail") else "",
            persona=failure.get("persona") or "-", model=failure.get("model") or "-",
            request=failure["request"][:2500], answer=(failure.get("answer") or "(none)")[:1500],
            log=(failure.get("work_log") or "(no code was run)")[-3000:], how=how,
            replays=replay_text(rows, models))}]).text
        got = parse_json(reply, "{")
        if isinstance(got, dict) and got.get("category") in CATEGORIES:
            return {"category": got["category"], "component": str(got.get("component") or "")[:200],
                    "evidence": str(got.get("evidence") or "")[:1500], "why": str(got.get("why") or "")[:800],
                    "reply": reply[-4000:]}
    return {"category": "unknown", "component": "", "evidence": "", "why": "no reading could be parsed",
            "reply": reply[-4000:]}


def keep(reading: dict, failing_models: list[str]) -> tuple[bool, str]:
    """Work on it? The reading decides, except that a failure seen on 2+ models is kept whatever it says:
    a slip one model makes is noise, one several make points at what they were all given."""
    cat = reading["category"]
    if cat in KEEP:
        return True, f"{cat}" + (f" ({reading['component']})" if reading.get("component") else "")
    if len(failing_models) >= 2 and cat != "not_a_failure":
        return True, f"read as {cat}, but it failed on {len(failing_models)} models"
    if cat == "unknown":
        return True, "no reading could be parsed; kept (the replays showed a failure)"
    return False, f"{cat}: {reading.get('why', '')}"
