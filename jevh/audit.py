"""Spot checks: a model rereads a finished task - the request, the answer as delivered, the files handed
over and the work log - and says whether the answer is actually right. They catch what the final() gate
and the failure log's Jev check don't: confident wrong answers, which read fine and pass every check of
the answer's text. A wrong one is logged as a failure (source "audit", kind "wrong_answer") for the
refinement loop.

Live (2026-09-30), what they are for: a regex summary that said "6 examples (4 valid, 2 invalid), all
correct" over a log that tested 3 and 3 and never showed the results; a handed-over planets.csv that held
only the text "planets.csv"; CSV -> JSON that turned "Ana" into "name". None of them was logged.

Config (failures:): audit: share of finished tasks checked (0 = off); audit_model: "provider:model", or
null for the task's own model (no model swap in LM Studio; a model rereading its own work with the
evidence in front of it still catches slips like these - see evals/audit_cases.py)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PROMPT = """Check an AI assistant's finished task like a careful reviewer: is the answer it gave the user \
actually right?

## The request
{request}

## The answer the user got
{answer}

## Files handed over with it
{files}

## The work log (the code it ran and what came back)
{log}

## Check
- Every number, count, name and result the answer states: does the work log or a file back it up, and \
does it match exactly? (A count of test cases, a total, which row was picked, a file's contents.)
- Does it deliver what the request asked for - all the parts, in the format asked - and do the files \
hold what the request and the answer say they hold?
- Facts from the assistant's own knowledge (no code run) are fine if they are correct.
- Don't judge style or wording; only whether the user got a right and complete result.

Reply with ONLY a JSON object: {{"verdict": "right" | "wrong" | "unsure", "problems": ["<each thing \
that is wrong, with the evidence>"]}}"""

TEXT = {".txt", ".md", ".csv", ".json", ".py", ".js", ".css", ".html", ".htm", ".svg", ".xml", ".yaml", ".yml",
        ".toml", ".sql", ".ts", ".tsv", ".ini", ".cfg", ".log"}


def files_text(artifacts: list[str], cap: int = 2000) -> str:
    out = []
    for a in artifacts:
        p = Path(a)
        if not p.is_file():
            out.append(f"--- {p.name}: MISSING")
            continue
        size = p.stat().st_size
        if p.suffix.lower() in TEXT:
            text = p.read_text(encoding="utf-8", errors="replace")
            out.append(f"--- {p.name} ({size} bytes)\n{text[:cap]}" + ("\n[... cut]" if len(text) > cap else ""))
        else:
            out.append(f"--- {p.name} ({size} bytes, binary)")
    return "\n".join(out) or "(none)"


def parse(reply: str) -> dict[str, Any] | None:
    """The JSON object in a reply (fences and chatter around it allowed)."""
    text = re.sub(r"```(?:json)?", "", reply or "")
    for m in re.finditer(r"\{", text):
        try:
            got = json.JSONDecoder().raw_decode(text[m.start():])[0]
        except ValueError:
            continue
        if isinstance(got, dict) and "verdict" in got:
            return got
    return None


def audit(llm, request: str, answer: str, artifacts: list[str], log: str) -> dict[str, Any]:
    """-> {verdict: right | wrong | unsure, problems: [...]}; "unsure" when the reply can't be read."""
    reply = llm.chat([{"role": "user", "content": PROMPT.format(
        request=request[:3000], answer=answer[:3000], files=files_text(artifacts),
        log=(log or "(no code was run)")[-12000:])}]).text
    got = parse(reply)
    if not got or got.get("verdict") not in ("right", "wrong", "unsure"):
        return {"verdict": "unsure", "problems": [], "reply": (reply or "")[-1500:]}
    problems = [str(p) for p in got.get("problems") or [] if str(p).strip()]
    return {"verdict": got["verdict"], "problems": problems[:8]}
