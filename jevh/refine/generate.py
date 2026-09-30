"""The local model writes the test cases for a failure: a checker for the failed request itself,
then variants that hit the same weakness with different specifics. Every checker must tell a
good answer from a bad one (checks.discriminates) or its case is dropped - that is what lets a
local model write tests without a bad checker deciding the results."""
from __future__ import annotations

import json
import random
import re
from typing import Any

from . import checks
from .cases import Case

CHECKER_TYPES = """Checker types (JSON):
  {"type": "contains", "text": "..."}            the answer contains the text (case-insensitive)
  {"type": "not_contains", "text": "..."}
  {"type": "regex", "pattern": "..."}
  {"type": "number", "value": 412, "tolerance": 0.5}   a number in the answer, within tolerance
  {"type": "json_equals", "value": [...]}        the answer is exactly this JSON
  {"type": "artifact", "name": "*.html", "contains": "..."}   the task hands over such a file
  {"type": "python", "code": "def check(answer, artifacts):\\n    return ..."}   True/False from the answer text
  {"type": "jev", "question": "Does `answer` ...?", "min": 0.7}   a yes/no judgement of the answer (last resort)
  {"type": "all", "checks": [...]}  /  {"type": "any", "checks": [...]}
Prefer exact checks (number, contains, json_equals, artifact). Check what a correct result must
contain, not the wording. A checker must pass good_answer and fail bad_answer and a refusal."""

REPLAY_PROMPT = """An AI assistant harness failed a task. Write a checker that decides whether an answer to \
the SAME request succeeds.

Request: {request}

The failed answer (your checker must reject it): {answer}

Why it counts as a failure: {kind} - {summary}
{detail}
{types}

Reply with ONLY a JSON object: {{"checker": {{...}}, "good_answer": "<a short answer that does the task and \
passes your checker>"}}"""

VARIANTS_PROMPT = """An AI assistant harness failed a task. Write {n} NEW test tasks that would hit the same \
weakness, so a fix can be tested on more than this one case.

The failed task: {request}
Its answer: {answer}
Failure: {kind} ({source}) - {summary}
{detail}
What it did (end of its work log):
{log}

Rules:
- Each task is a complete request a user could send, on its own.
- Same kind of task and the same weakness, but different specifics (topic, site, numbers, files, wording):
  not copies of the failed task, and not all alike.
- Prefer tasks whose correct result can be checked exactly and doesn't change from day to day (computations,
  files to produce, fixed facts), unless the failure is about something else (e.g. live websites).
{avoid}
{types}

Reply with ONLY a JSON list: [{{"request": "...", "checker": {{...}}, "good_answer": "a short correct answer \
that passes the checker", "bad_answer": "an answer that shows the same failure and must fail the checker"}}, ...]"""


def parse_json(text: str, kind: str = "[") -> Any:
    """The first JSON list/object in a model's reply (code fences and chatter around it allowed)."""
    text = re.sub(r"```(?:json)?", "", text or "")
    close = "]" if kind == "[" else "}"
    start = text.find(kind)
    while start != -1:
        try:
            return json.JSONDecoder().raw_decode(text[start:])[0]
        except ValueError:
            end = text.rfind(close)
            if end > start:
                try:
                    return json.loads(text[start:end + 1])
                except ValueError:
                    pass
        start = text.find(kind, start + 1)
    return None


class Generator:
    """`llm`: a local model client (.chat(messages) -> .text); `judge`: Jev, for jev checkers."""

    def __init__(self, llm, judge: checks.Judge | None = None, cases: int = 10, holdout: float = 0.3,
                 rounds: int = 3, seed: int = 0):
        self.llm, self.judge = llm, judge
        self.n, self.holdout, self.rounds, self.seed = cases, holdout, rounds, seed
        self.dropped: list[dict] = []   # (request, why) of cases whose checker didn't hold up

    def ask(self, prompt: str) -> str:
        return self.llm.chat([{"role": "user", "content": prompt}]).text

    def replay(self, failure: dict) -> Case | None:
        """The failed request with a checker that rejects the answer it got."""
        for _ in range(self.rounds):
            got = parse_json(self.ask(REPLAY_PROMPT.format(
                request=failure["request"][:3000], answer=(failure.get("answer") or "(no answer)")[:2000],
                kind=failure["kind"], summary=failure["summary"], detail=_detail(failure), types=CHECKER_TYPES)), "{")
            if not isinstance(got, dict) or not isinstance(got.get("checker"), dict):
                self.dropped.append({"request": failure["request"][:200], "why": "no checker in the reply"})
                continue
            ok, why = checks.discriminates(got["checker"], str(got.get("good_answer", "")),
                                           failure.get("answer") or "", failure["request"], self.judge)
            if ok:
                return Case("replay", failure["request"], got["checker"], "dev", "replay", failure.get("persona") or None,
                            str(got.get("good_answer", "")), failure.get("answer") or "")
            self.dropped.append({"request": failure["request"][:200], "why": why})
        return None

    def variants(self, failure: dict, persona: str | None = None) -> list[Case]:
        kept: list[Case] = []
        seen = {_norm(failure["request"])}
        for _ in range(self.rounds):
            if len(kept) >= self.n:
                break
            avoid = ""
            if kept:
                avoid = "- Already written (write different ones):\n" + "\n".join(f"  - {c.request[:120]}" for c in kept)
            if self.dropped:
                avoid += "\n- These were dropped because their checker didn't hold up - avoid the same mistake:\n" + \
                    "\n".join(f"  - {d['request'][:80]}: {d['why'][:120]}" for d in self.dropped[-5:])
            got = parse_json(self.ask(VARIANTS_PROMPT.format(
                n=self.n - len(kept) + 2, request=failure["request"][:3000],
                answer=(failure.get("answer") or "(no answer)")[:1500], kind=failure["kind"],
                source=failure.get("source", ""), summary=failure["summary"], detail=_detail(failure),
                log=(failure.get("work_log") or "(no code was run)")[-2500:], avoid=avoid, types=CHECKER_TYPES)))
            for item in got if isinstance(got, list) else []:
                if not isinstance(item, dict) or not str(item.get("request", "")).strip():
                    continue
                req = str(item["request"]).strip()
                if _norm(req) in seen:
                    self.dropped.append({"request": req[:200], "why": "a duplicate"})
                    continue
                ok, why = checks.discriminates(item.get("checker"), str(item.get("good_answer", "")),
                                               str(item.get("bad_answer", "")), req, self.judge)
                if not ok:
                    self.dropped.append({"request": req[:200], "why": why})
                    continue
                seen.add(_norm(req))
                kept.append(Case(f"v{len(kept) + 1}", req, item["checker"], "dev", "variant", persona,
                                 str(item.get("good_answer", "")), str(item.get("bad_answer", ""))))
                if len(kept) >= self.n:
                    break
        # held out at random (seeded: the same attempt splits the same way when it resumes)
        order = list(range(len(kept)))
        random.Random(self.seed).shuffle(order)
        for i in order[:round(len(kept) * self.holdout)]:
            kept[i].split = "holdout"
        return kept


def _detail(failure: dict) -> str:
    d = (failure.get("detail") or "").strip()
    return f"Details: {d[:1500]}" if d else ""


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))[:160]
