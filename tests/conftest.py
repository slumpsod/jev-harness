from __future__ import annotations

import re
from typing import Any, Callable

import pytest

from jevh.config import ROOT, load_config
from jevh.decide import Answer, Response
from jevh.llm import LLMResult


class FakeSystemOne:
    """Answers every question via `rules(key, question, state) -> raw answer | None`,
    else a neutral default (noul 0.1, choice = first option, score 0)."""

    def __init__(self, rules: Callable[[str, dict, Any], dict | None] | None = None):
        self.rules = rules or (lambda k, q, s: None)
        self.calls: list[tuple[Any, dict]] = []

    def ask(self, state, questions):
        self.calls.append((state, questions))
        answers = {}
        for key, q in questions.items():
            raw = self.rules(key, q, state) or self._default(q)
            raw.setdefault("type", q["type"])
            answers[key] = Answer(key, q["type"], raw)
        return Response("fake", answers)

    def health(self):
        return {"adapter": "fake"}

    @staticmethod
    def _default(q):
        if q["type"] == "noul":
            return {"noul": 0.1, "confidence": 0.9}
        if q["type"] == "choice":
            first = next(iter(q["criteria"]))
            return {"choice": first, "probabilities": {k: (1.0 if k == first else 0.0) for k in q["criteria"]},
                    "confidence": 1.0}
        return {"score": 0.0, "confidence": 1.0}


def noul_ans(p: float) -> dict:
    return {"type": "noul", "noul": p, "confidence": max(p, 1 - p)}


def choice_ans(pick: str, conf: float, options) -> dict:
    rest = (1 - conf) / max(len(options) - 1, 1)
    return {"type": "choice", "choice": pick, "confidence": conf,
            "probabilities": {o: (conf if o == pick else rest) for o in options}}


class ScriptedLLM:
    """Returns canned replies in order; records every prompt it saw."""

    def __init__(self, replies: list[str], name: str = "scripted"):
        self.replies = list(replies)
        self.seen: list[list[dict]] = []
        self.name = name

    def chat(self, messages):
        self.seen.append([dict(m) for m in messages])
        text = self.replies.pop(0) if self.replies else "done"
        return LLMResult(text=text, input_tokens=10, output_tokens=5, model=self.name)


def py(code: str) -> str:
    return f"```python\n{code}\n```"


@pytest.fixture
def cfg(tmp_path):
    c = load_config(ROOT / "config.yaml")
    c.workspace = tmp_path / "ws"
    c.skills_dirs = [ROOT / "skills"]  # deterministic: local skills only
    c.limits["repl_timeout"] = 20
    # the final() gate asks Jev about every answer; tests about other things use scripted workers
    # and a fake Jev, so it is off here and switched on in tests/test_verify.py
    c.raw = {**c.raw, "verify": {**(c.raw.get("verify") or {}), "enabled": False}}
    return c
