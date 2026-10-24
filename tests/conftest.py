from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any, Callable

import pytest

# the tests run on config.yaml alone: a machine's config.local.yaml (its models, a hosted Jev) stays out
os.environ["JEVH_NO_LOCAL_CONFIG"] = "1"
# ...and offline: a real Nango key must not reach a server (without config.local.yaml, the cloud) from a test
for _var in ("NANGO_SECRET_KEY", "JEVH_NANGO_URL", "JEVH_NANGO_WRITES", "JEVH_HOME"):
    os.environ.pop(_var, None)
# ...and the key/URL nango.start() saves go to a folder of their own, never the real workspace's
os.environ["JEVH_NANGO_DIR"] = tempfile.mkdtemp(prefix="jevh-nango-test-")

from jevh.config import ROOT, load_config
from jevh.decide import Answer, Response
from jevh.llm import LLMResult


def git(cwd, *args: str) -> str:
    """`git *args` in `cwd` for a test, its output decoded as UTF-8. A machine without git skips the test with a reason
    instead of erroring with FileNotFoundError (issue #7: ~36 errors on a Windows machine without git on PATH)."""
    exe = shutil.which("git")
    if exe is None:
        pytest.skip("git is not installed (or not on PATH): this test needs a real repository")
    return subprocess.run([exe, *args], cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8",
                          errors="replace").stdout


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
        if isinstance(text, LLMResult):   # a scripted reply that also carries reasoning
            return text
        return LLMResult(text=text, input_tokens=10, output_tokens=5, model=self.name)


def py(code: str) -> str:
    return f"```python\n{code}\n```"


@pytest.fixture
def cfg(tmp_path):
    c = load_config(ROOT / "config.yaml")
    c.workspace = tmp_path / "ws"
    c.personas_dir = tmp_path / "personas"   # a copy: persona creation writes personas/learned/ here, not in the repo
    c.personas_dirs = [c.personas_dir]
    # (not personas/learned/: personas the harness drafted on this machine are runtime output, not part of the tests)
    shutil.copytree(ROOT / "personas", c.personas_dir, ignore=shutil.ignore_patterns("learned"))
    c.skills_dirs = [ROOT / "skills"]  # deterministic: local skills only
    c.limits["repl_timeout"] = 20
    # the final() gate asks Jev about every answer; tests about other things use scripted workers
    # and a fake Jev, so it is off here and switched on in tests/test_verify.py
    c.raw = {**c.raw, "verify": {**(c.raw.get("verify") or {}), "enabled": False}}
    # spot checks sample tasks at random and spend a model reply: off unless a test turns them on
    c.raw = {**c.raw, "failures": {**(c.raw.get("failures") or {}), "audit": 0}}
    # the Newsroom's tile summaries spend a model reply between turns (from the same scripted model): off unless a test
    # turns them on (tests/test_newsroom_summary.py)
    c.raw = {**c.raw, "newsroom": {**(c.raw.get("newsroom") or {}), "summary": False}}
    return c


@pytest.fixture
def nango_dir(tmp_path, monkeypatch):
    """A fresh folder for the key/URL nango.start() saves."""
    d = tmp_path / "nango-state"
    monkeypatch.setenv("JEVH_NANGO_DIR", str(d))
    return d


@pytest.fixture
def fake_nango(nango_dir, monkeypatch):
    """A fake Nango server on localhost (tests/fake_nango.py); yields it. Its `.seen` lists every request."""
    from fake_nango import serve

    from jevh.tools import nango
    srv = serve()
    monkeypatch.setenv("NANGO_SECRET_KEY", "sk-test")
    monkeypatch.setenv("JEVH_NANGO_URL", srv.url)
    monkeypatch.delenv("JEVH_NANGO_WRITES", raising=False)
    nango._cache.clear()
    nango._allowed.clear()
    yield srv
    srv.shutdown()
    nango._cache.clear()
