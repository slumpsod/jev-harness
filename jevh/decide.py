"""Thin client for the local System One adapter (openjev/systemone.py, POST /v1/systemone).

Shapes follow the TypeSafe wire format so the same questions run against laya, openjev,
semif, or the hosted API. Question builders and the Answer/Response shapes are adapted
from typesafe-skill-router (MIT, Ritesh Patel), trimmed to what the harness needs:
an optional API key for the hosted API, no disk cache, stdlib only.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

Question = dict
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
TOTALS = {"calls": 0, "seconds": 0.0, "tokens": 0}   # every ask() in this process


def noul(instructions) -> Question:
    """Yes/no question; the answer is P(yes)."""
    return {"type": "noul", "instructions": instructions}


def choice(instructions, criteria: Mapping[str, str | None]) -> Question:
    """Pick one option; the answer carries the full distribution."""
    if not criteria:
        raise ValueError("choice() needs at least one option")
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def score(instructions, criteria: Sequence[str]) -> Question:
    """Rate along ordered levels (2-10)."""
    levels = list(criteria)
    if len(levels) < 2:
        raise ValueError("score() needs at least two levels")
    return {"type": "score", "instructions": instructions, "criteria": levels}


class SystemOneError(RuntimeError):
    pass


@dataclass
class Answer:
    id: str
    kind: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def noul(self) -> float:
        return float(self.raw["noul"])

    @property
    def choice(self) -> str:
        return str(self.raw["choice"])

    @property
    def score(self) -> float:
        return float(self.raw["score"])

    @property
    def probabilities(self) -> dict[str, float]:
        return {k: float(v) for k, v in (self.raw.get("probabilities") or {}).items()}

    @property
    def confidence(self) -> float:
        return float(self.raw.get("confidence") or 0.0)


@dataclass
class Response:
    model: str
    answers: dict[str, Answer]
    latency_s: float = 0.0
    input_tokens: int = 0      # tokens the backend processed (semif bridge reports it)

    def __getitem__(self, key: str) -> Answer:
        return self.answers[key]

    def as_dict(self) -> dict[str, Any]:
        return {"model": self.model, "latency_s": round(self.latency_s, 3),
                "answers": {k: a.raw for k, a in self.answers.items()}}


class SystemOneClient:
    """One endpoint, typed answers. `backend` is the adapter's `model` selector."""

    def __init__(self, base_url: str = "http://127.0.0.1:33001", backend: str | None = "laya",
                 timeout: float = 300.0, max_retries: int = 2, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key  # only for the hosted API; the local adapter needs none
        self.backend = backend
        self.timeout = timeout
        self.max_retries = max_retries
        self.calls = 0

    def ask(self, state: Any, questions: Mapping[str, Question]) -> Response:
        payload: dict[str, Any] = {"state": state, "questions": dict(questions)}
        if self.backend:
            payload["model"] = self.backend
        started = time.perf_counter()
        raw = self._post(payload)
        self.calls += 1
        answers = {k: Answer(id=k, kind=str(v.get("type", "")), raw=dict(v))
                   for k, v in (raw.get("answers") or {}).items()}
        missing = set(questions) - set(answers)
        if missing:
            raise SystemOneError(f"adapter omitted answers for: {sorted(missing)}")
        resp = Response(model=str(raw.get("model", self.backend)), answers=answers,
                        latency_s=time.perf_counter() - started,
                        input_tokens=int((raw.get("usage") or {}).get("input_tokens") or 0))
        TOTALS["calls"] += 1   # read by the worker REPL to report its tools' Jev use
        TOTALS["seconds"] += resp.latency_s
        TOTALS["tokens"] += resp.input_tokens
        return resp

    def health(self) -> dict[str, Any]:
        with urllib.request.urlopen(f"{self.base_url}/health", timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        attempt = 0
        while True:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            req = urllib.request.Request(f"{self.base_url}/v1/systemone", data=body, method="POST",
                                         headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")
                if e.code in RETRY_STATUSES and attempt < self.max_retries:
                    attempt += 1
                    time.sleep(2.0 ** attempt)
                    continue
                raise SystemOneError(f"systemone {e.code}: {detail[:400]}") from e
            except urllib.error.URLError as e:
                if attempt < self.max_retries:
                    attempt += 1
                    time.sleep(2.0 ** attempt)
                    continue
                raise SystemOneError(f"systemone unreachable at {self.base_url}: {e.reason}") from e
