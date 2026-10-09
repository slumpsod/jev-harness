"""Thin client for the local System One adapter (openjev/systemone.py, POST /v1/systemone).

Shapes follow the TypeSafe wire format so the same questions run against laya, openjev,
semif, or the hosted API. Question builders and the Answer/Response shapes are adapted
from typesafe-skill-router (MIT, Ritesh Patel), trimmed to what the harness needs:
an optional API key for the hosted API, no disk cache, stdlib only.
"""
from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .errors import Cancelled

Question = dict
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
TOTALS = {"calls": 0, "seconds": 0.0, "tokens": 0}   # every ask() in this process
_TOTALS_LOCK = threading.Lock()   # (a browser tool's thread and the REPL's main thread both ask)


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
    """Jev could not give a usable answer: unreachable, timed out, or replied with something malformed. The one
    exception the client raises for any of these, so the harness's "Jev down: degrade, never crash" handlers
    (Harness._safe and friends, which catch only this) see every one of them."""


@dataclass
class Answer:
    id: str
    kind: str
    raw: dict[str, Any] = field(default_factory=dict)

    def _field(self, key: str, cast):
        try:
            return cast(self.raw[key])
        except (KeyError, TypeError, ValueError):
            raise SystemOneError(f"Jev's answer to {self.id!r} has no usable {key!r}: {str(self.raw)[:200]}") from None

    @property
    def noul(self) -> float:
        return self._field("noul", float)

    @property
    def choice(self) -> str:
        return self._field("choice", str)

    @property
    def score(self) -> float:
        return self._field("score", float)

    @property
    def probabilities(self) -> dict[str, float]:
        try:
            return {k: float(v) for k, v in (self.raw.get("probabilities") or {}).items()}
        except (AttributeError, TypeError, ValueError):
            raise SystemOneError(f"Jev's answer to {self.id!r} has unusable probabilities: "
                                 f"{str(self.raw)[:200]}") from None

    @property
    def confidence(self) -> float:
        try:
            return float(self.raw.get("confidence") or 0.0)
        except (TypeError, ValueError):
            return 0.0


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

    def ask(self, state: Any, questions: Mapping[str, Question], cancel: threading.Event | None = None) -> Response:
        """`cancel`: a set event ends the wait at once with Cancelled (the user pressed Stop). The request itself
        finishes on its own thread and its answer is dropped: a blocked read cannot be woken from another thread on
        Windows (see llm.py), and before this a Stop waited out every Jev call in flight, up to the full timeout."""
        payload: dict[str, Any] = {"state": state, "questions": dict(questions)}
        if self.backend:
            payload["model"] = self.backend
        started = time.perf_counter()
        raw = self._post(payload) if cancel is None else self._post_until(payload, cancel)
        self.calls += 1
        listed = raw.get("answers") or {}
        if not isinstance(listed, dict) or not all(isinstance(v, dict) for v in listed.values()):
            raise SystemOneError(f"adapter sent answers in an unknown shape: {str(listed)[:200]}")
        answers = {k: Answer(id=k, kind=str(v.get("type", "")), raw=dict(v)) for k, v in listed.items()}
        missing = set(questions) - set(answers)
        if missing:
            raise SystemOneError(f"adapter omitted answers for: {sorted(missing)}")
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        try:
            tokens = int(usage.get("input_tokens") or 0)
        except (TypeError, ValueError):
            tokens = 0
        resp = Response(model=str(raw.get("model", self.backend)), answers=answers,
                        latency_s=time.perf_counter() - started, input_tokens=tokens)
        with _TOTALS_LOCK:   # read by the worker REPL to report its tools' Jev use
            TOTALS["calls"] += 1
            TOTALS["seconds"] += resp.latency_s
            TOTALS["tokens"] += resp.input_tokens
        return resp

    def health(self) -> dict[str, Any]:
        with urllib.request.urlopen(f"{self.base_url}/health", timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))

    def _post_until(self, payload: dict[str, Any], cancel: threading.Event) -> dict[str, Any]:
        if cancel.is_set():
            raise Cancelled("stopped by the user")
        box: dict[str, Any] = {}
        done = threading.Event()

        def run() -> None:
            try:
                box["raw"] = self._post(payload)
            except BaseException as e:  # noqa: BLE001 - handed to the caller
                box["error"] = e
            finally:
                done.set()
        threading.Thread(target=run, name="jev-ask", daemon=True).start()
        while not done.wait(0.1):
            if cancel.is_set():
                raise Cancelled("stopped by the user while Jev was answering")
        if "error" in box:
            raise box["error"]
        return box["raw"]

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        # ("replace": a lone surrogate in a state - text a page or a worker produced - cannot be encoded as UTF-8)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8", "replace")
        attempt = 0
        while True:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            req = urllib.request.Request(f"{self.base_url}/v1/systemone", data=body, method="POST",
                                         headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    text = r.read()
            except urllib.error.HTTPError as e:
                try:
                    detail = e.read().decode("utf-8", "replace")
                except (OSError, http.client.HTTPException):
                    detail = ""
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
            except TimeoutError as e:
                # urllib wraps only what fails while the request is sent: a timeout waiting for the answer came out raw
                # and, not being a SystemOneError, crashed the turn instead of degrading it. Not retried: an adapter
                # that took the whole timeout is unlikely to answer the next try sooner.
                raise SystemOneError(f"systemone at {self.base_url} did not answer within {self.timeout:.0f}s") from e
            except (ConnectionError, http.client.HTTPException, OSError) as e:
                # the adapter dropped the connection (restarted mid-request: RemoteDisconnected), or cut its answer
                # short (IncompleteRead): also raw before, also crashed the turn
                if attempt < self.max_retries:
                    attempt += 1
                    time.sleep(2.0 ** attempt)
                    continue
                raise SystemOneError(f"systemone at {self.base_url} failed: {type(e).__name__}: {e}") from e
            try:
                data = json.loads(text.decode("utf-8"))
            except ValueError as e:   # (UnicodeDecodeError is a ValueError too)
                raise SystemOneError(f"systemone sent a reply that is not JSON: {text[:200]!r}") from e
            if not isinstance(data, dict):
                raise SystemOneError(f"systemone sent {type(data).__name__}, not an object: {str(data)[:200]}")
            return data
