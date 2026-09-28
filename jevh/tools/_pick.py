"""Choice helpers shared by the tool modules (browser, web).

Local semif answers a Choice over at most 16 options (the hosted API takes 255), so bigger
sets run as balanced rounds whose leaders meet in a final round. All the round questions go
in one request, so a 40-option pick costs two calls, not four.
"""
from __future__ import annotations

import math
import os
import re
from typing import Mapping

from jevh.decide import SystemOneClient, choice, noul

CLOUD_URL = "https://api.typesafe.ai"


def client_from_env() -> tuple[SystemOneClient, int]:
    """(client, max options per Choice) for the Jev the tools should use.

    JEVH_TOOLS_JEV=cloud uses the hosted API with TYPESAFE_API_KEY; anything else uses the
    local adapter at JEVH_JEV_URL with JEVH_JEV_BACKEND (the harness sets both)."""
    if os.environ.get("JEVH_TOOLS_JEV", "local").lower() == "cloud":
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise RuntimeError("JEVH_TOOLS_JEV=cloud needs TYPESAFE_API_KEY")
        return SystemOneClient(CLOUD_URL, os.environ.get("JEVH_CLOUD_MODEL", "jev-latest"),
                               timeout=60, api_key=key), 240
    return SystemOneClient(os.environ.get("JEVH_JEV_URL", "http://127.0.0.1:33001"),
                           os.environ.get("JEVH_JEV_BACKEND", "semif"), timeout=120), 16


def split(keys: list[str], cap: int) -> list[list[str]]:
    """Balanced chunks of at most `cap` (never a lone 1-option chunk)."""
    n = max(1, math.ceil(len(keys) / cap))
    size = math.ceil(len(keys) / n)
    return [keys[i:i + size] for i in range(0, len(keys), size)]


def choose(client: SystemOneClient, state, instructions: str, options: Mapping[str, str],
           cap: int, extra: Mapping | None = None, keep: int = 3) -> tuple[dict[str, float], dict]:
    """Distribution over `options` (key -> description), in rounds when it exceeds `cap`.

    `extra` questions ride along in the first request; their answers come back as the
    second value. Probabilities from the final round are relative to its finalists; keys
    that lost earlier rounds get 0."""
    keys = list(options)
    extra = dict(extra or {})
    if len(keys) == 1:
        answers = client.ask(state, extra).answers if extra else {}
        return {keys[0]: 1.0}, answers
    if len(keys) <= cap:
        res = client.ask(state, {**extra, "_pick": choice(instructions, dict(options))})
        return res["_pick"].probabilities, res.answers
    rounds = split(keys, cap)
    qs = {f"_r{i}": choice(instructions, {k: options[k] for k in chunk}) for i, chunk in enumerate(rounds)}
    res = client.ask(state, {**extra, **qs})
    finalists: list[str] = []
    for i in range(len(rounds)):
        probs = res[f"_r{i}"].probabilities
        finalists += sorted(probs, key=probs.get, reverse=True)[:keep]
    final = choose(client, state, instructions, {k: options[k] for k in finalists}, cap)[0]
    return {k: final.get(k, 0.0) for k in keys}, res.answers


def chunk_text(text: str, size: int = 700) -> list[str]:
    """Split on blank lines / lines into ~size-char passages."""
    parts, cur = [], ""
    for line in re.split(r"\n+", text):
        line = line.strip()
        if not line:
            continue
        if cur and len(cur) + len(line) > size:
            parts.append(cur)
            cur = ""
        cur = f"{cur}\n{line}" if cur else line
        while len(cur) > size * 1.5:
            parts.append(cur[:size])
            cur = cur[size:]
    if cur:
        parts.append(cur)
    return parts


MAX_PASSAGES = 48   # per page: at most 3 Choice rounds + a final (local semif), whatever the page size


def prefilter(question: str, passages: list[str], n: int = MAX_PASSAGES) -> list[str]:
    """Keep the n passages sharing the most words with the question, in page order. Cheap and
    local: Jev's cost has to stay bounded, and semif's memory grows with every long input
    (a 359k-char page once meant 608 passages and a VRAM spill on the 4080)."""
    if len(passages) <= n:
        return passages
    words = {w for w in re.findall(r"[a-z0-9]{3,}", question.lower())}
    scored = sorted(range(len(passages)), key=lambda i: -sum(w in passages[i].lower() for w in words))
    return [passages[i] for i in sorted(scored[:n])]


def pick_passages(client: SystemOneClient, cap: int, question: str, text: str, k: int = 3,
                  keep_at: float = 0.3, size: int = 700) -> list[str]:
    """The passages of `text` that help answer `question`: a Choice ranks them, then a yes/no
    check on the top k keeps the ones that actually help (same two-stage design as skill and
    memory picking). Long pages are keyword-prefiltered to MAX_PASSAGES first."""
    passages = prefilter(question, chunk_text(text[:200_000], size))
    if len(passages) <= 1:
        return passages
    opts = {f"p{i}": p[:400] for i, p in enumerate(passages)}
    probs, _ = choose(client, {"question": question}, "Which passage best answers `question`?", opts, cap)
    top = sorted(probs, key=probs.get, reverse=True)[:k]
    state = {"question": question, "passages": {key: passages[int(key[1:])] for key in top}}
    res = client.ask(state, {key: noul(f"Does `passages.{key}` contain information that helps answer "
                                       f"`question`?") for key in top})
    kept = [int(key[1:]) for key in top if res[key].noul >= keep_at] or [int(top[0][1:])]
    return [passages[i] for i in sorted(kept)]
