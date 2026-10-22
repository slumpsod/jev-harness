"""Light Jev: a small System One model (Laya, `jev.backend: laya`, openjev/laya_bridge.py, which runs on a CPU) makes the
decisions it was measured to make well, and the rest are handed off (config `jev.light`, the Settings tab):

- worker: the task's own model answers the same typed questions as one quick side question (`ask`), so the checks stay
  on; it costs a model call each (seconds on a local model, tokens on a hosted one);
- off: the decision is skipped, the way it is when Jev is down (each caller has its fallback: no failure check, file
  and test checks only, review every change, ...).

Which decisions go where was measured (openjev/README.md "Smaller and faster", evals/bench/system_one_swap.py): Laya
agrees with semif on 90-98% of skill, tool, memory, tier, collaborator, device, Newsroom and session decisions and on
40-80% of the reasoning-heavy checks below. UNVERIFIED in daily use.
"""
from __future__ import annotations

import inspect
import json
import re
import threading
import time
from typing import Any, Callable, Mapping

from .decide import Answer, Response, SystemOneError
from .llm import LLMError


class Skipped(SystemOneError):
    """A decision light Jev does not ask (checks: off, or no model to ask). Callers fall back as when Jev is down, but it
    is not an outage: the harness does not report it as one."""

# the decisions Jev hands off in light mode, by default (config jev.light.worker / jev.light.off replace these lists)
WORKER = ("follow_up", "verify", "failure_check", "progress", "risk", "severity", "learn_fact", "learn_same_fact",
          "learn_skill", "same_specialist")
# in the middle of a worker's own reply or one call per old step: skipped even when the worker answers the others
OFF = ("compact", "thinking_watch")
MAX_TOKENS = 1500   # (as the quick reply: a thinking model spends part of it thinking; jev.light.extra can turn that off)

PROMPT = """You are the decision step of an AI-assistant harness. Read the `state` and answer every question about it.
Reply with one JSON object and nothing else: {{"<question key>": {{"answer": ..., "p": ...}}, ...}}
- yes/no: "answer" is "yes" or "no"; "p" is the probability that the answer is yes (0 to 1).
- choice: "answer" is exactly one option name; "p" is how sure you are of it (0 to 1).
- score: "answer" is the level number (0 is the first level); "p" is how sure you are (0 to 1).

state:
{state}

questions:
{questions}"""


def settings(jev_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """{mode: full|light, checks: worker|off, worker: steps, off: steps} from config `jev`."""
    j = dict(jev_cfg or {})
    light = j.get("light") if isinstance(j.get("light"), dict) else {}
    mode = str(j.get("mode") or "full").lower()
    checks = str(light.get("checks") or "worker").lower()
    return {"mode": mode if mode in ("full", "light") else "full",
            "checks": checks if checks in ("worker", "off") else "worker",
            "backend": str(light.get("backend") or "laya"),
            "worker": tuple(light.get("worker") or WORKER), "off": tuple(light.get("off") or OFF),
            "max_tokens": int(light.get("max_tokens") or MAX_TOKENS), "extra": dict(light.get("extra") or {})}


def route(step: str, s: dict[str, Any]) -> str:
    """Where a decision goes: "jev", "worker" or "off"."""
    if s["mode"] != "light":
        return "jev"
    if step in s["off"]:
        return "off"
    if step in s["worker"]:
        return "worker" if s["checks"] == "worker" else "off"
    return "jev"


def _question(key: str, q: Mapping[str, Any]) -> str:
    ins = q.get("instructions")
    if isinstance(ins, dict):
        text = str(ins.get("question", ""))
        data = "".join(f"\n    {k}: {v}" for k, v in ins.items() if k != "question")
    else:
        text, data = str(ins), ""
    kind = str(q.get("type", "")).lower()
    line = f"- {key} ({'yes/no' if kind == 'noul' else kind}): {text}{data}"
    crit = q.get("criteria")
    if kind == "choice" and isinstance(crit, dict):
        line += "".join(f"\n    option {name}" + (f": {desc}" if desc else "") for name, desc in crit.items())
    elif kind == "score" and isinstance(crit, list):
        line += "".join(f"\n    level {i}: {lv}" for i, lv in enumerate(crit))
    return line


def prompt(state: Any, questions: Mapping[str, Mapping[str, Any]]) -> str:
    st = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
    return PROMPT.format(state=st, questions="\n".join(_question(k, q) for k, q in questions.items()))


def _p(v: Any, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(v)))
    except (TypeError, ValueError):
        return default


def parse(text: str, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """The model's reply as System One answers (the shapes decide.Answer reads). Raises SystemOneError when a question
    has no usable answer: the caller then falls back as it does when Jev is down."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, flags=re.S)
    try:
        got = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        got = None
    if not isinstance(got, dict):
        raise SystemOneError(f"the worker's answer to Jev's questions was not JSON: {text[:200]!r}")
    out: dict[str, dict[str, Any]] = {}
    for key, q in questions.items():
        a = got.get(key)
        if not isinstance(a, dict):
            raise SystemOneError(f"the worker did not answer {key!r}")
        kind, ans = str(q.get("type", "")).lower(), a.get("answer")
        if kind == "noul":
            yes = str(ans).strip().lower() in ("yes", "true", "1")
            p = _p(a.get("p"), 0.9 if yes else 0.1)
            if (p >= 0.5) != yes:   # "no" with p 0.9: the model gave its confidence in its answer, not P(yes) (live: a
                p = 1 - p           # flipped "accurate" bounced an honest "that file doesn't exist" answer twice)
            out[key] = {"type": "noul", "noul": round(p, 4), "probabilities": {"yes": round(p, 4), "no": round(1 - p, 4)},
                        "confidence": round(max(p, 1 - p), 4)}
            continue
        names = list(q["criteria"]) if kind == "choice" else [str(i) for i in range(len(q.get("criteria") or []))]
        pick = str(ans).strip()
        if pick not in names:
            low = {n.lower(): n for n in names}
            pick = low.get(pick.lower(), "")
        if not pick:
            raise SystemOneError(f"the worker's answer to {key!r} is not one of the options: {ans!r}")
        n = len(names)
        p = max(_p(a.get("p"), 0.8), 1.0 / n)
        probs = {o: round(p if o == pick else (1 - p) / (n - 1), 4) for o in names}
        if kind == "choice":
            out[key] = {"type": "choice", "choice": pick, "probabilities": probs, "confidence": round(p, 4)}
        else:
            out[key] = {"type": "score", "score": round(sum(int(o) * v for o, v in probs.items()), 4),
                        "probabilities": probs, "confidence": round(p, 4)}
    return out


LOCAL = re.compile(r"https?://(127\.0\.0\.1|localhost|\[::1\])[:/]")


def _no_thinking(llm) -> dict[str, Any]:
    """Thinking off for this request on a local OpenAI-style server (llama.cpp, LM Studio, vLLM read the chat template's
    enable_thinking): live, a 27B model spent ~28 s thinking before each one-line answer. A hosted API gets nothing
    extra (it may refuse an unknown field); jev.light.extra sets it for any server."""
    return {"chat_template_kwargs": {"enable_thinking": False}} if LOCAL.match(str(getattr(llm, "base_url", ""))) else {}


def ask(llm, state: Any, questions: Mapping[str, Mapping[str, Any]], s: dict[str, Any],
        cancel: threading.Event | None = None) -> Response:
    """Jev's questions answered by `llm` (the task's own model) in one short call, as a decide.Response."""
    started = time.perf_counter()
    kwargs: dict[str, Any] = {}
    try:
        params = inspect.signature(llm.chat).parameters
    except (TypeError, ValueError):
        params = {}
    takes_any = any(p.kind == p.VAR_KEYWORD for p in params.values())   # (usage.Metered passes extras on to the model)
    if "body_extra" in params or takes_any:   # a quick side question: a small budget, and no thinking
        kwargs["body_extra"] = {"max_tokens": s["max_tokens"], **_no_thinking(llm), **s["extra"]}
    if ("cancel" in params or takes_any) and cancel is not None:
        kwargs["cancel"] = cancel
    try:
        out = llm.chat([{"role": "user", "content": prompt(state, questions)}], **kwargs)
    except (LLMError, OSError) as e:   # the model down or failing: the caller falls back as when Jev is down (a Stop,
        raise SystemOneError(f"the task's model could not answer Jev's questions: {e}") from e   # Cancelled, passes)
    raw = parse(getattr(out, "text", "") or "", questions)
    name = str(getattr(llm, "model", "") or "worker")
    return Response(model=f"worker:{name}", answers={k: Answer(id=k, kind=v["type"], raw=v) for k, v in raw.items()},
                    latency_s=time.perf_counter() - started)


LLMSource = Callable[[], Any]   # () -> the model that answers handed-off decisions now (the running task's), or None
