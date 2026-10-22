"""UNVERIFIED (2026-10-21): measured on the Jev evals, not yet used day to day; it misses cases semif catches on the
strict gates (openjev/README.md "Smaller and faster"). semif stays the default backend.

Laya bridge: the System One wire format (POST /v1/systemone) over a Laya checkpoint fine-tuned on the harness's own
decisions - a small, fast stand-in for the SemIf bridge (semif_bridge.py, Qwen3.5-4B).

Laya (convaiinnovations/laya, Apache-2.0) is a non-autoregressive decision model: a ModernBERT-large encoder (421M) with a
head that scores each option of a typed question in one forward pass. The base checkpoints answer the harness's questions
near chance; the harness checkpoint (`laya-harness`) is Laya trained to give semif's probabilities on harness-shaped
decisions (evals/bench/system_one_swap.py says how it was made and measured). About 2 GB of GPU memory (semif: ~9 GB),
or none on the CPU.

`systemone.py` forwards here when the request's model is `laya` (config.yaml `jev.backend: laya`).

Run:
    python laya_bridge.py                       # :33004, GPU if there is one (pick it with CUDA_VISIBLE_DEVICES)
    LAYA_DEVICE=cpu python laya_bridge.py       # on the CPU

Settings (environment): LAYA_CHECKPOINT (a checkpoint folder; default openjev/models/laya-harness), LAYA_DEVICE (auto,
cuda or cpu), LAYA_THREADS (CPU threads; default 8), LAYA_BRIDGE_HOST (127.0.0.1), LAYA_BRIDGE_PORT (33004).
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
CHECKPOINT = os.environ.get("LAYA_CHECKPOINT") or str(HERE / "models" / "laya-harness")
DEVICE = os.environ.get("LAYA_DEVICE", "auto")

_agent = None
_gpu = threading.Lock()   # one request at a time: a forward pass is ~30 ms on a GPU, batching buys little here
STATS = {"requests": 0, "questions": 0, "busy_seconds": 0.0, "errors": 0}


class Question(BaseModel):
    type: str
    instructions: object
    criteria: object | None = None


class SystemOneRequest(BaseModel):
    state: object
    questions: dict[str, Question]
    model: str | None = None


def state_text(state: object) -> str:
    """The state as the checkpoint was trained on it: text as is, anything else as indented JSON (as semif_bridge.py)."""
    return state.strip() if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)


def _load() -> None:
    global _agent
    import torch
    import laya

    device = DEVICE
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(int(os.environ.get("LAYA_THREADS", "8")))
    print(f"loading {CHECKPOINT} on {device} ...", flush=True)
    t0 = time.time()
    _agent = laya.load(CHECKPOINT, device=device)
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)


def _answer(kind: str, a: dict) -> dict:
    """Laya's answer in the shape the harness reads (decide.Answer): noul, choice + probabilities, score."""
    if kind == "noul":
        p = float(a["noul"])
        return {"type": "noul", "noul": round(p, 4), "probabilities": {"yes": round(p, 4), "no": round(1 - p, 4)},
                "confidence": round(max(p, 1 - p), 4)}
    probs = {str(k): round(float(v), 4) for k, v in (a.get("probabilities") or {}).items()}
    if kind == "choice":
        best = max(probs, key=probs.get)
        return {"type": "choice", "choice": best, "probabilities": probs, "confidence": probs[best]}
    out = {"type": "score", "score": round(float(a["score"]), 4), "probabilities": probs,
           "confidence": max(probs.values()) if probs else 0.0}
    if a.get("legend"):
        out["legend"] = a["legend"]
    return out


app = FastAPI(title="Laya bridge (System One wire format)")


@app.on_event("startup")
async def _startup() -> None:
    threading.Thread(target=_load, daemon=True).start()


@app.get("/health")
def health():
    # "loaded" as laya-serve reports it, for adapters that wait on it
    return {"status": "up" if _agent is not None else "loading", "loaded": _agent is not None,
            "model": f"laya:{Path(CHECKPOINT).name}", "checkpoint": CHECKPOINT, "stats": STATS}


def _work(state: str, questions: dict) -> dict:
    with _gpu:
        t0 = time.time()
        try:
            out = _agent.predict(state, questions)
        finally:
            STATS["busy_seconds"] += time.time() - t0
    return out


@app.post("/v1/systemone")
async def systemone(req: SystemOneRequest):
    if _agent is None:
        raise HTTPException(503, "loading")
    questions = {k: q.model_dump(exclude_none=True) for k, q in req.questions.items()}
    try:
        out = await asyncio.to_thread(_work, state_text(req.state), questions)
    except ValueError as e:   # a question Laya can't read (e.g. options beyond its limits)
        STATS["errors"] += 1
        raise HTTPException(422, str(e)[:300]) from None
    STATS["requests"] += 1
    STATS["questions"] += len(questions)
    answers = {k: _answer(questions[k]["type"].lower(), a) for k, a in out["answers"].items() if k in questions}
    usage = out.get("usage") or {}
    return {"model": f"laya:{Path(CHECKPOINT).name}", "answers": answers,
            "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": 0}}


def main():
    import uvicorn
    uvicorn.run(app, host=os.environ.get("LAYA_BRIDGE_HOST", "127.0.0.1"),
                port=int(os.environ.get("LAYA_BRIDGE_PORT", "33004")), log_level="warning")


if __name__ == "__main__":
    main()
