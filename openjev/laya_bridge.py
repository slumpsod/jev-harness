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
    LAYA_ONNX=model.int8.onnx python laya_bridge.py   # on the CPU through ONNX Runtime (faster, less memory)

Settings (environment): LAYA_CHECKPOINT (a checkpoint folder; default openjev/models/laya-harness), LAYA_DEVICE (auto,
cuda or cpu), LAYA_ONNX (an ONNX export of the checkpoint, a path or a file name in the checkpoint folder; unset: PyTorch),
LAYA_THREADS (CPU threads; default 8), LAYA_BRIDGE_HOST (127.0.0.1), LAYA_BRIDGE_PORT (33004).
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
ONNX = os.environ.get("LAYA_ONNX", "")
# ModernBERT's local-attention layers see 64 tokens either side. In a batch padded by more than that, a padding position
# sees only padding, its attention row is all -inf, and the exported graph turns it into NaN that spreads into the real
# tokens (laya 0.4.13's ONNXAgent pads every question of a request to the longest: measured, 135 NaN answers in 300
# held-out harness requests). PyTorch's own attention kernel does not have this problem. laya_onnx.py fixes it in the
# graph; a file without that fix runs in groups of rows no more than PAD_GAP tokens apart (Grouped).
PAD_GAP = 48

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


class Grouped:
    """An ONNX Runtime session that runs a batch in groups of rows of similar length (no row padded by more than
    PAD_GAP tokens) and returns the outputs in the batch's row order, as one run would."""

    def __init__(self, session):
        self.session = session

    def run(self, names, inputs):
        import numpy as np

        lens = inputs["attention_mask"].sum(axis=1)
        groups, cur = [], []
        for i in np.argsort(-lens, kind="stable"):   # longest first: a group's first row is its longest
            if cur and lens[cur[0]] - lens[i] > PAD_GAP:
                groups.append(cur)
                cur = []
            cur.append(int(i))
        groups.append(cur)
        if len(groups) == 1:
            return self.session.run(names, inputs)
        outs = None
        for g in groups:
            n = int(lens[g[0]])
            part = {k: v[g][:, :n] if k in ("input_ids", "attention_mask") else v[g] for k, v in inputs.items()}
            got = self.session.run(names, part)
            if outs is None:
                outs = [np.empty((len(lens),) + r.shape[1:], dtype=r.dtype) for r in got]
            for o, r in zip(outs, got):
                o[g] = r
        return outs

    def __getattr__(self, name):
        return getattr(self.session, name)


def _onnx_path() -> str:
    p = Path(ONNX)
    return str(p if p.is_absolute() or p.exists() else Path(CHECKPOINT) / p)


def _load() -> None:
    global _agent
    import torch
    import laya

    threads = int(os.environ.get("LAYA_THREADS", "8"))
    t0 = time.time()
    if ONNX:
        import onnxruntime as ort
        from laya.onnx_agent import ONNXAgent

        path = _onnx_path()
        print(f"loading {CHECKPOINT} with {path} (ONNX Runtime, CPU) ...", flush=True)
        agent = ONNXAgent(CHECKPOINT, onnx_path=path)
        so = ort.SessionOptions()   # rebuilt with our thread count (ONNXAgent takes none)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = threads
        so.enable_cpu_mem_arena = os.environ.get("LAYA_ONNX_ARENA", "1") == "1"
        agent.session = None
        session = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        if "laya_nan_fix" not in session.get_modelmeta().custom_metadata_map:
            print("this ONNX file is not fixed for batched runs (python laya_onnx.py FILE): running each request in "
                  "groups of similar length instead, about twice as slow", flush=True)
            session = Grouped(session)
        agent.session = session
    else:
        device = DEVICE
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            torch.set_num_threads(threads)
        print(f"loading {CHECKPOINT} on {device} ...", flush=True)
        agent = laya.load(CHECKPOINT, device=device)
    _agent = agent   # only now: /health says "loaded" when _agent is set
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
            "model": f"laya:{Path(CHECKPOINT).name}", "checkpoint": CHECKPOINT,
            "runtime": f"onnx:{Path(ONNX).name}" if ONNX else "pytorch", "stats": STATS}


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
