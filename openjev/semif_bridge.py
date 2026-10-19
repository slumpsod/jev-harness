"""SemIf bridge: the System One wire format (POST /v1/systemone) over SemIf's direct option readout.

Jev, the harness's decision model, runs locally as TheoLeeCJ/SemIf-OpenJev (MIT): Qwen3.5-4B reading typed option
probabilities straight from its next-token logits, with no generation. This server loads it once and answers System One
requests; `systemone.py` beside it is the adapter the harness talks to (config.yaml `jev.url`, port 33001), and it
forwards here (port 33003, `jev.backend_health`). See README.md in this folder.

Mapping (System One question -> SemIf row)
  noul    state = evidence, question = instructions, options {yes, no}  -> noul = P(yes)
  choice  options = the criteria, each shown as "<name>. <description>" (the way TypeSafe shows an option's key). The
          direct readout has 16 answer slots (A-P), so a wider choice runs as 16-way pieces whose probabilities are
          normalized into one distribution.
  score   options = the criteria levels  -> score = the expected level index

Serving
- One request on the GPU at a time; a request whose caller hung up is skipped, or stopped between forwards (a caller
  that timed out used to leave its queued work running for minutes).
- Out of memory answers 503 instead of crashing, and the cache PyTorch keeps after an unusually large request is
  released.
- Shared-state caching (SEMIF_PREFIX_CACHE=0 turns it off): every question of a request starts with the same state, so
  the state runs once and each question continues from a copy of its cache (~38% faster on a 2k-token state with 5
  questions; yes/no answers move by ~0.003, and jev-harness's routing, skill, memory, tool, follow-up, learning and
  browser evals gave identical results with it on).
- Inputs over 2048 tokens run in 2048-token pieces through the cache: on a 16 GB RTX 4080 the peak stayed ~8.7 GB at any
  length, where whole inputs spilled into system RAM from ~6k tokens. Shorter inputs take SemIf's own single forward.
- SemIf's check that the answer letters tokenize cleanly (16 extra tokenizations per question) runs once per option
  count: that CPU work had cost as much as the GPU forward.
- GET /health: loading or up, with request, token, cancellation and timing counters.

Run (see README.md for the install):
    python semif_bridge.py                      # :33003; pick the GPU with CUDA_VISIBLE_DEVICES

Settings (environment): SEMIF_HOST (127.0.0.1), SEMIF_PORT (33003), SEMIF_PREFIX_CACHE (1), SEMIF_MAX_TOKENS (12288),
SEMIF_SRC (a SemIf-OpenJev checkout's src/ folder, if semif_phase1 is not installed).
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
import threading
import time

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

if os.environ.get("SEMIF_SRC"):
    sys.path.insert(0, os.environ["SEMIF_SRC"])

MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"   # the revision SemIf-OpenJev's quickstart pins
# 12K was measured safe on a 16 GB RTX 4080 (eager attention ran out of memory at ~16-20K); requests stay under it
MAX_TOKENS = int(os.environ.get("SEMIF_MAX_TOKENS", "12288"))
CHUNK = 16              # answer slots: A..P
PREFIX_CACHE = os.environ.get("SEMIF_PREFIX_CACHE", "1") == "1"
MIN_PREFIX = 256        # shared tokens worth caching
# Longer inputs run in pieces of this many tokens through the cache. On the 4080 whole inputs peaked at 9.0 GB (2.6k
# tokens), 11.7 GB (4.9k), 15.5 GB (7.1k, spilling into system RAM: 25 s) and 19.7 GB (8.9k: 55 s); in 2048-token pieces
# the peak stayed at 8.7 GB at any length (7.3k tokens: 1.4 s) with the same readout. The growth is the linear-attention
# layers' PyTorch fallback kernel, not the full attention (which is already SDPA).
PREFILL_CHUNK = 2048
BIG_REQUEST = 6000      # tokens in one row: release cached GPU memory afterwards

_gpu = asyncio.Lock()   # one request on the GPU at a time
_model = None
_tokenizer = None
_slot_cache: dict[int, list[int]] = {}
_verified: set[int] = set()
STATS = {"requests": 0, "questions": 0, "forwards": 0, "tokens": 0, "cancelled": 0, "oom": 0,
         "cache_hits": 0, "busy_seconds": 0.0, "max_row_tokens": 0}


def _semif():
    """SemIf's own prompt and loader (imported when first needed, so this module loads without torch for its tests)."""
    from semif_phase1 import core, direct
    return core, direct


def _load() -> None:
    global _model, _tokenizer
    print(f"loading {MODEL_ID}@{REVISION} ...", flush=True)
    t0 = time.time()
    core, _ = _semif()
    _model, _tokenizer, _ = core.load_causal_model(MODEL_ID, REVISION, device="auto", dtype="bfloat16")
    print(f"loaded in {time.time() - t0:.1f}s on {next(_model.parameters()).device}", flush=True)


class Question(BaseModel):
    type: str
    instructions: object
    criteria: object | None = None


class SystemOneRequest(BaseModel):
    state: object
    questions: dict[str, Question]
    model: str | None = None


class Cancelled(Exception):
    pass


def _state_text(state: object) -> str:
    return state.strip() if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)


def _instr_text(q: Question) -> str:
    """The question; structured instructions ({"question": ..., other fields}) put the other fields after it as data."""
    ins = q.instructions
    if isinstance(ins, str):
        return ins.strip()
    if isinstance(ins, dict):
        qtext = str(ins.get("question", "")).strip()
        block = "\n".join(f"{k}: {v}" for k, v in ins.items() if k != "question")
        return qtext + (f"\n\nData:\n{block}" if block else "")
    return str(ins).strip()


def _norm(probs: list[float]) -> list[float]:
    s = sum(probs) or 1.0
    return [p / s for p in probs]


# ---------------------------------------------------------------- questions -> rows -> answers

def _plan(questions: dict[str, Question]) -> list[tuple[str, str, list, dict]]:
    """Every question as (key, kind, rows, extra). A row is one SemIf readout (at most 16 options)."""
    plan = []
    for key, q in questions.items():
        qt = (q.type or "").lower()
        qtext = _instr_text(q)
        if qt == "noul":
            crit = q.criteria if isinstance(q.criteria, dict) else {}
            opts = [("yes", str(crit.get("true") or "Yes.")), ("no", str(crit.get("false") or "No."))]
            plan.append((key, qt, [(qtext, opts)], {}))
        elif qt == "choice":
            crit = q.criteria or {}
            if not isinstance(crit, dict) or not crit:
                raise HTTPException(422, f"question '{key}': choice needs a non-empty criteria map")
            names = list(crit.keys())
            pairs = list(zip(names, [f"{n}. {crit[n]}" if crit[n] else str(n) for n in names]))
            rows = [(qtext, pairs[i:i + CHUNK]) for i in range(0, len(pairs), CHUNK)]
            if any(len(r[1]) < 2 for r in rows):   # a lone leftover option can't be read out
                rows = [(qtext, pairs[i:i + CHUNK - 1]) for i in range(0, len(pairs), CHUNK - 1)]
            plan.append((key, qt, rows, {"names": names}))
        elif qt == "score":
            crit = q.criteria
            if not isinstance(crit, list) or len(crit) < 2:
                raise HTTPException(422, f"question '{key}': score needs a criteria list of >= 2 levels")
            if len(crit) > CHUNK:
                raise HTTPException(422, f"question '{key}': score supports at most {CHUNK} levels")
            levels = [str(c) for c in crit]
            plan.append((key, qt, [(qtext, [(lv, f"Level {i}: {lv}") for i, lv in enumerate(levels)])],
                         {"levels": levels}))
        else:
            raise HTTPException(422, f"question '{key}': unknown type '{q.type}'")
    return plan


def _answer(kind: str, probs: list[list[float]], extra: dict) -> dict:
    if kind == "noul":
        p = probs[0]
        yes = round(p[0], 4)
        return {"type": "noul", "noul": yes, "probabilities": {"yes": yes, "no": round(p[1], 4)},
                "confidence": round(max(p), 4)}
    if kind == "choice":
        names = extra["names"]
        p = _norm([x for row in probs for x in row]) if len(probs) > 1 else probs[0]
        best = max(range(len(names)), key=lambda i: p[i])
        return {"type": "choice", "choice": names[best],
                "probabilities": {n: round(p[i], 4) for i, n in enumerate(names)}, "confidence": round(p[best], 4)}
    levels, p = extra["levels"], probs[0]
    return {"type": "score", "score": round(sum(i * p[i] for i in range(len(levels))), 4),
            "legend": {str(i): levels[i] for i in range(len(levels))},
            "probabilities": {str(i): round(p[i], 4) for i in range(len(levels))}, "confidence": round(max(p), 4)}


# ---------------------------------------------------------------- GPU work (on a worker thread)

def _slots(n: int) -> list[int]:
    if n not in _slot_cache:
        core, _ = _semif()
        _slot_cache[n] = [_tokenizer.encode(c, add_special_tokens=False)[0] for c in core.LETTERS[:n]]
    return _slot_cache[n]


def _encode(row: dict) -> list[int]:
    """SemIf's exact prompt. Its encode_prompt re-tokenizes the whole prompt once per answer letter to prove the letters
    tokenize cleanly after it; every prompt ends with the same assistant header, so that runs once per option count."""
    core, direct = _semif()
    n = len(row["options"])
    if n not in _verified:
        ids = direct.encode_prompt(_tokenizer, row, MAX_TOKENS)[0]
        _verified.add(n)
        return ids
    prompt = _tokenizer.apply_chat_template(core.direct_messages(row), tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
    ids = _tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids) > MAX_TOKENS:
        raise ValueError(f"{len(ids)} input tokens exceed limit {MAX_TOKENS}; no truncation allowed")
    return ids


def _readout(logits, n: int) -> list[float]:
    import torch
    return torch.softmax(logits[_slots(n)].float(), -1).cpu().tolist()


def _run_rows(state: str, rows: list[tuple[str, list]], cancel: threading.Event, use_cache: bool) -> list[list[float]]:
    """Probabilities for every row. With use_cache, the tokens all rows share (the state) run once and each row continues
    from a copy of that cache."""
    import torch

    dev = next(_model.parameters()).device
    enc = []
    for qtext, opts in rows:
        row = {"id": "sysone", "state": state, "question": qtext,
               "options": [{"id": i, "description": t} for i, t in opts]}
        try:
            enc.append(_encode(row))
        except ValueError as e:
            raise HTTPException(413, str(e)) from None
    STATS["max_row_tokens"] = max(STATS["max_row_tokens"], max(len(e) for e in enc))
    shared = 0
    if use_cache and len(enc) > 1:
        shortest = min(len(e) for e in enc) - 1
        while shared < shortest and len({e[shared] for e in enc}) == 1:
            shared += 1

    def run(ids: list[int], cache=None, start: int = 0, keep: bool = False):
        """(last position's logits, cache) for ids[start:], continuing `cache`. A short input with nothing to keep takes
        SemIf's own single uncached forward; longer ones go in PREFILL_CHUNK pieces."""
        if cache is None and start == 0 and len(ids) <= PREFILL_CHUNK and not keep:
            o = _model(input_ids=torch.tensor([ids], device=dev),
                       attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device=dev),
                       use_cache=False, return_dict=True, logits_to_keep=1)
            STATS["forwards"] += 1
            return o.logits[0, -1], None
        pos = start
        while pos < len(ids):
            if cancel.is_set():
                raise Cancelled
            part = ids[pos:pos + PREFILL_CHUNK]
            o = _model(input_ids=torch.tensor([part], device=dev), past_key_values=cache, use_cache=True,
                       return_dict=True, logits_to_keep=1, cache_position=torch.arange(pos, pos + len(part), device=dev))
            cache, pos = o.past_key_values, pos + len(part)
            STATS["forwards"] += 1
        return o.logits[0, -1], cache

    out: list[list[float]] = []
    with torch.inference_mode():
        if shared >= MIN_PREFIX:
            if cancel.is_set():
                raise Cancelled
            _, pre = run(enc[0][:shared], keep=True)
            STATS["tokens"] += shared
            STATS["cache_hits"] += len(enc)
            for ids, (_, opts) in zip(enc, rows):
                if cancel.is_set():
                    raise Cancelled
                logits, _ = run(ids, copy.deepcopy(pre), shared)
                out.append(_readout(logits, len(opts)))
                STATS["tokens"] += len(ids) - shared
            del pre
        else:
            for ids, (_, opts) in zip(enc, rows):
                if cancel.is_set():
                    raise Cancelled
                logits, _ = run(ids)
                out.append(_readout(logits, len(opts)))
                STATS["tokens"] += len(ids)
    if max(len(e) for e in enc) >= BIG_REQUEST:
        torch.cuda.empty_cache()   # give the peak back instead of holding it
    return out


def _work(state: str, plan, cancel: threading.Event, use_cache: bool) -> dict:
    import torch

    rows = [r for _, _, rs, _ in plan for r in rs]
    try:
        probs = _run_rows(state, rows, cancel, use_cache)
    except torch.cuda.OutOfMemoryError:
        STATS["oom"] += 1
        torch.cuda.empty_cache()
        raise HTTPException(503, "input too large for the GPU's free memory; send a smaller state") from None
    answers, i = {}, 0
    for key, kind, rs, extra in plan:
        answers[key] = _answer(kind, probs[i:i + len(rs)], extra)
        i += len(rs)
    return answers


# ---------------------------------------------------------------- HTTP

app = FastAPI(title="SemIf bridge (System One wire format)")


@app.get("/health")
def health():
    loaded = _model is not None
    if loaded:
        import torch
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            STATS["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            STATS["gpu_free_gb"] = round(free / 2**30, 2)
    return {"status": "up" if loaded else "loading", "model": MODEL_ID, "revision": REVISION,
            "device": str(next(_model.parameters()).device) if loaded else None,
            "prefix_cache": PREFIX_CACHE, "queued_or_running": _gpu.locked(),
            "stats": {k: round(v, 2) if isinstance(v, float) else v for k, v in STATS.items()}}


@app.post("/v1/systemone")
async def systemone(req: SystemOneRequest, request: Request):
    if _model is None:
        raise HTTPException(503, "model still loading")
    if not req.state or not str(req.state).strip():
        raise HTTPException(422, "state is required")
    state = _state_text(req.state)
    plan = _plan(req.questions)
    cancel, finished = threading.Event(), asyncio.Event()

    async def watch() -> None:   # the caller hung up: drop the work instead of finishing it for nobody
        while not finished.is_set():
            if await request.is_disconnected():
                cancel.set()
                return
            await asyncio.sleep(0.2)

    watcher = asyncio.create_task(watch())
    try:
        async with _gpu:
            if cancel.is_set():
                raise Cancelled
            t0, tokens0 = time.perf_counter(), STATS["tokens"]
            try:
                answers = await asyncio.to_thread(_work, state, plan, cancel, PREFIX_CACHE)
                processed = STATS["tokens"] - tokens0   # exact: one request on the GPU at a time
            finally:
                STATS["busy_seconds"] += time.perf_counter() - t0
    except Cancelled:
        STATS["cancelled"] += 1
        raise HTTPException(499, "caller disconnected; work dropped") from None
    finally:
        finished.set()
        watcher.cancel()
    STATS["requests"] += 1
    STATS["questions"] += len(plan)
    return {"model": f"{MODEL_ID}@{REVISION}", "answers": answers,
            "usage": {"input_tokens": processed, "output_tokens": 0}}


def main():
    import uvicorn
    _load()
    plan = _plan({"q": Question(type="noul", instructions="Is this a probe?")})   # warm-up: the first real request
    _work("Probe state.", plan, threading.Event(), False)                          # isn't the slow one
    print(f"warm (prefix cache {'on' if PREFIX_CACHE else 'off'})", flush=True)
    port = int(os.environ.get("SEMIF_PORT") or os.environ.get("SEMFIF_PORT") or "33003")   # (SEMFIF_: the old spelling)
    uvicorn.run(app, host=os.environ.get("SEMIF_HOST", "127.0.0.1"), port=port, log_level="warning")


if __name__ == "__main__":
    main()
