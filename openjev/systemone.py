"""System One adapter: the endpoint jev-harness asks its decisions of (config.yaml `jev.url`, port 33001).

It serves the TypeSafe System One wire format and forwards each request to the SemIf bridge (`semif_bridge.py`, port
33003), which holds the model. Kept as its own small process so the harness - and other local tools that speak System
One - talk to one stable address while the bridge behind it is restarted, moved to another GPU or swapped for another
backend that speaks the same format (point SEMIF_URL at it).

    POST /v1/systemone
    {"state": "...",
     "questions": {"billing":    {"type": "noul",   "instructions": "..."},
                   "department": {"type": "choice", "instructions": "...", "criteria": {"billing": "...", "tech": "..."}},
                   "urgency":    {"type": "score",  "instructions": "...", "criteria": ["low", "medium", "high"]}},
     "model": "semif"}                                   (optional: semif, the default, or laya)
    -> {"model": "...", "answers": {...}, "usage": {"input_tokens": N, "output_tokens": 0}}

`state` may be text or structured data; `instructions` may be text or an object whose "question" is the question and
whose other fields are data. If the caller hangs up, the adapter hangs up on the bridge too, so the bridge drops the
work instead of finishing it for nobody (a caller's timeout once left its requests queued on the GPU for minutes).

Run:
    python systemone.py                    # :33001, forwarding to the bridge on :33003
    python systemone.py --health           # the adapter's view of the bridge, as JSON

`"model": "laya"` goes to the Laya bridge instead (`laya_bridge.py`, port 33004): a small model trained on the harness's
own decisions (config.yaml `jev.backend: laya`; README.md "A smaller Jev").

Settings (environment): SEMIF_URL (http://127.0.0.1:33003), LAYA_URL (http://127.0.0.1:33004), SYSTEMONE_HOST
(127.0.0.1), SYSTEMONE_PORT (33001), SEMIF_TIMEOUT (seconds a request may take, 300).
"""
from __future__ import annotations

import asyncio
import json
import os

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

SEMIF_URL = os.environ.get("SEMIF_URL", "http://127.0.0.1:33003").rstrip("/")
LAYA_URL = os.environ.get("LAYA_URL", "http://127.0.0.1:33004").rstrip("/")
SEMIF_TIMEOUT = float(os.environ.get("SEMIF_TIMEOUT", "300"))
BACKENDS = ("semif", "laya")

app = FastAPI(title="System One adapter (SemIf)")
_client: httpx.AsyncClient | None = None


class Question(BaseModel):
    type: str
    instructions: object
    criteria: object | None = None


class SystemOneRequest(BaseModel):
    state: object
    questions: dict[str, Question]
    model: str | None = None


def _bridge() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=SEMIF_TIMEOUT)
    return _client


@app.get("/health")
async def health():
    """Fast: the adapter, and whether each bridge answers (2 s at most). The Laya bridge is optional: down is normal when
    the harness uses semif."""
    async def probe(url: str) -> dict:
        try:
            r = await _bridge().get(f"{url}/health", timeout=2.0)
            sh = r.json() if r.status_code == 200 else {}
            return {"status": "up" if sh.get("status") == "up" else (sh.get("status") or f"http {r.status_code}"),
                    "model": sh.get("model")}
        except Exception as e:  # noqa: BLE001
            return {"status": f"down: {type(e).__name__}: {e}"[:200]}
    semif, laya = await asyncio.gather(probe(SEMIF_URL), probe(LAYA_URL))
    return {"adapter": "up", "default": "semif", "semif": semif, "semif_url": SEMIF_URL, "laya": laya,
            "laya_url": LAYA_URL}


@app.post("/v1/systemone")
async def systemone(req: SystemOneRequest, request: Request):
    backend = (req.model or "semif").strip().lower()
    if backend not in BACKENDS:
        raise HTTPException(422, f"unknown model '{req.model}' (this adapter serves: {', '.join(BACKENDS)})")
    if not req.state or not str(req.state).strip():
        raise HTTPException(422, "state is required")
    body = {"state": req.state, "questions": {k: q.model_dump(exclude_none=True) for k, q in req.questions.items()}}
    url = SEMIF_URL if backend == "semif" else LAYA_URL
    call = asyncio.create_task(_bridge().post(f"{url}/v1/systemone", json=body))
    while not call.done():
        if await request.is_disconnected():
            call.cancel()   # closes the connection: the bridge sees the disconnect and drops the work
            raise HTTPException(499, "caller disconnected")
        await asyncio.wait({call}, timeout=0.2)
    try:
        r = call.result()
    except httpx.HTTPError as e:
        raise HTTPException(503, f"{backend} bridge unreachable at {url}: {e}") from None
    if r.status_code != 200:
        raise HTTPException(r.status_code if r.status_code in (413, 422, 503) else 502,
                            f"{backend} error {r.status_code}: {r.text[:300]}")
    out = r.json()
    if not isinstance(out, dict) or "answers" not in out:
        raise HTTPException(502, f"{backend} returned a malformed body: {str(out)[:200]}")
    usage = out.get("usage") or {}
    return {"model": out.get("model") or backend, "answers": out["answers"],
            "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)}}


def main():
    import argparse

    import uvicorn
    ap = argparse.ArgumentParser(description="System One adapter for jev-harness (forwards to the SemIf bridge)")
    ap.add_argument("--host", default=os.environ.get("SYSTEMONE_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("SYSTEMONE_PORT", "33001")))
    ap.add_argument("--health", action="store_true", help="print the adapter's view of the bridge and exit")
    a = ap.parse_args()
    if a.health:
        print(json.dumps(asyncio.run(health()), indent=2))
        return
    print(f"System One adapter on http://{a.host}:{a.port} -> SemIf bridge {SEMIF_URL}, Laya bridge {LAYA_URL}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
