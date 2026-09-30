"""Exclusive model loading for an LM Studio server.

Two big models loaded at once spill out of VRAM and both crawl (measured: a one-word
reply took 15-19s with qwen3.8-27b + ornith both resident, vs ~1s alone). LM Studio's
JIT loading happily loads a second model next to the first, so the harness enforces
"one LLM at a time" itself:

- same model already loaded  -> go straight through; parallel requests stay parallel
- different model requested  -> wait for in-flight requests on the current model to
  finish, unload every other LLM, load the requested one, then continue

One gate per server URL, shared by every persona. Embedding models are left alone.
REST: GET /api/v1/models (loaded_instances), POST /api/v1/models/unload {instance_id},
POST /api/v1/models/load {model}.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Callable, Iterator

_GATES: dict[str, "ExclusiveModels"] = {}
_GATES_LOCK = threading.Lock()


def gate_for(base_url: str) -> "ExclusiveModels":
    """The shared gate for an LM Studio server (base_url may end in /v1)."""
    root = base_url.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    with _GATES_LOCK:
        if root not in _GATES:
            _GATES[root] = ExclusiveModels(root)
        return _GATES[root]


def _base(model: str) -> str:
    return model.split("@", 1)[0]  # "qwen3.8-27b@q6_k" -> "qwen3.8-27b"


class ExclusiveModels:
    def __init__(self, root: str, load_timeout: float = 600.0, linger: float = 8.0, max_wait: float = 180.0,
                 recheck: float = 30.0):
        self.root = root
        # `current` is this process's belief. Another process on the same server (refine's test runs, the
        # dashboard next to a CLI) may have swapped models since: after `recheck` idle seconds the gate asks
        # LM Studio again. Live: refine's writer believed Qwen was loaded while its test runs had swapped to
        # ornith, sent the request cold, and gave up after 33s with nothing streamed.
        self.recheck = recheck
        self.load_timeout = load_timeout
        # A worker alternates LLM calls with REPL steps. Without a linger, parallel hand-offs on
        # different models would swap on every step (~15s each). The current model stays for
        # `linger` seconds after its last request so the same worker's next step keeps it; a
        # waiter for another model gets it once the current one is idle that long (e.g. its worker
        # finished, or is blocked waiting on a hand-off) - or after `max_wait` regardless.
        self.linger = linger
        self.max_wait = max_wait
        self.current: str | None = None      # model the gate last ensured
        self.inflight = 0
        self.last_used = 0.0
        self._cond = threading.Condition()
        self.on_swap: Callable[[list[str], str, float], None] | None = None  # (unloaded, loaded, secs)

    # -- public ------------------------------------------------------------------------
    @contextmanager
    def use(self, model: str) -> Iterator[None]:
        with self._cond:
            if self.current is not None and self.inflight == 0 and time.monotonic() - self.last_used >= self.recheck:
                self.current = None   # idle a while: ask LM Studio what's loaded (_swap_to is a no-op if it's ours)
            deadline = time.monotonic() + self.max_wait
            while self.current is not None and not self._same(self.current, model):
                if self.inflight == 0:
                    idle = time.monotonic() - self.last_used
                    if idle >= self.linger or time.monotonic() >= deadline:
                        break
                    self._cond.wait(self.linger - idle)
                else:
                    self._cond.wait(max(0.05, deadline - time.monotonic()) if time.monotonic() < deadline else None)
            if self.current is None or not self._same(self.current, model):
                self._swap_to(model)  # holding the lock: nobody else starts a request mid-swap
                self.current = model
            self.inflight += 1
        try:
            yield
        finally:
            with self._cond:
                self.inflight -= 1
                self.last_used = time.monotonic()
                self._cond.notify_all()

    # -- LM Studio REST ----------------------------------------------------------------
    def _request(self, method: str, path: str, body: dict | None = None, timeout: float = 30.0) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(f"{self.root}{path}", data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"LM Studio {path}: HTTP {e.code} {e.read().decode('utf-8', 'replace')[:300]}") from e

    def _list(self) -> list[dict]:
        return self._request("GET", "/api/v1/models").get("models", [])

    def _models(self) -> list[dict]:
        """LLMs only: embedding models are small and never unloaded."""
        return [m for m in self._list() if m.get("type") == "llm"]

    def _loaded(self) -> list[tuple[str, str]]:
        """(model key, instance id) for every loaded LLM."""
        return [(m["key"], inst["id"]) for m in self._models() for inst in m.get("loaded_instances") or []]

    def _unload(self, instance_id: str) -> None:
        self._request("POST", "/api/v1/models/unload", {"instance_id": instance_id}, timeout=120)

    def _load(self, model: str) -> None:
        self._request("POST", "/api/v1/models/load", {"model": model}, timeout=self.load_timeout)

    # -- internals ---------------------------------------------------------------------
    @staticmethod
    def _same(a: str, b: str) -> bool:
        return a == b or _base(a) == _base(b)

    def _swap_to(self, model: str) -> None:
        started = time.time()
        loaded = self._loaded()
        keep = [i for k, i in loaded if self._same(model, k) or self._same(model, i)]
        dropped = [i for k, i in loaded if i not in keep]
        for inst in dropped:  # unload FIRST so the new model gets the whole GPU
            self._unload(inst)
        if not keep:
            keys = {m["key"] for m in self._models()}
            self._load(model if model in keys else _base(model))
        if (dropped or not keep) and self.on_swap:
            self.on_swap(dropped, model, time.time() - started)
