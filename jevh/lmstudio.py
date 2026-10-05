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

The same list says what each loaded model's context length is (loaded_instances[].config.context_length: the value
set in LM Studio's load settings, not the model's own maximum, which is max_context_length): `model_info`.
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


def _positive(value) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else None


def info_from(models: list[dict], model: str) -> dict | None:
    """What a GET /api/v1/models answer says about `model`, or None when it lists no such LLM:
      {"key", "loaded": bool,
       "context":  the context length it is loaded with (the smallest, if two instances are loaded), None: not loaded,
       "max":      the most the model itself supports,
       "parallel": the loaded instance's parallel slots}"""
    for m in models:
        if m.get("type", "llm") != "llm":
            continue
        instances = [i for i in m.get("loaded_instances") or [] if isinstance(i, dict)]
        if not (ExclusiveModels._same(model, str(m.get("key", ""))) or
                any(ExclusiveModels._same(model, str(i.get("id", ""))) for i in instances)):
            continue
        sized = []   # (context length, that instance's config)
        for inst in instances:
            cfg = inst.get("config") if isinstance(inst.get("config"), dict) else {}
            if (c := _positive(cfg.get("context_length"))) is not None:
                sized.append((c, cfg))
        sized.sort(key=lambda t: t[0])   # (never compare the configs)
        return {"key": m.get("key", ""), "loaded": bool(instances),
                "context": sized[0][0] if sized else None,
                "max": _positive(m.get("max_context_length")),
                "parallel": _positive(sized[0][1].get("parallel")) if sized else None}
    return None


class ExclusiveModels:
    def __init__(self, root: str, load_timeout: float = 600.0, linger: float = 8.0, max_wait: float = 180.0,
                 recheck: float = 30.0):
        self.root = root
        # What a model is loaded with (`model_info`) is asked of LM Studio at most every `info_ttl` seconds per model,
        # on a short timeout, and never under the swap lock: a worker asks before every step, and a swap can hold
        # the lock for minutes. If LM Studio cannot be asked, the last answer stands for `info_stale` seconds.
        self.detect = True
        self.info_ttl, self.info_stale, self.info_timeout = 10.0, 600.0, 3.0
        self._info: dict[str, tuple[float, dict | None]] = {}
        self._info_lock = threading.Lock()
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

    def model_info(self, model: str) -> dict | None:
        """The context LM Studio loaded `model` with (see `info_from`), or None: detection is off, LM Studio does not
        list the model, or it cannot be asked and there is no recent answer."""
        if not self.detect:
            return None
        key, now = _base(model), time.monotonic()
        with self._info_lock:
            hit = self._info.get(key)
        if hit and now - hit[0] < self.info_ttl:
            return hit[1]
        try:
            models = self._request("GET", "/api/v1/models", timeout=self.info_timeout).get("models", [])
        except Exception:  # noqa: BLE001 - LM Studio down or busy: the last answer, if it is recent
            return hit[1] if hit and now - hit[0] < self.info_stale else None
        info = info_from(models, model)
        with self._info_lock:
            self._info[key] = (now, info)
        return info

    def llms_info(self) -> list[dict]:
        """Every LLM LM Studio lists, loaded or not, as `info_from` describes it (the dashboard's status card and
        `jevh context`)."""
        return [i for m in self._models() if (i := info_from([m], str(m.get("key", ""))))]

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
