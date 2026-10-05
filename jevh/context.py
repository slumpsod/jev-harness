"""A model's context window: asked of LM Studio (or set in config), and what the harness does with it.

LM Studio loads each model with a context length chosen in its load settings (100096 tokens for Qwen3.8 27B here,
though the model itself supports 262144). The harness used to assume one number for every model: `limits.compact_tokens`,
written by hand for that Qwen. Now, before each step of a task, it asks LM Studio what the model it is using was
loaded with, and condenses older steps (worker._compact) once the prompt is `compact_at` of that window:

    jevh context            what the harness knows about each model's window and when it will condense
    jevh context --json     the same for a script
    jevh context --model lmstudio:qwen3.8-27b@q6_k     one model

Where a model's window comes from, first match wins:

  1. `context_length` in the model's block (a persona's `model:`, `models.fast`) or in its provider: what to assume.
     It tells the harness; it does not change what LM Studio loads. Use it where nothing can be asked (vLLM,
     llama.cpp, OpenAI), or to hold a model to less than it is loaded with.
  2. LM Studio (a provider with `exclusive: true`): GET /api/v1/models, loaded_instances[].config.context_length.
     A model that is not loaded yet has no window until LM Studio loads it (what it loads with is LM Studio's own
     default for that model): until then limits.compact_tokens stands. The answer is reused for 10 s, so a model
     reloaded with another context is noticed within moments, even mid-task.
  3. Unknown: limits.compact_tokens.

LM Studio's context length is for the whole model, shared by its parallel slots (its unified KV cache, the default):
each request may use all of it. With that cache turned off, N slots get 1/N each: say so with `context_length`.

Settings (config.yaml `context:`):

    context:
      detect: true       # ask LM Studio what each model is loaded with; false: never ask (limits.compact_tokens)
      compact_at: 0.6    # older steps condense once the prompt is this share of the window (but the reply's room
                         # is always kept: the window minus the model's max_tokens, a quarter if it may think freely)
      min: 16384         # a window below this is flagged: the instructions and tools alone take several thousand tokens
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

DEFAULTS = {"detect": True, "compact_at": 0.6, "min": 16384}
MIN_COMPACT = 2048   # never condense below this (a window this small cannot run the harness anyway)


def settings(raw: dict[str, Any] | None) -> dict[str, Any]:
    return {**DEFAULTS, **((raw or {}).get("context") or {})}


@dataclass(frozen=True)
class Window:
    """What a model client knows about its model's context."""
    tokens: int | None            # what one request can use; None: not known (yet)
    source: str                   # "lmstudio" (asked) | "config" (told)
    loaded: bool = True           # lmstudio: False while LM Studio has not loaded the model
    max: int | None = None        # the most the model itself supports (lmstudio)
    parallel: int | None = None   # the loaded instance's parallel slots (lmstudio)


@dataclass(frozen=True)
class Budget:
    """What a worker does with it: condense older steps once the estimated prompt passes `compact`."""
    compact: int
    window: int | None            # None: the window is not known, `compact` is limits.compact_tokens
    source: str = ""
    loaded: bool | None = None
    max: int | None = None
    parallel: int | None = None
    warn: str = ""                # said to the user: this window is too small for the harness

    def public(self) -> dict[str, Any]:
        return asdict(self)


def compact_tokens(window: int, reply: int | None, share: float) -> int:
    """The estimated prompt size at which older steps condense, for a `window`-token context: `share` of it, but
    never so late that the model's reply (`reply` tokens; a quarter of the window when it may think without limit)
    no longer fits behind the prompt."""
    room = min(reply or window // 4, window // 2)
    return max(MIN_COMPACT, min(int(window * share), window - room))


def budget(window: Window, reply: int | None, fallback: int, s: dict[str, Any]) -> Budget:
    """`window`'s compaction limit (`fallback`, limits.compact_tokens, while the window is not known)."""
    if not window.tokens:
        return Budget(fallback, None, window.source, window.loaded, window.max, window.parallel)
    warn = ""
    if window.tokens < int(s["min"]):
        how = "load it with more in LM Studio" if window.source == "lmstudio" else "use a model with more"
        warn = (f"{window.tokens:,} tokens of context is little for the harness (its instructions and tools take "
                f"several thousand): {how}, at least {int(s['min']):,}")
    return Budget(compact_tokens(window.tokens, reply, float(s["compact_at"])), window.tokens, window.source,
                  window.loaded, window.max, window.parallel, warn)


def configure(cfg) -> dict[str, Any]:
    """Apply `context.detect` to every LM Studio gate (Harness and `jevh context` both call it). -> the settings."""
    from .lmstudio import gate_for
    s = settings(cfg.raw)
    for prov in cfg.providers.values():
        if prov.get("exclusive") and prov.get("base_url"):
            gate_for(prov["base_url"]).detect = bool(s["detect"])
    return s


# -- jevh context -------------------------------------------------------------------------------------------------

def models_in_use(cfg, only: str | None = None) -> list[dict[str, Any]]:
    """The distinct model blocks the config puts to work, each with who uses it:
    [{"block": {provider, model, ...}, "used_by": [persona or tier, ...]}]."""
    from .personas import load_personas
    blocks: list[tuple[str, dict[str, Any]]] = []
    if only:
        provider, _, model = only.partition(":")
        blocks.append(("(asked for)", {"provider": provider, "model": model}))
    else:
        for name, p in load_personas(cfg.personas_dirs, []).items():
            blocks.append((name, {k: v for k, v in p.model.items() if k != "fallback"}))
        for tier, block in (cfg.models or {}).items():
            blocks.append((f"{tier} tier", dict(block)))
    out: list[dict[str, Any]] = []
    for who, block in blocks:
        key = (block.get("provider"), block.get("model"), block.get("context_length"), block.get("max_tokens", 8192),
               block.get("think"))
        for o in out:
            if o["key"] == key:
                o["used_by"].append(who)
                break
        else:
            out.append({"key": key, "block": block, "used_by": [who]})
    return [{"block": o["block"], "used_by": o["used_by"]} for o in out]


def describe(cfg, block: dict[str, Any], s: dict[str, Any]) -> dict[str, Any]:
    """What the harness knows and will do for one model block."""
    from .llm import make_llm
    name = f"{block.get('provider')}:{block.get('model')}"
    fallback = cfg.lim("compact_tokens")
    row: dict[str, Any] = {"model": name, "window": None, "source": "", "loaded": None, "max": None, "parallel": None,
                           "compact": fallback, "warn": "", "error": ""}
    try:
        llm = make_llm(block, cfg.providers)
        window = llm.window() if hasattr(llm, "window") else None
    except Exception as e:  # noqa: BLE001 - one model's problem must not hide the others
        row["error"] = f"{type(e).__name__}: {e}"[:200]
        return row
    if window is not None:
        row.update(budget(window, getattr(llm, "max_tokens", None), fallback, s).public())
    return row


def lmstudio_models(cfg) -> list[dict[str, Any]]:
    """Every LLM each LM Studio server lists, with what it is loaded with (or an error saying it could not be asked)."""
    from .lmstudio import gate_for
    out = []
    for name, prov in cfg.providers.items():
        if prov.get("exclusive") and prov.get("base_url"):
            try:
                out.append({"provider": name, "url": prov["base_url"], "models": gate_for(prov["base_url"]).llms_info()})
            except Exception as e:  # noqa: BLE001
                out.append({"provider": name, "url": prov["base_url"], "models": [], "error": str(e)[:200]})
    return out


def report(cfg, only: str | None = None) -> dict[str, Any]:
    s = configure(cfg)
    rows = []
    for m in models_in_use(cfg, only):
        row = describe(cfg, m["block"], s)
        row["used_by"] = m["used_by"]
        rows.append(row)
    return {"settings": s, "lmstudio": lmstudio_models(cfg), "models": rows, "fallback": cfg.lim("compact_tokens")}


def _n(v: Any) -> str:
    return f"{v:,}" if isinstance(v, int) else "-"


def render(r: dict[str, Any]) -> str:
    s, lines = r["settings"], []
    for lm in r["lmstudio"]:
        lines.append(f"LM Studio '{lm['provider']}' at {lm['url']}" + ("" if s["detect"] else "  (context.detect is off: not asked)"))
        if lm.get("error"):
            lines.append(f"  could not be asked: {lm['error']}")
        for m in lm["models"]:
            lines.append(f"  {m['key']:<34} loaded with {_n(m['context']):>9}   supports {_n(m['max']):>9}"
                         f"   parallel {m['parallel'] or '-'}" if m["loaded"] else
                         f"  {m['key']:<34} not loaded            supports {_n(m['max']):>9}")
        lines.append("")
    lines.append(f"What the harness does with each model it is set to use (condense older steps at {s['compact_at']:.0%} "
                 f"of the window; limits.compact_tokens = {r['fallback']:,} where the window is not known):")
    for m in r["models"]:
        if m["window"]:
            what = f"window {m['window']:,} ({'asked LM Studio' if m['source'] == 'lmstudio' else 'from config'})"
        elif m["loaded"] is False:
            what = "window not known: LM Studio has not loaded it yet"
        elif m["error"]:
            what = f"could not be set up: {m['error']}"
        else:
            what = "window not known (nothing to ask: set context_length to tell the harness)"
        lines.append(f"  {m['model']:<38} {what}  -> condenses at {m['compact']:,}"
                     f"   [{', '.join(m['used_by'])}]")
        if m.get("warn"):
            lines.append(f"      ! {m['warn']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="jevh context", description="What the harness knows about each model's context window.")
    ap.add_argument("--config")
    ap.add_argument("--model", help="one model, provider:model, e.g. lmstudio:qwen3.8-27b@q6_k")
    ap.add_argument("--json", action="store_true", help="print JSON")
    a = ap.parse_args(argv)
    from .config import load_config
    r = report(load_config(a.config), a.model)
    print(json.dumps(r, indent=2) if a.json else render(r))
    return 0
