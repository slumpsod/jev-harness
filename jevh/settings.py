"""The dashboard's Settings page: the few choices people actually change, without editing YAML.

    Model         which model every assistant runs on (models.strong); "each assistant's own" leaves it to their files
    Quick model   the model easy tasks go to (models.fast); "off" always uses the main one
    Web search    DuckDuckGo, SearXNG (your server's address) or Brave Search (an API key); `Test` tries it first
    Jev           full (the System One model makes every decision) or light (UNVERIFIED: a small model that runs on a
                  CPU makes the decisions it is good at; the rest go to the task's model, or are skipped: jevh/light.py)
    Chat glass    how see-through the desktop app's orb chat is (appearance.orb_glass, 0 solid to 100): saved at once,
                  on every computer signed in to this hub (the orb's chat page reads it with the conversation's state)

They are saved in `<workspace>/settings.yaml`, which jevh/config.py lays over every other config file, so a choice made in the
tab wins and `config.yaml` / `config.local.yaml` (with their comments) are never rewritten. Delete the file, or press Reset,
to go back to what the config files say. Only the sections in config.SETTINGS_KEYS are read from it.

A value of null in it removes what the files below it set (that is how "each assistant's own" and "off" are saved).
The file may hold a Brave key, so it is created readable by you only, and the key is never sent back to the page.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from . import light
from .config import SETTINGS_FILE, Config, load_config, read_settings
from .tools import web

ORB_GLASS = 10   # the chat's glass when nothing is saved (jevh/orb_chat_art.GLASS: the look it was made with)
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}   # the servers whose models are listed (a cloud provider is never probed)
DISCOVER_TTL = 10.0
_discovered: dict[str, tuple[float, list[dict]]] = {}
_lock = threading.Lock()


class SettingsError(ValueError):
    """A choice that cannot be saved; the message says what to change."""


def path(cfg: Config) -> Path:
    return Path(cfg.workspace) / SETTINGS_FILE


# -- what is set now --------------------------------------------------------------------------------------------------

def _model_text(block: Any) -> str:
    return f"{block['provider']}:{block['model']}" if isinstance(block, dict) and block.get("provider") and block.get("model") else ""


def view(cfg: Config) -> dict[str, Any]:
    """What the tab shows: the effective choices (config files and saved settings together) and where they are saved."""
    s = cfg.raw.get("search") or {}
    return {"model": _model_text(cfg.models.get("strong")), "fast": _model_text(cfg.models.get("fast")),
            "search": {"provider": str(s.get("provider") or "duckduckgo"), "searxng_url": str(s.get("searxng_url") or ""),
                       "brave_key": bool(s.get("brave_api_key") or os.environ.get("BRAVE_SEARCH_API_KEY"))},
            "appearance": appearance(cfg), "jev": jev_view(cfg), "file": str(path(cfg)), "saved": path(cfg).is_file()}


def jev_view(cfg: Config) -> dict[str, str]:
    """{mode: full|light, checks: worker|off}: light Jev and where its handed-off decisions go (jevh/light.py)."""
    s = light.settings(cfg.jev)
    return {"mode": s["mode"], "checks": s["checks"]}


def appearance(cfg: Config) -> dict[str, int]:
    """How the desktop app looks: {orb_glass: 0..100}."""
    a = cfg.raw.get("appearance") or {}
    try:
        glass = int(a.get("orb_glass", ORB_GLASS)) if isinstance(a, dict) else ORB_GLASS
    except (TypeError, ValueError):
        glass = ORB_GLASS
    return {"orb_glass": min(100, max(0, glass))}


def providers_view() -> list[dict[str, str]]:
    return [{"id": k, "label": v["label"], "needs": v["needs"], "help": v["help"]} for k, v in web.PROVIDERS.items()]


# -- which models can be picked ---------------------------------------------------------------------------------------

def _is_local(url: str) -> bool:
    return (urllib.parse.urlparse(str(url)).hostname or "") in LOCAL_HOSTS


def _get(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _discover(name: str, prov: dict[str, Any], timeout: float) -> list[dict]:
    """The models one local server offers: LM Studio's downloaded LLMs (which are loaded), or what an OpenAI-style server
    (llama.cpp, vLLM) lists. [] when it does not answer."""
    base = str(prov.get("base_url") or "").rstrip("/")
    try:
        if prov.get("exclusive"):   # LM Studio: its own list says which are loaded
            root = base[:-3] if base.endswith("/v1") else base
            models = _get(f"{root}/api/v1/models", timeout).get("models", [])
            return [{"value": f"{name}:{m['key']}", "label": m["key"], "group": name,
                     "loaded": bool(m.get("loaded_instances"))} for m in models if m.get("type", "llm") == "llm" and m.get("key")]
        return [{"value": f"{name}:{m['id']}", "label": m["id"], "group": name, "loaded": None}
                for m in _get(f"{base}/models", timeout).get("data", []) if isinstance(m, dict) and m.get("id")]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return []


def model_options(cfg: Config, timeout: float = 2.0, fresh: bool = False) -> dict[str, Any]:
    """{models: [{value: "provider:model", label, group, loaded}], down: [providers that did not answer]}. Only servers on
    this machine are asked (a cloud provider is not probed); anything else can be typed in as provider:model."""
    wanted = {n: p for n, p in cfg.providers.items() if p.get("type") == "openai" and _is_local(p.get("base_url", ""))}
    now = time.monotonic()
    out: list[dict] = []
    todo = []
    with _lock:
        for n, p in wanted.items():
            hit = _discovered.get(f"{n}|{p.get('base_url')}")
            if hit and not fresh and now - hit[0] < DISCOVER_TTL:
                out += hit[1]
            else:
                todo.append((n, p))
    if todo:
        with ThreadPoolExecutor(max_workers=len(todo)) as ex:
            found = list(ex.map(lambda np: _discover(np[0], np[1], timeout), todo))
        with _lock:
            for (n, p), models in zip(todo, found):
                _discovered[f"{n}|{p.get('base_url')}"] = (now, models)
        for models in found:
            out += models
    have = {m["group"] for m in out}
    return {"models": out, "down": [n for n in wanted if n not in have]}


# -- saving -----------------------------------------------------------------------------------------------------------

def _parse_model(cfg: Config, text: Any, what: str) -> dict[str, str] | None:
    """"provider:model" -> {provider, model}; "" -> None (leave it to the files). The provider must be one in the config."""
    text = str(text or "").strip()
    if not text:
        return None
    provider, _, model = text.partition(":")
    if provider not in cfg.providers or not model.strip():
        raise SettingsError(f"{what}: write it as provider:model, with a provider from "
                            f"{', '.join(cfg.providers)} (for example {next(iter(cfg.providers), 'lmstudio')}:qwen3.8-27b)")
    if len(model) > 200 or re.search(r"[\s\x00-\x1f]", model):
        raise SettingsError(f"{what}: that is not a model name")
    return {"provider": provider, "model": model}


def _clean_search(current: dict[str, Any], given: dict[str, Any]) -> dict[str, str]:
    provider = str(given.get("provider") or "").strip().lower()
    if provider not in web.PROVIDERS:
        raise SettingsError(f"Web search: choose one of {', '.join(web.PROVIDERS)}")
    out = {"provider": provider}
    url = str(given.get("searxng_url", current.get("searxng_url", "")) or "").strip().rstrip("/")
    if url and not re.fullmatch(r"https?://[^\s/]+[^\s]*", url):
        raise SettingsError("SearXNG address: write it as http://host:port")
    if provider == "searxng" and not url:
        raise SettingsError("SearXNG needs its server address (http://host:port)")
    if url:
        out["searxng_url"] = url
    key = given.get("brave_api_key")   # None: keep the saved key, "": remove it, text: replace it
    key = current.get("brave_api_key", "") if key is None else str(key).strip()
    if re.search(r"[\s\x00-\x1f]", key) or len(key) > 300:
        raise SettingsError("Brave API key: that does not look like a key")
    if provider == "brave" and not key and not os.environ.get("BRAVE_SEARCH_API_KEY"):
        raise SettingsError("Brave Search needs an API key")
    if key:
        out["brave_api_key"] = key
    return out


def _clean_jev(current: dict[str, Any], given: dict[str, Any]) -> dict[str, Any]:
    """Only jev.mode and jev.light.checks are the tab's: the addresses and backends stay in the config files."""
    out = {k: v for k, v in current.items() if k in ("mode", "light")}
    if "mode" in given:
        mode = str(given["mode"] or "").strip().lower()
        if mode not in ("full", "light"):
            raise SettingsError("Jev: choose full or light")
        out["mode"] = mode
    if "checks" in given:
        checks = str(given["checks"] or "").strip().lower()
        if checks not in ("worker", "off"):
            raise SettingsError("Light Jev's other checks: choose worker or off")
        out["light"] = {"checks": checks}
    return out


def save(cfg: Config, change: dict[str, Any]) -> dict[str, Any]:
    """Validate `change` ({model?, fast?, search?: {provider, searxng_url?, brave_api_key?}, jev?: {mode?, checks?}}), write it into the settings
    file (keeping what was saved before for the parts not mentioned), and load it into `cfg`. -> view(cfg).
    SettingsError, with nothing written, when a choice cannot be saved."""
    layer = read_settings(cfg.workspace)
    models = dict(layer.get("models") or {})
    for key, tier, what in (("model", "strong", "Model"), ("fast", "fast", "Quick model")):
        if key in change:
            models[tier] = _parse_model(cfg, change[key], what)   # None: saved as null, which removes what the files set
    if models:
        layer["models"] = models
    if isinstance(change.get("search"), dict):
        layer["search"] = _clean_search(layer.get("search") or {}, change["search"])
    if isinstance(change.get("jev"), dict):
        layer["jev"] = _clean_jev(layer.get("jev") if isinstance(layer.get("jev"), dict) else {}, change["jev"])
    unknown = set(change) - {"model", "fast", "search", "jev"}
    if unknown:
        raise SettingsError(f"unknown setting: {', '.join(sorted(unknown))}")
    _write(path(cfg), layer)
    reload(cfg)
    return view(cfg)


def save_appearance(cfg: Config, change: dict[str, Any]) -> dict[str, int]:
    """Save how the app looks ({orb_glass: 0..100}) at once: it changes no model, so it needs no Save and no idle hub.
    -> appearance(cfg)."""
    unknown = set(change) - {"orb_glass"}
    if unknown or "orb_glass" not in change:
        raise SettingsError(f"unknown setting: {', '.join(sorted(unknown)) or '(none)'}")
    glass = change["orb_glass"]
    if isinstance(glass, bool) or not isinstance(glass, (int, float)) or not 0 <= glass <= 100:
        raise SettingsError("Chat glass: a number from 0 to 100")
    layer = read_settings(cfg.workspace)
    layer["appearance"] = {**(layer.get("appearance") if isinstance(layer.get("appearance"), dict) else {}),
                           "orb_glass": round(glass)}
    _write(path(cfg), layer)
    reload(cfg)
    return appearance(cfg)


def reset(cfg: Config) -> dict[str, Any]:
    """Forget every saved choice: the config files say what is used again."""
    try:
        path(cfg).unlink()
    except FileNotFoundError:
        pass
    reload(cfg)
    return view(cfg)


def _write(target: Path, layer: dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".settings-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("# Saved by the dashboard's Settings page (jevh/settings.py); laid over every config file. Delete it to go back.\n")
            yaml.safe_dump(layer, f, sort_keys=False, allow_unicode=True)
        try:
            os.chmod(tmp, 0o600)   # (it can hold an API key; on Windows this only drops the read-only flag)
        except OSError:
            pass
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def reload(cfg: Config) -> None:
    """Load the files again into `cfg`, but only the sections the tab manages: a harness running on this Config sees the new
    models and search choice (it reads cfg.models and cfg.raw['search'] when it builds a model client or a worker's
    environment). Everything else in `cfg` stays as it is."""
    fresh = load_config(cfg.source, settings_dir=cfg.workspace)   # (the folder save() wrote to)
    cfg.models = fresh.models
    cfg.jev = fresh.jev
    for key in ("models", "search", "appearance", "jev"):
        if fresh.raw.get(key) is None:
            cfg.raw.pop(key, None)
        else:
            cfg.raw[key] = fresh.raw[key]


# -- trying a search choice before it is saved ------------------------------------------------------------------------

def test_search(cfg: Config, given: dict[str, Any]) -> dict[str, Any]:
    """Run one real search with the choice as it is on the page (a key left blank is the saved one). -> {ok, count, first,
    seconds} or {ok: False, error}."""
    try:
        cleaned = _clean_search(cfg.raw.get("search") or {}, given)
    except SettingsError as e:
        return {"ok": False, "error": str(e)}
    started = time.monotonic()
    try:
        hits = web.search_with(web.search_env(cleaned), "open source language models", 3)
    except web.SearchError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:  # noqa: BLE001 - a test must say what went wrong, whatever it was
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    took = round(time.monotonic() - started, 1)
    if not hits:
        return {"ok": False, "error": "The search worked but found nothing for the test query.", "seconds": took}
    return {"ok": True, "count": len(hits), "first": hits[0]["title"][:120], "seconds": took}
