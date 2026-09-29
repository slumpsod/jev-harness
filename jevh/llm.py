"""Chat model providers. Every persona picks one in its `model:` block.

- type openai:    any OpenAI-compatible /v1/chat/completions server (LM Studio, llama.cpp,
                  vLLM, OpenAI). stdlib HTTP, no SDK needed.
- type anthropic: Claude via the official `anthropic` SDK (imported lazily, so the
                  harness runs without it when no persona uses Claude).

Messages are plain [{"role": "system"|"user"|"assistant", "content": str}].
"""
from __future__ import annotations

import json
import threading
import os
import time
from pathlib import Path
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from .errors import Cancelled
from .lmstudio import gate_for

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    thinking_tokens: int = 0   # part of output_tokens spent in hidden reasoning
    reasoning: str = ""        # that hidden reasoning's text, where the server returns it
    tool_calls: list = field(default_factory=list)   # native calls: [{"id", "name", "arguments" (JSON text)}]


class LLM(Protocol):
    name: str

    def chat(self, messages: list[dict[str, str]]) -> LLMResult: ...


class LLMError(RuntimeError):
    pass


def _close_on_cancel(response, cancel: threading.Event, done: threading.Event) -> None:
    """Close a streaming response from outside when the user stops the turn: the read in the
    worker's thread fails at once (even while the server is still processing the prompt)."""
    while not done.wait(0.2):
        if cancel.is_set():
            try:
                response.close()
            except Exception:  # noqa: BLE001
                pass
            return


class _LiveFeed:
    """Streams a reply into a text file as it is generated, section by section, so a long think
    can be read (and judged) while it is still going."""

    def __init__(self, path, model: str):
        self.fh, self.section, self.n = None, None, 0
        if path:
            try:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                self.fh = open(path, "w", encoding="utf-8")
                self.fh.write(f"# {model} - started {time.strftime('%H:%M:%S')}\n")
            except OSError:
                self.fh = None

    def write(self, section: str, text: str) -> None:
        if self.fh is None or not text:
            return
        if section != self.section:
            self.fh.write(f"\n\n=== {section} ===\n")
            self.section = section
        self.fh.write(text)
        self.n += 1
        if self.n % 20 == 0:
            self.fh.flush()

    def close(self, note: str) -> None:
        if self.fh is not None:
            self.fh.write(f"\n\n# done {time.strftime('%H:%M:%S')}: {note}\n")
            self.fh.close()
            self.fh = None


class OpenAICompat:
    """Streams, so `idle_timeout` bounds the gap between tokens, not the whole reply
    (a local 27B at ~27 tok/s legitimately takes minutes on a long answer)."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 temperature: float | None = None, max_tokens: int | None = 8192,
                 idle_timeout: float = 180.0, extra: dict[str, Any] | None = None,
                 gate: Any = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.idle_timeout = idle_timeout
        self.extra = extra or {}
        self.name = f"{model}@{self.base_url}"
        self.on_progress = None  # callable(tokens_so_far) set by the UI, optional
        self.gate = gate         # lmstudio.ExclusiveModels: one model resident at a time
        self.unlimited_thinking = False  # think: unlimited -> no cap, and workers skip the first-step nudge
        self.native_tools = False        # provider/persona native_tools: true -> tools= in chat()

    def chat(self, messages: list[dict[str, str]], cancel: threading.Event | None = None,
             tools: list[dict] | None = None, live: str | os.PathLike | None = None) -> LLMResult:
        """`live`: a file the reasoning and reply stream into as they are generated (read it any time)."""
        if self.gate is None:
            return self._chat(messages, cancel, tools, live)
        with self.gate.use(self.model):
            return self._chat(messages, cancel, tools, live)

    def _chat(self, messages: list[dict[str, str]], cancel: threading.Event | None = None,
              tools: list[dict] | None = None, live: str | os.PathLike | None = None) -> LLMResult:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": True,
                                "stream_options": {"include_usage": True}, **self.extra}
        if tools:
            body["tools"] = tools
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", method="POST",
                                     data=json.dumps(body).encode("utf-8"), headers=headers)
        parts: list[str] = []
        thought: list[str] = []
        calls: dict[int, dict[str, str]] = {}   # streamed tool calls, assembled by index
        usage: dict[str, Any] = {}
        finish = None
        chunks = thinking = 0
        done = threading.Event()
        feed = _LiveFeed(live, self.model)
        try:
            with urllib.request.urlopen(req, timeout=self.idle_timeout) as r:
                if cancel is not None:  # a stop mid-generation closes the stream: the server stops too
                    threading.Thread(target=_close_on_cancel, args=(r, cancel, done), daemon=True).start()
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    evt = json.loads(data)
                    usage = evt.get("usage") or usage
                    for ch in evt.get("choices") or []:
                        delta = ch.get("delta") or {}
                        parts.append(delta.get("content") or "")
                        piece = delta.get("reasoning_content") or delta.get("reasoning")
                        if piece:
                            thinking += 1
                            thought.append(piece if isinstance(piece, str) else "")
                            feed.write("thinking", thought[-1])
                        if delta.get("content"):
                            feed.write("reply", delta["content"])
                        for tc in delta.get("tool_calls") or []:
                            slot = calls.setdefault(tc.get("index", len(calls)), {"id": "", "name": "", "arguments": ""})
                            slot["id"] = tc.get("id") or slot["id"]
                            fn = tc.get("function") or {}
                            slot["name"] += fn.get("name") or ""
                            slot["arguments"] += fn.get("arguments") or ""
                            feed.write("tool call", (fn.get("name") or "") + (fn.get("arguments") or ""))
                        finish = ch.get("finish_reason") or finish
                    chunks += 1
                    if self.on_progress and chunks % 50 == 0:
                        self.on_progress(chunks, thinking)
        except Exception as e:
            if cancel is not None and cancel.is_set():
                raise Cancelled("stopped by the user while the model was replying") from None
            if isinstance(e, urllib.error.HTTPError):
                raise LLMError(f"{self.name}: HTTP {e.code} {e.read().decode('utf-8', 'replace')[:300]}") from e
            if isinstance(e, urllib.error.URLError):
                raise LLMError(f"{self.name}: unreachable ({e.reason})") from e
            if isinstance(e, TimeoutError):
                raise LLMError(f"{self.name}: no tokens for {self.idle_timeout:.0f}s") from e
            raise
        finally:
            done.set()
            feed.close(f"{thinking} thinking chunks")
        if cancel is not None and cancel.is_set():
            raise Cancelled("stopped by the user while the model was replying")
        raw = "".join(parts)
        inline = "".join(m.group(0) for m in THINK_RE.finditer(raw))   # models that think inside <think>
        text = THINK_RE.sub("", raw).strip()
        reasoning = "".join(thought) or inline.replace("<think>", "").replace("</think>", "").strip()
        if finish == "length":
            text += (f"\n[output cut off at max_tokens: ~{thinking} thinking + "
                     f"~{chunks - thinking} answer tokens]")
        counted = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        return LLMResult(text=text, input_tokens=int(usage.get("prompt_tokens") or 0),
                         output_tokens=int(usage.get("completion_tokens") or 0), model=self.model,
                         thinking_tokens=int(counted if counted is not None else thinking), reasoning=reasoning,
                         tool_calls=[{**c, "id": c["id"] or f"call_{i}"} for i, c in sorted(calls.items())])


class AnthropicLLM:
    def __init__(self, model: str = "claude-opus-5", max_tokens: int = 16000,
                 effort: str | None = None, thinking: bool = True, fallbacks: str | None = "default"):
        import anthropic  # lazy: only needed when a persona uses Claude

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.thinking = thinking
        self.fallbacks = fallbacks
        self.name = model

    def chat(self, messages: list[dict[str, str]], cancel: threading.Event | None = None) -> LLMResult:
        if cancel is not None and cancel.is_set():
            raise Cancelled("stopped by the user")
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens, "messages": convo}
        if system:
            kwargs["system"] = system
        if self.thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if self.fallbacks:
            response = self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks=self.fallbacks, **kwargs)
        else:
            response = self.client.messages.create(**kwargs)
        if response.stop_reason == "refusal":
            return LLMResult(text="[the model declined this request]", model=response.model)
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return LLMResult(text=text, input_tokens=response.usage.input_tokens,
                         output_tokens=response.usage.output_tokens, model=response.model)


class WithFallback:
    """Try the primary model; on a provider error (down, no credentials) use the fallback."""

    def __init__(self, primary_factory, fallback: LLM):
        self._primary_factory = primary_factory
        self._primary: LLM | None = None
        self.fallback = fallback
        self.name = f"(primary)->{fallback.name}"

    def chat(self, messages: list[dict[str, str]], cancel: threading.Event | None = None) -> LLMResult:
        if self._primary is None and self._primary_factory is not None:
            try:
                self._primary = self._primary_factory()
                self.name = self._primary.name
            except Exception as e:  # noqa: BLE001 - missing SDK/credentials: don't retry every call
                self._primary_factory = None
                self.name = f"{self.fallback.name} (fallback: {type(e).__name__}: {e})"[:160]
        if self._primary is None:
            return self.fallback.chat(messages, cancel)
        try:
            return self._primary.chat(messages, cancel)
        except Cancelled:
            raise
        except Exception as e:  # noqa: BLE001 - provider down/refused this call: fall through
            self.name = f"{self.fallback.name} (fallback: {type(e).__name__})"
            return self.fallback.chat(messages, cancel)


def make_llm(model_cfg: dict[str, Any], providers: dict[str, dict[str, Any]]) -> LLM:
    def build(cfg: dict[str, Any]) -> LLM:
        pname = cfg["provider"]
        prov = providers.get(pname)
        if prov is None:
            raise LLMError(f"unknown provider '{pname}'")
        if prov["type"] == "openai":
            key_env = prov.get("api_key_env")
            unlimited = cfg.get("think") == "unlimited"  # let a thinking model think as long as it needs
            llm = OpenAICompat(base_url=prov["base_url"], model=cfg["model"],
                               api_key=os.environ.get(key_env) if key_env else None,
                               temperature=cfg.get("temperature"),
                               max_tokens=None if unlimited else cfg.get("max_tokens", 8192),
                               extra=cfg.get("extra"),
                               gate=gate_for(prov["base_url"]) if prov.get("exclusive") else None)
            llm.unlimited_thinking = unlimited
            # workers get a real `python` tool (the model's own tool-call format) instead of fenced blocks
            llm.native_tools = bool(cfg.get("native_tools", prov.get("native_tools", False)))
            return llm
        if prov["type"] == "anthropic":
            return AnthropicLLM(model=cfg.get("model", "claude-opus-5"),
                                max_tokens=cfg.get("max_tokens", 16000), effort=cfg.get("effort"),
                                thinking=cfg.get("thinking", True),
                                fallbacks=cfg.get("fallbacks", prov.get("fallbacks")))
        raise LLMError(f"provider '{pname}' has unknown type '{prov['type']}'")

    if model_cfg.get("fallback"):
        return WithFallback(lambda: build(model_cfg), build(model_cfg["fallback"]))
    return build(model_cfg)
