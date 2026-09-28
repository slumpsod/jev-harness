"""Chat model providers. Every persona picks one in its `model:` block.

- type openai:    any OpenAI-compatible /v1/chat/completions server (LM Studio, llama.cpp,
                  vLLM, OpenAI). stdlib HTTP, no SDK needed.
- type anthropic: Claude via the official `anthropic` SDK (imported lazily, so the
                  harness runs without it when no persona uses Claude).

Messages are plain [{"role": "system"|"user"|"assistant", "content": str}].
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .lmstudio import gate_for

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class LLM(Protocol):
    name: str

    def chat(self, messages: list[dict[str, str]]) -> LLMResult: ...


class LLMError(RuntimeError):
    pass


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

    def chat(self, messages: list[dict[str, str]]) -> LLMResult:
        if self.gate is None:
            return self._chat(messages)
        with self.gate.use(self.model):
            return self._chat(messages)

    def _chat(self, messages: list[dict[str, str]]) -> LLMResult:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": True,
                                "stream_options": {"include_usage": True}, **self.extra}
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
        usage: dict[str, Any] = {}
        finish = None
        chunks = thinking = 0
        try:
            with urllib.request.urlopen(req, timeout=self.idle_timeout) as r:
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
                        if delta.get("reasoning_content") or delta.get("reasoning"):
                            thinking += 1
                        finish = ch.get("finish_reason") or finish
                    chunks += 1
                    if self.on_progress and chunks % 50 == 0:
                        self.on_progress(chunks, thinking)
        except urllib.error.HTTPError as e:
            raise LLMError(f"{self.name}: HTTP {e.code} {e.read().decode('utf-8', 'replace')[:300]}") from e
        except urllib.error.URLError as e:
            raise LLMError(f"{self.name}: unreachable ({e.reason})") from e
        except TimeoutError as e:
            raise LLMError(f"{self.name}: no tokens for {self.idle_timeout:.0f}s") from e
        text = THINK_RE.sub("", "".join(parts)).strip()
        if finish == "length":
            text += (f"\n[output cut off at max_tokens: ~{thinking} thinking + "
                     f"~{chunks - thinking} answer tokens]")
        return LLMResult(text=text, input_tokens=int(usage.get("prompt_tokens") or 0),
                         output_tokens=int(usage.get("completion_tokens") or 0), model=self.model)


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

    def chat(self, messages: list[dict[str, str]]) -> LLMResult:
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

    def chat(self, messages: list[dict[str, str]]) -> LLMResult:
        if self._primary is None and self._primary_factory is not None:
            try:
                self._primary = self._primary_factory()
                self.name = self._primary.name
            except Exception as e:  # noqa: BLE001 - missing SDK/credentials: don't retry every call
                self._primary_factory = None
                self.name = f"{self.fallback.name} (fallback: {type(e).__name__}: {e})"[:160]
        if self._primary is None:
            return self.fallback.chat(messages)
        try:
            return self._primary.chat(messages)
        except Exception as e:  # noqa: BLE001 - provider down/refused this call: fall through
            self.name = f"{self.fallback.name} (fallback: {type(e).__name__})"
            return self.fallback.chat(messages)


def make_llm(model_cfg: dict[str, Any], providers: dict[str, dict[str, Any]]) -> LLM:
    def build(cfg: dict[str, Any]) -> LLM:
        pname = cfg["provider"]
        prov = providers.get(pname)
        if prov is None:
            raise LLMError(f"unknown provider '{pname}'")
        if prov["type"] == "openai":
            key_env = prov.get("api_key_env")
            return OpenAICompat(base_url=prov["base_url"], model=cfg["model"],
                                api_key=os.environ.get(key_env) if key_env else None,
                                temperature=cfg.get("temperature"), max_tokens=cfg.get("max_tokens", 8192),
                                extra=cfg.get("extra"),
                                gate=gate_for(prov["base_url"]) if prov.get("exclusive") else None)
        if prov["type"] == "anthropic":
            return AnthropicLLM(model=cfg.get("model", "claude-opus-5"),
                                max_tokens=cfg.get("max_tokens", 16000), effort=cfg.get("effort"),
                                thinking=cfg.get("thinking", True),
                                fallbacks=cfg.get("fallbacks", prov.get("fallbacks")))
        raise LLMError(f"provider '{pname}' has unknown type '{prov['type']}'")

    if model_cfg.get("fallback"):
        return WithFallback(lambda: build(model_cfg), build(model_cfg["fallback"]))
    return build(model_cfg)
