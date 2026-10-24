"""Chat model providers. Every persona picks one in its `model:` block.

- type openai:    any OpenAI-compatible /v1/chat/completions server (LM Studio, llama.cpp,
                  vLLM, OpenAI). stdlib HTTP, no SDK needed.
- type anthropic: Claude via the official `anthropic` SDK (imported lazily, so the
                  harness runs without it when no persona uses Claude).

Messages are plain [{"role": "system"|"user"|"assistant", "content": str}].
"""
from __future__ import annotations

import inspect
import json
import socket
import threading
import os
import time
from pathlib import Path
import re
import zlib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .context import Window
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
    interrupted: bool = False   # a `watch` callback stopped it mid-thinking (see OpenAICompat.chat)
    redirected: bool = False    # the `interrupt` event cut it: the user sent a message that changes the task
    decode_seconds: float = 0.0   # time spent producing the tokens, the prompt's reading excluded (0: not known)


class LLM(Protocol):
    name: str

    def chat(self, messages: list[dict[str, str]]) -> LLMResult: ...


class LLMError(RuntimeError):
    pass


# A chat template's own control tokens leaking into a tool call's arguments: live (Gemma 4 in LM Studio),
# after its final call it wrote "<tool_call|>" and then "<|channel>thought\n<channel|>" for 437 KB / 17 min.
TEMPLATE_TOKEN_RE = re.compile(r"<\|[A-Za-z_]{2,24}\|?>|<[A-Za-z_]{2,24}\|>")
RUNAWAY_TOKENS = 20        # this many template tokens in a call's arguments: stop the stream
MAX_CALL_CHARS = 200_000   # ...or arguments this long
# ...or a long call whose end is one short unit repeated (live, Gemma 4: "..._content_content_content" for
# 200 KB / 13 min). zlib ratio of the last 6000 chars: real calls from the ARC runs >= 0.109 (a grid pasted in
# code), the runaways 0.013-0.022; a pasted 30x30 all-zero grid is 0.019 but only ~2700 chars, hence the length.
RUNAWAY_MIN_CHARS = 12_000
RUNAWAY_RATIO = 0.035      # (a deliberately periodic 20 KB table of code: 0.05)


def _packs(text: str) -> float:
    """Compressed size / size: near 0 for one short unit repeated."""
    data = text.encode("utf-8", "replace")
    return len(zlib.compress(data, 6)) / max(1, len(data))


def clean_call_args(arguments: str) -> str:
    """A tool call's arguments up to the first leaked template token (what follows is the runaway)."""
    m = TEMPLATE_TOKEN_RE.search(arguments or "")
    return arguments[:m.start()] if m else arguments


FIRST_CHUNK_BASE = 30.0  # seconds a reply may take to start streaming, plus the prompt at PREFILL_TPS
PREFILL_TPS = 400.0       # conservative prompt-reading speed (tokens/s) for a local model


def _set_read_timeout(response, seconds: float) -> None:
    """Change a streaming response's socket read timeout mid-stream."""
    try:
        response.fp.raw._sock.settimeout(seconds)
    except AttributeError:
        pass


def _abort(response) -> None:
    """Drop a streaming response's connection from another thread: the server sees it and stops generating. Only
    shutdown() does it safely: response.close() takes the lock the reading thread holds while it waits for the
    server, so the caller would wait as long as the read does (measured: 30 s against a silent server), and
    socket.close() waits for the buffered reader to let go of the socket. The reading thread is left to finish
    on its own (the server closing its end ends it): the caller has stopped waiting for it (see _chat)."""
    try:
        response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
    except Exception:  # noqa: BLE001 - not a plain socket, or already gone
        pass


def _watch_thinking(response, watch, thought, parts, calls, stop, done, every: int, after: int) -> None:
    """Side thread: until the model starts acting (a tool call or a code block), show `watch` what it
    has written every `every` new chars (from `after` on) - its reasoning, and its reply text too
    (some models, e.g. Gemma 4 in LM Studio, reason in the reply: live, a 108 KB reply looped on
    "I'll call final(test_output)"). `watch(text, answering)` -> stop? closes the stream; `answering`:
    the model reasoned separately and is now writing its reply."""
    checked = 0
    while not done.wait(1.0):
        reply, reasoning = "".join(parts), "".join(thought)
        if calls:   # acting - unless the call itself runs away (template tokens looping inside it)
            args = "".join(c["arguments"] for c in list(calls.values()))
            if (len(TEMPLATE_TOKEN_RE.findall(args)) >= RUNAWAY_TOKENS or len(args) > MAX_CALL_CHARS
                    or (len(args) > RUNAWAY_MIN_CHARS and _packs(args[-6000:]) < RUNAWAY_RATIO)):
                stop.set()
                return
            continue
        if "```" in reply:
            return   # the model is acting: nothing to interrupt
        text = reasoning + ("\n" + reply if reply.strip() else "")
        if len(text) < after or len(text) - checked < every:
            continue
        checked = len(text)
        try:
            hit = watch(text, bool(reply.strip() and reasoning.strip()))
        except Exception:  # noqa: BLE001 - a failed check never stops the model
            hit = False
        if hit and not done.is_set():
            stop.set()
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
                 gate: Any = None, context_length: int | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.idle_timeout = idle_timeout
        self.extra = extra or {}
        self.name = f"{model}@{self.base_url}"
        self.on_progress = None  # callable(chunks, thinking_chunks, tokens_per_second) set by the UI, optional: every 50 chunks,
                                 # and once more with a rate of 0 when a call that reported any ends
        self.gate = gate         # lmstudio.ExclusiveModels: one model resident at a time
        self.context_length = context_length   # config: what to assume the window is (wins over asking the server)
        self.unlimited_thinking = False  # think: unlimited -> no cap, and workers skip the first-step nudge
        self.native_tools = False        # provider/persona native_tools: true -> tools= in chat()

    def window(self) -> Window | None:
        """The model's context window (jevh/context.py): what a llama.cpp server was started with, else what config
        says, else what LM Studio loaded it with. None when nothing can tell (detection off, the server unreachable)."""
        if self.gate is None:
            from . import context as ctxwin
            props = ctxwin.server_props(self.base_url) if ctxwin.DETECT["on"] else None
            if props:
                return Window(props["context"], "llama.cpp", True, None, props.get("slots"))
        if self.context_length:
            return Window(int(self.context_length), "config")
        info = self.gate.model_info(self.model) if self.gate is not None else None
        if info is None:
            return None
        return Window(info["context"], "lmstudio", info["loaded"], info["max"], info["parallel"])

    def chat(self, messages: list[dict[str, str]], cancel: threading.Event | None = None,
             tools: list[dict] | None = None, live: str | os.PathLike | None = None,
             watch: Callable[[str], bool] | None = None, watch_every: int = 4000,
             watch_after: int = 6000, interrupt: threading.Event | None = None,
             body_extra: dict[str, Any] | None = None) -> LLMResult:
        """`live`: a file the reasoning and reply stream into as they are generated (read it any time).
        `watch(reasoning_so_far) -> stop?`: while the reply is still pure thinking, called from a
        side thread every `watch_every` new chars once there are `watch_after`; True closes the
        stream and returns what was thought so far (interrupted=True).
        `interrupt`: an event that cuts the reply the same way whenever it is set (redirected=True): the
        server stops generating, and what was streamed so far comes back instead of a Cancelled.
        `body_extra`: fields laid over this one request's body (a smaller max_tokens for a quick side question)."""
        if self.gate is None:
            return self._chat(messages, cancel, tools, live, watch, watch_every, watch_after, interrupt, body_extra)
        with self.gate.use(self.model, cancel):   # (a Stop ends the wait for another model's turn on the GPU too)
            return self._chat(messages, cancel, tools, live, watch, watch_every, watch_after, interrupt, body_extra)

    def _chat(self, messages: list[dict[str, str]], cancel: threading.Event | None = None,
              tools: list[dict] | None = None, live: str | os.PathLike | None = None,
              watch: Callable[[str], bool] | None = None, watch_every: int = 4000,
              watch_after: int = 6000, interrupt: threading.Event | None = None,
              body_extra: dict[str, Any] | None = None) -> LLMResult:
        if interrupt is not None and interrupt.is_set():   # the redirect came before the request went out
            return LLMResult(text="", model=self.model, interrupted=True, redirected=True)
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": True,
                                "stream_options": {"include_usage": True}, **self.extra}
        if tools:
            body["tools"] = tools
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        if body_extra:
            body.update(body_extra)
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", method="POST",
                                     data=json.dumps(body).encode("utf-8"), headers=headers)
        parts: list[str] = []
        thought: list[str] = []
        calls: dict[int, dict[str, str]] = {}   # streamed tool calls, assembled by index
        st: dict[str, Any] = {"usage": {}, "finish": None, "chunks": 0, "thinking": 0, "streaming": False,
                              "r": None, "err": None, "t0": None, "t1": None, "timings": {}, "mark": None, "reported": False}
        finished = threading.Event()     # the reader thread is over
        ended = threading.Event()        # this call is over (the reader finished, or the caller stopped waiting for it)
        stopped = threading.Event()      # the caller stopped waiting: the reader must leave the results alone
        stop_watch = threading.Event()   # set by the watcher: cut the stream, keep what we have
        stop_interrupt = threading.Event()   # set when `interrupt` cut the stream for a redirect
        feed = _LiveFeed(live, self.model)
        # Until the first chunk, a shorter limit scaled to the prompt (time to read it in): live, Gemma 4 in LM
        # Studio sometimes generated for minutes (12k tokens once) without streaming any of it, and waiting
        # the full idle_timeout never helped. After the first chunk, idle_timeout bounds each gap.
        first_limit = min(self.idle_timeout, FIRST_CHUNK_BASE + len(req.data) / 3.2 / PREFILL_TPS)

        def read() -> None:
            """The stream, read on a thread of its own. A read that waits for a server still reading its prompt
            (LM Studio: nothing for ~22 s on 6k tokens) cannot be woken from another thread - on Windows neither
            close() nor shutdown() ends it - so the caller waits on this thread instead and can stop waiting at
            any moment: a stop, a redirect or the thinking watch. Dropping the connection then stops the server."""
            try:
                with urllib.request.urlopen(req, timeout=first_limit) as r:
                    st["r"] = r
                    if stopped.is_set():   # cut while still connecting: nothing is waiting for this any more
                        _abort(r)
                        return
                    for raw in r:
                        if stopped.is_set():
                            return
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        if not st["streaming"]:
                            st["streaming"] = True
                            _set_read_timeout(r, self.idle_timeout)
                        try:
                            evt = json.loads(data)
                        except ValueError:   # (raw JSONDecodeError before: the worker failed on it as a crash)
                            raise LLMError(f"{self.name}: the server sent a stream chunk that is not JSON: "
                                           f"{data[:200]!r}") from None
                        if not isinstance(evt, dict):
                            continue
                        if evt.get("error"):
                            # a server that fails mid-reply says so in the stream (llama.cpp, LM Studio: a context
                            # overflow, a crashed slot). Ignored, it read as an empty reply, and the worker was asked
                            # to continue, step after step, until the step limit
                            err = evt["error"]
                            raise LLMError(f"{self.name}: the server reported an error: "
                                           f"{str(err.get('message') if isinstance(err, dict) else err)[:300]}")
                        st["usage"] = evt.get("usage") or st["usage"]
                        st["timings"] = evt.get("timings") or st["timings"]   # (llama-server: its own count of the decode time)
                        for ch in evt.get("choices") or []:
                            delta = ch.get("delta") or {}
                            if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or delta.get("tool_calls"):
                                st["t1"] = time.monotonic()
                                st["t0"] = st["t0"] or st["t1"]
                            parts.append(delta.get("content") or "")
                            piece = delta.get("reasoning_content") or delta.get("reasoning")
                            if piece:
                                st["thinking"] += 1
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
                            st["finish"] = ch.get("finish_reason") or st["finish"]
                        st["chunks"] += 1
                        if self.on_progress and st["chunks"] % 50 == 0:
                            now = time.monotonic()   # the speed of the last 50 chunks (a chunk is about a token)
                            since, was = st["mark"] or (st["t0"] or now, 0)
                            st["mark"], st["reported"] = (now, st["chunks"]), True
                            self.on_progress(st["chunks"], st["thinking"], (st["chunks"] - was) / max(now - since, 1e-3))
            except BaseException as e:  # noqa: BLE001 - handed to the caller, which maps it to the harness's errors
                st["err"] = e
            finally:
                finished.set()

        threading.Thread(target=read, daemon=True, name="llm-stream").start()
        if watch is not None:
            threading.Thread(target=_watch_thinking, daemon=True, args=(
                None, watch, thought, parts, calls, stop_watch, ended, watch_every, watch_after)).start()
        cut = ""
        while not finished.wait(0.1):
            if cancel is not None and cancel.is_set():
                cut = "cancel"
            elif interrupt is not None and interrupt.is_set():
                stop_interrupt.set()
                cut = "interrupt"
            elif stop_watch.is_set():
                cut = "watch"
            if cut:
                break
        ended.set()   # (the thinking watch must not go on calling Jev for a reader nobody waits for)
        if cut:
            stopped.set()
            if st["r"] is not None:   # dropping the connection is what stops the server's generation
                _abort(st["r"])
        feed.close(f"{st['thinking']} thinking chunks")
        if self.on_progress and st["reported"]:   # (the call is over: nothing is being written now)
            self.on_progress(st["chunks"], st["thinking"], 0.0)
        try:
            if cut == "cancel":
                raise Cancelled("stopped by the user while the model was replying")
            if st["err"] is not None and not cut:
                raise st["err"]
        except Cancelled:
            raise
        except Exception as e:
            if cancel is not None and cancel.is_set():
                raise Cancelled("stopped by the user while the model was replying") from None
            if isinstance(e, urllib.error.HTTPError):
                raise LLMError(f"{self.name}: HTTP {e.code} {e.read().decode('utf-8', 'replace')[:300]}") from e
            elif isinstance(e, urllib.error.URLError):
                raise LLMError(f"{self.name}: unreachable ({e.reason})") from e
            elif isinstance(e, TimeoutError):
                dump = os.environ.get("JEVH_DUMP_STALLS")   # debugging: keep the request that stalled
                if dump:
                    try:
                        Path(dump).mkdir(parents=True, exist_ok=True)
                        (Path(dump) / f"stall-{time.strftime('%H%M%S')}.json").write_text(
                            json.dumps(body), encoding="utf-8")
                    except OSError:
                        pass
                limit = self.idle_timeout if st["streaming"] else first_limit
                raise LLMError(f"{self.name}: no tokens for {limit:.0f}s"
                               + ("" if st["streaming"] else " (nothing streamed)")) from e
            else:
                raise
        if cancel is not None and cancel.is_set():
            raise Cancelled("stopped by the user while the model was replying")
        usage, finish, chunks, thinking = st["usage"], st["finish"], st["chunks"], st["thinking"]
        raw = "".join(parts)
        inline = "".join(m.group(0) for m in THINK_RE.finditer(raw))   # models that think inside <think>
        text = THINK_RE.sub("", raw).strip()
        reasoning = "".join(thought) or inline.replace("<think>", "").replace("</think>", "").strip()
        if finish == "length":
            text += (f"\n[output cut off at max_tokens: ~{thinking} thinking + "
                     f"~{chunks - thinking} answer tokens]")
        counted = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        # decode time: the server's own figure (llama.cpp's timings), else the span the tokens streamed in - but only a
        # stream that came token by token (a server that holds the reply back and sends it at once says nothing)
        n_out = int(usage.get("completion_tokens") or 0)
        span = st["t1"] - st["t0"] if st["t0"] and st["t1"] and not cut else 0.0
        decode = float((st["timings"] or {}).get("predicted_ms") or 0) / 1000 or (
            span if span >= 0.3 and chunks >= n_out * 0.5 else 0.0)
        return LLMResult(decode_seconds=round(max(decode, 0.0), 3), text=text, input_tokens=int(usage.get("prompt_tokens") or 0),
                         output_tokens=int(usage.get("completion_tokens") or 0), model=self.model,
                         thinking_tokens=int(counted if counted is not None else thinking), reasoning=reasoning,
                         tool_calls=[{**c, "id": c["id"] or f"call_{i}", "arguments": clean_call_args(c["arguments"])}
                                     for i, c in sorted(calls.items())],
                         interrupted=stop_watch.is_set() or stop_interrupt.is_set(),
                         redirected=stop_interrupt.is_set())


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

    def chat(self, messages: list[dict[str, str]], cancel: threading.Event | None = None,
             interrupt: threading.Event | None = None) -> LLMResult:
        if cancel is not None and cancel.is_set():
            raise Cancelled("stopped by the user")
        if interrupt is not None and interrupt.is_set():   # (a reply in progress is not cut: it finishes first)
            return LLMResult(text="", model=self.model, interrupted=True, redirected=True)
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

    @staticmethod
    def _call(llm: LLM, messages: list[dict[str, str]], cancel, interrupt) -> LLMResult:
        """`interrupt` goes only to a model that takes it (the others finish their reply first)."""
        try:
            takes = interrupt is not None and "interrupt" in inspect.signature(llm.chat).parameters
        except (TypeError, ValueError):
            takes = False
        return llm.chat(messages, cancel, interrupt=interrupt) if takes else llm.chat(messages, cancel)

    def _ensure_primary(self) -> None:
        if self._primary is None and self._primary_factory is not None:
            try:
                self._primary = self._primary_factory()
                self.name = self._primary.name
            except Exception as e:  # noqa: BLE001 - missing SDK/credentials: don't retry every call
                self._primary_factory = None
                self.name = f"{self.fallback.name} (fallback: {type(e).__name__}: {e})"[:160]

    def window(self) -> Window | None:
        """The smaller of the two models' windows: a prompt that fits one must fit the other, since a failed call is
        repeated on the fallback with the same messages. If either is not known yet (LM Studio has not loaded it),
        so is this one: the other's window says nothing about it."""
        self._ensure_primary()
        found = [w for w in (getattr(m, "window", lambda: None)() for m in (self._primary, self.fallback) if m is not None)
                 if w is not None]
        pending = [w for w in found if not w.tokens]
        return pending[0] if pending else min(found, key=lambda w: w.tokens, default=None)

    def chat(self, messages: list[dict[str, str]], cancel: threading.Event | None = None,
             interrupt: threading.Event | None = None) -> LLMResult:
        self._ensure_primary()
        if self._primary is None:
            return self._call(self.fallback, messages, cancel, interrupt)
        try:
            return self._call(self._primary, messages, cancel, interrupt)
        except Cancelled:
            raise
        except Exception as e:  # noqa: BLE001 - provider down/refused this call: fall through
            self.name = f"{self.fallback.name} (fallback: {type(e).__name__})"
            return self._call(self.fallback, messages, cancel, interrupt)


def see_image(llm: LLM, path: str, question: str) -> str:
    """Ask `llm` about an image file (a browser screenshot). Local vision models (Qwen 3.5+, Gemma 4
    in LM Studio) take OpenAI-style image parts; Claude takes its own image block."""
    import base64
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    kind = "image/jpeg" if str(path).lower().endswith((".jpg", ".jpeg")) else "image/png"
    target = getattr(llm, "_inner", llm)   # (the harness's usage meter, jevh/usage.Metered, wraps every worker model)
    target = getattr(target, "_primary", None) or target
    if isinstance(target, AnthropicLLM):
        content = [{"type": "image", "source": {"type": "base64", "media_type": kind, "data": data}},
                   {"type": "text", "text": question}]
    else:
        content = [{"type": "text", "text": question},
                   {"type": "image_url", "image_url": {"url": f"data:{kind};base64,{data}"}}]
    try:
        return llm.chat([{"role": "user", "content": content}]).text
    except Exception as e:  # noqa: BLE001 - a text-only model says so instead of failing the step
        return f"(could not look at the image: {str(e).splitlines()[0][:200]} - this model may not take images)"


# How much the worker's model thinks before it answers (config `thinking.level`, the Settings tab). auto sends nothing: the
# model's own default (Qwen3.8 on Strata: xhigh, its longest). The others go in each request as `reasoning_effort`, which
# Strata, vLLM, llama.cpp and OpenAI's reasoning models read (Strata: none, low, medium, high = xhigh); for a Claude model
# they set its effort, and none turns its thinking off.
THINKING_LEVELS = ("auto", "none", "low", "medium", "high")
_LOCAL_URL = re.compile(r"https?://(127\.0\.0\.1|localhost|\[::1\])[:/]")


def thinking_level(raw: dict[str, Any] | None) -> str:
    """The configured level (`thinking: {level: ...}`), "auto" when none or an unknown one is set."""
    t = (raw or {}).get("thinking")
    level = str((t.get("level") if isinstance(t, dict) else t) or "auto").strip().lower()
    return level if level in THINKING_LEVELS else "auto"


def with_thinking(block: dict[str, Any], level: str, providers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """`block` (a model config) with the thinking level laid over it. A level the block sets itself wins (a persona's
    `extra: {reasoning_effort: ...}` or `effort:`): the setting is for every model that has none of its own."""
    if level in ("", "auto") or not isinstance(block, dict):
        return block
    prov = providers.get(block.get("provider")) or {}
    out = dict(block)
    if prov.get("type") == "anthropic":
        if level == "none":
            out.setdefault("thinking", False)
        elif not out.get("effort"):
            out["effort"] = level
        return out
    if prov.get("type") != "openai":
        return block
    extra = dict(out.get("extra") or {})
    ctk = dict(extra.get("chat_template_kwargs") or {})
    if "reasoning_effort" in extra or "reasoning" in extra or "enable_thinking" in ctk or "reasoning_effort" in ctk:
        return block
    extra["reasoning_effort"] = level
    if level == "none" and _LOCAL_URL.match(str(prov.get("base_url", ""))):
        extra["chat_template_kwargs"] = {**ctk, "enable_thinking": False}   # (llama.cpp and LM Studio read this one)
    out["extra"] = extra
    return out


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
                               gate=gate_for(prov["base_url"]) if prov.get("exclusive") else None,
                               context_length=cfg.get("context_length", prov.get("context_length")))
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
