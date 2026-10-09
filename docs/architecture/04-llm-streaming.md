# 04 · Model calls

A worker step calls `Worker._llm` (`worker.py:1021`). That goes through `Metered`
(`usage.py:132`), then the provider client, which for LM Studio also passes the one-model gate
(`lmstudio.py:69`). The reply streams in on a reader thread while the calling thread waits for it and can stop waiting at
any moment (a Stop, a message from the user, the watcher), then hangs up: see [16 Interjection](16-interjection.md).

## The call chain

```mermaid
sequenceDiagram
  autonumber
  participant W as Worker._llm
  participant M as Metered
  participant C as OpenAICompat.chat
  participant G as ExclusiveModels gate
  participant S as LM Studio
  participant WT as watch thread
  participant RT as reader thread
  participant F as thinking/step-NN.txt

  W->>M: chat(messages, tools=[python, final], cancel, live, watch, watch_every, watch_after, interrupt)
  M->>C: forwards only the kwargs the client accepts
  C->>G: use(model)
  alt another model is loaded
    Note over G: wait for in-flight requests on it<br/>and for its linger (8s) to pass, max 180s
    G->>S: GET /api/v1/models, POST unload (each other LLM), POST load {model}
    G-->>W: on_swap → event swap + usage row kind swap
  end
  C->>RT: start read(): POST /v1/chat/completions stream=true, then the SSE loop
  Note over C: first-chunk timeout = min(idle 180s, 30s + prompt tokens / 400)
  C->>WT: start _watch_thinking
  loop each SSE chunk (on the reader thread)
    S-->>RT: delta: reasoning_content | content | tool_calls
    RT->>F: append section thinking / reply / tool call
    RT-->>W: on_progress every 50 chunks → event progress
    WT->>WT: every 1s: check reply/reasoning size
  end
  Note over C: meanwhile the caller waits in 0.1s steps for the reader, a Stop, an interrupt or the watcher
  alt watcher says stop
    WT->>C: stop flag → C hangs up (interrupted = True)
  else user pressed Stop
    C->>S: hang up (socket shutdown) → Cancelled
  else the user sent a message (interrupt event)
    C->>S: hang up → the partial reply, redirected = True
  end
  C-->>M: LLMResult text, reasoning, tool_calls, tokens, interrupted, redirected
  M->>M: record kind llm/learn, tokens, thinking, seconds → usage.db
  M-->>W: LLMResult
```

## Why the reply is read on its own thread

A model still reading its prompt sends nothing for a long time (LM Studio sends the response headers at once and the
first chunk when the prompt is read: ~22 s for 6k tokens, measured). A read that waits for the server cannot be woken
from another thread on Windows: neither `response.close()` nor `socket.shutdown()` ends it (and `close()` blocks on
the lock the reading thread holds). Cutting the stream from a helper thread therefore worked only once the server next
spoke: Stop during prompt reading waited for the first token, and a correction sent 9.7 s into a step landed at 28.5 s.
So `OpenAICompat._chat` reads on a thread of its own and the caller polls; to cut, it stops waiting, drops the
connection with `shutdown()` (the server sees it and cancels the task) and leaves the reader to end when the server
closes its side. A reader abandoned this way never touches the results again (`stopped`).

## The watcher thread (`_watch_thinking`, `llm.py:100`)

While the model hasn't started acting yet (no tool call, no ```` ``` ```` in the reply), the
watcher reads what it has written so far about once a second:

```mermaid
flowchart TD
  T["every 1s"] --> A{"tool call streaming?"}
  A -- yes --> R{"runaway call?<br/>≥ 20 template tokens,<br/>> 200k chars, or > 12k chars<br/>that compress below 0.035"}
  R -- yes --> STOP["close stream<br/>(keep what came)"]
  R -- no --> T
  A -- no --> B{"code fence<br/>in the reply?"}
  B -- yes --> END["stop watching:<br/>model is acting"]
  B -- no --> C{"text ≥ watch_after (6000)<br/>and ≥ watch_every (4000) new chars?"}
  C -- no --> T
  C -- yes --> D["Worker.watch(text, answering)<br/>worker.py:1044"]
  D --> E{"repeating(text)?<br/>a 3+ word line 8× in last 4k chars<br/>worker.py:93"}
  E -- yes --> STOP2["stop: reason repeating"]
  E -- no --> F{"code has run this task,<br/>thinking watch on,<br/>not writing the final reply?"}
  F -- no --> T
  F -- yes --> G["Jev thinking_watch<br/>orchestrator.py:646"]
  G -- "testable / searching" --> STOP3["stop: pause the think"]
  G -- none --> T
```

The watcher and the thinking watch together make the step return `interrupted=True`. The worker
then keeps the reasoning, trims the reply, and adds a pause message saying *why*: "you were
working out by hand something code does exactly…", "trying one theory after another…", or "you
were repeating yourself" (`worker.py:585-619`). After two ignored pauses in a row the next check
waits twice as long (`watch_after × 2`, `worker.py:1054`). The Jev thinking watch is off by
default (`limits.thinking_watch: 0`); the repetition check always runs.

## Failure handling

| Failure | Where | What happens |
|---|---|---|
| Nothing streams within the first-chunk limit | `llm.py:205` | `LLMError "no tokens for Ns (nothing streamed)"` |
| A gap between chunks > `idle_timeout` (180s) | `llm.py:205` | `LLMError "no tokens for 180s"` |
| Either of the above | `worker.py:1057` | the step is retried up to 2 more times (`SILENT_RETRIES`); sampling differs on a retry |
| HTTP error / unreachable | `llm.py:205` | `LLMError`: the task fails (`stopped=error`, resumable) |
| `finish_reason: length` | `llm.py:205` | "[output cut off at max_tokens…]" appended → the worker is told to use smaller steps |
| Template tokens leaked into call args | `llm.py:70` | cut at the first leaked token (`clean_call_args`) |
| `<think>…</think>` inline | `llm.py:205` | moved out of the text into `reasoning` |
| Primary provider down | `llm.py:357` | `WithFallback` uses the persona's `fallback:` model |
| Claude refuses | `llm.py:333` | "[the model declined this request]"; server-side fallbacks when configured |

## LM Studio: one model at a time (`lmstudio.py`)

Two big models loaded at once spill out of VRAM, and both slow to a crawl. One `ExclusiveModels`
gate per server URL (`gate_for`, `lmstudio.py:30`) is shared by every persona and tier.

```mermaid
stateDiagram-v2
  [*] --> Idle
  Idle --> Serving: use(model), same model → inflight += 1
  Serving --> Serving: parallel requests for the same model
  Serving --> Lingering: last request done (last_used = now)
  Lingering --> Serving: same model within linger (8s)
  Lingering --> Swapping: other model waited ≥ linger<br/>or max_wait (180s) passed
  Idle --> Swapping: other model requested
  Swapping --> Serving: unload every other LLM, then load (12–23s)
  Lingering --> Rechecking: idle ≥ recheck (30s)
  Rechecking --> Swapping: forget what it believed is loaded,<br/>ask LM Studio again (a no-op swap if it's still ours)
```

`current` is only this process's belief. Another process on the same LM Studio (refine's test runs,
the dashboard next to a CLI) may have swapped models meanwhile, so after 30 idle seconds the gate
asks LM Studio again (`lmstudio.py:72`).

Swaps show up as the `swap` event (dashboard timeline markers) and as usage rows of kind `swap`
(`orchestrator.py:209`). Embedding models are never unloaded.

The same `GET /api/v1/models` says what context each loaded model has (`loaded_instances[].config.context_length`).
`ExclusiveModels.model_info(model)` reads it for the worker, which asks before every step to know where compaction
starts: a 3 s timeout, never under the swap lock, an answer reused for 10 s, and the last answer kept for 10 minutes if
LM Studio cannot be reached. [17 Context length](17-context-length.md).

## Metering (`Metered`, `usage.py:132`)

Every client from `Harness._llm` is wrapped. After each call it records one row:
`kind` (`llm`, or `learn` on the learning thread), `model` (the loaded one), `persona`, `task`,
`tier`, `input`, `output`, `thinking` tokens, `seconds` and `decode_seconds` (the tokens per second shown in a turn's footer,
[10](10-observability.md#usage-tracking-usagepy)). It takes these from the calling
thread's `_where()` tags. See [10](10-observability.md).
