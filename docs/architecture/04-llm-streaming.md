# 04 · Model calls

A worker step calls `Worker._llm` (`worker.py:1021`). That goes through `Metered`
(`usage.py:132`), then the provider client, which for LM Studio also passes the one-model gate
(`lmstudio.py:69`). The reply streams in, and two side threads can cut it short.

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
  participant CT as cancel thread
  participant F as thinking/step-NN.txt

  W->>M: chat(messages, tools=[python, final], cancel, live, watch, watch_every, watch_after)
  M->>C: forwards only the kwargs the client accepts
  C->>G: use(model)
  alt another model is loaded
    Note over G: wait for in-flight requests on it<br/>and for its linger (8s) to pass, max 180s
    G->>S: GET /api/v1/models, POST unload (each other LLM), POST load {model}
    G-->>W: on_swap → event swap + usage row kind swap
  end
  C->>S: POST /v1/chat/completions stream=true
  Note over C: first-chunk timeout = min(idle 180s, 30s + prompt tokens / 400)
  C->>CT: start _close_on_cancel
  C->>WT: start _watch_thinking
  loop each SSE chunk
    S-->>C: delta: reasoning_content | content | tool_calls
    C->>F: append section thinking / reply / tool call
    C-->>W: on_progress every 50 chunks → event progress
    WT->>WT: every 1s: check reply/reasoning size
  end
  alt watcher says stop
    WT->>C: close response (interrupted = True)
  else user pressed Stop
    CT->>C: close response → Cancelled
  end
  C-->>M: LLMResult text, reasoning, tool_calls, tokens, interrupted
  M->>M: record kind llm/learn, tokens, thinking, seconds → usage.db
  M-->>W: LLMResult
```

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
  F -- yes --> G["Jev thinking_watch<br/>orchestrator.py:638"]
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
(`orchestrator.py:208`). Embedding models are never unloaded.

## Metering (`Metered`, `usage.py:132`)

Every client from `Harness._llm` is wrapped. After each call it records one row:
`kind` (`llm`, or `learn` on the learning thread), `model` (the loaded one), `persona`, `task`,
`tier`, `input`, `output`, `thinking` tokens and `seconds`. It takes these from the calling
thread's `_where()` tags. See [10](10-observability.md).
