# 01 · Turn and task setup

A **turn** is one user message. `Harness.turn` (`orchestrator.py:238`) sets up cancellation and
the thread-local tags, then `_turn` (`orchestrator.py:249`) does the work. Every task in the turn,
whether the root or a hand-off, goes through `run_task` (`orchestrator.py:421`). `run_task` asks
Jev five questions and assembles what the worker will see.

## The turn loop

```mermaid
sequenceDiagram
  autonumber
  participant UI as CLI / Hub
  participant H as Harness._turn
  participant Mem as Memory (sqlite)
  participant J as Jev
  participant B as Board
  participant RT as run_task (root)
  participant LR as Learner thread

  UI->>H: turn(message, force=pin, resume)
  Note over H: learner.cancel.set() - learning yields the model<br/>new cancel Event, new Board(max_depth, max_tasks)
  H->>Mem: recent_text(session, 2)
  Mem-->>H: last 2 turns of this session as chunks
  alt persona pinned (force)
    H->>J: follow_up(message, recent) - step follow_up
    J-->>H: continues, replies nouls
  else routed
    H->>J: route_persona(message, personas, current, recent) - step route
    J-->>H: RouteDecision persona, confidence, probabilities, follows
  end
  H-->>UI: event route
  H->>B: open(None, "user", persona, message) - root Task
  opt route.follows
    H->>Mem: session_turns(session, follow_turns=2)
    Mem-->>H: user, assistant, notes per turn
    Note over H: root.history = those turns<br/>(real chat turns, not via the memory gate)
  end
  H->>RT: run_task(root, goal=message, allow_collab=True, resume)
  RT-->>H: WorkerResult answer, artifacts, transcript, stopped, llm
  alt root.status == done
    H->>Mem: append(persona, message, answer, session, turn_notes(result))
    opt learning and stopped in final/plain
      H->>LR: _learn(persona, message, result) - new thread
    end
  else cancelled
    H->>Mem: append(persona, message, STOPPED_NOTE)
  end
  H-->>UI: TurnResult persona, answer, artifacts, tasks, trace, seconds, usage, checkpoint
```

| Step | Code | Data |
|---|---|---|
| Recent context | `orchestrator.py:253`, `memory.py:132` | the session's last 2 turns, each clipped to 900 chars |
| Routing | `orchestrator.py:263`, `jev.py:165` | `{latest_request, recent_conversation}` → `RouteDecision` |
| Root task | `orchestrator.py:267`, `bus.py:91` | `Task(id, parent=None, requester="user", to, text)` |
| Follow-up history | `orchestrator.py:271`, `memory.py:137` | `[(user, assistant, notes)]` × `limits.follow_turns` |
| Memory write | `orchestrator.py:276`, `memory.py:111` | turn + `turn_notes()` (code run and outputs, 2k chars) |
| Learning | `orchestrator.py:278`, `:398` | runs on its own thread with the model that just ran the task |
| Result | `orchestrator.py:282` | `TurnResult` incl. `usage.summary(session, turn)` and a checkpoint path if resumable |

**Routing rule** (`jev.py:165-195`): if the message follows up (`continues ≥ 0.25` or
`replies ≥ 0.40`) and the Choice isn't confident (`< route 0.45`), stay with the current persona.
Otherwise take the pick if its confidence is `≥ 0.45`. Otherwise stay if `continues ≥ stick 0.55`.
Otherwise use `default_persona`. If Jev is down, `_safe` (`orchestrator.py:727`) falls back to the
current or default persona.

## Task setup: what `run_task` builds

```mermaid
flowchart TD
  T["Task<br/>text, inputs, history<br/>jev_text = history + latest"] --> S
  S["Jev: pick_skills<br/>steps skills, skills_round, skills_fits<br/>orchestrator.py:425"] --> MC
  MC["Memory.candidates<br/>own 20 + shared 6 + FTS search 10 + facts 6<br/>orchestrator.py:428"] --> MG
  MG["Jev: gate_context<br/>steps memory_rank, memory<br/>orchestrator.py:435"] --> CO
  CO{"allow_collab<br/>and persona.code?"} -- yes --> CL["Jev: collab<br/>step collab<br/>orchestrator.py:440"]
  CO -- "no (hand-offs)" --> RQ
  CL --> RQ["requestable =<br/>can_request + suggested<br/>orchestrator.py:442"]
  RQ --> TO["Jev: pick_tools<br/>steps tools, mcp_rank, mcp_fits<br/>orchestrator.py:444"]
  TO --> SYS["build_system()<br/>worker.py:414<br/>orchestrator.py:449"]
  TO --> CTX["ctx = request, memory,<br/>inputs, skills (+ project)<br/>orchestrator.py:451"]
  TO --> ENV["REPL env<br/>JEVH_BROWSER_*, JEVH_TASK_HOSTS,<br/>JEVH_PROJECT, JEVH_RULE_BLOCK<br/>orchestrator.py:453"]
  SYS --> TI["Jev: pick_tier<br/>step tier<br/>orchestrator.py:465"]
  TI --> LLM["_llm(persona, tier)<br/>cached Metered client<br/>orchestrator.py:694"]
  LLM --> NAT{"native_tools?"}
  NAT -- yes --> SYS2["rebuild system with<br/>native=True wording<br/>orchestrator.py:468"]
  NAT -- no --> HD
  SYS2 --> HD["handlers dict<br/>llm_query, jev, look, request,<br/>request_async, wait, mcp, recall,<br/>remember, ask_user,<br/>report_failure<br/>orchestrator.py:472"]
  HD --> WD["workdir =<br/>sessions/id/u{turn}t{task}-{persona}<br/>orchestrator.py:487"]
  WD --> WK["Worker(...)<br/>+ step_value, thinking_watch, escalate<br/>orchestrator.py:490-506"]
  WK --> RUN["worker.run(task.text, ctx, system,<br/>task.history, resume)<br/>orchestrator.py:511"]
  RUN --> UW["collect_unwaited hand-offs<br/>merge their artifacts<br/>orchestrator.py:530"]
  UW --> DONE["event done<br/>return WorkerResult"]
```

### The system prompt (`build_system`, `worker.py:414`)

Assembled in this order. Only the parts Jev picked are included:

1. `persona.system_prompt` + today's date (date only, so a local model's prompt cache holds all day).
2. `## Environment`: Python version, OS, the working folder, and which common packages import.
3. `## How you work` (`HOW_YOU_WORK`, `worker.py:334`): the REPL guide. Two wordings: native tool
   calls (`NATIVE_ACT`) or fenced ```` ```python ```` blocks (`TEXT_ACT`).
4. `REQUEST_GUIDE` + the roster of `requestable` personas, when there are any.
5. `## Tools in your REPL`: only the guides Jev picked (`tools/__init__.py` `TOOLS[name].guide`),
   with live notes (nango: which accounts are connected). The rest of the persona's tools are named
   in one line: "Also importable…".
6. `## MCP tools`: signatures of the MCP tools Jev picked.
7. Jev's collaborator suggestion line.
8. `## Skill: name`: the picked skill's body. Past 6,000 chars only the head goes in, and the full
   text sits in `ctx['skills'][name]`. Learned drafts are marked as unreviewed.

### What reaches the worker, and how

| Data | Channel | Size rule |
|---|---|---|
| The request | the task message | as is, plus `(In your REPL: …)` listing ctx keys and sizes (`describe_ctx`, `worker.py:398`) |
| Follow-up turns | real `user` / `assistant` messages before the task | 3,000 chars per reply, plus 2,500 chars of work notes |
| Memory snippets Jev kept | `ctx['memory']` in the REPL, not the prompt | ≤ `max_memory_chunks` (4) |
| Hand-off inputs | `ctx['inputs']` (artifact paths) | paths only |
| Skill full text | `ctx['skills']` | full |
| Project path | `ctx['project']` + `JEVH_PROJECT` env | only for personas with the `project` tool, and only while a project is open |

A plain chat persona (`code: false`) has no REPL. `Worker._chat` (`worker.py:1123`) puts the kept
memory and the inputs straight into the user message and makes one model call.

## Model tier and client (`_pick_tier`, `_llm`, `_model_cfg`)

```mermaid
flowchart LR
  A{"--model override?"} -- yes --> S1["strong<br/>(override model)"]
  A -- no --> B{"models.fast set<br/>and persona.fast_ok?"}
  B -- no --> S2["strong"]
  B -- yes --> C{"tier_mode<br/>fast / strong?"}
  C -- yes --> S3["forced tier"]
  C -- auto --> D{"follow-up to a turn<br/>that escalated?"}
  D -- yes --> S4["strong"]
  D -- no --> E["Jev pick_tier<br/>P(strong) ≥ 0.55 → strong<br/>else fast<br/>Jev down → strong"]
```

`_llm` (`orchestrator.py:694`) caches one client per `(persona, tier)`. Each is wrapped in
`Metered` (`usage.py:132`), which records every call, and gets an `on_progress` hook that emits
`progress` events (token counts while streaming). `_model_cfg` (`orchestrator.py:707`) swaps in
`models.fast`, or `models.strong` from `config.local.yaml`, keeping the persona's sampling settings.
