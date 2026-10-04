# 03 · Worker step loop and REPL

`Worker.run` (`worker.py:539`) is the heart of every task. Each **step** has three parts: build
the context, call the model, and act on the reply (run its code, or accept an answer). The
conversation is a plain `messages` list, and the harness rewrites it in a few places
(compaction, pauses, notes).

## One step

```mermaid
flowchart TD
  START["run(task, ctx, system, history, resume)<br/>worker.py:539"] --> MSG
  MSG["messages = system + prior turns + task message<br/>task = request + (In your REPL: ctx keys) + FIRST_STEP_NUDGE*<br/>worker.py:562-566"] --> REPL
  REPL["Repl start: spawn jevh.repl.server<br/>init tools, set ctx<br/>worker.py:569-572"] --> RES{"resume?"}
  RES -- yes --> RN["restore repl_state.pkl<br/>add resume note<br/>worker.py:930"]
  RES -- no --> LOOP
  RN --> LOOP
  LOOP["for step in 1..max_steps<br/>worker.py:575"] --> CK["_check: cancel?<br/>worker.py:1009"]
  CK --> CP["_compact(messages)<br/>see 05<br/>worker.py:754"]
  CP --> CTX["_report_context → event context<br/>worker.py:726"]
  CTX --> CALL["_llm(messages) → LLMResult<br/>see 04<br/>worker.py:1021"]
  CALL --> APP["append assistant msg<br/>(+ reasoning_content, last 40k chars)<br/>worker.py:518, 579"]
  APP --> INT{"interrupted by<br/>the watcher?"}
  INT -- yes --> PAUSE["replace reply, add pause note<br/>(testable / searching / repeating)<br/>worker.py:585-619"] --> LOOP
  INT -- no --> PARSE{"what came back?"}
  PARSE -- "native tool_calls" --> NT["call_to_code per call<br/>python→code, final→print(final(..)),<br/>web(search=q)→web.search(q)<br/>worker.py:623-637"]
  PARSE -- "fenced python /<br/>tool_call text" --> FB["CODE_RE or TOOL_CALL_RE,<br/>XML calls translated<br/>worker.py:639-650"]
  PARSE -- "cut off at max_tokens /<br/>open fence" --> CUT["'work in smaller steps'<br/>worker.py:651"] --> LOOP
  PARSE -- "empty (only thinking)" --> EMP["'continue from your reasoning'<br/>worker.py:661"] --> LOOP
  PARSE -- "plain text, no code" --> PL["final() gate on the text<br/>worker.py:673"]
  PL -- bounced --> LOOP
  PL -- accepted --> DONE1["stopped = plain"]
  NT --> RUN
  FB --> RUN
  RUN["repl.run(code)<br/>event code, output<br/>worker.py:680"] --> FIN{"ex.final set?"}
  FIN -- yes --> GATE["_bounce: final() gate<br/>see 07<br/>worker.py:966"]
  GATE -- bounced --> LOOP
  GATE -- accepted --> DONE2["stopped = final<br/>answer + artifacts (or harvested files)"]
  FIN -- no --> REP["append output as tool result or<br/>'Output:' user msg + notes<br/>worker.py:700"]
  REP --> SAVE["_save checkpoint + REPL snapshot<br/>worker.py:702, 940"]
  SAVE --> ST{"_stalled?<br/>see 05"}
  ST -- "stop (2nd strike, after wrap-up step)" --> DONE3["stopped = stalled<br/>files harvested"]
  ST -- no --> LOOP
  LOOP -- "max_steps reached" --> DONE4["stopped = max_steps<br/>files harvested"]
```

`*` The first-step nudge is skipped for `think: unlimited` models (`worker.py:563`).

### Notes the harness adds to a step's output

These are appended to the text the model gets back after running code (`worker.py:700`):

| Note | Trigger | Code |
|---|---|---|
| "Not run: there is no `x` tool…" | a native call to an unknown tool | `worker.py:632` |
| "Your <tool_call> text ran as Python: …" | XML-style calls written as text | `worker.py:642` |
| "The same error has now happened 3 times in a row…" | same last error line 3× | `_stuck_note`, `worker.py:1095` |
| "[2 steps left…]" / "[LAST STEP…]" | 2 or 1 steps left | `_steps_left_note`, `worker.py:1084` |
| Progress-check nudge / wrap-up order | see [05](05-stamina.md) | `worker.py:875` |

### How a result becomes a message

- **Native tools** (`native_tools: true`): the assistant message carries `tool_calls` (arguments
  re-encoded as valid JSON, `valid_args`, `worker.py:263`). The output goes back as a `tool`
  message with the first call's id. Extra calls get "(ran together with the first call)"
  (`_replies`, `worker.py:1075`).
- **Text mode**: one `user` message, `Output:\n…`.
- On the last step of a stalled worker, the model is offered only the `final` tool
  (`worker.py:1024`).

### Ways a task ends (`WorkerResult.stopped`)

| stopped | Meaning | Resumable |
|---|---|---|
| `final` | `final()` was called and passed the gate, or was accepted with problems listed | no |
| `plain` | a reply with no code, accepted as the answer | no |
| `max_steps` | step limit hit; files the worker wrote are copied to artifacts (`_harvest`, `worker.py:1106`) | yes |
| `stalled` | two progress strikes and the wrap-up step used | yes |
| `cancelled` / `error` | Stop pressed, or an exception (e.g. the model server died) | yes |

### Messages from the user while it runs

The loop is a `while step < self.limit` (a step spent hearing the user out raises the limit by one). At the top of
each step `_deliver_steers` appends the user's pending messages to the conversation (the end only, so the model
server's cache of everything before stays valid); a model reply cut by `steer()` comes back `redirected` and is kept
as context without anything in it running (`_redirect`); a `final()` while a message waits is held (`STEER_HELD`).
Code that is running is never cut. See [16 Interjection](16-interjection.md).

## The REPL process and its protocol

Each worker gets its own Python process: `python -u -m jevh.repl.server` with its working folder
as cwd, `JEVH_ARTIFACTS` pointing at the session's artifact store, and the env from `run_task`
(`repl/client.py:70`). The protocol is JSON lines. The server answers on a private duplicate of
the original stdout, and its fd 1 is pointed at stderr, so prints from child processes can't
corrupt the protocol (`repl/server.py:32-33`).

```mermaid
sequenceDiagram
  participant W as Worker (harness thread)
  participant RC as Repl client
  participant RD as reader thread
  participant RS as REPL server
  participant H as handlers (orchestrator)

  RC->>RS: spawn process (cwd = task folder, env)
  RS-->>RD: {"type": "ready"}
  RC->>RS: {"op": "init", "tools": {"web": "jevh.tools.web", ...}}
  Note over RS: import each tool into G<br/>tools with _host get the call channel (browser.look)
  RS-->>RC: {"type": "result", output, error}
  W->>RC: set("ctx", ctx)
  RC->>RS: {"op": "set", "name": "ctx", "value": {...}}
  W->>RC: run(code)
  RC->>RS: {"op": "exec", "code": "..."}
  Note over RS: ast parse, remember imports/defs,<br/>exec, print repr of a trailing expression
  RS-->>RC: {"type": "call", "name": "request", "args": [...], "kwargs": {...}}
  RC->>H: handlers["request"](*args) (time not charged to the timeout)
  H-->>RC: value or exception
  RC->>RS: {"op": "reply", "value": ..., "error": null or "Type: msg"}
  RS-->>RC: {"type": "result", output, error, final, usage}
  RC-->>W: ExecResult(output truncated to output_cap 4000, final, usage)
```

| Channel | Code | Details |
|---|---|---|
| Spawn and ready | `repl/client.py:70-90` | waits up to 30s for `ready`; tool import errors go to `stderr_tail` |
| Round trip | `repl/client.py:141` | restarts a dead REPL; on timeout (`limits.repl_timeout`, 1800s) kills the process tree and restarts: "[timed out…all variables were lost]" |
| Callbacks | `repl/client.py:160`, `:166` | any `{"type":"call"}` is dispatched to `handlers[name]`; exceptions come back as `RuntimeError` in the worker's code |
| Output cap | `repl/client.py:31` | 60% head + 40% tail, "… [N chars omitted; use peek() or slice] …" |
| Secret scrub | `repl/server.py:354` | an exec's output has unmistakable secret formats (`GOCSPX-…`, Slack/GitHub tokens, AWS key ids, Stripe live keys, private keys) replaced by `[secret hidden]` before it goes back to the model |
| Tool Jev usage | `repl/server.py:424-430` | `decide.TOTALS` delta per exec → `ExecResult.usage` → event `tool_usage` → usage.db `jev_tools` |
| Kill | `repl/client.py:39` | process group (POSIX) or `taskkill /T` (Windows): Chromium and jobs go with it |
| `input()` | `repl/server.py:63` | disabled: "ask the user by finishing with final()" (the worker has `ask_user` instead) |

### What the worker's code can call (`repl/server.py:251`)

| Helper | Runs where | Goes to |
|---|---|---|
| `peek`, `save_artifact` (a `data=` string that is only an existing file's name is treated as `path=`, `repl/server.py:181`), `load_artifact`, `list_artifacts`, `start_job`, `job_status`, `job_wait` | in the REPL | local files, subprocesses |
| `final(answer, artifacts, answer_var)` | in the REPL | sets `_FINAL`; non-string answers become JSON; bare artifact names resolve into the store; `changed` = files `project` touched |
| `llm_query(prompt, data)` | callback | the task's own model, one message (data clipped to 60k) |
| `jev(state, questions)` | callback | `Jev._ask("worker_jev")` |
| `request`, `request_async`, `wait`, `request_all` | callback | `Handles`, see [06](06-handoffs.md) |
| `ask_user(question, options, secret=False)` | callback | `Harness._ask_user`, see [10](10-observability.md); `secret=True`: typed hidden, `(hidden)` in events |
| `recall(query, k)`, `remember(fact)` | callback | Memory FTS + Jev gate; Learner fact store |
| `report_failure(what, detail)` | callback | `FailureWatch.report`: a `model` row in the failure log; the worker carries on ([11](11-failure-log.md)) |
| `mcp(tool, **args)` | callback | `McpHub.call` (only tools the persona's `mcp` globs allow) |
| `browser.look(q)` (through `_host`) | callback `look` | `see_image(llm, screenshot, q)`: the worker's own model looks at the image |
| `web`, `browser`, `project`, `nango`, `comfy` | in the REPL | see [08](08-tools.md) |

### Snapshot and restore

After every step, `_save` asks the REPL to `snapshot` its user variables
(`repl/server.py:292`). Imports and top-level `def` / `class` sources are kept as source text.
Values are pickled (dill if installed); anything over 16 MB or unpicklable is skipped. On resume,
`restore` replays imports, then defs, then values, and the worker is told which came back
(`RESTORED_NOTE`).
