# 05 · Stamina: compaction, progress check, checkpoint and resume

These are the parts that let a worker run 20 to 60 steps without overflowing its context, going
in circles, or losing work when something stops it.

## Compaction (`Worker._compact`, `worker.py:750`)

Compaction runs at the start of every step, before the model call. It only fires when the
conversation is past `limits.compact_tokens` (60k estimated tokens at 3.2 chars per token). It
makes **no model call**: Jev scores each old step, and plain code writes the log.

```mermaid
flowchart TD
  A["size = chars / 3.2"] --> B{"size > compact_tokens<br/>(60,000)?"}
  B -- no --> X["no-op"]
  B -- yes --> C["body = messages after the task message<br/>recent = last 2 × keep_steps (6) messages,<br/>cut moved forward to an assistant turn<br/>(never splits a tool call from its result)"]
  C --> D{"old part ≥ 10% of<br/>compact_tokens?"}
  D -- no --> X
  D -- yes --> E["group old messages into steps<br/>_steps() worker.py:115"]
  E --> F["each step → entry<br/>brief: _condense() one-line notes<br/>full: _full() THOUGHT 400 / CODE 1500 / OUTPUT 2000<br/>worker.py:839, 854"]
  F --> G["now = condensed recent steps (last 2k)"]
  G --> H["_score: Jev compact per step<br/>'holds info finishing the task will use again?'<br/>cached by step text<br/>worker.py:797 → orchestrator.py:612"]
  H --> I["keep in full if P ≥ compact_keep 0.8"]
  I --> J["entries = earlier entries + new ones"]
  J --> K["soft = max(8,000 chars,<br/>0.6 × compact_tokens × 3.2 − system/task/recent)"]
  K --> L["_fit worker.py:813<br/>1. over min(soft, 16k): re-score full steps fresh,<br/>demote the least needed to notes<br/>2. over 16k (PROGRESS_CHARS): drop the<br/>lowest-scored entries"]
  L --> M{"new log ≥ 90% of<br/>what it replaces?"}
  M -- yes --> X2["skip: would barely save<br/>(scores stay cached)"]
  M -- no --> N["task message :=<br/>task text + '[Progress so far …]' + log<br/>messages = system + prior + task msg + recent"]
  N --> O["event compact<br/>tokens_before, tokens_after, full, notes, dropped"]
```

What survives compaction, and where:

| Kept | Where it lives afterwards |
|---|---|
| System prompt, follow-up turns | unchanged, before the task message |
| The task text | first part of the rewritten task message |
| Old steps Jev scored as needed | in full inside the progress log (capped sizes) |
| Other old steps | one-line notes: `- note:`, `- ran: <first 8 lines>`, `-> <output 400 chars>` |
| The last 3 steps | verbatim messages |
| Variables, files, the project | untouched (the REPL isn't touched) |

The compaction state (`_entries`, `_dropped`) is saved in `checkpoint.json` so a resumed task
keeps its log. The dashboard's **context ring** comes from the `context` event
(`_report_context`, `worker.py:722`). It is sent every step and splits the estimated prompt into
instructions, tools, skills, task, work and output, out of `compact_tokens`.

## Progress check (`Worker._stalled`, `worker.py:871`)

```mermaid
stateDiagram-v2
  [*] --> Working
  Working --> Check: every progress_every (8) steps
  Check --> Working: new page URLs appeared in outputs<br/>(strikes reset)
  Check --> Working: Jev progress circles < 0.6<br/>(strikes reset)
  Check --> Strike1: circles ≥ 0.6
  Strike1 --> Working: note added to last output<br/>'step back, find the root cause…'<br/>or 'you printed the same result 3× - hand it over'
  Strike1 --> WrapUp: next check also ≥ 0.6
  WrapUp --> Stopped: the step after wrap-up<br/>(only the final tool offered)
  WrapUp --> [*]: worker calls final() - accepted as is,<br/>problems listed, never bounced
  Stopped --> [*]: stopped = stalled, files harvested,<br/>last result quoted, resumable
```

- The Jev question gets the task text (1.5k) and the condensed last 16 messages (4k).
- `new_pages` (`worker.py:64`) counts only URLs that appeared in tool **outputs**, not URLs the
  worker typed. A page it hadn't reached before counts as progress, however similar the steps look.
- Each check emits `progress_check` `{circles, stuck, strike, new_pages}`.

## Other stamina guards

| Guard | Code | Effect |
|---|---|---|
| Same error 3× | `worker.py:1091` | "Stop retrying it. Try a genuinely different approach, or call final()" |
| 2 steps / 1 step left | `worker.py:1080` | "save your deliverables… call final()" / "LAST STEP" |
| Repetition in a reply | `worker.py:93` | the stream is stopped and the worker told to do what it keeps saying it will do |
| Thinking watch | `orchestrator.py:622` | see [04](04-llm-streaming.md#the-watcher-thread-_watch_thinking-llmpy100) |
| Step limit | `worker.py:710` | files harvested, the last output quoted, resumable |
| Hidden reasoning kept | `worker.py:514` | the last 40k chars of each step's reasoning go back as `reasoning_content` (`limits.keep_reasoning`) |

## Checkpoint and resume

```mermaid
sequenceDiagram
  participant W as Worker
  participant R as REPL
  participant D as task folder
  participant H as Harness
  participant UI as CLI /resume or dashboard

  loop after every step
    W->>R: snapshot(repl_state.pkl)
    W->>D: checkpoint.json (atomic write)<br/>persona, task_text, task_idx, progress_log,<br/>compact entries, step, finished, stopped, workdir, messages
  end
  Note over W,D: stopped / crashed / harness restarted
  UI->>H: resumable() - scans sessions/*/*/checkpoint.json<br/>unfinished, or stopped in max_steps/stalled/error/cancelled
  UI->>H: resume(path)
  H->>H: turn(task_text, force=persona, resume=data)<br/>route.follows = False
  H->>W: run_task → Worker(workdir = old folder).run(resume=data)
  W->>W: messages = new system prompt + saved messages<br/>restore task_idx, compaction state
  W->>R: restore(repl_state.pkl)
  R-->>W: restored vars, defs, failed
  W->>W: add RESTORED_NOTE (or RESUME_NOTE) to the last user message
  H->>D: old checkpoint: resumed_by = session
```

Code: `_save` `worker.py:940`, `resumable` `orchestrator.py:274`, `resume` `orchestrator.py:294`,
`_add_resume_note` `worker.py:926`, `snapshot` / `restore` `repl/server.py:279`, `:306`.
