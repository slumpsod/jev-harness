# 16. Interjection: talking to a task while it runs

Status: built (`jevh/interject.py`, plus `worker.py`, `orchestrator.py`, `llm.py`, the dashboard and the CLI) and
run live on 2026-10-03/04 on Qwen 3.8 27B (LM Studio, llama.cpp) with the semif Jev. The classifier is scored by
`evals/interject_cases.py` (209 of 214 labelled messages right at 0.4: dev 68/68, held-out 39/40, fresh 48/48, and two sets of
messages that are easy to take the wrong way, 29/30 and 25/28; one correction, a plain statement of fact, was answered instead of
added, no other request or stop; the fresh and explanation sets are no longer untouched tests) and the
whole feature by `evals/bench/interject.py`. The last *full* bench run (every check passed on both reps;
`evals/bench/interject_last_summary.txt`) is from before the default below was changed: the changed part was re-run
afterwards (`interject_add_summary.txt`, and the classifier above), the other stages do not depend on it. The
retry-delay experiment: `evals/bench/interject_retry_summary.txt`. **Not run on vLLM**: see "Model servers" for what is
expected there and what to check.

**Default (changed 2026-10-04 after a first use on the phone):** a plain message is added to the task's prompt. The first
design had Jev also queue "another request" and stop on "stop"; a correction sent while a picture was being made
("Actually could you just tell me what's on my calendar") was filed as another request (0.63) and queued behind the
task. Queueing and stopping are now only the `/queue` and `/stop` commands.

## Why

A running task had one control, Stop. The dashboard disabled its message box while a turn ran and the CLI only
took Ctrl-C and answers to `ask_user()`. Hermes Agent, running on the same LM Studio, shows the behaviour wanted:
send "Actually could you check my calendar" while it checks your email and it answers "Redirected current run".
Two of the notebook's ideas ask for the same thing: interrupt or redirect a conversation, and a quick reply that
says what is happening while the main task works.

## What Hermes does (read in its source on this machine, `hermes-agent` at 064f2f6b09)

- `gateway/run_busy.py`: a message that arrives while the agent runs is handled by `busy_input_mode`:
  **interrupt** (default; with `redirect()` where the agent supports it), **queue** (the next turn, FIFO) or **steer**
  (`agent.steer()`: delivered after the next tool batch, nothing interrupted). `/stop`, `/new`, `/queue`, `/steer`,
  `/btw` and `/bg` work while it is busy. The acknowledgement reads "Redirected current run (iteration 1/400)".
- `agent/interrupt_control.py`: `redirect(text)` cancels *only the model request* in flight (the completed messages
  are kept, the partial reasoning becomes assistant context, the correction is appended as a real user message
  and the loop retries). While a tool runs it degrades to `steer()`: **a tool is never killed to deliver guidance**.
  A response that crosses a redirect is discarded and rebuilt from the correction.
- `agent/side_question.py`: `/btw <question>` answers from a snapshot of the conversation, by a fork of the running
  agent with the same system prompt and the same tool list (every tool call denied), so the model server's prefix
  cache is hit. "Prompt-cache prefix is sacred": history is only ever appended to.

## What we took, and what is different

The same shape: cut only the model's reply, never the code; append, never rewrite; a side question on a copy of what
the task knows. Different: **there is no mode to pick**: a plain message is simply added to the prompt. One Jev
call (a Choice, ~0.4 s) only notices a question about progress, which is answered on the side instead, and a message
that swaps the request. The final() gate and the memory record know about the correction, and a redirect does not cost
the task a step.

| You send | What happens |
|---|---|
| a plain message (or `/steer <text>`, which skips Jev's check) | added to the prompt: the lead worker gets it as a user message |
| a **question about progress** (`/btw <question>`, `/status`) | an instant line from the worker's state, then the task's own model answers on a copy of what it knows; the task is untouched |
| `/queue <text>` | queued; runs as its own turn when this one ends, by itself |
| `/stop` | like the Stop button |

Jev's question (`Jev.interject_kind`) has five options (correction, **replace**, status, separate, stop). Only two
things act: "status" leading with at least `thresholds.interject_status` (0.4) gives the quick reply, and a "replace"
score of at least `thresholds.interject_replace` (0.5) makes the message stand alone as the request the gate judges
(`Task.current`) instead of being added to it. "separate" and "stop" never act (they are commands); their options stay in
the Choice because they keep such a message from being mistaken for a question. A question added to the prompt instead
could end the task with a status report; a message taken for a question never reaches the task (one of the 146 that were not questions was: a statement of fact, "the crash only happens with a zero-byte file").
With Jev down every message is added to the prompt.

## How a correction is applied

`Harness.interject(text)` finds the lead worker (`_lead`, the root task's) and calls `Worker.steer(text)`:

1. **The model is reading its prompt or writing a reply**: the reply is cut (`llm.chat(interrupt=...)` returns what it
   had, `redirected=True`). What it had written stays in the conversation as an assistant message with its code blocks
   cut, marked "[stopped before it ran: the user sent a message]"; nothing in it runs. The user's message follows as a
   user message and the step is asked again. That step is **not charged** to `max_steps`. The worker first waits
   `interject.retry_delay` (0.5 s; a Stop ends the wait) so that llama.cpp can free the cancelled slot: see "Model servers".
2. **Code is running**: it is left alone. The message arrives at the next step, after the code's output.
3. **A reply that arrived just as the message did** (the model finished first): not acted on, like (1).
4. **`final()` was just called**: held. The tool result says it was not accepted and the message follows; the worker
   answers it and calls `final()` again.
5. **Too late** (the worker is already finished, or the turn is handing its answer over): the message becomes the
   next request in the queue.

The conversation is only ever appended to, so everything the model server already holds in its cache stays valid
(see "Model servers"). The task as the gate and the progress checks judge it is `Task.current`: the original request
plus `UPDATED` ("the user then said ...; where it conflicts with the request above, it wins") for each correction,
or the message alone when it replaces the request. The deterministic file check follows the latest message when it
names a file (`verify.check(files_from=...)`): live, a redirect from `fib.py` to `primes.py` was bounced with "the
request asks for fib.py, but you handed over primes.py", and the worker then produced the old file too. The
correction is also kept in the turn's memory notes ("[While this ran, the user also said ...]") and in the task's
checkpoint text, so compaction (which rebuilds the task message) cannot drop it.

## Cutting a model reply (`OpenAICompat._chat`)

The stream is read on a **thread of its own** and the calling thread waits in 0.1 s steps for it, a Stop, an
`interrupt` or the thinking watch. This is why: a model still reading its prompt sends nothing for a long time
(LM Studio sends the headers at once and the first chunk when the prompt is read: ~22 s for 6k tokens), and a read
waiting for the server **cannot be woken from another thread** on Windows: neither `response.close()` nor
`socket.shutdown()` ends it (measured), and `close()` even blocks on the lock the reading thread holds. The first
version closed the response and a correction sent 9.7 s into a step landed at 28.5 s, when the first token came; the
same bug meant Stop during prompt reading waited for the first token. Now the caller stops waiting, drops the
connection with `shutdown()` (the server sees it and stops) and leaves the reader to finish by itself. Cut at once:
0.7 s in `tests/test_interject.py` against a server that goes silent; live, the redirect now lands in the same second as
the message.

## The quick reply

`Harness.aside(question)`. The instant line (`interject.status_line`: "high_end_dev is working on it (step 1 of up to
60): running code: import time; ... That has taken 8s so far.") needs no model. Then the task's own model, on the same
LM Studio instance (the gate never swaps for it), answers in its own words on a copy of what the worker knows, as an
`aside` event, metered as its own usage kind. Two contexts (`interject.aside_context`):

- **digest** (default): the task and the last six steps in a few lines, ~250 tokens. It shares nothing with the task's
  own prompt, so it cannot disturb the server's cache of that.
- **full**: the worker's own conversation as it stands plus one message (a step whose code is still running is left
  out: a conversation must not end on unanswered tool calls). Its start is the worker's own prompt, so a server that
  keeps one prompt cache for all requests reads almost none of it again.

If the model gives nothing (an error, a Stop, a budget `aside_max_tokens` spent thinking) the instant line is what
stays. A thinking model thinks: LM Studio ignores `enable_thinking`, `reasoning_effort` and `/no_think` (the worker's
notes say so), so the reply takes a few seconds.

Measured on Qwen 3.8 27B (LM Studio, 2 slots; single runs, a 45 s `time.sleep` task so the model was idle when asked;
the bench's medians over two reps are 6.4 s for digest and 4.5 s for full, and across its passes the reply took 4.5-11.6 s):

| | prompt | read from the server | answer in | the task's next call read |
|---|---|---|---|---|
| digest, code running | 243 tokens | 244 | 7.8 s | 141 of 2003 prompt tokens |
| full, code running | 1,941 tokens | **78** (96% cached) | 5.8 s | 310 of 2173 |
| no question (control) | | | | 441 of 2302 |
| digest, model mid-reply | 225 tokens | 91 (the other slot) | 5.0 s | the task was not touched, it finished normally |
| full, model mid-reply | 2,029 tokens | **2,029** (the other slot, cold) | 8.6 s | |

Neither mode made the task's next call cost more than the control when the task was idle. With the model mid-reply
the `full` prompt goes to the other slot with nothing cached (all 2,029 tokens read; with a 20k-token conversation
that is ~30 s at this machine's 700 tokens/s), so **digest is the default on LM Studio**; a server that shares prefix
blocks between running requests (vLLM) makes `full` nearly free.

## The queue and Stop

`/queue` appends to `Harness._queue`; `take_queued()` hands the next one to whoever runs
turns: the dashboard's turn thread runs them by itself, in order (`Hub._run_turn`, with a small lock so a message sent
between two queued turns is never lost), and the CLI does the same after a turn. **Stop drops the queue** (and says
so: the `cancel` event carries what was dropped, and the dashboard puts it back in the message box) because the
usual meaning of Stop is stop everything.

## Surfaces

- **Dashboard**: the message box stays usable while a turn runs ("Add to the running task's prompt (/queue to run it
  after, /stop)"). What you sent appears in the turn with what became of it (added to the task, swapping the task,
  queued for after, quick reply, stopped); the
  quick reply shows the instant line and swaps in the model's answer; the queue shows above the box.
  `POST /api/chat` returns `{"interjected": {...}}` instead of a 409 while a turn runs; `kind` can be sent to skip Jev.
- **CLI**: while a turn runs a non-blocking line reader (`cli.LineReader`: `msvcrt` on Windows, `select` elsewhere)
  takes what you type; Ctrl-C still stops. A message sent as the turn ends is queued, not lost.

## Model servers: LM Studio (llama.cpp) and vLLM

Measured in LM Studio's own log on this machine (`evals/bench/lmstudio_log.py` reads it):

- A client hang-up cancels the task at the next batch boundary: "Client disconnected" then "stop: cancel task" a
  fraction of a second later, and the slot is released within a batch (~3 s while a 2k-token prompt was being read).
- **A redirect while the model was writing** usually makes the server read 6-10% of the prompt again (73 of ~2,200
  tokens in the first run, 140-290 in the bench): the slot's cache holds the whole conversation. **It is a race**:
  the retry is sent the moment the reply is cut, and llama.cpp frees the cancelled slot a moment after the hang-up. A
  retry that arrives first starts on the other slot with nothing cached and reads the whole prompt again (2 of 7
  retries with no pause; 4.5 s here, ~30 s for a 20k-token conversation, and the prompt then sits in both slots).
  `interject.retry_delay` makes the worker wait before asking again: **0 of 15 cold retries at 0.5, 1 and 2 s** (5 reps
  each; `evals/bench/interject.py --stage retry`), so 0.5 s is the default (the final full bench, at 0.5 s, agrees); 0 on
  a server without slots (vLLM).
  **A cut while the model was still reading its prompt** made the server read all of it again with no pause (2022 of
  2021 tokens in both bench reps: the cancel only lands at the next batch, and the retry started on the other slot) and
  29% with the 0.5 s pause (590 of 2021 in both reps: the cancelled slot was freed holding the part it had read).
  Waiting for the reply to finish instead would be far slower.
- Slots are chosen by prompt similarity (`--slot-prompt-similarity`, default 0.10), so a conversation that only grows
  hits its slot's cache (typical steps read 100-500 tokens of 2-4k). A prompt that diverges from a slot's cache can be
  restored from a host-memory prompt cache (`--cache-ram`, default 8 GiB). The models here are hybrid
  (`qwen35`, `qwen35moe`: recurrent layers), which llama.cpp covers with context checkpoints (`--ctx-checkpoints`,
  default 32 per slot).
- Two decodes at once (2 slots) each run at ~22.8 tokens/s against ~31 alone.
- **Both models loaded at once crawl** (the note in `lmstudio.py`): with ornith loaded next to Qwen the prompt was read
  at ~30 tokens/s and generated at 0.9 tokens/s; with Qwen alone 708 and ~31. The harness's gate keeps one model
  loaded, but another client on the same server (Hermes) is not under it, and a task that needs the fast tier unloads
  the model that client was using: pin the model (`--model`) when benchmarking next to it.

vLLM (from its documentation, fetched 2026-10-03; **none of this was run here**):

- Prefix caching is on by default in the V1 engine (`--no-enable-prefix-caching` turns it off). Cached blocks are
  reference-counted and shared by every request that uses them, running ones included, only full blocks are cached,
  and they are evicted LRU once nothing references them. A redirect's retry and a `full` quick reply, which both start
  with the conversation the server just served, should therefore cost almost no prefill and **no extra KV memory** for
  the shared start: set `interject.aside_context: full` there.
- Hybrid (Mamba-style) models need `--mamba-cache-mode align` for prefix caching; hits then resume only at block
  boundaries (`--mamba-cache-mode` defaults to `none`, which turns it off for those layers).
- A thinking model can be told not to think per request: `interject: {aside_extra: {chat_template_kwargs:
  {enable_thinking: false}}}` (laid over the quick reply's request body only).
- To use it: a provider with `type: openai` and the server's `base_url` (the existing `vllm` provider, or the vllm-pro
  gateway on `:1235` with its key in `api_key_env`), no `exclusive`. Nothing here depends on LM Studio; the interrupt is
  a hang-up like Stop's. To check on a real vLLM: that a hang-up frees the request, and the prefix-cache hit counters
  (its `/metrics`) rise for a redirect retry and a `full` quick reply.

## Limits and known gaps

- Only the lead worker hears you. While it waits on a hand-off (inside `request()`), a correction waits for it; the
  hand-off's own worker is not told.
- Code is never cut for a message. A long step delays a correction until its output; Stop is the way to cut it.
- A model that cannot be cut mid-reply (Claude through the SDK) finishes the reply first, then the message applies.
- A plain message that reads like another request ("also write me a haiku") is added to the prompt, and the model
  decides the order (live: it did both). `/queue` runs it as its own turn after.
- A plain "stop" is added to the prompt, not obeyed: `/stop` or the Stop button stops.
- A question the check misses is added to the prompt (1 of 40 labelled questions), where the model may answer it in
  its next step or, with no code in the reply, end the task with that answer.
- A redirect during prompt reading costs a second reading of the prompt (see above). Waiting for the first token and
  cutting then would reuse what was read, at the price of the redirect landing only when the reading ends: not built.
- The quick reply needs a second slot to be quick: with one slot (`parallel: 1`) it waits behind the task's own reply.
