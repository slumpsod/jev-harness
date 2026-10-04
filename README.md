# jev-harness

A lightweight multi-persona agent harness. A System One decision model (**Jev**:
semif / laya / openjev behind `openjev/systemone.py`) is the gatekeeper for everything a worker model sees.
Workers (any LLM: LM Studio, llama.cpp, vLLM, Claude) see only the request plus what Jev lets through,
and they act by writing Python in their own sandboxed REPL.

**Contributing (people and coding agents): read [AGENTS.md](AGENTS.md) first.** Anything built in the cloud or
remotely must ship with a thorough test bench that runs locally against live models.

```
user ─► Jev: route persona ─► Jev: gate memory ─► Jev: pick skills ─► Jev: suggest collaborators
                                                                              │
          worker (persona LLM) ◄── prompt = persona + REPL guide + picked skills only
            │  ```python``` ─► private REPL (ctx, peek, llm_query, jev, request, save_artifact, final)
            │                                   │
            │                request("front_end_designer", brief, inputs=[paths])
            │                                   └─► Jev approves? ─► fresh worker, sees only the brief + inputs
            ▼
         final(answer, artifacts)
```

Each decision is one batched `/v1/systemone` call (about 0.2s warm on semif).

**Architecture:** [docs/architecture](docs/architecture/README.md) maps every loop and what data
crosses between them, with Mermaid diagrams and `file:line` references.

## Run

On a Mac, Linux, or with the hosted Jev instead of a local one: see [SETUP.md](SETUP.md). A machine's own
settings (models, Jev, paths) go in `config.local.yaml`, laid over `config.yaml` and ignored by git.

The `jevh` command works from any folder once installed (see [the `jevh` command](#the-jevh-command)).

```bash
openjev\start_systemone.bat          # adapter on :33001 (Jev); backend set in config.yaml
jevh                                 # chat; /help for commands
jevh dashboard                       # the dashboard (see Dashboard)
jevh --as fitness_coach -c "plan my week"   # one-shot, pinned persona
python -m pytest -q                  # offline tests (fake Jev + scripted LLMs, a fake MCP server)
python evals/routing_cases.py --sweep   # live routing/collab accuracy + threshold sweep
python evals/skill_cases.py            # live skill-picking accuracy + sweep (--offline to re-sweep)
python evals/memory_cases.py           # live memory-gating accuracy + sweep
python evals/tier_cases.py             # live fast/strong tier accuracy + sweep
python evals/tool_cases.py             # live tool-guide picking accuracy + sweep
python evals/browser_cases.py --live   # Jev browser agent on a local test shop (+ Wikipedia)
python evals/bench/browser_tasks.py      # 10 real-website tasks, full harness turns (flights, tickets, forms...)
python evals/bench/persona_creation.py   # persona creation, live: Jev checks, the writer, the gate, routing, full turns
                                         #   (--stage, --split dev|held, --reps, --budget/--resume for a runner with a time cap)
python evals/bench/nango_setup.py        # connecting an account: a simulated user + the real local Nango (Docker), multi-turn
python evals/interject_cases.py        # a message sent while a task runs: a question about progress, or anything else (--holdout, --fresh, --offline)
python evals/bench/interject.py --model lmstudio:qwen3.8-27b@q6_k   # messages to a running task, live: redirect speed, the gate, quick reply, queue, Stop
                                         #   (--stage, --reps; --stage retry measures the pause before a retry; needs nothing else using the
                                         #   model; reads LM Studio's log for the server-side cost)
python evals/learn_cases.py            # the three learning gates (fact / same fact / skill)
python evals/hook_route_cases.py     # skill/hook router: "could a program check this?" (NOT TUNED yet: run it, then set route_hook)
python evals/recall_cases.py 4         # memory reach when the history is buried under 160 newer turns
python evals/failure_cases.py          # the failure log's Jev check (gave up / missed the request) + sweep
python evals/verify_evidence_cases.py  # does the final() check see evidence deep inside long tool outputs
python evals/compact_cases.py          # which earlier steps compaction must keep in full
python evals/button_intent_cases.py    # browser: a click the goal asked for vs one with a side effect
python evals/expect_cases.py           # browser.test: does the page show the expected result
python evals/find_cases.py             # project.find: does it point at where a question is answered
python evals/risk_cases.py             # is a code change risky enough for a full review
python evals/severity_cases.py         # a risky change: strong-model or fast-model review
python evals/rule_cases.py             # does an edit break the project's AGENTS.md / CLAUDE.md rules
python evals/search_cases.py           # a long think: searching theories, or checking one idea
python evals/thinking_watch_cases.py   # when to pause a long think and test the idea in code
# most evals take --holdout (the held-out cases) and/or --offline (re-sweep saved scores)
jevh failures                          # logged failures; `export out.jsonl` to study them
jevh review                            # flagged sessions; `flag <session> "note"`, `run <id>` (a model's verdict)
jevh notebook                          # your notebook; `idea "..."`, `problem "..."`, `done <id>`, `-q words`
```

CLI commands: `/as <persona>`, `/auto`, `/tier auto|fast|strong`, `/project <path>` (or `--project`),
`/watch [on|off]` (the browser tool in a visible window), `/facts`, `/forget <id>`, `/usage`, `/failures`,
`/resume [n|list]`, `/flag [note]`, `/note <text>`, `/flags`, `/review`, `/verdict failure|not_failure [why]`
(see Session review), `/idea <text>`, `/problem <text>`, `/jot <text>`, `/notebook` (see Notebook), `/personas`, `/persona approve|activate|delete <name>` and `/forge` (personas the harness
created), `/skills`, `/trace` (raw Jev answers for the last turn),
`/verbose` (show code + REPL output), `/jev` (adapter health), `/update [check]` (see Updating), `/reset`, `/quit`.
While a task runs, what you type is added to its prompt (see Talking to a running task), except a question about
progress, which is answered on the side. `/queue <text>` holds a message for after, `/btw <question>` asks, `/stop`
stops, `/steer <text>` always adds.

## The `jevh` command

| Command | Does |
|---|---|
| `jevh` | chat in the terminal (`jevh --help` for its options) |
| `jevh dashboard` | the dashboard in your browser |
| `jevh update` | update the harness from GitHub |
| `jevh home` | where your own things are; `jevh home move` moves them out of the checkout |
| `jevh failures`, `jevh review` | the failure log, flagged sessions |
| `jevh notebook` | your notebook of ideas and problems |

`pip install -e .` inside the venv puts `jevh` on the PATH while that venv is active. To have it in every
terminal with no venv to activate, install it once with [pipx](https://pipx.pypa.io)
(`brew install pipx` on a Mac, `py -m pip install --user pipx` on Windows, then `pipx ensurepath` and open a new terminal):

```bash
pipx install -e ".[web,browser]"       # from the repo folder; -e keeps it pointing at this checkout
pipx run --spec playwright playwright install chromium   # the browser tool's Chromium, if not installed yet
```

Because it's editable, it always runs this folder's code: `jevh update` (or a `git pull`) updates the
global command too, and memory, logs and `config.local.yaml` stay in this folder whichever folder you start
it from. Every command also still works as `python -m jevh.cli`, `python -m jevh.web` and so on.

## Updating

```bash
jevh update            # or /update in the chat, or the Harness version card in the dashboard's Status tab
jevh update --check    # only list what's new on GitHub
jevh update --test     # also run the test suite on the new version; undo the update if it fails
jevh update --undo     # back to the version before the last update (/update undo, or "Undo update")
jevh update --set-channel main     # every change as it lands, instead of stable (default: stable)
jevh update --snapshots            # data snapshots taken before updates; --restore-data <name> puts one back
```

**Channels.** Every push to `main` runs the tests on GitHub (`.github/workflows/stable.yml`); when
they pass, the `stable` branch moves to that commit. `jevh update` follows `stable` by default, so a
commit that breaks the tests never reaches you. `--set-channel main` (or the channel menu in the
dashboard) follows `main` instead; `--channel main` does it for one update.

An update either works or leaves everything as it was:

- **Your edits are kept.** Edited harness files (a persona, a skill, the code) are saved with `git stash`,
  carried onto the new version and put back. A dry-run merge checks first that they still apply; if
  one clashes with the update (you and the update changed the same lines), it stops before anything
  moves and says which file. A file you added that the update would overwrite stops it the same way.
  `config.local.yaml` and `workspace/` are ignored by git, so updates never touch them.
- **New dependencies are installed.** If `pyproject.toml` changed, the update runs `pip install -e`
  with the extras you already have.
- **A broken version is rolled back.** The new version must import (with `--test`, or the dashboard's
  "run the tests first" box, it must pass the tests too). If that or the install fails, you're put back on
  the old version with your edits as they were.
- **You can undo it.** The version you came from is kept for `--undo`, which also keeps your edits.
- **Your data is snapshotted first.** The workspace databases, learned skills and config files are
  copied to `<workspace>/snapshots/` (the last 5 are kept), since a newer version may change data in a
  way the older one can't read; `--restore-data` puts a snapshot back.
- **A broken persona or skill is skipped.** One that can't be loaded (bad YAML, missing fields, an
  unknown model provider), or an MCP server that won't start, is left out with the reason (shown when
  the CLI starts and in the dashboard's Status tab) instead of stopping the harness.

The fewer of your own things live in the checkout, the less an update has to work around: see
[Your things, outside the checkout](docs/your-data.md) (`jevh home move` does the moving).

Copies stay in git as `refs/jevh/pre-update` (the old version) and `refs/jevh/pre-update-edits` (your
edits, `git stash apply refs/jevh/pre-update-edits`). Local commits that aren't on GitHub stop the update
(`git pull` merges them). Restart the CLI or dashboard afterwards to load the new code. The dashboard
won't update during a turn.

## Adding things

- **Persona**: drop a YAML in `personas/` (your own ones: `~/.jevh/personas/`, see [your data](docs/your-data.md)). Jev routes on `description`, so make it concrete.
  `model` picks the provider from `config.yaml` (an optional `fallback:` is used if the primary errors);
  `skills_allowed` holds globs over the skill roster; `can_request` lists who it may delegate to (Jev's
  collaborator suggestions add to this per turn); `tools` are modules from `jevh/tools/` pre-imported
  into its REPL (`web`, `browser`, `project`, `nango`, `comfy`); `mcp` holds globs over MCP tools ("server.tool");
  `code: false` makes it a plain chat persona.
- **MCP server**: add it under `mcp_servers:` in `config.yaml` (`command`, `args`, `env`) and give a persona
  `mcp: ["server.*"]`. Servers start on first use.
- **Skill**: `skills/<name>/SKILL.md` with `name` and `description` frontmatter. Extra roots (e.g. a Hermes
  skills folder) go in `skills_dirs` in `config.local.yaml` (hidden `.archive` dirs are skipped). A picked skill's first 6k chars go in the prompt, and the
  full text is in `ctx['skills'][name]`.
- **Model provider**: any OpenAI-compatible server works as `type: openai`. Claude uses the `anthropic`
  SDK (`pip install anthropic`, and set `ANTHROPIC_API_KEY` or run `ant auth login`).

## How context stays clean

- Memory is on disk (`workspace/memory/*.jsonl`). The candidates are a persona's own recent turns plus a
  few from other personas, and only snippets Jev scores as relevant (`thresholds.memory`) reach the worker,
  as `ctx['memory']`.
- **Follow-ups keep their thread.** The routing call also asks two follow-up questions: does the message
  continue the recent topic, and is it a reply to the assistant's last message? If either passes,
  this chat's last 2 turns go along as real conversation turns, and Jev's other decisions judge the
  message together with that exchange. This check is needed because a follow-up judged alone
  ("4 days, full gym", or an answer to the assistant's own clarifying question) has nothing to match
  in the memory gate: live, the previous turn scored 0 and the worker asked "your message looks cut
  off?". `followup_cases.py`: 14/14 follow-ups kept, 0/8 new topics joined. A new topic still starts clean.
- Delegated workers get no conversation history, only the brief and the artifact paths they were handed.
- Big data never goes into prompts: the worker sees `ctx` key names and sizes, and reads values with `peek()`.
- Hard limits hold no matter what Jev says: delegation depth, tasks per turn, cycle guard, per-exec
  timeout (the REPL is restarted on timeout), and an output cap.

## Measured (semif backend, 2026-09-27)

| Decision | Eval | Result | Per turn |
|---|---|---|---|
| Persona routing | `routing_cases.py`, 37 requests (33 + 4 "connect my account", 2026-10-02) | 97% top-1 (92% before `daily_assistant`'s description mentioned accounts: it sent 2 of the 4 to the dev persona) | ~0.2s |
| Follow-up detection | `followup_cases.py`, 22 message pairs | 14/14 follow-ups kept, 0/8 new topics joined | (in the routing call) |
| Collaborators | `routing_cases.py` | 94% exact, 0 needless on the 33; with the 4 new, 89% exact, 2 needless ("set up Nango" suggests the dev persona; a design request suggests `daily_assistant`) | ~0.2s |
| Skill picking | `skill_cases.py`, 63 requests (your 52 Hermes-router cases + 11) | 94% right skill, 93% correct "no skill", 3 needless | ~1.5s |
| Skill picking: `nango-setup` | the 8 new cases in `skill_cases.py` (7 "connect my ..." + an OAuth-concepts control, among 15 no-skill controls), harness-local skills only (the Hermes cases need your skills folder) | 7/7 pick it (fits 0.47-0.95); none of the 15 "no skill" controls (the OAuth one included, fits 0.07) loads it | ~1.5s |
| Memory gating | `memory_cases.py`, 18 requests over an 18-turn history | 95% of needed turns passed, 0 leaks | ~1.5s |
| Model tier (fast/strong) | `tier_cases.py`, 39 tasks + 15 held out | 100% and 15/15, 0 hard tasks sent fast | ~0.2s |
| Tool guides | `tool_cases.py`, 59 requests x 5 tools | 55/59 exact, 3 needless (nango: 15/15 picked incl. 5 set-up requests, 1 needless: "how do I use the Gmail API from Python") | ~0.3s |
| final() answer check | `verify_cases.py`, 22 tuning + 2 held-out sets of 12 | 12/12 tuning; fresh held-out 4/6 bad caught, 0 false alarms | ~0.3s |
| Progress check (long tasks) | `progress_cases.py`, 14 work logs | 6/6 stalls caught, 0/8 healthy runs interrupted | every 8 steps |
| Browser agent | `browser_cases.py`, 16 tuning + 12 held-out + 2 Wikipedia goals | 29-30/30, 0 false "done" | ~2.3s per goal |
| Connecting an account | `evals/bench/nango_setup.py -n 3` (2026-10-02; live: Qwen/Ornith workers, semif Jev, the real local Nango; a simulated user and a stand-in browser). 3 scenarios x 3 runs | no-login app connected 3/3 (55-73 s, 4-5 steps); Google Calendar walk-through 3/3 (steps one at a time, secret hidden, integration created, `redirect_uri_mismatch` diagnosed; median 10 min, 27-34 steps); assistant drives a practice OAuth console 3/3 (right ID and secret stored, nothing leaked) | minutes: the user's part dominates |
| Learning gates | `learn_cases.py`, 25 + 10 + 7 cases | 12/12 facts, 5/5 replacements, 3/3 skills; 0 false positives | ~0.2s each |
| Memory reach | `recall_cases.py 4`, history under 160 newer turns | 14/19 needed turns reached (recent-window design: 1/19) | ~1.3s |

Skill picking and memory both use the same two-stage design. One **Choice** ranks all candidates;
it's relative, so it separates sibling skills, and it's cheap because a Choice is scored in one pass.
Then a yes/no check runs on the top 3–4 only. The first versions put a yes/no on every candidate:
skills got 44% clean picks with 25 needless loads, memory passed only 68% of the context actually
needed, and each cost ~0.18s per candidate. Semif caps a Choice at 16 options, so `Jev.choice_probs`
runs bigger sets as rounds of 16 or fewer plus a final. A picked skill's head goes in the prompt and the
full text sits in `ctx['skills']`; in a live radon quiz, the worker searched it to correct a wrong
premise in the question.

Routing details: `evals/routing_cases.py` has 33 labelled requests: persona top-1 is **97%**. Collaborator suggestions are
**94% exact with 0 needless** at `collab: 0.6`, after rewording the question from "requires expertise"
(73%) to "a deliverable the specialist must produce that the lead can't". Semif's confidences are very
peaky (often 1.00), so tune thresholds with the sweep rather than trusting the raw confidence.

### Live end-to-end (qwen3.8-27b@q6_k workers, semif Jev)

"I want an app to look at my workouts and track my progress" sent to `fitness_coach` goes like this:
Jev suggests high_end_dev + front_end_designer, and the coach writes a spec and delegates the build.
The dev (fresh context: 0 memory, no skills) builds and checks a single-file tracker in 6 steps.
The coach then replies in its own voice with how to use the tracker plus progression advice.
It took about 7 minutes and 6 Jev calls. The app (localStorage, e1RM + weekly-volume SVG charts, JSON export) was verified in a browser.

Lessons from getting there (all handled in code now):
- **Thinking models plan forever.** Qwen3.8 in LM Studio spent 16k+ hidden reasoning tokens designing the
  whole app and produced zero code. `enable_thinking`, `reasoning_effort` and `/no_think` were all ignored
  through the API. A one-line nudge in the task message ("short plan + first python block, build
  incrementally") cut thinking to ~130 tokens. See `FIRST_STEP_NUDGE` in `jevh/worker.py`.
- Local models stream, so the timeout is an idle timeout, not a total one. Cut-off replies are detected and the
  model is told to work in smaller steps; the CLI shows live token counts, including thinking tokens.
- Workers get a "2 steps left, call final()" warning. If they still hit the limit, the files they wrote are
  collected as artifacts instead of being lost.
- Failed turns are not written to memory.

## Dashboard

```bash
jevh dashboard                     # http://127.0.0.1:8740/ opens in your browser
jevh dashboard --no-browser --port 8740 --model lmstudio:ornith-1.5-35b-a3b
```

The chat is the whole screen. Everything else is one click away on the left rail and opens beside the chat; the
panel stays closed until you open a tab, the browser tool goes live, or a turn makes a file (then the Artifacts
panel opens on it). Click the same tab again, press Esc, or use the X to close it; drag its edge to resize it.

**On a phone** it becomes one screen: the chat fills it, the tabs move to a bar along the bottom, and a tab opens as a
full-screen panel over the chat (the X at its top closes it). Controls are sized for a finger, text fields are 16px so
iOS doesn't zoom in when you tap them, and the notch and home bar are respected. It is tested down to a 320px-wide
phone, in landscape, and on a tablet.

- **Timeline**: one lane per task, positioned in time and coloured by tier (fast/strong). It shows nested hand-offs, refused requests, LM Studio swap markers, and "parallel saved Xs".
- **Inspector**: every Jev decision for the selected turn, with probabilities and threshold markers: routing, tier, skill Choice + fits, memory kept/checked, collaborators and hand-off approvals.
- **Artifacts**: files produced in this session. HTML is previewed live in a sandboxed iframe, images and video inline, and code or text as source.
- **Memory**: facts (with "forget"), learned skills and lesson notes.
- **Browser**: the worker's docked browser, live.
- **Review**: sessions flagged for review (see below), with their notes, a model's review and your verdict.
- **Status**: the harness version, with new commits on GitHub and an Update button (a count on the tab when there are any); which model LM Studio has loaded (plus the gate's state), Jev adapter and semif health, ComfyUI, GPU memory, and the swap history.

**Tool calls are summarised, not dumped.** A turn that ran 31 steps says so in one plain sentence ("Searched the web
3 times, looked at 6 files, edited 2 files and ran 4 commands · 1 failed"). Click it for one line per step
("Read jevh/web/server.py"), and click a step for its code and result. While a turn runs it is one live line
("Reading jevh/web/server.py..."), and the model's reasoning, when you have it open, reads as a quiet paragraph
under it. **Appearance** (the palette at the bottom of the rail) sets how much you see: Compact, Steps,
Verbatim or Hidden, plus the theme, accent, type, width, density and corners. See [docs/dashboard.md](docs/dashboard.md).

The persona, model tier, Watch browser, project folder and context ring sit under the message box. Pin a persona or
set the tier (Jev decides / fast / strong) there. Click an answer's avatar to inspect that turn,
and reload any time: the event history replays.

**The message box works while a task runs** (see Talking to a running task): what you send is added to the task's
prompt, or answered at once if it is a question; `/queue` makes it wait its turn (the queue shows above the box). Each
message shows in the turn with what became of it. On a phone the same box and the same Stop button are there.

**Security:** the agent executes code, so the dashboard binds to 127.0.0.1 only.
- It rejects foreign Host headers, which blocks DNS rebinding.
- Every API call needs a per-launch token embedded in the page.
- It refuses POSTs from other origins, which blocks CSRF from websites you visit.
- It serves files only from the workspace. HTML artifacts get a CSP sandbox, so a generated page can't reach the dashboard.

**From your phone (Tailscale):** the dashboard stays bound to 127.0.0.1; `tailscale serve` is the way in, and it is
tailnet-only (never `funnel`). Run `tailscale serve --bg --https=10000 http://127.0.0.1:8740`, then
`jevh dashboard --no-browser --allow-host <this-machine>.<tailnet>.ts.net:10000`. It prints a link ending
`?key=...`; open it once on the phone, which keeps a cookie. The page carries the API token, so the key is what
stops other devices on your tailnet from fetching it. The key is new each launch, so keep the link private; opening a
new launch's link replaces the cookie the phone kept from the last one.

### Session review

When a session went wrong in a way no detector caught, flag it: **Flag for review** at the top of the chat (or the
**flag** on any answer, to point a note at that turn), or `/flag <note>` in the CLI. Add as many notes as you
like on what looked wrong. In the **Review** tab, **Ask a model to review** has a model (config
`review.model`, else `models.strong`) read the whole session against your notes: the conversation, every
task's work log, and the failures already logged. It lists the problems it found (severity, turn, the note
each bears on, the evidence) and says **failure** or **not a failure**; you can overrule it either way. A
failure verdict is written to the failure log as `review_failure` (source `review`), so `jevh.refine` can
work on it; overruling it with "not a failure" marks that row ignored. Flags and notes live in
`workspace/reviews.db`; `jevh review` does the same from a terminal, for any session
(`sessions` lists them).

### Notebook

A place to jot down ideas and problems to come back to. In the dashboard, **Jot** (top bar) opens the
**Notebook** tab with the cursor in the box: pick *Idea*, *Problem* or *Note*, write (a little markdown
works), and press Ctrl+Enter. Entries can be marked done (a problem: solved), pinned to the top, edited,
deleted, filtered by kind and searched. An entry remembers the session it was written in; for a problem
that was really a session going wrong, **Flag session** sends it to the Review tab with the entry as the
note. In the CLI: `/idea`, `/problem` and `/jot` write an entry, and `/notebook [ideas|problems|notes|done|all|words]`
lists them (`/notebook done|reopen|pin|unpin|rm|show <id>`); `jevh notebook` does the same from a terminal.
The notebook is only for you, so nothing in it reaches a worker or Jev. It lives in `workspace/notebook.db`.

## Fast tier: Jev picks the model per task

Before each task, including every hand-off, Jev asks one two-way Choice on the task text: *fast
model* (quick answers, lookups, short scripts, small edits) or *strong model* (deep reasoning, hard
debugging, big builds). Easy tasks go to `models.fast` in `config.yaml` (ornith). Hard ones keep the
persona's own model. The call takes ~0.2s. If Jev is down, the persona's own model is used, since a slow right
answer beats a fast wrong one.

- `evals/tier_cases.py`: 39 labelled tasks at **100%**, and 15 held-out tasks at **15/15**, with 0 hard tasks sent to fast.
  Giving Jev the persona's description made it *worse*: simple dev and design tasks drifted toward "strong".
  So it sees the task text only.
- Opt a persona out with `fast_ok: false`. Delete the `models:` block to turn tiering off.
- CLI: `--tier auto|fast|strong`, or `/tier ...` mid-chat. The trace shows `tier: fast (P(strong)=0.01 vs 0.55)`.
- **One model at a time in LM Studio** (`jevh/lmstudio.py`, `exclusive: true` on the provider).
  LM Studio will happily JIT-load a second big model next to the first. Both then spill out of VRAM, and
  a one-word reply took 15–19s. The harness **unloads** the current model before **loading** the next one,
  over LM Studio's REST API, and waits for in-flight requests first. A swap takes about 12–23s. A warm ornith
  call is ~0.5s again. The trace shows `LM Studio: unloaded X -> loaded Y (12s)`.
- **Linger** (`swap_linger: 8`): after its last request, a model stays loaded briefly, so the same worker's
  next step keeps it. In parallel turns with mixed tiers, that means one swap per worker rather than one per step.
  It was verified live: 3 ornith steps finished, then a single swap to Qwen. A waiter for the other model gets it once the
  current one has been idle that long, or after `max_wait` (180s).

## Parallel hand-offs

Workers can hand off several jobs at once. Each approved hand-off runs on its own thread, with its own REPL process:

```python
results = request_all([("front_end_designer", brief), ("high_end_dev", spec)])   # both at once
h = request_async("comfyui_manager", "hero image ...")   # keep working...
img = wait(h)                                             # ...then collect
```

`request()` is `request_async()` followed by `wait()`. `wait(h, timeout=...)` returns `{"status": "running"}` if the hand-off isn't finished yet.
If a hand-off is never waited on, it is still finished and its files are kept, so no work is lost. The task board is locked,
so parallel requests can't exceed the depth/task limits between them. The CLI shows one progress line
for every persona that's generating, each task's duration, and a summary such as `hand-offs: 285s of work in 206s wall
(parallel saved 79s)`. Your semif bridge serializes forward passes, so parallel Jev calls queue safely, and
LM Studio runs concurrent requests in parallel (measured with 2 streams at once: 3.5s vs 3.1s for one).

Test with one fast model for every persona: `jevh --model lmstudio:ornith-1.5-35b-a3b`.

Safeguards added after watching a weaker model (ornith) work:
- If the same error repeats 3 times, the worker is told to stop retrying and report the blocker.
- Workers get a "last step, call final() now" warning.
- An unreachable ComfyUI raises `ComfyDown` ("not running, don't retry").
- `import comfy` works inside the REPL.
- `final()` without artifacts still ships the files the worker wrote.
- `save_artifact` keeps the source file's extension.

## The final() gate: work is checked before it's accepted

`final()` used to be taken on trust. Now `jevh/verify.py` checks every worker's work (hand-offs
too) before it's handed over:
- **Files:** each artifact exists and isn't empty; pages have no placeholder text (lorem ipsum,
  `[Your Company]`, TODO) and no links to files that don't exist; pages load in a headless browser
  without script errors or a blank body; Python compiles; JSON parses; images are images.
- **Project:** if the worker changed files in the open project, the project's tests run
  (pytest / npm test, detected; or `verify.project_command`), and they must pass.
- **The answer:** Jev compares it with the work log. Does it claim things the log doesn't show
  happening? Does it deliver what was asked, or only a plan? Is it an accurate report? And does it
  say plainly what isn't done? Plain rules add what Jev can't judge: counting deliverables
  ("three logos", one file), "checked it in the browser" with no browser use, and "next I'll…" stops.

Problems go back to the worker, whose REPL state is intact, as "not accepted yet: …". After
`verify.max_bounces` (2) the answer is accepted with the unresolved problems listed under it,
never silently. The Inspector shows every check and score.

Measured (`evals/verify_cases.py`, answer check):
- Tuning set: 12/12 bad answers caught, 0/10 false alarms.
- First held-out set: only 1/5 caught before I added the "accurate report" question.
- Second, fresh held-out set, run once: 4/6 caught, 0/6 false alarms.

It catches loud false claims reliably and about two-thirds of subtle ones (a number or item that
differs from the log). It hasn't flagged an honest partial answer yet. The file, page and test
checks are exact.

Live on uncapped Qwen: fixing the tip-calculator project, the gate ran the project's tests itself
before accepting. Building a bakery landing page (30 KB, real copy, no placeholders), the page was
opened in a headless browser and passed. (Both runs also showed the gate sending back correct work
listed by a bare file name; `final()` now resolves those.)

## Long tasks

Workers used to stop after 10-24 steps, with 120s per code block and a conversation that only
grew. Now:
- **Bigger budgets:** 20-60 steps per persona (dev 60, designer 40), and 30 minutes per code
  block (Stop ends one sooner). Long commands run as background jobs:
  `j = start_job("npm run build")`, `job_status(j)`, `job_wait(j)`. They end with the REPL.
- **Compaction:** past `limits.compact_tokens` (60k; Qwen has 100k of context loaded), older
  steps condense into a progress log inside the task message: code heads plus output snippets.
  The last `keep_steps` (3) steps stay verbatim, and variables and files are untouched. If the
  log itself grows past ~12k chars, the model summarizes it (what's done, what failed and why,
  what's left). It only fires when there's real old material, never just because the system
  prompt is big.
- **Progress check:** every `progress_every` (8) steps, Jev asks whether the recent steps repeat
  the same attempt with the same result (`evals/progress_cases.py`: 6/6 stalls caught, 0/8 healthy
  runs interrupted, over thresholds 0.3-0.8). The first time, the worker is told to step back and
  change approach. The second time in a row, it stops, keeps its files, and can be resumed. It
  replaced a "stuck?" wording that interrupted a healthy test-fixing run.
- **Checkpoint and resume:** after every step the worker's conversation is saved to
  `checkpoint.json` in its working folder. A task that was stopped, hit its step limit, stalled,
  crashed, or was cut off by a harness restart can be resumed in the same folder, with its files
  intact and a fresh REPL:
  - "Resume this task" under the answer in the dashboard;
  - `/resume` (or `/resume list`, `/resume 2`) in the CLI;
  - `Harness.resume()`.

Live on uncapped Qwen, a multi-part job ("make tipcalc a proper package: CLI with an exact-cent
breakdown, thorough tests, README") first crashed at step 2. That was a harness bug: a display
handler's error killed the worker. Event handlers are now shielded, and that's covered by a test.
The resumed run took 12 more steps and 17.5 min: 62.0k tokens in, 32.7k out, 26.8k of them
thinking. It restructured the code into a package with integer-cent math (100 split 3 ways ->
38.34 / 38.33 / 38.33), 25 tests and a README, and the final() gate ran the tests itself before
accepting. Compaction didn't fire in that run (about 5k tokens per step even with the threshold at
12k), so it is covered by tests so far, not by a live run.

## Usage tracking

Every model call, Jev call and LM Studio swap is recorded in `workspace/usage.db` (`jevh/usage.py`),
tagged with session, turn, task, persona, model and tier:
- **Model calls:** prompt, answer and hidden-thinking tokens (LM Studio reports
  `reasoning_tokens`), plus time. Background learning calls are recorded separately as `learn`.
- **Jev calls:** the harness's own decisions (`jev`), and the Jev calls made inside a worker's
  browser and web tools (`jev_tools`), reported by the REPL after each code block. The semif
  bridge now returns how many tokens each request processed.
- **Swaps:** LM Studio model swaps (seconds).
- **Cost:** only for models you price in config (`prices: {model: {input: $/M, output: $/M}}`).
  Local models cost nothing here.

Where you see it:
- Each answer shows a line like `12.1k in / 597 out (308 thinking) · model 24s · Jev 19.0s, 33 calls`,
  in both the CLI footer and the dashboard.
- `/usage` in the CLI covers this session, the last 24h and the last 7 days.
- The dashboard shows each turn's usage under its answer and the session total in the context popover;
  `GET /api/usage` has the breakdowns by kind, model and persona, and per-turn history.

It showed right away that on a web lookup, 16 of the 19s of Jev time is the web tool's
passage picking, not the harness's decisions (3.1s for 14).

## Qwen thinks as long as it needs

Personas on Qwen use `think: unlimited`: no `max_tokens` cap and no "reply with a short plan
first" nudge. Quality over speed. The fast tier (ornith) keeps its cap and nudge. Learning prompts
don't ask for brevity either (a new turn still drops a learning call in progress).

## Stop and ask

- **Stop** (dashboard button, or Ctrl-C in the CLI) ends the turn everywhere at once:
  - A model reply in progress is cut off mid-stream, so LM Studio stops generating (measured live:
    261 ms after the click, 100 tokens into a long essay). While the model is still *reading its prompt* (nothing
    streams yet) Stop used to wait for the first token, ~22 s on a 6k-token prompt: closing the connection from
    another thread does not wake a read that is waiting for the server (Windows), so a reply is now read on its own
    thread and Stop, like a redirect, stops waiting at once and hangs up.
  - Running REPL code is killed along with everything it started (e.g. the browser's Chromium).
  - Hand-offs stop, and new ones are refused.

  The stopped message stays in the chat history, so the next one can redirect it: live, "actually,
  just give me one short paragraph instead" after stopping a 3000-word essay was recognised as a
  follow-up and answered with the paragraph.
- **ask_user(question, options=None)** lets a worker ask something mid-task and wait, keeping its
  variables and an open browser page. In the dashboard the question appears in the chat with option
  buttons, and the message box answers it. The CLI prompts in the terminal. The browser tool's guide
  uses it to confirm orders, payments, sending and deleting before `allow_irreversible=True`.
  Unanswered questions time out after `limits.ask_timeout` (30 min) and the worker carries on.
- Background learning yields to you: starting a new turn drops a learning call still in progress
  (on Qwen, writing a skill thought ~1100-1400 hidden tokens, 45-60s, while you'd be waiting).

## Talking to a running task

A message sent while a task runs is **added to the task's prompt**: the message box (and the CLI) stay open and what you
type reaches the running worker. Only a question about progress is handled differently, and queueing and stopping are
commands, never guessed from the words:

| You send | What happens |
|---|---|
| a plain message ("actually check my calendar", "use km", "also write a haiku") | added to the task's prompt: the model's reply in progress is cut and asked for again with your message (the code it was writing never runs); code that is running is left alone and your message arrives with its output; a `final()` in flight is held until the worker has heard you. The step is not charged to its step limit. `/steer <text>` always adds, skipping the check below |
| a **question about progress** ("how's it going?") | an instant line from the worker's state ("step 1 of up to 60: running code: import time; ..."), then the task's own model answers in its own words from what it has done so far, while the task carries on. Jev tells (one Choice call, ~0.4 s); `/btw <question>` or `/status` says so. Added to the prompt instead, a question could end the task with a status report |
| `/queue <text>` | held until the task ends, then runs as the next turn by itself, in order (shown above the message box) |
| `/stop` | like the Stop button (which also drops what was queued, and puts it back in the message box) |

The same check notices a message that swaps the request ("check my calendar instead"): it is added all the same, but it
then stands alone as the request the final() gate judges. Any other message is added to the request, and where they
disagree the latest wins. The gate's file check follows the latest message when it names a file (live: a redirect from
`fib.py` to `squares.py` was first bounced for not handing over `fib.py`).

How it was checked (2026-10-03/04, Qwen 3.8 27B in LM Studio, semif Jev; `evals/interject_cases.py`,
`evals/bench/interject.py`; details and the vLLM notes in [docs/architecture/16-interjection.md](docs/architecture/16-interjection.md)):

- **Telling a question from everything else** (156 labelled messages in three sets): 155 right (dev 67/68, held-out
  40/40, fresh 48/48). Every question but one ("why did you pick that search query?", which is added to the prompt)
  was answered on the side, and **no correction, other request or stop was answered instead of added**. The first
  design had Jev also queue "another request" and stop on "stop" (66/68, 35/40, 46/48 on four-way labels, 0 mistaken
  stops), but in use a correction ("Actually could you just tell me what's on my calendar", sent while a picture was
  being made) was filed as another request (0.63) and queued behind the task: queueing and stopping are commands now.
  The fresh set is not an untouched test any more (the `replace` wording was widened and a bar lowered after it was
  seen). A "stop" guard question (does the message ask for anything else?) was tried and dropped: plain stops scored
  0.5-0.85 on it. "Actually check my calendar instead" scored as a stop at 0.95 in the first wording, which is why the
  replace option exists. The live check of the new default (`evals/bench/interject.py --stage add`, 2 reps, all
  checks): "also write me a haiku about autumn", sent while a script was being written, was added to the prompt, not
  queued, reached the task in 0.8 s, and the task did both.
- **Cutting a reply**: a correction sent 9.7 s into a step landed at 28.5 s (the first token), not at once: a read that
  waits for the server cannot be woken from another thread. Fixed (reader thread); now the cut lands in the same second,
  and Stop during prompt reading with it. A redirect while the model was writing makes the server read 6-10% of the
  prompt again (the rest from its cache), but the retry races the server freeing the cancelled slot: with no pause 2 of 7
  retries started on the other slot and read the whole prompt again; with a pause of 0.5-2 s (`interject.retry_delay`,
  default 0.5) 0 of 15 did (0 of 5 at 0.5 s, and the final bench at 0.5 s agrees). One while the model was still reading the prompt made it read all of it again with no pause and
  29% with the pause (two reps each).
- **Quick reply**: 4-12 s on Qwen (a thinking model; digest median 6.4 s, full 4.5 s over two bench reps), the task
  untouched; the digest context (default) is safe at any
  moment, the full context is cheap only while the task is idle (all 2,029 tokens were read cold when it was not).
- Found on the way: with Qwen and ornith both loaded, LM Studio read prompts at ~30 tokens/s and generated at 0.9
  (708 and ~31 with Qwen alone); the harness's own gate keeps one model loaded, Hermes, on the same server, does not.
- The bench also found a gate quirk older than this work: with `final()` called without a file list the harness collects the worker's files as `<persona>_<name>`, and the check for a file the request names compared names exactly, so `primes.py` was "not handed over" while `high_end_dev_primes.py` was. It now counts.
- Not run on vLLM; the doc lists what its documentation says and what to check.

## Tools: web, browser, project, nango, MCP

A persona lists its tools; per task Jev decides which tools' guides go into the prompt (one noul
each, ~0.6s for four; `thresholds.tool`). The rest are still importable and named in one line, so a
missed pick costs one `help(name)`, not the capability. The Inspector shows the scores.

- **web**: `web.ask(question)` searches (DuckDuckGo, no key), lets Jev rank the results, reads the
  best pages in parallel (a slow site is skipped), and returns only the passages that answer the
  question: e.g. 30k chars of Wikipedia -> 1-2k chars, in ~3s. `web.search`, `web.read(url, question)`.
- **browser**: a Jev-driven Chromium. The worker states one outcome per call
  (`browser.do("add the blue mug in size L to the cart")`), and Jev picks every click, field and dropdown,
  so the worker's model never reads a page snapshot. The design is adapted from
  [jev-browser](https://github.com/Ying-Kai-Liao/jev-browser) (MIT): the planner states outcomes, Jev
  decides, Playwright acts. It was reworked for local semif, whose latency grows with every question
  and which caps a Choice at 16 options:
  - The action kind follows from the element (field -> type, dropdown -> select, else click) instead
    of a separate question.
  - Pages with more than 16 elements are narrowed to a group of them first.
  - Values go into fields whose label obviously matches them (email -> "Email") without asking Jev.
  - Before clicking a form's button, one small check ("does a setting still differ from the goal?")
    stops premature submits.
  - Orders, payments, sending and deleting stop with `needs_confirmation` unless the user asked
    for it (`allow_irreversible=True`).
  - A goal that says "save" or "continue" isn't done until something was actually submitted.

  Results on `evals/browser_cases.py`: 16/16 tuning goals, 10/12 on the held-out pages on the first
  run (both misses were multi-part forms, since fixed), then 29-30/30 with 0 false "done". That's
  ~2s per goal on semif and ~5s on big Wikipedia pages. `browser.read(question)` works like
  `web.read`. Set `tools_jev: cloud` (and TYPESAFE_API_KEY) to use the hosted Jev instead: 255
  options per Choice, but page text leaves the machine.

  **Real websites** (`evals/bench/browser_tasks.py`): 10 full harness turns on live sites - Google
  Flights, BlazeDemo, Red Rocks -> AXS tickets, Target, the demoqa form, books/quotes.toscrape, play2048,
  Booking.com, Hacker News. Nothing is bought, booked or signed into: tasks stop at sign-in or payment,
  and a simulated user says "stop here" to confirmations. The local eval never exercised most of what
  failed there, all fixed and covered by tests:
  - A worker that started Playwright itself broke every later `browser` call ("Sync API inside the
    asyncio loop"). The shared session now runs on its own thread.
  - Links opening a new tab: the tab registered after the check, the click was banned as "no effect",
    and the next event's link was clicked (three wrong concerts). New tabs are now waited for and followed.
    A link is never "irreversible" ("Buy Tickets" only goes to the ticket site).
  - Bot walls (AXS's "Just a moment...", Target answering its own APIs with HTTP 435 so search said "no
    match") are reported as `blocked`. The agent does not get past them: no stealth, no faked headers.
  - React forms: labels in the next column, placeholders that are examples ("name@example.com"),
    autocomplete suggestions as plain divs (picked when one matches what was typed), date pickers that
    silently revert a format ("12 March 1990" -> retried as "12 Mar 1990" and read back), radio/checkbox
    choices given as values ({"Hobbies": "Reading"}), forms that render a second after load (Booking.com).
  - `do()` never submits a form the goal didn't ask to send (it clicked Submit after "fill name and email"),
    never presses Enter in a form field, and tells identical buttons apart by their row ("Choose This
    Flight" x5 -> "(in: 9696 Aer Lingus $200.98)").
  - A mistyped site the request names is corrected ("blavedemo.com"); a 404 is an error, not "done".
  - New: `browser.press(keys)` (games, Escape), `browser.html(css)` (scraping what text doesn't show);
    `read()` lists what the form fields hold, so a worker can check its form without reloading it.
  - A stalled worker gets one last step with only the `final` tool to report what it found and what
    blocked it, instead of the user getting raw REPL output ("105 / Are You Human?").

  Most browser tasks go to the fast tier; its slips (retyped URLs, guessed URLs, reloading a half-filled
  form) are much of what still fails. `--tier strong` runs every task on the strong model. Last runs
  (2026-09-29): fast tier (ornith) 6/10 plus a partial, strong tier (Qwen 3.8 27B) 8/10 - the strong
  model also filled and submitted the whole demoqa form and BlazeDemo's passenger details. Both miss
  play2048 (a canvas game) and Booking.com (its search button didn't take the chosen dates); every miss
  was reported as one, with where it stopped.
- **project**: the user's folder (dashboard "Project" field, `--project`, `/project`).
  - `tree`, `grep` and `read` (numbered lines).
  - `find(question)`: Jev ranks the files by path and first line.
  - `edit` (exact, unique replacement) and `write`, confined to the folder, with every original kept
    for `undo()`.
  - `sh(cmd)`: runs a command in the project folder.
  - Only personas that list it (dev and designer) get it, and only while a project is open.
- **nango**: the user's own accounts (Gmail, Google Calendar and Drive, GitHub, Slack, Notion, HubSpot,
  and ~700 other APIs) through [Nango](https://github.com/NangoHQ/nango). Nango holds the OAuth tokens
  and refreshes them. The worker calls each app's own REST API through Nango's proxy
  (`nango.get("google-mail", "/gmail/v1/users/me/messages", {...})`), so no token reaches a prompt.
  - **Setup is a conversation.** Say "Connect my Google Calendar" (or "Set up Nango"). The `nango-setup`
    skill has the assistant start Docker and Nango for you the first time, read you each step at the
    provider's website one message at a time (opening the page in your own browser), ask for the app's
    Client ID and then its secret (typed hidden: not in the transcript, memory or logs), open Nango's sign-in
    page and wait for you, and check the connection. Details, and what to do by hand: [SETUP.md](SETUP.md#connected-accounts-nango);
    how it works: [architecture 08](docs/architecture/08-tools.md#nango-setup-the-walk-through-and-what-code-does-for-it).
    It can also click through the provider's console in a window you sign in on (it never types a
    password; if Google refuses an automated sign-in it says so and carries on in your browser). Untested
    on Google's own console (it needs your sign-in): run live only against a practice console page
    (`evals/browser_site/oauth_console.html`, the `drive-mock` scenario).
  - The tool is offered before Nango is set up, so the request reaches it; Jev's tool pick sees which
    accounts are connected ("my calendar" with Google Calendar connected).
  - **Changes are approved by the user, not the model.** A POST/PUT/PATCH/DELETE that isn't a search or
    query goes to the user as a question (`Allow` / `Allow all <app> changes this task` / `Don't allow`)
    showing the method, path, params and body. A request with `base_url=` (Nango then sends your token to
    that host) is always asked, even a read. It's asked from inside the tool, so the worker can't approve itself,
    and an email it read can't talk it into forwarding the inbox. `tools.nango_writes: deny` turns
    changes off; `allow` skips the question.
  - Limit: this gate guards the tool's normal path, not a sandbox. The worker's REPL can read
    `NANGO_SECRET_KEY` (or `workspace/nango/secret_key`), like any other key the harness runs with.
  - Tool pick (`tool_cases.py`, live on semif): all 10 account requests scored >= 0.99 (a draft email 0.87),
    everything else <= 0.25 except "how do I use the Gmail API from Python" (1.00, a needless guide).
- **MCP**: stdio JSON-RPC client (`jevh/mcp.py`). The servers' tools form one catalog, and Jev picks
  the few that fit a task (Choice + fits noul, like skills). Only those signatures reach the prompt:
  `mcp("server.tool", arg=value)`. A broken server is reported and skipped.

Seen live (ornith as every worker, semif as Jev):
- "latest Playwright Python version": correct (1.63.0, 2026-09-15, checked on PyPI) in 20s.
- "search Wikipedia for tidal locking, which Pluto moon..., screenshot": Charon plus a screenshot, in 26s.
- "tests in my project are failing, fix them": found `tip_pct` not divided by 100, fixed it, and
  both tests passed, in 22s.

## Memory and learning

- **Everything is searchable.** Turns live in SQLite with a full-text index (`workspace/memory/memory.db`,
  Porter stemming; the old jsonl history is imported). Memory candidates are:
  - the persona's recent turns and a few from other personas, as before;
  - any past turns that share words with the request (`limits.memory_search`);
  - remembered facts: those matching the request, topped up with the newest (`limits.memory_facts`).

  Jev's gate is unchanged. With the history buried under 160 newer turns, 14/19 needed turns reached
  the worker, against 1/19 for the recent-window design, with no extra latency.
  Workers can also `recall(query)` (search everything, Jev keeps the matches) and `remember(fact)`.
- **Facts.** After a successful turn, one Jev noul asks whether the user stated something lasting
  (12/12, 0 false positives on `learn_cases.py`). Only then does the task's model, already loaded,
  write 1-3 short facts. A new fact retires an older one it makes redundant or outdated (5/5, 0
  false), so "trains at Gold's Gym" is replaced, not contradicted. Live: "my gym is closed on
  Sundays..." was remembered, and the next day's plan from a different persona respected it.
- **Skills.** After a multi-step task, a noul asks whether it involved non-obvious steps, fixes or
  settings (3/3, 0 false). If a skill was loaded, lessons are added:
  - to the SKILL.md itself for skills the harness owns;
  - to `skills/_lessons/<name>.md` for others, e.g. Hermes skills, whose files are never edited.

  Otherwise a new `skills/learned/learned-<name>/SKILL.md` is written. **Learned skills start as
  drafts**: workers see them labelled "learned from one run, verify its claims". Approve or delete
  them in the dashboard's Memory tab. This matters: live, ornith returned Hacker News's *all-time*
  top stories for "current top 5" and a skill claiming "current" was learned from it.
- Everything runs on a background thread after the turn has returned. The Memory tab lists facts
  (with "forget"), learned skills and lesson notes. `learning: false` in config turns it all off.

## Persona creation

When no persona fits, the harness drafts one (`jevh/forge.py`, design in
`docs/architecture/15-persona-creation.md`). Two triggers:

- **A request no persona fits.** After a turn the general assistant (the default persona) handled, Jev is
  asked which assistant has the expertise the request needs, with "none of these" as an option; it does not
  depend on how confident the router was (live, specialist requests went to the general assistant at
  0.85-1.0). Requests it says none covers are pooled, and when `min_cluster` that suit the same subject
  have gathered (all of the pool is compared, and a cluster must suit each other, not just the newest
  request), a persona is drafted for them.
- **A learned skill no persona wants.** A new learned skill is shown to Jev blind (a Choice over the
  personas plus "none of these"); if none claims it, a persona is drafted around it at once.

A draft is a file in `personas/learned/` (not committed) and is **live straight away**, so you can try
it, but fenced: no MCP, no `can_request`, the REPL tools in `persona_creation.tools` (none by default),
and the skill that made it, which it now **owns**: no other persona is offered that skill. Before it goes
live it must pass a routing gate: win the requests that made it, and take none of the last
`gate_probes` requests that an existing persona clearly has the expertise for. One that fails is written **inactive** (with the reason)
for you to switch on or delete. A new description that Jev finds to suit the same specialist as an
existing persona is not created, and at most `max_live_drafts` drafts wait for approval.

Review in the dashboard's Memory tab (approve / switch on / delete, and "check learned skills" for
skills learned before) or the CLI (`/personas`, `/persona approve|activate|delete <name>`, `/forge`).
Deleting a draft gives its skills back to the personas that were offered them. `persona_creation.enabled:
false` turns it all off.

**Live bench** (`python evals/bench/persona_creation.py`, 2026-10-02, semif Jev, Qwen 3.8 27B writing, 3 reps
per case; last summary in `evals/bench/persona_creation_last_summary.txt`). The first run found that persona creation did
not work end to end: the "no assistant fits" trigger never fired (it was only asked when routing was unsure, and
routing was never unsure about a specialist request: 0 of 14 asked), and the routing gate rejected every draft
once memory held the request the persona came from (0 of 27 went live). Fixed, then measured again:
- **Trigger:** specialist requests are pooled 39/42 (92%), ordinary ones 6/60 (10%) over four case sets (tuning,
  held-out, two fresh ones, the last written before it was scored). The flagged ordinary ones are chores no
  persona description lists (naming ideas, a slogan, "how does compound interest work", a cover letter). A
  flagged request only pools: a draft needs 3 that suit each other, then your approval.
- **Clustering** (shuffled streams of 62 flagged requests: 9 real domains, one-offs, chores): all 9 domains
  formed on both streams, 2 of 20 clusters mixed (insulation-business chores: a slogan, crew names, R-value).
  The version before compared only the last 8 pooled requests with no coherence check and formed 1 of 18 domains.
- **Drafts:** 27/27 (9 domains x 3) written and live, 0 steals, 27/27 fence and validity checks, 54/54 new
  in-domain requests routed to the draft, 24/24 orphan-skill drafts live and owning their skill, 0 of 9 skills
  with a home given one, 27s per draft. The three-reps rows include the held-out domains.
- **No harm to what worked:** the 37-request routing set is 36/37 with and without 9 drafts live; no draft takes a
  request an existing persona owns; a near-duplicate of `fitness_coach` and a general-purpose catch-all are
  refused; 4 of 4 hand-written duplicates stopped. Full turns: three requests pooled, a draft written in the
  background, the fourth request routed to it (conf 0.99-1.0) and answered, for a tuning and a held-out domain.
- **Known gaps:** about 1 in 10 ordinary requests is pooled (see above; describing what `daily_assistant` is for
  would keep them with it); the writer sometimes returns an empty reply (one retry now); a lukewarm group
  (cricket tactics vs rules) can need a 4th request; chess "endgames" and "openings" do not suit each other.

`persona_creation.enabled: false` turns it off. Moving skills between existing personas is a later step.

## Failure log

Every way a task goes wrong is written to `workspace/failures.db` (`jevh/failures.py`), to improve
the harness from later. Logging only watches the harness's events; it never changes how a task runs.
Failures come from five sources:
- **harness**: a crash, the step limit, a stall, going in circles (progress check), a reply cut
  off or empty, Jev unreachable, a hand-off refused, the user pressing Stop.
- **gate**: the final() gate sent work back (`gate_bounce`, with the answer it rejected), or
  accepted it with problems still listed (`gate_unresolved`).
- **jev**: after every task that ended normally, one Jev call reads the request, the answer and the
  work log: does the answer say it couldn't (fully) do it (`gave_up`), or does it not deliver what
  was asked (`missed_request`)? `failures.jev_check: false` turns it off. The questions aren't
  tuned yet: run `evals/failure_cases.py` and set `failure_gave_up` / `failure_missed` from its sweep.
- **model**: a worker calls `report_failure(what, detail)` in its REPL: a tool that misbehaved, a
  skill or instruction that was wrong. It's logged and the worker carries on.
- **hook**: your own checks in `failures.hooks`, run on any harness event. A hook is a shell
  command (the event arrives as JSON on stdin; print a summary or `{kind, summary, detail}`, or
  exit 2, to flag a failure) or a Python function `module:fn(event)`. A `done` event carries the
  task, the result and the work log. A hook that breaks is shown, never logged as the task's failure.

  ```yaml
  failures:
    hooks:
      - name: no_sources
        on: [done]
        command: python checks/sources.py
  ```

Each row keeps the request, answer, work log, persona, model and the task's working folder (its
`checkpoint.json` holds the whole conversation), so a failure can be studied after the session is
gone. Rows have a triage `status` (open / reviewed / fixed / ignored) and a note.

Where you see them: new failures show in the CLI and the dashboard's activity feed as they happen;
`/failures [n|kind]` in the CLI; `GET /api/failures` and `POST /api/failures/status` on the dashboard
server; and `jevh failures` (list, `--kind`, `export out.jsonl`, `status <id> fixed "note"`, and
`add <kind> "<summary>" --workdir <task folder>` for a failure no detector caught: it reads the request,
persona, work log and answer from the task's checkpoint).

## Refinement: failures become tested, approved changes

`jevh/refine/` turns logged failures into harness changes, with local models doing the work
and you approving each change. Claude isn't part of it: tests and fixes are written by a local
model (`refine.model`, default `models.strong`, else the fixer persona's model), and a
non-local model is refused.

```bash
python -m jevh.refine groups                 # what it would work on: open failures by kind, persona and cause
python -m jevh.refine nightly --until 07:00  # overnight: carry on unfinished attempts, then start new ones
python -m jevh.refine status                 # every attempt, its stage and objective
python -m jevh.refine report <id>            # the approval report
python -m jevh.refine approve <id>           # merge it; its failures are marked fixed
python -m jevh.refine reject <id> "why"      # drop it; the reason is shown to the fixer next time
```

One attempt per failure group (same kind, persona and cause: failures of one kind can have different
causes), in `workspace/refine/<id>/`, through stages that survive a restart (a night cut short carries
on the next night). The cheap questions come first - does it happen again, and is it the harness's to
fix? - so a night isn't spent writing tests for a slip one model made once:
1. **Replay checker.** The local model writes a checker for the failed request. It must pass a good
   answer and fail the failed one and a plain refusal. For a task that hands over files it also gives
   the files a correct run would hand over; they're written to disk and the checker must pass them
   (and still pass when the answer only names the file).
2. **Cross-model replay.** The failed request runs on every local model, up to `repro_runs` times each
   (a model stops after 2 failures). If less than `min_repro` of the replays fail, there's nothing to
   measure and the attempt is dropped.
3. **Triage: the local model's own reading.** It reads the failure, its work log and the replays and
   says whether the harness caused it (`harness_caused`), could have prevented it
   (`harness_preventable`), or it was the model alone (`model_only`) or no failure (`not_a_failure`),
   with the component and the evidence. Only the first two go on; the others are dropped and their
   failures marked (reviewed / ignored). A failure that came back on 2+ models goes on whatever the
   reading says: a slip one model makes is noise, one several make points at what they were all given.
4. **Tests.** About 10 variants aimed at that cause (same weakness, different specifics), each with a
   checker that must hold up the same way; about 30% are held out and the fixer never sees them.
   Control tasks that already work (`evals/refine_controls.json`) catch a fix that breaks other things.
5. **Baseline.** Every case on every local model, on the old code. The replay runs often enough to
   expect about 3 failures (`repro_target`, up to `max_runs`): a 1-in-4 failure shows nothing in 2
   runs. Then a variant that never failed becomes a regression check (counted with the controls), and
   the local model looks at one that failed every run on every model: if the result was right and
   the checker wrong, it's set aside (out of the verdict, listed in the report).
6. **Fix.** The local model, as a harness worker with the `project` tool, changes a git worktree
   of the harness (branch `refine/<id>`), never your checkout. It gets the triage reading. Its edits
   to `jevh/refine/`, `tests/` and `evals/` are thrown away (a fix can't change how it's scored), and
   the offline tests must pass. A fixer that stopped before final() is flagged in the report.
7. **Compare.** The same matrix on the new code. Each run is its own process and workspace, and it's
   scored by the case's checker. Verdict: **improves** (the failing cases' pass rate up by at least
   `min_gain`, held-out cases not worse, no model losing controls or regression checks), **mixed**,
   or **no gain** (then another fix try, up to `fix_tries`).
   - **Cost kinds** (`cost_kinds`: `gate_bounce`): the task got done, with needless extra work (the
     final() gate sent a correct answer back). These are scored on gate bounces per run, which must
     fall by `min_cost_gain`, with the pass rate not falling. Bounces come from the harness's own
     gate, so a fix could win by making it lenient: the gate's offline eval
     (`evals/verify_cases.py --guard`) must catch as many bad answers as before with no more false
     alarms, or the fix is refused.
8. **Report** for improves / mixed: the failure, its replays per model, the triage reading, what
   changed (the fixer's account plus the diff), per-model old → new pass counts, bounces, model
   calls, time and tokens for dev, held-out, regression and control cases, set-aside checkers, and
   caveats (small samples with a p-value, models that got worse or slower, broken controls,
   thrown-away edits, dropped tests, an unfinished fixer).

Scale: cases × models × `runs` × 2 versions, plus the sized replay, at several minutes a run on a
local model; about 140 runs with 10 cases, 7 controls and 2 models, which is overnight. Lower
`runs` or set `refine.models` to trade confidence for time. Test writing uses `refine.max_tokens`
(65536) per reply: Qwen 3.8 27B took 18-21k tokens per reply, and 8192 or 32768 cut its lists off.

Seen live (2026-09-30), the gaps these stages close: all 4 gate bounces logged were correct answers
(a checker built to reject them is a wrong test); a local checker looked for file contents in
`artifacts` (which holds paths) and would have failed the correct planets.csv; its variants all
passed on the old code, and an XML checker that accepted one layout failed every run - so a real
fix scored "mixed" at p = 0.32.

Limits for now: checkers the local model writes can be `python` code; they run in their own
process, with the same trust as worker code. The cost guard is offline: a fix that rewords a Jev
question isn't seen by it. The dashboard doesn't show proposals yet; use the CLI.

## Limits / next

- The REPL is a separate process with its own working dir, not a security sandbox (no WSL/Docker isolation yet).
  With `project.sh` and the browser, a worker can now act on your files and the web: keep watching turns,
  and hold off on unattended use until the sandbox exists.
- Learning trusts the run it learns from: a wrong answer can become a (draft) skill. Review drafts.
- Skill picking still loads a needless skill now and then (live: `claude-code` on a bug-fix task).
  Lessons from such a run are now declined ("wasn't really about this task") rather than attached.
- Parallel hand-offs share one GPU model server, so speedups depend on its concurrency (LM Studio: yes).
- VRAM: semif (~10GB) sits on the 4080 and the workers' LLM on the 3090, so ComfyUI jobs need one of them unloaded.
