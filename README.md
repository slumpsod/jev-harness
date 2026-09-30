# jev-harness

A lightweight multi-persona agent harness. A System One decision model (**Jev**:
semif / laya / openjev behind `openjev/systemone.py`) is the gatekeeper for everything a worker model sees.
Workers (any LLM: LM Studio, llama.cpp, vLLM, Claude) see only the request plus what Jev lets through,
and they act by writing Python in their own sandboxed REPL.

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

## Run

On a Mac, Linux, or with the hosted Jev instead of a local one: see [SETUP.md](SETUP.md). A machine's own
settings (models, Jev, paths) go in `config.local.yaml`, laid over `config.yaml` and ignored by git.

```bash
openjev\start_systemone.bat          # adapter on :33001 (Jev); backend set in config.yaml
python -m jevh.cli                   # chat; /help for commands
python -m jevh.cli --as fitness_coach -c "plan my week"   # one-shot, pinned persona
python -m pytest -q                  # 126 offline tests (fake Jev + scripted LLMs, a fake MCP server)
python evals/routing_cases.py --sweep   # live routing/collab accuracy + threshold sweep
python evals/skill_cases.py            # live skill-picking accuracy + sweep (--offline to re-sweep)
python evals/memory_cases.py           # live memory-gating accuracy + sweep
python evals/tier_cases.py             # live fast/strong tier accuracy + sweep
python evals/tool_cases.py             # live tool-guide picking accuracy + sweep
python evals/browser_cases.py --live   # Jev browser agent on a local test shop (+ Wikipedia)
python evals/bench/browser_tasks.py      # 10 real-website tasks, full harness turns (flights, tickets, forms...)
python evals/learn_cases.py            # the three learning gates (fact / same fact / skill)
python evals/recall_cases.py 4         # memory reach when the history is buried under 160 newer turns
```

CLI commands: `/as <persona>`, `/auto`, `/project <path>` (or `--project`), `/facts`, `/forget <id>`,
`/personas`, `/skills`, `/trace` (raw Jev answers for the last turn), `/verbose` (show code + REPL output),
`/jev` (adapter health), `/reset`, `/quit`.

## Adding things

- **Persona**: drop a YAML in `personas/`. Jev routes on `description`, so make it concrete.
  `model` picks the provider from `config.yaml` (an optional `fallback:` is used if the primary errors);
  `skills_allowed` holds globs over the skill roster; `can_request` lists who it may delegate to (Jev's
  collaborator suggestions add to this per turn); `tools` are modules from `jevh/tools/` pre-imported
  into its REPL (`web`, `browser`, `project`, `nango`, `comfy`); `mcp` holds globs over MCP tools ("server.tool");
  `code: false` makes it a plain chat persona.
- **MCP server**: add it under `mcp_servers:` in `config.yaml` (`command`, `args`, `env`) and give a persona
  `mcp: ["server.*"]`. Servers start on first use.
- **Skill**: `skills/<name>/SKILL.md` with `name` and `description` frontmatter. The Hermes skills folder is
  also a root (hidden `.archive` dirs are skipped). A picked skill's first 6k chars go in the prompt, and the
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
| Persona routing | `routing_cases.py`, 33 requests | 97% top-1 | ~0.2s |
| Follow-up detection | `followup_cases.py`, 22 message pairs | 14/14 follow-ups kept, 0/8 new topics joined | (in the routing call) |
| Collaborators | `routing_cases.py` | 94% exact, 0 needless | ~0.2s |
| Skill picking | `skill_cases.py`, 63 requests (your 52 Hermes-router cases + 11) | 94% right skill, 93% correct "no skill", 3 needless | ~1.5s |
| Memory gating | `memory_cases.py`, 18 requests over an 18-turn history | 95% of needed turns passed, 0 leaks | ~1.5s |
| Model tier (fast/strong) | `tier_cases.py`, 39 tasks + 15 held out | 100% and 15/15, 0 hard tasks sent fast | ~0.2s |
| Tool guides | `tool_cases.py`, 54 requests x 5 tools | 50/54 exact, 3 needless (nango: 10/10 picked, 1 needless) | ~0.3s |
| final() answer check | `verify_cases.py`, 22 tuning + 2 held-out sets of 12 | 12/12 tuning; fresh held-out 4/6 bad caught, 0 false alarms | ~0.3s |
| Progress check (long tasks) | `progress_cases.py`, 14 work logs | 6/6 stalls caught, 0/8 healthy runs interrupted | every 8 steps |
| Browser agent | `browser_cases.py`, 16 tuning + 12 held-out + 2 Wikipedia goals | 29-30/30, 0 false "done" | ~2.3s per goal |
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
python -m jevh.web                 # http://127.0.0.1:8740/ opens in your browser
python -m jevh.web --no-browser --port 8740 --model lmstudio:ornith-1.5-35b-a3b
```

The chat is on the left, and every event streams in live on the right:
- **Timeline**: one lane per task, positioned in time and coloured by tier (fast/strong). It shows nested hand-offs, refused requests, LM Studio swap markers, and "parallel saved Xs".
- **Inspector**: every Jev decision for the selected turn, with probabilities and threshold markers: routing, tier, skill Choice + fits, memory kept/checked, collaborators and hand-off approvals.
- **Artifacts**: files produced in this session. HTML is previewed live in a sandboxed iframe, images and video inline, and code or text as source.
- **Status**: which model LM Studio has loaded (plus the gate's state), Jev adapter and semif health, ComfyUI, GPU memory, and the swap history.

In the header you can pin a persona and set the tier (Jev decides / fast / strong). Click an answer to inspect that turn,
and reload any time: the event history replays. **Security:** the agent executes code, so the dashboard binds to 127.0.0.1 only.
- It rejects foreign Host headers, which blocks DNS rebinding.
- Every API call needs a per-launch token embedded in the page.
- It refuses POSTs from other origins, which blocks CSRF from websites you visit.
- It serves files only from the workspace. HTML artifacts get a CSP sandbox, so a generated page can't reach the dashboard.

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

Test with one fast model for every persona: `python -m jevh.cli --model lmstudio:ornith-1.5-35b-a3b`.

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
- The dashboard's Usage tab has totals, breakdowns by kind, model and persona, and per-turn history.

It showed right away that on a web lookup, 16 of the 19s of Jev time is the web tool's
passage picking, not the harness's decisions (3.1s for 14).

## Qwen thinks as long as it needs

Personas on Qwen use `think: unlimited`: no `max_tokens` cap and no "reply with a short plan
first" nudge. Quality over speed. The fast tier (ornith) keeps its cap and nudge. Learning prompts
don't ask for brevity either (a new turn still drops a learning call in progress).

## Stop and ask

- **Stop** (dashboard button, or Ctrl-C in the CLI) ends the turn everywhere at once:
  - A model reply in progress is cut off mid-stream, so LM Studio stops generating (measured live:
    261 ms after the click, 100 tokens into a long essay).
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
  - Setup: [SETUP.md](SETUP.md#connected-accounts-nango). It's offered only when `NANGO_SECRET_KEY` is set,
    and Jev's tool pick sees which accounts are connected ("my calendar" with Google Calendar connected).
  - **Changes are approved by the user, not the model.** A POST/PUT/PATCH/DELETE that isn't a search or
    query goes to the user as a question (`Allow` / `Allow all <app> changes this task` / `Don't allow`)
    showing the method, path, params and body. A request with `base_url=` (Nango then sends your token to
    that host) is always asked, even a read. It's asked from inside the tool, so the worker can't approve itself,
    and an email it read can't talk it into forwarding the inbox. `tools.nango_writes: deny` turns
    changes off; `allow` skips the question.
  - Not connected yet: `nango.connect("notion")` returns a Connect link the worker hands to the user.
  - Limit: this gate guards the tool's normal path, not a sandbox. The worker's REPL can read
    `NANGO_SECRET_KEY` from its environment, like any other key the harness runs with.
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

## Limits / next

- The REPL is a separate process with its own working dir, not a security sandbox (no WSL/Docker isolation yet).
  With `project.sh` and the browser, a worker can now act on your files and the web: keep watching turns,
  and hold off on unattended use until the sandbox exists.
- Learning trusts the run it learns from: a wrong answer can become a (draft) skill. Review drafts.
- Skill picking still loads a needless skill now and then (live: `claude-code` on a bug-fix task).
  Lessons from such a run are now declined ("wasn't really about this task") rather than attached.
- Parallel hand-offs share one GPU model server, so speedups depend on its concurrency (LM Studio: yes).
- VRAM: semif (~10GB) sits on the 4080 and the workers' LLM on the 3090, so ComfyUI jobs need one of them unloaded.
