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

```bash
openjev\start_systemone.bat          # adapter on :33001 (Jev); backend set in config.yaml
python -m jevh.cli                   # chat; /help for commands
python -m jevh.cli --as fitness_coach -c "plan my week"   # one-shot, pinned persona
python -m pytest -q                  # 51 offline tests (fake Jev + scripted LLMs)
python evals/routing_cases.py --sweep   # live routing/collab accuracy + threshold sweep
python evals/skill_cases.py            # live skill-picking accuracy + sweep (--offline to re-sweep)
python evals/memory_cases.py           # live memory-gating accuracy + sweep
python evals/tier_cases.py             # live fast/strong tier accuracy + sweep
```

CLI commands: `/as <persona>`, `/auto`, `/personas`, `/skills`, `/trace` (raw Jev answers for the last turn),
`/verbose` (show code + REPL output), `/jev` (adapter health), `/reset`, `/quit`.

## Adding things

- **Persona**: drop a YAML in `personas/`. Jev routes on `description`, so make it concrete.
  `model` picks the provider from `config.yaml` (an optional `fallback:` is used if the primary errors);
  `skills_allowed` holds globs over the skill roster; `can_request` lists who it may delegate to (Jev's
  collaborator suggestions add to this per turn); `tools` are modules from `jevh/tools/` pre-imported
  into its REPL; `code: false` makes it a plain chat persona.
- **Skill**: `skills/<name>/SKILL.md` with `name` and `description` frontmatter. The Hermes skills folder is
  also a root (hidden `.archive` dirs are skipped). A picked skill's first 6k chars go in the prompt, and the
  full text is in `ctx['skills'][name]`.
- **Model provider**: any OpenAI-compatible server works as `type: openai`. Claude uses the `anthropic`
  SDK (`pip install anthropic`, and set `ANTHROPIC_API_KEY` or run `ant auth login`).

## How context stays clean

- Memory is on disk (`workspace/memory/*.jsonl`). The candidates are a persona's own recent turns plus a
  few from other personas, and only snippets Jev scores as relevant (`thresholds.memory`) reach the worker,
  as `ctx['memory']`.
- Delegated workers get no conversation history, only the brief and the artifact paths they were handed.
- Big data never goes into prompts: the worker sees `ctx` key names and sizes, and reads values with `peek()`.
- Hard limits hold no matter what Jev says: delegation depth, tasks per turn, cycle guard, per-exec
  timeout (the REPL is restarted on timeout), and an output cap.

## Measured (semif backend, 2026-09-27)

| Decision | Eval | Result | Per turn |
|---|---|---|---|
| Persona routing | `routing_cases.py`, 33 requests | 97% top-1 | ~0.2s |
| Collaborators | `routing_cases.py` | 94% exact, 0 needless | ~0.2s |
| Skill picking | `skill_cases.py`, 62 requests (your 52 Hermes-router cases + 10) | 94% right skill, 93% correct "no skill", 3 needless | ~1.5s |
| Memory gating | `memory_cases.py`, 18 requests over an 18-turn history | 95% of needed turns passed, 0 leaks | ~1.5s |
| Model tier (fast/strong) | `tier_cases.py`, 39 tasks + 15 held out | 100% and 15/15, 0 hard tasks sent fast | ~0.2s |

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
- VRAM: ornith (~21GB) and qwen3.8-27b can't both sit on the 3090, so a "strong" task on a local
  persona makes LM Studio swap models. A cloud strong model (e.g. Claude for the dev) avoids that.

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

## Limits / next

- The REPL is a separate process with its own working dir, not a security sandbox (no WSL/Docker isolation yet).
- Parallel hand-offs share one GPU model server, so speedups depend on its concurrency (LM Studio: yes).
- VRAM: semif (~10GB) sits on the 4080 and the workers' LLM on the 3090, so ComfyUI jobs need one of them unloaded.
