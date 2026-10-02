# 13 · Refinement: from logged failures to a tested fix

`python -m jevh.refine nightly` turns open failures into harness changes you approve. A local
model writes tests for the failure, a local model changes a separate copy of the harness, and both
versions run the same tests on every local model. You get a report with the numbers and the diff,
and nothing is merged until you approve it.

Everything runs on **local models** (`runner.is_local`): no Claude API, no keyed cloud endpoint.
The fixer works in a **git worktree** on branch `refine/<id>`, never in the running harness.

## The stages of one attempt

One attempt per failure group: the same kind, persona and cause (`Refiner.groups`,
`refine/loop.py:198`). Each attempt lives in `workspace/refine/<id>/` and saves its stage after
every step, so a night cut short carries on the next night. The cheap questions come first: does
it happen again, and is it the harness's to fix?

```mermaid
stateDiagram-v2
  [*] --> new: a failure group is picked
  new --> checked: the local model writes a checker<br/>for the failed request (the replay)
  new --> dropped: no checker held up
  checked --> replayed: replay on every local model<br/>(up to repro_runs 4 each, stop after 2 failures)
  checked --> dropped: failed < min_repro (10%)
  replayed --> triaged: triage: harness_caused / harness_preventable<br/>(or failed on 2+ models)
  replayed --> dropped: model_only / not_a_failure<br/>(failures marked reviewed / ignored)
  triaged --> generated: variants aimed at that cause<br/>+ control tasks
  triaged --> dropped: fewer than min_cases (6) held up
  generated --> baseline: every case × model × runs on the OLD code<br/>never-failed variants become regression checks,<br/>always-failed ones get their checker reviewed
  baseline --> fixed: the fixer changes the worktree,<br/>offline tests pass, committed on refine/id
  baseline --> dropped: no working change in fix_tries (2)
  fixed --> proposed: same matrix on the NEW code<br/>verdict improves / mixed → report.md
  fixed --> baseline: no_gain, tries left
  fixed --> dropped: no_gain, no tries left
  proposed --> approved: you approve: merged --no-ff,<br/>failures marked fixed
  proposed --> rejected: you reject: the reason goes<br/>to the fixer next time
```

## What crosses between the parts

```mermaid
flowchart LR
  FL[("failures.db<br/>open failures")] --> G["Refiner.groups<br/>kind · persona · cause"]
  G --> GEN["Generator<br/>refine/generate.py<br/>replay + variants"]
  GEN -- "checkers must discriminate:<br/>pass good, fail bad + a refusal<br/>refine/checks.py:119" --> CASES[("cases.json<br/>replay · dev · holdout ·<br/>control · regress · suspect")]
  CASES --> RUN["run_matrix<br/>refine/runner.py:145"]
  RUN -- "one process per run<br/>python -m jevh.refine.run_case<br/>own workspace, learning off" --> OLD["old code<br/>(base worktree)"]
  RUN --> NEW["new code<br/>(refine/id worktree)"]
  OLD --> RES[("results.jsonl<br/>scored in THIS process")]
  NEW --> RES
  RES --> TRI["triage<br/>refine/triage.py:115"]
  TRI --> FIX["fixer turn<br/>persona high_end_dev + project tool<br/>refine/fixer.py:153"]
  FIX -- "edits to jevh/refine/, tests/, evals/<br/>thrown away · pytest must pass" --> NEW
  RES --> REP["report.verdict / render<br/>refine/report.py:62, :129"]
  REP --> MD[("report.md<br/>old vs new per model and split + diff")]
  MD --> YOU["you: approve / reject"]
```

### Why the scores can be trusted

- **The checkers don't belong to the harness.** A checker is JSON the local model writes
  (`contains`, `regex`, `number`, `json_equals`, `artifact`, `python`, `jev`, `all` / `any`;
  `refine/checks.py`). It's kept only if it passes a good answer and fails a bad one and a plain
  refusal. So a fix that only makes the final() gate more lenient can't score as an improvement.
- **The fix can't change how it's scored.** Runs execute in the code tree being tested, but scoring
  happens in the refiner's own process. The fixer's edits to `jevh/refine/`, `tests/` and `evals/`
  are thrown away (`PROTECTED`, `refine/fixer.py:20`).
- **The fixer sees only the dev cases.** Held-out cases show whether the fix generalizes. Control
  tasks (`evals/refine_controls.json`) catch a fix that breaks something else.
- **Rare failures get enough runs.** The replay runs often enough to expect `repro_target` (3)
  failures on the old code, up to `max_runs` (8) per model.
- **"Cost" failures are scored on bounces.** For `cost_kinds` (`gate_bounce`: a correct answer the
  gate sent back), the measure is gate bounces per run, which must halve. A cost guard
  (`evals/verify_cases.py --guard`) checks that the gate didn't get more lenient.

### The verdict (`refine/report.py:62`)

| Verdict | Meaning |
|---|---|
| `improves` | the failing cases pass clearly more often (≥ `min_gain` 0.15, Fisher test), held-out cases don't do worse, no model loses more than `control_slack` (1) control + regression runs |
| `mixed` | it helps overall, but a held-out split or some model's controls got worse |
| `no_gain` | anything else: another fix try, or the attempt is dropped |

## Commands (`python -m jevh.refine`)

| Command | Does |
|---|---|
| `groups` | open failures grouped by kind, persona and cause |
| `models` | the local models tests run on, and the one that writes tests and fixes |
| `nightly [--until 07:00 \| --hours 8]` | carry on unfinished attempts, then start new ones |
| `start <failure id>`, `resume <attempt id>` | one attempt now |
| `status`, `report <attempt id>` | every attempt's stage; the approval report |
| `approve <attempt id>` | merge the branch; its failures are marked fixed (`refine/loop.py:542`) |
| `reject <attempt id> "why"` | drop it; the reason is shown to the fixer next time (`refine/loop.py:564`) |
