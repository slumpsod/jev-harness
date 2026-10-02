# 12 · Session review

Some failures no detector can see: the totals were wrong, a source was never cited, the plan
ignored what you said. Session review lets you **flag a session**, write notes on what looked
wrong, and have a model read the whole session against those notes. The model lists the problems
and gives a verdict. A "failure" verdict also goes into the failure log, so refinement
([13](13-refine.md)) works on it like any other failure.

## The flow

```mermaid
sequenceDiagram
  autonumber
  participant U as You (dashboard / CLI)
  participant RV as Reviewer (review.py:414)
  participant RL as ReviewLog (reviews.db)
  participant REC as session_record()
  participant M as reviewing model
  participant FL as FailureLog

  U->>RV: flag(session, note, turn) - "Flag for review" / /flag
  RV->>RL: flags row (status open, title from the first request) + first note
  U->>RL: more notes (/note, the dashboard), each may point at a turn
  U->>RV: run(flag) - /review, POST /api/reviews/run (on a thread)
  RV->>REC: gather the session
  Note over REC: every turn in memory (user + assistant)<br/>each task folder u{turn}t{task}-{persona}: request,<br/>how it ended, steps, answer, work log (checkpoint.json)<br/>failures already logged for the session
  RV->>M: review_prompt(record, notes, budget 60k chars)<br/>work logs cut to fit, latest part kept
  M-->>RV: one JSON object: verdict, confidence, summary,<br/>problems [{what, evidence, note}], notes_checked
  RV->>RL: set_review (status decided, verdict_by model)
  RV->>FL: sync_failure: verdict failure → one row, source review, kind review_failure
  U->>RV: decide(flag, failure | not_failure, why) - overrules the model
  RV->>FL: not_failure → that row set to ignored
```

| Step | Code |
|---|---|
| What the model reads | `session_record`, `review.py:218`; `review_prompt`, `review.py:300` |
| Parsing the reply (first JSON object; verdict must be `failure` or `not_failure`) | `parse_review`, `review.py:320` |
| Which model reviews: `review.model`, else `--model`, else `models.strong`, else the default persona's; never `think: unlimited`, budget `review.max_tokens` (32,768) | `reviewer_block`, `review.py:339` |
| Running it | `Reviewer.run`, `review.py:441` |
| Keeping the failure log in step with the verdict | `sync_failure`, `review.py:380` |

## Storage (`workspace/reviews.db`)

```mermaid
erDiagram
  flags ||--o{ notes : "has"
  flags |o--o| failures : "failure_id (verdict failure)"
  flags {
    int id
    text session "unique"
    text title
    text status "open | decided | dismissed"
    text verdict "failure | not_failure"
    text verdict_by "model | user"
    text reason
    text review "the model's last review, JSON"
    int failure_id
  }
  notes {
    int id
    int flag_id
    real ts
    int turn
    text text
  }
```

## Where to use it

| Place | Commands |
|---|---|
| CLI | `/flag [note]`, `/note <text>`, `/flags [all]`, `/review [id]` |
| Dashboard | "Flag for review", notes, run, verdict (`/api/reviews`, `/api/reviews/flag`, `/note`, `/run`, `/api/sessions`) |
| Command line | `python -m jevh.review [sessions \| flag \| note \| show \| run \| verdict \| dismiss]` |
| Events | `review_flag`, `review_start`, `review_done`, `review_error` (dashboard) |
