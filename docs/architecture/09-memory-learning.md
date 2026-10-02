# 09 · Memory and learning

Nothing from the past reaches a worker unless Jev lets it through. There are two ways in: the
**memory gate** at task setup, and `recall()` when the worker asks. A third path, for
**follow-ups**, sends the last exchange as real conversation turns.

## Storage (`memory.py`)

```mermaid
erDiagram
  turns ||--o| turn_notes : "notes for follow-ups"
  turns ||--|| turns_fts : "FTS5 (porter)"
  facts ||--|| facts_fts : "FTS5 (porter)"
  turns {
    int id
    text persona
    text session
    real ts
    text user
    text assistant
  }
  turn_notes {
    int turn_id
    text notes
  }
  facts {
    int id
    real ts
    text text
    text source
    int active
  }
```

`Memory.append` (`memory.py:111`) also writes one JSON line to `<persona>.jsonl` and
`timeline.jsonl`. On a fresh `memory.db`, `timeline.jsonl` is imported.

## Three paths from the past to a worker

```mermaid
flowchart LR
  subgraph DB["memory.db"]
    T["turns"]
    N["turn_notes"]
    F["facts (active)"]
  end

  T -- "own(persona, 20)<br/>shared(others, 6)<br/>search(request, 10) BM25" --> C["candidates<br/>memory.py:186"]
  F -- "facts(request, 6)<br/>topped up with newest" --> C
  C --> G["Jev gate_context<br/>choice rank → top 4 → noul ≥ .20<br/>max 4"]
  G --> CTX["ctx['memory']<br/>(in the REPL, listed by size)"]

  T -- "search(query, 24)" --> RC["recall(query) callback"]
  F -- "facts(query, 8)" --> RC
  RC --> G2["Jev gate_context"] --> RET["returned to the worker's code"]

  T -- "session_turns(session, 2)" --> FU{"route.follows?"}
  N --> FU
  FU -- yes --> HIS["task.history → real user/assistant<br/>messages (+ work notes)<br/>and task.jev_text for every decision"]
```

- Candidates that are already in the follow-up history are removed first (`orchestrator.py:439`).
- `task.jev_text` (`bus.py:29`) is what every per-task Jev decision judges. For a follow-up it is
  the earlier exchange plus "Latest message: …", so "4 days, full gym" still gets the workout
  skill.
- Failed turns aren't written. A stopped turn is written with "[stopped by the user before
  finishing]", so a redirect has context.

## Learning after a turn (`learn.py`)

Runs on a background thread after a turn that ended in `final` or `plain`. It uses the model that
just ran the task, which is already loaded, so there's no LM Studio swap. A new turn sets
`learner.cancel`, and any learning call in progress is dropped (`Cancelled`).

```mermaid
sequenceDiagram
  participant H as Harness._learn
  participant L as Learner.after_turn
  participant J as Jev
  participant M as task's model (learn kind)
  participant DB as memory.db
  participant SK as skills/

  H->>L: persona, message, answer, transcript, steps, used skills, llm
  L->>J: learn_fact: does the user state a lasting fact?
  alt P ≥ fact 0.88
    L->>M: FACT_PROMPT → JSON list of 0–3 facts
    loop each fact
      L->>DB: facts(fact, 4) similar old facts
      L->>J: learn_same_fact: is old made redundant or outdated by new?
      L->>DB: retire old (≥ 0.5), add new
    end
  end
  opt steps ≥ 3
    L->>J: learn_skill: did work_log need non-obvious steps, fixes or settings?
    alt P ≥ 0.5 and a skill was loaded
      L->>M: LESSON_PROMPT → 1–4 bullets or NONE
      L->>SK: append to its SKILL.md (ours) or skills/_lessons/name.md (e.g. Hermes)
    else P ≥ 0.5, no skill loaded
      L->>M: SKILL_PROMPT → SKILL.md
      L->>SK: skills/learned/learned-slug/SKILL.md (draft: true)
    end
  end
  L-->>H: event learned {facts, retired, skill, skill_action, scores}
  H->>H: skill written → reload roster (next turn sees it)
```

Learned skills are drafts until the user approves them in the dashboard's Memory tab
(`/api/skills/review`). A draft's body goes into prompts with a "treat as hints, check its claims"
header. `remember(fact)` from the REPL uses the same `_store_facts` path, retiring outdated facts.
