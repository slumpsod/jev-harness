# 14. Persona creation (design, not yet built)

Status: design agreed in conversation; nothing implemented yet.

## Why

Skills and personas are the harness's context-isolation mechanism: a persona's `skills_allowed`
decides which skill guides can reach a worker's prompt. Two gaps weaken that:

- A skill that fits no persona (typically a learned one) lands under `learned-*`, which several
  personas allow. Every one of them can be shown it, which is the context poisoning the split is
  meant to prevent.
- A request that fits no persona is routed to the current or default persona
  (`Jev.route_persona`, `low confidence for X; default`). The task runs with the wrong persona's
  prompt, tools and skills.

When neither fits, the harness should create a new persona rather than stretch an existing one.

## Scope (v1)

**Create new personas only.** Moving or narrowing skills on existing personas is a later step
(see "Later: blind skill assignment").

## Two triggers

1. **No fit (routing).** Low confidence alone is not enough: it also happens when two good personas
   split the probability. Add a "none of these" option to the persona Choice, the way
   `SKILL_NONE_OPTION` works for skills, or a noul "does any assistant here cover this request?".
   Only a high "none" score counts as no fit. The request is logged to a proposal pool.
2. **Orphaned skill.** A skill (usually `learned-*`) that no persona's `skills_allowed` would
   pick on its own: ask Jev to Choice persona for the skill's description with a none option, and
   treat a high "none" as an orphan.

One event does not create a persona. Events go to a proposal pool and a persona is drafted when
similar ones cluster (a few no-fit requests close in embedding/Choice space, or an orphaned skill
whose description is far from every persona). Orphaned skills may draft immediately; they are the
more reliable signal.

## Drafts are live, but fenced

Drafts are routable straight away, so they can be tested in real use before being committed to.
That makes the fence matter:

- Minimal permissions: no `mcp`, no `can_request`, a short `tools` list, `skills_allowed` limited
  to the skills that triggered it. Same model as the other personas; the system prompt comes from
  a template plus a description written by the task's model.
- Labelled `draft: true` everywhere the persona shows (CLI route line, dashboard).
- Stored apart from hand-written personas, e.g. `personas/learned/` (gitignored like
  `skills/learned/`). `load_personas` only globs `personas/*.y*ml` today, so it needs to load this
  folder too and set the flag.
- Eval gate before going live: it must not steal cases that `evals/routing_cases.py` currently
  routes correctly, and its own triggering requests must route to it. Fails -> stays inactive.
- Cap on live drafts, and a near-duplicate check against existing descriptions (merge into the
  existing draft instead of making a second).
- Promote / delete in the dashboard; unused drafts are flagged from usage tracking.

## Later: blind skill assignment (noted, not in v1)

Idea: have Jev assign each skill to a persona blind, repeat it 10 times, and award the skill to the
persona it picks most consistently.

Notes for when it is built:

- Jev's Choice is probably near-deterministic for identical input, so 10 identical rounds give one
  answer ten times. Vary the rounds: shuffle option order (position bias), reword the question,
  and use the mean probability over rounds rather than the count of wins.
- Blind = the persona is described by its description only, the skill by name, description and
  excerpt; nothing about where it currently lives.
- Award only on a stable win: leader in most rounds and a margin over second place. A close
  result can mean the skill belongs to both, or to a persona that does not exist yet, which feeds
  the orphan path above.
- Risk: a wrong assignment silently removes a skill from the persona that needed it. Make the
  first version additive (propose, show the vote table), keep removal behind live testing, and
  check against the requests that previously loaded that skill (the usage log has them).
- `skills_allowed` is a hand-written glob list in persona YAML. Assignments should be an overlay
  file (e.g. `skills/assignments.yaml`) read at load time rather than edits to those files.

## Open items

- Clustering method for the proposal pool (Jev nouls vs. embeddings).
- How many drafts may be live, and how long before an unused one is flagged.
- Whether a draft's system prompt is written once or refined as more requests route to it.
