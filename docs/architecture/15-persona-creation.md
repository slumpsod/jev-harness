# 15. Persona creation

Status: built (`jevh/forge.py`), run live on 2026-10-02 with `evals/bench/persona_creation.py`: **not working end
to end**. The trigger never fires (routing confidence stays above 0.45, and `no_fit` scores 0.44-0.62 against
0.80) and the routing gate holds back every real draft (the request a persona came from counts as stolen). The
writer, the fence, the skill-ownership hand-over and approve/delete work. Numbers are in the README's Persona
creation section.

## Why

Skills and personas are the harness's context-isolation mechanism: a persona's `skills_allowed`
decides which skill guides can reach a worker's prompt. Two gaps weakened that:

- A skill that fits no persona (typically a learned one) lands under `learned-*`, which several
  personas allow. Every one of them can be shown it: context poisoning.
- A request that fits no persona is routed to the current or default persona
  (`Jev.route_persona`, `low confidence for X; default`), with the wrong prompt, tools and skills.

When neither fits, the harness drafts a new persona rather than stretch an existing one.

## Scope (v1)

**Creates new personas only.** Moving or narrowing skills on existing personas is a later step
(see "Later: blind skill assignment").

## The two triggers

1. **No fit (routing).** Low confidence alone also happens when two good personas split the
   probability, so `Jev.route_persona` asks one more noul, `no_fit` ("does the request fall outside
   what every one of the assistants is for?"), but only on a low-confidence route that is not a
   follow-up. A confident route costs no extra call, and the tuned routing Choice is unchanged (no
   "none of these" option was added to it). The decision carries `no_fit`; the orchestrator hands the
   request to `Forge.note_no_fit` after the turn, on the learning thread, using the task's own
   already-loaded model.
   The request goes into a pool (`workspace/persona_proposals.json`, newest `max_pending` kept). Jev
   compares each new one with the last 8 pooled ones (`same_specialist`). When `min_cluster` similar
   ones have gathered, one persona is drafted for them and they leave the pool.
2. **Orphaned skill.** After a learning run creates a skill, `Forge.note_skill` shows it to Jev
   blind: a Choice over every active persona's description plus "none of these". If "none of these"
   leads and reaches `orphan`, a persona is drafted around it immediately. Only unowned skills under
   `skills/learned/` (the ones the harness may edit) are considered. `/forge` and the dashboard's
   "check learned skills" run this over skills learned before.

A turn that was pinned (`/as`, the dashboard's persona picker) never feeds either trigger.

## What a draft is

A plain YAML file in `personas/learned/` (gitignored), loaded beside the hand-written personas
(`load_personas`; a hand-written persona wins a name clash, an unreadable file is skipped). It is live
straight away, so it can be tried in use, but fenced:

- `draft: true`, shown as a draft in the CLI, the dashboard and the persona picker.
- `mcp: []`, `can_request: []`, `tools: persona_creation.tools` (empty by default), `max_steps: 30`,
  the default persona's model.
- No `skills_allowed` globs: the skill that made it is **owned** by it (`owner:` in the skill's
  frontmatter, `Persona.offered_skill`). An owned skill is offered to its owner only, so it leaves
  every other persona's context. Ownership is given when the persona goes live, not before: a skill
  owned by an inactive persona would be offered to nobody. Deleting the persona gives it back.
- The persona's description and system prompt are written by the model that just ran the task, from
  the requests or the skill, shown the existing personas' descriptions so it makes a distinct one.

### The gate

Before it goes live a draft must pass `Forge.gate`:

- it wins at least half of the requests that made it (the routing Choice alone, no follow-up rules),
  and
- it takes at most `max_steals` (0) of the last `gate_probes` (8) earlier requests that other
  personas handled (read from memory; the triggering requests themselves are not counted).

A draft that fails is written `active: false` with the reason in `source`. It is out of routing
until you switch it on. The gate re-routes recent requests rather than running
`evals/routing_cases.py` (which needs the eval corpus and a live backend); that remains the check
for tuning.

### Limits

- `max_live_drafts` (4): drafts waiting for approval; no more are made until some are approved or deleted.
- A near-duplicate check: if Jev finds the new description to suit the same specialist as an existing
  persona (`same_specialist`), it is not created.

## Review

Dashboard, Memory tab: approve (keep for good), switch on (inactive -> live, stays a draft), delete;
pending unhandled requests are listed. CLI: `/personas`, `/persona approve|activate|delete <name>`,
`/forge`. The harness reloads its persona list after each (`Harness.reload_personas`).

## Later: blind skill assignment (noted, not built)

Idea: have Jev assign each skill to a persona blind, repeat it 10 times, and award the skill to the
persona it picks most consistently.

- Jev's Choice is probably near-deterministic for identical input, so 10 identical rounds give one
  answer ten times. Vary the rounds: shuffle option order (position bias), reword the question,
  and use the mean probability over rounds rather than the count of wins.
- Blind = the persona is described by its description only, the skill by name, description and
  excerpt; nothing about where it currently lives. (`Jev.skill_home` already does one such round.)
- Award only on a stable win: leader in most rounds and a margin over second place. A close
  result can mean the skill belongs to both, or to a persona that does not exist yet, which feeds
  the orphan trigger.
- Risk: a wrong assignment silently removes a skill from the persona that needed it. Make the
  first version additive (propose, show the vote table), keep removal behind live testing, and
  check against the requests that previously loaded that skill (the usage log has them). The
  `owner:` mechanism above is the natural place to record an award.

## Open items

- Tune `no_fit`, `orphan`, `same_specialist` and the gate on semif (`evals/persona_cases.py`).
- A draft only gets the tools in `persona_creation.tools`; a web-ish specialist needs them widened by hand.
- Clustering compares a new request with the last 8 pooled ones only; pairwise similarity is not transitive.
- How long before an unused draft is flagged (usage tracking is there; nothing reads it yet).
- Whether a draft's system prompt is refined as more requests route to it.
- No-fit requests that earlier turns already sent to another persona count against the gate as "stolen".
