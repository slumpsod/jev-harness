# 15. Persona creation

Status: built (`jevh/forge.py`) and run live on 2026-10-02 with `evals/bench/persona_creation.py` (semif Jev, Qwen
3.8 27B writing; the last summary is `evals/bench/persona_creation_last_summary.txt`). The first run showed it did not work end
to end (the trigger never fired; the gate rejected every real draft). After the fixes below: 27/27 drafts live
with realistic memory and 0 steals, specialist requests pooled 39/42 and ordinary ones 6/60, all 9 domains
formed from a mixed stream of 62 requests, the 37-request routing set unchanged with 9 drafts live, and full
turns that pool, draft and then route to the new persona. Numbers are in the README's Persona creation section.

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

1. **No fit.** After a turn that the default (catch-all) persona handled, on the learning thread and so
   off the critical path, `Jev.no_fit` asks which assistant has the expertise the request needs: a Choice
   over every active persona's description plus "none of these", the same form as the skill check. When
   P(none of these) reaches `no_fit` (0.60) the request is handed to `Forge.note_no_fit`, with the task's
   own already-loaded model. A pinned persona (`/as`, the dashboard picker) and a request the router sent to
   a specialist never feed it, and a persona the harness created counts as an assistant, so a request it
   covers is not pooled again. **It does not depend on routing confidence.** The first version asked a yes/no
   question only when the route was unsure (confidence under `thresholds.route`); live, Jev sent specialist
   requests (chess, guitar, tenancy...) to the general assistant at 0.85-1.0, so that was never asked.
   Asking for "the expertise this request needs", not just "who is this for", keeps generic chores
   (proofreading, arithmetic, a short bio) with the general assistant.
   The request goes into a pool (`workspace/persona_proposals.json`, newest `max_pending` (40) kept; the
   same text again is not pooled twice). Jev compares each new one with all of the pool (`compare_last`,
   `same_specialist`: "would one person who is an expert in A's subject also be an expert in B's?"), keeps
   the ones at `same_specialist` (0.60) or above, and then keeps only a group that suits each other
   (`coherent`: similarity is not transitive, so a new request linked to two requests that have nothing to
   do with each other must not gather them). When `min_cluster` have gathered, the pool is checked once more
   (an existing assistant may be for them by now: nothing is written), then one persona is drafted for them
   and they leave the pool. One request at a time goes through `note_no_fit`: learning threads can overlap.
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
  personas handled (read from memory; the triggering requests themselves are not counted). A request is
  **stolen** only when the draft would win it **and** Jev says an existing assistant has the expertise
  for it (`Jev.no_fit` over the other personas, under `no_fit`). A request that merely landed on the
  catch-all persona because nothing was for it is what the draft exists to take. The request a persona
  came from, the task a skill was learned from, and every earlier request in the same domain are all in
  memory under the catch-all persona, so counting them as stolen rejected every real draft (0 of 27 went
  live in the first live bench; 24 of 27 had, but only because that bench's memory held none of them).

A draft that fails is written `active: false` with the reason in `source`. It is out of routing
until you switch it on. The gate re-routes recent requests rather than running
`evals/routing_cases.py` (which needs the eval corpus and a live backend); that remains the check
for tuning. The router itself also protects the incumbents: with a catch-all draft ("any everyday question...
calendar and email...") present, calendar and email requests still went to `daily_assistant` at 0.97-0.99,
so a draft only takes what it is better at.

### Limits

- `max_live_drafts` (4): drafts waiting for approval; no more are made until some are approved or deleted.
- Two checks against duplicating an existing persona: the pooled requests are asked about again before the
  model writes anything (an existing assistant now having the expertise means nothing is made and the
  requests leave the pool), and a new description that Jev finds to suit the same specialist as an existing
  persona (`same_specialist`) is not created.
- A reply from the model that cannot be used leaves the requests pooled; the reply is kept in the
  `persona_skipped` event (the live bench met this a few times and could not reproduce it).

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

- A draft only gets the tools in `persona_creation.tools`; a web-ish specialist needs them widened by hand.
- Requests that no persona description lists but a general assistant would do (naming ideas, slogans,
  "how does compound interest work", "what does R-value mean") are flagged too, in about 1 of 10 ordinary
  requests; they pool, and three similar ones make a draft you then approve or delete. Describing what the
  general assistant is for (writing, brainstorming, general explanations) in `daily_assistant`'s description
  would keep them with it.
- Chess "endgames" and "openings" requests do not suit each other (0.2-0.5), so a stream of both forms two
  small groups, or none, instead of one chess persona.
- A pair's similarity is read in one direction; the two directions straddle 0.60 for 3 of 54 genuine pairs.
  Averaging both directions was measured and kept fewer genuine pairs without helping.
- How long before an unused draft is flagged (usage tracking is there; nothing reads it yet).
- Whether a draft's system prompt is refined as more requests route to it.
