"""Jev: the gatekeeper. Every decision about what a worker model gets to see is one
batched /v1/systemone call, and every call is recorded in `trace` for the CLI.

Workers never talk to Jev about routing; they only see the result.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import Config
from .decide import SystemOneClient, SystemOneError, choice, noul, score
from .personas import Persona
from .skills import Skill

CHOICE_MAX = 16          # semif accepts 2-16 options per Choice (TypeSafe API: 255)

# Question wording lives here so evals/ can A/B phrasings against the same code path.
# Skills (tuned on evals/skill_cases.py, semif): a relative Choice separates sibling skills,
# which absolute nouls rate ~0.99 alike; the "describe how to do the specific task" noul
# ranks well but runs low, so it is only a floor on the Choice's pick.
SKILL_CHOICE_Q = "Which of these skills, if any, is the right one to load for the latest request?"
SKILL_NONE_OPTION = ("none_of_these", "No skill guide is needed; the request can be handled directly.")
SKILL_FITS_Q = "Does this skill guide describe how to do the specific task the latest request asks for?"
SKILL_TOP_N = 3            # Choice leaders that get a fits noul
# Memory (tuned on evals/memory_cases.py, semif): a noul per snippet costs ~0.18s each and the
# "contain information needed" wording passed only 68% of needed turns; a Choice to rank the
# history, then the "refer to or depend on" noul on the leaders: 95% passed, 0 leaks, ~1.6s.
# P(none) is NOT a usable veto here (needy requests scored up to 0.98 on it).
MEMORY_CHOICE_Q = "Which earlier snippet does the latest request depend on, if any?"
MEMORY_NONE_OPTION = ("none_of_these", "None - the request stands on its own.")
MEMORY_Q = "Does the latest request refer to or depend on information in this earlier snippet?"
MEMORY_TOP_N = 4           # Choice leaders that get the noul

# Model tier (tuned on evals/tier_cases.py, semif): a two-way Choice on the task text alone
# (a 3-level Score scored 97%, nouls 77-82%); 42/42 labelled + held-out tasks at any threshold
# 0.45-0.65, so 0.55 sits mid-plateau.
# Follow-ups (tuned on evals/followup_cases.py, semif): the routing call's "continues the topic"
# noul caught 13/14 follow-ups at 0.25 with 0/8 new topics joined; its miss ("something older"
# after a movie tip, 0.06) is caught by the reply noul (>= 0.4, which joined 0/8 new topics).
# Together: 14/14, 0/8. "only makes sense together with" rated every new topic ~0.75: unusable.
CONTINUES_Q = "Is the latest request a continuation of the recent conversation's topic?"
REPLY_Q = "Is the latest message a reply to the assistant's last message, or a follow-up to it?"

# Persona creation (jevh/forge.py); measured live by evals/persona_cases.py and evals/bench/persona_creation.py.
# "Is this a request no assistant is for?" is a Choice over the assistants plus "none of these", like the skill
# check below. A yes/no noul on the same question rated specialist and ordinary requests alike (0.44-0.62 vs
# 0.12-0.59), because the general assistant's description reads as covering everything. The Choice separates
# them, and asking for the *expertise* a request needs (not just "is it for them") keeps generic chores
# (proofreading, arithmetic, a short bio) with the general assistant: "which assistant is for this request"
# flagged 5 of 19 such ordinary requests, this wording 1. Asked off the critical path (orchestrator._learn).
NO_FIT_Q = "Which assistant has the expertise this request needs, if any?"
NO_FIT_NONE = ("none_of_these", "None of these assistants: the request needs expertise in a specialist field "
                                "that none of them covers.")
SKILL_HOME_Q = "Which assistant would use this skill guide, if any?"
SKILL_HOME_NONE = ("none_of_these", "None of these assistants: the guide is about something they are not for.")
# Asking whether "one specialist assistant would be right for both" rated unrelated specialist requests as high as
# 0.75 (a chess question against Japanese grammar), 11 of 220 cross-area pairs over 0.6; asking whether one *expert in
# the subject* would be an expert in the other's separates 54 same-domain from 220 cross-area pairs perfectly (AUC 1.00).
SAME_SPECIALIST_Q = "Would one person who is an expert in `request_a`'s subject also be an expert in `request_b`'s subject?"
SAME_SPECIALIST_TOP_N = 8    # personas compared per call in the near-duplicate check (forge._twin)

ROUTE_Q = "Which assistant is best suited to handle the user's latest request?"

TIER_Q = "Which model should handle this task?"
TIER_OPTIONS = {"fast": "a small fast model - good for quick answers, lookups, short scripts and simple edits",
                "strong": "a large careful model - needed for deep reasoning, hard debugging, long or complex builds"}

# A message the user sent while a task runs (jevh/interject.py; evals/interject_cases.py). A message is added to the
# running task's prompt unless it is a question about progress, which the task's model answers on the side, and unless
# it swaps the request for another, which the final() gate must know. Words alone do not tell those apart ("check my
# calendar" can be either, or an addition), so Jev reads the message against the task and what the assistant is doing
# at this moment. Another request and a stop are /queue and /stop commands, not guesses; their options stay in the
# Choice because they keep such a message from being mistaken for a question.
INTERJECT_Q = ("The user sent `message` while the assistant works on `task` (`doing` is what it is doing at this "
               "moment). What is the message?")
INTERJECT_OPTIONS = {
    "correction": "It changes, narrows or adds to the task the assistant is doing: a correction, an extra requirement, "
                  "a preference, or advice on how to go about it.",
    "replace": "It swaps the task, or what the assistant is producing, for something else (\"check my calendar instead\", \"make it primes instead of Fibonacci\", \"never mind that, do this\").",
    "status": "It asks how the work is going, what the assistant is doing, or what it has found so far, and asks "
              "for nothing new.",
    "separate": "It is another request, unrelated to the current task, that can wait until the task is finished.",
    "stop": "It only tells the assistant to stop, cancel or drop the task, and asks for nothing in its place.",
}
# (First wording had one "stop" option: "Actually check my calendar instead" scored stop 0.95, a redirect taken for a
# cancel. A yes/no "asks for something besides stopping" as a guard was no better: plain stops scored 0.5-0.85 on it.
# A separate "replace" option, counted as a correction, is the relative choice the Choice can make.)


def decide_interject(probs: dict[str, float], status_at: float) -> str:
    """Pure decision rule (shared with evals/interject_cases.py): "status" when the message is clearly a question
    about progress (it leads the Choice with at least `status_at`), else "correction": the message is added to the
    task's prompt. "replace" counts as a correction, and another request or a stop is never acted on from here
    (/queue and /stop are commands). Adding is the forgiving error: a question steered into the task is answered in
    its reply, while a message taken for a question never reaches it."""
    kinds = {k: v for k, v in probs.items() if k in ("correction", "status", "separate", "stop")}
    if not kinds:
        return "correction"
    kinds["correction"] = kinds.get("correction", 0.0) + probs.get("replace", 0.0)
    top = max(kinds, key=lambda k: kinds[k])
    return "status" if top == "status" and kinds["status"] >= status_at else "correction"


# Tools (tuned on evals/tool_cases.py): which REPL tools' guides the worker's prompt carries.
TOOL_Q = "Would the assistant need this tool to handle the latest request?"
# MCP tools: a catalog can be large, so the same Choice + fits design as skills.
MCP_CHOICE_Q = "Which of these tools, if any, does the latest request need?"
MCP_NONE_OPTION = ("none_of_these", "No tool is needed for this request.")
MCP_FITS_Q = "Does the latest request need this tool?"
MCP_TOP_N = 3


def decide_skills(probs: dict[str, float], fits: dict[str, float], fits_t: float,
                  override: float, veto: float, k: int = 1) -> list[str]:
    """Pure decision rule (shared with evals/skill_cases.py).
    - P(none) >= veto            -> nothing
    - Choice leader >= override  -> the leader (fits is not needed to confirm)
    - else rank by P(choice) * fits and keep those with fits >= fits_t, up to k"""
    if probs.get(SKILL_NONE_OPTION[0], 0.0) >= veto or not fits:
        return []
    leader = max(fits, key=lambda n: probs.get(n, 0.0))
    if probs.get(leader, 0.0) >= override:
        return [leader]
    ranked = sorted(fits, key=lambda n: -probs.get(n, 0.0) * fits[n])
    return [n for n in ranked if fits[n] >= fits_t][:k]


def decide_memory(scores: Sequence[float], threshold: float, cap: int) -> list[int]:
    """Indices of kept snippets, in original order (shared with evals/memory_cases.py)."""
    ranked = sorted((i for i, s in enumerate(scores) if s >= threshold), key=lambda i: -scores[i])
    return sorted(ranked[:cap])


@dataclass
class RouteDecision:
    persona: str
    confidence: float
    probabilities: dict[str, float]
    continues: float | None
    reason: str
    replies: float | None = None
    follows: bool = False      # the message follows up on the recent exchange: it goes along as context


@dataclass
class SkillDecision:
    skills: list[str]
    none: float | None = None                              # P(no skill needed)
    probs: dict[str, float] = field(default_factory=dict)  # Choice over allowed skills
    fits: dict[str, float] = field(default_factory=dict)   # fits noul, Choice leaders only


@dataclass
class MemoryDecision:
    kept: list[str]
    scores: list[float]


@dataclass
class TierDecision:
    tier: str                  # "fast" | "strong"
    p_strong: float | None
    reason: str


@dataclass
class ToolDecision:
    tools: list[str]                                        # built-in tool guides for the prompt
    scores: dict[str, float] = field(default_factory=dict)
    mcp: list[str] = field(default_factory=list)            # "server.tool" keys
    mcp_probs: dict[str, float] = field(default_factory=dict)
    mcp_fits: dict[str, float] = field(default_factory=dict)


@dataclass
class CollabDecision:
    suggested: list[str]
    scores: dict[str, float]
    complexity: float | None = None


@dataclass
class Approval:
    approved: bool
    reason: str
    scores: dict[str, float] = field(default_factory=dict)


def _clip(text: str, n: int) -> str:
    text = text.strip()
    return text if len(text) <= n else text[: n - 3] + "..."


class Jev:
    def __init__(self, cfg: Config, client: SystemOneClient | None = None):
        self.cfg = cfg
        # the hosted Jev (url: https://api.typesafe.ai) takes a key; the local adapter doesn't
        key_env = cfg.jev.get("api_key_env")
        api_key = os.environ.get(key_env) if key_env else None
        if client is None and key_env and not api_key:
            raise RuntimeError(f"config jev.api_key_env: set the {key_env} environment variable to your Jev API key")
        self.client = client or SystemOneClient(cfg.jev.get("url", "http://127.0.0.1:33001"),
                                                backend=cfg.jev.get("backend", "laya"),
                                                timeout=float(cfg.jev.get("timeout", 300)), api_key=api_key)
        self.choice_max = int(cfg.jev.get("choice_max", CHOICE_MAX))   # options per Choice the backend takes
        self.trace: list[dict[str, Any]] = []
        self.on_call = None   # (step, response, n_questions): usage tracking

    def _ask(self, step: str, state: Any, questions: dict) -> Any:
        resp = self.client.ask(state, questions)
        if self.on_call is not None:
            self.on_call(step, resp, len(questions))
        self.trace.append({"step": step, "model": resp.model, "latency_s": round(resp.latency_s, 3),
                           "answers": {k: a.raw for k, a in resp.answers.items()}})
        return resp

    # -- 1. which persona ----------------------------------------------------------
    def route_persona(self, request: str, personas: dict[str, Persona], current: str | None = None,
                      recent: str = "") -> RouteDecision:
        default = self.cfg.default_persona
        if len(personas) == 1:
            only = next(iter(personas))
            return RouteDecision(only, 1.0, {only: 1.0}, None, "only persona")
        questions = {"persona": choice(ROUTE_Q, {name: p.description for name, p in personas.items()})}
        state: dict[str, Any] = {"latest_request": request}
        if recent:
            state["recent_conversation"] = _clip(recent, 1500)
            questions["continues"] = noul(CONTINUES_Q)
            questions["replies"] = noul(REPLY_Q)
        r = self._ask("route", state, questions)
        pick, conf = r["persona"].choice, r["persona"].confidence
        cont = r["continues"].noul if "continues" in questions else None
        rep = r["replies"].noul if "replies" in questions else None
        follows = self._follows(cont, rep)
        probs = r["persona"].probabilities
        if follows and current and current in personas and pick != current and conf < self.cfg.t("route"):
            return RouteDecision(current, conf, probs, cont, f"follow-up; stays with {current}", rep, True)
        if pick == current:
            return RouteDecision(pick, conf, probs, cont, "choice (same persona)", rep, follows)
        if conf >= self.cfg.t("route"):
            return RouteDecision(pick, conf, probs, cont, "choice", rep, follows)
        if current and cont is not None and cont >= self.cfg.t("stick"):
            return RouteDecision(current, conf, probs, cont, f"low confidence for {pick}; conversation continues",
                                 rep, follows)
        return RouteDecision(default if default in personas else pick, conf, probs, cont,
                             f"low confidence for {pick}; default", rep, follows)

    def pick_persona(self, request: str, personas: dict[str, Persona]) -> str:
        """The persona the routing Choice alone would pick (no follow-up or fallback rules)."""
        probs = self.choice_probs("route_probe", {"latest_request": request}, ROUTE_Q,
                                  {n: p.description for n, p in personas.items()})
        return max(probs, key=probs.get)

    def no_fit(self, request: str, personas: dict[str, Persona]) -> float | None:
        """P(none of these assistants is for the request), whichever one routing sent it to; None when persona
        creation is off or Jev could not be asked (a turn must never fail because of this extra check)."""
        if not (self.cfg.raw.get("persona_creation") or {}).get("enabled", True):
            return None
        none_key, none_desc = NO_FIT_NONE
        try:
            probs = self.choice_probs("route_fit", {"latest_request": request}, NO_FIT_Q,
                                      {n: p.description for n, p in personas.items()}, {none_key: none_desc})
        except SystemOneError:
            return None
        return probs.get(none_key, 0.0)

    def skill_home(self, skill: Skill, personas: dict[str, Persona]) -> tuple[str | None, float, dict[str, float]]:
        """Which persona a skill belongs with, blind to where it currently lives: (best persona,
        P(none of them), probabilities). The best is None when "none of these" leads."""
        none_key, none_desc = SKILL_HOME_NONE
        state = {"skill_guide": f"{skill.name}: {skill.description}\n{skill.excerpt[:500]}"}
        probs = self.choice_probs("skill_home", state, SKILL_HOME_Q,
                                  {n: p.description for n, p in personas.items()}, {none_key: none_desc})
        none = probs.get(none_key, 0.0)
        named = {n: v for n, v in probs.items() if n in personas}
        best = max(named, key=named.get) if named else None
        return (None if none >= max(named.values(), default=0.0) else best), none, probs

    def same_specialist(self, request: str, others: Sequence[str]) -> list[float]:
        """For each earlier request, P(one specialist assistant suits both it and `request`)."""
        if not others:
            return []
        qs = {f"s{i}": noul({"question": SAME_SPECIALIST_Q, "request_a": request, "request_b": o})
              for i, o in enumerate(others)}
        r = self._ask("same_specialist", {"request_a": request, "earlier_requests": list(others)}, qs)
        return [r[f"s{i}"].noul for i in range(len(others))]

    def _follows(self, cont: float | None, rep: float | None) -> bool:
        return (cont or 0.0) >= self.cfg.t("follow_topic") or (rep or 0.0) >= self.cfg.t("follow_reply")

    def follow_up(self, request: str, recent: str) -> tuple[bool, float, float]:
        """For a pinned persona (no routing call): (follows, continues, replies)."""
        r = self._ask("follow_up", {"latest_request": request, "recent_conversation": _clip(recent, 1500)},
                      {"continues": noul(CONTINUES_Q), "replies": noul(REPLY_Q)})
        cont, rep = r["continues"].noul, r["replies"].noul
        return self._follows(cont, rep), cont, rep

    # -- 2. which skills -----------------------------------------------------------
    def pick_skills(self, request: str, persona: Persona, roster: dict[str, Skill]) -> SkillDecision:
        cands = [s for s in roster.values() if persona.offered_skill(s)]
        if not cands:
            return SkillDecision([])
        by_name = {s.name: s for s in cands}
        state = {"latest_request": request, "assistant": f"{persona.name}: {persona.description}"}
        none_key, none_desc = SKILL_NONE_OPTION
        probs = self.choice_probs("skills", state, SKILL_CHOICE_Q,
                                  {s.name: s.description for s in cands}, {none_key: none_desc})
        none = probs.get(none_key, 0.0)
        if none >= self.cfg.t("skill_none_veto"):
            return SkillDecision([], none, probs)
        top = [n for n in sorted(probs, key=lambda n: -probs[n]) if n in by_name][:SKILL_TOP_N]
        r = self._ask("skills_fits", state, {f"f{i}": noul({
            "question": SKILL_FITS_Q,
            "skill": f"{n}: {by_name[n].description}\n{_clip(by_name[n].excerpt, 500)}"})
            for i, n in enumerate(top)})
        fits = {n: r[f"f{i}"].noul for i, n in enumerate(top)}
        picked = decide_skills(probs, fits, self.cfg.t("skill_fits"), self.cfg.t("skill_override"),
                               self.cfg.t("skill_none_veto"), self.cfg.lim("max_skills"))
        return SkillDecision(picked, none, probs, fits)

    def choice_probs(self, step: str, state: Any, question: str, options: dict[str, str | None],
                     final_extra: dict[str, str] | None = None) -> dict[str, float]:
        """A Choice over any number of options. Backends cap a Choice (semif: 2-16, and it
        rejects some counts above that), so large sets run as balanced rounds of <= CHOICE_MAX
        whose top few go to a final round. `final_extra` options (e.g. a "none" outcome) join
        the final round only. Options knocked out early get probability 0."""
        names = list(options)
        cap = getattr(self, "choice_max", CHOICE_MAX)   # hosted Jev takes far more options than semif
        if len(names) + len(final_extra or {}) <= cap:
            r = self._ask(step, state, {"pick": choice(question, {**options, **(final_extra or {})})})
            return r["pick"].probabilities
        groups = -(-len(names) // (cap - 1))            # ceil, leaving room in each round
        size = -(-len(names) // groups)                         # balanced round sizes
        keep = max(1, (cap - len(final_extra or {})) // groups)
        finalists: list[str] = []
        for start in range(0, len(names), size):
            chunk = names[start:start + size]
            if len(chunk) == 1:
                finalists += chunk
                continue
            r = self._ask(f"{step}_round", state, {"pick": choice(question, {n: options[n] for n in chunk})})
            p = r["pick"].probabilities
            finalists += sorted(chunk, key=lambda n: -p.get(n, 0.0))[:keep]
        final = {n: options[n] for n in finalists}
        return self.choice_probs(step, state, question, final, final_extra)

    # -- 2b. which tools -------------------------------------------------------------
    def pick_tools(self, request: str, tools: dict[str, str], mcp: dict[str, str] | None = None) -> ToolDecision:
        """`tools`: the persona's built-in tools (name -> description), one noul each in one
        call. `mcp`: MCP catalog (key -> description): Choice ranks it, fits noul on the top."""
        state = {"latest_request": request}
        scores: dict[str, float] = {}
        if tools:
            r = self._ask("tools", state, {f"t_{n}": noul({"question": TOOL_Q, "tool": d}) for n, d in tools.items()})
            scores = {n: r[f"t_{n}"].noul for n in tools}
        picked = [n for n in tools if scores.get(n, 0.0) >= self.cfg.t("tool")]
        dec = ToolDecision(picked, scores)
        if mcp:
            none_key, none_desc = MCP_NONE_OPTION
            probs = self.choice_probs("mcp_rank", state, MCP_CHOICE_Q, {k: _clip(d, 300) for k, d in mcp.items()},
                                      {none_key: none_desc})
            top = [k for k in sorted(probs, key=lambda k: -probs[k]) if k in mcp][:MCP_TOP_N]
            if top and probs.get(none_key, 0.0) < self.cfg.t("skill_none_veto"):
                r = self._ask("mcp_fits", state, {f"m{i}": noul({"question": MCP_FITS_Q, "tool": f"{k}: {_clip(mcp[k], 400)}"})
                                                  for i, k in enumerate(top)})
                fits = {k: r[f"m{i}"].noul for i, k in enumerate(top)}
                dec.mcp = [k for k in top if fits[k] >= self.cfg.t("mcp_fits")][:self.cfg.lim("max_mcp_tools")]
                dec.mcp_fits = fits
            dec.mcp_probs = probs
        return dec

    # -- 3. which memory -----------------------------------------------------------
    def gate_context(self, request: str, chunks: Sequence[str]) -> MemoryDecision:
        if not chunks:
            return MemoryDecision([], [])
        state = {"latest_request": request}
        if len(chunks) <= MEMORY_TOP_N:
            top = list(range(len(chunks)))
        else:
            probs = self.choice_probs("memory_rank", state, MEMORY_CHOICE_Q,
                                      {f"m{i}": _clip(c, 300) for i, c in enumerate(chunks)},
                                      dict([MEMORY_NONE_OPTION]))
            ranked = sorted((k for k in probs if k != MEMORY_NONE_OPTION[0]), key=lambda k: -probs[k])
            top = [int(k[1:]) for k in ranked[:MEMORY_TOP_N]]
        r = self._ask("memory", state, {f"m{i}": noul({"question": MEMORY_Q, "snippet": _clip(chunks[i], 800)})
                                        for i in top})
        scores = [r[f"m{i}"].noul if i in top else 0.0 for i in range(len(chunks))]
        kept = decide_memory(scores, self.cfg.t("memory"), self.cfg.lim("max_memory_chunks"))
        return MemoryDecision([chunks[i] for i in kept], scores)

    # -- 3b. which model tier ---------------------------------------------------------
    def pick_tier(self, task: str, persona: Persona) -> TierDecision:
        """Fast model for quick work, the persona's own model for hard work. Errs toward
        strong: a hard task on the fast model costs quality, the reverse only costs speed."""
        # task only: with the persona description in the state, simple dev/designer tasks
        # drifted to P(strong) ~0.6 (39/42); task-only scored 42/42 at thresholds 0.45-0.65
        r = self._ask("tier", {"task": task}, {"tier": choice(TIER_Q, TIER_OPTIONS)})
        p = r["tier"].probabilities.get("strong", 0.0)
        t = self.cfg.t("tier_strong")
        return TierDecision("strong" if p >= t else "fast", p, f"P(strong)={p:.2f} vs {t:.2f}")

    def interject_kind(self, message: str, task: str, doing: str = "") -> dict[str, float]:
        """What a message sent while a task runs is: probabilities over correction / replace / status / separate /
        stop (decide_interject: a question about progress or, else, a correction; replace counts as a correction)."""
        r = self._ask("interject", {"task": task[:1500], "doing": doing[:300], "message": message[:1500]},
                      {"kind": choice(INTERJECT_Q, INTERJECT_OPTIONS)})
        return r["kind"].probabilities

    # -- 4. who else ---------------------------------------------------------------
    def collab(self, request: str, lead: Persona, personas: dict[str, Persona]) -> CollabDecision:
        others = {n: p for n, p in personas.items() if n != lead.name}
        if not others:
            return CollabDecision([], {})
        questions = {"complexity": score("How much work does this request involve?",
                                         ["a quick answer", "a small task",
                                          "a multi-part project needing several skills"])}
        # Phrasing picked on evals/routing_cases.py (semif): "requires expertise" said yes to the
        # dev for anything technical; asking for a deliverable the lead can't make scored 31/33.
        for n, p in others.items():
            questions[f"needs_{n}"] = noul({
                "question": "Does the request ask for a deliverable that this specialist must produce, "
                            "which the lead assistant cannot produce itself?",
                "lead_assistant": lead.description, "specialist": p.description})
        r = self._ask("collab", {"latest_request": request,
                                 "lead_assistant": f"{lead.name}: {lead.description}"}, questions)
        scores = {n: r[f"needs_{n}"].noul for n in others}
        complexity = r["complexity"].score
        # a quick answer never pulls in a team
        suggested = [] if complexity < 0.5 else sorted(
            (n for n, s in scores.items() if s >= self.cfg.t("collab")), key=lambda n: -scores[n])
        return CollabDecision(suggested, scores, complexity)

    # -- 5. may A ask B for this? ----------------------------------------------------
    def approve_request(self, requester: Persona, target: Persona, task: str, goal: str) -> Approval:
        questions = {
            "in_scope": noul({"question": "Is this task within the specialist's expertise?",
                              "specialist": target.description, "task": task}),
            "necessary": noul({"question": "Does this task directly help accomplish the user's goal?",
                               "user_goal": _clip(goal, 800), "task": task}),
        }
        r = self._ask("approve", {"requesting_assistant": requester.name, "target": target.name,
                                  "task": task}, questions)
        scores = {k: r[k].noul for k in questions}
        t = self.cfg.t("approve")
        if scores["in_scope"] < t:
            return Approval(False, f"{target.name} is not the right specialist for this task", scores)
        if scores["necessary"] < t:
            return Approval(False, "task does not appear necessary for the user's goal", scores)
        return Approval(True, "approved", scores)
