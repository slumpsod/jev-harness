"""Jev: the gatekeeper. Every decision about what a worker model gets to see is one
batched /v1/systemone call, and every call is recorded in `trace` for the CLI.

Workers never talk to Jev about routing; they only see the result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import Config
from .decide import SystemOneClient, choice, noul, score
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
TIER_Q = "Which model should handle this task?"
TIER_OPTIONS = {"fast": "a small fast model - good for quick answers, lookups, short scripts and simple edits",
                "strong": "a large careful model - needed for deep reasoning, hard debugging, long or complex builds"}


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
        self.client = client or SystemOneClient(cfg.jev.get("url", "http://127.0.0.1:33001"),
                                                backend=cfg.jev.get("backend", "laya"),
                                                timeout=float(cfg.jev.get("timeout", 300)))
        self.trace: list[dict[str, Any]] = []

    def _ask(self, step: str, state: Any, questions: dict) -> Any:
        resp = self.client.ask(state, questions)
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
        questions = {"persona": choice(
            "Which assistant is best suited to handle the user's latest request?",
            {name: p.description for name, p in personas.items()})}
        state: dict[str, Any] = {"latest_request": request}
        if current and recent:
            state["recent_conversation"] = _clip(recent, 1500)
            questions["continues"] = noul(
                "Is the latest request a continuation of the recent conversation's topic?")
        r = self._ask("route", state, questions)
        pick, conf = r["persona"].choice, r["persona"].confidence
        cont = r["continues"].noul if "continues" in questions else None
        if pick == current:
            return RouteDecision(pick, conf, r["persona"].probabilities, cont, "choice (same persona)")
        if conf >= self.cfg.t("route"):
            return RouteDecision(pick, conf, r["persona"].probabilities, cont, "choice")
        if current and cont is not None and cont >= self.cfg.t("stick"):
            return RouteDecision(current, conf, r["persona"].probabilities, cont,
                                 f"low confidence for {pick}; conversation continues")
        return RouteDecision(default if default in personas else pick, conf,
                             r["persona"].probabilities, cont, f"low confidence for {pick}; default")

    # -- 2. which skills -----------------------------------------------------------
    def pick_skills(self, request: str, persona: Persona, roster: dict[str, Skill]) -> SkillDecision:
        cands = [s for s in roster.values() if persona.allows_skill(s.name)]
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
        if len(names) + len(final_extra or {}) <= CHOICE_MAX:
            r = self._ask(step, state, {"pick": choice(question, {**options, **(final_extra or {})})})
            return r["pick"].probabilities
        groups = -(-len(names) // (CHOICE_MAX - 1))            # ceil, leaving room in each round
        size = -(-len(names) // groups)                         # balanced round sizes
        keep = max(1, (CHOICE_MAX - len(final_extra or {})) // groups)
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
