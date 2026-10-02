"""Persona creation: a new persona when a request or a learned skill fits none of the existing ones.

Two triggers (docs/architecture/15-persona-creation.md):

- No fit. Routing is unsure and a Jev noul says no assistant is for the request: it goes into a
  pool of unhandled requests. When enough similar ones have gathered (Jev compares each new one
  with the pool), one persona is drafted for them.
- Orphaned skill. A newly learned skill is shown to Jev blind (a Choice over the personas plus
  "none of these"); if none of them wants it, a persona is drafted around it at once.

A draft is a plain file in personas/learned/ and is live straight away, so it can be tried in use,
but fenced: no MCP, no `can_request`, the REPL tools in `persona_creation.tools` (none by default),
and the only skill it is offered is the one that made it (that skill is owned by it, so every other
persona stops being offered it). Before it goes live it must pass a routing gate: it has to win the
requests that made it and take none of the recent requests other personas handled. A draft that
fails is written inactive for you to look at. Approve or delete drafts in the dashboard or the CLI.
"""
from __future__ import annotations

import copy
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from .config import Config
from .jev import SAME_SPECIALIST_TOP_N, Jev
from .learn import slug
from .memory import Memory
from .personas import LEARNED_DIR, Persona, load_learned, load_personas
from .skills import Skill, parse_skill

DEFAULTS = {"enabled": True, "min_cluster": 3, "max_live_drafts": 4, "max_pending": 40, "tools": [],
            "max_steps": 30, "gate_probes": 8, "max_steals": 0}
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,40}$")

PERSONA_PROMPT = """You are adding a specialist assistant to an assistant harness. A router picks the \
assistant for each request from the assistants' descriptions, so the description decides when this one is \
used: make it concrete, say what kind of requests it handles, and make it clearly different from the \
existing assistants. Keep it narrow: a specialist, not a second general assistant.

{evidence}

Existing assistants:
{existing}

Reply with ONLY a JSON object:
{{"name": "<2-3 word snake_case name>", "description": "<one or two sentences: the requests this assistant \
handles>", "system_prompt": "<4-8 lines: who the assistant is, how it works, what it leaves to others>"}}"""


@dataclass
class Gate:
    ok: bool
    wins: int = 0
    probes: int = 0                     # requests it should win (the ones that made it)
    steals: list[str] = field(default_factory=list)   # earlier requests of other personas it would take
    checked: int = 0                    # earlier requests re-routed
    note: str = ""


@dataclass
class Created:
    name: str
    path: str
    active: bool
    gate: Gate
    trigger: str                        # "no_fit" | "orphan_skill"
    skill: str | None = None


class Forge:
    def __init__(self, cfg: Config, jev: Jev, memory: Memory, skills_home: Path,
                 chat: Callable[[Any, str], str], on_event: Callable[[str, dict], None] | None = None):
        self.cfg, self.jev, self.memory = cfg, jev, memory
        self.skills_home = Path(skills_home)
        self.chat = chat                                  # (llm, prompt) -> text; stops when the user moves on
        self.on_event = on_event or (lambda kind, data: None)
        self.pool_path = cfg.workspace / "persona_proposals.json"
        self._lock = threading.Lock()

    # -- settings --------------------------------------------------------------------
    @property
    def settings(self) -> dict[str, Any]:
        return {**DEFAULTS, **(self.cfg.raw.get("persona_creation") or {})}

    @property
    def enabled(self) -> bool:
        return bool(self.settings["enabled"])

    @property
    def persona_dirs(self) -> list[Path]:
        return list(self.cfg.personas_dirs) or [Path(self.cfg.personas_dir)]

    @property
    def learned_dir(self) -> Path:
        """Created personas go beside your own personas (the home folder's, when there is one)."""
        return self.persona_dirs[0] / LEARNED_DIR

    # -- trigger 1: a request no persona fits ------------------------------------------
    def note_no_fit(self, request: str, llm: Any, source: str = "") -> Created | None:
        """Pool an unhandled request; draft a persona once enough similar ones have gathered."""
        if not self.enabled:
            return None
        cfg = self.settings
        with self._lock:
            pending = self._read_pool()
        earlier = pending[-SAME_SPECIALIST_TOP_N:]
        sims = self.jev.same_specialist(request, [p["text"] for p in earlier])
        similar = [p for p, v in zip(earlier, sims) if v >= self.cfg.t("same_specialist")]
        if len(similar) + 1 >= int(cfg["min_cluster"]):
            requests = [p["text"] for p in similar] + [request]
            made = self._create(requests, None, llm, "no_fit",
                                f"{len(requests)} requests no persona fit, e.g. \"{request[:120]}\"")
            if made is not None:
                with self._lock:
                    left = [p for p in self._read_pool() if p["text"] not in set(requests)]
                    self._write_pool(left)
                return made
        with self._lock:
            pool = self._read_pool()
            pool.append({"text": request, "ts": time.time(), "source": source})
            self._write_pool(pool[-int(cfg["max_pending"]):])
        self.on_event("persona_proposal", {"request": request, "similar": len(similar), "needed": int(cfg["min_cluster"])})
        return None

    def pending(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._read_pool()

    def _read_pool(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self.pool_path.read_text(encoding="utf-8"))
            return [p for p in data.get("pending", []) if isinstance(p, dict) and p.get("text")]
        except (OSError, ValueError, AttributeError):
            return []

    def _write_pool(self, pending: list[dict[str, Any]]) -> None:
        self.pool_path.parent.mkdir(parents=True, exist_ok=True)
        self.pool_path.write_text(json.dumps({"pending": pending}, ensure_ascii=False, indent=1), encoding="utf-8")

    # -- trigger 2: a learned skill no persona wants -------------------------------------
    def note_skill(self, skill: Skill, llm: Any) -> Created | None:
        """Draft a persona around a learned skill that Jev, blind, gives to none of the personas."""
        if not self.enabled or skill.owner or not self._is_learned(skill):
            return None
        personas = load_personas(self.persona_dirs)
        best, none, probs = self.jev.skill_home(skill, personas)
        if best is not None or none < self.cfg.t("orphan"):
            return None                                       # some persona is the skill's home
        return self._create([skill.description], skill, llm, "orphan_skill",
                            f"learned skill {skill.name} (none of the personas fits it: {none:.2f})")

    def sweep(self, roster: dict[str, Skill], llm: Any) -> list[Created]:
        """note_skill() for every learned skill that has no owner yet (the CLI's /forge)."""
        made = []
        for skill in list(roster.values()):
            c = self.note_skill(skill, llm)
            if c:
                made.append(c)
        return made

    def _is_learned(self, skill: Skill) -> bool:
        return (self.skills_home / "learned").resolve() in Path(skill.path).resolve().parents

    # -- drafting ---------------------------------------------------------------------
    def _create(self, requests: list[str], skill: Skill | None, llm: Any, trigger: str, source: str) -> Created | None:
        cfg = self.settings
        existing = load_personas(self.persona_dirs, include_inactive=True)
        if sum(1 for p in load_learned(self.persona_dirs) if p.draft) >= int(cfg["max_live_drafts"]):
            self.on_event("persona_skipped", {"reason": f"{cfg['max_live_drafts']} drafts are waiting for approval"})
            return None
        evidence = (f"A how-to guide learned from a task that fits none of them:\nname: {skill.name}\n"
                    f"description: {skill.description}\n{skill.excerpt[:500]}" if skill else
                    "Requests that none of them could handle well:\n" + "\n".join(f"- {r[:300]}" for r in requests[:6]))
        reply = self.chat(llm, PERSONA_PROMPT.format(
            evidence=evidence, existing="\n".join(f"- {n}: {p.description}" for n, p in existing.items())))
        spec = _parse_spec(reply)
        if spec is None:
            self.on_event("persona_skipped", {"reason": "the model's persona description was unusable"})
            return None
        name = self._unique_name(spec["name"], existing)
        # one persona's purpose should not repeat another's: a near-duplicate is not created
        twin = self._twin(spec["description"], existing)
        if twin:
            self.on_event("persona_skipped", {"reason": f"too close to {twin}", "name": name})
            return None
        base = existing.get(self.cfg.default_persona) or next(iter(existing.values()))
        persona = Persona(
            name=name, description=spec["description"], system_prompt=spec["system_prompt"],
            model=copy.deepcopy(base.model), skills_allowed=[skill.name] if skill else [], tools=list(cfg["tools"]), can_request=[], mcp=[],
            max_steps=int(cfg["max_steps"]), code=True, fast_ok=base.fast_ok, draft=True, active=True, source=source)
        triggers = [skill.description] if skill else requests
        gate = self.gate(persona, existing, triggers)
        persona.active = gate.ok
        if not gate.ok:
            persona.source += f" | not live: {gate.note}"
        path = self._write(persona)
        if persona.active:
            self._claim(persona)
        made = Created(name, str(path), gate.ok, gate, trigger, skill.name if skill else None)
        self.on_event("persona_drafted", {"created": made, "persona": persona})
        return made

    def _unique_name(self, wanted: str, existing: dict[str, Persona]) -> str:
        base = slug(wanted).replace("-", "_")
        base = base if NAME_RE.match(base) else "specialist"
        name, n = base, 2
        while name in existing or (self.learned_dir / f"{name}.yaml").exists():
            name, n = f"{base}_{n}", n + 1
        return name

    def _twin(self, description: str, existing: dict[str, Persona]) -> str | None:
        names = list(existing)
        for i in range(0, len(names), SAME_SPECIALIST_TOP_N):
            chunk = names[i:i + SAME_SPECIALIST_TOP_N]
            scores = self.jev.same_specialist(description, [existing[n].description for n in chunk])
            for n, v in zip(chunk, scores):
                if v >= self.cfg.t("same_specialist"):
                    return n
        return None

    # -- the routing gate --------------------------------------------------------------
    def gate(self, persona: Persona, existing: dict[str, Persona], triggers: list[str]) -> Gate:
        """The draft must win what made it and take none of what other personas recently handled."""
        field_ = {n: p for n, p in existing.items() if p.active}
        field_[persona.name] = persona
        wins = sum(1 for t in triggers if self.jev.pick_persona(t, field_) == persona.name)
        stolen, checked = [], 0
        seen: set[str] = set(triggers)   # the requests that made it were handled by another persona; that is the point
        for turn in self.memory.recent_requests(int(self.settings["gate_probes"]) * 3):
            if checked >= int(self.settings["gate_probes"]):
                break
            if turn.persona == persona.name or turn.persona not in field_ or turn.user in seen:
                continue
            seen.add(turn.user)
            checked += 1
            if self.jev.pick_persona(turn.user, field_) == persona.name:
                stolen.append(turn.user[:120])
        won = wins * 2 >= len(triggers)
        ok = won and len(stolen) <= int(self.settings["max_steals"])
        note = "" if ok else ("wins too few of the requests that made it" if not won else
                              f"would take {len(stolen)} request(s) other personas handled")
        return Gate(ok, wins, len(triggers), stolen, checked, note)

    # -- files and review ----------------------------------------------------------------
    def _write(self, persona: Persona) -> Path:
        path = self.learned_dir / f"{persona.name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: getattr(persona, k) for k in ("name", "description", "system_prompt", "model", "skills_allowed",
                                                 "tools", "can_request", "mcp", "max_steps", "code", "fast_ok",
                                                 "draft", "active", "source")}
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8")
        return path

    def drafts(self) -> list[Persona]:
        """Created personas still waiting for approval, and any that failed the gate."""
        return [p for p in load_learned(self.persona_dirs) if p.draft or not p.active]

    def approve(self, name: str) -> bool:
        """Keep a draft for good (and switch it on if the gate had left it inactive)."""
        return self._edit(name, {"draft": False, "active": True})

    def activate(self, name: str) -> bool:
        """Switch an inactive draft on; it stays a draft."""
        return self._edit(name, {"active": True})

    def _claim(self, persona: Persona) -> None:
        """The learned skills a live persona was made for become its own: no other persona is offered
        them. (Not before it is live: a skill owned by an inactive persona would be offered to nobody.)"""
        for f in (self.skills_home / "learned").glob("*/SKILL.md"):
            s = parse_skill(f)
            if s and not s.owner and s.name in persona.skills_allowed:
                set_owner(f, persona.name)

    def delete(self, name: str) -> bool:
        """Remove a created persona; the skills it owned go back to being offered as before."""
        path = self.learned_dir / f"{name}.yaml"
        if not path.is_file():
            return False
        path.unlink()
        for f in (self.skills_home / "learned").glob("*/SKILL.md"):
            s = parse_skill(f)
            if s and s.owner == name:
                set_owner(f, None)
        return True

    def _edit(self, name: str, changes: dict[str, Any]) -> bool:
        path = self.learned_dir / f"{name}.yaml"
        if not path.is_file():
            return False
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data.update(changes)
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8")
        if data.get("active", True):
            self._claim(Persona(**data))
        return True


def _parse_spec(text: str) -> dict[str, str] | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    name, desc, prompt = (str(data.get(k) or "").strip() for k in ("name", "description", "system_prompt"))
    if not name or not (20 <= len(desc) <= 500) or len(prompt) < 20:
        return None
    return {"name": name, "description": " ".join(desc.split()), "system_prompt": prompt}


def set_owner(skill_file: Path, owner: str | None) -> None:
    """Add, change or (owner=None) remove `owner:` in a skill file's frontmatter."""
    text = skill_file.read_text(encoding="utf-8")
    m = re.match(r"^---\s*\n(.*?)\n---", text, re.S)
    if not m:
        return
    head = re.sub(r"^owner:.*\n?", "", m.group(1) + "\n", flags=re.M).rstrip("\n")
    if owner:
        head += f"\nowner: {owner}"
    skill_file.write_text(f"---\n{head}\n---" + text[m.end():], encoding="utf-8")
