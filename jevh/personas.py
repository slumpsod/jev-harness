"""Persona definitions: one YAML file per persona in personas/."""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Persona:
    name: str
    description: str                      # what Jev routes on; keep it concrete
    system_prompt: str
    model: dict[str, Any]                 # {provider, model, temperature, max_tokens, fallback: {...}}
    skills_allowed: list[str] = field(default_factory=lambda: ["*"])
    tools: list[str] = field(default_factory=list)          # modules pre-imported into the REPL
    can_request: list[str] = field(default_factory=list)    # persona names, or ["*"]
    max_steps: int = 12
    code: bool = True                     # False = plain chat persona, no REPL
    fast_ok: bool = True                  # Jev may send easy tasks to config models.fast
    mcp: list[str] = field(default_factory=list)            # globs over "server.tool" MCP keys
    draft: bool = False                   # created by the harness (jevh/forge.py) and not yet approved
    active: bool = True                   # False: written but kept out of routing (failed the routing gate)
    source: str = ""                      # what led to it, for a draft: shown when reviewing

    def allows_skill(self, skill_name: str) -> bool:
        return any(fnmatch.fnmatch(skill_name, pat) for pat in self.skills_allowed)

    def offered_skill(self, skill) -> bool:
        """Whether this persona may be shown `skill`: one owned by a persona goes to that persona
        alone (it is how a created persona keeps a skill out of every other persona's context)."""
        if skill.owner:
            return skill.owner == self.name
        return self.allows_skill(skill.name)

    def allows_mcp(self, key: str) -> bool:
        return any(fnmatch.fnmatch(key, pat) for pat in self.mcp)

    def may_request(self, other: str) -> bool:
        return other != self.name and ("*" in self.can_request or other in self.can_request)


LEARNED_DIR = "learned"   # personas/learned/: personas the harness created, apart from the hand-written ones


def load_personas(directory: Path, include_inactive: bool = False) -> dict[str, Persona]:
    """The hand-written personas in `directory`, plus the harness-created ones in `directory/learned/`.
    A created persona never replaces a hand-written one of the same name, and an unreadable one is
    skipped: a bad draft must not stop the harness from starting. Inactive drafts (failed the routing
    gate) are left out unless `include_inactive`."""
    personas: dict[str, Persona] = {}
    for path in sorted(Path(directory).glob("*.y*ml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data.setdefault("name", path.stem)
        p = Persona(**data)
        personas[p.name] = p
    if not personas:
        raise FileNotFoundError(f"no persona files in {directory}")
    for p in load_learned(directory):
        if p.name not in personas and (p.active or include_inactive):
            personas[p.name] = p
    return personas


def load_learned(directory: Path) -> list[Persona]:
    """Every readable persona in `directory/learned/`, drafts and inactive ones included."""
    out: list[Persona] = []
    for path in sorted((Path(directory) / LEARNED_DIR).glob("*.y*ml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            data.setdefault("name", path.stem)
            out.append(Persona(**data))
        except (OSError, yaml.YAMLError, TypeError, ValueError):
            continue
    return out
