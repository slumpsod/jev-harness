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

    def allows_skill(self, skill_name: str) -> bool:
        return any(fnmatch.fnmatch(skill_name, pat) for pat in self.skills_allowed)

    def may_request(self, other: str) -> bool:
        return other != self.name and ("*" in self.can_request or other in self.can_request)


def load_personas(directory: Path) -> dict[str, Persona]:
    personas: dict[str, Persona] = {}
    for path in sorted(Path(directory).glob("*.y*ml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data.setdefault("name", path.stem)
        p = Persona(**data)
        personas[p.name] = p
    if not personas:
        raise FileNotFoundError(f"no persona files in {directory}")
    return personas
