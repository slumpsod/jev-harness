"""Skill roster: every <root>/**/SKILL.md (Hermes / Claude skill format).

Frontmatter needs `name` and `description`; the body is what gets injected into a
worker's system prompt when Jev picks the skill.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
EXCERPT_CHARS = 700


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    path: str

    @property
    def excerpt(self) -> str:
        return self.body[:EXCERPT_CHARS]


def parse_skill(path: Path) -> Skill | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = FRONTMATTER_RE.match(text)
    meta: dict = {}
    if m:
        try:
            meta = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        body = text[m.end():]
    else:
        body = text
    name = str(meta.get("name") or path.parent.name).strip()
    desc = " ".join(str(meta.get("description") or "").split())
    if not name or not desc:
        return None
    return Skill(name=name, description=desc, body=body.strip(), path=str(path))


def load_roster(roots: Iterable[Path]) -> dict[str, Skill]:
    roster: dict[str, Skill] = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in sorted(root.rglob("SKILL.md")):
            if any(part.startswith(".") for part in path.relative_to(root).parts):
                continue  # .archive/ and other hidden dirs
            skill = parse_skill(path)
            if skill and skill.name not in roster:  # first root wins on name clashes
                roster[skill.name] = skill
    return roster
