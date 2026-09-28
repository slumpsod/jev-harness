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
    draft: bool = False    # learned from one run and not yet approved by the user

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
    return Skill(name=name, description=desc, body=body.strip(), path=str(path), draft=bool(meta.get("draft")))


def load_roster(roots: Iterable[Path], lessons: Path | None = None) -> dict[str, Skill]:
    """`lessons`: folder of <skill name>.md notes the harness learned while using a skill it
    doesn't own (e.g. a Hermes skill); they are appended to that skill's body."""
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
    if lessons and Path(lessons).is_dir():
        for note in Path(lessons).glob("*.md"):
            s = roster.get(note.stem)
            if s:
                extra = note.read_text(encoding="utf-8", errors="replace").strip()
                body = f"{s.body}\n\n## Lessons from past runs\n{extra}"
                roster[s.name] = Skill(s.name, s.description, body, s.path, s.draft)
    return roster
