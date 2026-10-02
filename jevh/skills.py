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


def parse_skill(path: Path, problems: list[dict] | None = None) -> Skill | None:
    """The skill at `path`, or None (with the reason in `problems`) when it can't be used."""
    def skip(why: str) -> None:
        if problems is not None:
            problems.append({"kind": "skill", "name": path.parent.name, "path": str(path), "error": why, "skipped": True})

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return skip(f"can't read it: {e}")
    m = FRONTMATTER_RE.match(text)
    meta: dict = {}
    if m:
        try:
            meta = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError as e:
            mark = getattr(e, "problem_mark", None)
            return skip(f"bad frontmatter YAML: {getattr(e, 'problem', None) or 'parse error'}"
                        + (f" (line {mark.line + 2})" if mark is not None else ""))
        if not isinstance(meta, dict):
            return skip("frontmatter is not a YAML mapping")
        body = text[m.end():]
    else:
        body = text
    name = str(meta.get("name") or path.parent.name).strip()
    desc = " ".join(str(meta.get("description") or "").split())
    if not name or not desc:
        return skip("no description in its frontmatter (---\nname: ...\ndescription: ...\n---)")
    return Skill(name=name, description=desc, body=body.strip(), path=str(path), draft=bool(meta.get("draft")))


def load_roster(roots: Iterable[Path], lessons: Path | Iterable[Path] | None = None,
                problems: list[dict] | None = None) -> dict[str, Skill]:
    """`lessons`: folder(s) of <skill name>.md notes the harness learned while using a skill it
    doesn't own (e.g. a Hermes skill); they are appended to that skill's body. A skill that can't
    be used is skipped, with the reason in `problems`."""
    roster: dict[str, Skill] = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in sorted(root.rglob("SKILL.md")):
            if any(part.startswith(".") for part in path.relative_to(root).parts):
                continue  # .archive/ and other hidden dirs
            skill = parse_skill(path, problems)
            if skill and skill.name not in roster:  # first root wins on name clashes
                roster[skill.name] = skill
    folders = [lessons] if isinstance(lessons, (str, Path)) else list(lessons or [])
    notes: dict[str, list[str]] = {}
    for folder in folders:
        for note in sorted(Path(folder).glob("*.md")) if Path(folder).is_dir() else []:
            notes.setdefault(note.stem, []).append(note.read_text(encoding="utf-8", errors="replace").strip())
    for name, extra in notes.items():
        s = roster.get(name)
        if s:
            body = f"{s.body}\n\n## Lessons from past runs\n" + "\n".join(extra)
            roster[s.name] = Skill(s.name, s.description, body, s.path, s.draft)
    return roster
