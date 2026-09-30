"""Test cases for one refinement attempt: the failure replayed, variants of it the local model
wrote, and control tasks that already work (a fix must not break them).

split: dev      the fixer sees these (with how the old harness did on them)
       holdout  never shown to the fixer: they show whether the fix generalizes
       control  unrelated tasks that work today: they catch a fix that breaks other things"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from ..config import ROOT

CONTROLS = ROOT / "evals" / "refine_controls.json"


@dataclass
class Case:
    id: str
    request: str
    checker: dict
    split: str = "dev"            # dev | holdout | control
    origin: str = "variant"       # replay | variant | control
    persona: str | None = None    # pinned persona (None: Jev routes)
    good_answer: str = ""         # what validated the checker
    bad_answer: str = ""
    good_files: dict = field(default_factory=dict)   # {name: content} a correct run hands over (validation)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Case":
        return cls(**{f.name: d[f.name] for f in fields(cls) if f.name in d})


def save(path: Path, cases: list[Case]) -> None:
    path.write_text(json.dumps([c.to_dict() for c in cases], indent=1, ensure_ascii=False), encoding="utf-8")


def load(path: Path) -> list[Case]:
    return [Case.from_dict(d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]


def controls(path: Path | None = None) -> list[Case]:
    p = Path(path or CONTROLS)
    if not p.is_file():
        return []
    out = []
    for i, d in enumerate(json.loads(p.read_text(encoding="utf-8"))):
        c = Case.from_dict({"id": f"control-{i + 1}", **d})
        c.split, c.origin = "control", "control"
        out.append(c)
    return out
