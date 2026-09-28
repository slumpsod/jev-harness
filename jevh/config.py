"""Load config.yaml and resolve its relative paths."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    raw: dict[str, Any]
    root: Path
    jev: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    limits: dict[str, int] = field(default_factory=dict)
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)
    tools: dict[str, Any] = field(default_factory=dict)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)   # e.g. {'fast': {...}}
    default_persona: str = "daily_assistant"
    personas_dir: Path = ROOT / "personas"
    skills_dirs: list[Path] = field(default_factory=list)
    workspace: Path = ROOT / "workspace"

    def t(self, name: str) -> float:
        return float(self.thresholds[name])

    def lim(self, name: str) -> int:
        return int(self.limits[name])


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path or os.environ.get("JEVH_CONFIG") or ROOT / "config.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.resolve().parent

    def rel(p: str) -> Path:
        q = Path(os.path.expandvars(os.path.expanduser(p)))
        return q if q.is_absolute() else base / q

    return Config(
        raw=raw,
        root=base,
        jev=raw.get("jev", {}),
        thresholds=raw.get("thresholds", {}),
        limits=raw.get("limits", {}),
        providers=raw.get("providers", {}),
        tools=raw.get("tools", {}),
        models=raw.get("models") or {},
        default_persona=raw.get("default_persona", "daily_assistant"),
        personas_dir=rel(raw.get("personas_dir", "personas")),
        skills_dirs=[rel(p) for p in raw.get("skills_dirs", ["skills"])],
        workspace=rel(raw.get("workspace", "workspace")),
    )
