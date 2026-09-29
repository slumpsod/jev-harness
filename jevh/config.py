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
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    default_persona: str = "daily_assistant"
    personas_dir: Path = ROOT / "personas"
    skills_dirs: list[Path] = field(default_factory=list)
    workspace: Path = ROOT / "workspace"

    def t(self, name: str) -> float:
        return float(self.thresholds[name] if name in self.thresholds else DEFAULT_THRESHOLDS[name])

    def lim(self, name: str) -> int:
        return int(self.limits[name] if name in self.limits else DEFAULT_LIMITS[name])


# Settings added after the first config.yaml: an older config keeps working with the tuned values.
DEFAULT_THRESHOLDS = {"tool": 0.65, "mcp_fits": 0.5, "fact": 0.88, "same_fact": 0.5, "learn_skill": 0.5,
                      "follow_topic": 0.25, "follow_reply": 0.4,
                      "verify_claims": 0.8, "verify_delivers": 0.8, "verify_honest": 0.8,
                      "verify_accurate": 0.4, "stall": 0.6, "testable": 0.55,
                      "compact_keep": 0.8}
DEFAULT_LIMITS = {"max_mcp_tools": 3, "memory_search": 10, "memory_facts": 6, "follow_turns": 2,
                  "ask_timeout": 1800, "progress_every": 8, "compact_tokens": 60000, "keep_steps": 3,
                  "keep_reasoning": 40000, "thinking_watch": 0, "watch_every": 4000, "watch_after": 6000}


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
        mcp_servers=raw.get("mcp_servers") or {},
        default_persona=raw.get("default_persona", "daily_assistant"),
        personas_dir=rel(raw.get("personas_dir", "personas")),
        skills_dirs=[rel(p) for p in raw.get("skills_dirs", ["skills"])],
        workspace=rel(raw.get("workspace", "workspace")),
    )
