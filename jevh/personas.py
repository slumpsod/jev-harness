"""Persona definitions: one YAML file per persona in personas/."""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field, fields
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

    def allows_skill(self, skill_name: str) -> bool:
        return any(fnmatch.fnmatch(skill_name, pat) for pat in self.skills_allowed)

    def allows_mcp(self, key: str) -> bool:
        return any(fnmatch.fnmatch(key, pat) for pat in self.mcp)

    def may_request(self, other: str) -> bool:
        return other != self.name and ("*" in self.can_request or other in self.can_request)


REQUIRED = ("description", "system_prompt", "model")


def _why(e: Exception) -> str:
    mark = getattr(e, "problem_mark", None)
    if isinstance(e, yaml.YAMLError) and mark is not None:
        return f"bad YAML: {getattr(e, 'problem', None) or 'parse error'} (line {mark.line + 1})"
    return (str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}")[:300]
LISTS = ("skills_allowed", "tools", "can_request", "mcp")


def load_personas(directories: Path | list[Path], problems: list[dict] | None = None) -> dict[str, Persona]:
    """Every persona file in `directories`; on a name clash the first folder wins (your own folder goes
    first, so your persona replaces a built-in one). A file that can't be used is skipped and noted in
    `problems` ({kind, name, path, error}) instead of stopping the harness; an unknown key is dropped."""
    dirs = [directories] if isinstance(directories, (str, Path)) else list(directories)
    problems = problems if problems is not None else []
    personas: dict[str, Persona] = {}
    keys = {f.name for f in fields(Persona)}
    for directory in dirs:
        for path in sorted(Path(directory).glob("*.y*ml")):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                if not isinstance(data, dict):
                    raise ValueError("not a YAML mapping")
                data.setdefault("name", path.stem)
                if data["name"] in personas:
                    continue
                missing = [k for k in REQUIRED if not data.get(k)]
                if missing:
                    raise ValueError(f"missing {', '.join(missing)}")
                if not isinstance(data["model"], dict) or not data["model"].get("provider"):
                    raise ValueError("model needs a provider (model: {provider: ..., model: ...})")
                bad = [k for k in LISTS if k in data and not isinstance(data[k], list)]
                if bad:
                    raise ValueError(f"{', '.join(bad)} must be a list")
                extra = sorted(set(data) - keys)
                if extra:
                    problems.append({"kind": "persona", "name": data["name"], "path": str(path),
                                     "error": f"ignored unknown key(s): {', '.join(extra)}", "skipped": False})
                p = Persona(**{k: v for k, v in data.items() if k in keys})
            except Exception as e:  # noqa: BLE001 - one bad file must not stop the harness
                problems.append({"kind": "persona", "name": path.stem, "path": str(path),
                                 "error": _why(e), "skipped": True})
                continue
            personas[p.name] = p
    if not personas:
        raise FileNotFoundError(f"no usable persona files in {', '.join(map(str, dirs))}"
                                + (f" ({problems[-1]['error']})" if problems else ""))
    return personas
