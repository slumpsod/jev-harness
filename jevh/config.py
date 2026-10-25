"""Load config.yaml and resolve its relative paths.

Settings are laid on top of each other, last wins:
  config.yaml (in the checkout, updated with it)
  <home>/config.yaml (your own folder outside the checkout: JEVH_HOME, default ~/.jevh, if it exists;
                      its relative paths are relative to that folder)
  config.local.yaml (next to config.yaml, ignored by git)
  <workspace>/settings.yaml (what the dashboard's Settings page saved: models and web search only; see jevh/settings.py)
With a home folder, <home>/personas and <home>/skills are searched before the built-in ones, so a
persona or skill of yours with the same name replaces the built-in one; learned skills go to
<home>/skills. See docs/your-data.md.
"""
from __future__ import annotations

import os
import sys
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
    personas_dir: Path = ROOT / "personas"                       # the built-in personas
    skills_dirs: list[Path] = field(default_factory=list)          # first one gets learned skills
    workspace: Path = ROOT / "workspace"
    personas_dirs: list[Path] = field(default_factory=list)       # searched in order; first wins on a name
    home: Path | None = None                                       # your folder outside the checkout
    source: Path | None = None                                     # the config.yaml this was loaded from (settings.reload)

    def t(self, name: str) -> float:
        # light Jev (jevh/light.py) has thresholds of its own for the decisions its small model makes: its confidence is
        # not semif's, which the thresholds below were tuned on
        if str((self.jev or {}).get("mode") or "") == "light":
            over = ((self.jev or {}).get("light") or {}).get("thresholds")
            if isinstance(over, dict) and name in over:
                return float(over[name])
        return float(self.thresholds[name] if name in self.thresholds else DEFAULT_THRESHOLDS[name])

    def lim(self, name: str) -> int:
        return int(self.limits[name] if name in self.limits else DEFAULT_LIMITS[name])


# Settings added after the first config.yaml: an older config keeps working with the tuned values.
DEFAULT_THRESHOLDS = {"no_fit": 0.6, "owned": 0.3, "orphan": 0.5, "same_specialist": 0.6, "tool": 0.65, "mcp_fits": 0.5, "fact": 0.88, "same_fact": 0.5, "learn_skill": 0.5,
                      "follow_topic": 0.25, "follow_reply": 0.4,
                      "verify_claims": 0.8, "verify_delivers": 0.8, "verify_honest": 0.8,
                      "verify_accurate": 0.4, "stall": 0.6, "testable": 0.55, "searching": 0.25,
                      "compact_keep": 0.8, "rule_block": 0.8,
                      "review_risk": 0.65, "review_severe": 0.2, "failure_gave_up": 0.85, "failure_missed": 0.85,
                      "route_hook": 0.85,
                      "interject_status": 0.4, "interject_replace": 0.5, "newsroom_about": 0.7}
DEFAULT_LIMITS = {"max_mcp_tools": 3, "memory_search": 10, "memory_facts": 6, "follow_turns": 2,
                  "ask_timeout": 1800, "progress_every": 8, "compact_tokens": 60000, "keep_steps": 3,
                  "keep_reasoning": 40000, "thinking_watch": 0, "watch_every": 4000, "watch_after": 6000,
                  "follow_always": 1, "session_context": 1, "narrate": 1}


def shell_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """`env` (default os.environ) with this Python's folder first on PATH, so a shell command's `python`
    is the harness's own interpreter. On macOS and Linux outside an activated venv there is often no
    `python` at all (only python3): project.sh("python -m pytest") failed with "python: not found"."""
    env = dict(os.environ if env is None else env)
    here = os.path.dirname(sys.executable)
    env["PATH"] = here + os.pathsep + env.get("PATH", "") if here else env.get("PATH", "")
    return env


def merged(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """`base` with `over` laid on top, nested dicts merged key by key (a None value removes the key)."""
    out = dict(base)
    for k, v in over.items():
        if v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merged(out[k], v)
        else:
            out[k] = v
    return out


PATH_KEYS = ("workspace", "personas_dir")
PATH_LISTS = ("skills_dirs", "personas_dirs")


def home_path() -> Path:
    """Where your own folder goes (it is used only if it exists): JEVH_HOME, default ~/.jevh."""
    return Path(os.path.expandvars(os.path.expanduser(os.environ.get("JEVH_HOME") or "~/.jevh")))


def _expand(p: str, base: Path) -> str:
    q = Path(os.path.expandvars(os.path.expanduser(str(p))))
    return str(q if q.is_absolute() else base / q)


SETTINGS_FILE = "settings.yaml"          # in the workspace: what the dashboard's Settings page saved
SETTINGS_KEYS = ("models", "search", "appearance", "jev", "thinking")   # the only sections that file may set; anything else is ignored


def read_settings(workspace: Path) -> dict[str, Any]:
    """The Settings page's saved choices (only SETTINGS_KEYS), or {} when there are none or the file cannot be read: a
    hand-edited, half-written or locked settings file must never stop the harness from starting."""
    try:
        data = yaml.safe_load((Path(workspace) / SETTINGS_FILE).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError, UnicodeDecodeError) as e:
        if not isinstance(e, FileNotFoundError):
            print(f"jevh: ignoring {Path(workspace) / SETTINGS_FILE}: {type(e).__name__}", file=sys.stderr)
        return {}
    return {k: v for k, v in data.items() if k in SETTINGS_KEYS} if isinstance(data, dict) else {}


def load_config(path: str | os.PathLike | None = None, settings_dir: str | os.PathLike | None = None) -> Config:
    """`settings_dir`: where to read the Settings page's saved file from, when that is not the configured workspace (a config
    whose `workspace` was changed after loading: the settings are saved where the config now says, so they are read there)."""
    path = Path(path or os.environ.get("JEVH_CONFIG") or ROOT / "config.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.resolve().parent
    # your own folder outside the checkout (tests ignore it unless they set JEVH_HOME)
    home = home_path() if os.environ.get("JEVH_HOME") or not os.environ.get("JEVH_NO_LOCAL_CONFIG") else None
    home = home if home and home.is_dir() else None
    if home and (home / "config.yaml").is_file():
        over = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}
        for k in PATH_KEYS:   # relative paths in it are relative to the home folder
            if isinstance(over.get(k), str):
                over[k] = _expand(over[k], home)
        for k in PATH_LISTS:
            if isinstance(over.get(k), list):
                over[k] = [_expand(p, home) for p in over[k]]
        raw = merged(raw, over)
    # this machine's own settings (models, Jev, paths), kept out of git: config.local.yaml next to it
    local = base / "config.local.yaml"
    if local.is_file() and not os.environ.get("JEVH_NO_LOCAL_CONFIG"):
        raw = merged(raw, yaml.safe_load(local.read_text(encoding="utf-8")) or {})

    def rel(p: str) -> Path:
        q = Path(os.path.expandvars(os.path.expanduser(p)))
        return q if q.is_absolute() else base / q

    # what the dashboard's Settings page saved (jevh/settings.py): last, so a choice made there wins over every file above
    if not os.environ.get("JEVH_NO_LOCAL_CONFIG"):
        raw = merged(raw, read_settings(Path(settings_dir) if settings_dir else rel(raw.get("workspace", "workspace"))))

    def dirs(own: Path | None, listed: list[Path]) -> list[Path]:
        out: list[Path] = []
        for d in ([own] if own else []) + listed:
            if d.resolve() not in [o.resolve() for o in out]:
                out.append(d)
        return out

    personas_dir = rel(raw.get("personas_dir", "personas"))
    return Config(
        raw=raw,
        root=base,
        jev=raw.get("jev", {}),
        thresholds=raw.get("thresholds", {}),
        limits=raw.get("limits", {}),
        providers=raw.get("providers", {}),
        tools=raw.get("tools", {}),
        models={k: v for k, v in (raw.get("models") or {}).items() if v},   # (a null tier is "not set")
        mcp_servers=raw.get("mcp_servers") or {},
        default_persona=raw.get("default_persona", "daily_assistant"),
        personas_dir=personas_dir,
        skills_dirs=dirs(home / "skills" if home else None, [rel(p) for p in raw.get("skills_dirs", ["skills"])]),
        workspace=rel(raw.get("workspace", "workspace")),
        personas_dirs=dirs(home / "personas" if home else None,
                           [rel(p) for p in raw.get("personas_dirs", [])] + [personas_dir]),
        home=home,
        source=path,
    )
