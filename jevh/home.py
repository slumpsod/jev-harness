"""Move your own things out of the checkout into a home folder, so updates never touch them.

    jevh home              # where your things are now, and what `jevh home move` would do (changes nothing)
    jevh home move         # do it (asks first; --yes to skip the question). Stop the CLI and dashboard first.

The home folder is JEVH_HOME, default ~/.jevh. Once it exists, the harness lays <home>/config.yaml over
config.yaml and looks in <home>/personas and <home>/skills before the built-in ones (jevh/config.py).

What moves (see docs/your-data.md for the why):
- config.local.yaml -> <home>/config.yaml (relative paths in it are made absolute, so they still point
  at the same folders), plus any settings you changed in config.yaml itself (only the changed keys;
  config.yaml goes back to the shipped version)
- workspace/ (memory, logs, sessions) -> <home>/workspace, when it is inside the checkout
- skills/learned and skills/_lessons -> <home>/skills/
- lessons the harness appended to a built-in SKILL.md -> <home>/skills/_lessons/<name>.md (the built-in
  file goes back to the shipped version, and keeps getting updates)
- personas and skills you added -> <home>/personas, <home>/skills
- built-in personas and skills you edited -> a copy in <home> (yours replaces the built-in one by name),
  and the checkout's file goes back to the shipped version. Your copy no longer gets updates to it.
Edits to the code itself are only listed: keep them on a git branch, or send them upstream.
Nothing is deleted: everything is moved or copied before a checkout file is reset.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from .config import PATH_KEYS, PATH_LISTS, ROOT, home_path, load_config, merged

LESSONS_HEADING = "## Lessons from past runs"
HOME_README = """# Your jev-harness folder

The harness keeps your own things here, outside its code checkout, so updates never touch them.

- config.yaml   your settings, laid over the checkout's config.yaml (relative paths are relative to this folder)
- personas/     your personas; one with a built-in persona's name replaces it
- skills/       your skills (same rule), learned skills (skills/learned) and lesson notes (skills/_lessons)
- workspace/    memory, failure log, reviews, notebook, newsroom, usage, sessions and settings.yaml (what the Settings page saved)

Set JEVH_HOME to use another folder. See docs/your-data.md in the checkout.
"""


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def changed_files() -> tuple[list[str], list[str]]:
    """(tracked files you edited, files you added that git doesn't ignore), relative to the checkout."""
    try:
        edited = git("diff", "--name-only", "HEAD").splitlines()
        added = git("ls-files", "--others", "--exclude-standard").splitlines()
    except (OSError, subprocess.CalledProcessError):
        return [], []
    return edited, added


def yaml_diff(old, new):
    """The keys of `new` that differ from `old`, as an overlay for config.merged() (None removes a key)."""
    if not isinstance(old, dict) or not isinstance(new, dict):
        return new
    out = {}
    for k in set(old) | set(new):
        if k not in new:
            out[k] = None
        elif k not in old:
            out[k] = new[k]
        elif old[k] != new[k]:
            out[k] = yaml_diff(old[k], new[k]) if isinstance(old[k], dict) and isinstance(new[k], dict) else new[k]
    return out


def appended(rel: str, heading: str | None = LESSONS_HEADING) -> str | None:
    """If the only change to a tracked file is text added at its end (after `heading`, if given:
    lessons the harness appended to a SKILL.md), that text."""
    try:
        shipped = git("show", f"HEAD:{rel}").rstrip()
        now = (ROOT / rel).read_text(encoding="utf-8")
    except (OSError, subprocess.CalledProcessError):
        return None
    if not now.startswith(shipped):
        return None
    extra = now[len(shipped):]
    if heading:
        if heading not in extra:
            return None
        extra = extra.split(heading, 1)[1]
    return extra.strip("\n") + "\n" if extra.strip() else None


def plan(home: Path | None = None) -> list[dict]:
    """What `move` would do: [{what, src, dst, how, note}] with how in move | copy-reset | lessons |
    config | settings | list."""
    home = Path(home or home_path())
    cfg = load_config()
    steps: list[dict] = []
    edited, added = changed_files()
    local = cfg.root / "config.local.yaml"
    if local.is_file():
        steps.append({"what": "your settings (config.local.yaml)", "src": local, "dst": home / "config.yaml",
                      "how": "config", "note": "relative paths made absolute"})
    if "config.yaml" in edited:
        steps.append({"what": "settings you changed in config.yaml", "src": cfg.root / "config.yaml",
                      "dst": home / "config.yaml", "how": "settings", "note": "only the changed keys; config.yaml is reset"})
        edited.remove("config.yaml")
    ws = Path(cfg.workspace)
    if ws.is_dir() and ROOT in ws.resolve().parents:
        steps.append({"what": "your data (workspace)", "src": ws, "dst": home / "workspace", "how": "move",
                      "note": "memory, logs, sessions"})
    skills = ROOT / "skills"
    tracked = set(git("ls-files", "skills").splitlines()) if (ROOT / ".git").exists() else set()
    for sub, what in (("learned", "learned skill"), ("_lessons", "lesson notes for")):
        for p in sorted((skills / sub).iterdir()) if (skills / sub).is_dir() else []:
            rel = p.relative_to(ROOT).as_posix()
            if rel in tracked:   # shipped with the harness (edits to it are handled below)
                continue
            steps.append({"what": f"{what} {p.stem}", "src": p, "dst": home / "skills" / sub / p.name,
                          "how": "lessons" if sub == "_lessons" else "move",
                          "lessons": p.read_text(encoding="utf-8", errors="replace") if sub == "_lessons" else None})
    for rel in added:
        parts = Path(rel).parts
        if parts[0] == "personas" and len(parts) == 2:
            steps.append({"what": f"your persona {Path(rel).stem}", "src": ROOT / rel, "dst": home / rel, "how": "move"})
        elif parts[0] == "skills" and len(parts) >= 3 and parts[1] not in ("learned", "_lessons", ".archive"):
            d = ROOT / "skills" / parts[1]
            if not any(s["src"] == d for s in steps):
                shipped = git("ls-files", f"skills/{parts[1]}").strip()
                steps.append({"what": f"your skill {parts[1]}", "src": d, "dst": home / "skills" / parts[1],
                              "how": "copy-reset" if shipped else "move",
                              "note": "files added to a built-in skill" if shipped else ""})
        elif parts[0] not in ("personas", "skills"):
            steps.append({"what": f"a file you added: {rel}", "src": ROOT / rel, "dst": None, "how": "list",
                          "note": "stays here; move it yourself if it's yours"})
    for rel in edited:
        parts = Path(rel).parts
        if parts[0] == "personas" and len(parts) == 2:
            steps.append({"what": f"your edits to the persona {Path(rel).stem}", "src": ROOT / rel, "dst": home / rel,
                          "how": "copy-reset", "note": "your copy replaces the built-in one"})
        elif rel.startswith("skills/_lessons/") and (notes := appended(rel, None)):
            steps.append({"what": f"lessons added to {Path(rel).name}", "src": ROOT / rel,
                          "dst": home / "skills" / "_lessons" / Path(rel).name, "how": "lessons", "lessons": notes})
        elif parts[0] == "skills" and len(parts) >= 3 and parts[1] != "_lessons":
            name = parts[1]
            notes = appended(rel) if parts[-1] == "SKILL.md" else None
            if notes:
                steps.append({"what": f"lessons learned for the skill {name}", "src": ROOT / rel,
                              "dst": home / "skills" / "_lessons" / f"{name}.md", "how": "lessons", "lessons": notes})
            elif not any(s["src"] == ROOT / "skills" / name for s in steps):
                steps.append({"what": f"your edits to the skill {name}", "src": ROOT / "skills" / name,
                              "dst": home / "skills" / name, "how": "copy-reset",
                              "note": "your copy replaces the built-in one"})
        else:
            steps.append({"what": f"code you edited: {rel}", "src": ROOT / rel, "dst": None, "how": "list",
                          "note": "stays here; keep it on a git branch, or send it upstream"})
    return steps


def _write_config(path: Path, over: dict) -> None:
    current = yaml.safe_load(path.read_text(encoding="utf-8")) or {} if path.is_file() else {}
    path.write_text(yaml.safe_dump(merged(current, over), sort_keys=False, allow_unicode=True), encoding="utf-8")


def move(home: Path | None = None) -> list[str]:
    """Carry out plan(). Returns what was done, one line each."""
    home = Path(home or home_path())
    steps = plan(home)
    done: list[str] = []
    for d in ("personas", "skills"):
        (home / d).mkdir(parents=True, exist_ok=True)
    if not (home / "README.md").exists():
        (home / "README.md").write_text(HOME_README, encoding="utf-8")
    # settings first: config.yaml edits, then config.local.yaml on top (it won over them before too)
    for s in sorted((s for s in steps if s["how"] in ("settings", "config")), key=lambda s: s["how"] != "settings"):
        if s["how"] == "settings":
            shipped = yaml.safe_load(git("show", "HEAD:config.yaml")) or {}
            _write_config(s["dst"], yaml_diff(shipped, yaml.safe_load(s["src"].read_text(encoding="utf-8")) or {}))
            git("checkout", "--", "config.yaml")
        else:
            over = yaml.safe_load(s["src"].read_text(encoding="utf-8")) or {}
            for k in PATH_KEYS:   # they were relative to the checkout; now they'd be relative to home
                if isinstance(over.get(k), str):
                    over[k] = str((ROOT / os.path.expanduser(over[k])).resolve())
            for k in PATH_LISTS:
                if isinstance(over.get(k), list):
                    over[k] = [str((ROOT / os.path.expanduser(p)).resolve()) for p in over[k]]
            _write_config(s["dst"], over)
            shutil.move(str(s["src"]), str(home / "config.local.yaml.old"))   # the original, for reference
        done.append(f"{s['what']} -> {s['dst']}")
    for s in steps:
        how, src, dst = s["how"], s["src"], s["dst"]
        if how in ("settings", "config"):
            continue
        if how == "list":
            done.append(f"left in place: {s['what']} ({s['note']})")
            continue
        if dst.exists() and how != "lessons":
            done.append(f"skipped {s['what']}: {dst} already exists")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if how == "move":
            shutil.move(str(src), str(dst))
            if s["what"] == "your data (workspace)":
                _write_config(home / "config.yaml", {"workspace": "workspace"})
        elif how == "copy-reset":
            rel = src.relative_to(ROOT).as_posix()
            shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)
            git("checkout", "--", rel)
            for extra in git("ls-files", "--others", "--exclude-standard", rel).splitlines():
                (ROOT / extra).unlink()   # files you added inside a built-in skill: copied above
        elif how == "lessons":
            with dst.open("a", encoding="utf-8") as f:
                f.write(s["lessons"])
            rel = src.relative_to(ROOT).as_posix()
            if rel in git("ls-files", rel).splitlines():
                git("checkout", "--", rel)   # a shipped file: back to the shipped text
            else:
                src.unlink()
        done.append(f"{s['what']} -> {dst}")
    return done


def show(steps: list[dict], home: Path) -> str:
    verbs = {"move": "move", "copy-reset": "copy, then reset the checkout's file", "lessons": "move the lessons",
             "config": "move", "settings": "move", "list": "leave"}
    lines = [f"home folder: {home} ({'exists' if home.is_dir() else 'not created yet'})"]
    if not steps:
        lines.append("nothing of yours is left in the checkout: updates won't touch your things")
    for s in steps:
        to = f" -> {s['dst']}" if s["dst"] else ""
        lines.append(f"  {verbs[s['how']]}: {s['what']}{to}" + (f"  ({s['note']})" if s.get("note") else ""))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="jevh home", description="Move your own things out of the checkout.")
    ap.add_argument("action", nargs="?", choices=["show", "move"], default="show")
    ap.add_argument("--yes", action="store_true", help="don't ask before moving")
    a = ap.parse_args(argv)
    home = home_path()
    steps = plan(home)
    print(show(steps, home))
    if a.action == "show" or not any(s["how"] != "list" for s in steps):
        if a.action == "show" and steps:
            print("\njevh home move does this (stop the CLI and dashboard first)")
        return 0
    if not a.yes and input("\nmove them now? (stop the CLI and dashboard first) [y/N] ").strip().lower() not in ("y", "yes"):
        return 1
    for line in move(home):
        print(f"  {line}")
    cfg = load_config()   # check the result loads
    print(f"done. The harness now reads {cfg.home or home} first; restart it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
