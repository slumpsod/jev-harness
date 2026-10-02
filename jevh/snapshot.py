"""Snapshots of your data, taken before every update (and undo) so a version change can be undone
for the data too: a newer version may change a database in a way the older code can't read.

    jevh update --snapshots                 # list them
    jevh update --restore-data <name>       # put one back (stop the CLI and dashboard first)

What is saved: the workspace (memory, failures, reviews, usage) except the per-session folders
(sessions/, which only grow and aren't changed by updates), learned skills and lesson notes, and your
config files (config.local.yaml, <home>/config.yaml). Databases are copied with SQLite's backup API,
so a copy taken while the harness runs is still consistent. Files over 1 GB are left out and listed.
The newest KEEP snapshots are kept, in <workspace>/snapshots/.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import time
from pathlib import Path

KEEP = 5
MAX_FILE = 1 << 30
SKIP_DIRS = {"sessions", "snapshots"}
DB_SIDECARS = ("-wal", "-shm", "-journal")


def folder(cfg) -> Path:
    return Path(cfg.workspace) / "snapshots"


def items(cfg) -> list[tuple[str, Path]]:
    """(label, path) of everything a snapshot holds; the label names it inside the snapshot."""
    home = Path(cfg.skills_dirs[0])
    out = [("workspace", Path(cfg.workspace)), ("skills-learned", home / "learned"),
           ("skills-lessons", home / "_lessons"), ("config.local.yaml", Path(cfg.root) / "config.local.yaml")]
    if getattr(cfg, "home", None):
        out.append(("home-config.yaml", Path(cfg.home) / "config.yaml"))
    return [(label, p) for label, p in out if p.exists()]


def _copy_file(src: Path, dst: Path, skipped: list[str]) -> None:
    if src.name.endswith(DB_SIDECARS):
        return   # part of a database: the backup below includes what they hold
    if src.stat().st_size > MAX_FILE:
        skipped.append(str(src))
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".db":
        with sqlite3.connect(f"file:{src}?mode=ro", uri=True) as a, sqlite3.connect(dst) as b:
            a.backup(b)
        b.close()
        a.close()
    else:
        shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path, skipped: list[str], top_skip: set[str] = frozenset()) -> None:
    for p in src.iterdir():
        if p.name in top_skip:
            continue
        if p.is_dir():
            _copy_tree(p, dst / p.name, skipped)
        elif p.is_file():
            _copy_file(p, dst / p.name, skipped)


def take(cfg, reason: str, version: str = "", keep: int | None = KEEP) -> Path | None:
    """Snapshot the data now. Returns its folder (None when there is nothing to save)."""
    found = items(cfg)
    if not found:
        return None
    name = time.strftime("%Y%m%d-%H%M%S") + f"-{reason}" + (f"-{version}" if version else "")
    dest = folder(cfg) / name
    n = 1
    while dest.exists():   # two in the same second
        n += 1
        dest = folder(cfg) / f"{name}-{n}"
    skipped: list[str] = []
    manifest = {"name": dest.name, "created": time.time(), "reason": reason, "version": version, "items": [],
                "skipped": skipped}
    for label, src in found:
        if src.is_dir():
            _copy_tree(src, dest / label, skipped, SKIP_DIRS if label == "workspace" else set())
        else:
            _copy_file(src, dest / label, skipped)
        manifest["items"].append({"label": label, "path": str(src), "dir": src.is_dir()})
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if keep:
        prune(cfg, keep)
    return dest


def listing(cfg) -> list[dict]:
    """Snapshots, newest first (their manifests)."""
    out = []
    root = folder(cfg)
    for d in root.iterdir() if root.is_dir() else []:
        try:
            out.append(json.loads((d / "manifest.json").read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return sorted(out, key=lambda m: m.get("created", 0), reverse=True)


def prune(cfg, keep: int = KEEP) -> None:
    for m in listing(cfg)[keep:]:
        shutil.rmtree(folder(cfg) / m["name"], ignore_errors=True)


def restore(cfg, name: str) -> dict:
    """Put a snapshot's data back. The data as it is now is snapshotted first ("before-restore"), so a
    restore can be undone the same way. Run it with the CLI and dashboard stopped."""
    src = folder(cfg) / name
    try:
        manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise FileNotFoundError(f"no snapshot {name} in {folder(cfg)}") from None
    safety = take(cfg, "before-restore", keep=None)   # pruned after, so the one being restored survives
    for it in manifest["items"]:
        saved, dest = src / it["label"], Path(it["path"])
        if it["dir"]:
            keep = SKIP_DIRS if it["label"] == "workspace" else set()
            if dest.is_dir():   # replace its contents, except the folders a snapshot leaves out
                for p in dest.iterdir():
                    if p.name not in keep:
                        shutil.rmtree(p) if p.is_dir() else p.unlink()
            dest.mkdir(parents=True, exist_ok=True)
            if saved.is_dir():
                _copy_tree(saved, dest, [])
        elif saved.is_file():
            for side in DB_SIDECARS:   # a stale WAL next to a restored database would be replayed into it
                Path(str(dest) + side).unlink(missing_ok=True)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(saved, dest)
    prune(cfg)
    return {"restored": name, "safety": safety.name if safety else None}
