"""Update the harness to the latest version on GitHub (the checkout's upstream branch).

    python -m jevh.update            # show what's new, then fast-forward to it
    python -m jevh.update --check    # only show what's new

The same is /update in the CLI chat and "Check for updates" in the dashboard's Status tab.
Only a fast-forward is ever done: local commits or edits that would be overwritten stop the
update instead of being lost, and config.local.yaml and workspace/ are ignored by git anyway.
Code already imported keeps running, so restart the CLI or dashboard after an update.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_COMMITS = 30
DEPS_FILES = ("pyproject.toml",)
REINSTALL = 'pip install -e ".[web,browser,dev]"'


class UpdateError(RuntimeError):
    pass


def git(*args: str, root: Path = ROOT, timeout: float = 60) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}   # a private repo fails instead of waiting for a password
    try:
        p = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=timeout, env=env,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except FileNotFoundError:
        raise UpdateError("git is not installed (or not on PATH)") from None
    except subprocess.TimeoutExpired:
        raise UpdateError(f"git {args[0]} timed out after {timeout:.0f}s") from None
    if p.returncode:
        raise UpdateError((p.stderr or p.stdout).strip() or f"git {args[0]} failed ({p.returncode})")
    return p.stdout.strip()


def _commits(spec: str, root: Path) -> list[dict]:
    out = git("log", f"-{MAX_COMMITS}", "--format=%h%x09%cs%x09%s", spec, root=root)
    return [dict(zip(("sha", "date", "subject"), line.split("\t", 2))) for line in out.splitlines() if line]


def check(root: Path | None = None, fetch: bool = True) -> dict:
    """Where this checkout is against its upstream branch. `fetch` asks GitHub first; without it
    the answer is as of the last fetch. Never raises: problems come back in `error`."""
    root = Path(root or ROOT)
    out: dict = {"root": str(root), "branch": None, "upstream": None, "current": None, "latest": None,
                 "behind": 0, "ahead": 0, "incoming": [], "dirty": [], "deps_changed": False,
                 "fetched": False, "can_update": False, "error": None}
    try:
        if not (root / ".git").exists():
            raise UpdateError(f"{root} is not a git checkout: reinstall it with git clone to get updates")
        head = git("log", "-1", "--format=%h%x09%cs%x09%s", root=root).split("\t", 2)
        out["current"] = dict(zip(("sha", "date", "subject"), head))
        branch = git("rev-parse", "--abbrev-ref", "HEAD", root=root)
        if branch == "HEAD":
            raise UpdateError("no branch is checked out (detached HEAD): git checkout main first")
        out["branch"] = branch
        try:
            out["upstream"] = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}", root=root)
        except UpdateError:
            raise UpdateError(f"branch {branch} doesn't track a GitHub branch: git checkout main, "
                              f"or git branch --set-upstream-to origin/{branch}") from None
        out["dirty"] = git("diff", "--name-only", "HEAD", root=root).splitlines()   # edited tracked files
        if fetch:
            remote = git("config", "--get", f"branch.{branch}.remote", root=root) or "origin"
            git("fetch", "--quiet", remote, root=root, timeout=90)
            out["fetched"] = True
        latest = git("log", "-1", "--format=%h%x09%cs%x09%s", "@{upstream}", root=root).split("\t", 2)
        out["latest"] = dict(zip(("sha", "date", "subject"), latest))
        ahead, behind = git("rev-list", "--left-right", "--count", "HEAD...@{upstream}", root=root).split()
        out["ahead"], out["behind"] = int(ahead), int(behind)
        if out["behind"]:
            out["incoming"] = _commits("HEAD..@{upstream}", root)
            changed = git("diff", "--name-only", "HEAD", "@{upstream}", root=root).splitlines()
            out["deps_changed"] = any(f in DEPS_FILES for f in changed)
        if out["behind"] and out["ahead"]:
            raise UpdateError(f"this checkout has {out['ahead']} local commit(s) that aren't on {out['upstream']}: "
                              "merge or rebase them yourself (git pull)")
        out["can_update"] = out["behind"] > 0
    except UpdateError as e:
        out["error"] = str(e)
    return out


def apply(root: Path | None = None) -> dict:
    """Fetch and fast-forward to the upstream branch. Returns the check before it plus `updated`,
    `from`, `to`, `commits` and `files` (how many came in). Raises UpdateError when it can't."""
    root = Path(root or ROOT)
    st = check(root, fetch=True)
    if st["error"]:
        raise UpdateError(st["error"])
    if not st["behind"]:
        return {**st, "updated": False, "from": st["current"]["sha"], "to": st["current"]["sha"], "commits": 0, "files": 0}
    before = git("rev-parse", "HEAD", root=root)
    try:
        git("merge", "--ff-only", "--quiet", "@{upstream}", root=root)
    except UpdateError as e:
        # git refuses when an edited file would be overwritten; nothing has changed then
        raise UpdateError(f"{e}\n(commit or stash your edits first: git stash; then update; then git stash pop)") from None
    files = git("diff", "--name-only", before, "HEAD", root=root).splitlines()
    return {**st, "updated": True, "from": st["current"]["sha"], "to": git("rev-parse", "--short", "HEAD", root=root),
            "commits": st["behind"], "files": len(files), "can_update": False, "behind": 0}


def describe(st: dict) -> str:
    """A few lines for a terminal."""
    if st.get("error") and not st.get("current"):
        return st["error"]
    cur = st["current"]
    lines = [f"version: {cur['sha']} ({cur['date']}) {cur['subject']}" + (f"  [{st['branch']}]" if st["branch"] else "")]
    if st.get("error"):
        lines.append(st["error"])
    elif not st["behind"]:
        lines.append(f"up to date with {st['upstream']}" + ("" if st["fetched"] else " (as of the last fetch)"))
    else:
        lines.append(f"{st['behind']} new commit(s) on {st['upstream']}:")
        lines += [f"  {c['sha']} {c['date']} {c['subject']}" for c in st["incoming"]]
        if st["behind"] > len(st["incoming"]):
            lines.append(f"  ... and {st['behind'] - len(st['incoming'])} more")
        if st["deps_changed"]:
            lines.append(f"dependencies changed: run {REINSTALL} after updating")
    if st["dirty"]:
        lines.append(f"local edits: {', '.join(st['dirty'][:8])}" + (" ..." if len(st["dirty"]) > 8 else ""))
    return "\n".join(lines)


def done(r: dict) -> str:
    if not r["updated"]:
        return "already up to date"
    msg = f"updated {r['from']} -> {r['to']} ({r['commits']} commits, {r['files']} files)."
    if r["deps_changed"]:
        msg += f" Dependencies changed: run {REINSTALL}."
    return msg + " Restart to use the new version."


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="jevh.update", description="Update the harness from GitHub.")
    ap.add_argument("--check", action="store_true", help="only show what's new")
    a = ap.parse_args(argv)
    st = check()
    print(describe(st))
    if a.check or st["error"] or not st["behind"]:
        return 1 if st["error"] else 0
    try:
        print(done(apply()))
    except UpdateError as e:
        print(f"update failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
