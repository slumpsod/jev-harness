"""Update the harness to the latest version on GitHub (the checkout's upstream branch).

    jevh update            # show what's new, then update to it (or python -m jevh.update)
    jevh update --check    # only show what's new
    jevh update --test     # also run the test suite on the new version; undo the update if it fails
    jevh update --undo     # go back to the version before the last update

The same is /update in the CLI chat and the Harness version card in the dashboard's Status tab.

An update either works or leaves things exactly as they were:
- Your edits to the harness's files are saved (git stash), carried onto the new version and put back.
  Before anything moves, a dry-run merge checks they apply cleanly; if one clashes with the update,
  the update stops and nothing changes. A copy also stays at refs/jevh/pre-update-edits.
- A file you added that the update would overwrite stops it too. config.local.yaml and workspace/
  are ignored by git, so they're never touched.
- If pyproject.toml changed, the new dependencies are installed (with the extras you already have).
- The new version must import (and with --test, pass the tests). If installing or that check fails,
  the update is rolled back: the old version, your edits as they were.
- The version you came from is kept (refs/jevh/pre-update) for --undo.
Local commits that aren't on GitHub stop the update (merge them yourself with git pull).
Code already imported keeps running, so restart the CLI or dashboard after an update.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_COMMITS = 30
DEPS_FILES = ("pyproject.toml",)
REINSTALL = 'pip install -e ".[web,browser,dev]"'
EXTRAS = {"web": "fastapi", "browser": "playwright", "dev": "pytest", "claude": "anthropic"}  # extra -> a module it installs
SMOKE = ["jevh.cli", "jevh.orchestrator", "jevh.update"]   # must import after an update (+ the dashboard if installed)
BACKUP_REF, LANDED_REF, EDITS_REF = "refs/jevh/pre-update", "refs/jevh/post-update", "refs/jevh/pre-update-edits"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class UpdateError(RuntimeError):
    pass


def run(cmd: list[str], root: Path, timeout: float) -> subprocess.CompletedProcess:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}   # a private repo fails instead of waiting for a password
    try:
        return subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=timeout, env=env,
                              creationflags=_NO_WINDOW)
    except FileNotFoundError:
        raise UpdateError(f"{cmd[0]} is not installed (or not on PATH)") from None
    except subprocess.TimeoutExpired:
        raise UpdateError(f"{' '.join(cmd[:3])} timed out after {timeout:.0f}s") from None


def git(*args: str, root: Path = ROOT, timeout: float = 60) -> str:
    p = run(["git", *args], root, timeout)
    if p.returncode:
        raise UpdateError((p.stderr or p.stdout).strip() or f"git {args[0]} failed ({p.returncode})")
    return p.stdout.strip()


def _ref(name: str, root: Path) -> str | None:
    p = run(["git", "rev-parse", "--verify", "--quiet", name + "^{commit}"], root, 30)
    return p.stdout.strip() or None


def _info(rev: str, root: Path) -> dict:
    return dict(zip(("sha", "date", "subject"), git("log", "-1", "--format=%h%x09%cs%x09%s", rev, root=root).split("\t", 2)))


def _commits(spec: str, root: Path) -> list[dict]:
    out = git("log", f"-{MAX_COMMITS}", "--format=%h%x09%cs%x09%s", spec, root=root)
    return [dict(zip(("sha", "date", "subject"), line.split("\t", 2))) for line in out.splitlines() if line]


def _tail(text: str, n: int = 12) -> str:
    return "\n".join(text.strip().splitlines()[-n:])


def obstacles(target: str, root: Path) -> tuple[list[str], list[str]]:
    """What would stop moving this checkout to `target`: (edited files whose edits clash with the
    change, untracked files it would overwrite). Touches nothing."""
    changed = git("diff", "--name-only", "--no-renames", "HEAD", target, root=root).splitlines()
    added = git("diff", "--name-only", "--no-renames", "--diff-filter=A", "HEAD", target, root=root).splitlines()
    in_way = sorted(f for f in added if (root / f).exists())
    dirty = git("diff", "--name-only", "HEAD", root=root).splitlines()
    overlap = sorted(set(dirty) & set(changed))
    if not overlap:
        return [], in_way
    # both sides changed these files: a dry-run merge of the edits onto the target says whether they still apply
    edits = git("stash", "create", root=root)
    p = run(["git", "merge-tree", "--write-tree", "--name-only", "--no-messages", "--merge-base", "HEAD",
             target, edits], root, 60)
    if p.returncode == 0:
        return [], in_way
    if p.returncode == 1:
        return sorted(set(p.stdout.split("\n")[1:]) - {""}), in_way
    return overlap, in_way   # git older than 2.40: no dry run, so any file both sides changed counts


def _pip(root: Path) -> str | None:
    """Install the checkout's dependencies, with the extras already installed. None when done, or why
    it was skipped; raises UpdateError when pip fails."""
    extras = [x for x, mod in EXTRAS.items() if importlib.util.find_spec(mod)]
    spec = f"{root}[{','.join(extras)}]" if extras else str(root)
    p = run([sys.executable, "-m", "pip", "install", "--quiet", "-e", spec], root, 1800)
    if p.returncode and "No module named pip" in p.stderr:
        return f"no pip in {sys.executable}: reinstall the harness yourself (e.g. pipx install --force -e .)"
    if p.returncode:
        raise UpdateError(f"installing the new dependencies failed:\n{_tail(p.stderr or p.stdout)}")
    return None


def _verify(root: Path, test: bool) -> None:
    mods = SMOKE + (["jevh.web.server"] if SMOKE and importlib.util.find_spec("fastapi") else [])
    if mods:
        p = run([sys.executable, "-c", "import " + ", ".join(mods)], root, 180)
        if p.returncode:
            raise UpdateError(f"the new version doesn't start:\n{_tail(p.stderr)}")
    if test:
        p = run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"], root, 3600)
        if p.returncode:
            raise UpdateError(f"the tests fail on the new version:\n{_tail(p.stdout + p.stderr)}")


def _move(target: str, root: Path, test: bool = False) -> dict:
    """Move this branch to `target` carrying the local edits, or leave everything as it was."""
    before = git("rev-parse", "HEAD", root=root)
    target = git("rev-parse", target, root=root)
    if test and not importlib.util.find_spec("pytest"):
        raise UpdateError(f"running the tests needs pytest: {REINSTALL}")
    conflicts, in_way = obstacles(target, root)
    if conflicts:
        raise UpdateError(f"your edits to {', '.join(conflicts)} clash with the update. Nothing was changed. "
                          "Save a copy of them, git checkout -- <file> to drop them, update, then redo them.")
    if in_way:
        raise UpdateError(f"the update adds {', '.join(in_way)}, which you already have as your own file(s). "
                          "Nothing was changed. Rename or move them, then update.")
    files = git("diff", "--name-only", before, target, root=root).splitlines()
    deps = any(f in DEPS_FILES for f in files)
    dirty = git("diff", "--name-only", "HEAD", root=root).splitlines()
    saved = None
    if dirty:
        git("stash", "push", "--quiet", "-m", f"jevh update: your edits on {before[:7]}", root=root)
        saved = git("rev-parse", "stash@{0}", root=root)
        git("update-ref", EDITS_REF, saved, root=root)
    note, pipped = None, False
    try:
        git("reset", "--hard", "--quiet", target, root=root)
        if saved:
            git("stash", "apply", "--quiet", saved, root=root)
        if deps:
            pipped = True
            note = _pip(root)
        _verify(root, test)
    except BaseException as e:
        rollback = _restore(before, saved, root, pipped)
        if isinstance(e, UpdateError):
            raise UpdateError(f"{e}\n{rollback}") from None
        raise
    if saved:
        _drop(saved, root)
    short = lambda sha: git("rev-parse", "--short", sha, root=root)   # same form as %h in _info
    return {"from": short(before), "to": short(target), "files": len(files), "kept": dirty, "deps_changed": deps,
            "deps_note": note}


def _restore(before: str, saved: str | None, root: Path, pipped: bool) -> str:
    """Back to `before` with the saved edits, as they were. Returns what happened, for the message."""
    try:
        git("reset", "--hard", "--quiet", before, root=root)
        if saved:
            try:
                git("stash", "apply", "--quiet", "--index", saved, root=root)
            except UpdateError:
                git("stash", "apply", "--quiet", saved, root=root)
            _drop(saved, root)
    except UpdateError as e:
        return (f"Rolling back failed too ({e}). Your previous version is {before[:7]} (git reset --hard {before[:7]})"
                + (f" and your edits are saved in {EDITS_REF} (git stash apply {EDITS_REF})." if saved else "."))
    if pipped:
        try:
            _pip(root)
        except UpdateError:
            return f"Rolled back to {before[:7]}, but reinstalling its dependencies failed: run {REINSTALL}."
    return f"Rolled back: you're on {before[:7]} again" + (" with your edits as they were." if saved else ".")


def _drop(saved: str, root: Path) -> None:
    """Drop our stash entry (a copy stays at EDITS_REF)."""
    for i, sha in enumerate(git("stash", "list", "--format=%H", root=root).splitlines()):
        if sha == saved:
            git("stash", "drop", "--quiet", f"stash@{{{i}}}", root=root)
            return


def check(root: Path | None = None, fetch: bool = True) -> dict:
    """Where this checkout is against its upstream branch. `fetch` asks GitHub first; without it
    the answer is as of the last fetch. Never raises: problems come back in `error`."""
    root = Path(root or ROOT)
    out: dict = {"root": str(root), "branch": None, "upstream": None, "current": None, "latest": None,
                 "behind": 0, "ahead": 0, "incoming": [], "dirty": [], "conflicts": [], "in_way": [],
                 "deps_changed": False, "fetched": False, "can_update": False, "previous": None, "error": None}
    try:
        if not (root / ".git").exists():
            raise UpdateError(f"{root} is not a git checkout: reinstall it with git clone to get updates")
        out["current"] = _info("HEAD", root)
        head = git("rev-parse", "HEAD", root=root)
        if (prev := _ref(BACKUP_REF, root)) and _ref(LANDED_REF, root) == head:
            out["previous"] = _info(prev, root)   # the last update can be undone
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
        out["latest"] = _info("@{upstream}", root)
        ahead, behind = git("rev-list", "--left-right", "--count", "HEAD...@{upstream}", root=root).split()
        out["ahead"], out["behind"] = int(ahead), int(behind)
        if out["behind"] and out["ahead"]:
            raise UpdateError(f"this checkout has {out['ahead']} local commit(s) that aren't on {out['upstream']}: "
                              "merge them yourself (git pull)")
        if out["behind"]:
            out["incoming"] = _commits("HEAD..@{upstream}", root)
            changed = git("diff", "--name-only", "HEAD", "@{upstream}", root=root).splitlines()
            out["deps_changed"] = any(f in DEPS_FILES for f in changed)
            out["conflicts"], out["in_way"] = obstacles("@{upstream}", root)
        out["can_update"] = out["behind"] > 0 and not out["conflicts"] and not out["in_way"]
    except UpdateError as e:
        out["error"] = str(e)
    return out


def apply(root: Path | None = None, test: bool = False) -> dict:
    """Fetch and move to the upstream branch, keeping local edits (see the module doc). Returns the
    check before it plus `updated`, `from`, `to`, `commits`, `files`, `kept` (the edited files carried
    over) and `deps_note`. Raises UpdateError when it can't, with everything as it was."""
    root = Path(root or ROOT)
    st = check(root, fetch=True)
    if st["error"]:
        raise UpdateError(st["error"])
    if not st["behind"]:
        sha = st["current"]["sha"]
        return {**st, "updated": False, "from": sha, "to": sha, "commits": 0, "files": 0, "kept": [], "deps_note": None}
    before = git("rev-parse", "HEAD", root=root)
    r = _move("@{upstream}", root, test)
    git("update-ref", BACKUP_REF, before, root=root)
    git("update-ref", LANDED_REF, "HEAD", root=root)
    return {**st, **r, "updated": True, "commits": st["behind"], "can_update": False, "behind": 0,
            "previous": _info(before, root)}


def undo(root: Path | None = None) -> dict:
    """Go back to the version before the last update, keeping local edits."""
    root = Path(root or ROOT)
    st = check(root, fetch=False)
    if not st["previous"]:
        raise UpdateError("nothing to undo: no update was made here, or the code moved on since")
    r = _move(BACKUP_REF, root)
    for ref in (BACKUP_REF, LANDED_REF):
        git("update-ref", "-d", ref, root=root)
    return {**r, "undone": True}


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
            lines.append("dependencies changed: they'll be installed with the update")
    if st["dirty"]:
        lines.append(f"your edits ({', '.join(st['dirty'][:8])}{' ...' if len(st['dirty']) > 8 else ''}) "
                     + ("are kept across the update" if st["behind"] and not st["conflicts"] else "stay as they are"))
    if st["conflicts"]:
        lines.append(f"can't update: your edits to {', '.join(st['conflicts'])} clash with it")
    if st["in_way"]:
        lines.append(f"can't update: it adds {', '.join(st['in_way'])}, which you already have as your own file(s)")
    if st["previous"]:
        lines.append(f"before the last update: {st['previous']['sha']} (jevh update --undo goes back)")
    return "\n".join(lines)


def done(r: dict) -> str:
    if r.get("undone"):
        msg = f"back on {r['to']} (from {r['from']})."
    elif not r["updated"]:
        return "already up to date"
    else:
        msg = f"updated {r['from']} -> {r['to']} ({r['commits']} commits, {r['files']} files)."
    if r["kept"]:
        msg += f" Your edits to {len(r['kept'])} file(s) were kept."
    if r["deps_changed"]:
        msg += f" {r['deps_note']}." if r["deps_note"] else " New dependencies installed."
    return msg + " Restart to use it."


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="jevh update", description="Update the harness from GitHub, keeping your edits.")
    ap.add_argument("--check", action="store_true", help="only show what's new")
    ap.add_argument("--test", action="store_true", help="run the test suite on the new version; undo the update if it fails")
    ap.add_argument("--undo", action="store_true", help="go back to the version before the last update")
    a = ap.parse_args(argv)
    try:
        if a.undo:
            print(done(undo()))
            return 0
        st = check()
        print(describe(st))
        if a.check or st["error"] or not st["can_update"]:
            return 1 if st["error"] or st["behind"] else 0
        if a.test:
            print("updating, then running the tests (a few minutes)...")
        print(done(apply(test=a.test)))
    except UpdateError as e:
        print(f"{'undo' if a.undo else 'update'} failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
