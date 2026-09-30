"""The fix: the local model, as a harness worker with the `project` tool, changes a git worktree
of the harness (branch refine/<id>) - never the running one. Then:
- changes to the measuring tools (jevh/refine/, tests/, evals/) are thrown away: a fix can't
  change how it is scored;
- the harness's offline tests must pass (refine.test_command);
- the change is committed on the branch, where the comparison runs it and approval merges it."""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..config import shell_env
from . import checks
from .cases import Case

PROTECTED = ("jevh/refine/", "tests/", "evals/")
AUTHOR = ["-c", "user.name=jev-refine", "-c", "user.email=jev-refine@localhost"]

FIX_PROMPT = """This project is the jev-harness itself: an agent harness where Jev (a fast decision model) \
gates what worker models see. It has a recurring failure. Change the harness so tasks like these succeed.

## The failure ({count} logged like it)
Kind: {kind} ({source}) - {summary}
Persona: {persona}, model: {model}
Request: {request}
The answer it gave: {answer}
{detail}
End of its work log:
{log}
{others}
## Test tasks (the old harness's results). Others like them are held back to check the fix generalizes.
{cases}
{rejected}
## Rules
- Fix the cause in the harness, not these tasks: no special cases for these requests, sites, files or numbers.
- Prefer the smallest change that works: a threshold or limit in config.yaml, the wording of a prompt or of a
  Jev question, a skill in skills/, and only then code in jevh/. Read the code around the problem first
  (project.find / project.grep / project.read); the README explains the design.
- Don't edit jevh/refine/, tests/ or evals/: changes there are thrown away.
- Run the offline tests before you finish: project.sh("{test_command}", timeout=1800). They must pass.
- Finish with final(): one short paragraph - what you changed, and why it fixes the cause."""


@dataclass
class Fix:
    summary: str = ""                     # the fixer's own account
    files: list[str] = field(default_factory=list)
    diff: str = ""
    reverted: list[str] = field(default_factory=list)   # protected files it touched (thrown away)
    tests_ok: bool = False
    tests_out: str = ""
    commit: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.files) and self.tests_ok and not self.error


def git(repo: Path, *args: str, check: bool = True) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {(p.stderr or p.stdout).strip()[-400:]}")
    return p.stdout


def make_tree(repo: Path, dest: Path, branch: str | None, start: str = "HEAD") -> Path:
    """A worktree of `start` (on a new branch, or detached), with this machine's config.local.yaml
    (git ignores it)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if branch:
        git(repo, "worktree", "add", "-B", branch, str(dest), start)
    else:
        git(repo, "worktree", "add", "--detach", str(dest), start)
    local = repo / "config.local.yaml"
    if local.is_file():
        shutil.copy2(local, dest / "config.local.yaml")
    return dest


def reset_tree(tree: Path, base: str) -> None:
    git(tree, "reset", "--hard", base)
    git(tree, "clean", "-fdq", "-e", "config.local.yaml")


def remove_tree(repo: Path, dest: Path, branch: str | None = None) -> None:
    git(repo, "worktree", "remove", "--force", str(dest), check=False)
    shutil.rmtree(dest, ignore_errors=True)
    git(repo, "worktree", "prune", check=False)
    if branch:
        git(repo, "branch", "-D", branch, check=False)


def run_tests(tree: Path, command: str, timeout: float = 1800) -> tuple[bool, str]:
    env = shell_env()
    env["PYTHONPATH"] = str(tree) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        p = subprocess.run(command, shell=True, cwd=tree, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"tests timed out after {timeout:.0f}s"
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode == 0, out[-3000:]


def changed_files(tree: Path) -> list[str]:
    git(tree, "add", "-A")
    return [f for f in git(tree, "diff", "--cached", "--name-only").splitlines()
            if f.strip() and f != "config.local.yaml"]


def prompt(failure: dict, group: list[dict], cases: list[Case], old_rows: list[dict], rejections: list[dict],
           test_command: str, earlier: list[str] | None = None) -> str:
    """The fixer's task. It sees dev cases only - never the held-out ones."""
    lines = []
    for c in cases:
        if c.split != "dev":
            continue
        rows = [r for r in old_rows if r["case"] == c.id]
        res = f"{sum(r['passed'] for r in rows)}/{len(rows)} passed" if rows else "not run yet"
        why = next((r["why"] for r in rows if not r["passed"]), "")
        lines.append(f"- {c.request[:400]}\n  done when: {checks.describe(c.checker)[:200]} | old: {res}"
                     + (f" ({why[:120]})" if why else ""))
    others = [f for f in group if f.get("id") != failure.get("id")][:6]
    rejected = ""
    if rejections or earlier:
        rejected = "\n## Earlier attempts at this that didn't make it\n" + "\n".join(
            [f"- {r.get('change', '')[:200]} -> rejected: {r.get('reason', '')[:200]}" for r in rejections[-5:]]
            + [f"- {e[:300]}" for e in earlier or []]) + "\n"
    return FIX_PROMPT.format(
        count=len(group), kind=failure["kind"], source=failure.get("source", ""), summary=failure["summary"],
        persona=failure.get("persona") or "-", model=failure.get("model") or "-", request=failure["request"][:2500],
        answer=(failure.get("answer") or "(none)")[:1500],
        detail=f"Details: {failure['detail'][:1500]}" if failure.get("detail") else "",
        log=(failure.get("work_log") or "(no code was run)")[-2500:],
        others=("\nOther failures like it:\n" + "\n".join(f"- {f['summary'][:160]} (request: {f['request'][:120]})"
                                                      for f in others) + "\n") if others else "",
        cases="\n".join(lines) or "(none)", rejected=rejected, test_command=test_command)


def fix(tree: Path, task: str, run_turn: Callable[[Path, str], str], test_command: str,
        test_timeout: float = 1800, message: str = "refine: fix") -> Fix:
    """`run_turn(tree, task) -> the worker's answer` runs the harness turn on the tree."""
    out = Fix()
    try:
        out.summary = run_turn(tree, task)
    except Exception as e:  # noqa: BLE001
        out.error = f"the fixer's turn failed: {type(e).__name__}: {e}"
    files = changed_files(tree)
    for f in files:
        if f.startswith(PROTECTED):
            git(tree, "reset", "-q", "--", f)
            if git(tree, "ls-files", "--", f).strip():
                git(tree, "checkout", "--", f)
            else:
                (tree / f).unlink(missing_ok=True)
            out.reverted.append(f)
    out.files = [f for f in changed_files(tree)]
    if not out.files:
        out.error = out.error or "no change was made" + (" (outside the protected paths)" if out.reverted else "")
        return out
    out.diff = git(tree, "diff", "--cached")
    out.tests_ok, out.tests_out = run_tests(tree, test_command, test_timeout)
    if not out.tests_ok:
        out.error = "the offline tests fail with this change"
        return out
    git(tree, *AUTHOR, "commit", "-q", "-m", message + "\n\n" + (out.summary or "")[:2000])
    out.commit = git(tree, "rev-parse", "HEAD").strip()
    return out
