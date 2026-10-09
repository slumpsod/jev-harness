"""The user's project folder, pre-imported as `project` when a project is open
(CLI --project / /project, or the dashboard's project field).

    project.root                             # absolute path of the project
    project.tree(".", depth=2)               # folder outline (skips .git, node_modules, venvs, builds)
    project.find("where are API routes defined?")   # search + Jev -> ["src/api.py:42: ...", ...]
    project.read(project.find(q)[0])         # the lines around a find() hit
    project.find_files(q)                    # Jev ranks whole files instead
    project.grep(r"def handle_\\w+", "**/*.py")     # -> ["path:line: text", ...]
    project.read("src/app.py", 40, 120)      # numbered lines (at most 400 at a time)
    project.edit("src/app.py", old, new)     # exact text replacement; `old` must be unique
    project.write("notes/todo.md", text)     # create or overwrite a file
    project.sh("pytest -q", timeout=300)     # run a shell command in the project -> {code, out}
    project.changed()                        # files you changed in this task
    project.undo("src/app.py")               # restore a file (or all changed files) as it was
    project.overrides()                      # project rules you overrode with rule_ok=

Edits and writes stay inside the project folder; the original of every file you touch is
kept so undo() works. Read before you edit, and run the tests after. Edits that break the
project's own rules (RULES.md / AGENTS.md / CLAUDE.md) are blocked with the rule quoted
(jevh/tools/rules.py); rule_ok="why" lets one through when the rule doesn't fit the case.
"""
from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "env", ".mypy_cache", ".pytest_cache",
             "dist", "build", ".next", ".cache", ".idea", ".vscode", "target", ".tox", "site-packages"}
TEXT_EXT = {".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".md", ".txt", ".html", ".css", ".scss", ".yaml",
            ".yml", ".toml", ".ini", ".cfg", ".sh", ".ps1", ".bat", ".rs", ".go", ".java", ".kt", ".c", ".h",
            ".cpp", ".hpp", ".cs", ".rb", ".php", ".sql", ".vue", ".svelte", ".xml", ".env.example", ".mjs"}
_env_root = os.environ.get("JEVH_PROJECT")
root = str(Path(_env_root).resolve()) if _env_root else None
_BACKUPS = Path(".project_backups")
_touched: dict[str, bool] = {}   # rel path -> existed before


def _root() -> Path:
    if not root:
        raise RuntimeError("No project is open. Ask the user to open one (dashboard project field, "
                           "or /project <path> in the CLI), or call project.open(path).")
    return Path(root)


def open(path: str) -> str:  # noqa: A001 - `project.open(path)` reads naturally
    global root
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise NotADirectoryError(path)
    root = str(p)
    return root


def _path(rel: str) -> Path:
    base = _root()
    p = (base / rel).resolve()
    if p != base and base not in p.parents:
        raise PermissionError(f"{rel} is outside the project folder {base}")
    return p


def _rel(path: str) -> str:
    """The project-relative POSIX path `path` names, however it was written ("src/a.py", "./src/a.py", the absolute
    path): one file must have one backup. Keyed by the spelling, a second spelling copied the already-edited file
    over the original's backup (undo then restored the edit), and an absolute one was "backed up" onto itself."""
    return _path(path).relative_to(_root()).as_posix()


def _git_files(base: Path) -> list[str] | None:
    """The repo's files minus what .gitignore excludes (tracked + untracked), or None if `base`
    isn't a git repo. Without it, find/grep walked generated data - session folders, saved runs."""
    if not (base / ".git").exists():
        return None
    try:
        r = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=base,
                           capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return sorted({p for p in r.stdout.decode("utf-8", "replace").split("\0") if p})


def _files(glob: str = "**/*"):
    base = _root()
    listed = _git_files(base)
    if listed is not None:
        for rel in listed:
            parts = rel.split("/")
            if any(p in SKIP_DIRS or p.startswith(".") for p in parts[:-1]) or not (base / rel).is_file():
                continue
            if fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(parts[-1], glob):
                yield rel
        return
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for f in sorted(filenames):
            rel = (Path(dirpath) / f).relative_to(base).as_posix()
            if fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(f, glob):
                yield rel


def _is_text(rel: str) -> bool:
    return Path(rel).suffix.lower() in TEXT_EXT or Path(rel).name in {"Dockerfile", "Makefile", "README"}


def tree(path: str = ".", depth: int = 2, limit: int = 300) -> str:
    base = _path(path)
    lines: list[str] = []

    def walk(d: Path, level: int):
        if level > depth or len(lines) >= limit:
            return
        entries = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        for e in entries:
            if e.name in SKIP_DIRS or e.name.startswith("."):
                continue
            lines.append("  " * level + e.name + ("/" if e.is_dir() else ""))
            if len(lines) >= limit:
                lines.append("  ... (more)")
                return
            if e.is_dir():
                walk(e, level + 1)

    walk(base, 0)
    return "\n".join(lines)


def grep(pattern: str, glob: str = "**/*", limit: int = 60, ignore_case: bool = False) -> list[str]:
    rx = re.compile(pattern, re.I if ignore_case else 0)
    out = []
    for rel in _files(glob):
        if not _is_text(rel):
            continue
        try:
            text = _path(rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                out.append(f"{rel}:{n}: {line.strip()[:200]}")
                if len(out) >= limit:
                    return out + [f"... (stopped at {limit} matches; narrow the pattern or glob)"]
    return out


LOCATION_RE = re.compile(r"^(.+?):(\d+):")


def read(path: str, start: int = 1, end: int | None = None) -> str:
    """Numbered lines. `path` may be a find() result ("src/app.py:42: ..."): the lines around it."""
    m = LOCATION_RE.match(path)
    if m and not (_root() / path).exists():
        path, line = m.group(1), int(m.group(2))
        if start == 1 and end is None:
            start, end = max(1, line - 10), line + 40
    lines = _path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    end = min(end or len(lines), start + 399, len(lines))
    body = "\n".join(f"{n:>5}  {lines[n - 1]}" for n in range(max(1, start), end + 1))
    more = f"\n[... {len(lines) - end} more lines; read(path, {end + 1}) for the next part]" if end < len(lines) else ""
    return body + more


def _backup(path: str) -> None:
    rel = _rel(path)
    if rel in _touched:
        return
    p = _path(rel)
    _touched[rel] = p.exists()
    if p.exists():
        dest = _BACKUPS / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)


class RuleViolation(PermissionError):
    """An edit the project's rules forbid (see jevh.tools.rules)."""


_overrides: list[dict] = []


def _enforce(rel: str, old: str, new: str, existed: bool, rule_ok: str | None) -> None:
    from jevh.tools import rules
    rel = _path(rel).relative_to(_root()).as_posix()
    v = rules.check(_root(), rel, old, new, existed)
    if v is None:
        return
    if rule_ok:
        _overrides.append({"file": rel, "rule": v.rule.text, "why": rule_ok})
        print(f"[rule override - {rel}: \"{v.rule.text}\" - {rule_ok}]")
        return
    raise RuleViolation(v.message(rel))


def overrides() -> list[dict]:
    """Rules you overrode with rule_ok= in this task (say so in your final answer)."""
    return list(_overrides)


def edit(path: str, old: str, new: str, count: int | None = None, rule_ok: str | None = None) -> str:
    p = _path(path)
    text = p.read_text(encoding="utf-8")
    n = text.count(old)
    if n == 0:
        raise ValueError(f"`old` text not found in {path}; read() the file and copy the text exactly")
    if count is None and n > 1:
        raise ValueError(f"`old` appears {n} times in {path}; include more surrounding lines, or pass count=")
    _enforce(path, old, new, True, rule_ok)
    _backup(path)
    p.write_text(text.replace(old, new, count or 1), encoding="utf-8", newline="")
    return f"edited {path} ({min(n, count or 1)} replacement{'s' if (count or 1) > 1 else ''})"


def write(path: str, text: str, rule_ok: str | None = None) -> str:
    p = _path(path)
    existed = p.exists()
    before = p.read_text(encoding="utf-8", errors="replace") if existed else ""
    _enforce(path, before, text, existed, rule_ok)
    _backup(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="")
    return f"wrote {path} ({len(text)} chars)"


def sh(cmd: str, timeout: float = 120) -> dict:
    """Run a shell command with the project as working directory. Its stdin is empty: a command that asks a
    question gets an end of file instead of waiting."""
    from jevh import procs   # (kills what the command started on a timeout: plain subprocess.run waited for it)
    try:
        r = procs.run(cmd, shell=True, cwd=_root(), timeout=timeout)
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or "") + (e.stderr or "")) if isinstance(e.stdout, str) else ""
        return {"code": None, "out": f"[timed out after {timeout}s]\n{out[-3000:]}"}
    out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
    return {"code": r.returncode, "out": out if len(out) <= 4000 else "[...]\n" + out[-4000:]}


def changed() -> list[str]:
    return sorted(_touched)


def undo(path: str | None = None) -> list[str]:
    done = []
    for rel in [_rel(path)] if path else list(_touched):
        existed = _touched.pop(rel, None)
        if existed is None:
            continue
        if existed:
            shutil.copy2(_BACKUPS / rel, _path(rel))
        else:
            _path(rel).unlink(missing_ok=True)
        done.append(rel)
    return done


STOP_WORDS = {"where", "what", "which", "how", "does", "the", "and", "for", "that", "this", "with", "from", "into",
               "when", "who", "why", "are", "was", "were", "has", "have", "can", "get", "set", "its", "our", "your",
               "code", "file", "files", "function", "method", "class", "defined", "define", "find", "done", "made",
               "make", "handle", "handled", "handles", "used", "use", "uses", "there", "then", "than", "them", "they",
               "about", "out", "whether", "decide", "decides", "decided", "work", "works", "worked", "happen"}
DEF_RE = re.compile(r"^\s*(?:async\s+def|def|class|function|export|const|let|var|pub\s+fn|fn|func|interface|type)\b"
                    r"|^\s*[A-Z][A-Z0-9_]{2,}\s*=|^\s*@\w+\.(?:get|post|put|delete|route)\(")
FIND_CANDIDATES = 48    # snippets Jev ranks per find() (at most 3 Choice rounds + a final on local semif)
FIND_Q = "Which code snippet is where `question` is answered?"
FIND_FILES = 48         # files Jev ranks in find_files() after the keyword prefilter


def _keywords(question: str) -> set[str]:
    """The question's words worth searching for: no stop words; plurals and the parts of
    snake_case / camelCase names too ("API routes" -> api, routes, route)."""
    out = set()
    for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", question):
        parts = [w] + re.findall(r"[a-z]{3,}|[A-Z][a-z]{2,}", w)
        for p in parts:
            lw = p.lower().strip("_")
            if len(lw) < 3 or lw in STOP_WORDS:
                continue
            out.add(lw)
            if lw.endswith("es") and len(lw) > 5:
                out.add(lw[:-2])
            if lw.endswith("s") and len(lw) > 4:
                out.add(lw[:-1])
    return out


def _candidates(question: str, n: int = FIND_CANDIDATES) -> list[tuple[str, int, str, str]]:
    """(path, line number, the line, snippet around it) for the lines sharing the most keywords with `question`: plain
    search, best line per 6-line window, definitions and matching paths first."""
    words = _keywords(question)
    if not words:
        return []
    scored = []
    for rel in _files():
        if not _is_text(rel):
            continue
        try:
            lines = _path(rel).read_text(encoding="utf-8", errors="replace").splitlines()[:8000]
        except OSError:
            continue
        in_path = sum(w in rel.lower() for w in words)
        best: dict[int, tuple[float, int]] = {}
        for i, line in enumerate(lines):
            low = line.lower()
            hit = sum(w in low for w in words)
            if not hit:
                continue
            score = 2 * hit + in_path + (1.5 if DEF_RE.match(line) else 0)
            window = i // 6
            if window not in best or score > best[window][0]:
                best[window] = (score, i)
        for score, i in best.values():
            snippet = " | ".join(ln.strip() for ln in lines[max(0, i - 1):i + 3] if ln.strip())
            scored.append((score, rel, i + 1, lines[i].strip()[:150], snippet[:300]))
    scored.sort(key=lambda x: -x[0])
    return [c[1:] for c in scored[:n]]


def _in_files(question: str, files: list[str], n: int = FIND_CANDIDATES) -> list[tuple[str, int, str, str]]:
    """(path, line number, the line, snippet) inside `files`: their definitions (def, class,
    constant, route), each shown with its docstring and scored on the question's keywords
    anywhere in its body. Only definitions: offered keyword lines too, Jev picked docstrings,
    comments and messages that restate the question over the code that does it. A file with no
    definitions (docs, config) offers its keyword-matching lines instead."""
    words = _keywords(question)
    scored = []
    for rank, rel in enumerate(files):
        try:
            lines = _path(rel).read_text(encoding="utf-8", errors="replace").splitlines()[:8000]
        except OSError:
            continue
        heads = [i for i, ln in enumerate(lines) if DEF_RE.match(ln)]
        if not heads:
            for i, line in enumerate(lines):
                hit = sum(w in line.lower() for w in words)
                if hit:
                    snippet = " | ".join(ln.strip() for ln in lines[max(0, i - 1):i + 2] if ln.strip())
                    scored.append((2 * hit - 0.5 * rank, rel, i + 1, line.strip()[:150], snippet[:300]))
            continue
        for h, i in enumerate(heads):
            body = lines[i:min(heads[h + 1] if h + 1 < len(heads) else len(lines), i + 80)]
            low = "\n".join(body).lower()
            found = sum(w in low for w in words)
            named = sum(w in lines[i].lower() for w in words)
            doc = next((ln.strip().strip('"\'#/* ') for ln in body[1:5] if ln.strip().strip('"\'#/* ')), "")
            snippet = lines[i].strip() + (f" | {doc}" if doc else "")
            scored.append((2 * found + 2 * named - 0.5 * rank, rel, i + 1, lines[i].strip()[:150], snippet[:300]))
    scored.sort(key=lambda x: -x[0])
    return [c[1:] for c in scored[:n]]


def find(question: str, k: int = 5, files: int = 3) -> list[str]:
    """Where in the project `question` is answered -> ["path:line: text", ...], best first;
    project.read() takes one of these directly. Two steps: Jev picks the `files` most relevant
    files (find_files), then ranks the definitions and keyword-matching lines inside them.
    (evals/find_cases.py: one Jev pick over 48 keyword-matched lines from anywhere put the
    right line first 9/16 on the tuning questions but 2/8 held out - it was fooled by
    look-alike lines in other files, where picking the file first is steadier.)"""
    from jevh.tools._pick import choose, client_from_env
    top = find_files(question, files)
    cands = _in_files(question, top)
    if not cands:
        return [f"{rel}:1:" for rel in top[:k]]
    opts = {f"s{i}": f"{rel}:{n}: {snip}" for i, (rel, n, _, snip) in enumerate(cands)}
    jev, cap = client_from_env()
    probs, _ = choose(jev, {"question": question}, FIND_Q, opts, cap)
    ranked = sorted((key for key in probs if probs[key] > 0), key=probs.get, reverse=True)[:k]
    return [f"{rel}:{n}: {line}" for rel, n, line, _ in (cands[int(key[1:])] for key in ranked)]


def find_files(question: str, k: int = 5, prefilter: int = FIND_FILES) -> list[str]:
    """Files most likely relevant to `question`: Jev ranks files by path and first meaningful
    line, after a keyword prefilter to `prefilter` files (the ones mentioning the most of the
    question's keywords, in path or content)."""
    from jevh.tools._pick import choose, client_from_env
    files = [rel for rel in _files() if _is_text(rel)]
    if not files:
        return []
    words = _keywords(question)
    if len(files) > prefilter and words:
        def hits(rel):
            try:
                text = _path(rel).read_text(encoding="utf-8", errors="replace")[:200_000].lower()
            except OSError:
                text = ""
            return sum(w in rel.lower() for w in words) * 3 + sum(w in text for w in words)
        files = sorted(files, key=hits, reverse=True)[:prefilter]
    opts = {f"f{i}": f"{rel}: {_head(rel)}" for i, rel in enumerate(files)}
    jev, cap = client_from_env()
    probs, _ = choose(jev, {"question": question}, "Which file is most relevant to `question`?", opts, cap)
    return [files[int(key[1:])] for key in sorted(probs, key=probs.get, reverse=True)[:k] if probs[key] > 0]


def _head(rel: str) -> str:
    """First meaningful line: a docstring, comment or heading."""
    try:
        with _path(rel).open(encoding="utf-8", errors="replace") as f:
            for _, line in zip(range(40), f):
                s = line.strip().strip('"\'#/*-<>! ').strip()
                if len(s) > 12 and not s.startswith(("import ", "from ", "use ", "package ", "require(")):
                    return s[:90]
    except OSError:
        pass
    return ""
