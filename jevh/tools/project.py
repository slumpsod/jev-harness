"""The user's project folder, pre-imported as `project` when a project is open
(CLI --project / /project, or the dashboard's project field).

    project.root                             # absolute path of the project
    project.tree(".", depth=2)               # folder outline (skips .git, node_modules, venvs, builds)
    project.find("where are API routes defined?")   # Jev ranks the files -> a few likely paths
    project.grep(r"def handle_\\w+", "**/*.py")     # -> ["path:line: text", ...]
    project.read("src/app.py", 40, 120)      # numbered lines (at most 400 at a time)
    project.edit("src/app.py", old, new)     # exact text replacement; `old` must be unique
    project.write("notes/todo.md", text)     # create or overwrite a file
    project.sh("pytest -q", timeout=300)     # run a shell command in the project -> {code, out}
    project.changed()                        # files you changed in this task
    project.undo("src/app.py")               # restore a file (or all changed files) as it was

Edits and writes stay inside the project folder; the original of every file you touch is
kept so undo() works. Read before you edit, and run the tests after.
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


def _files(glob: str = "**/*"):
    base = _root()
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


def read(path: str, start: int = 1, end: int | None = None) -> str:
    lines = _path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    end = min(end or len(lines), start + 399, len(lines))
    body = "\n".join(f"{n:>5}  {lines[n - 1]}" for n in range(max(1, start), end + 1))
    more = f"\n[... {len(lines) - end} more lines; read(path, {end + 1}) for the next part]" if end < len(lines) else ""
    return body + more


def _backup(rel: str) -> None:
    if rel in _touched:
        return
    p = _path(rel)
    _touched[rel] = p.exists()
    if p.exists():
        dest = _BACKUPS / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)


def edit(path: str, old: str, new: str, count: int | None = None) -> str:
    p = _path(path)
    text = p.read_text(encoding="utf-8")
    n = text.count(old)
    if n == 0:
        raise ValueError(f"`old` text not found in {path}; read() the file and copy the text exactly")
    if count is None and n > 1:
        raise ValueError(f"`old` appears {n} times in {path}; include more surrounding lines, or pass count=")
    _backup(path)
    p.write_text(text.replace(old, new, count or 1), encoding="utf-8", newline="")
    return f"edited {path} ({min(n, count or 1)} replacement{'s' if (count or 1) > 1 else ''})"


def write(path: str, text: str) -> str:
    p = _path(path)
    _backup(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="")
    return f"wrote {path} ({len(text)} chars)"


def sh(cmd: str, timeout: float = 120) -> dict:
    """Run a shell command with the project as working directory."""
    try:
        r = subprocess.run(cmd, shell=True, cwd=_root(), capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or "") + (e.stderr or "")) if isinstance(e.stdout, str) else ""
        return {"code": None, "out": f"[timed out after {timeout}s]\n{out[-3000:]}"}
    out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
    return {"code": r.returncode, "out": out if len(out) <= 4000 else "[...]\n" + out[-4000:]}


def changed() -> list[str]:
    return sorted(_touched)


def undo(path: str | None = None) -> list[str]:
    done = []
    for rel in [path] if path else list(_touched):
        existed = _touched.pop(rel, None)
        if existed is None:
            continue
        if existed:
            shutil.copy2(_BACKUPS / rel, _path(rel))
        else:
            _path(rel).unlink(missing_ok=True)
        done.append(rel)
    return done


def find(question: str, k: int = 5) -> list[str]:
    """Files most likely relevant to `question`: Jev ranks every file by its path and first
    meaningful line (keyword-prefiltered to 240 on big repos)."""
    from jevh.tools._pick import choose, client_from_env
    files = [rel for rel in _files() if _is_text(rel)]
    if not files:
        return []
    words = {w for w in re.findall(r"[a-z0-9_]{3,}", question.lower())}
    if len(files) > 240:
        def hits(rel):
            try:
                head = _path(rel).read_text(encoding="utf-8", errors="replace")[:4000].lower()
            except OSError:
                head = ""
            return sum(w in rel.lower() for w in words) * 3 + sum(w in head for w in words)
        files = sorted(files, key=hits, reverse=True)[:240]
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
