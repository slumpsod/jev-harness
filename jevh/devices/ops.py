"""What a device does when the hub asks: files, a shell, a screenshot, and the browser and desktop tools, run here.

No networking: `Ops.run(op, args, kwargs, call)` is given a `Call` (its cancel flag, and `host()` to ask the harness), so
the agent and the tests drive it the same way.

Guards (the user chose "full access, keep the confirmations"; like every tool here, these catch mistakes, they are not a
sandbox - a shell can do anything):
- files.delete moves to <device home>/trash, files.write / files.put over a file keep the old one in <device home>/backups:
  both can be undone, so neither asks.
- A shell command that destroys data or stops the machine (RISKY below) returns needs_confirmation until it is repeated
  with allow_irreversible=True, which the worker passes after the user says yes.
- The browser and desktop tools keep their own confirmations (orders, sends, deletes, passwords).
- The device's user can pause it (tray icon): every call is then refused.
"""
from __future__ import annotations

import fnmatch
import inspect
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import wire

MAX_READ = 200_000        # characters files.read returns at most
MAX_OUT = 20_000          # characters of a shell command's output
MAX_LIST = 1000
FIND_SECONDS = 30.0
BROWSER_IDLE = 15 * 60    # an idle browser on a device is closed after this long

# commands that destroy data or stop the machine: asked about first (cmd, PowerShell and POSIX spellings)
RISKY = re.compile(r"""(?ix)
    \brm\s+(-\w*\s+)*-\w*[rf] | \brm\s+-- | \brmdir\b | \brd\s+/s | \bdel\b | \berase\b | \bformat\s+[a-z]: |
    \bremove-item\b | \bri\s | \bclear-content\b | \bclear-disk\b | \bformat-volume\b | \bdiskpart\b |
    \bcipher\s+/w | \bshutdown\b | \brestart-computer\b | \bstop-computer\b | \blogoff\b |
    \breg(\.exe)?\s+delete\b | \bremove-itemproperty\b | \bbcdedit\b | \bvssadmin\s+delete\b | \bwmic\b.*\bdelete\b |
    \bgit\s+(reset\s+--hard|clean\s+-\w*f|push\s+(.*\s)?(-f|--force)) | \bmkfs | \bdd\s+if= |
    \btaskkill\b | \bstop-process\b | \bkill\s+-9 | \bpkill\b | \bkillall\b |
    \bset-executionpolicy\b | \bnetsh\s+advfirewall\b | \bremove-appxpackage\b | \bwinget\s+uninstall\b |
    \bmove-item\b.*-force | >\s*[a-z]:\\windows
""")

FILE_OPS = ("list", "stat", "read", "find", "write", "fetch", "put", "move", "copy", "mkdir", "delete", "open")
TOOL_SKIP = {"session", "goal_values", "named_gone", "final_click", "nav_target", "norm_text", "label_of", "kind_of",
             "region_word", "build_elements", "describe", "window_line", "signature", "corrected_url", "dropped_search",
             "bot_check_exempt", "unresolved_hint", "click_label", "field_matches", "find_chrome", "close_kept",
             "chrome_args", "off_goal_link"}


class Refused(RuntimeError):
    """The device will not do this (paused, unknown op): said to the worker as is."""


@dataclass
class Call:
    """One call in progress on this device."""
    id: str
    op: str
    host: Callable[..., Any] = lambda *a, **k: None
    cancelled: threading.Event = field(default_factory=threading.Event)
    proc: subprocess.Popen | None = None


def device_home() -> Path:
    """This device's own folder: its key, trash, backups and log. Not ~/.jevh: the harness uses that folder for a user's
    personas and skills once it exists, so a device must not create it."""
    return Path(os.path.expandvars(os.path.expanduser(os.environ.get("JEVH_DEVICE_HOME") or "~/.jevh-device")))


def _path(p: str | os.PathLike, must_exist: bool = False) -> Path:
    """A path the worker gave: ~ and %VARS% expanded, relative to the user's home folder."""
    q = Path(os.path.expandvars(os.path.expanduser(str(p).strip().strip('"'))))
    if not q.is_absolute():
        q = Path.home() / q
    if must_exist and not q.exists():
        raise FileNotFoundError(f"{q} does not exist on this device")
    return q


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _entry(p: Path) -> dict:
    try:
        st = p.stat()
        return {"name": p.name, "path": str(p), "dir": p.is_dir(), "bytes": None if p.is_dir() else st.st_size,
                "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))}
    except OSError as e:
        return {"name": p.name, "path": str(p), "error": type(e).__name__}


def _clip(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    half = n // 2
    return f"{text[:half]}\n... [{len(text) - n} characters cut] ...\n{text[-half:]}"


class Ops:
    def __init__(self, home: Path | None = None):
        self.home = home or device_home()
        self.artifacts = self.home / "artifacts"
        self.paused = False
        self.ui_lock = threading.Lock()   # the browser, the desktop, the shell and screenshots: one at a time
        self.current: Call | None = None  # the call holding ui_lock: a tool's ask_user is routed to it
        self._local = threading.local()
        self._tools: dict[str, Any] = {}
        self._used: dict[str, float] = {}

    # ------------------------------------------------------------------ dispatch
    def run(self, op: str, args: list | None = None, kwargs: dict | None = None, call: Call | None = None):
        if self.paused:
            raise Refused("paused on this device by its user: the assistant may not use it until they resume it")
        call = call or Call("local", op)
        args, kwargs = list(args or []), dict(kwargs or {})
        group, _, name = op.partition(".")
        if op == "info":
            return self.info()
        if group == "files" and name in FILE_OPS:
            return getattr(self, f"files_{name}")(*args, **kwargs)
        if group in ("sh", "screenshot", "browser", "desktop"):
            with self.ui_lock:
                self.current, self._local.call = call, call
                try:
                    if op == "sh":
                        return self.sh(*args, call=call, **kwargs)
                    if op == "screenshot":
                        return self.screenshot(*args, **kwargs)
                    if name:
                        return self.tool(group, name, args, kwargs)
                finally:
                    self.current, self._local.call = None, None
        raise Refused(f"unknown operation {op!r}")

    def call_now(self) -> Call | None:
        return getattr(self._local, "call", None) or self.current

    # ------------------------------------------------------------------ about this device
    def info(self) -> dict:
        drives = []
        if os.name == "nt":
            drives = [f"{d}:\\" for d in "CDEFGHIJKLMNOPQRSTUVWXYZ" if os.path.exists(f"{d}:\\")]
        return {"host": platform.node(), "platform": f"{platform.system()} {platform.release()}",
                "user": os.environ.get("USERNAME") or os.environ.get("USER") or "", "home": str(Path.home()),
                "drives": drives, "paused": self.paused, "tools": self.tools_available(),
                "folders": {k: str(Path.home() / k) for k in ("Desktop", "Documents", "Downloads", "Pictures")
                            if (Path.home() / k).is_dir()}}

    @staticmethod
    def tools_available() -> dict[str, str]:
        """{tool: "ok" or why not}: what the worker can use here."""
        import importlib.util as iu
        out = {"files": "ok", "sh": "ok"}
        out["browser"] = "ok" if iu.find_spec("playwright") else "not installed (pip install playwright; " \
                                                                  "python -m playwright install chromium)"
        if os.name != "nt":
            out["desktop"] = "Windows only"
        else:
            out["desktop"] = "ok" if iu.find_spec("comtypes") else "not installed (pip install comtypes pillow)"
        out["screenshot"] = "ok" if iu.find_spec("PIL") else "not installed (pip install pillow)"
        return out

    # ------------------------------------------------------------------ files
    def files_list(self, path: str = "~", pattern: str = "*", recursive: bool = False, limit: int = 200) -> dict:
        root = _path(path, must_exist=True)
        if root.is_file():
            return {"path": str(root), "entries": [_entry(root)]}
        it = root.rglob(pattern) if recursive else root.glob(pattern)
        entries, more = [], False
        for p in it:
            if len(entries) >= min(int(limit), MAX_LIST):
                more = True
                break
            entries.append(_entry(p))
        entries.sort(key=lambda e: (not e.get("dir"), e["name"].lower()))
        return {"path": str(root), "entries": entries, "more": more}

    def files_stat(self, path: str) -> dict:
        return _entry(_path(path, must_exist=True))

    def files_read(self, path: str, start: int | None = None, end: int | None = None) -> str:
        p = _path(path, must_exist=True)
        if p.is_dir():
            raise IsADirectoryError(f"{p} is a folder: use files.list")
        raw = p.read_bytes()[: MAX_READ * 4]
        if b"\0" in raw[:8192]:
            raise ValueError(f"{p.name} is not text ({p.stat().st_size} bytes): use files.fetch to bring it here")
        text = raw.decode("utf-8", errors="replace")
        if start is not None or end is not None:
            lines = text.splitlines()
            a = max(int(start or 1), 1)
            b = min(int(end or len(lines)), len(lines))
            return "\n".join(f"{i}: {lines[i - 1]}" for i in range(a, b + 1))
        return _clip(text, MAX_READ)

    def files_find(self, name: str = "*", root: str = "~", contains: str | None = None, limit: int = 50) -> dict:
        """Files whose name matches `name` (a glob, or plain words: every word in the name) under `root`, newest first;
        `contains`: only text files holding that text. Stops after FIND_SECONDS."""
        base = _path(root, must_exist=True)
        words = [] if any(c in name for c in "*?[") else name.lower().split()
        started, hits, looked = time.time(), [], 0
        skip = {"node_modules", ".git", "AppData", "$Recycle.Bin", "Windows", "__pycache__", ".venv", "venv"}
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in skip and not d.startswith(".")]
            for f in filenames:
                looked += 1
                low = f.lower()
                if words and not all(w in low for w in words):
                    continue
                if not words and not fnmatch.fnmatch(low, name.lower()):
                    continue
                p = Path(dirpath) / f
                if contains:
                    try:
                        if p.stat().st_size > 5_000_000 or contains.lower() not in p.read_text(
                                encoding="utf-8", errors="ignore").lower():
                            continue
                    except OSError:
                        continue
                hits.append(_entry(p))
            if time.time() - started > FIND_SECONDS or len(hits) >= 5 * limit:
                break
        hits.sort(key=lambda e: e.get("modified", ""), reverse=True)
        return {"root": str(base), "hits": hits[: int(limit)], "looked_at": looked,
                "stopped_early": time.time() - started > FIND_SECONDS}

    def _keep(self, p: Path, where: str) -> str | None:
        """Move or copy `p` aside (backups/ or trash/) so a change can be undone; -> where it went."""
        if not p.exists():
            return None
        dest = self.home / where / _stamp() / (p.drive.rstrip(":") or "root") / p.relative_to(p.anchor)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if where == "trash":
            shutil.move(str(p), str(dest))
        elif p.is_dir():
            shutil.copytree(p, dest)
        else:
            shutil.copy2(p, dest)
        return str(dest)

    def files_write(self, path: str, text: str, append: bool = False) -> dict:
        p = _path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        backup = self._keep(p, "backups") if p.exists() and not append else None
        with open(p, "a" if append else "w", encoding="utf-8", newline="") as f:
            f.write(text)
        return {"path": str(p), "bytes": p.stat().st_size, "backup": backup}

    def files_fetch(self, path: str) -> dict:
        p = _path(path, must_exist=True)
        if p.is_dir():
            self.artifacts.mkdir(parents=True, exist_ok=True)
            zipped = shutil.make_archive(str(self.artifacts / f"{p.name}-{_stamp()}"), "zip", root_dir=p)
            try:
                return {"path": str(p), "zipped": True, "file": wire.pack_file(zipped)}
            finally:
                os.remove(zipped)
        return {"path": str(p), "file": wire.pack_file(p)}

    def files_put(self, path: str, data: dict, overwrite: bool = False) -> dict:
        p = _path(path)
        if p.is_dir():
            p = p / wire.safe_name(data["__file__"].get("name", "file"))
        if p.exists() and not overwrite:
            return {"status": "exists", "path": str(p), "note": "a file is already there: pass overwrite=True to replace "
                                                                "it (the old one is kept in the device's backups)"}
        p.parent.mkdir(parents=True, exist_ok=True)
        backup = self._keep(p, "backups") if p.exists() else None
        p.write_bytes(wire.file_bytes(data))
        return {"status": "done", "path": str(p), "bytes": p.stat().st_size, "backup": backup}

    def files_move(self, src: str, dst: str) -> dict:
        a, b = _path(src, must_exist=True), _path(dst)
        if b.is_dir():
            b = b / a.name
        if b.exists():
            return {"status": "exists", "path": str(b), "note": "something is already there: move or delete it first"}
        b.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(a), str(b))
        return {"status": "done", "path": str(b)}

    def files_copy(self, src: str, dst: str) -> dict:
        a, b = _path(src, must_exist=True), _path(dst)
        if b.is_dir():
            b = b / a.name
        if b.exists():
            return {"status": "exists", "path": str(b), "note": "something is already there"}
        b.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if a.is_dir() else shutil.copy2)(a, b)
        return {"status": "done", "path": str(b)}

    def files_mkdir(self, path: str) -> dict:
        p = _path(path)
        p.mkdir(parents=True, exist_ok=True)
        return {"status": "done", "path": str(p)}

    def files_delete(self, path: str) -> dict:
        p = _path(path, must_exist=True)
        if p == Path.home() or p == Path(p.anchor) or len(p.parts) <= 2:
            raise Refused(f"will not delete {p}: it is a drive or a top folder")
        where = self._keep(p, "trash")
        return {"status": "done", "path": str(p), "moved_to": where,
                "note": "moved to the device's trash folder (it can be put back from there)"}

    def files_open(self, path: str) -> dict:
        """Open a file or folder in its own app on this device, for the user to see."""
        p = _path(path, must_exist=True)
        if os.name == "nt":
            os.startfile(str(p))  # noqa: S606 - the user's own file, in its default app
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(p)])
        return {"status": "done", "path": str(p)}

    # ------------------------------------------------------------------ shell
    def sh(self, cmd: str, timeout: float = 120, cwd: str | None = None, shell: str = "powershell",
           allow_irreversible: bool = False, call: Call | None = None) -> dict:
        if RISKY.search(cmd) and not allow_irreversible:
            return {"status": "needs_confirmation", "code": None, "out": "",
                    "note": "this command can destroy data or stop programs or the machine: ask the user, then repeat "
                            "with allow_irreversible=True"}
        if os.name == "nt" and shell in ("powershell", "pwsh"):
            exe = shutil.which("pwsh") if shell == "pwsh" else None
            argv: Any = [exe or "powershell", "-NoProfile", "-NonInteractive", "-Command", cmd]
            use_shell = False
        else:
            argv, use_shell = cmd, True
        kw: dict[str, Any] = {}
        if os.name == "nt":
            kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            kw["start_new_session"] = True
        proc = subprocess.Popen(argv, shell=use_shell, cwd=str(_path(cwd)) if cwd else str(Path.home()),
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", **kw)
        if call is not None:
            call.proc = proc
        out: list[str] = []
        reader = threading.Thread(target=lambda: out.append(proc.stdout.read()), daemon=True)
        reader.start()
        deadline = time.time() + float(timeout)
        status = "done"
        while proc.poll() is None:
            if call is not None and call.cancelled.is_set():
                status = "stopped"
            elif time.time() > deadline:
                status = "timed_out"
            if status != "done":
                from ..procs import kill_tree
                kill_tree(proc)
                break
            time.sleep(0.1)
        reader.join(5)
        return {"status": status, "code": proc.returncode, "out": _clip("".join(out), MAX_OUT)}

    # ------------------------------------------------------------------ screen
    def screenshot(self, name: str = "screen.png") -> dict:
        from PIL import ImageGrab
        self.artifacts.mkdir(parents=True, exist_ok=True)
        p = self.artifacts / wire.safe_name(name if name.lower().endswith(".png") else f"{name}.png")
        ImageGrab.grab(all_screens=True).save(p)
        return wire.pack_file(p)

    # ------------------------------------------------------------------ the browser and desktop tools, run here
    def _tool(self, group: str):
        if group not in self._tools:
            os.environ["JEVH_ARTIFACTS"] = str(self.artifacts)   # (both read it when imported)
            os.environ.setdefault("JEVH_BROWSER_HEADED", "1")     # the device's user sees what the assistant does
            import importlib
            mod = importlib.import_module(f"jevh.tools.{group}")
            mod._host = self._host_for_tool
            self._tools[group] = mod
        return self._tools[group]

    def tool_functions(self, group: str) -> list[str]:
        mod = self._tool(group)
        return sorted(n for n, f in vars(mod).items() if inspect.isfunction(f) and f.__module__ == mod.__name__
                      and not n.startswith("_") and n not in TOOL_SKIP)

    def tool(self, group: str, name: str, args: list, kwargs: dict):
        if name.startswith("_") or name in TOOL_SKIP:
            raise Refused(f"{group}.{name} is not offered")
        mod = self._tool(group)
        fn = getattr(mod, name, None)
        if not inspect.isfunction(fn) or fn.__module__ != mod.__name__:
            raise Refused(f"{group} has no function {name}(); it has: {', '.join(self.tool_functions(group))}")
        self._used[group] = time.time()
        return self.attach(fn(*args, **kwargs))

    def _host_for_tool(self, name: str, *args, **kwargs):
        """A tool's call into the harness (look: the worker's model reads a screenshot), sent through the call in progress;
        a screenshot path goes as the file itself."""
        call = self.call_now()
        if call is None:
            raise RuntimeError(f"{name}() needs a call in progress")
        return call.host(name, *[self.attach(a) for a in args], **kwargs)

    def ask_user(self, question: str, options: list | None = None, secret: bool = False):
        """The desktop and nango tools ask the user through `__main__._call("ask_user", ...)`: the agent points that here."""
        call = self.call_now()
        if call is None:
            raise RuntimeError("no call in progress to ask the user through")
        return call.host("ask_user", question, options, secret=secret)

    def attach(self, value):
        """A result with each path to a file this device made (in artifacts/) replaced by the file itself."""
        if isinstance(value, str) and len(value) < 1000:
            try:
                p = Path(value)
                if p.is_file() and p.resolve().is_relative_to(self.artifacts.resolve()):
                    return wire.pack_file(p)
            except (OSError, ValueError):
                pass
            return value
        if isinstance(value, dict):
            return {k: self.attach(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.attach(v) for v in value]
        return value

    def close_idle(self, idle: float = BROWSER_IDLE) -> bool:
        """Close the browser when nothing used it for `idle` seconds (it would otherwise stay open on the user's screen)."""
        mod = self._tools.get("browser")
        if mod is None or time.time() - self._used.get("browser", 0) < idle or getattr(mod, "_session", None) is None:
            return False
        if not self.ui_lock.acquire(blocking=False):
            return False
        try:
            mod.close()
            return True
        finally:
            self.ui_lock.release()

    def close(self) -> None:
        for mod in self._tools.values():
            try:
                mod.close()
            except Exception:  # noqa: BLE001
                pass


def summary(op: str, args: list, kwargs: dict) -> str:
    """One line for the device's activity log and its tray notice: the op and its first arguments, never a secret."""
    parts = []
    for a in args[:2]:
        parts.append("<file>" if wire.is_file(a) else json.dumps(a, default=str)[:120])
    for k, v in list(kwargs.items())[:4]:
        if re.search(r"pass|secret|token|key|pin|card", k, re.I):
            v = "***"
        elif isinstance(v, dict):
            v = {kk: ("***" if re.search(r"pass|secret|token|key|pin|card", str(kk), re.I) else vv)
                 for kk, vv in v.items()} if not wire.is_file(v) else "<file>"
        parts.append(f"{k}={json.dumps(v, default=str)[:80]}")
    return f"{op}({', '.join(parts)})"
