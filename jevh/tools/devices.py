"""devices: the user's other computers, signed in to this assistant (jevh/devices). In a worker's REPL.

    devices.list()                         -> [{name, online, paused, platform, tools, ...}]
    pc = devices.get("laptop")
    pc.files.list("~/Downloads"); pc.files.read(path); pc.files.find("invoice 2026", root="~/Documents")
    pc.files.write(path, text); pc.files.fetch(path) -> a path here; pc.files.send(local, remote)
    pc.files.move/copy/mkdir/delete/open
    pc.sh("Get-Process | Select -First 5")  -> {status, code, out}
    pc.screenshot()                        -> a path here
    pc.browser.do("..."), pc.desktop.do("...", window=...)   (the browser and desktop tools, run on that computer)

Every call goes to the harness (`_host("device", ...)`), which sends it to the device and brings files back into this
task's folder (from-devices/<device>/).
"""
from __future__ import annotations

import builtins
import os
from pathlib import Path

from jevh.devices import wire

_host = None   # set by the worker's REPL: calls into the harness

BROWSER = ("goto", "do", "read", "elements", "click", "fill", "select", "press", "hold", "look", "screenshot", "url", "html",
           "test", "visible", "close")
DESKTOP = ("windows", "where", "focus", "open", "do", "read", "elements", "click", "fill", "select", "key", "screenshot",
           "look", "close")


def _call(device: str | None, op: str, *args, _timeout: float | None = None, **kwargs):
    if _host is None:
        raise RuntimeError("devices works inside a worker's REPL (it asks the harness, which holds the connections)")
    if _timeout is not None:
        kwargs["_timeout"] = _timeout
    return _host("device", device, op, builtins.list(args), kwargs)


class _Remote:
    """A tool (browser, desktop) on a device: every function of the tool, run there."""

    def __init__(self, device: "Device", group: str, names: tuple):
        self._device, self._group, self._names = device, group, names

    def __getattr__(self, fn: str):
        if fn.startswith("_"):
            raise AttributeError(fn)

        def run(*args, **kwargs):
            return _call(self._device.name, f"{self._group}.{fn}", *args, **kwargs)
        run.__name__ = fn
        return run

    def __dir__(self):
        return builtins.list(self._names)

    def __repr__(self):
        return f"<{self._group} on {self._device.name}: {', '.join(self._names)}>"


class _Files:
    def __init__(self, device: "Device"):
        self._d = device

    def _c(self, op, *a, **k):
        return _call(self._d.name, f"files.{op}", *a, **k)

    def list(self, path: str = "~", pattern: str = "*", recursive: bool = False, limit: int = 200) -> dict:
        """-> {path, entries: [{name, path, dir, bytes, modified}], more}"""
        return self._c("list", path, pattern, recursive, limit)

    def stat(self, path: str) -> dict:
        return self._c("stat", path)

    def read(self, path: str, start: int | None = None, end: int | None = None) -> str:
        """Text (start/end: numbered lines). Not for binary files: fetch() them."""
        return self._c("read", path, start, end)

    def find(self, name: str = "*", root: str = "~", contains: str | None = None, limit: int = 50) -> dict:
        """-> {hits: [{name, path, bytes, modified}] newest first, looked_at, stopped_early}. `name`: a glob ("*.pdf")
        or words that must all be in the file name ("invoice march"); `contains`: text inside (text files)."""
        return self._c("find", name, root, contains, limit)

    def write(self, path: str, text: str, append: bool = False) -> dict:
        """Write text there (a file it replaces is kept in the device's backups)."""
        return self._c("write", path, text, append)

    def fetch(self, path: str, dest: str | None = None) -> str:
        """Bring a file (or a folder, zipped) from the device into this task's folder -> its path here."""
        got = self._c("fetch", path, _timeout=900)
        local = got["file"]
        if dest:
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            os.replace(local, dest)
            local = str(Path(dest).resolve())
        return local

    def send(self, local: str, path: str, overwrite: bool = False) -> dict:
        """Copy a file from here to the device (`path`: a folder or the new file's path)."""
        return self._c("put", path, wire.pack_file(local), overwrite, _timeout=900)

    def move(self, src: str, dst: str) -> dict:
        return self._c("move", src, dst)

    def copy(self, src: str, dst: str) -> dict:
        return self._c("copy", src, dst)

    def mkdir(self, path: str) -> dict:
        return self._c("mkdir", path)

    def delete(self, path: str) -> dict:
        """Moves it to the device's trash folder (it can be put back)."""
        return self._c("delete", path)

    def open(self, path: str) -> dict:
        """Open a file or folder in its own app on the device, for the user there to see."""
        return self._c("open", path)


class Device:
    def __init__(self, name: str):
        self.name = name
        self.files = _Files(self)
        self.browser = _Remote(self, "browser", BROWSER)
        self.desktop = _Remote(self, "desktop", DESKTOP)

    def info(self) -> dict:
        """-> {host, platform, user, home, drives, folders, tools, paused}"""
        return _call(self.name, "info")

    def sh(self, cmd: str, timeout: float = 120, cwd: str | None = None, shell: str = "powershell",
           allow_irreversible: bool = False) -> dict:
        """Run a command there (PowerShell by default; shell="cmd" for cmd.exe) -> {status, code, out}."""
        return _call(self.name, "sh", cmd, timeout=timeout, cwd=cwd, shell=shell, allow_irreversible=allow_irreversible,
                     _timeout=float(timeout) + 60)

    def screenshot(self, name: str = "screen.png") -> str:
        """The device's whole screen -> a PNG's path here."""
        return _call(self.name, "screenshot", name)

    def __repr__(self) -> str:
        return f"<device {self.name}>"


def list() -> builtins.list:  # noqa: A001 - the tool's own verb
    """The user's devices: [{name, online, paused, platform, tools, last_seen, ...}]; `here` is the hub itself."""
    return _call(None, "list")


def get(name: str) -> Device:
    """A device by name (any case, or the start of its name)."""
    for d in list():
        if d.get("name", "").lower() == str(name).lower():
            return Device(d["name"])
    hits = [d for d in list() if d.get("name", "").lower().startswith(str(name).lower())]
    if len(hits) == 1:
        return Device(hits[0]["name"])
    return Device(str(name))   # (the call says what is wrong: offline, unknown, ambiguous)


def copy(src: str, dst: str) -> dict:
    """Between computers: copy("laptop:~/Documents/a.pdf", "studio:~/Desktop/"); "here:path" is this task's folder."""
    (sd, sp), (dd, dp) = (s.split(":", 1) if ":" in s and len(s.split(":", 1)[0]) > 1 else ("here", s) for s in (src, dst))
    local = sp if sd == "here" else get(sd).files.fetch(sp)
    if dd == "here":
        Path(dp).parent.mkdir(parents=True, exist_ok=True)
        os.replace(local, dp) if sd != "here" else Path(dp).write_bytes(Path(local).read_bytes())
        return {"status": "done", "path": str(Path(dp).resolve())}
    return get(dd).files.send(local, dp)
