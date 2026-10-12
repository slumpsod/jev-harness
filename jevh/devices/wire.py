"""The messages between a hub and a device, one JSON object per WebSocket text message, keyed by `t`.

device -> hub   {"t": "hello", "name", "platform", "version", "paused", "tools": {name: ok|reason}, "info": {...}}
                {"t": "result", "id", "ok", "value", "error"}            the answer to a call
                {"t": "host", "id", "hid", "name", "args", "kwargs"}     a tool on the device asks the harness (look, ask_user)
                {"t": "jev", "rid", "body"}                              a tool's Jev request, for the hub to forward
                {"t": "state", "paused"}                                 the device's user paused or resumed it
hub -> device   {"t": "welcome", "device": {"id", "name"}, "env": {...}}
                {"t": "call", "id", "op", "args", "kwargs", "timeout"}
                {"t": "cancel", "id"}
                {"t": "host_reply", "hid", "value", "error"}
                {"t": "jev_reply", "rid", "status", "body"}

A file travels inside a value as {"__file__": {"name", "b64", "bytes"}}: a screenshot coming back, a file fetched from or
sent to a device. MAX_FILE caps one file; MAX_MESSAGE is what both ends accept in one message (base64 is 4/3 of the file).
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

PROTOCOL = 1
MAX_FILE = 48 * 1024 * 1024
MAX_MESSAGE = 72 * 1024 * 1024
PATH = "/api/devices/ws"
CLOSE_REVOKED = 4401    # the key is unknown or revoked: the device stops trying
CLOSE_REPLACED = 4409   # the same device connected again: the older connection goes


class TooBig(ValueError):
    pass


def pack_file(path: str | os.PathLike, name: str | None = None) -> dict:
    p = Path(path)
    size = p.stat().st_size
    if size > MAX_FILE:
        raise TooBig(f"{p.name} is {size / 1e6:.1f} MB; at most {MAX_FILE // 2**20} MB moves between devices in one call")
    return {"__file__": {"name": name or p.name, "b64": base64.b64encode(p.read_bytes()).decode("ascii"), "bytes": size}}


def is_file(value) -> bool:
    return isinstance(value, dict) and set(value) == {"__file__"} and isinstance(value["__file__"], dict)


def file_bytes(value: dict) -> bytes:
    return base64.b64decode(value["__file__"]["b64"])


def safe_name(name: str) -> str:
    """A file name from the other end, made safe to write here (no folders, no drive, no dots-only)."""
    base = Path(str(name).replace("\\", "/")).name.strip().strip(".")
    return "".join(c if c.isalnum() or c in " ._-()" else "_" for c in base)[:120] or "file"


def unpack_files(value, folder: Path):
    """`value` with every carried file written into `folder` and replaced by its path here."""
    if is_file(value):
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / safe_name(value["__file__"].get("name", "file"))
        stem, k = dest.stem, 2
        while dest.exists():
            dest = dest.with_name(f"{stem}-{k}{dest.suffix}")
            k += 1
        dest.write_bytes(file_bytes(value))
        return str(dest)
    if isinstance(value, dict):
        return {k: unpack_files(v, folder) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack_files(v, folder) for v in value]
    return value
