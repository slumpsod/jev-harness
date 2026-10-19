"""Files the user sends with a chat message (the dashboard's attach button, paste and drop; issue #5).

POST /api/upload?name=<file name> takes the file's bytes as the body (no multipart: nothing to install) and keeps it in
the session's folder, workspace/sessions/<id>/uploads/. The message that follows names the paths it got back; the
text the harness gets is the message with a note listing them (note()), so routing, Jev's tool pick, the worker and
the session's memory all see that files came with it, and the worker opens them from there (look_at() for an image or a
scanned PDF). The dashboard reads the note back out of the message to show the files in the chat (thumbnails, chips).
"""
from __future__ import annotations

import re
import threading
from pathlib import Path

MAX_BYTES = 25 * 1024 * 1024        # a file: a phone photo is 2-12 MB, a scanned PDF a few MB
MAX_FILES = 10                       # files with one message
MAX_SESSION_BYTES = 500 * 1024 * 1024   # all of a session's uploads together
IMAGES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
DOCS = {".pdf", ".txt", ".md", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml", ".log", ".docx", ".xlsx", ".xls",
        ".pptx", ".odt", ".ods", ".rtf", ".eml", ".zip"}
ALLOWED = IMAGES | DOCS              # (no programs, scripts or web pages)
HEAD = "[Attached by the user (saved on this computer):"
NOTE_RE = re.compile(r"\n*\[Attached by the user \(saved on this computer\):\n((?:- .+\n)+)\]\s*$")
_lock = threading.Lock()


class UploadError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def folder(session_dir: Path) -> Path:
    return Path(session_dir) / "uploads"


def kind_of(name: str) -> str:
    ext = Path(name).suffix.lower()
    return "image" if ext in IMAGES else "PDF" if ext == ".pdf" else "file"


def safe_name(name: str) -> str:
    """The file's own name, made safe for a folder on any system: no path, no odd characters, not too long."""
    base = re.split(r"[\\/]", str(name or ""))[-1].strip().strip(".")
    stem, ext = Path(base).stem, Path(base).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9 _.()-]+", "_", stem).strip(" ._")[:60] or "file"
    return stem + re.sub(r"[^a-z0-9.]", "", ext)[:8]


def check_name(name: str) -> str:
    clean = safe_name(name)
    ext = Path(clean).suffix.lower()
    if ext not in ALLOWED:
        raise UploadError(f"{ext or 'a file without an extension'} can't be sent: images, PDFs, text, CSV and "
                          "Office files can", 415)
    return clean


def used(session_dir: Path) -> int:
    d = folder(session_dir)
    return sum(p.stat().st_size for p in d.iterdir() if p.is_file()) if d.is_dir() else 0


def save(session_dir: Path, name: str, data: bytes) -> dict:
    """Keep one sent file: -> {path, name, bytes, kind}. The name gets a number in front, so two "image.png" pasted
    one after the other are two files."""
    clean = check_name(name)
    if not data:
        raise UploadError("the file is empty")
    if len(data) > MAX_BYTES:
        raise UploadError(f"the file is over {MAX_BYTES // (1024 * 1024)} MB", 413)
    with _lock:
        if used(session_dir) + len(data) > MAX_SESSION_BYTES:
            raise UploadError(f"this session's files are over {MAX_SESSION_BYTES // (1024 * 1024)} MB: start a new "
                              "session to send more", 413)
        d = folder(session_dir)
        d.mkdir(parents=True, exist_ok=True)
        n = 1 + max((int(m[1]) for p in d.iterdir() if (m := re.match(r"(\d+)-", p.name))), default=0)
        path = d / f"{n:03d}-{clean}"
        path.write_bytes(data)
    return {"path": str(path), "name": clean, "bytes": len(data), "kind": kind_of(clean)}


def resolve(session_dir: Path, paths) -> list[Path]:
    """The files a message names, each one sent to this session (anything else is refused, not passed on)."""
    if not paths:
        return []
    if not isinstance(paths, list) or len(paths) > MAX_FILES:
        raise UploadError(f"send at most {MAX_FILES} files with a message")
    root = folder(session_dir).resolve()
    out = []
    for p in paths:
        f = Path(str(p)).resolve()
        if not f.is_relative_to(root) or not f.is_file():
            raise UploadError(f"{Path(str(p)).name} is not one of this session's files: attach it again")
        if f not in out:
            out.append(f)
    return out


def size_text(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB" if n >= 1024 * 1024 else f"{max(1, round(n / 1024))} KB"


def note(files: list[Path]) -> str:
    lines = "".join(f"- {f} ({kind_of(f.name)}, {size_text(f.stat().st_size)})\n" for f in files)
    return f"{HEAD}\n{lines}]"


def with_files(message: str, files: list[Path]) -> str:
    """The message as the harness gets it: the user's text, then the note naming the files."""
    if not files:
        return message
    text = message.strip() or "(No message: only the attached file" + ("s" if len(files) > 1 else "") + ".)"
    return f"{text}\n\n{note(files)}"


def split(text: str) -> tuple[str, list[str]]:
    """(the user's own text, the attached paths) out of a message made by with_files()."""
    m = NOTE_RE.search(text or "")
    if not m:
        return text, []
    paths = [re.sub(r" \([^()]*\)$", "", line[2:]) for line in m[1].splitlines() if line.startswith("- ")]
    return text[:m.start()], paths
