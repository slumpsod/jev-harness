"""A program's output is read as UTF-8 (errors replaced), never in the locale's code page.

On Windows, subprocess's text mode decodes with the code page (cp1252 here): UTF-8 from git, Node (Playwright's
driver), Chromium or Tailscale holding a byte cp1252 has no character for (0x90, as in "А" = d0 90) raised
UnicodeDecodeError in subprocess's reader thread, and the output came back as None (issue #7)."""
import ast
import sys
import threading
from pathlib import Path

from jevh import update

ROOT = Path(__file__).resolve().parent.parent
CALLS = {"run", "Popen", "check_output", "check_call", "call"}
UTF8_ON_STDERR = "import sys; print('ok'); sys.stderr.buffer.write('\\u0410\\u2192 driver note\\n'.encode('utf-8'))"


def _text_mode_calls_without_encoding(root: Path) -> list[str]:
    bad = []
    for p in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in CALLS
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess"):
                continue
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            text = any(isinstance(kw.get(k), ast.Constant) and kw[k].value for k in ("text", "universal_newlines"))
            if text and "encoding" not in kw:
                bad.append(f"{p.relative_to(ROOT).as_posix()}:{node.lineno}")
    return bad


def test_every_subprocess_call_that_reads_text_names_its_encoding():
    assert _text_mode_calls_without_encoding(ROOT / "jevh") == []
    assert _text_mode_calls_without_encoding(ROOT / "tests") == []


def test_utf8_output_a_code_page_cannot_decode_comes_through(tmp_path):
    died = []
    hook, threading.excepthook = threading.excepthook, lambda a: died.append(a.exc_value)
    try:
        p = update.run([sys.executable, "-c", UTF8_ON_STDERR], tmp_path, 30)
    finally:
        threading.excepthook = hook
    assert not died, died                     # (before: UnicodeDecodeError in the reader thread, stderr None)
    assert p.stdout.strip() == "ok" and "А→ driver note" in p.stderr
