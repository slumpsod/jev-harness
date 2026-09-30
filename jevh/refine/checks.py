"""Checkers: did a test run do the task? They are how the refinement loop scores old vs new, so
they are independent of the harness's own failure detectors: a fix that only makes the final()
gate more lenient must not score as an improvement.

A checker is JSON, so the local model can write one:
  {"type": "contains", "text": "...", "case": false}    the answer contains the text
  {"type": "not_contains", "text": "..."}
  {"type": "regex", "pattern": "..."}
  {"type": "number", "value": 412, "tolerance": 0.5}     a number in the answer within tolerance
  {"type": "json_equals", "value": [...]}                the answer (or the JSON in it) equals value
  {"type": "artifact", "name": "*.html", "contains": "..."}   a handed-over file (glob), non-empty
  {"type": "python", "code": "def check(answer, artifacts): ..."}   -> bool or (bool, why)
  {"type": "jev", "question": "Does `answer` ...?", "min": 0.7}      Jev judges request + answer
  {"type": "all" | "any", "checks": [...]}

A checker is only used once it tells good from bad (discriminates()): it must pass a good
answer, and fail the bad one and a plain refusal."""
from __future__ import annotations

import fnmatch
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

Judge = Callable[[str, dict], float]   # (question, {"request", "answer"}) -> P(yes)
TYPES = {"contains", "not_contains", "regex", "number", "json_equals", "artifact", "python", "jev", "all", "any"}
REFUSALS = ["I'm sorry, I couldn't complete this task.", "(no answer)"]
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


class CheckerError(ValueError):
    pass


def problems(spec: Any, where: str = "checker") -> list[str]:
    """What is structurally wrong with a checker ([] = usable)."""
    if not isinstance(spec, dict):
        return [f"{where} is not an object"]
    t = spec.get("type")
    if t not in TYPES:
        return [f"{where}: unknown type {t!r}"]
    need = {"contains": "text", "not_contains": "text", "regex": "pattern", "number": "value", "json_equals": "value",
            "artifact": "name", "python": "code", "jev": "question", "all": "checks", "any": "checks"}[t]
    if need not in spec:
        return [f"{where}: {t} needs {need!r}"]
    if t in ("all", "any"):
        if not isinstance(spec["checks"], list) or not spec["checks"]:
            return [f"{where}: {t} needs a non-empty list of checks"]
        return [p for i, c in enumerate(spec["checks"]) for p in problems(c, f"{where}.{t}[{i}]")]
    if t == "regex":
        try:
            re.compile(spec["pattern"])
        except re.error as e:
            return [f"{where}: bad regex: {e}"]
    if t == "number" and not isinstance(spec["value"], (int, float)):
        return [f"{where}: number value must be a number"]
    if t == "python" and "def check" not in str(spec["code"]):
        return [f"{where}: python code must define check(answer, artifacts)"]
    return []


def check(spec: dict, answer: str, artifacts: list[str] | None = None, request: str = "",
          judge: Judge | None = None, files_assumed: bool = False) -> tuple[bool, str]:
    """-> (passed, why). `files_assumed`: artifact checks pass (validating a text-only good answer)."""
    answer = answer or ""
    artifacts = list(artifacts or [])
    t = spec.get("type")
    if t in ("all", "any"):
        results = [check(c, answer, artifacts, request, judge, files_assumed) for c in spec["checks"]]
        if t == "all":
            bad = [why for ok, why in results if not ok]
            return (not bad, bad[0] if bad else "all checks passed")
        good = [why for ok, why in results if ok]
        return (bool(good), good[0] if good else "; ".join(why for _, why in results))
    fold = (lambda s: s) if spec.get("case") else (lambda s: " ".join(s.lower().split()))
    if t == "contains":
        ok = fold(str(spec["text"])) in fold(answer)
        return ok, f"{'contains' if ok else 'missing'} {str(spec['text'])[:60]!r}"
    if t == "not_contains":
        ok = fold(str(spec["text"])) not in fold(answer)
        return ok, f"{'does not contain' if ok else 'contains'} {str(spec['text'])[:60]!r}"
    if t == "regex":
        ok = re.search(spec["pattern"], answer, 0 if spec.get("case") else re.IGNORECASE) is not None
        return ok, f"regex {spec['pattern'][:60]!r} {'matched' if ok else 'not found'}"
    if t == "number":
        tol = float(spec.get("tolerance", 1e-9))
        nums = [float(n.replace(",", "")) for n in NUMBER.findall(answer.replace("$", ""))]
        ok = any(abs(n - float(spec["value"])) <= tol for n in nums)
        return ok, f"number {spec['value']} {'found' if ok else 'not found'}"
    if t == "json_equals":
        ok = _json_in(answer) == spec["value"]
        return ok, "JSON matches" if ok else "JSON differs"
    if t == "artifact":
        if files_assumed:
            return True, "file check assumed"
        for a in artifacts:
            p = Path(a)
            if fnmatch.fnmatch(p.name.lower(), str(spec["name"]).lower()) and p.is_file() and p.stat().st_size:
                if not spec.get("contains") or fold(str(spec["contains"])) in fold(
                        p.read_text(encoding="utf-8", errors="replace")):
                    return True, f"file {p.name} ok"
        return False, f"no handed-over file {spec['name']}" + (f" containing {spec['contains']!r}"
                                                             if spec.get("contains") else "")
    if t == "python":
        return _python(spec["code"], answer, artifacts, request)
    if t == "jev":
        if judge is None:
            raise CheckerError("a jev check needs Jev")
        p = judge(str(spec["question"]), {"request": request[:2000], "answer": answer[:4000]})
        ok = p >= float(spec.get("min", 0.7))
        return ok, f"Jev {p:.2f} on {str(spec['question'])[:60]!r}"
    raise CheckerError(f"unknown checker type {t!r}")


def discriminates(spec: dict, good: str, bad: str, request: str = "", judge: Judge | None = None,
                  good_files: dict | None = None) -> tuple[bool, str]:
    """A checker is kept only if it passes `good` and fails `bad` and a plain refusal. `good_files`
    ({name: content}): what a correct run hands over, written to disk so file checks read real files.
    (Live: a python checker that looked for file contents in `artifacts` - which holds paths - passed a
    text-only good answer, then failed the real, correct planets.csv.)"""
    errs = problems(spec)
    if errs:
        return False, "; ".join(errs)
    try:
        with tempfile.TemporaryDirectory() as d:
            paths = _write_files(Path(d), good_files)
            ok, why = check(spec, good, paths, request, judge, files_assumed=not paths)
            if not ok:
                return False, f"rejects the good answer ({why})"
            if paths:   # a correct run's answer often only names the file: the file must be enough
                names = ", ".join(Path(p).name for p in paths)
                if not check(spec, f"Here is {names}.", paths, request, judge)[0]:
                    return False, ("passes only when the answer text repeats the file's content; check the file "
                                   "(artifacts holds its path)")
        for b in [bad, *REFUSALS]:
            if b and check(spec, b, [], request, judge)[0]:
                return False, f"accepts a bad answer: {b[:80]!r}"
    except Exception as e:  # noqa: BLE001 - a broken generated checker is just not used
        return False, f"{type(e).__name__}: {e}"
    return True, "ok"


def _write_files(folder: Path, files: dict | None) -> list[str]:
    paths = []
    for name, content in (files or {}).items() if isinstance(files, dict) else []:
        p = folder / (Path(str(name)).name or "file")
        if isinstance(content, str):
            p.write_text(content, encoding="utf-8")
        else:
            p.write_text(json.dumps(content), encoding="utf-8")
        paths.append(str(p))
    return paths


def describe(spec: dict) -> str:
    """One line for prompts and reports."""
    t = spec.get("type")
    if t in ("all", "any"):
        return f" {'and' if t == 'all' else 'or'} ".join(describe(c) for c in spec.get("checks", []))
    return {"contains": lambda: f"answer contains {spec.get('text')!r}",
            "not_contains": lambda: f"answer doesn't contain {spec.get('text')!r}",
            "regex": lambda: f"answer matches /{spec.get('pattern')}/",
            "number": lambda: f"answer gives {spec.get('value')}",
            "json_equals": lambda: "answer is the exact JSON expected",
            "artifact": lambda: f"hands over {spec.get('name')}" + (f" containing {spec.get('contains')!r}"
                                                                   if spec.get("contains") else ""),
            "python": lambda: "a code check of the answer",
            "jev": lambda: f"Jev: {spec.get('question')}"}.get(t, lambda: str(t))()


def uses_jev(spec: dict) -> bool:
    return spec.get("type") == "jev" or any(uses_jev(c) for c in spec.get("checks", []) if isinstance(c, dict))


def _json_in(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        pass
    m = re.search(r"[\[{]", text)
    if not m:
        return None
    try:
        return json.JSONDecoder().raw_decode(text[m.start():])[0]
    except ValueError:
        return None


RUNNER = """
import json, sys
data = json.loads(sys.stdin.read())
ns = {}
exec(compile(data["code"], "checker", "exec"), ns)
out = ns["check"](data["answer"], data["artifacts"])
ok, why = (out if isinstance(out, tuple) else (out, ""))
print(json.dumps({"ok": bool(ok), "why": str(why)}))
"""


def _python(code: str, answer: str, artifacts: list[str], request: str) -> tuple[bool, str]:
    """Generated checker code runs in its own process with a time limit."""
    try:
        p = subprocess.run([sys.executable, "-c", RUNNER], input=json.dumps(
            {"code": code, "answer": answer, "artifacts": artifacts, "request": request}),
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    except subprocess.TimeoutExpired:
        return False, "code check timed out"
    try:
        r = json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return False, f"code check failed: {p.stderr.strip()[-200:]}"
    return r["ok"], r["why"] or f"code check {'passed' if r['ok'] else 'failed'}"
