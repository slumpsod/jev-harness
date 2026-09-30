"""The test matrix: every case x model x repeat x code version, each run in its own process and
workspace, scored by the case's checker in this (the current) process - so a fix can't change
how it is scored.

Runs are appended to results.jsonl as they finish; a run already there is never repeated, so a
night cut short by the deadline carries on the next night. Runs are grouped by model (LM Studio
keeps one model loaded), and old and new take turns on each case, so both see the same conditions."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from ..config import Config, shell_env
from . import checks
from .cases import Case

Execute = Callable[[Path, dict, str, Path], dict]   # (code root, case, model, run workspace) -> result


def is_local(provider: dict[str, Any]) -> bool:
    """The refinement loop runs on local models only: no Claude API, no keyed cloud endpoint."""
    return provider.get("type") == "openai" and not provider.get("api_key_env")


def model_id(block: dict[str, Any]) -> str:
    return f"{block['provider']}:{block['model']}"


def local_models(cfg: Config, personas: dict | None = None) -> list[str]:
    """Every local model the harness runs on: config refine.models if set, else the tier models
    (models.strong replaces the personas' own) or the personas' models, plus models.fast; with
    refine.discover, also every chat model LM Studio has."""
    settings = cfg.raw.get("refine") or {}
    if settings.get("models"):
        found = [str(m) for m in settings["models"]]
    else:
        blocks = []
        if cfg.models.get("strong"):
            blocks.append(cfg.models["strong"])
        else:
            if personas is None:
                from ..personas import load_personas
                personas = load_personas(cfg.personas_dir)
            blocks += [p.model for p in personas.values()]
        if cfg.models.get("fast"):
            blocks.append(cfg.models["fast"])
        found = [model_id(b) for b in blocks if b.get("provider") and b.get("model")]
        if settings.get("discover"):
            found += discover(cfg)
    out = []
    for m in found:
        prov = cfg.providers.get(m.partition(":")[0])
        if prov is not None and is_local(prov) and m not in out:
            out.append(m)
    return out


def discover(cfg: Config) -> list[str]:
    """Chat models the exclusive (LM Studio) providers list."""
    out = []
    for name, prov in cfg.providers.items():
        if not (prov.get("exclusive") and prov.get("base_url")):
            continue
        try:
            with urllib.request.urlopen(prov["base_url"].rstrip("/") + "/models", timeout=5) as r:
                data = json.loads(r.read())
        except (OSError, ValueError):
            continue
        out += [f"{name}:{m['id']}" for m in data.get("data", []) if "embed" not in m.get("id", "").lower()]
    return out


def execute(root: Path, case: dict, model: str, ws: Path, timeout: float = 3600) -> dict:
    """One run in its own process, with the code in `root` (its jevh first on the path)."""
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "case.json").write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")
    out = ws / "result.json"
    env = shell_env()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "jevh.refine.run_case", "--case", str(ws / "case.json"), "--model", model,
           "--workspace", str(ws), "--out", str(out), "--config", str(root / "config.yaml")]
    started = time.time()
    code, err = _run(cmd, root, env, timeout, ws / "run.log")
    if out.is_file():
        res = json.loads(out.read_text(encoding="utf-8"))
    else:
        res = {"error": err or f"exit {code}, no result", "seconds": round(time.time() - started, 2)}
    ran = res.get("jevh_file")
    if ran and not Path(ran).resolve().is_relative_to(Path(root).resolve()):
        res["error"] = f"ran the wrong code: {ran} is not under {root}"
    return res


def _run(cmd: list[str], cwd: Path, env: dict, timeout: float, log: Path) -> tuple[int | None, str]:
    """Run to the end or the timeout; on timeout, kill it with everything it started (REPLs, browsers)."""
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT, **kw)
        try:
            return p.wait(timeout), ""
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
            else:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except OSError:
                    pass
            p.wait()
            return None, f"timed out after {timeout:.0f}s"


def key(row: dict) -> tuple:
    return row["version"], row["model"], row["case"], row["rep"]


def load_rows(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass   # a line cut off by a crash
    return rows


def score(case: Case, res: dict, judge: checks.Judge | None) -> tuple[bool, str]:
    if res.get("error"):
        return False, f"run error: {res['error'][:200]}"
    try:
        return checks.check(case.checker, res.get("answer", ""), res.get("artifacts", []), case.request, judge)
    except Exception as e:  # noqa: BLE001 - a checker that breaks scores the run as not done, with why
        return False, f"checker error: {type(e).__name__}: {e}"


def run_matrix(cases: list[Case], versions: dict[str, Path], models: list[str], runs: int, results: Path,
               runs_dir: Path, judge: checks.Judge | None = None, execute_fn: Execute | None = None,
               deadline: float | None = None, on_row: Callable[[dict], None] | None = None) -> bool:
    """Runs what isn't in `results` yet. True = complete, False = stopped at the deadline."""
    execute_fn = execute_fn or execute
    done = {key(r) for r in load_rows(results)}
    for model in models:
        for rep in range(runs):
            for case in cases:
                for version, root in versions.items():
                    row_key = (version, model, case.id, rep)
                    if row_key in done:
                        continue
                    if deadline is not None and time.time() >= deadline:
                        return False
                    ws = runs_dir / f"{version}-{_safe(model)}-{case.id}-r{rep}"
                    res = execute_fn(Path(root), case.to_dict(), model, ws)
                    passed, why = score(case, res, judge)
                    row = {"version": version, "model": model, "case": case.id, "rep": rep, "split": case.split,
                           "passed": passed, "why": why, "ts": time.time(), "workspace": str(ws),
                           **{k: res.get(k) for k in ("seconds", "input", "output", "thinking", "model_seconds",
                                                      "jev_calls", "status", "failures", "error")},
                           "answer": str(res.get("answer", ""))[:1500]}
                    with open(results, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    done.add(row_key)
                    if on_row:
                        on_row(row)
    return True


def _safe(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)[:60]
