"""Run one test case as one harness turn and write what happened as JSON.

    python -m jevh.refine.run_case --case case.json --model lmstudio:qwen3.8-27b \\
        --workspace runs/x --out result.json [--config path/config.yaml]

The refinement runner starts this in the code tree being tested (the current harness, or a fix's
worktree), so the harness that runs is that tree's. Each run gets its own workspace: memory,
failures and artifacts from one run never leak into the next. Learning is off (it would write
skills into the tree), and ask_user() gets no answer, so an unattended run can't hang on it."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ASK_TIMEOUT = 5   # seconds a worker's ask_user() waits: nobody answers overnight


def run_one(case: dict[str, Any], cfg, model: str | None, jev=None, llm_factory=None) -> dict[str, Any]:
    import jevh
    from jevh.orchestrator import Harness

    cfg.limits["ask_timeout"] = ASK_TIMEOUT
    cfg.raw = {**cfg.raw, "learning": False}
    kw = {"jev": jev} if jev is not None else {}
    if llm_factory is not None:
        kw["llm_factory"] = llm_factory
    started = time.time()
    h = Harness(cfg, model_override=model or None, **kw)
    h.learning = False
    out: dict[str, Any] = {"jevh_file": str(Path(jevh.__file__).resolve()), "model": model or ""}
    try:
        r = h.turn(case["request"], force=case.get("persona") or None)
        by = r.usage.get("by_kind", {})
        llm = [by.get("llm") or {}, by.get("learn") or {}]
        jev_ = [by.get("jev") or {}, by.get("jev_tools") or {}]
        root = r.tasks[0] if r.tasks else None
        out.update({
            "answer": r.answer, "artifacts": list(r.artifacts), "persona": r.persona,
            "seconds": round(r.seconds, 2), "input": sum(x.get("input", 0) for x in llm),
            "output": sum(x.get("output", 0) for x in llm), "thinking": sum(x.get("thinking", 0) for x in llm),
            "model_seconds": round(sum(x.get("seconds", 0) for x in llm), 2),
            "jev_calls": sum(x.get("calls", 0) for x in jev_), "status": root.status if root else "",
            "calls": sum(x.get("calls", 0) for x in llm),   # model steps: what a needless bounce costs
            "tasks": len(r.tasks), "failures": [f["kind"] for f in h.failures.list(limit=200)]})
        out["bounces"] = out["failures"].count("gate_bounce")
    except Exception as e:  # noqa: BLE001 - reported as the run's error, never raised
        out.update({"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:],
                    "seconds": round(time.time() - started, 2)})
    finally:
        h.wait_learning(5)
    return out


def main(argv: list[str] | None = None) -> int:
    from jevh.config import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    cfg.workspace = Path(a.workspace)
    case = json.loads(Path(a.case).read_text(encoding="utf-8"))
    res = run_one(case, cfg, a.model)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
