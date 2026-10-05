"""Live check of context-length detection against the LM Studio on this machine (read-only: nothing is loaded,
unloaded or sent to a model).

    python evals/bench/context_detect.py            # one line per check, a summary, raw results beside it
    python evals/bench/context_detect.py --reps 20  # ask more often (stability, cost per question)

What it compares, for every LLM the LM Studio in config.yaml lists:
  - what the harness resolves (ExclusiveModels.model_info, the call a worker makes before each step) with the raw
    GET /api/v1/models answer read here directly (the parsing is right);
  - the same with `lms ps --json`, LM Studio's own CLI, when `lms` is on the PATH (an independent source: the
    context LM Studio itself says each loaded model has);
  - a model that is not loaded has no context yet (and nothing is loaded by asking);
  - the whole path a worker uses: for each model the config puts to work, make_llm(...).window() and the compaction
    limit built from it (jevh.context.report), against the same numbers;
  - the answer is the same on every repetition, and how long one uncached question takes (it is asked before every
    step, at most every 10 s per model).
What it cannot check without changing your LM Studio: that a model reloaded with another context is noticed. That is
covered by tests/test_context.py with a fake server; to see it live, reload a model with another context length in
LM Studio and run `jevh context` again.

Results: evals/bench/context_detect_last_summary.txt and evals/bench/runs/context_detect/<time>/raw.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jevh import context as ctxwin  # noqa: E402
from jevh.config import load_config  # noqa: E402
from jevh.lmstudio import gate_for, info_from  # noqa: E402


def lms_ps() -> list[dict] | None:
    """`lms ps --json`: the loaded models as LM Studio's own CLI reports them (None: no `lms` here)."""
    exe = shutil.which("lms")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "ps", "--json"], capture_output=True, text=True, timeout=30, check=True).stdout
        return json.loads(out)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config")
    ap.add_argument("--reps", type=int, default=10, help="questions asked to measure stability and cost (default 10)")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    ctxwin.configure(cfg)
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    servers = [(n, p) for n, p in cfg.providers.items() if p.get("exclusive") and p.get("base_url")]
    if not servers:
        print("no exclusive (LM Studio) provider in the config: nothing to check")
        return 1
    raw: dict = {"started": time.strftime("%Y%m%d-%H%M%S"), "servers": {}}
    ps = lms_ps()
    raw["lms_ps"] = ps
    for pname, prov in servers:
        gate = gate_for(prov["base_url"])
        gate.info_ttl = 0.0   # every question goes to LM Studio
        try:
            listed = gate._request("GET", "/api/v1/models", timeout=10)["models"]
        except Exception as e:  # noqa: BLE001
            check(f"{pname}: LM Studio answers", False, f"{type(e).__name__}: {e}")
            continue
        llms = [m for m in listed if m.get("type", "llm") == "llm"]
        check(f"{pname}: LM Studio lists LLMs", bool(llms), f"{len(llms)} listed")
        raw["servers"][pname] = {"listed": listed}
        for m in llms:
            key = m["key"]
            inst = [i for i in m.get("loaded_instances") or []]
            direct = min((i["config"]["context_length"] for i in inst if i.get("config", {}).get("context_length")), default=None)
            got = gate.model_info(key)
            check(f"{key}: resolved as the raw answer says", got is not None and got["context"] == direct and
                  got["loaded"] == bool(inst) and got["max"] == m.get("max_context_length"),
                  f"harness {got and got['context']} | raw {direct} | loaded {bool(inst)}")
            if inst:
                if ps is not None:
                    theirs = [p for p in ps if p.get("modelKey") == key]
                    check(f"{key}: same context as `lms ps`", bool(theirs) and theirs[0].get("contextLength") == got["context"],
                          f"harness {got['context']} | lms ps {theirs[0].get('contextLength') if theirs else 'not listed'}")
                    if theirs and theirs[0].get("maxContextLength") is not None:
                        check(f"{key}: same maximum as `lms ps`", theirs[0]["maxContextLength"] == got["max"],
                              f"harness {got['max']} | lms ps {theirs[0]['maxContextLength']}")
                times, seen = [], set()
                for _ in range(max(a.reps, 1)):
                    t0 = time.perf_counter()
                    one = gate.model_info(key)
                    times.append((time.perf_counter() - t0) * 1000)
                    seen.add(one and one["context"])
                check(f"{key}: the same answer {a.reps} times", seen == {got["context"]}, f"{sorted(map(str, seen))}")
                raw["servers"][pname].setdefault("ask_ms", {})[key] = [round(t, 1) for t in times]
                check(f"{key}: asking is cheap (median under 250 ms)", statistics.median(times) < 250,
                      f"median {statistics.median(times):.0f} ms, worst {max(times):.0f} ms")
            else:
                check(f"{key}: not loaded, so no context yet", got is not None and got["context"] is None and not got["loaded"],
                      f"max {got and got['max']}")
        after = gate._request("GET", "/api/v1/models", timeout=10)["models"]
        check(f"{pname}: asking loaded and unloaded nothing",
              [(m["key"], len(m.get("loaded_instances") or [])) for m in after]
              == [(m["key"], len(m.get("loaded_instances") or [])) for m in listed])

    # the path a worker takes: the config's models -> make_llm(...).window() -> the compaction limit
    from jevh.llm import make_llm
    report = ctxwin.report(cfg)
    raw["report"] = report
    share = ctxwin.settings(cfg.raw)["compact_at"]
    for use, row in zip(ctxwin.models_in_use(cfg), report["models"]):
        provider, _, model = row["model"].partition(":")
        listed = raw["servers"].get(provider, {}).get("listed")
        if listed is None or row["source"] == "config":
            continue   # not an LM Studio model we listed above, or its window is set in config
        info = info_from(listed, model)
        reply = make_llm(use["block"], cfg.providers).max_tokens
        if info is not None and info["context"]:
            want = ctxwin.compact_tokens(info["context"], reply, share)
            ok = row["window"] == info["context"] and row["compact"] == want
            detail = f"window {row['window']} (LM Studio: {info['context']}) | limit {row['compact']} (expected {want})"
        else:   # not loaded: no window yet, the configured limits.compact_tokens stands
            ok = row["window"] is None and row["compact"] == cfg.lim("compact_tokens")
            detail = f"window {row['window']} | limit {row['compact']} (limits.compact_tokens {cfg.lim('compact_tokens')})"
        check(f"{row['model']}: a worker gets the window and the limit made of it", ok,
              f"{detail} | used by {', '.join(row['used_by'])}")

    failed = [c for c in checks if not c["ok"]]
    lines = [f"CONTEXT DETECTION  {raw['started']}  config={a.config or 'config.yaml'}  `lms ps`: {'used' if ps is not None else 'not available'}",
             ""]
    lines += [f"  {'ok  ' if c['ok'] else 'FAIL'} {c['check']}" + (f"   [{c['detail']}]" if c["detail"] else "") for c in checks]
    lines += ["", f"{len(checks) - len(failed)}/{len(checks)} checks passed" + (f"; failing: {', '.join(c['check'] for c in failed)}" if failed else "")]
    text = "\n".join(lines)
    print(text)
    run_dir = ROOT / "evals" / "bench" / "runs" / "context_detect" / raw["started"]
    run_dir.mkdir(parents=True, exist_ok=True)
    raw["checks"] = checks
    (run_dir / "raw.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    (ROOT / "evals" / "bench" / "context_detect_last_summary.txt").write_text(text + f"\n\nraw results in {run_dir.relative_to(ROOT).as_posix()}\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
