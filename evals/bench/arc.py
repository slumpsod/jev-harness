"""ARC-AGI-1 (public evaluation set): the harness vs the same model answering directly.

Both modes use the persona's own model block (Qwen, unlimited thinking), the same tasks and
exact-match scoring (the whole test output grid, one attempt: pass@1).

  base     one chat call: the train pairs and the test input in the prompt, the grid back.
  harness  a Harness turn on high_end_dev: the task is a JSON file the worker loads in its
           REPL, so it can write the rule as code and check it against every train pair.
           Fresh workspace per task (no memory between tasks), learning off, tier forced to
           strong, and NO web/browser/MCP (the public tasks' solutions are online).

    python evals/bench/arc.py --mode base --n 20
    python evals/bench/arc.py --mode harness --n 20
    python evals/bench/arc.py --report                   # both result files side by side
    python evals/bench/arc.py --watch                    # follow the running task's thinking live

While a run is going: create evals/bench/runs/SKIP to end the current task early (recorded as
skipped; its thinking files stay), or runs/STOP to end the run after the current task.

Data: git clone https://github.com/fchollet/ARC-AGI evals/bench/data/arc1 (not committed).
Tasks with more than one test input are skipped (about 1 in 10) to keep one grid per answer.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from jevh.config import load_config  # noqa: E402
from jevh.llm import make_llm  # noqa: E402

DATA = HERE / "data" / "arc1" / "data" / "evaluation"
RUNS = HERE / "runs"
SKIP = RUNS / "SKIP"   # touch to end the current task early
STOP = RUNS / "STOP"   # touch to end the run after the current task


def watch_skip(stop) -> tuple[threading.Event, threading.Event]:
    """Poll for the SKIP file; on it, call stop() (cancels the model call). -> (skipped, done)."""
    skipped, done = threading.Event(), threading.Event()

    def loop():
        while not done.wait(1.0):
            if SKIP.exists():
                try:
                    SKIP.unlink()
                except OSError:
                    pass
                skipped.set()
                stop()
                return
    threading.Thread(target=loop, daemon=True).start()
    return skipped, done


def live_files() -> list[Path]:
    return sorted(list(RUNS.glob("h*/ws/*/sessions/*/*/thinking/step-*.txt")) + list(RUNS.glob("base_live/*.txt")),
                  key=lambda f: f.stat().st_mtime)


def watch() -> None:
    """Print the newest live thinking file as it grows; switch to a newer one when it appears."""
    current, pos = None, 0
    while True:
        files = live_files()
        if files and files[-1] != current:
            current = files[-1]
            pos = max(0, current.stat().st_size - 4000)   # joining mid-think: just the latest part
            print(f"\n\n########## {current.relative_to(RUNS)}" + (" (latest 4 KB)" if pos else "") + "\n",
                  flush=True)
        if current is not None:
            with open(current, encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
            if chunk:
                print(chunk, end="", flush=True)
        time.sleep(1.0)
PERSONA = "high_end_dev"


def grid_text(g: list[list[int]]) -> str:
    return "[\n" + ",\n".join(" " + json.dumps(row) for row in g) + "\n]"


def parse_grid(text: str) -> list[list[int]] | None:
    """The last JSON list of lists of ints in `text`."""
    dec = json.JSONDecoder()
    starts = [i for i in range(len(text)) if text.startswith("[", i)]
    for i in reversed(starts):
        try:
            val, _ = dec.raw_decode(text[i:])
        except ValueError:
            continue
        if isinstance(val, list) and val and all(isinstance(r, list) and r and all(isinstance(x, int) for x in r)
                                                 for r in val):
            return val
    return None


def load_tasks(n: int, seed: int) -> list[tuple[str, dict]]:
    ids = sorted(p.stem for p in DATA.glob("*.json"))
    tasks = [(i, json.loads((DATA / f"{i}.json").read_text())) for i in ids]
    tasks = [(i, t) for i, t in tasks if len(t["test"]) == 1]
    random.Random(seed).shuffle(tasks)
    return tasks[:n]


PROMPT = """Here is an ARC puzzle. Each example maps an input grid to an output grid by the same hidden rule. \
Grids are lists of rows; each number 0-9 is a colour.

{examples}

Test input:
{test}

Work out the rule, apply it to the test input, and end your reply with the test output grid as JSON \
(a list of rows)."""

HARNESS_MSG = """Solve this ARC-AGI puzzle. The file {path} holds JSON {{"train": [{{"input": grid, "output": grid}}, ...], \
"test": [{{"input": grid}}]}}; grids are lists of rows of colours 0-9. Every train input maps to its output by one \
hidden rule. Load the file in your REPL, work out the rule, write it as a Python function and check that it \
reproduces EVERY train output exactly, then apply it to the test input. Call final() with only the predicted \
test output grid as JSON (a list of rows)."""


def examples_text(task: dict) -> str:
    return "\n\n".join(f"Example {k}\nInput:\n{grid_text(ex['input'])}\nOutput:\n{grid_text(ex['output'])}"
                       for k, ex in enumerate(task["train"], 1))


def run_base(cfg, task_id: str, task: dict, limit: float) -> dict:
    persona_model = load_persona_model(cfg)
    llm = make_llm(persona_model, cfg.providers)
    cancel = threading.Event()
    timer = threading.Timer(limit, cancel.set)
    timer.start()
    skipped, done = watch_skip(cancel.set)
    try:
        prompt = PROMPT.format(examples=examples_text(task), test=grid_text(task["test"][0]["input"]))
        try:
            res = llm.chat([{"role": "user", "content": prompt}], cancel=cancel,
                           live=RUNS / "base_live" / f"{task_id}.txt")
        except Exception:  # noqa: BLE001 - cancelled (time limit or SKIP)
            if not cancel.is_set():
                raise
            return {"answer": "", "input": 0, "output": 0, "thinking": 0,
                    "timed_out": not skipped.is_set(), "skipped": skipped.is_set()}
        return {"answer": res.text[-4000:], "input": res.input_tokens, "output": res.output_tokens,
                "thinking": getattr(res, "thinking_tokens", 0), "timed_out": False, "skipped": False}
    finally:
        timer.cancel()
        done.set()


TIER = "strong"   # --tier


def load_persona_model(cfg) -> dict:
    import yaml
    return yaml.safe_load((HERE.parent.parent / "personas" / f"{PERSONA}.yaml").read_text(encoding="utf-8"))["model"]


def run_harness(cfg, task_id: str, task: dict, limit: float, run_dir: Path) -> dict:
    from jevh.orchestrator import Harness
    ws = run_dir / "ws" / task_id
    ws.mkdir(parents=True, exist_ok=True)
    puzzle = ws / "puzzle.json"
    puzzle.write_text(json.dumps({"train": task["train"], "test": [{"input": t["input"]} for t in task["test"]]}),
                      encoding="utf-8")
    cfg.workspace = ws
    events: list[tuple[str, dict]] = []
    def on_event(k, d):
        events.append((k, d))
        if k == "thinking_watch":   # live: each Jev check of a long think
            print(f"    watch: {d['chars'] // 1000} KB thought, testable {d['p']:.2f}", flush=True)
        elif k == "code":
            first = next((ln for ln in str(d.get("code", "")).splitlines() if ln.strip()), "")
            print(f"    code: step {d.get('step')}: {first[:90]}", flush=True)
        elif k == "compact":
            print(f"    compact: step {d['step']}: ~{d['tokens_before']:,} -> ~{d['tokens_after']:,} tokens, "
                  f"{d['full']} steps kept in full, {d['notes']} as notes, {d['dropped']} left out", flush=True)
        elif k == "output" and "paused a long think" in str(d.get("output")):
            print(f"    watch: step {d.get('step')} paused to test its idea", flush=True)
    h = Harness(cfg, on_event=on_event)
    h.learning, h.tier_mode = False, TIER
    p = h.personas[PERSONA]
    p.tools, p.can_request = [], []
    if hasattr(p, "mcp"):
        p.mcp = []
    timer = threading.Timer(limit, h.cancel)
    timer.start()
    skipped, done = watch_skip(h.cancel)
    try:
        r = h.turn(HARNESS_MSG.format(path=puzzle), force=PERSONA)
    finally:
        timer.cancel()
        done.set()
        h.wait_learning()
    tot = (r.usage or {}).get("total", {})
    steps = [d.get("step") for k, d in events if k == "code"]
    return {"answer": r.answer[-4000:], "input": tot.get("input", 0), "output": tot.get("output", 0),
            "thinking": tot.get("thinking", 0), "timed_out": r.cancelled and not skipped.is_set(),
            "skipped": skipped.is_set(), "steps": max(steps or [0]),
            "bounces": sum(1 for k, d in events if k == "verify" and d.get("bounced")),
            "escalated": any(k == "escalate" for k, _ in events),
            "watch_checks": [(d["chars"], d["p"]) for k, d in events if k == "thinking_watch"],
            "pauses": sum(1 for k, d in events if k == "output" and "paused a long think" in str(d.get("output")))}


def report() -> None:
    rows = {}
    for f in sorted(RUNS.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            rows.setdefault(f.stem.replace("arc1_", ""), {})[r["id"]] = r   # one row per result file
    both = sorted(set.intersection(*(set(v) for v in rows.values()))) if rows else []
    for mode, rs in rows.items():
        n = len(rs)
        ok = sum(r["correct"] for r in rs.values())
        secs = sum(r["seconds"] for r in rs.values())
        print(f"{mode:14} {ok}/{n} = {ok / max(n, 1):.0%}  avg {secs / max(n, 1) / 60:.1f} min/task, "
              f"{sum(r['output'] for r in rs.values()) / max(n, 1):,.0f} out tokens/task, "
              f"no grid {sum(r['grid'] is None for r in rs.values())}, timeouts {sum(r['timed_out'] for r in rs.values())}, "
              f"skipped {sum(bool(r.get('skipped')) for r in rs.values())}")
    if both:
        print(f"\non the {len(both)} tasks every run did: " + ", ".join(
            f"{name} {sum(rs[i]['correct'] for i in both)}" for name, rs in rows.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["base", "harness"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--limit", type=float, default=45 * 60, help="seconds per task")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the results file (a harness version)")
    ap.add_argument("--watch", action="store_true", help="follow the running task's thinking live")
    ap.add_argument("--thinking-watch", action="store_true", help="harness: Jev pauses long thinks at a testable idea")
    ap.add_argument("--compact-tokens", type=int, help="harness: compact the context past this (test compaction)")
    ap.add_argument("--tier", choices=["strong", "fast"], default="strong", help="harness: the persona's model or the fast one")
    a = ap.parse_args()
    if a.report:
        return report()
    if a.watch:
        return watch()
    cfg = load_config()
    if a.thinking_watch:
        cfg.limits["thinking_watch"] = 1
    if a.compact_tokens:
        cfg.limits["compact_tokens"] = a.compact_tokens
    global TIER
    TIER = a.tier
    tasks = load_tasks(400, a.seed) if a.ids else load_tasks(a.n, a.seed)
    if a.ids:
        tasks = [t for t in tasks if t[0] in a.ids]
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"arc1_{a.mode}{'_' + a.tag if a.tag else ''}.jsonl"
    done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()
    for f in (SKIP, STOP):   # stale control files from an earlier run
        f.unlink(missing_ok=True)
    for k, (tid, task) in enumerate(tasks, 1):
        if tid in done:
            continue
        if STOP.exists():
            STOP.unlink(missing_ok=True)
            print("STOP file found: ending the run", flush=True)
            break
        t = time.time()
        try:
            r = (run_base(cfg, tid, task, a.limit) if a.mode == "base"
                 else run_harness(cfg, tid, task, a.limit, RUNS / ("h" + a.tag)))
            err = None
        except Exception as e:  # noqa: BLE001 - a crash is a result, keep going
            r, err = {"answer": "", "input": 0, "output": 0, "thinking": 0, "timed_out": False}, f"{type(e).__name__}: {e}"
        grid = parse_grid(r["answer"])
        want = task["test"][0]["output"]
        row = {"id": tid, "mode": a.mode, "correct": grid == want, "seconds": round(time.time() - t, 1),
               "grid": grid, "want_shape": [len(want), len(want[0])], "error": err, **r}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{k}/{len(tasks)}] {tid} {a.mode}: {'RIGHT' if row['correct'] else 'wrong'} "
              f"{row['seconds'] / 60:.1f} min, {row['output']:,} out ({row['thinking']:,} thinking)"
              f"{' TIMEOUT' if row['timed_out'] else ''}{' SKIPPED' if row.get('skipped') else ''}"
              f"{' no grid' if grid is None else ''}{' ERR ' + err if err else ''}",
              flush=True)


if __name__ == "__main__":
    main()
