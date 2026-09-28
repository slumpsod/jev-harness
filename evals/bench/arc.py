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


def run_base(cfg, task: dict, limit: float) -> dict:
    persona_model = load_persona_model(cfg)
    llm = make_llm(persona_model, cfg.providers)
    cancel = threading.Event()
    timer = threading.Timer(limit, cancel.set)
    timer.start()
    try:
        prompt = PROMPT.format(examples=examples_text(task), test=grid_text(task["test"][0]["input"]))
        res = llm.chat([{"role": "user", "content": prompt}], cancel=cancel)
        return {"answer": res.text[-4000:], "input": res.input_tokens, "output": res.output_tokens,
                "thinking": getattr(res, "thinking_tokens", 0), "timed_out": cancel.is_set()}
    finally:
        timer.cancel()


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
    h = Harness(cfg, on_event=lambda k, d: events.append((k, d)))
    h.learning, h.tier_mode = False, "strong"
    p = h.personas[PERSONA]
    p.tools, p.can_request = [], []
    if hasattr(p, "mcp"):
        p.mcp = []
    timer = threading.Timer(limit, h.cancel)
    timer.start()
    try:
        r = h.turn(HARNESS_MSG.format(path=puzzle), force=PERSONA)
    finally:
        timer.cancel()
        h.wait_learning()
    tot = (r.usage or {}).get("total", {})
    steps = [d.get("step") for k, d in events if k == "code"]
    return {"answer": r.answer[-4000:], "input": tot.get("input", 0), "output": tot.get("output", 0),
            "thinking": tot.get("thinking", 0), "timed_out": r.cancelled, "steps": max(steps or [0]),
            "bounces": sum(1 for k, d in events if k == "verify" and d.get("bounced")),
            "escalated": any(k == "escalate" for k, _ in events)}


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
              f"no grid {sum(r['grid'] is None for r in rs.values())}, timeouts {sum(r['timed_out'] for r in rs.values())}")
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
    a = ap.parse_args()
    if a.report:
        return report()
    cfg = load_config()
    tasks = load_tasks(400, a.seed) if a.ids else load_tasks(a.n, a.seed)
    if a.ids:
        tasks = [t for t in tasks if t[0] in a.ids]
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"arc1_{a.mode}{'_' + a.tag if a.tag else ''}.jsonl"
    done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()
    for k, (tid, task) in enumerate(tasks, 1):
        if tid in done:
            continue
        t = time.time()
        try:
            r = run_base(cfg, task, a.limit) if a.mode == "base" else run_harness(cfg, tid, task, a.limit, RUNS / "h")
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
              f"{' TIMEOUT' if row['timed_out'] else ''}{' no grid' if grid is None else ''}{' ERR ' + err if err else ''}",
              flush=True)


if __name__ == "__main__":
    main()
