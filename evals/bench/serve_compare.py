"""Live bench: the same real harness tasks on different ways of serving Qwen3.8 27B (llama.cpp, one GPU).

What it compares (CONFIGS below; all ~70k+ context, all on the ONE GPU named by --gpu, never another):
  fast         Qwen3.8-27B UD-Q4_K_XL + DFlash2 draft (4 tokens), q8 KV cache
  recommended  Qwen3.8-27B UD-Q5_K_M  + MTP (2 tokens),            q8 KV cache
  precision    Qwen3.8-27B UD-Q6_K    + MTP (2 tokens),            q4_0 KV cache

What it runs: full harness turns (Harness.turn on high_end_dev, the real worker/gate/Jev code paths) on
  - ARC-AGI-1 puzzles (evals/bench/arc.py's task loader and prompt): exact-match on the test grid;
  - coding tasks with hidden pytest suites (evals/bench/serve_tasks.py): score = share of hidden tests passed.
Per task the three servers are started one after another (the order rotates so none is always first or last,
and a run stopped half-way still leaves a balanced comparison). Every persona and tier is pointed at the server
under test, so LM Studio is never touched. Needs: llama.cpp b11450+ with CUDA (llama-server), the three GGUFs,
the DFlash2 draft GGUF, a running Jev (config.yaml jev.url), and nothing else on the chosen GPU.

    python evals/bench/serve_compare.py --server C:/path/llama-server.exe --models-dir C:/path/models --dry
    python evals/bench/serve_compare.py --server ... --models-dir ... --coding 6 --arc 6
    python evals/bench/serve_compare.py --server ... --models-dir ... --holdout          # held-out ARC tasks
    python evals/bench/serve_compare.py --report        # offline: re-read the saved rows, no server, no model

Results: evals/bench/runs/serve_compare/rows.jsonl (resumable: a finished (task, config, rep) is never re-run),
evals/bench/serve_compare_raw.json and serve_compare_summary.txt beside this file.
Local models vary run to run: use --reps 2+ before trusting a gap between two servers.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

RUNS = HERE / "runs" / "serve_compare"
ROWS = RUNS / "rows.jsonl"
RAW = HERE / "serve_compare_raw.json"
SUMMARY = HERE / "serve_compare_summary.txt"
PERSONA = "high_end_dev"
PORT = 8090

CONFIGS = {
    "fast": dict(label="Q4_K_XL + DFlash2(4), q8 KV", model="Qwen3.8-27B-UD-Q4_K_XL.gguf",
                 draft="Qwen3.8-27B-DFlash2-Q8_0.gguf", spec="draft-dflash", n=4, kv="q8_0"),
    "recommended": dict(label="Q5_K_M + MTP(2), q8 KV", model="Qwen3.8-27B-UD-Q5_K_M.gguf",
                        draft=None, spec="draft-mtp", n=2, kv="q8_0"),
    "precision": dict(label="Q6_K + MTP(2), q4 KV", model="Qwen3.8-27B-UD-Q6_K.gguf",
                      draft=None, spec="draft-mtp", n=2, kv="q4_0"),
    # not in the default set (--configs q4_mtp): the same Q4_K_XL with MTP instead of the DFlash2 draft (what runs in production at 100k)
    "q4_mtp": dict(label="Q4_K_XL + MTP(2), q8 KV", model="Qwen3.8-27B-UD-Q4_K_XL.gguf",
                   draft=None, spec="draft-mtp", n=2, kv="q8_0"),
    # control, not in the default set (--configs ...,control_nospec): the recommended model with speculation off, to
    # check that MTP/DFlash change speed only and not the answers
    "control_nospec": dict(label="Q5_K_M, no speculation, q8 KV", model="Qwen3.8-27B-UD-Q5_K_M.gguf",
                           draft=None, spec=None, n=0, kv="q8_0"),
}
DEFAULT_CONFIGS = ["fast", "recommended", "precision"]
HELD_ARC = ["c48954c1", "e7639916", "15663ba9", "e66aafb8", "759f3fd3", "351d6448", "a3f84088", "9b365c51"]


# --- the GPU: only the one asked for -----------------------------------------------------------------------------

def find_gpu(name: str) -> tuple[str, int]:
    """(uuid, index) of the single GPU whose name contains `name`; anything else is an error, never a guess."""
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    hits = [(m.group(3), int(m.group(1))) for m in re.finditer(r"GPU (\d+): (.*?) \(UUID: (GPU-[0-9a-f-]+)\)", out)
            if name.lower() in m.group(2).lower()]
    if len(hits) != 1:
        raise SystemExit(f"--gpu {name!r} must match exactly one GPU, found {len(hits)}:\n{out}")
    return hits[0]


def gpu_used_mib(index: int) -> int:
    out = subprocess.run(["nvidia-smi", "-i", str(index), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip().split()[0])


def wait_gpu_free(index: int, limit_mib: int = 1500, timeout: float = 90) -> None:
    t = time.time()
    while gpu_used_mib(index) > limit_mib:
        if time.time() - t > timeout:
            raise SystemExit(f"GPU {index} still has {gpu_used_mib(index)} MiB in use: something else is on it "
                             f"(LM Studio's model? `lms unload --all`). Not starting a server on a shared card.")
        time.sleep(2)


# --- the server -------------------------------------------------------------------------------------------------

def find_model(name: str, dirs: list[Path]) -> Path:
    for d in dirs:
        hit = next(d.rglob(name), None) if d.exists() else None
        if hit:
            return hit
    raise SystemExit(f"{name} not found under {', '.join(map(str, dirs))} (see --models-dir)")


class Server:
    def __init__(self, cfg: dict, binary: str, models: list[Path], uuid: str, gpu_index: int, ctx: int, parallel: int,
                 tag: str):
        self.cfg, self.binary, self.uuid, self.gpu_index = cfg, binary, uuid, gpu_index
        self.model = find_model(cfg["model"], models)
        self.draft = find_model(cfg["draft"], models) if cfg["draft"] else None
        self.ctx, self.parallel, self.tag, self.proc = ctx, parallel, tag, None
        self.log = RUNS / f"server_{tag}.log"

    def args(self) -> list[str]:
        c = self.cfg
        a = [self.binary, "-m", str(self.model), "--host", "127.0.0.1", "--port", str(PORT), "-c", str(self.ctx),
             "-ngl", "999", "-fa", "on", "-ctk", c["kv"], "-ctv", c["kv"], "--parallel", str(self.parallel),
             "--kv-unified", "--jinja", "--no-webui", "--no-mmproj", "-fit", "off", "--metrics"]
        if c["spec"]:
            a += ["--spec-type", c["spec"], "--spec-draft-n-max", str(c["n"])]
        if self.draft:
            a += ["-md", str(self.draft)]   # (-ngld/-ctkd/-ctvd make the DFlash2 draft fail to load in b11451)
        return a

    def start(self) -> None:
        wait_gpu_free(self.gpu_index)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": self.uuid}   # the one GPU: the server cannot see any other
        RUNS.mkdir(parents=True, exist_ok=True)
        with open(self.log, "w") as out:
            self.proc = subprocess.Popen(self.args(), env=env, stdout=out, stderr=subprocess.STDOUT)
        t = time.time()
        while time.time() - t < 240:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.tag}: llama-server exited ({self.proc.returncode}):\n" + self.tail())
            try:
                if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)).get("status") == "ok":
                    break
            except Exception:  # noqa: BLE001 - not up yet
                time.sleep(2)
        else:
            self.stop()
            raise RuntimeError(f"{self.tag}: llama-server not healthy after 240 s:\n" + self.tail())
        text = self.log.read_text(errors="ignore")
        if "4080" in text or "4090" in text:   # belt and braces: this must never happen with CUDA_VISIBLE_DEVICES set
            self.stop()
            raise RuntimeError(f"{self.tag}: the server saw another GPU:\n" + self.tail())

    def tail(self, n: int = 12) -> str:
        return "\n".join(self.log.read_text(errors="ignore").splitlines()[-n:])

    def metrics(self) -> dict[str, float]:
        out = {}
        try:
            for line in urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=5).read().decode().splitlines():
                m = re.match(r"llamacpp:(\w+)\s+([-0-9.e+]+)$", line)
                if m:
                    out[m.group(1)] = float(m.group(2))
        except Exception:  # noqa: BLE001 - metrics are extra, never fatal
            pass
        return out

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(30)


# --- the tasks --------------------------------------------------------------------------------------------------

def seed_task(task: dict, ws: Path) -> None:
    ws.mkdir(parents=True, exist_ok=True)
    for name, text in task["files"].items():
        (ws / name).write_text(text, encoding="utf-8")


def grade_coding(task: dict, ws: Path) -> dict:
    """Run the hidden tests against what the worker wrote in `ws`."""
    hidden = ws / "_hidden"
    hidden.mkdir(exist_ok=True)
    (hidden / f"test_{task['id']}_hidden.py").write_text(task["tests"], encoding="utf-8")
    for f in ws.glob("*.py"):   # the hidden suite imports the worker's modules from the task folder
        if not f.name.startswith("test_"):
            shutil.copy(f, hidden / f.name)
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "--tb=line", "-p", "no:cacheprovider", str(hidden)],
                       capture_output=True, text=True, cwd=str(hidden), timeout=300)
    out = p.stdout + p.stderr
    passed = int((re.search(r"(\d+) passed", out) or [0, 0])[1])
    failed = int((re.search(r"(\d+) failed", out) or [0, 0])[1])
    errors = int((re.search(r"(\d+) error", out) or [0, 0])[1])
    total = passed + failed + errors
    if total == 0:   # nothing collected (the module does not import): count the suite's tests as failed
        total = max(1, len(re.findall(r"^def test_", task["tests"], re.M)))
    return {"passed": passed, "total": total, "all_passed": passed == total and failed == 0 and errors == 0,
            "tail": out.strip().splitlines()[-4:]}


def arc_tasks(n: int, holdout: bool) -> list[tuple[str, dict]]:
    import arc
    if holdout:
        return [t for t in arc.load_tasks(400, 0) if t[0] in HELD_ARC][:n]
    return arc.load_tasks(n, 0)


# --- one harness turn -------------------------------------------------------------------------------------------

def run_turn(message: str, ws: Path, block: dict, base_url: str, limit: float) -> dict:
    """A real Harness turn with every persona and tier on `block`. -> usage, steps, timing, answer."""
    from jevh.config import load_config
    from jevh.orchestrator import Harness
    cfg = load_config()
    cfg.providers["serve_bench"] = {"type": "openai", "base_url": base_url, "native_tools": True}
    cfg.models = {"fast": dict(block), "strong": dict(block)}   # nothing falls back to LM Studio
    cfg.raw.setdefault("failures", {})["audit"] = 0               # no background re-check outliving the server
    cfg.workspace = ws
    ws.mkdir(parents=True, exist_ok=True)
    events: list[tuple[str, dict]] = []
    h = Harness(cfg, on_event=lambda k, d: events.append((k, d)))
    h.learning, h.tier_mode = False, "strong"
    for p in h.personas.values():
        p.model = dict(block)
    p = h.personas[PERSONA]
    p.tools, p.can_request = [], []
    if hasattr(p, "mcp"):
        p.mcp = []
    timer = threading.Timer(limit, h.cancel)
    timer.start()
    t = time.time()
    try:
        r = h.turn(message, force=PERSONA)
    finally:
        timer.cancel()
        h.wait_learning()
    tot = (r.usage or {}).get("total", {})
    steps = [d.get("step") for k, d in events if k == "code"]
    return {"answer": r.answer[-4000:], "seconds": round(time.time() - t, 1), "input": tot.get("input", 0),
            "output": tot.get("output", 0), "thinking": tot.get("thinking", 0), "timed_out": bool(r.cancelled),
            "steps": max(steps or [0]),
            "bounces": sum(1 for k, d in events if k == "verify" and d.get("bounced")),
            "escalated": any(k == "escalate" for k, _ in events)}


def run_item(item: dict, cfgname: str, srv: Server, a) -> dict:
    block = {"provider": "serve_bench", "model": "qwen3.8-27b", "think": "unlimited", "context_length": a.ctx}
    ws = RUNS / "ws" / f"{item['key']}__{cfgname}__r{item['rep']}"
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    m0 = srv.metrics()
    row = {"task": item["key"], "kind": item["kind"], "config": cfgname, "rep": item["rep"], "error": None}
    try:
        if item["kind"] == "code":
            task = item["task"]
            seed_task(task, ws)
            msg = task["prompt"].replace("{dir}", str(ws))
            r = run_turn(msg, ws, block, f"http://127.0.0.1:{PORT}/v1", a.limit_code)
            g = grade_coding(task, ws)
            row.update(r, passed=g["passed"], total=g["total"], correct=g["all_passed"], tail=g["tail"])
        else:
            import arc
            tid, task = item["task"]
            ws.mkdir(parents=True, exist_ok=True)
            puzzle = ws / "puzzle.json"
            puzzle.write_text(json.dumps({"train": task["train"], "test": [{"input": t["input"]} for t in task["test"]]}),
                              encoding="utf-8")
            r = run_turn(arc.HARNESS_MSG.format(path=puzzle), ws, block, f"http://127.0.0.1:{PORT}/v1", a.limit_arc)
            grid = arc.parse_grid(r["answer"])
            row.update(r, correct=grid == task["test"][0]["output"], grid=grid is not None)
    except Exception as e:  # noqa: BLE001 - a crash is a result, keep going
        row.update(error=f"{type(e).__name__}: {e}", correct=False, seconds=0, output=0, thinking=0, input=0, steps=0,
                   bounces=0, timed_out=False)
    m1 = srv.metrics()
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in ("tokens_predicted_total", "tokens_predicted_seconds_total",
                                                 "prompt_tokens_total", "prompt_seconds_total")}
    row["server"] = {"decode_tps": round(d["tokens_predicted_total"] / d["tokens_predicted_seconds_total"], 1)
                     if d["tokens_predicted_seconds_total"] else None,
                     "prefill_tps": round(d["prompt_tokens_total"] / d["prompt_seconds_total"], 1)
                     if d["prompt_seconds_total"] else None,
                     "decoded": int(d["tokens_predicted_total"]), "prefilled": int(d["prompt_tokens_total"])}
    return row


# --- report -----------------------------------------------------------------------------------------------------

def load_rows() -> list[dict]:
    return [json.loads(line) for line in ROWS.read_text(encoding="utf-8").splitlines()] if ROWS.exists() else []


def report(rows: list[dict] | None = None) -> str:
    rows = load_rows() if rows is None else rows
    if not rows:
        return "no results yet"
    names = [c for c in CONFIGS if any(r["config"] == c for r in rows)]   # (the control only has bugfix rows)
    by = {c: [r for r in rows if r["config"] == c] for c in names}
    keys = sorted({(r["task"], r["rep"]) for r in rows})
    done_all = [k for k in keys if all(any((r["task"], r["rep"]) == k for r in by[c]) for c in names)]
    ok_all = [k for k in done_all if all(next(r for r in by[c] if (r["task"], r["rep"]) == k)["correct"] for c in names)]
    L = [f"serve_compare: {len(rows)} runs, {len(done_all)} (task, rep) pairs done on every server, "
         f"{len(ok_all)} solved by every server\n"]
    head = f"{'':14}{'server':30}{'solved':>10}{'code':>7}{'ARC':>7}{'test%':>7}{'min/task':>10}{'min (solved by all)':>21}" \
           f"{'out tok':>9}{'tok/s':>7}{'srv dec':>8}{'srv pre':>8}{'steps':>6}{'t/o':>5}{'err':>5}"
    L += [head, "-" * len(head)]
    for c in names:
        rs = by[c]
        code = [r for r in rs if r["kind"] == "code"]
        arc = [r for r in rs if r["kind"] == "arc"]
        secs = sum(r["seconds"] for r in rs)
        outt = sum(r["output"] for r in rs)
        dec = [r["server"] for r in rs if r["server"].get("decode_tps")]
        pre = [r["server"] for r in rs if r["server"].get("prefill_tps")]
        both = [r for r in rs if (r["task"], r["rep"]) in ok_all]
        pct = statistics.mean(r["passed"] / r["total"] for r in code) * 100 if code else float("nan")
        L.append(f"{c:14}{CONFIGS[c]['label']:30}{sum(r['correct'] for r in rs):>4}/{len(rs):<5}"
                 f"{sum(r['correct'] for r in code):>3}/{len(code):<3}{sum(r['correct'] for r in arc):>3}/{len(arc):<3}"
                 f"{pct:>6.0f}%{secs / len(rs) / 60:>10.1f}"
                 f"{(statistics.mean(r['seconds'] for r in both) / 60 if both else float('nan')):>21.1f}"
                 f"{outt / len(rs):>9,.0f}{outt / secs if secs else 0:>7.1f}"
                 f"{(sum(d['decoded'] for d in dec) / sum(d['decoded'] / d['decode_tps'] for d in dec) if dec else 0):>8.1f}"
                 f"{(sum(d['prefilled'] for d in pre) / sum(d['prefilled'] / d['prefill_tps'] for d in pre) if pre else 0):>8.0f}"
                 f"{statistics.mean(r['steps'] for r in rs):>6.1f}{sum(r['timed_out'] for r in rs):>5}"
                 f"{sum(bool(r['error']) for r in rs):>5}")
    L += ["", "per task (minutes, RIGHT/wrong; code tasks also show tests passed):"]
    for k in keys:
        cells = []
        for c in names:
            r = next((r for r in by[c] if (r["task"], r["rep"]) == k), None)
            if r is None:
                cells.append(f"{c}: -")
            else:
                extra = f" {r['passed']}/{r['total']}" if r["kind"] == "code" else ""
                cells.append(f"{c}: {'RIGHT' if r['correct'] else 'wrong'}{extra} {r['seconds'] / 60:.1f}m"
                             f"{' TIMEOUT' if r['timed_out'] else ''}{' ERR' if r['error'] else ''}")
        L.append(f"  {k[0]:10} r{k[1]}  " + "   ".join(cells))
    L += ["", "tok/s = output tokens / wall seconds of the whole turn (includes reading, tools, Jev); srv dec / srv pre = the "
              "server's own decode and prompt-read speed over the same runs. min (solved by all) compares time only on the "
              "tasks every server got right, so a server is not rewarded for giving up early."]
    return "\n".join(L)


# --- main -------------------------------------------------------------------------------------------------------

def plan(a) -> list[dict]:
    from serve_tasks import TASKS
    items = []
    codes = list(TASKS.values())[: a.coding]
    arcs = arc_tasks(a.arc, a.holdout) if a.arc else []
    for rep in range(1, a.reps + 1):
        items += [{"key": t["id"], "kind": "code", "task": t, "rep": rep} for t in codes]
    for rep in range(1, a.arc_reps + 1):
        items += [{"key": tid, "kind": "arc", "task": (tid, t), "rep": rep} for tid, t in arcs]
    if a.only:
        items = [i for i in items if i["key"] in a.only]
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=os.environ.get("LLAMA_SERVER") or shutil.which("llama-server"),
                    help="llama-server binary (env LLAMA_SERVER)")
    ap.add_argument("--models-dir", action="append", default=[], help="folder(s) holding the GGUFs (searched recursively)")
    ap.add_argument("--gpu", default="3090", help="name of the ONE GPU to use (must match exactly one)")
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    ap.add_argument("--coding", type=int, default=6, help="coding tasks (of 6)")
    ap.add_argument("--arc", type=int, default=6, help="ARC tasks")
    ap.add_argument("--holdout", action="store_true", help="ARC: the held-out task ids instead of the dev ones")
    ap.add_argument("--reps", type=int, default=1, help="repeats of each coding task")
    ap.add_argument("--arc-reps", type=int, default=1, help="repeats of each ARC task")
    ap.add_argument("--interleave", action="store_true", help="swap servers per task instead of running one server through everything")
    ap.add_argument("--only", nargs="*", help="run just these task ids")
    ap.add_argument("--ctx", type=int, default=80000)
    ap.add_argument("--parallel", type=int, default=2, help="server slots (LM Studio here uses 2)")
    ap.add_argument("--limit-code", type=float, default=15 * 60, help="seconds per coding task")
    ap.add_argument("--limit-arc", type=float, default=20 * 60, help="seconds per ARC task")
    ap.add_argument("--dry", action="store_true", help="show the plan and the server command lines, run nothing")
    ap.add_argument("--report", "--offline", action="store_true", help="re-read saved rows and print the summary")
    a = ap.parse_args()
    if a.report:
        print(report())
        return
    names = [c for c in a.configs.split(",") if c]
    uuid, gpu_index = find_gpu(a.gpu)
    models = [Path(d) for d in a.models_dir] + [Path.home() / ".lmstudio" / "models"]
    items = plan(a)
    print(f"GPU {gpu_index} ({a.gpu}, {uuid}); {len(items)} tasks x {len(names)} servers = {len(items) * len(names)} runs")
    if a.dry:
        for n in names:
            print(n, " ".join(Server(CONFIGS[n], a.server or "llama-server", models, uuid, gpu_index, a.ctx, a.parallel, n).args()))
        for i in items:
            print(f"  {i['kind']:5} {i['key']} r{i['rep']}")
        return
    if not a.server:
        raise SystemExit("--server (or env LLAMA_SERVER) is required")
    RUNS.mkdir(parents=True, exist_ok=True)
    done = {(r["task"], r["config"], r["rep"]) for r in load_rows()}

    def record(row: dict, n: int, item: dict) -> None:
        with open(ROWS, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{n + 1}/{len(items)}] {item['key']:10} r{item['rep']} {row['config']:12} "
              f"{'RIGHT' if row['correct'] else 'wrong'} {row['seconds'] / 60:5.1f} min  {row['output']:>7,} out "
              f"{row['steps']:>2} steps  server {row['server']['decode_tps']} tok/s"
              f"{' TIMEOUT' if row['timed_out'] else ''}{' ERR ' + row['error'] if row['error'] else ''}", flush=True)

    if a.interleave:   # per task, every server in turn (rotating who goes first): fair if stopped early, many model swaps
        for n, item in enumerate(items):
            order = names[n % len(names):] + names[:n % len(names)]
            for cfgname in order:
                if (item["key"], cfgname, item["rep"]) in done:
                    continue
                srv = Server(CONFIGS[cfgname], a.server, models, uuid, gpu_index, a.ctx, a.parallel, cfgname)
                try:
                    srv.start()
                    record(run_item(item, cfgname, srv, a), n, item)
                finally:
                    srv.stop()
    else:   # default: one server up, every task on it, then the next server (one model load each)
        for cfgname in names:
            todo = [(n, i) for n, i in enumerate(items) if (i["key"], cfgname, i["rep"]) not in done]
            if not todo:
                continue
            srv = Server(CONFIGS[cfgname], a.server, models, uuid, gpu_index, a.ctx, a.parallel, cfgname)
            try:
                srv.start()
                for n, item in todo:
                    if srv.proc.poll() is not None:   # it died: bring it back for the next task
                        srv.start()
                    record(run_item(item, cfgname, srv, a), n, item)
            finally:
                srv.stop()
    rows = load_rows()
    RAW.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    SUMMARY.write_text(report(rows), encoding="utf-8")
    print("\n" + report(rows))


if __name__ == "__main__":
    main()
