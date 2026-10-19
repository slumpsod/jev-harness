"""Live bench: long tasks and context compaction. Does a worker still finish well when its context is condensed,
and how late can that happen?

Designed for: Qwen3.8 27B on llama-server (config.local.yaml: provider llamacpp, 76k window, 2 slots, native tools) as
the worker, the semif Jev (config.yaml jev.url) scoring the steps. Needs both running; touches nothing else. It
never starts or stops a server (Jed's llama-server may be in use: it only sends requests to it).

What it runs: full harness turns (Harness.turn on high_end_dev) on the long tasks in evals/bench/long_tasks.py
(`rulebook`: 18 pages of policy to read, a policy.py to write, a hidden pytest file with 80 cases), once per
VARIANT (a set of settings; see VARIANTS) and seed, several repeats. For each run it records

  correct / tests passed          the hidden tests, as in serve_compare.py
  compactions, tokens before/after, entries dropped    from the harness's own `compact` events
  history reads                   how often the worker looked something up in history.md (variants with `archive`)
  re-reads                        how often the worker opened a rule page it had already read ("lost it" cost),
                                  and how many of those came after a compaction
  steps, minutes, in/out tokens, thinking tokens       the cost of the run
  prompt estimate vs real         the harness estimates tokens at 3.2 chars each; the server counts them. This is
                                  what an 80% limit stands on, so the error is measured on every call
  peak prompt, share of window    the largest real prompt sent, and whether any call failed on the window

    python evals/bench/long_horizon.py --dry                             # the plan, nothing runs
    python evals/bench/long_horizon.py --variants base,at80 --seeds 1,2 --reps 2
    python evals/bench/long_horizon.py --window 40000 ...                # the harness is told the window is 40k (so
                                                                         # compaction happens after fewer steps; the
                                                                         # server really has 76k: no overflow risk)
    python evals/bench/long_horizon.py --holdout                         # seeds the variants were not tuned on
    python evals/bench/long_horizon.py --report                          # offline: re-read saved rows (the --offline
                                                                         # sweep of other evals), no model
    python evals/bench/long_horizon.py --mode headroom                   # how full can a prompt get before a long
                                                                         # thinking reply stops fitting (real window)

Results: evals/bench/runs/long_horizon/rows.jsonl (resumable: a finished (variant, seed, rep) is not re-run),
evals/bench/long_horizon_raw.json and long_horizon_summary.txt beside this file.
Local models vary a lot from run to run: 1 rep says nothing about a gap of one task. Use --reps 3+.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

RUNS = HERE / "runs" / "long_horizon"
ROWS = RUNS / "rows.jsonl"
RAW = HERE / "long_horizon_raw.json"
SUMMARY = HERE / "long_horizon_summary.txt"
PERSONA = "high_end_dev"
DEV_SEEDS = [1, 2, 3]
HOLD_SEEDS = [11, 12, 13]

# name -> settings laid over the config for the run: {"context": {...}, "limits": {...}, "compaction": {...}}
VARIANTS: dict[str, dict] = {
    "base": {"label": "as shipped: compact at 60% of the window", "context": {"compact_at": 0.6}},
    "at80": {"label": "compact at 80% (the reply's room still comes off the top)", "context": {"compact_at": 0.8}},
    # the changes of 2026-10-08, one at a time and together (jevh/context.py settings), each at 80%
    "at80_measure": {"label": "80% + prompt sized from the server's own token count",
                     "context": {"compact_at": 0.8, "measure": True}},
    "at80_archive": {"label": "80% + condensed steps saved in full in history.md",
                     "context": {"compact_at": 0.8, "archive": True}},
    "at80_all": {"label": "80% + measure + archive + note tails + log share 25%",
                 "context": {"compact_at": 0.8, "measure": True, "archive": True, "note_tail": True, "log_share": 0.25}},
    "base_all": {"label": "60% + measure + archive + note tails + log share 25%",
                 "context": {"compact_at": 0.6, "measure": True, "archive": True, "note_tail": True, "log_share": 0.25}},
}


def flat_code(events: list[tuple[str, dict]]) -> list[tuple[int, str]]:
    return [(d.get("step", 0), d.get("code", "")) for k, d in events if k == "code"]


def rereads(events: list[tuple[str, dict]], marks: list[str]) -> dict:
    """Work on the rule pages after the first compaction: code blocks that open the rules folder again. (Before the
    first compaction the worker is reading them for the first time; after it, a block that opens them again is a
    re-read, i.e. what condensing the context cost.)"""
    first_compact = min([d.get("step", 10 ** 6) for k, d in events if k == "compact"] or [10 ** 6])
    touch = [(step, code) for step, code in flat_code(events) if re.search(r"rule_|rules[/\\]|\.glob|listdir", code)]
    return {"rule_blocks": len(touch), "reread_after_compact": sum(1 for step, _ in touch if step > first_compact),
            "reread": sum(1 for step, _ in touch if step > first_compact),
            "history_reads": sum("history.md" in code for _, code in flat_code(events))}


# -- one harness turn ------------------------------------------------------------------------------------------------

class CallLog:
    """Every model call of a run: what the harness estimated for the prompt, and what the server counted."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.errors: list[str] = []

    def wrap(self):
        from jevh import llm as llmmod
        orig = llmmod.OpenAICompat._chat
        log = self

        def chat(self_, messages, *a, **kw):
            est = sum(len(str(m.get("content", ""))) + len(str(m.get("reasoning_content", ""))) for m in messages) / 3.2
            #      (what the worker counts: not the code in native tool calls, which the server does)
            t = time.time()
            try:
                r = orig(self_, messages, *a, **kw)
            except Exception as e:  # noqa: BLE001 - recorded, then the harness decides what it means
                log.errors.append(f"{type(e).__name__}: {str(e)[:200]}")
                raise
            log.rows.append({"est": round(est), "real": r.input_tokens, "out": r.output_tokens, "think": r.thinking_tokens,
                             "secs": round(time.time() - t, 1), "decode": r.decode_seconds})
            return r
        llmmod.OpenAICompat._chat = chat
        return lambda: setattr(llmmod.OpenAICompat, "_chat", orig)


def run_turn(task: dict, variant: dict, ws: Path, window: int, limit: float) -> dict:
    from jevh import context as ctxwin
    from jevh.config import load_config
    from jevh.orchestrator import Harness
    cfg = load_config()
    for key in ("context", "limits"):
        for k, v in (variant.get(key) or {}).items():
            (cfg.raw.setdefault(key, {}) if key == "context" else cfg.limits)[k] = v
    cfg.raw.setdefault("failures", {})["audit"] = 0     # no background re-check outliving the run
    cfg.workspace = ws
    ws.mkdir(parents=True, exist_ok=True)
    real_props = ctxwin.server_props
    if window:   # the harness is told the window is smaller than the server's, so compaction comes sooner
        ctxwin.server_props = lambda base_url, *a, **k: {**(real_props(base_url) or {"slots": 2, "model": "", "file": "", "build": ""}),
                                                         "context": window}
    calls = CallLog()
    unwrap = calls.wrap()
    events: list[tuple[str, dict]] = []
    h = Harness(cfg, on_event=lambda k, d: events.append((k, d)))
    h.learning, h.tier_mode = False, "strong"
    p = h.personas[PERSONA]
    p.tools, p.can_request = [], []
    if hasattr(p, "mcp"):
        p.mcp = []
    timer = threading.Timer(limit, h.cancel)
    timer.start()
    t = time.time()
    err = None
    try:
        r = h.turn(task["prompt"], force=PERSONA)
    except Exception as e:  # noqa: BLE001 - a crash is a result
        r, err = None, f"{type(e).__name__}: {e}"
    finally:
        timer.cancel()
        unwrap()
        ctxwin.server_props = real_props
        h.wait_learning()
    usage = ((r.usage or {}).get("total", {}) if r else {})
    compacts = [d for k, d in events if k == "compact"]
    ctx_events = [d for k, d in events if k == "context"]
    c = calls.rows
    ratios = [x["real"] / x["est"] for x in c if x["est"] and x["real"]]
    win = next((d.get("window") for d in ctx_events if d.get("window")), None)
    return {"answer": (r.answer[-1500:] if r else ""), "seconds": round(time.time() - t, 1),
            "input": usage.get("input", 0), "output": usage.get("output", 0), "thinking": usage.get("thinking", 0),
            "timed_out": bool(r and r.cancelled), "error": err or (calls.errors[-1] if calls.errors else None),
            "call_errors": len(calls.errors), "steps": max([d.get("step", 0) for k, d in events if k == "code"] or [0]),
            "bounces": sum(1 for k, d in events if k == "verify" and d.get("bounced")),
            "compactions": len(compacts),
            "compact_events": [{k: d.get(k) for k in ("step", "tokens_before", "tokens_after", "full", "notes", "dropped")}
                               for d in compacts],
            "limit": ctx_events[-1]["limit"] if ctx_events else None, "window": win,
            "peak_real": max([x["real"] for x in c] or [0]), "peak_est": max([x["est"] for x in c] or [0]),
            "real_over_est": round(statistics.median(ratios), 3) if ratios else None,
            "prefill_s": round(sum(max(0.0, x["secs"] - x["decode"]) for x in c), 1),
            "decode_s": round(sum(x["decode"] for x in c), 1), "calls": len(c),
            "calls_log": c, **rereads(events, task["marks"])}


def grade(task: dict, ws: Path) -> dict:
    """The hidden tests against the worker's policy.py. It is looked for where the prompt said (the task folder), then in
    the worker's own folder and its artifacts: a worker that wrote it with a relative path is not wrong about the rules."""
    import serve_compare as sc
    if not (ws / "policy.py").exists():
        found = sorted(ws.glob("sessions/*/*/policy.py")) + sorted(ws.glob("sessions/*/artifacts/policy.py"))
        if found:
            shutil.copy(found[-1], ws / "policy.py")
    return sc.grade_coding({"id": task["id"], "tests": task["tests"]}, ws)


def stall_stopped(ws: Path) -> bool:
    """Did the progress check end the run (two 'same attempt repeating' flags, then the harness's wrap-up message)?
    Read from the saved conversation, so it works on runs made before this was recorded."""
    for cp in ws.glob("sessions/*/*/checkpoint.json"):
        try:
            msgs = json.loads(cp.read_text(encoding="utf-8")).get("messages", [])
        except (OSError, ValueError):
            continue
        if any("The harness is stopping this task" in str(m.get("content", "")) for m in msgs):
            return True
    return False


def run_item(item: dict, a) -> dict:
    import long_tasks
    vname, seed, rep = item["variant"], item["seed"], item["rep"]
    ws = RUNS / "ws" / f"{vname}__s{seed}__r{rep}"
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True)
    task = long_tasks.make_task(seed, ws)
    task["prompt"] = task["prompt"].replace(str(ws), str(ws).replace("\\", "/"))
    row = {"variant": vname, "seed": seed, "rep": rep, "window_told": a.window or None}
    try:
        row.update(run_turn(task, VARIANTS[vname], ws, a.window, a.limit))
        g = grade(task, ws)
        row.update(passed=g["passed"], total=g["total"], correct=g["all_passed"], tail=g["tail"],
                   stall_stopped=stall_stopped(ws))
    except Exception as e:  # noqa: BLE001
        row.update(error=f"{type(e).__name__}: {e}", correct=False, passed=0, total=80, seconds=0, steps=0, compactions=0,
                   reread=0, reread_after_compact=0, timed_out=False, output=0, input=0, thinking=0, bounces=0)
    return row


# -- headroom: how full can a prompt be --------------------------------------------------------------------------------

def count_tokens(llm, msgs: list[dict]) -> int:
    """The server's own token count for a prompt (llama-server POST /tokenize on the chat text; close enough to the
    chat template's count to size a prompt by). The harness itself estimates 3.2 chars a token."""
    import urllib.request
    text = "\n".join(m["content"] for m in msgs)
    root = llm.base_url[:-3] if llm.base_url.endswith("/v1") else llm.base_url
    req = urllib.request.Request(root + "/tokenize", data=json.dumps({"content": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    return len(json.load(urllib.request.urlopen(req, timeout=60))["tokens"]) + 8 * len(msgs)


def run_headroom(a) -> None:
    """Fill the prompt to a share of the real window with realistic work history, then ask for one step that needs
    long thinking (the rulebook's reasoning, unlimited): does the reply fit, how long does the prompt take to read, and
    does a cached prompt start (what a normal step after a normal step sees) differ from a cold one (what the step
    after a compaction sees, because compaction rewrites the start of the prompt)?"""
    from jevh import context as ctxwin
    from jevh.config import load_config
    from jevh.llm import make_llm
    import long_tasks
    cfg = load_config()
    llm = make_llm({**cfg.models["strong"], "think": "unlimited"}, cfg.providers)   # as the persona's own settings do
    win = (llm.window().tokens if llm.window() else None) or 76000
    props = ctxwin.server_props(cfg.providers["llamacpp"]["base_url"], fresh=True)
    print(f"window {win:,} (server says {props and props['context']:,}); max_tokens {llm.max_tokens}")
    tmp = RUNS / "headroom"
    shutil.rmtree(tmp, ignore_errors=True)
    task = long_tasks.make_task(5, tmp)
    pages = [(p.name, p.read_text()) for p in sorted((tmp / "rules").glob("*.md"))]
    system = "You are a careful programmer working in a Python REPL. Answer with the next step."
    ask = ("Write out, rule by rule, which of these 18 rules a PRO customer in the EAST region with 12 items, a fragile "
           "express order of 61000 cents and coupon SAVE10 hits, and the total after each, then give the final result.")
    rows = []
    for share in a.shares:
        target = int(win * share)
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": task["prompt"]}]
        i = 0
        while count_tokens(llm, msgs) < target - 1500:
            name, text = pages[i % len(pages)]
            msgs.append({"role": "assistant", "content": f"Reading {name} (pass {i // len(pages) + 1})."})
            msgs.append({"role": "user", "content": f"Output:\n[{name}, pass {i // len(pages) + 1}]\n{text}"})
            i += 1
        msgs.append({"role": "assistant", "content": "I have read the pages."})
        msgs.append({"role": "user", "content": ask})
        for temp in ("cold", "warm"):
            t = time.time()
            try:
                r = llm.chat(msgs)
                row = {"share": share, "run": temp, "prompt": r.input_tokens, "out": r.output_tokens, "think": r.thinking_tokens,
                       "secs": round(time.time() - t, 1), "decode": r.decode_seconds, "ok": bool(r.text.strip()),
                       "cut": "output cut off" in r.text, "error": None}
            except Exception as e:  # noqa: BLE001
                row = {"share": share, "run": temp, "error": f"{type(e).__name__}: {str(e)[:160]}", "ok": False}
            rows.append(row)
            print(row, flush=True)
            if row.get("error"):
                break
            msgs[-2] = {"role": "assistant", "content": "I have read the pages. "}   # keep the prefix; (warm = same prefix)
    out = RUNS / "headroom_rows.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nsaved {out}")


# -- report ----------------------------------------------------------------------------------------------------------

def load_rows() -> list[dict]:
    return [json.loads(line) for line in ROWS.read_text(encoding="utf-8").splitlines()] if ROWS.exists() else []


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else float("nan")


def report(rows: list[dict] | None = None) -> str:
    rows = load_rows() if rows is None else rows
    if not rows:
        return "no results yet"
    names = [v for v in VARIANTS if any(r["variant"] == v for r in rows)] + \
            sorted({r["variant"] for r in rows} - set(VARIANTS))
    by = {v: [r for r in rows if r["variant"] == v] for v in names}
    L = [f"long_horizon: {len(rows)} runs ({', '.join(f'{v} {len(by[v])}' for v in names)}); window told to the harness: "
         f"{sorted({r.get('window_told') or 'real' for r in rows}, key=str)}\n"]
    head = (f"{'variant':12}{'runs':>5}{'solved':>8}{'tests%':>8}{'compacts':>9}{'reread':>7}{'after':>6}{'steps':>7}{'min':>6}"
            f"{'out k':>7}{'prefill s':>10}{'peak%win':>9}{'real/est':>9}{'dropped':>8}{'stall':>6}{'err':>5}{'t/o':>5}")
    L += [head, "-" * len(head)]
    for v in names:
        rs = by[v]
        win = [r["peak_real"] / r["window"] for r in rs if r.get("window") and r.get("peak_real")]
        drop = [max([e.get("dropped") or 0 for e in r.get("compact_events", [])] or [0]) for r in rs]
        L.append(f"{v:12}{len(rs):>5}{sum(r['correct'] for r in rs):>4}/{len(rs):<3}"
                 f"{mean([100 * r['passed'] / max(1, r['total']) for r in rs]):>7.0f}%"
                 f"{mean([r['compactions'] for r in rs]):>9.1f}{mean([r['reread'] for r in rs]):>7.1f}"
                 f"{mean([r['reread_after_compact'] for r in rs]):>6.1f}{mean([r['steps'] for r in rs]):>7.1f}"
                 f"{mean([r['seconds'] for r in rs]) / 60:>6.1f}{mean([r['output'] for r in rs]) / 1000:>7.1f}"
                 f"{mean([r.get('prefill_s') for r in rs]):>10.0f}{100 * mean(win):>8.0f}%"
                 f"{mean([r.get('real_over_est') for r in rs]):>9.2f}{mean(drop):>8.1f}"
                 f"{sum(bool(r.get('stall_stopped')) for r in rs):>6}"
                 f"{sum(bool(r.get('error')) for r in rs):>5}{sum(bool(r.get('timed_out')) for r in rs):>5}")
    L += ["", "per run (RIGHT = every hidden test passed):"]
    for r in sorted(rows, key=lambda r: (r["seed"], r["rep"], r["variant"])):
        ce = ", ".join(f"{e['tokens_before']}->{e['tokens_after']}" for e in r.get("compact_events", [])[:5])
        L.append(f"  {r['variant']:10} s{r['seed']} r{r['rep']}  {'RIGHT' if r['correct'] else 'wrong'} {r['passed']}/{r['total']}"
                 f"  {r['steps']:>3} steps {r['seconds'] / 60:5.1f} min  compactions {r['compactions']} [{ce}]"
                 f"  reread {r['reread']} (after compaction {r['reread_after_compact']})"
                 f"{'  ERR ' + str(r['error'])[:80] if r.get('error') else ''}{'  TIMEOUT' if r.get('timed_out') else ''}")
    L += ["", "reread = a rule page opened again after the worker had read it; after = of those, the ones after the first",
          "compaction. real/est = the server's token count over the harness's estimate (3.2 chars a token); prefill s =",
          "time the server spent reading prompts. peak%win = the largest real prompt over the window the harness was told."]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["e2e", "headroom"], default="e2e")
    ap.add_argument("--variants", default="base,at80,at80_all", help="comma list of: " + ", ".join(VARIANTS))
    ap.add_argument("--seeds", default=None, help="comma list (default: dev seeds; --holdout: the held-out ones)")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--window", type=int, default=0, help="tell the harness the window is this big (0: ask the server)")
    ap.add_argument("--limit", type=float, default=40 * 60, help="seconds per run")
    ap.add_argument("--shares", type=lambda s: [float(x) for x in s.split(",")], default=[0.5, 0.6, 0.7, 0.75, 0.8, 0.9])
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--report", "--offline", action="store_true")
    ap.add_argument("--regrade", action="store_true", help="grade every saved run's folder again (a grader fix) and update the rows")
    a = ap.parse_args()
    if a.regrade:
        import long_tasks
        rows = load_rows()
        for r in rows:
            ws = RUNS / "ws" / f"{r['variant']}__s{r['seed']}__r{r['rep']}"
            if ws.exists():
                g = grade(long_tasks.make_task(r["seed"], ws), ws)
                r.update(passed=g["passed"], total=g["total"], correct=g["all_passed"], tail=g["tail"],
                         stall_stopped=stall_stopped(ws))
        ROWS.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        print(report(rows))
        return
    if a.report:
        print(report())
        return
    if a.mode == "headroom":
        run_headroom(a)
        return
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else (HOLD_SEEDS if a.holdout else DEV_SEEDS)
    names = [v for v in a.variants.split(",") if v]
    # rotate the variant order per (seed, rep), so a stopped run is balanced and no variant always runs first/last
    items = []
    for rep in range(1, a.reps + 1):
        for si, seed in enumerate(seeds):
            order = names[(si + rep) % len(names):] + names[:(si + rep) % len(names)]
            items += [{"variant": v, "seed": seed, "rep": rep} for v in order]
    done = {(r["variant"], r["seed"], r["rep"]) for r in load_rows()}
    print(f"{len(items)} runs ({len(names)} variants x {len(seeds)} seeds x {a.reps} reps); window told: {a.window or 'real'}")
    if a.dry:
        for i in items:
            print(f"  {i['variant']:10} seed {i['seed']} rep {i['rep']}{'  (done)' if (i['variant'], i['seed'], i['rep']) in done else ''}")
        return
    RUNS.mkdir(parents=True, exist_ok=True)
    for n, item in enumerate(items):
        if (item["variant"], item["seed"], item["rep"]) in done:
            continue
        row = run_item(item, a)
        with open(ROWS, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{n + 1}/{len(items)}] {item['variant']:10} s{item['seed']} r{item['rep']} "
              f"{'RIGHT' if row['correct'] else 'wrong'} {row['passed']}/{row['total']} {row['steps']} steps "
              f"{row['seconds'] / 60:.1f} min compactions {row['compactions']} reread {row['reread']}"
              f"{' ERR ' + str(row['error'])[:100] if row.get('error') else ''}", flush=True)
    rows = load_rows()
    RAW.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    SUMMARY.write_text(report(rows), encoding="utf-8")
    print("\n" + report(rows))


if __name__ == "__main__":
    main()
