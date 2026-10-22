"""Swapping Jev's model: the harness's own decision evals, old backend against new, on live models.

Built for replacing semif (openjev/semif_bridge.py, Qwen3.5-4B, ~9 GB of GPU) with laya (openjev/laya_bridge.py, a
421M Laya checkpoint fine-tuned on semif's answers to harness-shaped decisions, ~2 GB of GPU or the CPU). Both run behind
the adapter at config jev.url (openjev/systemone.py), which picks the bridge by the request's model.

    python evals/bench/system_one_swap.py                    # all stages, old semif vs new laya
    python evals/bench/system_one_swap.py --stage evals      # every Jev-only eval (evals/*_cases.py) on each backend
    python evals/bench/system_one_swap.py --stage replay     # every request the evals sent the old backend, sent to the
                                                             #   new one: does it decide the same?
    python evals/bench/system_one_swap.py --stage latency --repeat 3
    python evals/bench/system_one_swap.py --holdout          # replay only the held-out half (see below)
    python evals/bench/system_one_swap.py --offline          # re-sweep the saved raw results, no models needed
    python evals/bench/system_one_swap.py --old semif --new laya --evals routing_cases,tier_cases

Needs: the adapter (openjev/systemone.py) with both bridges up: semif_bridge.py (:33003) and laya_bridge.py (:33004,
LAYA_CHECKPOINT = the fine-tuned checkpoint). The evals stage takes ~25 min on semif and ~10 min on laya.

What it measures
- evals: each eval's own summary, old and new side by side (the rows whose counts differ are printed). The evals are
  labelled: this is accuracy, with the thresholds config.yaml ships (tuned on semif).
- replay: agreement with the old backend on the same requests: yes/no answers on the same side of 0.5, mean |dP|, the
  same choice leader, mean total-variation distance, score distance. Split: a request is held out (--holdout) when its
  eval is a *_holdout set or its user request hashes odd (the split the laya checkpoint was trained on: only the even
  half and generated look-alikes of it were used to train it, so --holdout is the honest number).
- latency: seconds per request, one request at a time, on a fixed sample, --repeat times.

The new backend is wrong for the harness if, on --holdout, yes/no agreement falls under ~0.85 or choice agreement under
~0.70, or if any eval's shipped-threshold row loses more than a case or two that semif got right.
Status: UNVERIFIED as a whole - written and run 2026-10-21 on 3 evals (tier, learn, failure) end to end; the full
45-eval comparison in openjev/README.md was made with the research scripts it was built from, not with this file.
No committed raw results yet: a full run with --save-committed makes the reference.
Results: system_one_swap_raw.json and system_one_swap_summary.txt (a live run writes to workspace/evals/<backend>/bench/;
see evals/_out.py), the evals' own files under that folder's swap_<backend>/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVALS = HERE.parent
ROOT = EVALS.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EVALS))

import _out  # noqa: E402  (evals/_out.py: where a run's results go)
from jevh.config import load_config  # noqa: E402

RAW = HERE / "system_one_swap_raw.json"
SUMMARY = HERE / "system_one_swap_summary.txt"

# the evals that only ask Jev (no worker model, no browser): (script, args)
EVALS_JEV = [
    ("routing_cases", ["--sweep"]), ("skill_cases", []), ("memory_cases", []), ("tier_cases", []),
    ("tool_cases", []), ("tool_cases", ["--holdout"]), ("followup_cases", []), ("failure_cases", []),
    ("learn_cases", []), ("compact_cases", []), ("compact_cases", ["--holdout"]),
    ("expect_cases", []), ("expect_cases", ["--holdout"]), ("risk_cases", []), ("risk_cases", ["--holdout"]),
    ("severity_cases", []), ("severity_cases", ["--holdout"]), ("rule_cases", []), ("rule_cases", ["--holdout"]),
    ("search_cases", []), ("search_cases", ["--holdout"]), ("thinking_watch_cases", []),
    ("thinking_watch_cases", ["--holdout"]), ("verify_cases", []), ("verify_cases", ["--holdout"]),
    ("verify_evidence_cases", []), ("progress_cases", []), ("interject_cases", []), ("interject_cases", ["--holdout"]),
    ("persona_cases", []), ("persona_cases", ["--holdout"]), ("recall_cases", []), ("hook_route_cases", []),
    ("button_intent_cases", []), ("button_intent_cases", ["--holdout"]), ("desktop_cases", []),
    ("desktop_cases", ["--holdout"]), ("devices_cases", []), ("devices_cases", ["--holdout"]),
    ("session_category_cases", []), ("session_category_cases", ["--holdout"]), ("newsroom_cases", []),
    ("newsroom_cases", ["--holdout"]), ("find_cases", []), ("find_cases", ["--holdout"]),
]
CASE_KEYS = ("latest_request", "user_message", "request", "task", "goal", "message", "request_a", "new_fact")


def run_name(ev: str, args: list[str]) -> str:
    return ev + "".join("_" + a.strip("-") for a in args)


# ---------------------------------------------------------------- recording proxy (captures what the evals ask)

class Recorder:
    """A local HTTP proxy in front of the adapter: forwards every POST /v1/systemone and keeps request + answers."""

    def __init__(self, target: str):
        self.target, self.tag, self.rows, self.lock = target.rstrip("/"), "", [], threading.Lock()
        rec = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _fwd(self, method: str):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) if method == "POST" else None
                req = urllib.request.Request(rec.target + self.path, data=body, method=method,
                                             headers={"Content-Type": "application/json"})
                t0 = time.time()
                try:
                    with urllib.request.urlopen(req, timeout=600) as r:
                        status, data = r.status, r.read()
                except urllib.error.HTTPError as e:
                    status, data = e.code, e.read()
                secs = time.time() - t0
                if method == "POST" and status == 200 and self.path.startswith("/v1/systemone"):
                    q, a = json.loads(body), json.loads(data)
                    with rec.lock:
                        rec.rows.append({"tag": rec.tag, "state": q.get("state"), "questions": q.get("questions"),
                                         "answers": a.get("answers"), "secs": round(secs, 4)})
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                self._fwd("POST")

            def do_GET(self):
                self._fwd("GET")

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"


def run_evals(backend: str, url: str, out_dir: Path, names: set[str], recorder: Recorder | None) -> dict:
    home = Path(tempfile.mkdtemp(prefix=f"swap_{backend}_"))
    (home / "config.yaml").write_text(f"jev:\n  url: {recorder.url if recorder else url}\n  backend: {backend}\n")
    env = dict(os.environ, JEVH_HOME=str(home), JEVH_EVAL_OUT=str(out_dir), PYTHONIOENCODING="utf-8",
               JEVH_JEV_URL=recorder.url if recorder else url, JEVH_JEV_BACKEND=backend)
    logs = out_dir / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    status = {}
    for ev, args in EVALS_JEV:
        n = run_name(ev, args)
        if names and ev not in names and n not in names:
            continue
        if recorder:
            recorder.tag = n
        t0 = time.time()
        with open(logs / f"{n}.log", "w", encoding="utf-8") as f:
            try:
                rc = subprocess.run([sys.executable, str(EVALS / f"{ev}.py"), *args], cwd=ROOT, env=env, stdout=f,
                                    stderr=subprocess.STDOUT, timeout=3600).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        status[n] = {"rc": rc, "secs": round(time.time() - t0, 1)}
        print(f"  {backend:8} {n:36} rc={rc} {status[n]['secs']}s", flush=True)
    return status


ROWS = re.compile(r"\d+/\d+|\d+%")


def summary_rows(path: Path) -> dict[str, str]:
    """An eval's summary lines that carry counts, keyed by their text with the counts taken out."""
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ROWS.search(line):
                out.setdefault(re.sub(r"\b\d+/\d+\b|\b\d+(?:\.\d+)?%", "#", line).strip(), line.strip())
    return out


# ---------------------------------------------------------------- replay

def held_out(row: dict) -> bool:
    if row["tag"].endswith("_holdout"):
        return True
    st = row["state"]
    key = json.dumps(st, sort_keys=True)[:400]
    if isinstance(st, dict):
        for k in CASE_KEYS:
            if st.get(k):
                key = json.dumps(st[k], sort_keys=True)[:400]
                break
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % 2 == 1


def ask(url: str, backend: str, row: dict) -> tuple[dict | None, float, str | None]:
    body = json.dumps({"state": row["state"], "questions": row["questions"], "model": backend}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["answers"], time.time() - t0, None
    except Exception as e:  # noqa: BLE001
        return None, time.time() - t0, f"{type(e).__name__}: {e}"[:200]


def agreement(rows: list[dict]) -> dict:
    m: dict[str, list] = {k: [] for k in ("noul_agree", "noul_dp", "choice_agree", "choice_tv", "score_d")}
    errors = 0
    for r in rows:
        if not r.get("new"):
            errors += 1
            continue
        for k, q in (r["questions"] or {}).items():
            a, b = (r["answers"] or {}).get(k), r["new"].get(k)
            if not a or not b:
                continue
            if q["type"] == "noul":
                m["noul_agree"].append((a["noul"] >= .5) == (b["noul"] >= .5))
                m["noul_dp"].append(abs(a["noul"] - b["noul"]))
            elif q["type"] == "choice":
                pa, pb = a["probabilities"], b["probabilities"]
                m["choice_agree"].append(max(pa, key=pa.get) == max(pb, key=pb.get))
                m["choice_tv"].append(.5 * sum(abs(pa.get(o, 0) - pb.get(o, 0)) for o in set(pa) | set(pb)))
            elif q["type"] == "score":
                m["score_d"].append(abs(a["score"] - b["score"]) / (len(q["criteria"]) - 1))
    out = {"requests": len(rows), "errors": errors}
    for k, v in m.items():
        out[k] = round(statistics.mean(v), 3) if v else None
    out.update(noul_n=len(m["noul_agree"]), choice_n=len(m["choice_agree"]), score_n=len(m["score_d"]))
    return out


def latency(url: str, backend: str, rows: list[dict], repeat: int) -> dict:
    for r in rows[:3]:
        ask(url, backend, r)
    secs = []
    for _ in range(repeat):
        for r in rows:
            ans, dt, err = ask(url, backend, r)
            if err is None:
                secs.append(dt)
    secs.sort()
    if not secs:
        return {"n": 0}
    return {"n": len(secs), "p50": round(secs[len(secs) // 2], 3), "p90": round(secs[int(len(secs) * .9)], 3),
            "mean": round(statistics.mean(secs), 3)}


def health(url: str) -> dict:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=5) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200]}


# ---------------------------------------------------------------- report

def report(raw: dict, holdout: bool) -> list[str]:
    old, new = raw["old"], raw["new"]
    lines = [f"System One swap: old {old} vs new {new}  ({raw.get('when', '')})"]
    rows = [r for r in raw.get("replay", []) if not holdout or held_out(r)]
    if rows:
        lines.append(f"\n== replay: {new} on the requests the evals sent {old} "
                     f"({'held-out half' if holdout else 'all'}: {len(rows)} requests)")
        cols = ("requests", "errors", "noul_agree", "noul_dp", "choice_agree", "choice_tv", "score_d")
        lines.append(f"  {'eval':36} " + " ".join(f"{c:>12}" for c in cols))
        tot = agreement(rows)
        lines.append(f"  {'ALL':36} " + " ".join(f"{str(tot[c]):>12}" for c in cols))
        for tag in sorted({r["tag"] for r in rows}):
            a = agreement([r for r in rows if r["tag"] == tag])
            lines.append(f"  {tag:36} " + " ".join(f"{str(a[c]):>12}" for c in cols))
        worst = sorted((r for r in rows if r.get("new")), key=lambda r: -max(
            [abs(r["answers"][k]["noul"] - r["new"][k]["noul"]) for k, q in r["questions"].items()
             if q["type"] == "noul" and k in r["answers"] and k in r["new"]] or [0]))[:8]
        lines.append("  largest yes/no disagreements:")
        for r in worst:
            st = json.dumps(r["state"], ensure_ascii=False)[:90]
            ds = {k: (round(r["answers"][k]["noul"], 2), round(r["new"][k]["noul"], 2)) for k, q in r["questions"].items()
                  if q["type"] == "noul" and k in r["answers"] and k in r["new"]}
            lines.append(f"    {r['tag'][:24]:24} {st}  {dict(list(ds.items())[:3])}")
    if raw.get("latency"):
        lines.append("\n== latency (seconds per request, one at a time)")
        for b, v in raw["latency"].items():
            lines.append(f"  {b:8} {v}")
    if raw.get("health"):
        lines.append("\n== bridges (adapter /health)")
        lines.append(f"  {json.dumps(raw['health'])[:400]}")
    if raw.get("evals"):
        lines.append(f"\n== evals: each eval's own summary rows that differ (old {old} / new {new})")
        for n, d in raw["evals"].items():
            diff = d.get("diff", [])
            lines.append(f"  {n}: rc old={d['rc'].get(old)} new={d['rc'].get(new)}, {len(diff)} of {d['rows']} rows differ")
            for o, nw in diff[:12]:
                lines.append(f"     old: {o[:140]}\n     new: {nw[:140]}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old", default="semif")
    ap.add_argument("--new", default="laya")
    ap.add_argument("--stage", default="all", choices=("all", "evals", "replay", "latency"))
    ap.add_argument("--evals", default="", help="only these evals (comma list of script names)")
    ap.add_argument("--holdout", action="store_true", help="report the held-out half of the replay")
    ap.add_argument("--repeat", type=int, default=2, help="latency passes")
    ap.add_argument("--latency-n", type=int, default=100)
    ap.add_argument("--offline", action="store_true")
    _out.add_args(ap)
    a = ap.parse_args()
    out = _out.Out.of(a)
    if a.offline:
        raw = json.loads(out.read(RAW).read_text(encoding="utf-8"))
        print("\n".join(report(raw, a.holdout)))
        return
    cfg = load_config()
    url = cfg.jev.get("url", "http://127.0.0.1:33001")
    names = set(filter(None, a.evals.split(",")))
    path = out.write(RAW)
    raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if raw.get("old") not in (None, a.old) or raw.get("new") not in (None, a.new):
        raw = {}
    raw.update(old=a.old, new=a.new, when=time.strftime("%Y-%m-%d %H:%M"), health=health(url))
    if a.stage in ("all", "evals"):
        rec = Recorder(url)
        base = path.parent
        status = {a.old: run_evals(a.old, url, base / f"swap_{a.old}", names, rec),
                  a.new: run_evals(a.new, url, base / f"swap_{a.new}", names, None)}
        raw["replay"] = [r for r in raw.get("replay", []) if names and r["tag"] not in status[a.old]] + rec.rows
        raw["evals"] = {}
        for n in status[a.old]:
            ro = summary_rows(base / f"swap_{a.old}" / "_logs" / f"{n}.log")
            rn = summary_rows(base / f"swap_{a.new}" / "_logs" / f"{n}.log")
            raw["evals"][n] = {"rc": {a.old: status[a.old][n]["rc"], a.new: status[a.new].get(n, {}).get("rc")},
                               "rows": len(ro), "diff": [(ro[k], rn.get(k, "(missing)")) for k in ro if ro[k] != rn.get(k)]}
    if a.stage in ("all", "evals", "replay"):
        todo = [r for r in raw.get("replay", []) if "new" not in r]
        print(f"replay: {len(todo)} requests to {a.new}", flush=True)
        for i, r in enumerate(todo):
            r["new"], r["new_secs"], r["new_error"] = ask(url, a.new, r)
            if i % 200 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    if a.stage in ("all", "latency"):
        rows = [r for r in raw.get("replay", []) if held_out(r)]
        rows = rows[:: max(1, len(rows) // a.latency_n)][: a.latency_n]
        if rows:
            raw["latency"] = {b: latency(url, b, rows, a.repeat) for b in (a.old, a.new)}
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    lines = report(raw, a.holdout)
    out.write(SUMMARY).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved {out.where(path)}")


if __name__ == "__main__":
    main()
