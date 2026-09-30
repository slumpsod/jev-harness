"""Old vs new: completion (the checkers), time and tokens, per model and split; the verdict; and
the report a human approves or rejects from.

Verdict (thresholds in config refine:):
  improves  failing cases (dev + holdout) pass clearly more often (>= min_gain), the held-out cases
            don't do worse, and no model loses more than control_slack control runs
  mixed     it helps overall, but a held-out split or some model's controls got worse
  no_gain   anything else
Time and tokens don't decide it (quality first); a rise past cost_warn is listed as a caveat."""
from __future__ import annotations

from math import comb
from typing import Any

SPLITS = ("dev", "holdout", "control")


def stats(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    ok = [r for r in rows if r.get("passed")]
    avg = (lambda k: round(sum(float(r.get(k) or 0) for r in rows) / n, 1) if n else 0.0)
    return {"n": n, "passed": len(ok), "rate": (len(ok) / n) if n else 0.0, "seconds": avg("seconds"),
            "tokens": round(sum(float(r.get("input") or 0) + float(r.get("output") or 0) for r in rows) / n) if n else 0,
            "thinking": avg("thinking"), "errors": sum(1 for r in rows if r.get("error"))}


def table(rows: list[dict], version: str) -> dict[str, dict[str, dict]]:
    """{model: {split: stats}} for one version, plus model "all"."""
    mine = [r for r in rows if r["version"] == version]
    out: dict[str, dict[str, dict]] = {}
    for model in sorted({r["model"] for r in mine}) + ["all"]:
        sel = mine if model == "all" else [r for r in mine if r["model"] == model]
        out[model] = {s: stats([r for r in sel if r["split"] == s]) for s in SPLITS}
        out[model]["target"] = stats([r for r in sel if r["split"] in ("dev", "holdout")])
    return out


def p_better(old_pass: int, old_n: int, new_pass: int, new_n: int) -> float:
    """One-sided Fisher exact test: P(new passes at least this much more often by chance)."""
    total, passes = old_n + new_n, old_pass + new_pass
    if not old_n or not new_n:
        return 1.0
    denom = comb(total, passes)
    return min(1.0, sum(comb(new_n, k) * comb(old_n, passes - k)
                        for k in range(new_pass, min(new_n, passes) + 1) if passes - k <= old_n) / denom)


def verdict(rows: list[dict], old: str, new: str, settings: dict | None = None) -> dict[str, Any]:
    s = settings or {}
    min_gain, slack, cost_warn = float(s.get("min_gain", 0.15)), int(s.get("control_slack", 1)), \
        float(s.get("cost_warn", 0.25))
    a, b = table(rows, old), table(rows, new)
    models = [m for m in b if m != "all" and m in a]
    ta, tb = a.get("all", {}).get("target", stats([])), b.get("all", {}).get("target", stats([]))
    gain = tb["rate"] - ta["rate"]
    hold_a, hold_b = a.get("all", {}).get("holdout", stats([])), b.get("all", {}).get("holdout", stats([]))
    caveats, regress = [], []
    for m in models:
        ca, cb = a[m]["control"], b[m]["control"]
        if ca["passed"] - cb["passed"] > slack:
            regress.append(f"{m}: controls {ca['passed']}/{ca['n']} -> {cb['passed']}/{cb['n']}")
        ma, mb = a[m]["target"], b[m]["target"]
        if mb["rate"] < ma["rate"]:
            caveats.append(f"{m} did worse on the failing cases: {ma['passed']}/{ma['n']} -> {mb['passed']}/{mb['n']}")
        for k, label in (("seconds", "time"), ("tokens", "tokens")):
            if ma[k] and (mb[k] - ma[k]) / ma[k] > cost_warn:
                caveats.append(f"{m}: {label} per run up {100 * (mb[k] - ma[k]) / ma[k]:.0f}% "
                               f"({ma[k]:g} -> {mb[k]:g})")
    holdout_worse = hold_b["n"] and hold_b["rate"] < hold_a["rate"]
    if gain >= min_gain and not holdout_worse and not regress:
        v = "improves"
    elif gain > 0:
        v = "mixed"
    else:
        v = "no_gain"
    if holdout_worse:
        caveats.append(f"held-out cases did worse: {hold_a['passed']}/{hold_a['n']} -> {hold_b['passed']}/{hold_b['n']}")
    caveats += [f"controls got worse on {r}" for r in regress]
    p = p_better(ta["passed"], ta["n"], tb["passed"], tb["n"])
    if p > 0.1 and gain > 0:
        caveats.append(f"the gain could be chance: p = {p:.2f} with {tb['n']} runs per version")
    if tb["n"] < 20:
        caveats.append(f"only {tb['n']} runs of the failing cases per version")
    if hold_b["n"] == 0:
        caveats.append("no held-out cases were run")
    errs = sum(b[m]["target"]["errors"] + b[m]["control"]["errors"] for m in models)
    if errs:
        caveats.append(f"{errs} new-version runs ended in a run error (crash or timeout)")
    return {"verdict": v, "gain": round(gain, 3), "p": round(p, 3), "caveats": caveats, "old": a, "new": b}


def _cell(s: dict) -> str:
    return f"{s['passed']}/{s['n']}" if s["n"] else "-"


def _delta(a: float, b: float) -> str:
    if not a:
        return f"{b:g}"
    return f"{a:g} → {b:g} ({100 * (b - a) / a:+.0f}%)"


def render(state: dict, v: dict, cases: list, fix: dict, extra_caveats: list[str] | None = None) -> str:
    """The approval report (markdown)."""
    f = state["failure"]
    icon = {"improves": "✅ Improves", "mixed": "⚠️ Mixed", "no_gain": "❌ No improvement"}[v["verdict"]]
    ta, tb = v["old"]["all"]["target"], v["new"]["all"]["target"]
    out = [f"# Refinement {state['id']}: {f['kind']} ({f.get('persona') or '-'})", "",
           f"**Verdict: {icon}.** Failing cases passed {_cell(ta)} → {_cell(tb)} "
           f"({100 * ta['rate']:.0f}% → {100 * tb['rate']:.0f}%, p = {v['p']}).", ""]
    out += ["## The failure that caught it", "",
            f"- #{f['id']} **{f['kind']}** ({f.get('source', '')}): {f['summary']}",
            f"- Request: {f['request'][:500]}",
            f"- Persona {f.get('persona') or '-'}, model {f.get('model') or '-'}, logged "
            f"{state.get('group_size', 1)} time(s) like this (ids: {', '.join(map(str, state.get('group', [f['id']])))})"]
    rep = state.get("repro")
    if rep:
        out.append(f"- Reproduced {rep['failed']}/{rep['runs']} times on {rep['model']} before the fix")
    out += ["", "## What changed", "", fix.get("summary") or "(the fixer gave no summary)", "",
            f"Files: {', '.join(fix.get('files', [])) or '-'}" + (f" · commit {fix['commit'][:10]}" if fix.get("commit") else "")
            + f" · branch `{state.get('branch', '')}`", ""]
    diff = fix.get("diff", "")
    if diff:
        out += ["<details><summary>Diff</summary>", "", "```diff", diff[:12000] + ("\n[... cut]" if len(diff) > 12000 else ""),
                "```", "</details>", ""]
    out += ["## How much it improved", "",
            "| model | split | passed old → new | time per run (s) | tokens per run | thinking |",
            "|---|---|---|---|---|---|"]
    for m in [m for m in v["new"] if m in v["old"]]:
        for s in ("dev", "holdout", "control"):
            a, b = v["old"][m][s], v["new"][m][s]
            if a["n"] or b["n"]:
                out.append(f"| {m} | {s} | {_cell(a)} → {_cell(b)} | {_delta(a['seconds'], b['seconds'])} | "
                           f"{_delta(a['tokens'], b['tokens'])} | {_delta(a['thinking'], b['thinking'])} |")
    caveats = v["caveats"] + list(extra_caveats or [])
    out += ["", "## Caveats", ""] + ([f"- {c}" for c in caveats] or ["- none found"])
    by = {c.split: 0 for c in cases}
    for c in cases:
        by[c.split] += 1
    out += ["", "## How it was tested", "",
            f"- Cases: {', '.join(f'{n} {s}' for s, n in by.items())}; {state.get('runs', 1)} run(s) of each per model "
            f"and version. The fixer saw only the dev cases.",
            f"- Tests written by {state.get('generator_model', '-')}; fix by {state.get('fixer_model', '-')}"
            + (f" (try {state.get('try', 1)})" if state.get("try") else "") + ".",
            f"- Dropped test cases (checker didn't tell good from bad): {state.get('dropped', 0)}",
            f"- Models: {', '.join(state.get('models', []))}", "",
            f"Approve: `python -m jevh.refine approve {state['id']}` · Reject: "
            f"`python -m jevh.refine reject {state['id']} \"why\"`"]
    return "\n".join(out) + "\n"
