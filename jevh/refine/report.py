"""Old vs new: completion (the checkers), cost (time, tokens, model calls, gate bounces), per model
and split; the verdict; and the report a human approves or rejects from.

The failing cases are the dev and held-out cases that failed at least once on the old code: a variant
that never did is a regression check (split "regress", counted with the controls), and one that failed
every run everywhere is set aside as a suspect checker (split "suspect", out of the verdict).

Verdict (thresholds in config refine:), for a failure where the task wasn't done (objective "pass"):
  improves  the failing cases pass clearly more often (>= min_gain), held-out cases don't do worse, and
            no model loses more than control_slack control + regression runs
  mixed     it helps overall, but a held-out split or some model's controls got worse
  no_gain   anything else
For a failure where the task got done with needless extra work (objective "cost", e.g. a correct answer
the final() gate sent back): the same, with gate bounces per run on the failing cases falling by at least
min_cost_gain (a share) in place of the pass rate rising, and the pass rate not falling.
Otherwise time and tokens don't decide it (quality first); a rise past cost_warn is listed as a caveat."""
from __future__ import annotations

from math import comb
from typing import Any

from .checks import describe

SPLITS = ("dev", "holdout", "control", "regress")
GUARDS = ("control", "regress")   # a fix must not make these worse


def stats(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    ok = [r for r in rows if r.get("passed")]
    avg = (lambda k: round(sum(float(r.get(k) or 0) for r in rows) / n, 2) if n else 0.0)
    return {"n": n, "passed": len(ok), "rate": (len(ok) / n) if n else 0.0, "seconds": avg("seconds"),
            "tokens": round(sum(float(r.get("input") or 0) + float(r.get("output") or 0) for r in rows) / n) if n else 0,
            "thinking": avg("thinking"), "calls": avg("calls"), "bounces": avg("bounces"),
            "errors": sum(1 for r in rows if r.get("error"))}


def table(rows: list[dict], version: str, splits: dict[str, str] | None = None) -> dict[str, dict[str, dict]]:
    """{model: {split: stats}} for one version, plus model "all". `splits`: each case's split now (a row
    keeps the one its case had when it ran)."""
    split = (lambda r: (splits or {}).get(r["case"], r["split"]))
    mine = [r for r in rows if r["version"] == version]
    out: dict[str, dict[str, dict]] = {}
    for model in sorted({r["model"] for r in mine}) + ["all"]:
        sel = mine if model == "all" else [r for r in mine if r["model"] == model]
        out[model] = {s: stats([r for r in sel if split(r) == s]) for s in SPLITS}
        out[model]["target"] = stats([r for r in sel if split(r) in ("dev", "holdout")])
        out[model]["guard"] = stats([r for r in sel if split(r) in GUARDS])
    return out


def p_better(old_pass: int, old_n: int, new_pass: int, new_n: int) -> float:
    """One-sided Fisher exact test: P(new passes at least this much more often by chance)."""
    total, passes = old_n + new_n, old_pass + new_pass
    if not old_n or not new_n:
        return 1.0
    denom = comb(total, passes)
    return min(1.0, sum(comb(new_n, k) * comb(old_n, passes - k)
                        for k in range(new_pass, min(new_n, passes) + 1) if passes - k <= old_n) / denom)


def verdict(rows: list[dict], old: str, new: str, settings: dict | None = None, splits: dict[str, str] | None = None,
            objective: str = "pass") -> dict[str, Any]:
    s = settings or {}
    min_gain, slack, cost_warn = float(s.get("min_gain", 0.15)), int(s.get("control_slack", 1)),         float(s.get("cost_warn", 0.25))
    a, b = table(rows, old, splits), table(rows, new, splits)
    models = [m for m in b if m != "all" and m in a]
    ta, tb = a.get("all", {}).get("target", stats([])), b.get("all", {}).get("target", stats([]))
    hold_a, hold_b = a.get("all", {}).get("holdout", stats([])), b.get("all", {}).get("holdout", stats([]))
    caveats, regress = [], []
    for m in models:
        ga, gb = a[m]["guard"], b[m]["guard"]
        if ga["passed"] - gb["passed"] > slack:
            regress.append(f"{m}: controls and regression checks {ga['passed']}/{ga['n']} -> {gb['passed']}/{gb['n']}")
        ma, mb = a[m]["target"], b[m]["target"]
        if mb["rate"] < ma["rate"]:
            caveats.append(f"{m} did worse on the failing cases: {ma['passed']}/{ma['n']} -> {mb['passed']}/{mb['n']}")
        for k, label in (("seconds", "time"), ("tokens", "tokens")):
            if ma[k] and (mb[k] - ma[k]) / ma[k] > cost_warn:
                caveats.append(f"{m}: {label} per run up {100 * (mb[k] - ma[k]) / ma[k]:.0f}% "
                               f"({ma[k]:g} -> {mb[k]:g})")
    if objective == "cost":
        # needless extra work: bounces per run on the failing cases, as a share of the old
        gain = (ta["bounces"] - tb["bounces"]) / ta["bounces"] if ta["bounces"] else 0.0
        holdout_worse = hold_b["n"] and hold_b["bounces"] > hold_a["bounces"]
        need, done_worse = float(s.get("min_cost_gain", 0.5)), tb["rate"] < ta["rate"]
        if done_worse:
            caveats.append(f"the failing cases were done less often: {ta['passed']}/{ta['n']} -> {tb['passed']}/{tb['n']}")
    else:
        gain = tb["rate"] - ta["rate"]
        holdout_worse = hold_b["n"] and hold_b["rate"] < hold_a["rate"]
        need, done_worse = min_gain, False
    if gain >= need and not holdout_worse and not regress and not done_worse:
        v = "improves"
    elif gain > 0:
        v = "mixed"
    else:
        v = "no_gain"
    if holdout_worse:
        caveats.append(f"held-out cases did worse: {hold_a['passed']}/{hold_a['n']} passed, {hold_a['bounces']:g} "
                       f"bounces per run -> {hold_b['passed']}/{hold_b['n']}, {hold_b['bounces']:g}")
    caveats += [f"got worse on {r}" for r in regress]
    p = p_better(ta["passed"], ta["n"], tb["passed"], tb["n"])
    if objective == "pass" and p > 0.1 and gain > 0:
        caveats.append(f"the gain could be chance: p = {p:.2f} with {tb['n']} runs per version")
    if tb["n"] < 20:
        caveats.append(f"only {tb['n']} runs of the failing cases per version")
    if ta["n"] == 0:
        caveats.append("no case failed on the old code: there was nothing for a fix to improve")
    if hold_b["n"] == 0:
        caveats.append("no held-out case failed on the old code, so none shows whether the fix generalizes")
    errs = sum(b[m]["target"]["errors"] + b[m]["guard"]["errors"] for m in models)
    if errs:
        caveats.append(f"{errs} new-version runs ended in a run error (crash or timeout)")
    return {"verdict": v, "gain": round(gain, 3), "p": round(p, 3), "caveats": caveats, "old": a, "new": b,
            "objective": objective}


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
    if v.get("objective") == "cost":
        head = (f"Gate bounces per run on the failing cases {ta['bounces']:g} → {tb['bounces']:g} "
                f"({100 * v['gain']:.0f}% fewer); they were done {_cell(ta)} → {_cell(tb)}.")
    else:
        head = (f"Failing cases passed {_cell(ta)} → {_cell(tb)} "
                f"({100 * ta['rate']:.0f}% → {100 * tb['rate']:.0f}%, p = {v['p']}).")
    out = [f"# Refinement {state['id']}: {f['kind']} ({f.get('persona') or '-'})", "",
           f"**Verdict: {icon}.** {head}", ""]
    out += ["## The failure that caught it", "",
            f"- #{f['id']} **{f['kind']}** ({f.get('source', '')}): {f['summary']}",
            f"- Request: {f['request'][:500]}",
            f"- Persona {f.get('persona') or '-'}, model {f.get('model') or '-'}, logged "
            f"{state.get('group_size', 1)} time(s) like this (ids: {', '.join(map(str, state.get('group', [f['id']])))})"]
    rep = state.get("repro")
    if rep:
        out.append("- Replayed before the fix, failed: " + ", ".join(
            f"{m} {r['failed']}/{r['runs']}" for m, r in rep.items()))
    tri = state.get("triage")
    if tri:
        out += ["", "## Why it's the harness's to fix (the local model's reading)", "",
                f"- **{tri.get('category')}**" + (f", {tri['component']}" if tri.get("component") else "")
                + f": {tri.get('why', '')}",
                f"- Evidence: {str(tri.get('evidence', ''))[:600]}", f"- Decision: {tri.get('decision', '')}"]
    out += ["", "## What changed", "", fix.get("summary") or "(the fixer gave no summary)", "",
            f"Files: {', '.join(fix.get('files', [])) or '-'}" + (f" · commit {fix['commit'][:10]}" if fix.get("commit") else "")
            + f" · branch `{state.get('branch', '')}`", ""]
    diff = fix.get("diff", "")
    if diff:
        out += ["<details><summary>Diff</summary>", "", "```diff", diff[:12000] + ("\n[... cut]" if len(diff) > 12000 else ""),
                "```", "</details>", ""]
    out += ["## How much it improved", "",
            "| model | split | passed old → new | bounces per run | model calls per run | time per run (s) "
            "| tokens per run |", "|---|---|---|---|---|---|---|"]
    for m in [m for m in v["new"] if m in v["old"]]:
        for s in ("dev", "holdout", "regress", "control"):
            a, b = v["old"][m][s], v["new"][m][s]
            if a["n"] or b["n"]:
                out.append(f"| {m} | {s} | {_cell(a)} → {_cell(b)} | {a['bounces']:g} → {b['bounces']:g} | "
                           f"{a['calls']:g} → {b['calls']:g} | {_delta(a['seconds'], b['seconds'])} | "
                           f"{_delta(a['tokens'], b['tokens'])} |")
    caveats = v["caveats"] + list(extra_caveats or [])
    out += ["", "## Caveats", ""] + ([f"- {c}" for c in caveats] or ["- none found"])
    by = {c.split: 0 for c in cases}
    for c in cases:
        by[c.split] += 1
    moved = state.get("moved") or {}
    if moved.get("suspect"):
        sus = [c for c in cases if c.id in moved["suspect"]]
        review = state.get("checker_review") or {}
        out += ["", "## Set aside: checkers that failed every run", "",
                "These failed on every run on the old code, and the local model judged their checker wrong for a "
                "result it looked at; they are not in the verdict. Worth a look:"]
        out += [f"- {c.id}: {c.request[:160]} (checks: {describe(c.checker)[:160]}) - {review.get(c.id, '')}"
                for c in sus]
    out += ["", "## How it was tested", "",
            f"- Cases: {', '.join(f'{n} {s}' for s, n in by.items())}; {state.get('runs', 1)} run(s) of each per model "
            f"and version. The fixer saw only the dev cases.",
            f"- Tests written by {state.get('generator_model', '-')}; fix by {state.get('fixer_model', '-')}"
            + (f" (try {state.get('try', 1)})" if state.get("try") else "") + ".",
            f"- Dropped test cases (checker didn't tell good from bad): {state.get('dropped', 0)}",
            f"- Regression checks (variants that never failed on the old code): {len(moved.get('regress') or [])}",
            f"- Models: {', '.join(state.get('models', []))}", "",
            f"Approve: `python -m jevh.refine approve {state['id']}` · Reject: "
            f"`python -m jevh.refine reject {state['id']} \"why\"`"]
    return "\n".join(out) + "\n"
