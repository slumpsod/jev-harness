"""Live eval: does Jev give the worker the `devices` tool for the right requests - and only those - and does adding the tool
change what it picks for everything else?

It asks Jev through the harness's own code (`Jev.pick_tools`, the `tools` threshold from config.yaml), with the daily
assistant's tools, the way `Harness._pick_tools` builds them: the devices description says which computers are connected
(CONNECTED below, as if two were online).

    python evals/devices_cases.py                  # the cases, old roster vs new, 3 repeats each
    python evals/devices_cases.py --holdout        # the held-out cases only (written apart from the description; don't tune on them)
    python evals/devices_cases.py --regression     # also every case of evals/tool_cases.py (what already worked), old vs new
    python evals/devices_cases.py --offline        # re-sweep the saved raw scores (thresholds 0.2-0.9), no Jev needed
    python evals/devices_cases.py --repeat 5

Designed for the semif Jev backend (the adapter at config jev.url, 4B System One model); any backend works.
Raw scores: evals/devices_cases_raw.json (or _holdout_raw.json); summary printed and saved as evals/devices_cases_summary.txt.

A case is right when every needed tool is picked and nothing outside needed+either is. For `devices` what matters most:
- fire: requests about another computer by name or as "my laptop / other PC" must pick devices;
- lookalikes: requests about laptops in general, this computer's own files and apps, websites, or the user's accounts must
  NOT pick devices (a needless guide costs ~600 tokens of every such prompt, and pulls the worker toward the wrong computer);
- old vs new: on the same requests, the other tools' picks should not move when devices is added.
The feature is wrong if the fire cases fall under ~80% or the lookalikes pick devices more than ~10% of the time.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevh.config import load_config  # noqa: E402
from jevh.decide import SystemOneClient  # noqa: E402
from jevh.jev import Jev  # noqa: E402
from jevh.tools import TOOLS  # noqa: E402

HERE = Path(__file__).resolve().parent
RAW = HERE / "devices_cases_raw.json"
RAW_HOLDOUT = HERE / "devices_cases_holdout_raw.json"
SUMMARY = HERE / "devices_cases_summary.txt"
CONNECTED = ["bilbo", "studio-pc"]
W, B, DK, DV, N, NR = "web", "browser", "desktop", "devices", "nango", "newsroom"
OLD = [W, B, DK, N, NR]            # daily_assistant before the change
NEW = [W, B, DK, DV, N, NR]        # ... and after

# (request, needed, either, kind): kind = fire | lookalike | borderline
CASES = [
    # fire: another computer, by name or as "my laptop / other PC"
    ("What's in the Downloads folder on bilbo?", {DV}, set(), "fire"),
    ("Copy report.pdf from my laptop's Documents to this PC.", {DV}, set(), "fire"),
    ("Run git status in ~/code/site on studio-pc and tell me what changed.", {DV}, set(), "fire"),
    ("Take a screenshot of bilbo's screen.", {DV}, set(), "fire"),
    ("Is Spotify open on my laptop?", {DV}, {DK}, "fire"),
    ("Find the tax PDF I saved on my other computer last week.", {DV}, set(), "fire"),
    ("How much free disk space does bilbo have?", {DV}, set(), "fire"),
    ("On studio-pc, open Chrome and check whether my site loads.", {DV}, {B, W}, "fire"),
    ("Send the zip in bilbo's Downloads to studio-pc's Desktop.", {DV}, set(), "fire"),
    ("Close the error dialog that's showing on my laptop.", {DV}, {DK}, "fire"),
    ("Which of my computers are online right now?", {DV}, set(), "fire"),
    ("Restart the dev server running in a terminal on studio-pc.", {DV}, set(), "fire"),
    ("Read the notes.txt on bilbo's desktop to me.", {DV}, set(), "fire"),
    ("On my laptop, log into the Hydro One site and download my latest bill.", {DV}, {B, W}, "fire"),
    # lookalikes: laptops in general, this computer, websites, accounts
    ("Which laptop should I buy for programming under $1500?", {W}, {B}, "lookalike"),
    ("My laptop's battery drains fast. What can I do about it?", set(), {W}, "lookalike"),
    ("What's in my Downloads folder?", {DK}, {DV}, "borderline"),
    ("Open Notepad and write down my shopping list.", {DK}, set(), "lookalike"),
    ("Take a screenshot of example.com.", {B}, set(), "lookalike"),
    ("Compare the MacBook Air and the Dell XPS 13.", {W}, {B}, "lookalike"),
    ("Write a short poem about my old computer.", set(), set(), "lookalike"),
    ("Check my email for anything from my accountant.", {N}, set(), "lookalike"),
    ("How do I set up remote desktop between two Windows PCs?", set(), {W}, "lookalike"),
    ("Explain how file sync works in Dropbox.", set(), {W}, "lookalike"),
    ("Which apps do I have open right now?", {DK}, {DV}, "borderline"),
    ("Keep an eye on news about the new Surface laptops and tell me when one launches.", {NR}, {W}, "lookalike"),
    # borderline
    ("Open the file I was working on yesterday.", set(), {DK, DV}, "borderline"),
    ("Back up my photos to the other machine.", {DV}, set(), "borderline"),
]
# written after the description was settled, apart from it; run with --holdout and do not tune on them
HOLDOUT = [
    ("Grab the latest invoice from bilbo and summarize it.", {DV}, set(), "fire"),
    ("On my other PC, what Windows version is installed?", {DV}, set(), "fire"),
    ("Delete the temp files in studio-pc's Downloads older than a month.", {DV}, set(), "fire"),
    ("Is the build finished on my laptop? It was running in PowerShell.", {DV}, {DK}, "fire"),
    ("Put this CSV on my laptop's desktop.", {DV}, set(), "fire"),
    ("Pause the music playing on studio-pc.", {DV}, {DK}, "fire"),
    ("Is a gaming laptop worth it for video editing?", set(), {W}, "lookalike"),
    ("Rename draft.txt on my Desktop to final.txt.", {DK}, {DV}, "borderline"),
    ("Find hotels in Denver for Friday on booking.com.", {B}, {W}, "lookalike"),
    ("What's on my calendar tomorrow?", {N}, set(), "lookalike"),
    ("How do I share a folder between my Mac and my PC?", set(), {W}, "lookalike"),
    ("Click Next in the installer window that's open.", {DK}, set(), "lookalike"),
]


def roster(names: list[str]) -> dict[str, str]:
    own = {t: TOOLS[t].description for t in names}
    if DV in own:   # (as Harness._pick_tools says it)
        own[DV] = own[DV].replace("{connected}", ", ".join(CONNECTED))
    return own


def right(need: set, either: set, picked: set) -> bool:
    return need <= picked <= need | either


def run_live(cases, repeat: int, regression: bool) -> dict:
    cfg = load_config()
    jev = Jev(cfg, SystemOneClient(cfg.jev["url"], cfg.jev.get("backend", "semif"), timeout=300))
    out = {"threshold": cfg.t("tool"), "connected": CONNECTED, "cases": [], "regression": []}
    t0 = time.time()
    for req, need, either, kind in cases:
        row = {"request": req, "need": sorted(need), "either": sorted(either), "kind": kind, "old": [], "new": []}
        for _ in range(repeat):
            row["old"].append(jev.pick_tools(req, roster(OLD)).scores)
            row["new"].append(jev.pick_tools(req, roster(NEW)).scores)
        out["cases"].append(row)
        print(f"  {kind:10} {req[:70]:70} devices p={statistics.mean(s.get(DV, 0) for s in row['new']):.2f}", flush=True)
    if regression:   # what already worked: evals/tool_cases.py, restricted to the daily assistant's tools
        from tool_cases import CASES as TOOL_CASES
        keep = set(OLD)
        for req, need, either in TOOL_CASES:
            if not (need | either) <= keep | {"project", "comfy"}:
                continue
            row = {"request": req, "need": sorted(need & keep), "either": sorted(either & keep), "old": [], "new": []}
            for _ in range(repeat):
                row["old"].append(jev.pick_tools(req, roster(OLD)).scores)
                row["new"].append(jev.pick_tools(req, roster(NEW)).scores)
            out["regression"].append(row)
    out["seconds"] = round(time.time() - t0, 1)
    out["jev_calls"] = jev.client.calls
    return out


def sweep(raw: dict, t: float) -> list[str]:
    lines = []
    for part in ("cases", "regression"):
        rows = raw.get(part) or []
        if not rows:
            continue
        lines.append(f"\n== {part} at threshold {t:.2f} ({len(rows)} requests x {len(rows[0]['new'])} repeats)")
        totals = {"old": [0, 0], "new": [0, 0]}
        by_kind: dict[str, list[int]] = {}
        moved = fp = 0
        failing = []
        for r in rows:
            need, either = set(r["need"]), set(r["either"])
            for arm in ("old", "new"):
                for s in r[arm]:
                    picked = {k for k, p in s.items() if p >= t}
                    if arm == "old":   # the old roster cannot pick devices: a fire case is judged on the rest
                        ok = right(need - {DV}, either | {DV}, picked) if DV in need else right(need, either, picked)
                    else:
                        ok = right(need, either, picked)
                    totals[arm][0] += ok
                    totals[arm][1] += 1
                    if arm == "new":
                        k = r.get("kind", "regression")
                        by_kind.setdefault(k, [0, 0])
                        by_kind[k][0] += ok
                        by_kind[k][1] += 1
                        fp += DV in picked and DV not in need | either
            for so, sn in zip(r["old"], r["new"]):
                moved += {k for k, p in so.items() if p >= t} != {k for k, p in sn.items() if p >= t and k != DV}
            rate = sum(right(need, either, {k for k, p in s.items() if p >= t}) for s in r["new"]) / len(r["new"])
            if rate < 1:
                mean = {k: round(statistics.mean(s.get(k, 0) for s in r["new"]), 2) for k in r["new"][0]}
                failing.append(f"  {rate:4.0%} right  {r['request'][:64]:64} need={r['need']} either={r['either']} {mean}")
        for arm in ("old", "new"):
            lines.append(f"  {arm}: {totals[arm][0]}/{totals[arm][1]} right ({totals[arm][0] / totals[arm][1]:.0%})")
        for k, (ok, n) in sorted(by_kind.items()):
            lines.append(f"    new, {k}: {ok}/{n} ({ok / n:.0%})")
        lines.append(f"  devices picked where it was not wanted: {fp}; other tools' picks changed by adding devices: {moved}")
        # the decision this change added, on its own (the totals above also count Jev's older habit of adding desktop and
        # nango to file requests, which is the same with or without devices)
        fire = [s.get(DV, 0) >= t for r in rows if DV in r["need"] for s in r["new"]]
        others = [s.get(DV, 0) >= t for r in rows if DV not in r["need"] + r["either"] for s in r["new"]]
        if fire:
            lines.append(f"  devices recall (requests that need it): {sum(fire)}/{len(fire)} ({sum(fire) / len(fire):.0%})")
        if others:
            lines.append(f"  devices false picks (requests that must not get it): {sum(others)}/{len(others)}")
        if failing:
            lines.append("  not always right (new):")
            lines += failing
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="re-sweep the saved raw scores")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--regression", action="store_true", help="also evals/tool_cases.py's cases, old vs new")
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    path = RAW_HOLDOUT if a.holdout else RAW
    if a.offline:
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        sys.path.insert(0, str(HERE))
        raw = run_live(HOLDOUT if a.holdout else CASES, a.repeat, a.regression)
        path.write_text(json.dumps(raw, indent=1), encoding="utf-8")
    lines = [f"devices tool pick ({'held-out' if a.holdout else 'cases'}); config threshold {raw['threshold']}; "
             f"{raw.get('jev_calls', '?')} Jev calls, {raw.get('seconds', '?')} s"]
    lines += sweep(raw, raw["threshold"])
    lines.append("\nthreshold sweep (new roster, cases): " + ", ".join(
        f"{t:.1f}: {sum(right(set(r['need']), set(r['either']), {k for k, p in s.items() if p >= t}) for r in raw['cases'] for s in r['new'])}"
        f"/{sum(len(r['new']) for r in raw['cases'])}" for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)))
    text = "\n".join(lines)
    print(text)
    if not a.offline:
        SUMMARY.with_name(SUMMARY.stem + ("_holdout" if a.holdout else "") + ".txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
