"""Live bench for persona creation (jevh/forge.py): real Jev, real drafting model, real routing, real turns.

The offline tests (tests/test_forge.py) use a fake Jev and a scripted writer, so they show the code runs.
This shows whether the feature WORKS: does Jev notice a request no assistant is for, does the model write a
sound, narrow persona, does the routing gate keep it from stealing other personas' work, does the draft then
win its own domain, and is everything that already worked still routed as before.

    python evals/bench/persona_creation.py                        # every stage, both splits, 3 reps
    python evals/bench/persona_creation.py --stage trigger,regress
    python evals/bench/persona_creation.py --split held           # only the held-out domains
    python evals/bench/persona_creation.py --reps 5 --model lmstudio:ornith-1.5-35b-a3b
    python evals/bench/persona_creation.py --report               # the last run's summary

Needs: the Jev adapter (config jev.url) and LM Studio with the default persona's model. Nothing touches your
real workspace, personas or skills: every scenario runs in runs/persona_creation/<stamp>/<scenario>/ on a
copy of personas/. The dev domains are the ones the thresholds were first looked at; the held domains were
written afterwards, so a result that holds on both is not just fitted to the cases.

Stages
  decisions  the three Jev questions on labelled cases (evals/persona_cases.py), per split
  trigger    the REAL route (confidence, no_fit threshold) on specialist and ordinary requests: which would be pooled
  forge      clusters of requests -> the drafting model writes a persona -> the routing gate -> checks, x reps.
             Also: a cluster an existing persona handles, scattered requests, approve / delete
  orphan     learned skills no persona wants -> a persona around the skill, x reps; skills with a home -> none
  regress    the drafts all live at once: do they win new requests in their domain, and does the routing of
             evals/routing_cases.py (the same 37 requests) stay as it was without them
  e2e        full harness turns: requests -> pooled by the real route -> drafted in the background -> the next
             request is routed to the draft and answered

Pass targets (first guesses, printed beside each result; a miss is a finding, not a bug in the bench):
the Jev checks are listed in each stage. If the feature is wrong, expect: no_fit never fires at the shipped
threshold, drafts that fail the fence checks, a created persona that steals requests from the existing ones,
or the routing set getting worse once drafts are live.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

import persona_cases as pc  # noqa: E402
import routing_cases as rc  # noqa: E402

from jevh.config import load_config  # noqa: E402
from jevh.forge import NAME_RE  # noqa: E402
from jevh.jev import Jev  # noqa: E402
from jevh.orchestrator import Harness  # noqa: E402
from jevh.personas import load_learned, load_personas  # noqa: E402
from jevh.skills import Skill, parse_skill  # noqa: E402

RUNS = HERE / "runs" / "persona_creation"

# --------------------------------------------------------------------------- scenarios
# name -> (split, [requests that make the persona: 3 are enough, a 4th spare], [new requests in the domain])
DOMAINS = {
    "guitar": ("dev", [
        "How do I intonate a classical guitar's bridge saddle?",
        "My acoustic's 12th fret harmonic is sharp of the fretted note, what do I adjust?",
        "What string gauge and action height should I set for a nylon-string guitar?",
        "How do I check and set the neck relief on an acoustic guitar?"], [
        "Which truss rod direction fixes fret buzz on the low strings?",
        "How do I compensate the saddle for a wound G string?",
        "What's the right nut slot depth on an acoustic?"]),
    "chess": ("dev", [
        "Explain the Sicilian Defense and which lines to study first",
        "How do I punish the London System as black?",
        "What's a good opening repertoire for a 1400-rated club player?",
        "Which endgames should I drill first to improve at chess?"], [
        "How do I play against the Caro-Kann as white?",
        "Explain the plans in the Queen's Gambit Declined.",
        "What is the Lucena position and how do I win it?"]),
    "petg": ("dev", [
        "Which slicer settings stop stringing in PETG on a Bambu A1?",
        "My first layer won't stick on a textured PEI plate with PETG",
        "What nozzle temperature and retraction should I use for PETG?",
        "How do I dry filament and how long does PETG need?"], [
        "Why does PETG blob on the corners of my prints?",
        "What cooling fan speed is right for PETG bridges?",
        "How do I stop PETG sticking too well to the PEI sheet?"]),
    "spanish": ("dev", [
        "Drill me on Spanish preterite vs imperfect with exercises",
        "What's the difference between ser and estar, with examples to practise?",
        "Quiz me on Spanish subjunctive triggers",
        "Give me a week of A2 Spanish speaking drills"], [
        "When do I use por versus para in Spanish?",
        "Correct this Spanish sentence and explain the mistake.",
        "Teach me the Spanish conditional tense with practice sentences."]),
    "tenancy": ("dev", [
        "Review this tenancy agreement clause by clause for risks",
        "Is a 60-day notice period normal in a residential lease?",
        "Can my landlord keep the whole deposit for normal wear and tear?",
        "What should I check in a lease before signing as a tenant?"], [
        "Is an automatic rent-increase clause enforceable?",
        "What does a joint-and-several liability clause mean for roommates?",
        "How do I dispute deductions from my security deposit?"]),
    "dahlia": ("held", [
        "How do I overwinter dahlia tubers in zone 5?",
        "When should I start tomatoes indoors in zone 5 and how do I harden them off?",
        "What soil mix and spacing works for a raised vegetable bed in a short-season garden?",
        "How do I save seed from heirloom beans?"], [
        "When do I lift dahlias after the first frost?",
        "How deep and how far apart do I plant garlic cloves in autumn?",
        "What is the best way to store gladiolus corms over winter?"]),
    "cricket": ("held", [
        "Explain cricket's LBW rule and how DRS works",
        "What is the Duckworth-Lewis-Stern method in a rain-affected match?",
        "How does a captain set a field for a left-arm spinner?",
        "What are the differences between T20, ODI and Test cricket rules?"], [
        "What counts as a no-ball in cricket?",
        "How does the powerplay work in a one-day international?",
        "Explain the follow-on rule in Test cricket."]),
    "jazz": ("held", [
        "Teach me ii-V-I chord voicings on piano",
        "What scales work over a minor 7b5 chord?",
        "How do I comp rootless voicings in a jazz trio?",
        "Show me how to reharmonize a blues with tritone substitutions"], [
        "What are shell voicings and when do I use them on piano?",
        "How do I voice an altered dominant chord?",
        "What's the difference between Dorian and Aeolian over a minor chord?"]),
    "sourdough": ("held", [
        "What's a good sourdough starter feeding schedule for a 75% hydration loaf?",
        "Why is my sourdough loaf dense with a tight crumb?",
        "How long should I bulk ferment sourdough at 24C?",
        "How do I shape a batard and score it?"], [
        "How do I know my starter is ready to bake with?",
        "What does the autolyse step do for sourdough?",
        "Why does my sourdough crust come out pale?"]),
}
# a cluster an existing persona already handles: it must not become a second fitness coach
FITS_EXISTING = [
    "Make me a 4 day upper/lower split for strength.",
    "How should I progress my bench press, I've stalled at 80kg?",
    "How much protein should I eat to build muscle at 85kg?"]
# three requests from three domains: never a cluster, nothing to draft
SCATTERED = ["Explain the Sicilian Defense and which lines to study first",
             "How do I overwinter dahlia tubers in zone 5?",
             "Which slicer settings stop stringing in PETG on a Bambu A1?"]

# turns that other personas handled before the drafts, so the gate has real requests to check for stealing
PRIOR = [(q, p) for q, p, _ in rc.CASES][::3][:12]

# what a created persona must not have (the fence), checked on every draft
FENCE = {"mcp": [], "can_request": [], "tools": []}

# --realistic-memory: in real use the request that started a persona (or the task a skill was learned from) is
# already in memory, handled by another persona, and the gate re-routes recent requests to look for stealing.
# Off, the bench seeds only unrelated history (which can hide a gate that rejects every real draft).
REALISTIC: list[bool] = [False]
SKILL_ORIGIN = {  # learned skill -> the kind of request it was learned from (a domain's first request)
    "learned-guitar-intonation": "guitar", "learned-chess-opening-study": "chess", "learned-petg-stringing-fix": "petg",
    "learned-tenancy-clause-review": "tenancy", "learned-dahlia-overwintering": "dahlia",
    "learned-sourdough-hydration": "sourdough", "learned-jazz-voicings": "jazz", "learned-cricket-dls-method": "cricket"}

LOG: list[str] = []
CACHE: list[Path] = []          # [dir]: each finished rep is kept here, so a stopped run can be resumed
DEADLINE: list[float] = []      # [time]: --budget; a rep that would start after it pauses the run


class Paused(Exception):
    """The time budget is used up: rerun with --resume <stamp> to carry on."""


def cached(key: str, fn):
    """The result of one rep: from the cache when it was already done, else run it (unless the budget is spent)."""
    f = CACHE[0] / f"{key}.json" if CACHE else None
    if f is not None and f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    if DEADLINE and time.time() > DEADLINE[0]:
        raise Paused(key)
    r = fn()
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
    return r


def say(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


# --------------------------------------------------------------------------- scaffolding
def make_cfg(where: Path, model: str | None = None):
    """A config on a throwaway workspace, a copy of personas/ and a fresh skills folder."""
    cfg = load_config()
    shutil.rmtree(where, ignore_errors=True)
    cfg.workspace = where / "ws"
    cfg.personas_dir = where / "personas"
    cfg.personas_dirs = [cfg.personas_dir]
    shutil.copytree(ROOT / "personas", cfg.personas_dir)
    cfg.skills_dirs = [where / "skills", ROOT / "skills"]
    (where / "skills").mkdir(parents=True, exist_ok=True)
    return cfg


class Rig:
    """One harness on a throwaway folder, with every Jev same_specialist score and event recorded."""

    def __init__(self, where: Path, model: str | None = None, seed: bool = True, thresholds: dict | None = None):
        self.cfg = make_cfg(where)
        self.cfg.thresholds = {**self.cfg.thresholds, **(thresholds or {})}
        self.events: list[tuple[str, dict]] = []
        self.h = Harness(self.cfg, on_event=lambda k, d: self.events.append((k, d)), model_override=model)
        self.jev, self.forge = self.h.jev, self.h.forge
        self.sims: list[list[float]] = []
        real = self.jev.same_specialist
        self.jev.same_specialist = lambda r, o: (self.sims.append(real(r, o)) or self.sims[-1])
        self.llm = self.h._llm(self.h.personas[self.cfg.default_persona])
        if seed:
            for q, p in PRIOR:
                self.h.memory.append(p, q, "ok", "seed")

    def personas(self, inactive: bool = False):
        return load_personas(self.cfg.personas_dirs, include_inactive=inactive)

    def skipped(self) -> list[str]:
        return [d.get("reason", "") for k, d in self.events if k == "persona_skipped"]


def check_draft(spec_path: str, skill: str | None = None) -> dict:
    """Programmatic checks on one written draft. Returns {check: ok} plus the text to read."""
    import yaml
    d = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    ok = {
        "name_valid": bool(NAME_RE.match(d["name"])),
        "description_20_500": 20 <= len(d["description"]) <= 500,
        "prompt_present": len(d["system_prompt"]) >= 20,
        "marked_draft": d.get("draft") is True,
        "no_mcp": d.get("mcp") == [], "no_can_request": d.get("can_request") == [],
        "no_tools": d.get("tools") == [],
        "only_its_skill": d.get("skills_allowed") == ([skill] if skill else []),
    }
    return {"checks": ok, "name": d["name"], "description": d["description"], "system_prompt": d["system_prompt"],
            "active": d.get("active", True)}


# --------------------------------------------------------------------------- stages
def stage_decisions(splits: list[str], out: Path) -> dict:
    res = {}
    for split in splits:
        held = split == "held"
        nf, sm, sk = (pc.NO_FIT_HELD, pc.SAME_HELD, pc.SKILLS_HELD) if held else (pc.NO_FIT, pc.SAME, pc.SKILLS)
        raw = pc.run_live(nf, sm, sk, out / f"decisions_{split}_raw.json")
        cfg = load_config()
        t = {k: cfg.t(k) for k in ("no_fit", "same_specialist", "orphan")}

        def rate(scores, labels, thr):
            tp = sum(s >= thr and l for s, l in zip(scores, labels))
            fp = sum(s >= thr and not l for s, l in zip(scores, labels))
            return tp, sum(labels), fp, len(labels) - sum(labels)
        nfs, nfl = raw["no_fit"], [c[1] for c in nf]
        sms, sml = raw["same"], [c[2] for c in sm]
        oks, okl = [r["none"] for r in raw["orphan"]], [c[2] for c in sk]
        res[split] = {
            "thresholds": t,
            "no_fit": rate(nfs, nfl, t["no_fit"]), "same_specialist": rate(sms, sml, t["same_specialist"]),
            "orphan": rate(oks, okl, t["orphan"]),
            "no_fit_best": max(((rate(nfs, nfl, x / 100), x / 100) for x in range(30, 91, 5)),
                               key=lambda r: (r[0][2] == 0, r[0][0] - 3 * r[0][2])),
            "no_fit_scores": {"positive": sorted(round(s, 2) for s, l in zip(nfs, nfl) if l),
                              "negative": sorted(round(s, 2) for s, l in zip(nfs, nfl) if not l)},
        }
    return res


def stage_trigger(splits: list[str], out: Path) -> dict:
    """The real route, as the harness runs it on a first message (no current persona): is the request
    pooled for a persona? Pooled = route confidence under route threshold AND no_fit >= the threshold."""
    cfg = load_config()
    jev = Jev(cfg)
    personas = load_personas(cfg.personas_dirs)
    res = {}
    for split in splits:
        cases = pc.NO_FIT_HELD if split == "held" else pc.NO_FIT
        rows = []
        for q, unfit in cases:
            r = jev.route_persona(q, personas, None, "")
            pooled = r.no_fit is not None and r.no_fit >= cfg.t("no_fit")
            rows.append({"request": q, "specialist": unfit, "routed": r.persona, "confidence": round(r.confidence, 2),
                         "asked_no_fit": r.no_fit is not None, "no_fit": None if r.no_fit is None else round(r.no_fit, 2),
                         "pooled": pooled})
        res[split] = rows
    (out / "trigger.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


ONLY: list[str] = []   # --domains: run only these (a quick look); empty = all


def _cluster_rep(out: Path, model, name: str, split: str, reqs: list, probes: list, rep: int) -> dict:
    rig = Rig(out / f"forge-{name}-r{rep}", model)
    if REALISTIC[0]:   # an earlier request in this domain that wasn't pooled, handled by the default persona
        rig.h.memory.append(rig.cfg.default_persona, reqs[3], "ok", "origin")
    t0 = time.time()
    made = None
    for q in reqs[:3]:
        made = rig.forge.note_no_fit(q, rig.llm, f"bench/{name}")
    secs = time.time() - t0
    row = {"domain": name, "split": split, "rep": rep, "seconds": round(secs), "skipped": rig.skipped(),
           "similarity": [[round(x, 2) for x in s] for s in rig.sims], "created": made is not None,
           "pending": len(rig.forge.pending())}
    if made is not None:
        row.update({"active": made.active, "gate": {"wins": made.gate.wins, "probes": made.gate.probes,
                    "steals": made.gate.steals, "checked": made.gate.checked, "note": made.gate.note}})
        row.update(check_draft(made.path))
        if made.active:   # does it take what it was made for, among the real personas?
            field = rig.personas()
            row["probe_wins"] = sum(rig.jev.pick_persona(p, field) == made.name for p in probes)
            row["probes"] = len(probes)
            row["spare_wins"] = rig.jev.pick_persona(reqs[3], field) == made.name
    say(f"  forge {name} r{rep}: " + (f"{made.name} active={made.active} wins {made.gate.wins}/{made.gate.probes} "
        f"steals {len(made.gate.steals)} probes {row.get('probe_wins')}/{len(probes)}" if made else
        f"NOT created (pooled {row['pending']}; similarity {row['similarity']}) {row['skipped']}") + f"  ({secs:.0f}s)")
    return row


def _negative_rep(out: Path, model, label: str, reqs: list, rep: int) -> dict:
    rig = Rig(out / f"forge-{label}-r{rep}", model)
    made = None
    for q in reqs:
        made = rig.forge.note_no_fit(q, rig.llm, f"bench/{label}") or made
    row = {"rep": rep, "created": made is not None, "name": made.name if made else None,
           "skipped": rig.skipped(), "pending": len(rig.forge.pending()),
           "similarity": [[round(x, 2) for x in s] for s in rig.sims]}
    if made:
        row.update(check_draft(made.path))
        row["gate"] = {"wins": made.gate.wins, "probes": made.gate.probes, "steals": made.gate.steals}
    say(f"  forge {label} r{rep}: " + (f"CREATED {made.name}" if made else
        f"nothing created, pending={row['pending']} {row['skipped']}"))
    return row


def _lifecycle(out: Path, model, clusters: list) -> dict | None:
    """approve / delete on the first active draft that was made."""
    first = next((r for r in clusters if r.get("active")), None)
    if not first:
        return None
    rig = Rig(out / "forge-lifecycle", model)
    made = None
    for q in DOMAINS[first["domain"]][1][:3]:
        made = rig.forge.note_no_fit(q, rig.llm, "bench/lifecycle")
    if not made:
        return None
    live = made.name in rig.personas()
    approved = rig.forge.approve(made.name)
    kept = not next((p for p in load_learned(rig.cfg.personas_dirs) if p.name == made.name), None).draft
    deleted = rig.forge.delete(made.name)
    lc = {"live_when_made": live, "approve": approved and kept, "delete": deleted and made.name not in rig.personas()}
    say(f"  lifecycle: {lc}")
    return lc


def stage_forge(splits: list[str], reps: int, out: Path, model: str | None) -> dict:
    res = {"clusters": [], "fits_existing": [], "scattered": [], "lifecycle": None}
    for name, (split, reqs, probes) in DOMAINS.items():
        if split not in splits or (ONLY and name not in ONLY):
            continue
        for rep in range(reps):
            res["clusters"].append(cached(f"forge-{name}-r{rep}", lambda n=name, sp=split, rq=reqs, pr=probes, r=rep:
                                          _cluster_rep(out, model, n, sp, rq, pr, r)))
    for label, reqs in (("fits_existing", FITS_EXISTING), ("scattered", SCATTERED)):
        for rep in range(max(1, reps - 1)):
            res[label].append(cached(f"forge-{label}-r{rep}", lambda lb=label, rq=reqs, r=rep:
                                     _negative_rep(out, model, lb, rq, r)))
    res["lifecycle"] = cached("forge-lifecycle", lambda: _lifecycle(out, model, res["clusters"]))
    (out / "forge.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def write_skill(cfg, name: str, desc: str) -> Path:
    d = cfg.skills_dirs[0] / "learned" / name
    d.mkdir(parents=True, exist_ok=True)
    f = d / "SKILL.md"
    f.write_text(f"---\nname: {name}\ndraft: true\ndescription: {desc}\n---\n\n# {name}\n\n"
                 f"## Steps\n1. {desc}\n2. Check the result against what was asked.\n", encoding="utf-8")
    return f


def _orphan_rep(out: Path, model, split: str, name: str, desc: str, orphan: bool, rep: int) -> dict:
    rig = Rig(out / f"orphan-{name}-r{rep}", model)
    if REALISTIC[0] and name in SKILL_ORIGIN:   # the task the skill was learned from, answered by the default persona
        rig.h.memory.append(rig.cfg.default_persona, DOMAINS[SKILL_ORIGIN[name]][1][0], "ok", "origin")
    path = write_skill(rig.cfg, name, desc)
    made = rig.forge.note_skill(parse_skill(path), rig.llm)
    row = {"skill": name, "split": split, "should_create": orphan, "rep": rep, "created": made is not None,
           "skipped": rig.skipped()}
    if made:
        row.update(check_draft(made.path, name))
        row["active"] = made.active
        row["owned"] = parse_skill(path).owner == made.name if made.active else None
        rig.forge.delete(made.name)
        row["delete_restores"] = parse_skill(path).owner == ""
    say(f"  orphan {name} r{rep}: " + (f"created {made.name} active={made.active} owned={row['owned']}"
        if made else f"none {row['skipped']}") + ("" if bool(made) == orphan else "   <-- WRONG"))
    return row


def stage_orphan(splits: list[str], reps: int, out: Path, model: str | None) -> list:
    res = []
    for split in splits:
        for name, desc, orphan in (pc.SKILLS_HELD if split == "held" else pc.SKILLS):
            for rep in range(reps if orphan else 1):
                res.append(cached(f"orphan-{name}-r{rep}", lambda sp=split, n=name, d=desc, o=orphan, r=rep:
                                  _orphan_rep(out, model, sp, n, d, o, r)))
    (out / "orphan.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def stage_regress(forge_res: dict, out: Path) -> dict:
    """All the drafts made in the forge stage live at once, against the routing set and every request that
    belongs to an existing persona."""
    cfg = load_config()
    jev = Jev(cfg)
    base = load_personas(cfg.personas_dirs)
    drafts = {}
    seen = set()
    for r in forge_res.get("clusters", []):
        if r.get("active") and r["domain"] not in seen:
            seen.add(r["domain"])
            drafts[r["name"]] = r
    from jevh.personas import Persona
    withd = dict(base)
    for n, r in drafts.items():
        b = base[cfg.default_persona]
        withd[n] = Persona(name=n, description=r["description"], system_prompt=r["system_prompt"],
                           model=copy.deepcopy(b.model), skills_allowed=[], draft=True)
    rows, stolen = [], []
    for q, want, _collab in rc.CASES:
        a = jev.route_persona(q, base, None, "").persona
        b = jev.route_persona(q, withd, None, "").persona
        rows.append({"request": q, "expected": want, "without": a, "with": b})
        if b in drafts:
            stolen.append((q, b))
    # the requests that are unfit-specialist in other domains must not land on the wrong draft either
    cross = []
    for dom, (split, reqs, probes) in DOMAINS.items():
        mine = [n for n, r in drafts.items() if r["domain"] == dom]
        for p in probes:
            got = jev.route_persona(p, withd, None, "").persona
            cross.append({"domain": dom, "request": p, "routed": got, "own": got in mine,
                          "wrong_draft": got in drafts and got not in mine, "has_draft": bool(mine)})
    res = {"drafts": list(drafts), "routing": rows, "stolen": stolen, "cross": cross,
           "top1_without": sum(r["without"] == r["expected"] for r in rows), "top1_with": sum(r["with"] == r["expected"] for r in rows),
           "n": len(rows)}
    (out / "regress.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


E2E_DOMAINS = ["chess"]


# the shipped thresholds, then the trigger forced (ask the no_fit question on every route, at the level the
# decisions stage found workable): shows whether the rest of the path works once something reaches the pool
E2E_VARIANTS = [("shipped", {}), ("trigger forced", {"no_fit": 0.45, "route": 1.01})]


def stage_e2e(out: Path, model: str | None, domains: list[str]) -> list[dict]:
    """Real turns. Requests are asked in order; the real route decides what is pooled; the third makes a persona in the
    background; the fourth is routed to it and answered."""
    res = []
    for (dom, (variant, thr)) in [(d, v) for d in domains for v in E2E_VARIANTS]:
        res.append(cached(f"e2e-{dom}-{variant.replace(' ', '_')}",
                          lambda dom=dom, variant=variant, thr=thr: _e2e_run(out, model, dom, variant, thr)))
    (out / "e2e.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def _e2e_run(out: Path, model, dom: str, variant: str, thr: dict) -> dict:
    if True:
        split, reqs, probes = DOMAINS[dom]
        rig = Rig(out / f"e2e-{dom}-{variant.replace(' ', '_')}", model, thresholds=thr)
        rig.h.learning = False                 # no memory/skill learning noise: only persona creation
        rows = []
        t0 = time.time()
        for i, q in enumerate(reqs[:3] + [probes[0]]):
            t = rig.h.turn(q)
            rig.h.wait_learning(600)
            ev = [d for k, d in reversed(rig.events) if k == "route"]
            dec = ev[0]["decision"] if ev else None
            rows.append({"turn": i + 1, "request": q, "persona": t.persona, "routed_reason": dec.reason if dec else "",
                         "no_fit": getattr(dec, "no_fit", None), "confidence": round(getattr(dec, "confidence", 0), 2),
                         "answer_chars": len(t.answer or ""), "seconds": round(t.seconds),
                         "drafts_after": [p.name for p in rig.forge.drafts()]})
            rig.h.reload_personas()
            say(f"  e2e {dom}/{variant} turn {i + 1}: persona={t.persona} no_fit={rows[-1]['no_fit']} drafts={rows[-1]['drafts_after']} "
                f"answer={rows[-1]['answer_chars']} chars ({rows[-1]['seconds']}s)")
        created = [p for p in rig.forge.drafts()]
        return {"domain": dom, "variant": variant, "turns": rows, "created": [p.name for p in created],
                "last_routed_to_draft": rows[-1]["persona"] in [p.name for p in created],
                "answered": rows[-1]["answer_chars"] > 0, "skipped": rig.skipped(),
                "pending": len(rig.forge.pending()), "seconds": round(time.time() - t0)}


# --------------------------------------------------------------------------- summary
def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def frac(a: int, b: int) -> str:
    return f"{a}/{b}" + (f" ({100 * a // b}%)" if b else "")


def summarize(r: dict) -> str:
    L: list[str] = []
    add = L.append
    add("PERSONA CREATION BENCH  " + r.get("stamp", "") + f"  model={r.get('model') or 'default persona model'}  reps={r.get('reps')}"
        + ("  [realistic memory: the originating request is already in memory]" if r.get("realistic_memory") else ""))
    if "decisions" in r:
        add("\n== decisions (labelled cases; thresholds are the config's)")
        for split, d in r["decisions"].items():
            t = d["thresholds"]
            for key in ("no_fit", "same_specialist", "orphan"):
                tp, pos, fp, neg = d[key]
                good = tp >= 0.7 * pos and fp == 0
                add(f"  [{split}] {key:<16} @{t[key]:.2f}: caught {frac(tp, pos)}, false positives {fp}/{neg}   "
                    f"[target: >=70% caught, 0 false]  {verdict(good)}")
            (btp, bpos, bfp, bneg), bthr = d["no_fit_best"]
            add(f"  [{split}] no_fit best threshold on these cases: {bthr:.2f} -> caught {btp}/{bpos}, false {bfp}/{bneg}")
            add(f"  [{split}] no_fit scores: specialist {d['no_fit_scores']['positive']}  ordinary {d['no_fit_scores']['negative']}")
    if "trigger" in r:
        add("\n== trigger (the real route: is the request pooled for a persona?)")
        for split, rows in r["trigger"].items():
            spec = [x for x in rows if x["specialist"]]
            ordy = [x for x in rows if not x["specialist"]]
            pooled_spec = sum(x["pooled"] for x in spec)
            pooled_ord = sum(x["pooled"] for x in ordy)
            asked = sum(x["asked_no_fit"] for x in spec)
            add(f"  [{split}] specialist requests pooled {frac(pooled_spec, len(spec))} (the no_fit question was even asked on "
                f"{frac(asked, len(spec))}); ordinary requests pooled {pooled_ord}/{len(ordy)}   "
                f"[target: >=50% pooled, 0 ordinary]  {verdict(pooled_spec * 2 >= len(spec) and pooled_ord == 0)}")
    if "forge" in r:
        f = r["forge"]
        add("\n== forge (drafting model + routing gate), per domain over reps")
        by: dict[str, list] = {}
        for row in f["clusters"]:
            by.setdefault(row["domain"], []).append(row)
        tot = {"n": 0, "created": 0, "active": 0, "fence": 0, "steals": 0, "probe": 0, "probes": 0, "secs": 0}
        for dom, rows in by.items():
            c = sum(x["created"] for x in rows)
            a = sum(bool(x.get("active")) for x in rows)
            pw = sum(x.get("probe_wins", 0) for x in rows)
            pn = sum(x.get("probes", 0) for x in rows)
            add(f"  {dom:<10}[{rows[0]['split']:<4}] created {c}/{len(rows)}, live {a}/{len(rows)}, new requests won "
                f"{pw}/{pn}   names: {', '.join(x.get('name', '-') for x in rows)}")
            for x in rows:
                tot["n"] += 1
                tot["created"] += x["created"]
                tot["active"] += bool(x.get("active"))
                tot["fence"] += all(x["checks"].values()) if x.get("checks") else 0
                tot["steals"] += len(x.get("gate", {}).get("steals", []))
                tot["probe"] += x.get("probe_wins", 0)
                tot["probes"] += x.get("probes", 0)
                tot["secs"] += x["seconds"]
        n = max(tot["n"], 1)
        add(f"  all clusters: created {frac(tot['created'], tot['n'])}, live {frac(tot['active'], tot['n'])} "
            f"[target >=80% live]  {verdict(tot['active'] * 5 >= tot['n'] * 4)}")
        add(f"  fence + validity checks pass on {frac(tot['fence'], tot['created'])} created drafts [target 100%]  "
            f"{verdict(tot['fence'] == tot['created'])}")
        add(f"  gate steals recorded: {tot['steals']} [target 0]  {verdict(tot['steals'] == 0)};  "
            f"new in-domain requests routed to the draft: {frac(tot['probe'], tot['probes'])} [target >=80%]  "
            f"{verdict(tot['probes'] * 4 <= tot['probe'] * 5)}")
        add(f"  mean time to draft one persona: {tot['secs'] // n}s (3 requests + the writer + the gate)")
        for label in ("fits_existing", "scattered"):
            rows = f[label]
            made = sum(x["created"] for x in rows)
            add(f"  {label}: persona created in {made}/{len(rows)} runs [target 0]  {verdict(made == 0)}"
                + ("".join(f"   -> {x['name']}" for x in rows if x["created"])))
        if f.get("lifecycle"):
            lc = f["lifecycle"]
            add(f"  lifecycle: live when made {lc['live_when_made']}, approve {lc['approve']}, delete {lc['delete']}  "
                f"{verdict(all(lc.values()))}")
        add("  drafts written (read these: name / description):")
        shown = set()
        for row in f["clusters"]:
            if row.get("name") and row["domain"] not in shown:
                shown.add(row["domain"])
                add(f"    {row['domain']}: {row['name']}: {row['description']}")
        bad = [(x["domain"], k) for x in f["clusters"] if x.get("checks") for k, v in x["checks"].items() if not v]
        if bad:
            add(f"  FAILED CHECKS: {bad}")
        miss = [(x["domain"], x["skipped"]) for x in f["clusters"] if not x["created"]]
        if miss:
            add(f"  not created: {miss}")
    if "orphan" in r:
        rows = r["orphan"]
        pos = [x for x in rows if x["should_create"]]
        neg = [x for x in rows if not x["should_create"]]
        mk = sum(x["created"] for x in pos)
        live = sum(bool(x.get("active")) for x in pos)
        add("\n== orphan (a learned skill no persona wants)")
        add(f"  orphans that got a persona {frac(mk, len(pos))}, and went LIVE {frac(live, len(pos))} [target >=75% live]; "
            f"skills with a home that got one {sum(x['created'] for x in neg)}/{len(neg)} [target 0]  "
            f"{verdict(live * 4 >= len(pos) * 3 and not any(x['created'] for x in neg))}")
        act = [x for x in pos if x.get("active")]
        add(f"  live drafts that own their skill {sum(bool(x.get('owned')) for x in act)}/{len(act)}; delete gives the skill "
            f"back {sum(bool(x.get('delete_restores')) for x in pos if x['created'])}/{sum(x['created'] for x in pos)}  "
            f"[target all]")
        bad = [(x["skill"], k) for x in pos if x.get("checks") for k, v in x["checks"].items() if not v]
        if bad:
            add(f"  FAILED CHECKS: {bad}")
    if "regress" in r:
        g = r["regress"]
        add("\n== regress (every draft live at once)")
        add(f"  routing_cases top-1 without drafts {frac(g['top1_without'], g['n'])}, with {len(g['drafts'])} drafts live "
            f"{frac(g['top1_with'], g['n'])}; requests an existing persona owned that a draft took: {len(g['stolen'])} "
            f"[target 0 and no drop]  {verdict(not g['stolen'] and g['top1_with'] >= g['top1_without'])}")
        for q, d in g["stolen"]:
            add(f"    STOLEN by {d}: {q}")
        own = [x for x in g["cross"] if x["has_draft"]]
        add(f"  new in-domain requests that land on their own draft with ALL drafts live: {frac(sum(x['own'] for x in own), len(own))}; "
            f"on a different draft: {sum(x['wrong_draft'] for x in g['cross'])} [target 0]")
    if "e2e" in r:
        add("\n== e2e (full turns)")
        for x in r["e2e"]:
            add(f"  {x['domain']} ({x['variant']}): draft created {x['created'] or 'NO'}; the 4th request routed to it: {x['last_routed_to_draft']}; "
                f"answered: {x['answered']}; pending in pool {x['pending']}; {x['seconds']}s total   "
                f"{verdict(bool(x['created']) and x['last_routed_to_draft'] and x['answered'])}")
            for t in x["turns"]:
                add(f"    turn {t['turn']}: -> {t['persona']} (conf {t['confidence']}, no_fit {t['no_fit']}) {t['routed_reason']}")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="decisions,trigger,forge,orphan,regress,e2e")
    ap.add_argument("--split", default="all", choices=["dev", "held", "all"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--model", help="put every persona on one model, e.g. lmstudio:ornith-1.5-35b-a3b")
    ap.add_argument("--domains", default="", help="only these domains in the forge stage, e.g. chess,petg (a quick look)")
    ap.add_argument("--e2e-domains", default=",".join(E2E_DOMAINS))
    ap.add_argument("--report", action="store_true", help="print the last run's summary (--resume STAMP: that run's, re-rendered)")
    ap.add_argument("--realistic-memory", action="store_true",
                    help="seed the request a persona or skill came from into memory, as real use has it (see REALISTIC)")
    ap.add_argument("--resume", metavar="STAMP", help="carry on a run that was paused or stopped (its finished reps are kept)")
    ap.add_argument("--budget", type=int, default=0, metavar="SECONDS",
                    help="pause after this long (finishing the rep in hand); --resume carries on. For a runner with a time cap")
    a = ap.parse_args()
    if a.report:
        if a.resume:   # re-render from the saved results (after a change to the summary)
            f = RUNS / a.resume / "results.json"
            res = json.loads(f.read_text(encoding="utf-8"))
            text = summarize(res) + f"\n\ntotal {res.get('seconds')}s; raw results in {f.parent}"
            (f.parent / "summary.txt").write_text(text, encoding="utf-8")
            print(text)
            return
        last = sorted(RUNS.glob("*/summary.txt"), key=lambda q: q.stat().st_mtime)
        print(last[-1].read_text(encoding="utf-8") if last else "no runs yet")
        return
    ONLY[:] = [d for d in a.domains.split(",") if d]
    REALISTIC[0] = a.realistic_memory
    stamp = a.resume or time.strftime("%Y%m%d-%H%M%S")
    out = RUNS / stamp
    out.mkdir(parents=True, exist_ok=True)
    CACHE[:] = [out / "cache"]
    if a.budget:
        DEADLINE[:] = [time.time() + a.budget]
    splits = ["dev", "held"] if a.split == "all" else [a.split]
    stages = a.stage.split(",")
    r: dict = {"stamp": stamp, "model": a.model, "reps": a.reps, "splits": splits, "realistic_memory": a.realistic_memory}
    t0 = time.time()
    try:
        if "decisions" in stages:
            say("decisions ...")
            r["decisions"] = cached("decisions", lambda: stage_decisions(splits, out))
        if "trigger" in stages:
            say("trigger ...")
            r["trigger"] = cached("trigger", lambda: stage_trigger(splits, out))
        if "forge" in stages or "regress" in stages:
            say("forge ...")
            r["forge"] = stage_forge(splits, a.reps, out, a.model)
        if "orphan" in stages:
            say("orphan ...")
            r["orphan"] = stage_orphan(splits, a.reps, out, a.model)
        if "regress" in stages:
            say("regress ...")
            r["regress"] = cached("regress", lambda: stage_regress(r["forge"], out))
        if "e2e" in stages:
            say("e2e ...")
            r["e2e"] = stage_e2e(out, a.model, [d for d in a.e2e_domains.split(",") if d])
    except Paused as p:
        rest = " ".join(x for x in sys.argv[1:] if x not in ("--resume", stamp))
        print(f"PAUSED at {p} after the time budget. Carry on with: "
              f"python evals/bench/persona_creation.py --resume {stamp} {rest}")
        sys.exit(2)
    r["seconds"] = round(time.time() - t0)
    (out / "results.json").write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
    text = summarize(r) + f"\n\ntotal {r['seconds']}s; raw results in {out}"
    (out / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
