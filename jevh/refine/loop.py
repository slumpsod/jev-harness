"""The refinement loop: logged failures -> tests -> a fix -> old vs new -> a report to approve.

One attempt per failure group (same kind, persona and cause), in workspace/refine/<id>/, moving
through stages that each survive a restart, so a night cut short carries on the next night. The cheap
questions come first - does it happen again, and is it the harness's to fix? - so a night isn't spent
writing tests for a slip one model made once:

  new         the local model writes a checker for the failed request (the replay)
                                                        (dropped: no checker held up)
  checked     cross-model replay: the request on every local model, up to repro_runs each (a model
              stops after repro_stop failures)          (dropped: it doesn't happen again)
  replayed    triage: the local model reads the failure and the replays - harness_caused /
              harness_preventable / model_only / not_a_failure, the component, the evidence
                                                        (dropped, and the failures marked: not the harness's)
  triaged     the local model writes variants aimed at that cause, plus the control tasks
                                                        (dropped: too few checkers held up)
  generated   baseline: every case on every local model, on the old code; the replay runs as often as
              its failure rate needs. Then variants that never failed become regression checks, and the
              local model looks at one that failed every run everywhere: a wrong checker is set aside.
  baseline    the local model changes a worktree of the harness (branch refine/<id>) and the
              offline tests must pass                   (dropped: no working change in N tries)
  fixed       the same matrix on the new code, then the verdict:
              improves / mixed -> proposed (report.md); no_gain -> another fix try, or dropped
  proposed    waiting for you: approve (merged, the failures marked fixed) or reject (the reason
              is shown to the fixer the next time this kind of failure comes up)

A failure whose task got done with needless extra work (refine.cost_kinds: a correct answer the
final() gate sent back) is scored on cost - gate bounces per run - instead of completion.

"Old" is the commit the attempt started from, in its own worktree: edits in your checkout and
approvals merged meanwhile don't change what the fix is compared against. Everything runs on
local models (config refine.model, else models.strong, else the fixer persona's model)."""
from __future__ import annotations

import dataclasses
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable

from ..config import Config
from ..decide import noul
from ..failures import FailureLog
from . import cases as cases_mod
from . import checks, fixer, report, runner, triage
from .cases import Case
from .generate import Generator

DEFAULTS: dict[str, Any] = {
    "model": None,              # "provider:model" that writes tests and fixes (local only)
    "max_tokens": 65536,        # per test-writing reply, thinking included (the fixer's turn uses its persona's)
    "models": None,             # models the cases run on (default: every local model in the config)
    "discover": False,          # also every chat model LM Studio lists
    "cases": 10,                # variants to write per failure (plus the replayed failure itself)
    "min_cases": 6,             # fewer checkers held up -> the attempt is dropped
    "holdout": 0.3,             # share of variants the fixer never sees
    "runs": 2,                  # runs of each case per model and version
    "repro_runs": 4,            # replays of the failed request per model, before any variant is written
    "repro_stop": 2,            # ...a model stops once it failed this often (enough to know it happens)
    "min_repro": 0.1,           # share of all replays that must fail, or there's nothing to measure
    "repro_target": 3,          # the replay runs often enough to expect this many failures on the old code
    "max_runs": 8,              # ...up to this many per model and version
    "fix_tries": 2,             # changes the fixer may try per attempt
    "min_gain": 0.15,           # verdict: the failing cases' pass rate must rise this much
    "min_cost_gain": 0.5,       # verdict (cost kinds): gate bounces per run must fall by this share
    "control_slack": 1,         # ...with no model losing more than this many control + regression runs
    "cost_warn": 0.25,          # time/tokens per run up more than this -> a caveat
    "case_timeout": 3600,       # seconds per run
    "test_command": "python -m pytest -q -x",
    "test_timeout": 1800,
    "fixer_persona": "high_end_dev",
    "skip_kinds": ["stopped_by_user", "jev_unavailable"],   # not the harness's to fix
    # the task got done, with needless extra work: scored on gate bounces per run, and the logged answer may be
    # right (live: 4 of 4 bounced answers were), so the replay checker passes a correct result instead of rejecting it
    "cost_kinds": ["gate_bounce"],
    # ...and bounces come from the harness's own gate, so a fix could win by making it lenient: for a cost kind the
    # gate's offline eval must catch as many bad answers as before, with no more false alarms (old tree vs new)
    "cost_guard": "python evals/verify_cases.py --guard",
    "retry_days": 14,           # a failure group attempted within this many days isn't picked again
    "controls": None,           # control tasks file (default evals/refine_controls.json)
}
OPEN = ("new", "checked", "replayed", "triaged", "generated", "baseline", "fixed")
UNFINISHED = re.compile(r"^\[[^\]\n]*(?:step limit|going in circles|crashed|stopped)", re.I)


def cause(summary: str) -> str:
    """What a failure's summary says went wrong, without its numbers: failures of one kind can have
    different causes (live: gate bounces for "doesn't match the work log" and for "16" in "16:9")."""
    s = re.sub(r"\d+(?:[.,:]\d+)*", "#", summary or "")
    return re.split(r"(?<=[.!?])\s", s, maxsplit=1)[0].strip()[:140]


class Refiner:
    def __init__(self, cfg: Config, *, llm=None, judge: checks.Judge | None = None,
                 execute_fn: runner.Execute | None = None, run_turn: Callable[[Path, str], str] | None = None,
                 log: Callable[[str], None] | None = None):
        self.cfg = cfg
        self.s = {**DEFAULTS, **(cfg.raw.get("refine") or {})}
        self.repo = Path(cfg.root)
        self.home = Path(cfg.workspace) / "refine"
        self.home.mkdir(parents=True, exist_ok=True)
        self.failures = FailureLog(Path(cfg.workspace) / "failures.db")
        self._llm, self._judge, self._run_turn = llm, judge, run_turn
        self.execute_fn = execute_fn or (lambda root, case, model, ws: runner.execute(
            root, case, model, ws, float(self.s["case_timeout"])))
        self._log = log or print

    def log(self, msg: str) -> None:
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
        with open(self.home / "refine.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        self._log(line)

    # -- models ------------------------------------------------------------------------
    def model_block(self) -> dict[str, Any]:
        """The writer's model with its settings: refine.model as a block, or the block the harness already
        runs that model with (models.strong / the personas'), with refine.max_tokens as its budget.
        Live (Qwen 3.8 27B, ~30 tok/s): a bare "provider:model" got the 8192-token default and every variant
        reply was cut off, so the attempt was dropped with 1 test case. Unlimited, its replies took 18-21k
        tokens (~11 min each); a 32k cap cut two 12-case variant replies off. The budget stays finite because
        nothing watches these calls the way a worker's thinking is watched."""
        from ..personas import load_personas
        m = self.s["model"]
        if isinstance(m, dict):
            block = dict(m)
        else:
            personas = load_personas(self.cfg.personas_dir)
            if not m:
                block = dict(self.cfg.models.get("strong") or personas[self.s["fixer_persona"]].model)
            else:
                known = [b for b in (self.cfg.models.get("strong"), self.cfg.models.get("fast"),
                                     *(p.model for p in personas.values())) if b and b.get("provider") and b.get("model")]
                block = next((dict(b) for b in known if runner.model_id(b) == m),
                             {"provider": m.partition(":")[0], "model": m.partition(":")[2]})
        block.pop("think", None)
        block["max_tokens"] = int(self.s["max_tokens"])
        return block

    def model(self) -> str:
        """The local model that writes tests and fixes."""
        m = runner.model_id(self.model_block())
        prov = self.cfg.providers.get(m.partition(":")[0])
        if prov is None or not runner.is_local(prov):
            raise RuntimeError(f"refine runs on local models only; {m} isn't one (set refine.model)")
        return m

    def models(self) -> list[str]:
        return runner.local_models(self.cfg)

    def llm(self):
        if self._llm is None:
            from ..llm import make_llm
            self.model()   # local only
            self._llm = make_llm(self.model_block(), self.cfg.providers)
        return self._llm

    def judge(self) -> checks.Judge:
        if self._judge is None:
            from ..jev import Jev
            jev = Jev(self.cfg)
            self._judge = lambda q, state: jev._ask("refine_check", state, {"q": noul(q)})["q"].noul
        return self._judge

    # -- picking -----------------------------------------------------------------------
    def attempts(self) -> list[dict]:
        out = []
        for p in sorted(self.home.glob("*/state.json")):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass
        return out

    def groups(self) -> list[dict]:
        """Open failures by (kind, persona, cause), most frequent first, with whether they were tried lately."""
        recent = time.time() - float(self.s["retry_days"]) * 86400
        tried = {i for a in self.attempts() if a["created"] >= recent or a["stage"] in ("proposed", "approved")
                 for i in a.get("group", [])}
        skip = set(self.s["skip_kinds"] or [])
        groups: dict[tuple, list[dict]] = {}
        for f in self.failures.list(status="open", limit=100000):
            if f["kind"] not in skip and (f.get("request") or "").strip():
                groups.setdefault((f["kind"], f["persona"], cause(f["summary"])), []).append(f)
        out = [{"kind": k, "persona": p, "cause": c, "count": len(fs), "ids": [f["id"] for f in fs], "latest": fs[0],
                "tried": all(f["id"] in tried for f in fs)} for (k, p, c), fs in groups.items()]
        return sorted(out, key=lambda g: (g["tried"], -g["count"], -g["latest"]["ts"]))

    def pick(self) -> dict | None:
        return next((g for g in self.groups() if not g["tried"]), None)

    # -- an attempt ----------------------------------------------------------------------
    def dir(self, aid: str) -> Path:
        return self.home / aid

    def state(self, aid: str) -> dict:
        return json.loads((self.dir(aid) / "state.json").read_text(encoding="utf-8"))

    def save(self, st: dict) -> None:
        st["updated"] = time.time()
        p = self.dir(st["id"]) / "state.json"
        p.write_text(json.dumps(st, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

    def start(self, failure_id: int) -> str:
        f = self.failures.get(failure_id)
        if f is None:
            raise LookupError(f"no failure #{failure_id}")
        why = cause(f["summary"])
        group = [g["id"] for g in self.failures.list(kind=f["kind"], status="open", limit=100000)
                 if g["persona"] == f["persona"] and cause(g["summary"]) == why] or [f["id"]]
        stem = time.strftime("%Y%m%d-%H%M%S") + f"-f{f['id']}"
        aid, n = stem, 1
        while self.dir(aid).exists():   # two attempts on one failure within a second
            n += 1
            aid = f"{stem}-{n}"
        self.dir(aid).mkdir(parents=True)
        base = fixer.git(self.repo, "rev-parse", "HEAD").strip()
        st = {"id": aid, "failure": {k: v for k, v in f.items()}, "group": group, "group_size": len(group),
              "cause": why, "objective": "cost" if f["kind"] in (self.s["cost_kinds"] or []) else "pass",
              "stage": "new", "created": time.time(), "base": base, "branch": f"refine/{aid}",
              "tree": str(self.dir(aid) / "tree"), "models": self.models(), "runs": int(self.s["runs"]),
              "generator_model": self.model(), "fixer_model": self.model(), "tries": [], "dropped": 0}
        if not st["models"]:
            raise RuntimeError("no local models to test on (config refine.models)")
        self.save(st)
        self.log(f"{aid}: started on #{f['id']} {f['kind']} ({f['persona']}): {f['summary'][:100]}")
        return aid

    def run(self, aid: str, deadline: float | None = None) -> dict:
        """Advance an attempt until it's proposed, dropped, or the deadline."""
        st = self.state(aid)
        while st["stage"] in OPEN:
            if deadline is not None and time.time() >= deadline:
                break
            before = (st["stage"], len(st["tries"]))
            st = getattr(self, "_" + st["stage"])(st, deadline)
            self.save(st)
            if (st["stage"], len(st["tries"])) == before:   # a stage that stopped for the deadline
                break
        return st

    def cases(self, st: dict) -> list[Case]:
        return cases_mod.load(self.dir(st["id"]) / "cases.json")

    def rows(self, st: dict) -> list[dict]:
        return runner.load_rows(self.dir(st["id"]) / "results.jsonl")

    def base_tree(self, commit: str) -> Path:
        """The old code: the attempt's starting commit in its own (shared) worktree."""
        return fixer.make_tree(self.repo, self.home / f"base-{commit[:12]}", None, commit)

    def _matrix(self, st: dict, cs: list[Case], versions: dict[str, Path], models: list[str], runs: int,
                deadline: float | None) -> bool:
        return runner.run_matrix(cs, versions, models, runs, self.dir(st["id"]) / "results.jsonl",
                                 self.dir(st["id"]) / "runs", self.judge(), self.execute_fn, deadline,
                                 lambda r: self.log(f"{st['id']}: {r['version']} {r['model']} {r['case']} "
                                                    f"r{r['rep']}: {'pass' if r['passed'] else 'FAIL'} ({r['why'][:80]})"
                                                    + (f" [{r['bounces']} bounce(s)]" if r.get("bounces") else "")))

    def failed(self, st: dict, row: dict) -> bool:
        """Did this run show the failure? Not done - or, for a cost kind, the same failure was logged again
        (the gate sent the answer back)."""
        return not row["passed"] or (st.get("objective") == "cost"
                                     and st["failure"]["kind"] in (row.get("failures") or []))

    def _drop(self, st: dict, reason: str) -> dict:
        st["stage"], st["reason"] = "dropped", reason
        self.log(f"{st['id']}: dropped - {reason}")
        tree = Path(st["tree"])
        if tree.exists():
            fixer.remove_tree(self.repo, tree, st["branch"])
        return st

    def _generator(self, st: dict) -> Generator:
        t = st.get("triage") or {}
        reading = f"{t['category']}" + (f" ({t['component']})" if t.get("component") else "") + \
            (f": {t['why']}" if t.get("why") else "") if t else ""
        return Generator(self.llm(), self.judge(), int(self.s["cases"]), float(self.s["holdout"]),
                         seed=int(st["failure"]["id"]), cost=st.get("objective") == "cost", cause=reading)

    def _keep_writing(self, st: dict, gen: Generator) -> None:
        """What the writer produced (dropped cases, raw replies), added to the attempt's files."""
        for name, new in (("dropped.json", gen.dropped), ("replies.json", gen.replies)):
            p = self.dir(st["id"]) / name
            old = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else []
            p.write_text(json.dumps(old + new, indent=1, ensure_ascii=False), encoding="utf-8")
        st["dropped"] = int(st.get("dropped") or 0) + len(gen.dropped)

    def _new(self, st: dict, deadline) -> dict:
        gen = self._generator(st)
        replay = gen.replay(st["failure"])
        self._keep_writing(st, gen)
        if replay is None:
            return self._drop(st, "no checker for the failed request held up")
        cases_mod.save(self.dir(st["id"]) / "cases.json", [replay])
        st["stage"] = "checked"
        self.log(f"{st['id']}: replay checker: {checks.describe(replay.checker)[:120]}")
        return st

    def _checked(self, st: dict, deadline) -> dict:
        """Cross-model replay: does it happen again, and on which models?"""
        replay = self.cases(st)[0]
        old = {"old": self.base_tree(st["base"])}
        for model in st["models"]:
            for rep in range(int(self.s["repro_runs"])):
                mine = [r for r in self.rows(st) if r["case"] == "replay" and r["model"] == model and r["version"] == "old"]
                if sum(self.failed(st, r) for r in mine) >= int(self.s["repro_stop"]):
                    break
                if not self._matrix(st, [replay], old, [model], rep + 1, deadline):
                    return st
        rows = [r for r in self.rows(st) if r["case"] == "replay" and r["version"] == "old"]
        st["repro"] = {m: {"runs": sum(r["model"] == m for r in rows),
                           "failed": sum(r["model"] == m and self.failed(st, r) for r in rows)} for m in st["models"]}
        failed, runs = sum(v["failed"] for v in st["repro"].values()), len(rows)
        st["repro_rate"] = round(failed / max(runs, 1), 3)
        seen = ", ".join(f"{m.partition(':')[2]} {v['failed']}/{v['runs']}" for m, v in st["repro"].items())
        if st["repro_rate"] < float(self.s["min_repro"]) or not failed:
            return self._drop(st, f"doesn't reproduce: failed {seen}")
        self.log(f"{st['id']}: reproduced: failed {seen}")
        st["stage"] = "replayed"
        return st

    def _replayed(self, st: dict, deadline) -> dict:
        """The local model's own reading of the failure and its replays: is it the harness's to fix?"""
        f, replay = st["failure"], self.cases(st)[0]
        rows = [dict(r, reproduced=self.failed(st, r)) for r in self.rows(st) if r["case"] == "replay"]
        how = (f"the final() gate sends its answer back, or its result fails this check: {checks.describe(replay.checker)}"
               if st.get("objective") == "cost" else f"its result fails this check: {checks.describe(replay.checker)}")
        reading = triage.triage(self.llm(), f, rows, st["models"], how)
        failing = [m for m, v in st["repro"].items() if v["failed"]]
        go, decision = triage.keep(reading, failing)
        st["triage"] = {k: v for k, v in reading.items() if k != "reply"} | {"failing_models": failing,
                                                                            "decision": decision}
        (self.dir(st["id"]) / "triage.json").write_text(json.dumps(reading, indent=1, ensure_ascii=False),
                                                         encoding="utf-8")
        self.log(f"{st['id']}: triage: {decision[:200]}")
        if not go:
            status = "ignored" if reading["category"] == "not_a_failure" else "reviewed"
            for fid in st["group"]:
                self.failures.set_status(fid, status, f"refine triage {st['id']}: {decision}"[:500])
            return self._drop(st, f"not the harness's to fix - {decision}")
        st["stage"] = "triaged"
        return st

    def _triaged(self, st: dict, deadline) -> dict:
        f = st["failure"]
        gen = self._generator(st)
        variants = gen.variants(f, f.get("persona") or None)
        self._keep_writing(st, gen)
        if 1 + len(variants) < int(self.s["min_cases"]):
            return self._drop(st, f"only {1 + len(variants)} test cases held up (need {self.s['min_cases']})")
        replay = self.cases(st)[0]
        # run the replay often enough to see the failure a few times on the old code: a 1-in-4 failure shows
        # nothing in 2 runs (live: planets.csv, 1 of 3, then "mixed" at p = 0.32)
        rate = max(float(st.get("repro_rate") or 1.0), 1e-3)
        replay.runs = max(int(st["runs"]), min(int(self.s["max_runs"]), math.ceil(float(self.s["repro_target"]) / rate)))
        cases_mod.save(self.dir(st["id"]) / "cases.json", [replay, *variants, *cases_mod.controls(self.s["controls"])])
        st["stage"] = "generated"
        self.log(f"{st['id']}: {1 + len(variants)} test cases ({len(gen.dropped)} dropped); "
                 f"the replay runs {replay.runs}x per model")
        return st

    def _generated(self, st: dict, deadline) -> dict:
        """Baseline on the old code, then sort the variants by what they showed there."""
        cs = self.cases(st)
        if not self._matrix(st, cs, {"old": self.base_tree(st["base"])}, st["models"], st["runs"], deadline):
            return st
        old = [r for r in self.rows(st) if r["version"] == "old"]
        moved, why = {"regress": [], "suspect": []}, {}
        for c in cs:
            if c.origin != "variant":
                continue
            mine = [r for r in old if r["case"] == c.id]
            fails = [r for r in mine if self.failed(st, r)]
            if mine and not fails:
                c.split = "regress"   # never failed: it can't show a gain, but a fix must not break it
            elif mine and len(fails) == len(mine):
                # failed every run everywhere: a hard case, or a wrong checker - the local model looks at a run
                wrong, why[c.id] = triage.review_checker(self.llm(), c, fails[0])
                if not wrong:
                    continue
                c.split = "suspect"
            else:
                continue
            moved[c.split].append(c.id)
        cases_mod.save(self.dir(st["id"]) / "cases.json", cs)
        st["moved"], st["checker_review"] = moved, why
        if any(moved.values()):
            self.log(f"{st['id']}: regression checks {moved['regress'] or '-'}; suspect checkers {moved['suspect'] or '-'}")
        st["stage"] = "baseline"
        return st

    def _baseline(self, st: dict, deadline) -> dict:
        k = len(st["tries"]) + 1
        tree = fixer.make_tree(self.repo, Path(st["tree"]), st["branch"], st["base"])
        if k > 1:
            fixer.reset_tree(tree, st["base"])
        f = st["failure"]
        group = [g for g in (self.failures.get(i) for i in st["group"]) if g]
        earlier = [f"try {t['k']} changed {', '.join(t.get('files') or []) or 'nothing'}: {t.get('summary', '')[:200]}"
                   f" -> {t.get('outcome', '')}" for t in st["tries"]]
        task = fixer.prompt(f, group, self.cases(st), [r for r in self.rows(st) if r["version"] == "old"],
                            [r for r in self.rejections() if r.get("kind") == f["kind"]], self.s["test_command"],
                            earlier, st.get("triage"))
        self.log(f"{st['id']}: fix try {k} on {st['fixer_model']}")
        fx = fixer.fix(tree, task, self._run_turn or self._turn, self.s["test_command"], float(self.s["test_timeout"]),
                       f"refine {st['id']}: {f['kind']} ({f.get('persona') or '-'})")
        (self.dir(st["id"]) / f"try{k}.diff").write_text(fx.diff, encoding="utf-8")
        t = {"k": k, "summary": fx.summary, "files": fx.files, "reverted": fx.reverted, "tests_ok": fx.tests_ok,
             "tests_out": fx.tests_out[-1500:], "commit": fx.commit, "error": fx.error,
             # live: the fixer hit its step limit mid-sentence; its change was fine, its "summary" was not one
             "unfinished": bool(UNFINISHED.match(fx.summary or ""))}
        if fx.ok and st.get("objective") == "cost" and self.s.get("cost_guard"):
            t["guard"] = self._guard(st, tree)
            if t["guard"].get("error"):
                fx.error = t["guard"]["error"]
        if not fx.ok or t.get("guard", {}).get("error"):
            t["outcome"] = fx.error
            st["tries"].append(t)
            self.log(f"{st['id']}: try {k}: {fx.error}")
            return st if k < int(self.s["fix_tries"]) else self._drop(st, f"no working change in {k} tries ({fx.error})")
        st["tries"].append(t)
        st["stage"] = "fixed"
        return st

    def _guard(self, st: dict, tree: Path) -> dict:
        """The cost guard on the old and the new code: {old, new} (each {caught, bad, false}), or {error}."""
        out = {}
        for name, root in (("old", self.base_tree(st["base"])), ("new", tree)):
            ok, text = fixer.run_tests(root, self.s["cost_guard"], float(self.s["test_timeout"]))
            try:
                out[name] = json.loads(text.strip().splitlines()[-1])
            except (ValueError, IndexError):
                return {"error": f"the cost guard didn't run on the {name} code: {text.strip()[-300:]}"}
        a, b = out["old"], out["new"]
        if b["caught"] < a["caught"] or b["false"] > a["false"]:
            out["error"] = (f"the final() gate got more lenient: its eval caught {a['caught']}/{a['bad']} bad answers "
                            f"with {a['false']} false alarms, now {b['caught']}/{b['bad']} with {b['false']}")
        return out

    def _fixed(self, st: dict, deadline) -> dict:
        t = st["tries"][-1]
        new = f"new{t['k']}"
        cs = self.cases(st)
        live = [c for c in cs if c.split != "suspect"]   # a suspect checker measures nothing: not run again
        if not self._matrix(st, live, {"old": self.base_tree(st["base"]), new: Path(st["tree"])},
                            st["models"], st["runs"], deadline):
            return st
        v = report.verdict(self.rows(st), "old", new, self.s, {c.id: c.split for c in cs}, st.get("objective", "pass"))
        t["verdict"], t["gain"] = v["verdict"], v["gain"]
        extra = [f"the fixer's edits to {', '.join(t['reverted'])} were thrown away (protected)"] if t["reverted"] else []
        if t.get("unfinished"):
            extra.append("the fixer didn't finish its turn (it stopped before final()): its summary is not an account "
                         "of the change - read the diff")
        if st.get("dropped"):
            extra.append(f"{st['dropped']} generated test cases were dropped (their checker didn't hold up)")
        md = report.render({**st, "try": t["k"]}, v, cs,
                           {**t, "diff": (self.dir(st["id"]) / f"try{t['k']}.diff").read_text(encoding="utf-8")}, extra)
        (self.dir(st["id"]) / f"report-try{t['k']}.md").write_text(md, encoding="utf-8")
        self.log(f"{st['id']}: try {t['k']}: {v['verdict']} (gain {v['gain']:+.2f})")
        if v["verdict"] in ("improves", "mixed"):
            (self.dir(st["id"]) / "report.md").write_text(md, encoding="utf-8")
            st["stage"], st["verdict"] = "proposed", v["verdict"]
            return st
        ta, tb = v["old"]["all"]["target"], v["new"]["all"]["target"]
        t["outcome"] = f"no gain: failing cases {ta['passed']}/{ta['n']} -> {tb['passed']}/{tb['n']}" + (
            f", bounces per run {ta['bounces']:g} -> {tb['bounces']:g}" if st.get("objective") == "cost" else "")
        if t["k"] < int(self.s["fix_tries"]):
            st["stage"] = "baseline"
            return st
        return self._drop(st, f"no change improved it in {t['k']} tries")

    def _turn(self, tree: Path, task: str) -> str:
        """The fixer's harness turn: the current harness, its fixer persona on the local model, the
        worktree open as its project."""
        from ..orchestrator import Harness
        from .run_case import ASK_TIMEOUT
        cfg = dataclasses.replace(self.cfg, workspace=self.home / "fixer", limits={**self.cfg.limits,
                                  "ask_timeout": ASK_TIMEOUT}, raw={**self.cfg.raw, "learning": False})
        h = Harness(cfg, model_override=self.model())
        h.learning = False
        h.set_project(str(tree))
        r = h.turn(task, force=self.s["fixer_persona"])
        return r.answer

    # -- after the night ---------------------------------------------------------------
    def nightly(self, deadline: float) -> list[dict]:
        """Carry on unfinished attempts, then start new ones, until the deadline or nothing is left."""
        done = []
        for a in sorted((a for a in self.attempts() if a["stage"] in OPEN), key=lambda a: a["created"]):
            if time.time() >= deadline:
                return done
            done.append(self.run(a["id"], deadline))
        while time.time() < deadline:
            g = self.pick()
            if g is None:
                self.log("nothing left to work on")
                break
            done.append(self.run(self.start(g["latest"]["id"]), deadline))
        self.cleanup()
        return done

    def cleanup(self) -> None:
        """Base worktrees no unfinished attempt needs."""
        needed = {a["base"][:12] for a in self.attempts() if a["stage"] in OPEN}
        for d in self.home.glob("base-*"):
            if d.name[5:] not in needed:
                fixer.remove_tree(self.repo, d)

    def proposals(self) -> list[dict]:
        return [a for a in self.attempts() if a["stage"] == "proposed"]

    def rejections(self) -> list[dict]:
        p = self.home / "rejections.jsonl"
        return runner.load_rows(p) if p.is_file() else []

    def approve(self, aid: str) -> str:
        st = self.state(aid)
        if st["stage"] != "proposed":
            raise ValueError(f"{aid} is {st['stage']}, not waiting for approval")
        if fixer.git(self.repo, "status", "--porcelain", "--untracked-files=no").strip():
            raise RuntimeError("your checkout has uncommitted changes: commit or stash them first")
        t = st["tries"][-1]
        msg = f"Refine {aid}: {st['failure']['kind']} ({st['failure'].get('persona') or '-'})\n\n{t['summary'][:1500]}"
        try:
            fixer.git(self.repo, "merge", "--no-ff", st["branch"], "-m", msg)
        except RuntimeError:
            fixer.git(self.repo, "merge", "--abort", check=False)
            raise
        commit = fixer.git(self.repo, "rev-parse", "HEAD").strip()
        for fid in st["group"]:
            self.failures.set_status(fid, "fixed", f"refine {aid}: merged {commit[:10]}")
        fixer.remove_tree(self.repo, Path(st["tree"]), st["branch"])
        st["stage"], st["merged"] = "approved", commit
        self.save(st)
        self.log(f"{aid}: approved, merged as {commit[:10]}")
        return commit

    def reject(self, aid: str, reason: str) -> None:
        st = self.state(aid)
        if st["stage"] != "proposed":
            raise ValueError(f"{aid} is {st['stage']}, not waiting for approval")
        t = st["tries"][-1]
        with open(self.home / "rejections.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": aid, "kind": st["failure"]["kind"], "persona": st["failure"].get("persona"),
                                 "change": t.get("summary", ""), "files": t.get("files", []), "reason": reason,
                                 "ts": time.time()}, ensure_ascii=False) + "\n")
        fixer.remove_tree(self.repo, Path(st["tree"]), st["branch"])
        st["stage"], st["reason"] = "rejected", reason
        self.save(st)
        self.log(f"{aid}: rejected - {reason}")
