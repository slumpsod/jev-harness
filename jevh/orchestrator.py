"""One user turn: route -> gate memory -> pick skills -> suggest collaborators -> run
the worker, serving its request() calls through Jev's approval on the task board.

Hand-offs run in parallel: each approved sub-task gets its own thread (and its own REPL
process); request() is request_async() + wait()."""
from __future__ import annotations

import itertools
import os
import threading
import time
import uuid
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from .bus import Board, LimitError, Task
from .config import Config, load_config
from .decide import SystemOneError
from .jev import CollabDecision, Jev, MemoryDecision, RouteDecision, SkillDecision, TierDecision
from .llm import LLM, make_llm
from .lmstudio import gate_for
from .memory import Memory
from .personas import Persona, load_personas
from .skills import load_roster
from .worker import Worker, WorkerResult, build_system

Event = Callable[[str, dict], None]


class Handles:
    """One task's hand-offs. request_async() returns a handle id the worker's code keeps;
    wait() blocks on one or a list. Unknown handles raise; unfinished ones on timeout come
    back as {"status": "running"} so the worker can do other work and wait again."""

    def __init__(self, start: Callable[..., Future], task: Task, goal: str, requestable: dict[str, str]):
        self._start, self.task, self.goal, self.requestable = start, task, goal, requestable
        self._futs: dict[str, Future] = {}
        self._waited: set[str] = set()
        self._ids = itertools.count(1)

    def start(self, to: str, text: str, inputs: list[str] | None = None) -> str:
        h = f"{self.task.id}.{next(self._ids)}"
        self._futs[h] = self._start(self.task, to, text, list(inputs or []), self.goal, self.requestable)
        return h

    def wait(self, h: str | list[str], timeout: float | None = None) -> dict | list[dict]:
        if isinstance(h, (list, tuple)):
            deadline = None if timeout is None else time.monotonic() + timeout
            return [self.wait(x, None if deadline is None else max(0.0, deadline - time.monotonic()))
                    for x in h]
        if h not in self._futs:
            raise KeyError(f"unknown handle {h!r}; use the value request_async() returned")
        try:
            res = self._futs[h].result(timeout)
        except FutureTimeout:
            return {"status": "running", "answer": "", "artifacts": [], "handle": h}
        self._waited.add(h)
        return res

    def collect_unwaited(self) -> list[dict]:
        out = [f.result() for h, f in self._futs.items() if h not in self._waited]
        self._waited.update(self._futs)
        return out


@dataclass
class TurnResult:
    persona: str
    answer: str
    artifacts: list[str]
    tasks: list[Task]
    trace: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0


class Harness:
    def __init__(self, cfg: Config | None = None, jev: Jev | None = None,
                 llm_factory: Callable[[Persona], LLM] | None = None, on_event: Event | None = None,
                 session: str | None = None, model_override: str | None = None):
        self.cfg = cfg or load_config()
        self.jev = jev or Jev(self.cfg)
        self.personas = load_personas(self.cfg.personas_dir)
        self.roster = load_roster(self.cfg.skills_dirs)
        self.memory = Memory(self.cfg.workspace / "memory")
        self.session = session or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.session_dir = self.cfg.workspace / "sessions" / self.session
        self.artifacts = self.session_dir / "artifacts"
        self.current: str | None = None
        self.on_event = on_event or (lambda kind, data: None)
        self.model_override = model_override
        self.tier_mode = "auto"  # auto (Jev decides) | fast | strong
        # the factory gets the persona with its model block already resolved for the tier
        self._llm_factory = llm_factory or (lambda p: make_llm(p.model, self.cfg.providers))
        self._llms: dict[str, LLM] = {}
        self._llm_lock = threading.Lock()
        self.board = Board()
        for prov in self.cfg.providers.values():  # surface LM Studio model swaps in the trace
            if prov.get("exclusive") and prov.get("base_url"):
                gate = gate_for(prov["base_url"])
                gate.linger = float(prov.get("swap_linger", gate.linger))
                gate.on_swap = lambda dropped, model, secs: self.on_event(
                    "swap", {"unloaded": dropped, "loaded": model, "seconds": secs})
        for key, env in (("comfy_url", "JEVH_COMFY_URL"), ("comfy_workflows", "JEVH_COMFY_WORKFLOWS")):
            if self.cfg.tools.get(key):  # REPL children inherit these
                os.environ.setdefault(env, str(self.cfg.tools[key]))

    # -- public ----------------------------------------------------------------------
    def turn(self, message: str, force: str | None = None) -> TurnResult:
        started = time.time()
        self.jev.trace = []
        self.board = Board(self.cfg.lim("max_depth"), self.cfg.lim("max_tasks_per_turn"))
        if force:
            if force not in self.personas:
                raise KeyError(f"unknown persona '{force}'")
            route = RouteDecision(force, 1.0, {}, None, "forced by user")
        else:
            route = self._safe("route", lambda: self.jev.route_persona(
                message, self.personas, self.current, self.memory.recent_text(self.session)),
                RouteDecision(self.current or self.cfg.default_persona, 0.0, {}, None, "jev unavailable"))
        self.on_event("route", {"decision": route})
        root = self.board.open(None, "user", route.persona, message)
        result = self.run_task(root, goal=message, allow_collab=True)
        if root.status == "done":  # a crashed turn's error text must not become "memory"
            self.memory.append(route.persona, message, result.answer, self.session)
        self.current = route.persona
        return TurnResult(route.persona, result.answer, result.artifacts, self.board.tasks,
                          list(self.jev.trace), time.time() - started)

    # -- tasks -----------------------------------------------------------------------
    def run_task(self, task: Task, goal: str, allow_collab: bool) -> WorkerResult:
        persona = self.personas[task.to]
        skills = self._safe("skills", lambda: self.jev.pick_skills(task.text, persona, self.roster),
                            SkillDecision([]))
        self.on_event("skills", {"task": task, "decision": skills})
        candidates = self.memory.candidates(persona.name, self.cfg.lim("memory_candidates_own"),
                                            self.cfg.lim("memory_candidates_shared"))
        mem = self._safe("memory", lambda: self.jev.gate_context(task.text, candidates),
                         MemoryDecision([], []))
        self.on_event("memory", {"task": task, "decision": mem, "candidates": len(candidates)})
        collab = CollabDecision([], {})
        if allow_collab and persona.code:
            collab = self._safe("collab", lambda: self.jev.collab(task.text, persona, self.personas), collab)
            self.on_event("collab", {"task": task, "decision": collab})
        requestable = {n: p.description for n, p in self.personas.items()
                       if persona.may_request(n) or n in collab.suggested}
        picked = [self.roster[s] for s in skills.skills]
        system = build_system(persona, picked, requestable, collab.suggested)
        ctx = {"request": task.text, "memory": mem.kept, "inputs": task.inputs,
               "skills": {s.name: s.body for s in picked}}
        tier = self._pick_tier(task, persona)
        self.on_event("tier", {"task": task, "decision": tier})
        llm = self._llm(persona, tier.tier)
        handles = Handles(self._start_request, task, goal, requestable)
        handlers = {
            "llm_query": lambda prompt, data=None: llm.chat([{"role": "user", "content": (
                prompt if data is None else f"{prompt}\n\n---\n{data}")}]).text,
            "jev": lambda state, questions: self.jev.client.ask(state, questions).as_dict(),
            "request": lambda to, text, inputs=None: handles.wait(handles.start(to, text, inputs)),
            "request_async": lambda to, text, inputs=None: handles.start(to, text, inputs),
            "wait": lambda h, timeout=None: handles.wait(h, timeout),
        }
        worker = Worker(persona, llm, self.session_dir / f"t{task.id}-{persona.name}", self.artifacts,
                        handlers, {t: f"jevh.tools.{t}" for t in persona.tools},
                        self.cfg.lim("repl_timeout"), self.cfg.lim("output_cap"),
                        lambda kind, data: self.on_event(kind, {**data, "task_id": task.id}))
        task.started = time.time()
        self.on_event("start", {"task": task, "model": getattr(llm, "name", "?")})
        try:
            result = worker.run(task.text, ctx, system)
            task.status = "done"
        except Exception as e:  # noqa: BLE001 - a dead model server must not kill the harness
            result = WorkerResult(answer=f"[{persona.name} failed: {type(e).__name__}: {e}]",
                                  stopped="error")
            task.status = "failed"
        # hand-offs the worker started but never waited for: finish them and keep their files
        for res in handles.collect_unwaited():
            result.artifacts = list(result.artifacts) + [a for a in res["artifacts"] if a not in result.artifacts]
            self.on_event("unwaited", {"task": task, "result": res})
        task.answer, task.artifacts = result.answer, result.artifacts
        task.finished = time.time()
        self.on_event("done", {"task": task, "result": result})
        return result

    def _start_request(self, parent: Task, to: str, text: str, inputs: list[str], goal: str,
                       requestable: dict[str, str]) -> Future:
        """Checks + Jev approval run in the caller's thread (fast); the approved sub-task runs
        on its own thread. Always returns a Future of {status, answer, artifacts}."""
        requester = self.personas[parent.to]

        def rejected(reason: str) -> Future:
            t = self.board.reject(parent, requester.name, to, text, reason)
            self.on_event("request", {"task": t, "approved": False, "reason": reason})
            f: Future = Future()
            f.set_result({"status": "rejected", "answer": reason, "artifacts": []})
            return f

        if to not in self.personas:
            return rejected(f"unknown assistant '{to}'. Available: {', '.join(requestable)}")
        if to not in requestable:
            return rejected(f"{requester.name} may not call {to}. Available: {', '.join(requestable)}")
        try:
            self.board.check(parent, to)  # fail fast before spending a Jev call
        except LimitError as e:
            return rejected(str(e))
        approval = self._safe("approve", lambda: self.jev.approve_request(
            requester, self.personas[to], text, goal), None)
        if approval is not None and not approval.approved:
            return rejected(f"Jev declined: {approval.reason}")
        try:
            sub = self.board.open_checked(parent, requester.name, to, text, inputs)
        except LimitError as e:  # a parallel sibling took the last slot meanwhile
            return rejected(str(e))
        self.on_event("request", {"task": sub, "approved": True,
                                  "scores": approval.scores if approval else {}})
        fut: Future = Future()

        def run() -> None:
            try:
                res = self.run_task(sub, goal, allow_collab=False)
                fut.set_result({"status": sub.status, "answer": res.answer, "artifacts": res.artifacts})
            except BaseException as e:  # noqa: BLE001 - never leave a waiter hanging
                sub.status = "failed"
                fut.set_result({"status": "failed", "answer": f"{type(e).__name__}: {e}", "artifacts": []})

        threading.Thread(target=run, name=f"task-{sub.id}-{to}", daemon=True).start()
        return fut

    # -- helpers ---------------------------------------------------------------------
    def _pick_tier(self, task: Task, persona: Persona) -> TierDecision:
        if self.model_override:
            return TierDecision("strong", None, "--model override")
        if not self.cfg.models.get("fast") or not persona.fast_ok:
            return TierDecision("strong", None, "no fast tier for this persona")
        if self.tier_mode in ("fast", "strong"):
            return TierDecision(self.tier_mode, None, "forced by user")
        # Jev down -> persona's own model: a slow right answer beats a fast wrong one
        return self._safe("tier", lambda: self.jev.pick_tier(task.text, persona),
                          TierDecision("strong", None, "jev unavailable"))

    def _llm(self, persona: Persona, tier: str = "strong") -> LLM:
        key = (persona.name, tier)
        with self._llm_lock:  # parallel hand-offs may build the same persona's client at once
            if key not in self._llms:
                llm = self._llm_factory(replace(persona, model=self._model_cfg(persona, tier)))
                for target in (llm, getattr(llm, "fallback", None)):
                    if target is not None and hasattr(target, "on_progress"):
                        target.on_progress = lambda n, thinking=0, name=persona.name: self.on_event(
                            "progress", {"persona": name, "chunks": n, "thinking": thinking})
                self._llms[key] = llm
            return self._llms[key]

    def _model_cfg(self, persona: Persona, tier: str = "strong") -> dict[str, Any]:
        """The model block a persona runs on. --model puts every persona on one model
        (testing); the fast tier swaps in config models.fast. Both keep the persona's own
        temperature/max_tokens and drop its fallback/Claude-only settings."""
        if self.model_override:
            provider, _, model = self.model_override.partition(":")
            swap = {"provider": provider, "model": model}
        elif tier == "fast" and self.cfg.models.get("fast"):
            swap = dict(self.cfg.models["fast"])
        else:
            return persona.model
        cfg = {k: v for k, v in persona.model.items() if k not in ("fallback", "effort", "thinking")}
        cfg.update(swap)
        return cfg

    def _safe(self, step: str, fn: Callable[[], Any], fallback: Any) -> Any:
        """Jev down -> degrade (default persona, no skills/memory/collab), never crash."""
        try:
            return fn()
        except SystemOneError as e:
            self.on_event("jev_error", {"step": step, "error": str(e)})
            return fallback
