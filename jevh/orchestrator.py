"""One user turn: route -> gate memory -> pick skills -> suggest collaborators -> run
the worker, serving its request() calls through Jev's approval on the task board.

Hand-offs run in parallel: each approved sub-task gets its own thread (and its own REPL
process); request() is request_async() + wait()."""
from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path
import threading
import time
import traceback
import uuid
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from .bus import Board, LimitError, Task
from .config import Config, load_config
from .decide import SystemOneError, noul
from .errors import Cancelled
from .jev import CollabDecision, Jev, MemoryDecision, RouteDecision, SkillDecision, TierDecision, ToolDecision
from .learn import Learner, seen_text, work_log
from .llm import LLM, make_llm
from .lmstudio import gate_for
from .mcp import McpHub
from .memory import Memory
from .personas import Persona, load_personas
from .skills import load_roster
from .tools import TOOLS
from .usage import Metered, Usage
from .verify import Verifier
from .worker import Worker, WorkerResult, build_system

Event = Callable[[str, dict], None]
RESUMABLE = {"max_steps", "stalled", "error", "cancelled"}
# Progress check (tuned on evals/progress_cases.py, semif): 6/6 stalls caught with 0/8 healthy runs
# interrupted at any threshold 0.3-0.8; "is it stuck?" interrupted a healthy test-fixing run.
# Thinking watch (limits.thinking_watch): while a worker's step is still pure thinking, Jev reads
# the end of it; once it holds an idea code could check, the step is paused and the model is asked
# to test it (its reasoning is kept). ARC bench: a model reasoned 80-100 KB in its head about a rule
# it could have run against the examples in seconds - and read block sizes off printed grids by eye,
# misread a 2x4 as 2x3, and built a wrong theory on it. Only watched once the task has run code
# (there is data in the REPL): advice, writing and planning thinks are never paused.
# evals/thinking_watch_cases.py: 11/11 caught, 0/8 wrong pauses at 0.55 (first wording: 7/9, and it
# let both live misreadings through).
TESTABLE_Q = ("Is `thinking` doing by hand something code could do exactly on the task's data - reading off, "
              "listing or measuring values, positions, sizes or counts, or checking a rule, pattern or "
              "calculation against the data?")
# Compaction: which old steps the rest of the task still needs in full (the rest become one-line notes).
# evals/compact_cases.py at 0.8: 11/11 needed kept, 2/15 needless (both superseded test runs); held out
# 5/7 and 1/7 - the two misses were 1-3 line steps whose note already holds all of their code. A
# "will finishing still need what step found or did" wording ran flat (9/11 needed at 8/15 needless).
COMPACT_Q = ("Does `step` hold information that finishing `task` will use again, such as data values, "
             "column or function names, file contents, sources, URLs or a finding about what does or does not work?")
CIRCLES_Q = ("Is `work_log` repeating the same or nearly the same attempt and getting the same result, "
             "without learning anything new?")
STOPPED_NOTE = "[stopped by the user before finishing]"


def turn_notes(result: WorkerResult, cap: int = 2000) -> str:
    """What a turn's answer was based on, kept with it in memory: a follow-up like "where did
    you get that?" once saw only the reply (plus a 'Not verified' note) and confessed to making
    up a price it had in fact read off the store's page."""
    log = work_log(result.transcript, cap=cap, answer=result.answer)
    return log or "(no code or tools were run for this reply: it came from the conversation and the model itself)"


def _shielded(callback: Event) -> Event:
    """Events are for display. A bug in a UI's handler must never stop the work that emitted
    it (seen live: a printer's KeyError failed a long task at step 2)."""
    def safe(kind: str, data: dict) -> None:
        try:
            callback(kind, data)
        except Exception:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
    return safe


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
    cancelled: bool = False
    usage: dict[str, Any] = field(default_factory=dict)   # Usage.summary() for this turn
    checkpoint: str | None = None     # the lead task's checkpoint, when the turn can be resumed


@dataclass
class PendingQuestion:
    """A worker waiting in ask_user() for the user's reply."""
    id: str
    task_id: int
    persona: str
    question: str
    options: list[str]
    asked: float
    answer: str | None = None
    event: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "task_id": self.task_id, "persona": self.persona, "question": self.question,
                "options": self.options, "asked": self.asked}


class Harness:
    def __init__(self, cfg: Config | None = None, jev: Jev | None = None,
                 llm_factory: Callable[[Persona], LLM] | None = None, on_event: Event | None = None,
                 session: str | None = None, model_override: str | None = None):
        self.cfg = cfg or load_config()
        self.jev = jev or Jev(self.cfg)
        self.personas = load_personas(self.cfg.personas_dir)
        self.skills_home = self.cfg.skills_dirs[0]    # where learned skills and lessons are written
        self.roster = load_roster(self.cfg.skills_dirs, self.skills_home / "_lessons")
        self.memory = Memory(self.cfg.workspace / "memory")
        self.mcp = McpHub(self.cfg.mcp_servers)
        self.project: str | None = None             # the user's project folder, if one is open
        self.learning = bool(self.cfg.raw.get("learning", True))
        self.learn_sync = False                     # tests: learn inline instead of on a thread
        self.session = session or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.session_dir = self.cfg.workspace / "sessions" / self.session
        self.artifacts = self.session_dir / "artifacts"
        self.current: str | None = None
        self.on_event = _shielded(on_event or (lambda kind, data: None))
        self.model_override = model_override
        self.tier_mode = "auto"  # auto (Jev decides) | fast | strong
        self._escalated = False  # the last turn's fast model was replaced by the strong one
        # show the browser tool's window, with Jev's picks outlined (dashboard toggle, CLI /watch)
        self.watch_browser = bool(self.cfg.tools.get("browser_watch")) or os.environ.get("JEVH_BROWSER_HEADED") == "1"
        # the factory gets the persona with its model block already resolved for the tier
        self._llm_factory = llm_factory or (lambda p: make_llm(p.model, self.cfg.providers))
        self._llms: dict[str, LLM] = {}
        self._llm_lock = threading.Lock()
        self.board = Board()
        for prov in self.cfg.providers.values():  # surface LM Studio model swaps in the trace
            if prov.get("exclusive") and prov.get("base_url"):
                gate = gate_for(prov["base_url"])
                gate.linger = float(prov.get("swap_linger", gate.linger))
                gate.on_swap = lambda dropped, model, secs: (
                    self.on_event("swap", {"unloaded": dropped, "loaded": model, "seconds": secs}),
                    self._record(kind="swap", model=model, seconds=secs, calls=1))
        for key, env in (("comfy_url", "JEVH_COMFY_URL"), ("comfy_workflows", "JEVH_COMFY_WORKFLOWS"),
                         ("tools_jev", "JEVH_TOOLS_JEV")):
            if self.cfg.tools.get(key):  # REPL children inherit these
                os.environ.setdefault(env, str(self.cfg.tools[key]))
        os.environ.setdefault("JEVH_JEV_URL", str(self.cfg.jev.get("url", "http://127.0.0.1:33001")))
        os.environ.setdefault("JEVH_JEV_BACKEND", str(self.cfg.jev.get("backend", "semif")))
        self.verifier = Verifier(self.jev, self.cfg)
        self.usage = Usage(self.cfg.workspace / "usage.db", self.cfg.raw.get("prices"))
        self.turn_no = 0
        self._tl = threading.local()                 # which turn/task/persona this thread works for
        self.jev.on_call = lambda step, resp, n: self._record(
            kind="jev", model=resp.model, input_tokens=resp.input_tokens, seconds=resp.latency_s,
            persona=getattr(self._tl, "persona", ""), task=getattr(self._tl, "task", None))
        self.learner = Learner(self.jev, self.memory, self.skills_home, self.cfg.t,
                               lambda kind, data: self.on_event(kind, data))
        self._learn_threads: list[threading.Thread] = []
        self._cancel = threading.Event()             # set by cancel(); every worker of the turn checks it
        self._running = False
        self._workers: set[Worker] = set()           # running workers (their REPLs are killed on cancel)
        self._questions: dict[str, PendingQuestion] = {}
        self._qlock = threading.Lock()
        self._qids = itertools.count(1)

    # -- public ----------------------------------------------------------------------
    def turn(self, message: str, force: str | None = None, resume: dict | None = None) -> TurnResult:
        self.learner.cancel.set()               # background learning yields the model to this turn
        self.turn_no += 1
        self._tl.turn, self._tl.persona, self._tl.task, self._tl.kind = self.turn_no, "", None, "llm"
        self._cancel = threading.Event()
        self._running = True
        try:
            return self._turn(message, force, resume)
        finally:
            self._running = False

    def _turn(self, message: str, force: str | None, resume: dict | None = None) -> TurnResult:
        started = time.time()
        self.jev.trace = []
        self.board = Board(self.cfg.lim("max_depth"), self.cfg.lim("max_tasks_per_turn"))
        recent = self.memory.recent_text(self.session)
        if force:
            if force not in self.personas:
                raise KeyError(f"unknown persona '{force}'")
            route = RouteDecision(force, 1.0, {}, None, "forced by user")
            if recent:
                follows, cont, rep = self._safe("follow_up", lambda: self.jev.follow_up(message, recent),
                                                (False, None, None))
                route.follows, route.continues, route.replies = follows, cont, rep
        else:
            route = self._safe("route", lambda: self.jev.route_persona(
                message, self.personas, self.current, recent),
                RouteDecision(self.current or self.cfg.default_persona, 0.0, {}, None, "jev unavailable"))
        self.on_event("route", {"decision": route})
        root = self.board.open(None, "user", route.persona, message)
        if resume:  # continuing an interrupted task: its own conversation is the context
            route.follows = False
        if route.follows:  # the exchange this answers goes along as real context, not via the memory gate
            root.history = [(t.user, t.assistant, t.notes) for t in self.memory.session_turns(
                self.session, self.cfg.lim("follow_turns"))]
        result = self.run_task(root, goal=message, allow_collab=True, resume=resume)
        cancelled = root.status == "cancelled"
        if root.status == "done":  # a crashed turn's error text must not become "memory"
            self.memory.append(route.persona, message, result.answer, self.session, turn_notes(result))
            if self.learning and result.stopped in ("final", "plain"):
                self._learn(route.persona, message, result)
        elif cancelled:  # kept, so a redirect ("no, just the USPS part") has the request as context
            self.memory.append(route.persona, message, STOPPED_NOTE, self.session)
        self.current = route.persona
        return TurnResult(route.persona, result.answer, result.artifacts, self.board.tasks,
                          list(self.jev.trace), time.time() - started, cancelled,
                          self.usage.summary(self.session, self.turn_no),
                          str(root.checkpoint) if (cancelled or result.stopped in RESUMABLE) and root.checkpoint
                          else None)

    # -- resume ----------------------------------------------------------------------
    def resumable(self, limit: int = 10) -> list[dict[str, Any]]:
        """Interrupted lead tasks (stopped, step limit, stalled, crashed or harness restarted), newest first."""
        out = []
        for ck in sorted((self.cfg.workspace / "sessions").glob("*/*/checkpoint.json"),
                         key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(ck.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if data.get("finished") and data.get("stopped") not in RESUMABLE:
                continue
            if data.get("resumed_by"):
                continue
            task = data.get("task_text", "").split("\n\n(In your REPL:")[0]
            out.append({"path": str(ck), "persona": data.get("persona"), "task": task[:300], "step": data.get("step"),
                        "stopped": data.get("stopped") or "interrupted", "saved": data.get("saved")})
            if len(out) >= limit:
                break
        return out

    def resume(self, path: str | None = None) -> TurnResult:
        """Continue an interrupted task in its own working folder (files intact, REPL fresh)."""
        if path is None:
            found = self.resumable(1)
            if not found:
                raise LookupError("nothing to resume")
            path = found[0]["path"]
        ck = Path(path)
        data = json.loads(ck.read_text(encoding="utf-8"))
        data["checkpoint_path"] = str(ck)
        task = data.get("task_text", "").split("\n\n(In your REPL:")[0]
        result = self.turn(task, force=data["persona"], resume=data)
        try:  # mark the old checkpoint as taken over, so it isn't offered twice
            data.pop("checkpoint_path", None)
            data["resumed_by"] = self.session
            ck.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        return result

    # -- stop / ask ------------------------------------------------------------------
    def cancel(self) -> bool:
        """Stop the running turn: every worker (hand-offs too) stops at its next check, a model
        reply in progress is cut off, running REPL code (and a browser it opened) is killed, and
        anyone waiting in ask_user() is released. Returns False when nothing was running."""
        if self._cancel.is_set() or not self._running:
            return False
        self._cancel.set()
        self.on_event("cancel", {"workers": len(self._workers)})  # first: the stop is often done before the kills return
        for w in list(self._workers):
            if w.repl is not None:
                w.repl.close(kill=True)
        with self._qlock:
            waiting = list(self._questions.values())
        for q in waiting:
            q.event.set()
        return True

    def pending_questions(self) -> list[dict[str, Any]]:
        with self._qlock:
            return [q.public() for q in self._questions.values()]

    def answer(self, qid: str, text: str) -> bool:
        with self._qlock:
            q = self._questions.get(qid)
        if q is None:
            return False
        q.answer = str(text)
        q.event.set()
        return True

    def _ask_user(self, task: Task, persona: Persona, question: str, options: list | None) -> str:
        q = PendingQuestion(f"q{next(self._qids)}", task.id, persona.name, str(question).strip(),
                            [str(o) for o in (options or [])][:8], time.time())
        with self._qlock:
            self._questions[q.id] = q
        self.on_event("question", {"task": task, "question": q.public()})
        deadline = time.monotonic() + self.cfg.lim("ask_timeout")
        try:
            while not q.event.wait(0.5):
                if time.monotonic() > deadline:
                    break
        finally:
            with self._qlock:
                self._questions.pop(q.id, None)
        if self._cancel.is_set():
            raise Cancelled("stopped by the user")
        if q.answer is None:
            self.on_event("question_timeout", {"task": task, "question": q.public()})
            return (f"(no answer from the user within {self.cfg.lim('ask_timeout') // 60} minutes: continue with "
                    f"your best judgement, or finish and say what you need)")
        self.on_event("answered", {"task": task, "question": q.public(), "answer": q.answer})
        return q.answer

    def set_project(self, path: str | None) -> str | None:
        """Open a project folder for personas with the `project` tool (None closes it)."""
        if path:
            p = Path(path).expanduser().resolve()
            if not p.is_dir():
                raise NotADirectoryError(str(p))
            self.project = str(p)
        else:
            self.project = None
        return self.project

    def wait_learning(self, timeout: float | None = None) -> None:
        for t in list(self._learn_threads):
            t.join(timeout)

    def _learn(self, persona: str, message: str, result: WorkerResult) -> None:
        if result.llm is None:
            return
        used = [self.roster[n] for n in result.skills if n in self.roster]

        self.learner.cancel = threading.Event()  # a fresh flag for this learning run
        turn_no = self.turn_no

        def run() -> None:
            self._tl.turn, self._tl.persona, self._tl.task, self._tl.kind = turn_no, persona, None, "learn"
            out = self.learner.after_turn(persona, message, result.answer, result.transcript, result.steps,
                                          used, result.llm, self.session)
            if out.skill:  # a new or improved skill is picked up from the next turn on
                self.roster = load_roster(self.cfg.skills_dirs, self.skills_home / "_lessons")

        if self.learn_sync:
            run()
            return
        t = threading.Thread(target=run, name="learn", daemon=False)
        self._learn_threads = [x for x in self._learn_threads if x.is_alive()] + [t]
        t.start()

    # -- tasks -----------------------------------------------------------------------
    def run_task(self, task: Task, goal: str, allow_collab: bool, resume: dict | None = None) -> WorkerResult:
        persona = self.personas[task.to]
        turn_no = getattr(self._tl, "turn", self.turn_no)
        self._tl.turn, self._tl.persona, self._tl.task, self._tl.kind = turn_no, persona.name, task.id, "llm"
        skills = self._safe("skills", lambda: self.jev.pick_skills(task.jev_text, persona, self.roster),
                            SkillDecision([]))
        self.on_event("skills", {"task": task, "decision": skills})
        candidates = self.memory.candidates(persona.name, self.cfg.lim("memory_candidates_own"),
                                            self.cfg.lim("memory_candidates_shared"), task.jev_text,
                                            self.cfg.lim("memory_search"),
                                            self.cfg.lim("memory_facts"))
        if task.history:  # those turns are already in the conversation itself
            said = [f"User: {h[0]}\n" for h in task.history]
            candidates = [c for c in candidates if not any(s in c for s in said)]
        mem = self._safe("memory", lambda: self.jev.gate_context(task.jev_text, candidates),
                         MemoryDecision([], []))
        self.on_event("memory", {"task": task, "decision": mem, "candidates": len(candidates)})
        collab = CollabDecision([], {})
        if allow_collab and persona.code:
            collab = self._safe("collab", lambda: self.jev.collab(task.jev_text, persona, self.personas), collab)
            self.on_event("collab", {"task": task, "decision": collab})
        requestable = {n: p.description for n, p in self.personas.items()
                       if persona.may_request(n) or n in collab.suggested}
        tools = self._pick_tools(task, persona)
        self.on_event("tools", {"task": task, "decision": tools})
        catalog = self.mcp.catalog() if tools.mcp else {}
        picked = [self.roster[s] for s in skills.skills]
        system = build_system(persona, picked, requestable, collab.suggested, tools.tools,
                              [catalog[k] for k in tools.mcp if k in catalog])
        ctx = {"request": task.text, "memory": mem.kept, "inputs": task.inputs,
               "skills": {s.name: s.body for s in picked}}
        env = {}
        if "browser" in persona.tools:
            env["JEVH_BROWSER_HEADED"] = "1" if self.watch_browser else "0"
        if self.project and "project" in persona.tools:
            ctx["project"] = self.project
            env["JEVH_PROJECT"] = self.project
            env["JEVH_RULE_BLOCK"] = str(self.cfg.t("rule_block"))
        tier = self._pick_tier(task, persona)
        self.on_event("tier", {"task": task, "decision": tier})
        llm = self._llm(persona, tier.tier)
        if getattr(llm, "native_tools", False):   # acting is a `python` tool call, not a fenced block
            system = build_system(persona, picked, requestable, collab.suggested, tools.tools,
                                  [catalog[k] for k in tools.mcp if k in catalog], native=True)
        handles = Handles(self._start_request, task, goal, requestable)
        handlers = {
            "llm_query": lambda prompt, data=None: llm.chat([{"role": "user", "content": (
                prompt if data is None else f"{prompt}\n\n---\n{data}")}]).text,
            "jev": lambda state, questions: self.jev._ask("worker_jev", state, questions).as_dict(),
            "request": lambda to, text, inputs=None: handles.wait(handles.start(to, text, inputs)),
            "request_async": lambda to, text, inputs=None: handles.start(to, text, inputs),
            "wait": lambda h, timeout=None: handles.wait(h, timeout),
            "mcp": lambda key, arguments: self._mcp_call(persona, key, arguments),
            "recall": lambda query, k=5: self._recall(query, k),
            "remember": lambda fact: self.learner.remember(str(fact), f"{self.session}/{persona.name}"),
            "ask_user": lambda question, options=None: self._ask_user(task, persona, question, options),
        }
        workdir = Path(resume["workdir"]) if resume and resume.get("workdir") else \
            self.session_dir / f"u{turn_no}t{task.id}-{persona.name}"   # task ids restart every turn
        task.checkpoint = workdir / "checkpoint.json"
        worker = Worker(persona, llm, workdir, self.artifacts,
                        handlers, {t: f"jevh.tools.{t}" for t in persona.tools},
                        self.cfg.lim("repl_timeout"), self.cfg.lim("output_cap"),
                        lambda kind, data: self._on_worker(kind, data, task), env,
                        self._cancel, self._verify_fn(task) if persona.code and self.verifier.enabled else None,
                        int(self.verifier.settings.get("max_bounces", 2)),
                        self._progress_fn(task), self.cfg.t("stall"), self.cfg.lim("progress_every"),
                        self.cfg.lim("compact_tokens"), self.cfg.lim("keep_steps"), task.checkpoint)
        worker.keep_reasoning = self.cfg.lim("keep_reasoning")
        worker.step_value, worker.keep_at = self._compact_fn(task), self.cfg.t("compact_keep")
        if self.cfg.lim("thinking_watch"):
            worker.thinking_watch = self._watch_fn(task)
            worker.watch_every, worker.watch_after = self.cfg.lim("watch_every"), self.cfg.lim("watch_after")
        if tier.tier == "fast" and self.tier_mode != "fast":   # quality first: a bounced fast answer escalates
            worker.escalate = lambda: self._llm(persona, "strong")
        task.started = time.time()
        self.on_event("start", {"task": task, "model": getattr(llm, "name", "?")})
        self._workers.add(worker)
        try:
            if self._cancel.is_set():
                raise Cancelled("stopped by the user")
            result = worker.run(task.text, ctx, system, task.history, resume)
            result.skills, result.llm = list(skills.skills), worker.llm   # the loaded one, if it escalated
            if task.parent is None:   # its follow-ups start on the strong model: no swap back and forth
                self._escalated = worker.escalated
            task.status = "done"
        except Cancelled:
            result = WorkerResult(answer=STOPPED_NOTE, stopped="cancelled")
            task.status = "cancelled"
        except Exception as e:  # noqa: BLE001 - a dead model server must not kill the harness
            if self._cancel.is_set():  # the stop killed the REPL under it mid-call: that's the stop, not a failure
                result = WorkerResult(answer=STOPPED_NOTE, stopped="cancelled")
                task.status = "cancelled"
            else:
                result = WorkerResult(answer=f"[{persona.name} failed: {type(e).__name__}: {e}]",
                                      stopped="error")
                task.status = "failed"
        finally:
            self._workers.discard(worker)
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

        if self._cancel.is_set():
            return rejected("the user stopped this turn")
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
    def _where(self) -> dict[str, Any]:
        return {"turn": getattr(self._tl, "turn", self.turn_no), "persona": getattr(self._tl, "persona", ""),
                "task": getattr(self._tl, "task", None), "kind": getattr(self._tl, "kind", "llm")}

    def _record(self, **kw) -> None:
        w = self._where()
        kw.setdefault("persona", w["persona"])
        kw.setdefault("task", w["task"])
        try:
            self.usage.record(session=self.session, turn=w["turn"], **kw)
        except Exception:  # noqa: BLE001 - accounting must never break a turn
            pass

    def _on_worker(self, kind: str, data: dict, task: Task) -> None:
        if kind == "tool_usage":
            self._record(kind="jev_tools", model=str(self.cfg.jev.get("backend", "")), calls=data.get("calls", 0),
                         input_tokens=data.get("tokens", 0), seconds=data.get("seconds", 0.0))
            return
        self.on_event(kind, {**data, "task_id": task.id})

    def _pick_tools(self, task: Task, persona: Persona) -> ToolDecision:
        """Which tool guides (and MCP tools) this task's prompt carries."""
        if not persona.code:
            return ToolDecision([])
        own = {t: TOOLS[t].description for t in persona.tools
               if t in TOOLS and (t != "project" or self.project)}
        mcp = {}
        if persona.mcp and self.mcp:
            mcp = {k: t.description for k, t in self.mcp.catalog().items() if persona.allows_mcp(k)}
        if not own and not mcp:
            return ToolDecision([])
        fallback = ToolDecision(list(own))  # Jev down: all guides (capability beats a lean prompt)
        return self._safe("tools", lambda: self.jev.pick_tools(task.jev_text, own, mcp), fallback)

    def _progress_fn(self, task: Task):
        def circles(log: str) -> float:
            r = self.jev._ask("progress", {"request": task.text[:1500], "work_log": log[-4000:]},
                              {"circles": noul(CIRCLES_Q)})
            return r["circles"].noul
        return circles

    def _compact_fn(self, task: Task):
        """-> (old step texts, where the work is now) -> P(the rest of the task needs each step).
        One call per step: batching the steps into one call scored them lower (10/18 needed kept at
        0.8 vs 16/18) and was no faster (~0.2 s a step either way)."""
        def value(steps: list[str], now: str) -> list[float]:
            state = {"task": task.text[:1500], "now": now[-2000:]}
            return [self.jev._ask("compact", {**state, "step": s[:4500]}, {"info": noul(COMPACT_Q)})["info"].noul
                    for s in steps]
        return value

    def _watch_fn(self, task: Task):
        """-> (reasoning so far) -> pause? Jev on the end of a long think (see TESTABLE_Q)."""
        def testable(thinking: str) -> bool:
            r = self.jev._ask("thinking_watch", {"task": task.text[:1500], "thinking": thinking[-2500:]},
                              {"testable": noul(TESTABLE_Q)})
            self.on_event("thinking_watch", {"task": task, "p": round(r["testable"].noul, 3),
                                             "chars": len(thinking)})
            return r["testable"].noul >= self.cfg.t("testable")
        return testable

    def _verify_fn(self, task: Task):
        project = self.project

        def check(answer: str, artifacts: list[str], transcript: list[dict], changed: list[str],
                  task_idx: int | None = None):
            return self.verifier.check(task.text, answer, artifacts, work_log(transcript, answer=answer),
                                       project, changed, evidence=seen_text(transcript, task_idx))
        return check

    def _mcp_call(self, persona: Persona, key: str, arguments: dict) -> str:
        if not persona.allows_mcp(key):
            raise PermissionError(f"{persona.name} may not use MCP tool {key}")
        return self.mcp.call(key, dict(arguments or {}))

    def _recall(self, query: str, k: int = 5) -> list[str]:
        chunks = [t.chunk() for t in self.memory.search(query, 24)] + [f.chunk() for f in self.memory.facts(query, 8)]
        if not chunks:
            return []
        dec = self._safe("recall", lambda: self.jev.gate_context(query, chunks), MemoryDecision(chunks[:k], []))
        return dec.kept[:k]

    def _pick_tier(self, task: Task, persona: Persona) -> TierDecision:
        if self.model_override:
            return TierDecision("strong", None, "--model override")
        if not self.cfg.models.get("fast") or not persona.fast_ok:
            return TierDecision("strong", None, "no fast tier for this persona")
        if self.tier_mode in ("fast", "strong"):
            return TierDecision(self.tier_mode, None, "forced by user")
        if task.history and self._escalated:
            return TierDecision("strong", None, "follow-up to a turn the fast model couldn't finish")
        # Jev down -> persona's own model: a slow right answer beats a fast wrong one
        return self._safe("tier", lambda: self.jev.pick_tier(task.jev_text, persona),
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
                self._llms[key] = Metered(llm, self._record, self._where, tier,
                                          str(self._model_cfg(persona, tier).get("model", "")))
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
        cfg = {k: v for k, v in persona.model.items() if k not in ("fallback", "effort", "thinking", "think",
                                                                   "max_tokens")}
        cfg.update(swap)
        return cfg

    def _safe(self, step: str, fn: Callable[[], Any], fallback: Any) -> Any:
        """Jev down -> degrade (default persona, no skills/memory/collab), never crash."""
        try:
            return fn()
        except SystemOneError as e:
            self.on_event("jev_error", {"step": step, "error": str(e)})
            return fallback
