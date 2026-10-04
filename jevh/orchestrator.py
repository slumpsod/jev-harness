"""One user turn: route -> gate memory -> pick skills -> suggest collaborators -> run
the worker, serving its request() calls through Jev's approval on the task board.

Hand-offs run in parallel: each approved sub-task gets its own thread (and its own REPL
process); request() is request_async() + wait()."""
from __future__ import annotations

import itertools
import json
import os
import re
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

from . import interject
from .bus import Board, LimitError, Task
from .config import Config, load_config
from .decide import SystemOneError, noul
from .errors import Cancelled
from .failures import FailureLog, FailureWatch
from .forge import Forge
from .jev import (CollabDecision, Jev, MemoryDecision, RouteDecision, SkillDecision, TierDecision, ToolDecision,
                  decide_interject)
from .learn import Learner, seen_text, work_log
from .llm import LLM, make_llm, see_image
from .lmstudio import gate_for
from .mcp import McpHub
from .memory import Memory
from .personas import Persona, load_learned, load_personas
from .notebook import Notebook
from .review import Reviewer
from .skills import load_roster, parse_skill
from .tools import TOOLS
from .tools import nango
from .usage import Metered, Usage
from .verify import DELIVERABLE, Verifier, asks_for_secret
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
# ...and a think that is searching - theory after theory - rather than checking one idea. When checking,
# "compute this" was answered with code every time (the 8.3-min ARC run: 3/3); when searching, pauses
# were ignored 4 in a row and TESTABLE_Q mostly didn't fire (0.04-0.22 on 28 KB of theorising, and on a
# verbatim loop). Neither question separates the two alone - Qwen's checking is full of "hmm, wait" too
# - but "stuck" minus "checking" > 0.25 pauses the searching the watch let through: tuning 8/8 (1 extra
# pause while checking), held out 2/2 (2 extra); evals/search_cases.py (real thinking, 8 runs).
STUCK_Q = ("Is `thinking` stuck searching for a rule - trying idea after idea - rather than carefully checking "
           "one specific idea against the data?")
CHECKING_Q = "Is `thinking` checking one specific idea step by step against the data?"
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
    secret: bool = False     # a password or key: typed hidden, and never echoed into events
    event: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "task_id": self.task_id, "persona": self.persona, "question": self.question,
                "options": self.options, "asked": self.asked, "secret": self.secret}


class Harness:
    def __init__(self, cfg: Config | None = None, jev: Jev | None = None,
                 llm_factory: Callable[[Persona], LLM] | None = None, on_event: Event | None = None,
                 session: str | None = None, model_override: str | None = None):
        self.cfg = cfg or load_config()
        self.jev = jev or Jev(self.cfg)
        # a persona, skill or MCP server that can't be used is skipped and listed here, not fatal
        self.load_problems: list[dict] = []
        self.personas = load_personas(self.cfg.personas_dirs, self.load_problems)
        if llm_factory is None and not model_override:
            for name, p in list(self.personas.items()):
                if p.model.get("provider") not in self.cfg.providers:
                    self.load_problems.append({"kind": "persona", "name": name, "path": "", "skipped": True,
                                               "error": f"unknown model provider '{p.model.get('provider')}' "
                                                        f"(providers: {', '.join(self.cfg.providers) or 'none'})"})
                    if len(self.personas) > 1:
                        del self.personas[name]
        if self.cfg.default_persona not in self.personas:
            first = next(iter(self.personas))
            self.load_problems.append({"kind": "persona", "name": self.cfg.default_persona, "path": "", "skipped": True,
                                       "error": f"the default persona isn't available; using {first} instead"})
            self.cfg.default_persona = first
        self.skills_home = self.cfg.skills_dirs[0]    # where learned skills and lessons are written
        # lesson notes: yours (the first skills folder) and any shipped with a skills folder
        self.lesson_dirs = [Path(d) / "_lessons" for d in self.cfg.skills_dirs]
        self.roster = load_roster(self.cfg.skills_dirs, self.lesson_dirs, self.load_problems)
        self.memory = Memory(self.cfg.workspace / "memory")
        self.mcp = McpHub(self.cfg.mcp_servers)
        self.project: str | None = None             # the user's project folder, if one is open
        self.learning = bool(self.cfg.raw.get("learning", True))
        self.learn_sync = False                     # tests: learn inline instead of on a thread
        self.session = session or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.session_dir = self.cfg.workspace / "sessions" / self.session
        self.artifacts = self.session_dir / "artifacts"
        self.current: str | None = None
        ui = _shielded(on_event or (lambda kind, data: None))
        # every event also passes the failure watch, which logs what went wrong (and runs the hooks)
        self.failures = FailureLog(self.cfg.workspace / "failures.db")
        self.failure_watch = FailureWatch(
            self.failures, self.cfg.raw.get("failures") or {}, self.session, self._where,
            lambda tid: next((t for t in self.board.tasks if t.id == tid), None),
            lambda step, state, questions: self.jev._ask(step, state, questions), ui, self.cfg.t, self.cfg.root,
            spawn=self._background, audit_llm=self._audit_llm)

        def emit(kind: str, data: dict) -> None:
            ui(kind, data)
            self.failure_watch.on_event(kind, data)
        self.on_event = emit
        self.model_override = model_override
        self.tier_mode = "auto"  # auto (Jev decides) | fast | strong
        self._escalated = False  # the last turn's fast model was replaced by the strong one
        # show the browser tool's window, with Jev's picks outlined (dashboard toggle, CLI /watch)
        self.watch_browser = bool(self.cfg.tools.get("browser_watch")) or os.environ.get("JEVH_BROWSER_HEADED") == "1"
        self.browser_dock = False   # the dashboard sets it: watched browsers run hidden and show in its Browser panel
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
                         ("tools_jev", "JEVH_TOOLS_JEV"), ("nango_url", "JEVH_NANGO_URL"),
                         ("nango_writes", "JEVH_NANGO_WRITES")):
            if self.cfg.tools.get(key):  # REPL children inherit these
                os.environ.setdefault(env, str(self.cfg.tools[key]))
        os.environ["JEVH_NANGO_DIR"] = str(self.cfg.workspace / "nango")   # the key/URL nango.start() keeps
        os.environ.setdefault("JEVH_JEV_URL", str(self.cfg.jev.get("url", "http://127.0.0.1:33001")))
        os.environ.setdefault("JEVH_JEV_BACKEND", str(self.cfg.jev.get("backend", "semif")))
        self.verifier = Verifier(self.jev, self.cfg)
        self.usage = Usage(self.cfg.workspace / "usage.db", self.cfg.raw.get("prices"))
        # sessions the user flagged for review, with their notes; a model can judge one (jevh/review.py)
        self.review = Reviewer(self.cfg, self.memory, self.failures, model_override=model_override)
        # the user's own ideas and problems to come back to; nothing in it reaches a worker (jevh/notebook.py)
        self.notebook = Notebook(self.cfg.workspace / "notebook.db")
        self.turn_no = 0
        self.turns_seen: dict[int, dict[str, Any]] = {}   # turn -> request, answer, lead task's folder (for flag())
        self._tl = threading.local()                 # which turn/task/persona this thread works for
        self.jev.on_call = lambda step, resp, n: self._record(
            kind="jev", model=resp.model, input_tokens=resp.input_tokens, seconds=resp.latency_s,
            persona=getattr(self._tl, "persona", ""), task=getattr(self._tl, "task", None))
        self.learner = Learner(self.jev, self.memory, self.skills_home, self.cfg.t,
                               lambda kind, data: self.on_event(kind, data))
        # a persona drafted when a request or a learned skill fits none of them (jevh/forge.py)
        self.forge = Forge(self.cfg, self.jev, self.memory, self.skills_home, self.learner._chat,
                           lambda kind, data: self.on_event(kind, data))
        self._learn_threads: list[threading.Thread] = []
        self._cancel = threading.Event()             # set by cancel(); every worker of the turn checks it
        self._running = False
        self._workers: set[Worker] = set()           # running workers (their REPLs are killed on cancel)
        self._questions: dict[str, PendingQuestion] = {}
        self._qlock = threading.Lock()
        self._qids = itertools.count(1)
        # messages the user sends while a turn runs (jevh/interject.py)
        self._ilock = threading.Lock()
        self._phase = "idle"                         # idle | preparing (routing, picking) | working | finishing
        self._lead: Worker | None = None             # the running turn's lead worker: it hears the user's messages
        self._lead_task: Task | None = None
        self._early: list[tuple[str, bool]] = []     # corrections (text, replaces) sent before the lead worker exists
        self._queue: list[str] = []                  # requests sent while a turn ran: they run after it
        self._said: list[str] = []                   # this turn's corrections (kept with the turn in memory)
        self._goal = ""                               # the running turn's message (what a correction is judged against)

    # -- public ----------------------------------------------------------------------
    def problems(self) -> list[dict]:
        """Add-ons that were skipped: personas, skills, and MCP servers that failed to start."""
        return self.load_problems + [{"kind": "mcp", "name": n, "path": "", "error": e, "skipped": True}
                                     for n, e in self.mcp.errors.items()]

    def turn(self, message: str, force: str | None = None, resume: dict | None = None) -> TurnResult:
        self.learner.cancel.set()               # background learning yields the model to this turn
        self.turn_no += 1
        self._tl.turn, self._tl.persona, self._tl.task, self._tl.kind = self.turn_no, "", None, "llm"
        self._cancel = threading.Event()
        with self._ilock:
            self._phase, self._lead, self._lead_task, self._early, self._said = "preparing", None, None, [], []
            self._goal = message
        self._running = True
        try:
            return self._turn(message, force, resume)
        finally:
            self._running = False
            with self._ilock:   # a message that never reached the task is the next request, in the order it came
                left, self._early = [t for t, _ in self._early], []
                self._phase, self._lead, self._lead_task = "idle", None, None
                self._queue[0:0] = left
            if left:
                self.on_event("queued", {"texts": left, "late": True, "queue": list(self._queue)})

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
            self.memory.append(route.persona, message, result.answer, self.session,
                               turn_notes(result) + self._said_notes())
            learn = self.learning and result.stopped in ("final", "plain")
            # the catch-all persona ended up with it (the router's confidence in the catch-all says nothing: it
            # is the persona everything lands on): Jev decides, after the turn, whether any assistant is for it
            check_fit = self.forge.enabled and not force and route.persona == self.cfg.default_persona
            if learn or check_fit:
                self._learn(route.persona, message, result, learn, check_fit)
        elif cancelled:  # kept, so a redirect ("no, just the USPS part") has the request as context
            self.memory.append(route.persona, message, STOPPED_NOTE, self.session, self._said_notes().strip())
        self.current = route.persona
        self.turns_seen[self.turn_no] = {   # what flag() needs to log this turn as wrong later
            "request": message, "answer": result.answer, "persona": route.persona, "task": root.id,
            "artifacts": list(result.artifacts), "workdir": str(root.checkpoint.parent) if root.checkpoint else "",
            "model": self.failure_watch._models.get((self.turn_no, root.id), "")}
        for old in [k for k in self.turns_seen if k <= self.turn_no - 200]:
            del self.turns_seen[old]
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
        with self._ilock:   # Stop means stop: what was queued behind this turn goes too (the event hands it back)
            dropped, self._queue, self._early = [t for t, _ in self._early] + self._queue, [], []
        self.on_event("cancel", {"workers": len(self._workers), "dropped": dropped})  # first: the stop is often done before the kills return
        for w in list(self._workers):
            if w.repl is not None:
                w.repl.close(kill=True)
        with self._qlock:
            waiting = list(self._questions.values())
        for q in waiting:
            q.event.set()
        return True

    # -- messages sent while a turn runs (jevh/interject.py, docs/architecture/16-interjection.md) -------
    def queued(self) -> list[str]:
        """Requests sent while a turn ran, in order: each runs as its own turn once this one is over."""
        with self._ilock:
            return list(self._queue)

    def take_queued(self) -> str | None:
        """The next queued request (removed), or None. Whoever runs turns calls this when one ends."""
        with self._ilock:
            return self._queue.pop(0) if self._queue else None

    def enqueue(self, text: str) -> None:
        """Queue a request for after the current turn, for a surface that finds itself between two turns."""
        with self._ilock:
            self._queue.append(text)
        self.on_event("queued", {"texts": [text], "queue": self.queued()})

    def status(self) -> dict[str, Any] | None:
        """What the running turn is doing right now, or None when idle (interject.status_line words it)."""
        with self._ilock:
            phase, w = self._phase, self._lead
        if phase == "idle":
            return None
        if phase != "working" or w is None:
            doing = ("getting ready: choosing the assistant, skills and tools" if phase == "preparing"
                     else "finishing up: collecting its answer")
            return {"persona": None, "step": 0, "max_steps": 0, "doing": doing, "seconds": 0.0, "corrections": [],
                    "pending": False}
        return w.status()

    def _said_notes(self) -> str:
        """What the user added while the turn ran, for its memory record ("what did I tell you earlier?")."""
        if not self._said:
            return ""
        return "\n[While this ran, the user also said: " + " | ".join(f'"{s[:300]}"' for s in self._said) + "]"

    def interject(self, text: str, kind: str | None = None) -> dict[str, Any]:
        """A message the user sent while a turn runs. By default it is added to the running task's prompt, except a
        question about progress (Jev tells), which is answered on the side. `kind`: "correction" | "status" |
        "separate" | "stop", or None to let a /command in the text, else Jev, decide - Jev never picks separate or
        stop: queueing and stopping are the /queue and /stop commands. Returns {"outcome": ..., "kind": ...}:
          idle       nothing is running: send it as an ordinary message
          corrected  the lead worker hears it: a model reply in progress is cut and asked for again with it
          status     a question about progress: "answer" is the instant line; the task's model answers too, a
                     little later, as an "aside" event
          queued     (/queue, or a message that arrived after the task was done) another request: it runs as the
                     next turn
          stopped    (/stop) the turn was stopped
        The "interject" event carries the same, for the dashboard."""
        cfg = interject.settings(self.cfg.raw)
        command, body = interject.parse(text)
        text = body if command else str(text or "").strip()
        kind = kind or command
        if not self._running:
            return {"outcome": "idle", "text": text}
        if not cfg["enabled"]:
            return {"outcome": "refused", "text": text, "reason": "interject.enabled is false"}
        if not text and kind not in ("stop", "status"):
            return {"outcome": "empty", "text": text}
        st = self.status() or {}
        probs: dict[str, float] = {}
        if kind is None:
            if cfg["classify"]:
                task = self._lead_task.current if self._lead_task else self._goal
                probs = self._safe("interject", lambda: self.jev.interject_kind(text, task, st.get("doing", "")), {})
                kind = decide_interject(probs, self.cfg.t("interject_status"))
            else:
                kind = "correction"
        out: dict[str, Any] = {"outcome": "", "kind": kind, "text": text, "probs": probs}
        if kind == "stop":
            self.cancel()
            out["outcome"] = "stopped"
        elif kind == "separate":
            with self._ilock:
                self._queue.append(text)
                out.update(outcome="queued", queue=list(self._queue))
        elif kind == "status":
            out.update(outcome="status", answer=interject.status_line(st, len(self.queued())),
                       aside=bool(cfg["aside_model"]))   # (the task's own answer follows as an "aside" event)
        else:
            # a clear swap ("check my calendar instead") replaces the request; anything else adds to it
            replaces = bool(probs) and probs.get("replace", 0.0) >= self.cfg.t("interject_replace")
            out["replaces"] = replaces
            out["outcome"] = self._correct(text, bool(cfg["redirect"]), replaces)
            out["queue"] = self.queued()
        out["turn"] = self.turn_no
        self.on_event("interject", dict(out))
        if kind == "status" and cfg["aside_model"]:
            self._aside_async(text or "What is going on right now?")
        return out

    def _correct(self, text: str, redirect: bool, replaces: bool = False) -> str:
        """Give the lead worker a correction. Before it exists the message waits for it; once it is done there is
        no task left to correct, so the message becomes the next request."""
        with self._ilock:
            phase, w = self._phase, self._lead
            if phase == "preparing":
                self._early.append((text, replaces))
                return "corrected"
            if phase != "working" or w is None:
                self._queue.append(text)
                return "queued"
        w.steer(text, abort=redirect, replaces=replaces)
        return "corrected"

    def aside(self, question: str) -> dict[str, Any]:
        """The quick reply: the task's own model answers a question about the running task from what the worker
        has done so far, on a copy of it, while the worker carries on. Blocks for the reply (it is a model
        call): use _aside_async from a surface. Always returns an answer: if the model cannot give one (no
        worker yet, an error, it spent its budget thinking) the instant status line is what comes back."""
        st = self.status()
        if st is None:
            return {"answer": "Nothing is running.", "source": "none"}
        quick = interject.status_line(st, len(self.queued()))
        cfg = interject.settings(self.cfg.raw)
        with self._ilock:
            w, task = self._lead, self._lead_task
        if not cfg["aside_model"] or w is None or task is None:
            return {"answer": quick, "source": "state"}
        mode = cfg["aside_context"] if cfg["aside_context"] in ("digest", "full") else "digest"
        started = time.time()
        try:
            messages = interject.aside_messages(mode, w.persona.name, task.current, st["doing"], question,
                                                w.digest(), w.snapshot())
            tools = w.tool_schemas() if mode == "full" else None
            out = w.llm.chat(messages, cancel=self._cancel, **({"tools": tools} if tools else {}),
                             body_extra={"max_tokens": int(cfg["aside_max_tokens"]), **(cfg.get("aside_extra") or {})})
        except Cancelled:
            return {"answer": quick, "source": "state", "error": "stopped"}
        except Exception as e:  # noqa: BLE001 - a quick reply must never become a second problem
            return {"answer": quick, "source": "state", "error": f"{type(e).__name__}: {e}"[:200]}
        answer = (out.text or "").strip()
        if not answer:
            return {"answer": quick, "source": "state", "error": "the model gave no answer (spent its budget thinking?)",
                    "seconds": round(time.time() - started, 1), "context": mode}
        return {"answer": answer, "source": "model", "seconds": round(time.time() - started, 1), "context": mode,
                "input_tokens": out.input_tokens, "output_tokens": out.output_tokens}

    def _aside_async(self, question: str) -> None:
        turn_no = self.turn_no
        with self._ilock:
            lead, task = self._lead, self._lead_task

        def run() -> None:
            self._tl.turn, self._tl.persona, self._tl.task, self._tl.kind = (
                turn_no, lead.persona.name if lead else "", task.id if task else None, "aside")
            res = self.aside(question)
            self.on_event("aside", {"question": question, "turn": turn_no, **res})
        threading.Thread(target=run, name="aside", daemon=True).start()

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

    def _ask_user(self, task: Task, persona: Persona, question: str, options: list | None,
                  secret: bool = False) -> str:
        # a free-text question that asks for a password or key is asked hidden whatever the worker passed: the
        # answer would otherwise sit in the transcript, memory and logs (buttons: an approval, left alone)
        secret = bool(secret) or (not options and asks_for_secret(str(question)))
        q = PendingQuestion(f"q{next(self._qids)}", task.id, persona.name, str(question).strip(),
                            [] if secret else [str(o) for o in (options or [])][:8], time.time(), secret=bool(secret))
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
        self.on_event("answered", {"task": task, "question": q.public(),
                                   "answer": "(hidden)" if q.secret else q.answer})
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

    def flag(self, note: str = "", turn: int | None = None) -> dict[str, Any] | None:
        """The user says a turn's answer was wrong (dashboard thumbs-down, CLI /bad): log it as a failure,
        with the lead task's work log from its checkpoint, so the refinement loop can work on it. Confident
        wrong answers are the failures no detector logs (live: a regex test summary that miscounted its own
        cases, a handed-over CSV that held only its file name). -> the failure row, or None (no such turn)."""
        t = turn or self.turn_no
        seen = self.turns_seen.get(t)
        if seen is None:
            return None
        from .failures import from_workdir
        task = {}
        if seen["workdir"] and (Path(seen["workdir"]) / "checkpoint.json").is_file():
            try:
                task = from_workdir(Path(seen["workdir"]))
            except (OSError, ValueError):
                task = {}
        note = str(note or "").strip()
        files = ", ".join(Path(a).name for a in seen["artifacts"])
        row = self.failures.record(
            source="user", kind="user_flagged", summary=(note or "the user marked this answer as wrong")[:500],
            session=self.session, turn=t, task=seen["task"], persona=seen["persona"], model=seen["model"],
            request=seen["request"], answer=seen["answer"], work_log=task.get("work_log", ""), workdir=seen["workdir"],
            detail=(f"The user's note: {note}\n" if note else "") + (f"Files handed over: {files}" if files else ""))
        self.on_event("failure", {"failure": {k: row[k] for k in ("id", "source", "kind", "summary", "persona", "task")}})
        return row

    def _background(self, kind: str, fn, task: Task) -> None:
        """Run fn after the task, on its own thread, metered as `kind` for that task (spot checks); joined
        by wait_learning() like learning is."""
        turn_no = getattr(self._tl, "turn", self.turn_no)

        def run() -> None:
            self._tl.turn, self._tl.persona, self._tl.task, self._tl.kind = turn_no, task.to, task.id, kind
            fn()
        if self.learn_sync:
            run()
            return
        t = threading.Thread(target=run, name=kind, daemon=False)
        self._learn_threads = [x for x in self._learn_threads if x.is_alive()] + [t]
        t.start()

    def _audit_llm(self, persona: str, own):
        """The model a spot check uses: config failures.audit_model "same" (the task's own: no model swap)
        or "strong" (the persona's strong-tier model, for tasks the fast tier did)."""
        mode = str((self.cfg.raw.get("failures") or {}).get("audit_model") or "same")
        if mode == "strong" and persona in self.personas:
            return self._llm(self.personas[persona], "strong")
        return own

    def wait_learning(self, timeout: float | None = None) -> None:
        for t in list(self._learn_threads):
            t.join(timeout)

    def _learn(self, persona: str, message: str, result: WorkerResult, learn: bool = True,
               check_fit: bool = False) -> None:
        """After a turn, off the critical path: learn (facts, skills), then let the forge see what
        was learned (a skill no persona wants) and the request itself (when `check_fit` and Jev says no
        persona, a created one included, is for it)."""
        if result.llm is None:
            return
        used = [self.roster[n] for n in result.skills if n in self.roster]

        self.learner.cancel = threading.Event()  # a fresh flag for this learning run
        turn_no = self.turn_no

        def run() -> None:
            self._tl.turn, self._tl.persona, self._tl.task, self._tl.kind = turn_no, persona, None, "learn"
            out = None
            if learn:
                out = self.learner.after_turn(persona, message, result.answer, result.transcript, result.steps,
                                              used, result.llm, self.session)
                if out.skill:  # a new or improved skill is picked up from the next turn on
                    self.roster = load_roster(self.cfg.skills_dirs, self.lesson_dirs)
            made = False
            try:
                if out and out.skill and out.skill_action == "created":
                    skill = parse_skill(Path(out.skill))
                    made = bool(skill and self.forge.note_skill(skill, result.llm))
                if check_fit and self._unfit(message) and self.forge.note_no_fit(
                        message, result.llm, f"{self.session}/{persona}"):
                    made = True
            except Cancelled:
                self.on_event("learn_error", {"error": "persona creation skipped: a new turn started"})
            except Exception as e:  # noqa: BLE001 - best effort; never break a turn
                self.on_event("learn_error", {"error": f"persona creation: {type(e).__name__}: {e}"})
            if made:
                self.reload_personas()

        if self.learn_sync:
            run()
            return
        t = threading.Thread(target=run, name="learn", daemon=False)
        self._learn_threads = [x for x in self._learn_threads if x.is_alive()] + [t]
        t.start()

    def _unfit(self, message: str) -> bool:
        """Jev says no assistant (a created persona included) is for this request."""
        fit = self.jev.no_fit(message, self.personas)
        return fit is not None and fit >= self.cfg.t("no_fit")

    def forge_sweep(self) -> list:
        """Look at every learned skill that has no owner for one no persona wants (CLI /forge, dashboard)."""
        persona = self.personas.get(self.cfg.default_persona) or next(iter(self.personas.values()))
        self.learner.cancel = threading.Event()
        self._tl.turn, self._tl.persona, self._tl.task, self._tl.kind = self.turn_no, persona.name, None, "learn"
        try:
            return self.forge.sweep(self.roster, self._llm(persona))
        finally:
            self.reload_personas()

    def reload_personas(self) -> None:
        """Pick up personas the forge created, approved or deleted (the hand-written ones are untouched,
        so changes made to them in memory stay), and the ownership it gave learned skills."""
        fresh = load_personas(self.cfg.personas_dirs)
        learned = {p.name for p in load_learned(self.cfg.personas_dirs)}
        # one assignment, so a turn starting meanwhile never finds a persona missing
        self.personas = {n: (p if n in learned else self.personas.get(n, p)) for n, p in fresh.items()}
        self.roster = load_roster(self.cfg.skills_dirs, self.lesson_dirs)

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
        notes = {"nango": nango.summary()} if "nango" in tools.tools else {}
        system = build_system(persona, picked, requestable, collab.suggested, tools.tools,
                              [catalog[k] for k in tools.mcp if k in catalog], tool_notes=notes)
        ctx = {"request": task.text, "memory": mem.kept, "inputs": task.inputs,
               "skills": {s.name: s.body for s in picked}}
        env = {}
        if "browser" in persona.tools:
            env["JEVH_BROWSER_HEADED"] = "1" if self.watch_browser else "0"
            # the sites the request names: goto() points out a mistyped one (live: "blavedemo.com")
            env["JEVH_TASK_HOSTS"] = " ".join(dict.fromkeys(
                re.findall(r"(?:https?://)?((?:[a-z0-9-]+\.)+[a-z]{2,})(?=[/:\s)\"',]|$)", task.text.lower())))[:2000]
            if self.watch_browser and self.browser_dock:
                env["JEVH_BROWSER_DOCK"] = str(self.session_dir / "browsers")
        if self.project and "project" in persona.tools:
            ctx["project"] = self.project
            env["JEVH_PROJECT"] = self.project
            env["JEVH_RULE_BLOCK"] = str(self.cfg.t("rule_block"))
        tier = self._pick_tier(task, persona)
        self.on_event("tier", {"task": task, "decision": tier})
        llm = self._llm(persona, tier.tier)
        if getattr(llm, "native_tools", False):   # acting is a `python` tool call, not a fenced block
            system = build_system(persona, picked, requestable, collab.suggested, tools.tools,
                                  [catalog[k] for k in tools.mcp if k in catalog], native=True, tool_notes=notes)
        handles = Handles(self._start_request, task, goal, requestable)
        handlers = {
            "llm_query": lambda prompt, data=None: llm.chat([{"role": "user", "content": (
                prompt if data is None else f"{prompt}\n\n---\n{data}")}]).text,
            "jev": lambda state, questions: self.jev._ask("worker_jev", state, questions).as_dict(),
            # browser.look(): the worker's own model looks at a screenshot (canvas games, charts, boards)
            "look": lambda path, question: see_image(llm, path, question),
            "request": lambda to, text, inputs=None: handles.wait(handles.start(to, text, inputs)),
            "request_async": lambda to, text, inputs=None: handles.start(to, text, inputs),
            "wait": lambda h, timeout=None: handles.wait(h, timeout),
            "mcp": lambda key, arguments: self._mcp_call(persona, key, arguments),
            "recall": lambda query, k=5: self._recall(query, k),
            "remember": lambda fact: self.learner.remember(str(fact), f"{self.session}/{persona.name}"),
            "ask_user": lambda question, options=None, secret=False: self._ask_user(task, persona, question,
                                                                                    options, secret),
            "report_failure": lambda what, detail="": self.failure_watch.report(task, what, detail),
        }
        workdir = Path(resume["workdir"]) if resume and resume.get("workdir") else \
            self.session_dir / f"u{turn_no}t{task.id}-{persona.name}"   # task ids restart every turn
        task.checkpoint = workdir / "checkpoint.json"
        worker = Worker(persona, llm, workdir, self.artifacts,
                        handlers, {t: f"jevh.tools.{t}" for t in persona.tools},
                        self.cfg.lim("repl_timeout"), self.cfg.lim("output_cap"),
                        lambda kind, data: self._on_worker(kind, data, task), env,
                        self._cancel, self._verify_fn(task, persona) if persona.code and self.verifier.enabled else None,
                        int(self.verifier.settings.get("max_bounces", 2)),
                        self._progress_fn(task), self.cfg.t("stall"), self.cfg.lim("progress_every"),
                        self.cfg.lim("compact_tokens"), self.cfg.lim("keep_steps"), task.checkpoint)
        worker.keep_reasoning = self.cfg.lim("keep_reasoning")
        worker.retry_delay = float(interject.settings(self.cfg.raw)["retry_delay"])
        worker.step_value, worker.keep_at = self._compact_fn(task), self.cfg.t("compact_keep")
        if self.cfg.lim("thinking_watch"):
            worker.thinking_watch = self._watch_fn(task)
            worker.watch_every, worker.watch_after = self.cfg.lim("watch_every"), self.cfg.lim("watch_after")
        if tier.tier == "fast" and self.tier_mode != "fast":   # quality first: a bounced fast answer escalates
            worker.escalate = lambda: self._llm(persona, "strong")
        task.started = time.time()
        self.on_event("start", {"task": task, "model": getattr(llm, "name", "?")})
        self._workers.add(worker)
        if task.parent is None:   # the user's messages while the turn runs go to the lead worker (interject.py)
            with self._ilock:
                self._lead, self._lead_task, self._phase = worker, task, "working"
                early, self._early = self._early, []
            for text, replaces in early:   # sent while the harness was still choosing the skills and tools
                worker.steer(text, abort=False, replaces=replaces)
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
            left = worker.take_unapplied()   # messages that came too late for it to use: the next request
            if left and not self._cancel.is_set():
                with self._ilock:
                    self._queue[0:0] = left
                self.on_event("queued", {"texts": left, "late": True, "queue": self.queued()})
            if task.parent is None:
                with self._ilock:
                    if self._lead is worker:
                        self._phase = "finishing"
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
        if kind == "steer":   # the task is now what the user last asked for: the gate, the progress check and
            task.said.append((data["text"], bool(data.get("replaces"))))   # compaction all judge Task.current
            self._said.append(data["text"])
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
        if "nango" in own and (accounts := nango.summary()):   # Jev judges "my email" knowing Gmail is connected
            own["nango"] += f" ({accounts.rstrip('.')})"
        mcp = {}
        if persona.mcp and self.mcp:
            mcp = {k: t.description for k, t in self.mcp.catalog().items() if persona.allows_mcp(k)}
        if not own and not mcp:
            return ToolDecision([])
        fallback = ToolDecision(list(own))  # Jev down: all guides (capability beats a lean prompt)
        return self._safe("tools", lambda: self.jev.pick_tools(task.jev_text, own, mcp), fallback)

    def _progress_fn(self, task: Task):
        def circles(log: str) -> float:
            r = self.jev._ask("progress", {"request": task.current[:1500], "work_log": log[-4000:]},
                              {"circles": noul(CIRCLES_Q)})
            return r["circles"].noul
        return circles

    def _compact_fn(self, task: Task):
        """-> (old step texts, where the work is now) -> P(the rest of the task needs each step).
        One call per step: batching the steps into one call scored them lower (10/18 needed kept at
        0.8 vs 16/18) and was no faster (~0.2 s a step either way)."""
        def value(steps: list[str], now: str) -> list[float]:
            state = {"task": task.current[:1500], "now": now[-2000:]}
            return [self.jev._ask("compact", {**state, "step": s[:4500]}, {"info": noul(COMPACT_Q)})["info"].noul
                    for s in steps]
        return value

    def _watch_fn(self, task: Task):
        """-> (reasoning so far) -> why to pause it ("testable" / "searching") or None. Jev on the end
        of a long think: hand-work code does exactly (TESTABLE_Q), or theory after theory (STUCK_Q
        well above CHECKING_Q)."""
        def watch(thinking: str) -> str | None:
            r = self.jev._ask("thinking_watch", {"task": task.current[:1500], "thinking": thinking[-2500:]},
                              {"testable": noul(TESTABLE_Q), "stuck": noul(STUCK_Q), "checking": noul(CHECKING_Q)})
            p, search = r["testable"].noul, r["stuck"].noul - r["checking"].noul
            self.on_event("thinking_watch", {"task": task, "p": round(p, 3), "search": round(search, 3),
                                             "chars": len(thinking)})
            if p >= self.cfg.t("testable"):
                return "testable"
            return "searching" if search > self.cfg.t("searching") else None
        return watch

    def _verify_fn(self, task: Task, persona: Persona):
        project = self.project
        # the worker's REPL runs in its workdir, where project.py keeps the originals of files it touches
        backups = task.checkpoint.parent / ".project_backups" if task.checkpoint else None

        def review(prompt: str, tier: str) -> str:   # a risky change's full review (tier: the gate's pick)
            self.on_event("review", {"task": task, "persona": persona.name, "tier": tier})
            return self._llm(persona, tier).chat([{"role": "user", "content": prompt}], cancel=self._cancel).text

        def check(answer: str, artifacts: list[str], transcript: list[dict], changed: list[str],
                  task_idx: int | None = None):
            # the user changed the request while it ran and named a file: that file, not an earlier one, is due
            last = task.said[-1][0] if task.said else ""
            return self.verifier.check(task.current, answer, artifacts, work_log(transcript, answer=answer),
                                       project, changed, evidence=seen_text(transcript, task_idx),
                                       backups=backups, review=review,
                                       files_from=last if DELIVERABLE.search(last) else None)
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
        elif self.cfg.models.get("strong"):
            # another machine's models (config.local.yaml): every persona's own model is replaced, its
            # thinking and sampling settings kept
            return {**{k: v for k, v in persona.model.items() if k != "fallback"}, **self.cfg.models["strong"]}
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
