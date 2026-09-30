"""Task board for one user turn: who asked whom for what, plus the hard limits
(depth, task count, cycles) that hold no matter what Jev says."""
from __future__ import annotations

import dataclasses
import itertools
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    id: int
    parent: "Task | None"
    requester: str               # "user" or a persona name
    to: str
    text: str
    inputs: list[str] = field(default_factory=list)
    status: str = "open"         # open | done | rejected | failed
    started: float = 0.0         # time.time() when the worker started / finished
    finished: float = 0.0
    answer: str = ""
    artifacts: list[str] = field(default_factory=list)
    history: list[tuple] = field(default_factory=list)  # (user, assistant[, work notes]) turns this follows up on
    checkpoint: Any = None       # path of this task's worker checkpoint (resume)

    @property
    def jev_text(self) -> str:
        """What Jev judges for skills/tools/tier/memory: the message, plus the exchange it follows
        up on ("4 days, full gym" alone says nothing about workouts)."""
        if not self.history:
            return self.text
        prior = "\n".join(f"User: {u[:400]}\nAssistant: {a[:400]}" for u, a, *_ in self.history)
        return f"Earlier in this conversation:\n{prior}\n\nLatest message: {self.text}"

    @property
    def depth(self) -> int:
        return 0 if self.parent is None else self.parent.depth + 1

    def chain(self) -> list[str]:
        """Personas currently working up the call stack, innermost last."""
        return ([] if self.parent is None else self.parent.chain()) + [self.to]


def task_json(t: Task) -> dict[str, Any]:
    return {"id": t.id, "parent": t.parent.id if t.parent else None, "requester": t.requester, "to": t.to,
            "text": t.text[:600], "depth": t.depth, "status": t.status, "started": t.started,
            "finished": t.finished, "artifacts": list(t.artifacts), "answer": t.answer[:1500]}


def jsonable(x: Any) -> Any:
    """Harness events carry dataclasses (Task, decisions, WorkerResult); make them JSON-safe
    (for the dashboard and for failure hooks)."""
    if isinstance(x, Task):
        return task_json(x)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        d = {f.name: getattr(x, f.name) for f in dataclasses.fields(x) if f.name != "transcript"}
        return {k: jsonable(v) for k, v in d.items()}
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


class LimitError(RuntimeError):
    pass


class Board:
    """Thread-safe: parallel hand-offs open tasks from several worker threads."""

    def __init__(self, max_depth: int = 3, max_tasks: int = 8):
        self.max_depth = max_depth
        self.max_tasks = max_tasks
        self.tasks: list[Task] = []
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def check(self, parent: Task, to: str) -> None:
        if to in parent.chain():
            raise LimitError(f"cycle: {' -> '.join(parent.chain() + [to])}")
        if parent.depth + 1 > self.max_depth:
            raise LimitError(f"max delegation depth {self.max_depth} reached")
        if len(self.tasks) >= self.max_tasks:
            raise LimitError(f"max {self.max_tasks} tasks per turn reached")

    def open(self, parent: Task | None, requester: str, to: str, text: str,
             inputs: list[str] | None = None) -> Task:
        with self._lock:
            task = Task(next(self._ids), parent, requester, to, text, list(inputs or []))
            self.tasks.append(task)
            return task

    def open_checked(self, parent: Task, requester: str, to: str, text: str,
                     inputs: list[str] | None = None) -> Task:
        """check() + open() as one step, so parallel requests can't both slip under a limit."""
        with self._lock:
            self.check(parent, to)
            task = Task(next(self._ids), parent, requester, to, text, list(inputs or []))
            self.tasks.append(task)
            return task

    def reject(self, parent: Task | None, requester: str, to: str, text: str, reason: str) -> Task:
        task = self.open(parent, requester, to, text)
        task.status, task.answer = "rejected", reason
        return task
