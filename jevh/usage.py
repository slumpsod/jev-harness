"""Usage tracking: every model call, Jev call and LM Studio swap, stored per session / turn /
task / persona in workspace/usage.db, summed for the CLI footer, the dashboard's Usage tab and
(next) long-task budgets.

kinds
  llm        a worker's model call (thinking tokens counted separately, from LM Studio's
             completion_tokens_details.reasoning_tokens, else the streamed reasoning chunks)
  learn      a model call made by after-turn learning
  jev        a Jev call made by the harness (routing, skills, memory, tools, tier, verify, ...)
  jev_tools  Jev calls made inside a worker's tools (browser, web, project.find), reported by the REPL
  swap       an LM Studio model swap (seconds only)

Cost is only computed for models priced in config `prices:` ($ per million tokens), e.g.
    prices: {claude-opus-5: {input: 5.0, output: 25.0}}
Local models cost nothing here.
"""
from __future__ import annotations

import inspect
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class Usage:
    def __init__(self, path: Path, prices: dict[str, dict[str, float]] | None = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.prices = prices or {}
        self._lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS usage(
            ts REAL, session TEXT, turn INTEGER, task INTEGER, persona TEXT, kind TEXT, model TEXT, tier TEXT,
            calls INTEGER, input INTEGER, output INTEGER, thinking INTEGER, seconds REAL, cost REAL)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS usage_turn ON usage(session, turn)")
        self.db.commit()

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        p = self.prices.get(model) or self.prices.get(model.split("@")[0])
        if not p:
            return 0.0
        return (input_tokens * float(p.get("input", 0)) + output_tokens * float(p.get("output", 0))) / 1e6

    def record(self, *, session: str, turn: int, kind: str, model: str = "", persona: str = "", task: int | None = None,
               tier: str = "", calls: int = 1, input_tokens: int = 0, output_tokens: int = 0, thinking: int = 0,
               seconds: float = 0.0) -> dict[str, Any]:
        row = {"ts": time.time(), "session": session, "turn": turn, "task": task, "persona": persona, "kind": kind,
               "model": model, "tier": tier, "calls": calls, "input": int(input_tokens), "output": int(output_tokens),
               "thinking": int(thinking), "seconds": float(seconds),
               "cost": self.cost(model, input_tokens, output_tokens)}
        with self._lock:
            self.db.execute("INSERT INTO usage VALUES (:ts,:session,:turn,:task,:persona,:kind,:model,:tier,:calls,"
                            ":input,:output,:thinking,:seconds,:cost)", row)
            self.db.commit()
        return row

    def summary(self, session: str | None = None, turn: int | None = None, since: float | None = None) -> dict[str, Any]:
        """Totals plus breakdowns by kind, model and persona for a turn, a session or a time window."""
        where, args = [], []
        if session is not None:
            where.append("session=?")
            args.append(session)
        if turn is not None:
            where.append("turn=?")
            args.append(turn)
        if since is not None:
            where.append("ts>=?")
            args.append(since)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        cols = "SUM(calls), SUM(input), SUM(output), SUM(thinking), SUM(seconds), SUM(cost)"
        with self._lock:
            total = self.db.execute(f"SELECT {cols} FROM usage {clause}", args).fetchone()
            groups = {g: self.db.execute(f"SELECT {g}, {cols} FROM usage {clause} GROUP BY {g} "
                                         f"ORDER BY SUM(seconds) DESC", args).fetchall()
                      for g in ("kind", "model", "persona")}
            turns = self.db.execute(f"SELECT COUNT(DISTINCT session || ':' || turn) FROM usage {clause}", args).fetchone()[0]

        def pack(r):
            return {"calls": r[0] or 0, "input": r[1] or 0, "output": r[2] or 0, "thinking": r[3] or 0,
                    "seconds": round(r[4] or 0.0, 2), "cost": round(r[5] or 0.0, 4)}
        out = {"total": pack(total), "turns": turns}
        for g, rows in groups.items():
            out[f"by_{g}"] = {(r[0] or "-"): pack(r[1:]) for r in rows}
        return out

    def turns(self, session: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        where, args = ("WHERE session=?", [session]) if session else ("", [])
        with self._lock:
            rows = self.db.execute(
                f"SELECT session, turn, MIN(ts), SUM(CASE WHEN kind IN ('llm','learn') THEN input ELSE 0 END), "
                f"SUM(CASE WHEN kind IN ('llm','learn') THEN output ELSE 0 END), SUM(thinking), "
                f"SUM(CASE WHEN kind IN ('llm','learn') THEN seconds ELSE 0 END), "
                f"SUM(CASE WHEN kind LIKE 'jev%' THEN seconds ELSE 0 END), "
                f"SUM(CASE WHEN kind LIKE 'jev%' THEN calls ELSE 0 END), SUM(cost) "
                f"FROM usage {where} GROUP BY session, turn ORDER BY MIN(ts) DESC LIMIT ?", [*args, limit]).fetchall()
        return [{"session": r[0], "turn": r[1], "ts": r[2], "input": r[3], "output": r[4], "thinking": r[5],
                 "model_seconds": round(r[6], 1), "jev_seconds": round(r[7], 1), "jev_calls": r[8],
                 "cost": round(r[9], 4)} for r in rows]

    def close(self) -> None:
        with self._lock:
            self.db.close()


def brief(summary: dict[str, Any]) -> str:
    """One line for a turn: '12.3k in / 4.5k out (3.9k thinking) · model 85s · Jev 3.2s, 14 calls'."""
    k = summary.get("by_kind", {})
    llm = [k.get("llm"), k.get("learn")]
    tin = sum(x["input"] for x in llm if x)
    tout = sum(x["output"] for x in llm if x)
    think = sum(x["thinking"] for x in llm if x)
    msecs = sum(x["seconds"] for x in llm if x)
    jev = [k.get("jev"), k.get("jev_tools")]
    jsecs, jcalls = sum(x["seconds"] for x in jev if x), sum(x["calls"] for x in jev if x)
    parts = [f"{_k(tin)} in / {_k(tout)} out" + (f" ({_k(think)} thinking)" if think else "")]
    if msecs:
        parts.append(f"model {msecs:.0f}s")
    if jcalls:
        parts.append(f"Jev {jsecs:.1f}s, {jcalls} calls")
    if k.get("swap"):
        parts.append(f"swaps {k['swap']['seconds']:.0f}s")
    if summary["total"]["cost"]:
        parts.append(f"${summary['total']['cost']:.4f}")
    return " · ".join(parts)


def _k(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


class Metered:
    """Wraps an LLM so every call is recorded. `where()` says which session/turn/task/persona the
    calling thread is working for (hand-offs run on their own threads)."""

    def __init__(self, inner, record, where, tier: str, model: str):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_record", record)
        object.__setattr__(self, "_where", where)
        object.__setattr__(self, "_tier", tier)
        object.__setattr__(self, "_model", model)
        try:
            takes = "cancel" in inspect.signature(inner.chat).parameters
        except (TypeError, ValueError):
            takes = False
        object.__setattr__(self, "_takes_cancel", takes)

    def chat(self, messages, cancel=None):
        started = time.monotonic()
        res = None
        try:
            res = self._inner.chat(messages, cancel=cancel) if self._takes_cancel else self._inner.chat(messages)
            return res
        finally:
            w = self._where()
            self._record(kind=w.get("kind", "llm"), model=getattr(res, "model", "") or self._model,
                         persona=w.get("persona", ""), task=w.get("task"), tier=self._tier,
                         input_tokens=getattr(res, "input_tokens", 0), output_tokens=getattr(res, "output_tokens", 0),
                         thinking=getattr(res, "thinking_tokens", 0), seconds=time.monotonic() - started)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)
