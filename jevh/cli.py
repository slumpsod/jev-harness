"""Terminal chat for the harness.

    python -m jevh.cli [--as persona] [--verbose] [--config path]

Commands: /as <persona> (pin), /auto (let Jev route), /personas, /skills, /trace,
/verbose, /jev (health), /reset, /quit
"""
from __future__ import annotations

import argparse
import json
import sys
import threading

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape as esc
from rich.panel import Panel

from .config import load_config
from .orchestrator import Harness

console = Console()


def handoff_overlap(tasks) -> tuple[float, float]:
    """(summed hand-off seconds, wall seconds they covered). Nested hand-offs are counted
    inside their parent, so only depth-1 tasks are summed."""
    spans = sorted((t.started, t.finished) for t in tasks if t.depth == 1 and t.finished)
    work = sum(b - a for a, b in spans)
    wall, end = 0.0, float("-inf")
    for a, b in spans:
        if b > end:
            wall += b - max(a, end)
            end = b
    return work, wall


def _indent(task) -> str:
    return "  " * (task.depth + 1)


class Printer:
    """Events arrive from several worker threads at once (parallel hand-offs): printing is
    serialized, and concurrent generation shows as one combined progress line."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._lock = threading.Lock()
        self._live: dict[str, tuple[int, int]] = {}

    def __call__(self, kind: str, data: dict) -> None:
        with self._lock:
            self._print(kind, data)

    def _print(self, kind: str, data: dict) -> None:
        if kind == "route":
            d = data["decision"]
            top = sorted(d.probabilities.items(), key=lambda kv: -kv[1])[:3]
            alts = "  ".join(f"{k} {v:.2f}" for k, v in top)
            cont = f"  continues={d.continues:.2f}" if d.continues is not None else ""
            console.print(f"[bold cyan]jev[/] → [bold]{d.persona}[/] ({esc(d.reason)}; conf {d.confidence:.2f})"
                          f"[dim]  {alts}{cont}[/]")
        elif kind == "skills":
            d, t = data["decision"], data["task"]
            if d.skills or self.verbose:
                gate = f" (none {d.none:.2f})" if d.none is not None else ""
                console.print(f"[dim]{_indent(t)}skills: {', '.join(d.skills) or 'none'}{gate}[/]")
        elif kind == "memory":
            d, t = data["decision"], data["task"]
            console.print(f"[dim]{_indent(t)}memory: {len(d.kept)}/{data['candidates']} snippets passed[/]")
        elif kind == "collab":
            d, t = data["decision"], data["task"]
            if d.suggested:
                console.print(f"{_indent(t)}[magenta]collab suggested:[/] {', '.join(d.suggested)}"
                              f"[dim] (complexity {d.complexity:.2f})[/]")
        elif kind == "start":
            t = data["task"]
            console.print(f"[dim]{_indent(t)}{t.to} working on {esc(str(data['model']))}[/]")
        elif kind == "request":
            t = data["task"]
            if data["approved"]:
                console.print(f"{_indent(t)}[green]↳ {t.requester} → {t.to}:[/] {esc(' '.join(t.text.split())[:120])}")
            else:
                console.print(f"{_indent(t)}[red]✗ {t.requester} → {t.to} refused:[/] {esc(data['reason'])}")
        elif kind == "code":
            first = data["code"].strip().splitlines()[0][:100] if data["code"].strip() else ""
            tag = esc("[" + data["persona"] + " step " + str(data["step"]) + "]")
            console.print(f"[dim]    {tag} {esc(first)}[/]")
            if self.verbose:
                console.print(Panel(data["code"], title="code", border_style="dim"))
        elif kind == "output":
            if self.verbose or data["error"]:
                style = "red" if data["error"] else "dim"
                console.print(Panel(data["output"][-1500:] or "(no output)", title="output",
                                    border_style=style))
        elif kind == "done":
            t, r = data["task"], data["result"]
            self._live.pop(t.to, None)
            if t.depth > 0:
                console.print(f"[dim]{_indent(t)}{t.to} finished ({r.stopped}, {r.steps} steps, "
                              f"{t.finished - t.started:.0f}s)[/]")
        elif kind == "progress":
            self._live[data["persona"]] = (data["chunks"], data.get("thinking", 0))
            line = " | ".join(f"{p} {n} tok ({t} think)" for p, (n, t) in self._live.items())
            console.print(f"[dim]    generating: {esc(line)}[/]", end="\r")
        elif kind == "unwaited":
            t, r = data["task"], data["result"]
            console.print(f"[yellow]{_indent(t)}{t.to} never waited on a hand-off ({r['status']}); "
                          f"kept its {len(r['artifacts'])} artifact(s)[/]")
        elif kind == "jev_error":
            console.print(f"[yellow]jev unavailable during {data['step']}; degrading ({esc(data['error'][:120])})[/]")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="jevh")
    ap.add_argument("--config")
    ap.add_argument("--as", dest="pin", help="pin a persona (skip routing)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("-c", "--command", help="run one message and exit")
    ap.add_argument("--model", help="put every persona on one model for testing, e.g. lmstudio:qwen3-4b")
    a = ap.parse_args(argv)

    printer = Printer(a.verbose)
    h = Harness(load_config(a.config), on_event=printer, model_override=a.model)
    pin = a.pin
    last = None

    def run(msg: str) -> None:
        nonlocal last
        last = h.turn(msg, force=pin)
        console.print(Panel(Markdown(last.answer or "(empty)"), title=f"{last.persona}",
                            subtitle=f"{last.seconds:.1f}s · jev calls {len(last.trace)}",
                            border_style="cyan"))
        for art in last.artifacts:
            console.print(f"  [green]artifact:[/] {esc(art)}")
        work, wall = handoff_overlap(last.tasks)
        if work:
            console.print(f"  [dim]hand-offs: {work:.0f}s of work in {wall:.0f}s wall "
                          f"(parallel saved {work - wall:.0f}s)[/]")

    if a.command:
        run(a.command)
        return

    console.print(f"[bold]jev harness[/] · session {h.session} · personas: {', '.join(h.personas)}"
                  f" · skills: {len(h.roster)}  [dim](/help for commands)[/]")
    while True:
        try:
            msg = input("\nyou › ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        if msg.startswith("/"):
            cmd, _, arg = msg[1:].partition(" ")
            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "as":
                if arg not in h.personas:
                    console.print(f"unknown persona; choose from {', '.join(h.personas)}")
                else:
                    pin = arg
                    console.print(f"pinned to {pin} (/auto to let Jev route)")
            elif cmd == "auto":
                pin = None
                console.print("Jev routes again")
            elif cmd == "personas":
                for n, p in h.personas.items():
                    console.print(f"[bold]{n}[/] ({p.model.get('provider')}:{p.model.get('model')}) — {esc(p.description)}")
            elif cmd == "skills":
                for s in h.roster.values():
                    console.print(f"[bold]{s.name}[/] — {esc(s.description)}")
            elif cmd == "trace":
                console.print_json(json.dumps(last.trace if last else [], default=str))
            elif cmd == "verbose":
                printer.verbose = not printer.verbose
                console.print(f"verbose {'on' if printer.verbose else 'off'}")
            elif cmd == "jev":
                try:
                    console.print_json(json.dumps(h.jev.client.health()))
                except Exception as e:  # noqa: BLE001
                    console.print(f"[red]{esc(str(e))}[/]")
            elif cmd == "reset":
                h = Harness(h.cfg, on_event=printer, model_override=a.model)
                console.print(f"new session {h.session}")
            else:
                console.print(__doc__)
            continue
        try:
            run(msg)
        except KeyboardInterrupt:
            console.print("[yellow]interrupted[/]")


if __name__ == "__main__":
    sys.exit(main())
