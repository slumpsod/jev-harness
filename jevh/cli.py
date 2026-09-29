"""Terminal chat for the harness.

    python -m jevh.cli [--as persona] [--project path] [--watch] [--verbose] [--config path]

Ctrl-C during a turn stops it (model, running code, hand-offs); a worker's question is
asked right here in the terminal.

Commands: /as <persona> (pin), /auto (let Jev route), /tier auto|fast|strong,
/project <path> (open a project folder; "/project off" closes it), /watch [on|off] (see the
browser tool work in a visible window), /facts, /forget <id>, /usage,
/resume [n|list] (continue an interrupted task: stopped, step limit, stalled, crashed),
/personas, /skills, /trace, /verbose, /jev (health), /reset, /quit
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape as esc
from rich.panel import Panel

from .config import load_config
from .orchestrator import Harness
from .usage import brief

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
            if d.replies is not None:
                cont += f" replies={d.replies:.2f}" + (" -> follow-up: last turns go along" if d.follows else "")
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
        elif kind == "tools":
            d, t = data["decision"], data["task"]
            if d.tools or d.mcp or self.verbose:
                shown = ", ".join(d.tools + d.mcp) or "none"
                console.print(f"[dim]{_indent(t)}tools: {esc(shown)}[/]")
        elif kind == "compact":
            console.print(f"[dim]    {esc(data['persona'])}: context compacted at step {data['step']} "
                          f"(~{data['tokens_before']:,} -> ~{data['tokens_after']:,} tokens"
                          f"; {data.get('full', 0)} steps kept in full, {data.get('notes', 0)} as notes"
                          f"{', %d left out' % data['dropped'] if data.get('dropped') else ''})[/]")
        elif kind == "escalate":
            console.print(f"[yellow]    {esc(data['persona'])}: the fast model's answer was sent back; "
                          f"continuing on {esc(str(data['model']))}[/]")
        elif kind == "progress_check":
            if data["stuck"]:
                console.print(f"[yellow]    {esc(data['persona'])}: progress check at step {data['step']}: "
                              f"going in circles ({data['circles']:.2f}); "
                              f"{'stopping' if data['strike'] >= 2 else 'told to change approach'}[/]")
        elif kind == "verify":
            if data["ok"]:
                console.print(f"[dim]    {esc(data['persona'])}: final() verified "
                              f"({', '.join(k for k in data['checks']) or 'answer'})[/]")
            else:
                verb = "sent back" if data["bounced"] else "accepted with warnings"
                console.print(f"[yellow]    {esc(data['persona'])}: final() {verb}: "
                              + esc(" | ".join(p.splitlines()[0][:140] for p in data["problems"])) + "[/]")
        elif kind == "learned":
            out = data["learned"]
            for f in out.facts:
                console.print(f"[dim]    remembered: {esc(f)}[/]")
            for f in out.retired:
                console.print(f"[dim]    replaced: {esc(f)}[/]")
            if out.skill:
                verb = "learned a new skill (draft; /skills to review)" if out.skill_action == "created" \
                    else "added lessons to"
                console.print(f"[dim]    {verb}: {esc(out.skill)}[/]")
        elif kind == "learn_error":
            console.print(f"[yellow]    learning skipped: {esc(data['error'][:160])}[/]")
        elif kind == "collab":
            d, t = data["decision"], data["task"]
            if d.suggested:
                console.print(f"{_indent(t)}[magenta]collab suggested:[/] {', '.join(d.suggested)}"
                              f"[dim] (complexity {d.complexity:.2f})[/]")
        elif kind == "swap":
            gone = ", ".join(data["unloaded"]) or "nothing"
            console.print(f"[magenta]    LM Studio: unloaded {esc(gone)} -> loaded {esc(data['loaded'])} "
                          f"({data['seconds']:.0f}s)[/]")
        elif kind == "tier":
            d, t = data["decision"], data["task"]
            color = "green" if d.tier == "fast" else "blue"
            console.print(f"[dim]{_indent(t)}tier: [{color}]{d.tier}[/{color}] ({esc(d.reason)})[/]")
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
    ap.add_argument("--tier", choices=["auto", "fast", "strong"], default="auto",
                    help="model tier: Jev decides per task (auto), or force one")
    ap.add_argument("--model", help="put every persona on one model for testing, e.g. lmstudio:qwen3-4b")
    ap.add_argument("--project", help="open a project folder for personas with the project tool")
    ap.add_argument("--watch", action="store_true", help="show the browser tool's window with Jev's picks outlined")
    a = ap.parse_args(argv)

    printer = Printer(a.verbose)
    h = Harness(load_config(a.config), on_event=printer, model_override=a.model)
    h.tier_mode = a.tier
    if a.project:
        h.set_project(a.project)
    if a.watch:
        h.watch_browser = True
    pin = a.pin
    last = None

    def run(msg: str, resume_path: str | None = None, resuming: bool = False) -> None:
        """The turn runs on a thread so this one can answer ask_user() questions, and Ctrl-C
        stops the turn instead of abandoning it."""
        nonlocal last
        box: dict = {}

        def go() -> None:
            try:
                box["r"] = h.resume(resume_path) if resuming else h.turn(msg, force=pin)
            except BaseException as e:  # noqa: BLE001 - re-raised below in the main thread
                box["e"] = e

        worker = threading.Thread(target=go, daemon=True)
        worker.start()
        asked: set[str] = set()
        while worker.is_alive():
            try:
                worker.join(0.2)
                for q in h.pending_questions():
                    if q["id"] in asked:
                        continue
                    asked.add(q["id"])
                    opts = f"  [{' / '.join(q['options'])}]" if q["options"] else ""
                    console.print(f"\n[bold magenta]{esc(q['persona'])} asks:[/] {esc(q['question'])}{esc(opts)}")
                    h.answer(q["id"], input("answer › ").strip() or "(no answer)")
            except KeyboardInterrupt:
                if h.cancel():
                    console.print("[yellow]stopping... (the model, running code and hand-offs)[/]")
        if "e" in box:
            raise box["e"]
        last = box["r"]
        console.print(Panel(Markdown(last.answer or "(empty)"),
                            title=f"{last.persona}" + (" (stopped)" if last.cancelled else ""),
                            subtitle=f"{last.seconds:.1f}s · {brief(last.usage)}" if last.usage
                            else f"{last.seconds:.1f}s · jev calls {len(last.trace)}",
                            border_style="cyan"))
        for art in last.artifacts:
            console.print(f"  [green]artifact:[/] {esc(art)}")
        if last.checkpoint:
            console.print("  [yellow]This task can be resumed: /resume[/]")
        work, wall = handoff_overlap(last.tasks)
        if work:
            console.print(f"  [dim]hand-offs: {work:.0f}s of work in {wall:.0f}s wall "
                          f"(parallel saved {work - wall:.0f}s)[/]")

    if a.command:
        run(a.command)
        h.wait_learning()
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
            elif cmd == "tier":
                if arg in ("auto", "fast", "strong"):
                    h.tier_mode = arg
                console.print(f"tier: {h.tier_mode} (fast model: {h.cfg.models.get('fast', {}).get('model', 'none')})")
            elif cmd == "project":
                try:
                    h.set_project(None if arg.strip() in ("", "off", "close") else arg.strip())
                    console.print(f"project: {h.project or 'none'}")
                except NotADirectoryError as e:
                    console.print(f"[red]not a folder: {esc(str(e))}[/]")
            elif cmd == "watch":
                h.watch_browser = not h.watch_browser if arg.strip() not in ("on", "off") else arg.strip() == "on"
                console.print(f"watch browser: {'on (a visible window from the next task)' if h.watch_browser else 'off'}")
            elif cmd == "resume":
                found = h.resumable(10)
                if not found:
                    console.print("nothing to resume")
                elif arg.strip().isdigit() or not arg.strip():
                    pick = found[int(arg) - 1 if arg.strip().isdigit() else 0]
                    console.print(f"resuming {pick['persona']} (step {pick['step']}, {pick['stopped']}): "
                                  f"{esc(pick['task'][:100])}")
                    run("", pick["path"], True)
                if found and arg.strip() == "list":
                    for i, x in enumerate(found, 1):
                        console.print(f"  {i}. {x['persona']} step {x['step']} ({x['stopped']}): {esc(x['task'][:90])}")
            elif cmd == "usage":
                day = time.time() - 86400
                for label, s in (("this session", h.usage.summary(h.session)),
                                 ("last 24h", h.usage.summary(since=day)),
                                 ("last 7 days", h.usage.summary(since=time.time() - 7 * 86400))):
                    console.print(f"[bold]{label}[/] ({s['turns']} turns): {esc(brief(s))}")
                s = h.usage.summary(since=time.time() - 7 * 86400)
                for m, v in s["by_model"].items():
                    console.print(f"  [dim]{esc(m)}: {v['calls']} calls, {v['input']:,} in / {v['output']:,} out "
                                  f"({v['thinking']:,} thinking), {v['seconds']:.0f}s[/]")
            elif cmd == "facts":
                for f in h.memory.facts(None, 50):
                    console.print(f"[bold]{f.id}[/] {esc(f.chunk())}")
            elif cmd == "forget":
                if arg.strip().isdigit():
                    h.memory.retire_fact(int(arg))
                    console.print(f"forgot fact {arg.strip()}")
            elif cmd == "auto":
                pin = None
                console.print("Jev routes again")
            elif cmd == "personas":
                for n, p in h.personas.items():
                    console.print(f"[bold]{n}[/] ({p.model.get('provider')}:{p.model.get('model')}) — {esc(p.description)}")
            elif cmd == "skills":
                for s in h.roster.values():
                    tag = " [yellow](draft: learned, not reviewed)[/]" if s.draft else ""
                    console.print(f"[bold]{s.name}[/]{tag} — {esc(s.description)}")
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
                mode, project, watch = h.tier_mode, h.project, h.watch_browser
                h = Harness(h.cfg, on_event=printer, model_override=a.model)
                h.tier_mode, h.project, h.watch_browser = mode, project, watch
                console.print(f"new session {h.session}")
            else:
                console.print(__doc__)
            continue
        run(msg)


if __name__ == "__main__":
    sys.exit(main())
