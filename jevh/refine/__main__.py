"""python -m jevh.refine <command>

  groups                  open failures grouped by kind, persona and cause (what would be worked on)
  models                  the local models tests run on, and the one that writes tests and fixes
  nightly [--until 07:00 | --hours 8]   carry on unfinished attempts, then start new ones
  start <failure id> [--hours N]        one attempt on that failure, now
  resume <attempt id> [--hours N]
  status                  every attempt and its stage
  report <attempt id>     the approval report
  approve <attempt id>    merge the change; its failures are marked fixed
  reject <attempt id> "why"   drop it; the reason goes to the fixer next time
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time

from ..config import load_config
from .loop import Refiner


def deadline(until: str | None, hours: float | None) -> float | None:
    if hours:
        return time.time() + hours * 3600
    if until:
        h, _, m = until.partition(":")
        now = dt.datetime.now()
        t = now.replace(hour=int(h), minute=int(m or 0), second=0, microsecond=0)
        if t <= now:
            t += dt.timedelta(days=1)
        return t.timestamp()
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jevh.refine", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("groups")
    sub.add_parser("models")
    sub.add_parser("status")
    n = sub.add_parser("nightly")
    n.add_argument("--until", default="07:00")
    n.add_argument("--hours", type=float)
    for name in ("start", "resume"):
        p = sub.add_parser(name)
        p.add_argument("id")
        p.add_argument("--hours", type=float)
    sub.add_parser("report").add_argument("id")
    sub.add_parser("approve").add_argument("id")
    r = sub.add_parser("reject")
    r.add_argument("id")
    r.add_argument("reason")
    a = ap.parse_args(argv)
    ref = Refiner(load_config(a.config))

    if a.cmd == "groups":
        for g in ref.groups():
            f = g["latest"]
            print(f"{g['count']:>4}x {g['kind']:<18} {g['persona'] or '-':<20} {'(tried) ' if g['tried'] else ''}"
                  f"latest #{f['id']}: {g['cause'][:90]}")
    elif a.cmd == "models":
        print("tests run on: " + (", ".join(ref.models()) or "(none)"))
        print("tests and fixes written by: " + ref.model())
    elif a.cmd == "status":
        for s in ref.attempts():
            extra = s.get("verdict") or s.get("reason") or (s.get("triage") or {}).get("decision") or ""
            print(f"{s['id']:<28} {s['stage']:<11} {s['failure']['kind']:<18} {s.get('objective', 'pass'):<5} "
                  f"tries {len(s['tries'])} {extra[:80]}")
    elif a.cmd == "nightly":
        until = deadline(None if a.hours else a.until, a.hours)
        ref.log(f"nightly run until {time.strftime('%Y-%m-%d %H:%M', time.localtime(until))}")
        for s in ref.nightly(until):
            print(f"{s['id']}: {s['stage']}" + (f" - {s.get('verdict') or s.get('reason')}" if s['stage'] != 'new' else ""))
        waiting = ref.proposals()
        if waiting:
            print(f"\n{len(waiting)} change(s) waiting for approval: python -m jevh.refine report <id>")
    elif a.cmd in ("start", "resume"):
        aid = ref.start(int(a.id)) if a.cmd == "start" else a.id
        s = ref.run(aid, deadline(None, a.hours))
        print(f"{aid}: {s['stage']}" + (f" - {s.get('verdict') or s.get('reason') or ''}"))
    elif a.cmd == "report":
        p = ref.dir(a.id) / "report.md"
        if not p.is_file():
            reports = sorted(ref.dir(a.id).glob("report-try*.md"))
            p = reports[-1] if reports else p
        print(p.read_text(encoding="utf-8") if p.is_file() else f"no report yet for {a.id}")
    elif a.cmd == "approve":
        print(f"merged as {ref.approve(a.id)[:10]}")
    elif a.cmd == "reject":
        ref.reject(a.id, a.reason)
        print("rejected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
