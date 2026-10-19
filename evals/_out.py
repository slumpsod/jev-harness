"""Where the evals and benches keep what they measure.

The raw results and summaries committed beside each eval (evals/*_raw.json, evals/bench/*_summary.txt, ...) are the
numbers the README quotes and what `--offline` re-sweeps. A live run used to write over them: running an eval on your
machine, or against another Jev backend, replaced the committed results and dirtied the checkout (which `jevh update`
then had to stash). Now:

- A live run writes to `<workspace>/evals/<jev backend>/` (outside git; e.g. workspace/evals/semif/), keeping each
  file's name, with the benches' files under `bench/`. `--out DIR` (or the JEVH_EVAL_OUT environment variable) picks
  another folder.
- `--offline` re-sweeps the committed results, unless `--out DIR` / JEVH_EVAL_OUT says to read a folder of your own
  (e.g. `--offline --out workspace/evals/semif` re-sweeps your last live run).
- `--save-committed` writes the committed files themselves: a run whose numbers should become the new reference.

A path outside evals/ (a test's tmp file, an eval's own --cache PATH), or in a run's own git-ignored folder
(evals/bench/runs/...), is used as given, for reading and writing.

Usage in an eval:

    import _out                      # evals/ is on sys.path (a bench adds it)
    _out.add_args(ap)
    a = ap.parse_args()
    out = _out.Out.of(a)
    rows = json.loads(out.read(RAW).read_text())           # --offline
    out.write(RAW).write_text(json.dumps(rows))            # live (makes the folder)
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

EVALS = Path(__file__).resolve().parent
ROOT = EVALS.parent
ENV = "JEVH_EVAL_OUT"
UNTRACKED = (EVALS / "bench" / "runs", EVALS / "bench" / "data")   # git-ignored: each run's own folders


def add_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = ap.add_argument_group("where results go (see evals/_out.py)")
    g.add_argument("--out", metavar="DIR",
                   help=f"write results here instead of workspace/evals/<backend>/ (or set {ENV}); "
                        "with --offline, read them from here instead of the committed copy")
    g.add_argument("--save-committed", action="store_true",
                   help="write (and read) the committed results beside the eval: make this run the new reference")
    return ap


def default_dir() -> Path:
    """<workspace>/evals/<jev backend>: the config's workspace (git-ignored) and Jev backend."""
    try:
        from jevh.config import load_config
        cfg = load_config()
        workspace, backend = Path(cfg.workspace), str((cfg.jev or {}).get("backend") or "jev")
    except Exception:  # noqa: BLE001 - an eval must still run (and say where it wrote) without a usable config
        workspace, backend = ROOT / "workspace", "jev"
    return workspace / "evals" / (re.sub(r"[^A-Za-z0-9._-]+", "_", backend) or "jev")


class Out:
    def __init__(self, out: str | os.PathLike | None = None, save_committed: bool = False):
        given = out or os.environ.get(ENV) or None
        self.given = Path(given).expanduser().resolve() if given else None   # told where, by --out or the env
        self.save_committed = bool(save_committed)
        self._dir: Path | None = None

    @classmethod
    def of(cls, args: argparse.Namespace | None) -> "Out":
        return cls(getattr(args, "out", None), getattr(args, "save_committed", False))

    @property
    def dir(self) -> Path:
        """The folder a live run writes to."""
        if self._dir is None:
            self._dir = self.given or default_dir()
        return self._dir

    @staticmethod
    def _rel(committed: Path) -> Path | None:
        p = Path(committed).resolve()
        if any(p.is_relative_to(d) for d in UNTRACKED):
            return None   # a run's own folder (git-ignored): used as given
        try:
            return p.relative_to(EVALS)
        except ValueError:
            return None   # not one of the committed files: used as given

    def write(self, committed: str | os.PathLike) -> Path:
        """Where a live run saves `committed` (the file's path beside the eval). Its folder is made."""
        committed = Path(committed)
        rel = self._rel(committed)
        p = committed if rel is None or self.save_committed else self.dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def read(self, committed: str | os.PathLike) -> Path:
        """Where --offline reads `committed` from: the committed copy, unless --out / JEVH_EVAL_OUT named a folder."""
        committed = Path(committed)
        rel = self._rel(committed)
        if rel is None or self.save_committed or self.given is None:
            return committed
        return self.given / rel

    def where(self, p: str | os.PathLike) -> str:
        """`p` for a message: relative to the checkout when it is inside it."""
        p = Path(p).resolve()
        try:
            return p.relative_to(ROOT).as_posix()
        except ValueError:
            return str(p)
