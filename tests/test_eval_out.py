"""evals/_out.py: a live eval run never writes over the committed results (issue #8), --offline re-sweeps them, and
every eval and bench saves through it."""
import argparse
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals"))

import _out  # noqa: E402

RAW = ROOT / "evals" / "routing_raw.json"
BENCH_RAW = ROOT / "evals" / "bench" / "flagged_raw.json"


@pytest.fixture(autouse=True)
def no_env(monkeypatch, tmp_path):
    monkeypatch.delenv(_out.ENV, raising=False)
    monkeypatch.setattr(_out, "default_dir", lambda: tmp_path / "workspace" / "evals" / "semif")


def parse(*argv):
    return _out.Out.of(_out.add_args(argparse.ArgumentParser()).parse_args(list(argv)))


def test_a_live_run_writes_outside_git_and_offline_reads_the_committed_copy(tmp_path):
    out = parse()
    assert out.write(RAW) == tmp_path / "workspace" / "evals" / "semif" / "routing_raw.json"
    assert out.write(BENCH_RAW) == tmp_path / "workspace" / "evals" / "semif" / "bench" / "flagged_raw.json"
    assert out.write(BENCH_RAW).parent.is_dir()                      # the folder is made
    assert out.read(RAW) == RAW and out.read(BENCH_RAW) == BENCH_RAW


def test_out_picks_the_folder_for_both_writing_and_re_sweeping(tmp_path, monkeypatch):
    mine = tmp_path / "mine"
    for out in (parse("--out", str(mine)), (monkeypatch.setenv(_out.ENV, str(mine)), parse())[1]):
        assert out.write(RAW) == mine / "routing_raw.json" and out.read(RAW) == mine / "routing_raw.json"
        assert out.read(BENCH_RAW) == mine / "bench" / "flagged_raw.json"


def test_save_committed_is_the_only_way_to_replace_the_committed_results(tmp_path):
    out = parse("--save-committed", "--out", str(tmp_path / "ignored"))
    assert out.write(RAW) == RAW and out.read(RAW) == RAW


def test_a_path_outside_evals_is_used_as_given(tmp_path):
    p = tmp_path / "raw.json"      # a test's monkeypatched RAW, or an eval's own --cache PATH
    run = ROOT / "evals" / "bench" / "runs" / "desktop" / "20991231-000000" / "results.json"   # a run's own (ignored) folder
    for out in (parse(), parse("--out", str(tmp_path / "x")), parse("--save-committed")):
        assert out.write(p) == p and out.read(p) == p
        assert out.read(run) == run


def test_no_eval_or_bench_writes_beside_itself_any_more():
    """Every file that saves results goes through _out: a write_text/open-for-writing on a name like RAW, SUMMARY or a
    *_raw.json / *summary.txt path beside the script, outside `dest.write(...)`, would overwrite the committed copy."""
    bad = []
    for p in sorted((ROOT / "evals").rglob("*.py")):
        src = p.read_text(encoding="utf-8")
        for n, line in enumerate(src.splitlines(), 1):
            if re.search(r"\b(RAW|RAW_HOLDOUT|SUMMARY|SUMMARY_HOLDOUT|OUT|cache)\b(\[\w+\])?\.write_text\(", line):
                bad.append(f"{p.relative_to(ROOT).as_posix()}:{n}: {line.strip()}")
            beside = re.search(r"(HERE /|ROOT / \"evals\"|with_name\()[^#]*\.write_text\(", line)   # a path beside the eval
            if beside and "dest.write(" not in line:
                bad.append(f"{p.relative_to(ROOT).as_posix()}:{n}: {line.strip()}")
    assert bad == []   # (on the code before evals/_out.py this listed 31 writes)
