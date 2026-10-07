"""The coding tasks of evals/bench/serve_compare.py: every hidden suite passes on its reference solution, the seeded
bugs really fail, and the grader reads pytest's summary. Offline (no model, no server)."""
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals" / "bench"))

from serve_compare import grade_coding, seed_task  # noqa: E402
from serve_tasks import TASKS  # noqa: E402

REF = ROOT / "evals" / "bench" / "serve_tasks_ref"


@pytest.mark.parametrize("tid", sorted(TASKS))
def test_reference_passes_hidden_tests(tid, tmp_path):
    seed_task(TASKS[tid], tmp_path)
    ref = {"todocli": "todo.py", "bugfix": "textstats.py"}.get(tid, f"{tid}.py")
    shutil.copy(REF / ref, tmp_path / ref)
    r = grade_coding(TASKS[tid], tmp_path)
    assert r["total"] >= 5, r
    assert r["passed"] == r["total"], r


def test_seeded_bugs_fail_before_the_fix(tmp_path):
    seed_task(TASKS["bugfix"], tmp_path)
    r = grade_coding(TASKS["bugfix"], tmp_path)
    assert 0 < r["passed"] < r["total"], r   # some already right, the rest are the bugs to find


def test_nothing_written_scores_zero(tmp_path):
    seed_task(TASKS["ledger"], tmp_path)
    r = grade_coding(TASKS["ledger"], tmp_path)
    assert r["passed"] == 0 and r["total"] >= 1 and not r["all_passed"]
