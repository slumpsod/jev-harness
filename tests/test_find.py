"""project.find: keyword search + Jev -> exact locations; project files respect .gitignore."""
import subprocess

import pytest

from jevh.tools import _pick, project

APP = '''"""Orders app: routes, billing and the database."""
import sqlite3


def total_for(order):
    """Sum of an order's line prices."""
    return sum(line.price for line in order.lines)


def refund(order, amount):
    """Give money back on an order (partial refunds allowed)."""
    stripe.Refund.create(charge=order.charge_id, amount=amount)
'''


@pytest.fixture
def proj(tmp_path, monkeypatch):
    (tmp_path / "app").mkdir()
    (tmp_path / "app/billing.py").write_text(APP, encoding="utf-8")
    (tmp_path / "README.md").write_text("# Orders\nRefunds are handled in billing.\n", encoding="utf-8")
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace/run.log").write_text("refund refund refund\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("workspace/\n", encoding="utf-8")
    monkeypatch.setattr(project, "root", str(tmp_path))
    return tmp_path


class FakeJev:
    """Picks the option containing `prefer`; records what it was offered."""
    def __init__(self, prefer):
        self.prefer, self.offered = prefer, []

    def ask(self, state, questions):
        from jevh.decide import Answer, Response
        answers = {}
        for key, q in questions.items():
            opts = q["criteria"]
            self.offered.append(dict(opts))
            best = next((k for k, v in opts.items() if self.prefer in (v or "")), next(iter(opts)))
            answers[key] = Answer(key, "choice", {"choice": best, "confidence": 1.0,
                                                  "probabilities": {k: (0.9 if k == best else 0.1 / len(opts))
                                                                    for k in opts}})
        return Response("fake", answers)


def use_jev(monkeypatch, prefer):
    jev = FakeJev(prefer)
    monkeypatch.setattr(_pick, "client_from_env", lambda: (jev, 16))
    return jev


def test_files_respect_gitignore_in_a_git_repo(proj):
    subprocess.run(["git", "init", "-q"], cwd=proj, check=True)
    assert "workspace/run.log" not in list(project._files())
    assert "app/billing.py" in list(project._files())
    assert not any("workspace" in g for g in project.grep("refund"))


def test_without_git_everything_is_searched(proj):
    assert "workspace/run.log" in list(project._files())


def test_find_returns_the_definition_and_read_opens_it(proj, monkeypatch):
    subprocess.run(["git", "init", "-q"], cwd=proj, check=True)
    jev = use_jev(monkeypatch, "def refund")
    hits = project.find("where do we give money back to a customer?", k=2)
    assert hits[0].startswith("app/billing.py:10: def refund(order, amount):")
    body = project.read(hits[0])
    assert "stripe.Refund.create" in body and "   10  def refund" in body
    # step two offers definitions (with their docstrings), not the lines restating the question
    offered = " ".join(" ".join(o.values()) for o in jev.offered)
    assert "def refund(order, amount): | Give money back on an order" in offered
    assert "(partial refunds allowed)" not in offered.split("| Give money back")[0]     # no bare body lines


def test_keywords_drop_stop_words_and_split_names():
    kw = project._keywords("Where are the API routes for order_items defined?")
    assert {"api", "routes", "route", "order_items", "order", "items"} <= kw
    assert not {"where", "are", "the", "defined"} & kw
