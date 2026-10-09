"""The Newsroom's evals, offline: their code paths run, they score what they should, and they can FAIL (a bench that cannot
report a bad feature is no use). The numbers a real Jev gives are in evals/newsroom_cases.py and evals/bench/newsroom.py."""
import json
import sys
from types import SimpleNamespace

import pytest

from jevh import newsroom as nr
from jevh.config import ROOT

sys.path.insert(0, str(ROOT / "evals"))
sys.path.insert(0, str(ROOT / "evals" / "bench"))

import newsroom_cases as nc  # noqa: E402
import newsroom as bench  # noqa: E402


class FakeJev:
    """A Jev with an opinion: `eager` fires the newsroom on everything; otherwise only where the request says track/watch/..."""

    def __init__(self, cfg=None, eager=False):
        self.eager = eager
        self.client = SimpleNamespace(calls=0, ask=self._ask)

    def pick_tools(self, req, tools, mcp=None):
        want = any(w in req.lower() for w in ("keep an eye", "track", "follow @", "alert", "watch", "newsroom", "keep me", "let me know",
                                              "tell me whenever", "ping me", "widget", "notify me", "dashboard", "feed", "board"))
        want = want and not any(w in req.lower() for w in ("follow up", "follow me", "monitor my server", "remind me", "timer"))
        scores = {t: 0.1 for t in tools}
        if "newsroom" in tools:
            scores["newsroom"] = 0.95 if (want or self.eager) else 0.05
        return SimpleNamespace(tools=[], scores=scores)

    def route_persona(self, text, personas):
        return SimpleNamespace(persona="daily_assistant" if "news" in text.lower() or "keep" in text.lower() else next(iter(personas)),
                               confidence=0.9, probabilities={})

    def _ask(self, state, questions):
        item, interest = state["item"].lower(), state["interest"].lower()
        hit = any(w in item for w in interest.split() if len(w) > 5)
        return SimpleNamespace(answers={"match": SimpleNamespace(noul=0.9 if hit else 0.1)})


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(nc, "RAW", tmp_path / "raw.json")
    monkeypatch.setattr(nc, "SUMMARY", tmp_path / "summary.txt")
    return tmp_path


def run_cases(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["newsroom_cases.py", *argv])
    nc.main()


def test_the_decision_eval_runs_every_stage_and_reports_what_it_measured(paths, monkeypatch, capsys):
    monkeypatch.setattr(nc, "Jev", lambda cfg: FakeJev(cfg))
    run_cases(monkeypatch, "--stage", "tool,route,about")
    out = capsys.readouterr().out
    for part in ("== tool:", "== route:", "== about:", "regression (", "the other tools' scores with and without", "keywords only", "about only",
                 "keywords or about", "sweep"):
        assert part in out, part
    raw = json.loads((paths / "raw.json").read_text())
    assert len(raw["tool"]["dev"]) == len(nc.TOOL_DEV) and raw["tool"]["regress"]["old"] and "newsroom" not in raw["tool"]["regress"]["old"][0]["runs"][0]
    assert {r["part"] for r in raw["route"]["new"]} == {"newsroom", "existing"}
    assert (paths / "summary.txt").read_text().startswith("newsroom decisions")
    run_cases(monkeypatch, "--offline", "--stage", "tool,about")      # re-reads the saved scores, no Jev
    assert "== tool:" in capsys.readouterr().out


def test_the_decision_eval_names_a_description_that_fires_on_everything(paths, monkeypatch, capsys):
    monkeypatch.setattr(nc, "Jev", lambda cfg: FakeJev(cfg, eager=True))
    run_cases(monkeypatch, "--stage", "tool")
    out = capsys.readouterr().out
    assert "MISS (should not fire): What's the latest news on Nvidia?" in out
    assert "PICKED:" in out and "(target 0)" in out
    assert "fires on 100% of" in out


def test_the_held_out_sets_are_separate_from_the_dev_sets_and_the_labels_are_balanced():
    dev, held = {r for r, _ in nc.TOOL_DEV}, {r for r, _ in nc.TOOL_HOLDOUT}
    assert not dev & held
    for cases in (nc.TOOL_DEV, nc.TOOL_HOLDOUT):
        fire = sum(1 for _, f in cases if f)
        assert 10 <= fire <= len(cases) - 10, "need both kinds of request"
    for cases in (nc.ABOUT_DEV, nc.ABOUT_HOLDOUT):
        yes = sum(1 for c in cases if c[2])
        assert 0.35 < yes / len(cases) < 0.65
    assert not {c[1] for c in nc.ABOUT_DEV} & {c[1] for c in nc.ABOUT_HOLDOUT}


def test_the_about_cases_are_the_hard_kind_keywords_alone_get_wrong():
    """If the keywords already got these right the `about` alert would have nothing to prove."""
    rows = [{"about": a, "headline": h, "label": lab, "keywords": k, "runs": [0.5]} for a, h, lab, k in nc.ABOUT_DEV + nc.ABOUT_HOLDOUT]
    kw_wrong = sum(1 for r in rows if nc.about_decisions(r, 0.7)["kw"][0] != r["label"])
    assert kw_wrong >= 8, f"only {kw_wrong} cases that keywords get wrong"
    assert any(r["label"] and not nc.about_decisions(r, 0.7)["kw"][0] for r in rows), "no headline the keywords miss"
    assert any(not r["label"] and nc.about_decisions(r, 0.7)["kw"][0] for r in rows), "no keyword false alarm"


def test_real_items_can_be_exported_and_labelled(paths, monkeypatch, tmp_path):
    cfg = SimpleNamespace(workspace=tmp_path / "ws", raw={}, t=lambda n: 0.7, jev={})
    monkeypatch.setattr(nc, "load_config", lambda: cfg)
    room = nr.Newsroom(cfg.workspace / "newsroom.db")
    monkeypatch.setattr(nr, "settings", lambda raw: {})
    w = room.add_widget("news", {"topic": "x"}, alert={"keywords": ["quarter"], "about": "their results"})
    room._run("INSERT INTO items(widget_id, uid, title, url, source, summary, author, first_seen) VALUES (?,?,?,?,?,?,?,?)",
              (w["id"], "1", "Record quarter", "https://ex.test/1", "S", "Revenue up", "", 1.0))
    nc.export_items(str(tmp_path / "items.jsonl"))
    rows = [json.loads(x) for x in (tmp_path / "items.jsonl").read_text().splitlines()]
    assert rows[0]["about"] == "their results" and rows[0]["label"] is None and "Record quarter" in rows[0]["headline"]
    rows[0]["label"] = True
    (tmp_path / "items.jsonl").write_text(json.dumps(rows[0]) + "\n")
    assert nc.labelled(str(tmp_path / "items.jsonl")) == [("their results", "Record quarter - Revenue up", True, ["quarter"])]


# ---------------------------------------------------------------- the end-to-end bench, in its dry mode

@pytest.fixture
def bench_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(bench, "RAW", tmp_path / "raw.json")
    monkeypatch.setattr(bench, "SUMMARY", tmp_path / "summary.txt")
    monkeypatch.delenv("JEVH_NEWSROOM_ALLOW_LOCAL", raising=False)
    return tmp_path


def run_bench(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["newsroom.py", *argv])
    bench.main()


def test_the_bench_plumbing_works_end_to_end_with_a_scripted_worker(bench_paths, monkeypatch, capsys):
    run_bench(monkeypatch, "--dry", "--stage", "turns,dashboard", "--reps", "1",
              "--task", "nvidia_earnings,own_feed,remove_bitcoin,list_board,latest_nvidia,percent")
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "[track]" in out and "[manage]" in out and "[lookalike]" in out and "[ordinary]" in out
    assert "MISS" not in out.split("dashboard (")[0], out                     # every scripted turn did the right thing, so every check passes
    assert "ok   new nvidia_earnings: 1/1" in out
    assert "ok   old latest_nvidia: 1/1" in out and "ok   new latest_nvidia: 1/1" in out
    assert "dashboard (" in out and "MISS" not in out.split("dashboard (")[1], out
    raw = json.loads((bench_paths / "raw.json").read_text())
    assert {r["arm"] for r in raw["turns"] if r["task"] == "latest_nvidia"} == {"old", "new"}      # the lookalikes ran on both arms
    assert {r["arm"] for r in raw["turns"] if r["task"] == "nvidia_earnings"} == {"new"}            # ...the tracking ones only on this harness


def test_the_bench_reports_a_worker_that_makes_a_widget_for_a_one_off_question(bench_paths, monkeypatch, capsys):
    real = bench.dry_script

    def overeager(name, task, feed):
        if name == "latest_nvidia":
            return {"code": "newsroom.add('news', topic='Nvidia', alert=True)\nfinal('Here is the news you asked for, all of it, and more.')", "fire": True}
        return real(name, task, feed)
    monkeypatch.setattr(bench, "dry_script", overeager)
    run_bench(monkeypatch, "--dry", "--stage", "turns", "--reps", "1", "--arm", "new", "--task", "latest_nvidia")
    out = capsys.readouterr().out
    assert "MISS new latest_nvidia: 0/1" in out and "- no_widget: 0/1" in out


def test_the_bench_reports_a_worker_that_says_it_did_it_when_the_board_is_empty(bench_paths, monkeypatch, capsys):
    real = bench.dry_script
    monkeypatch.setattr(bench, "dry_script", lambda n, t, f: {"code": "final('Done! I will notify you about earnings.')", "fire": True}
                        if n == "nvidia_earnings" else real(n, t, f))
    run_bench(monkeypatch, "--dry", "--stage", "turns", "--reps", "1", "--task", "nvidia_earnings")
    out = capsys.readouterr().out
    assert "MISS new nvidia_earnings: 0/1" in out and "- one_news_widget_on_nvidia: 0/1" in out and "- notifies_on_earnings: 0/1" in out
    assert "answer_says_what_notifies" not in out      # (the answer did say it: the checks read the board, not the answer)


def test_the_real_source_list_covers_every_kind():
    assert {k for k, _ in bench.SOURCES} == set(nr.KINDS) - {"youtube"}, "a kind with no real source to check (youtube needs a channel id of yours)"
