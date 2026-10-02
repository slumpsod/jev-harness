"""Session review: flags and notes, the session a reviewer reads, the model's verdict, the user's
verdict, the failure log kept in step, and the command line."""
import json

import pytest

from jevh.jev import Jev
from jevh.orchestrator import Harness
from jevh.review import ReviewError, ReviewLog, main, parse_review, review_prompt, reviewer_block

from conftest import FakeSystemOne, ScriptedLLM, choice_ans, py


def rules(k, q, s):
    if k == "persona":
        return choice_ans("daily_assistant", 0.9, q["criteria"])


def verdict(v="failure", **extra):
    return json.dumps({"verdict": v, "confidence": 0.8, "summary": f"judged {v}",
                       "problems": [{"what": "the total is wrong", "turn": 1, "note": 1,
                                     "evidence": "print(2 + 2) -> 5", "severity": "high"}],
                       "notes_checked": [{"note": 1, "holds": True, "why": "the log shows 5"}], **extra})


def harness(cfg, replies, reviewer_replies=()):
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: ScriptedLLM(list(replies)))
    h.learning = False
    reviewer = ScriptedLLM(list(reviewer_replies), name="reviewer-model")
    h.review._factory = lambda block: reviewer
    return h, reviewer


def test_flags_notes_and_statuses(tmp_path):
    log = ReviewLog(tmp_path / "r.db")
    a = log.flag("s1", "the chart had the wrong units", turn=2, title="plot my runs")
    assert (a["session"], a["status"], a["title"]) == ("s1", "open", "plot my runs")
    assert [(n["turn"], n["text"]) for n in a["notes"]] == [(2, "the chart had the wrong units")]
    again = log.flag("s1", "and no source")          # one flag per session: a second flag adds the note
    assert again["id"] == a["id"] and len(again["notes"]) == 2
    log.flag("s2")
    assert [f["session"] for f in log.list()] == ["s2", "s1"]
    assert log.resolve(str(a["id"]))["session"] == "s1" and log.resolve("s2")["notes"] == []
    assert log.set_status(a["id"], "dismissed") and log.list(status="dismissed")[0]["id"] == a["id"]
    assert log.flag("s1")["status"] == "open"         # flagging again brings it back
    log.set_verdict(a["id"], "failure", "user", "wrong units")
    assert log.counts() == {"failure": 1, "open": 1}
    with pytest.raises(ValueError):
        log.set_verdict(a["id"], "maybe")
    with pytest.raises(ValueError):
        log.add_note(a["id"], "   ")
    with pytest.raises(KeyError):
        log.add_note(999, "x")


def test_parse_review_takes_fenced_json_and_rejects_no_verdict():
    r = parse_review("Here it is:\n```json\n" + verdict("not failure") + "\n```")
    assert r["verdict"] == "not_failure" and r["confidence"] == 0.8 and r["problems"][0]["severity"] == "high"
    with pytest.raises(ReviewError):
        parse_review("I think it failed.")
    with pytest.raises(ReviewError):
        parse_review('{"verdict": "unclear"}')


def test_model_review_reads_the_session_and_logs_a_failure(cfg):
    h, reviewer = harness(cfg, [py("print(2 + 2)"), py("final('2 + 2 = 5')")], [verdict("failure")])
    h.turn("what is 2 + 2?")
    flag = h.review.flag(h.session, "the sum is wrong", turn=1)
    assert flag["title"] == "what is 2 + 2?"
    out = h.review.run(flag["id"])
    prompt = reviewer.seen[0][-1]["content"]
    for part in ("1. (turn 1) the sum is wrong", "### Turn 1 (daily_assistant)", "Assistant: 2 + 2 = 5",
                 "Request: what is 2 + 2?", "CODE:\nprint(2 + 2)", "OUTPUT: 4"):
        assert part in prompt, part
    assert (out["verdict"], out["verdict_by"], out["status"]) == ("failure", "model", "decided")
    assert out["review"]["model"] == "reviewer-model" and out["review"]["problems"][0]["what"] == "the total is wrong"
    f = h.failures.get(out["failure_id"])
    assert (f["source"], f["kind"], f["session"], f["persona"]) == ("review", "review_failure", h.session,
                                                                     "daily_assistant")
    assert f["request"] == "what is 2 + 2?" and "the sum is wrong" in f["detail"] and "print(2 + 2)" in f["work_log"]


def test_the_users_verdict_overrules_the_model_and_the_failure_log_follows(cfg):
    h, _ = harness(cfg, [py("final('done')")], [verdict("failure"), verdict("failure")])
    h.turn("tidy my notes")
    fid = h.review.flag(h.session, "looks wrong")["id"]
    first = h.review.run(fid)
    out = h.review.decide(fid, "not_failure", "it was fine, I misread it")
    assert (out["verdict"], out["verdict_by"], out["reason"]) == ("not_failure", "user", "it was fine, I misread it")
    assert h.failures.get(first["failure_id"])["status"] == "ignored"
    again = h.review.run(fid)                          # a new model review doesn't undo the user's call
    assert again["verdict"] == "not_failure" and again["review"]["verdict"] == "failure"
    out = h.review.decide(fid, "failure")
    assert out["failure_id"] == first["failure_id"] and h.failures.get(first["failure_id"])["status"] == "open"
    assert len(h.failures.list(source="review")) == 1


def test_a_not_failure_verdict_logs_nothing_and_an_empty_session_cant_be_reviewed(cfg):
    h, _ = harness(cfg, [py("final('ok')")], [verdict("not_failure")])
    empty = h.review.flag("20990101-000000-none")
    with pytest.raises(ReviewError):
        h.review.run(empty["id"])
    assert h.review.running == set()
    h.turn("hi")
    out = h.review.run(h.review.flag(h.session)["id"])
    assert out["verdict"] == "not_failure" and out["failure_id"] is None and h.failures.list(source="review") == []


def test_a_bad_reply_is_an_error_and_the_flag_stays_open(cfg):
    h, _ = harness(cfg, [py("final('ok')")], ["no idea, sorry"])
    h.turn("hi")
    fid = h.review.flag(h.session, "check it")["id"]
    with pytest.raises(ReviewError, match="no JSON"):
        h.review.run(fid)
    assert h.review.reviews.get(fid)["status"] == "open"


def test_prompt_fits_long_work_logs_into_its_budget():
    rec = {"turns": [{"n": 1, "persona": "dev", "user": "build it", "assistant": "built"}], "failures": [],
           "tasks": [{"turn": 1, "task": i, "persona": "dev", "request": "build", "answer": "", "stopped": "",
                      "steps": 3, "work_log": "x" * 50000 + f"END{i}"} for i in (1, 2)]}
    p = review_prompt(rec, [{"turn": None, "text": "it broke"}], budget=20000)
    assert len(p) < 24000 and "END1" in p and "END2" in p and "1. it broke" in p


def test_reviewer_model_choice(cfg):
    cfg.raw = {**cfg.raw, "review": {"model": "lmstudio:judge-7b", "max_tokens": 4000}}
    assert reviewer_block(cfg) == {"provider": "lmstudio", "model": "judge-7b", "max_tokens": 4000}
    assert reviewer_block(cfg, "anthropic:claude-opus-5")["provider"] == "anthropic"
    cfg.raw = {**cfg.raw, "review": {}}
    cfg.models = {**cfg.models, "strong": {"provider": "lmstudio", "model": "big", "think": "unlimited"}}
    assert reviewer_block(cfg) == {"provider": "lmstudio", "model": "big", "max_tokens": 32768}
    assert reviewer_block(cfg, model_override="vllm:x")["model"] == "x"
    with pytest.raises(ValueError):
        reviewer_block(cfg, "no-provider")


def test_command_line(cfg, capsys, monkeypatch):
    h, _ = harness(cfg, [py("final('ok')")])
    h.turn("plan my week")
    monkeypatch.setattr("jevh.config.load_config", lambda path=None: cfg)
    assert main(["sessions"]) == 0 and h.session in capsys.readouterr().out
    assert main(["flag", h.session, "it skipped Tuesday", "--turn", "1"]) == 0
    assert main(["note", h.session, "and Friday"]) == 0
    assert main(["verdict", h.session, "failure", "missing days"]) == 0
    out = capsys.readouterr().out
    assert "note 1 (turn 1): it skipped Tuesday" in out and "your verdict: failure - missing days" in out
    assert main([]) == 0 and "[failure (user)]" in capsys.readouterr().out
    assert main(["dismiss", h.session]) == 0 and main(["show", "nope"]) == 1
    assert ReviewLog(cfg.workspace / "reviews.db").find(h.session)["status"] == "dismissed"
