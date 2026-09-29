"""Long tasks: compaction, progress checks, checkpoints + resume, background jobs."""
import json
from pathlib import Path

from jevh.jev import Jev
from jevh.orchestrator import Harness
from jevh.repl.client import Repl

from conftest import FakeSystemOne, ScriptedLLM, choice_ans, noul_ans, py


def rules(circles=0.05):
    def r(k, q, s):
        if k == "persona":
            return choice_ans("daily_assistant", 0.9, q["criteria"])
        if k == "circles":
            return noul_ans(circles)
    return r


def harness(cfg, replies, circles=0.05, events=None, **limits):
    cfg.limits.update(limits)
    llm = ScriptedLLM(replies)
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules(circles))), llm_factory=lambda p: llm,
                on_event=(lambda k, d: events.append((k, d))) if events is not None else None)
    h.learning = False
    return h, llm


def steps(n, final=True, big=False):
    body = "print('step {i} done'); print('y' * 3000)" if big else "print('step {i} done')"
    out = [py(body.format(i=i)) for i in range(n)]
    return out + ([py("final('all done')")] if final else [])


def test_compaction_keeps_the_conversation_small_and_well_formed(cfg):
    events = []
    h, llm = harness(cfg, steps(8, big=True), events=events, compact_tokens=3000, keep_steps=2)
    r = h.turn("do a long job")
    assert r.answer == "all done"
    compacts = [d for k, d in events if k == "compact"]
    assert compacts and all(c["tokens_after"] < c["tokens_before"] for c in compacts)
    last = llm.seen[-1]
    roles = [m["role"] for m in last]
    assert roles[0] == "system" and roles[1] == "user" and roles[2] == "assistant"
    assert all(a != b for a, b in zip(roles[1:], roles[2:]))       # strictly alternating after the system prompt
    assert "[Progress so far" in last[1]["content"] and "step 0 done" in last[1]["content"]
    assert "do a long job" in last[1]["content"]                     # the task itself is never condensed away
    assert "step 7 done" in last[-1]["content"]                      # the latest step is verbatim


def tail_steps(n):
    """Big steps whose output ends with a marker a one-line note (first 400 chars) would lose."""
    return [py(f"print('step {i} done'); print('y' * 3000); print('TA' + 'IL{i}')") for i in range(n)] + \
        [py("final('all done')")]


def compact_harness(cfg, replies, needed=lambda step: False, events=None, **limits):
    """Jev's compaction question answers high for the steps `needed(step text)` picks."""
    base = rules()

    def r(k, q, s):
        if k == "info":
            if isinstance(needed, Exception):
                raise needed
            return noul_ans(0.95 if needed(s["step"]) else 0.2)
        return base(k, q, s)
    cfg.limits.update(limits)
    llm = ScriptedLLM(replies)
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(r)), llm_factory=lambda p: llm,
                on_event=(lambda k, d: events.append((k, d))) if events is not None else None)
    h.learning = False
    return h, llm


def test_compaction_keeps_the_steps_jev_marks_needed_in_full(cfg):
    events = []
    h, llm = compact_harness(cfg, tail_steps(8), needed=lambda s: "step 1 done" in s, events=events,
                             compact_tokens=3000, keep_steps=2)
    assert h.turn("do a long job").answer == "all done"
    assert len(llm.seen) == 9                        # one call per step: compaction never calls the model
    log = llm.seen[-1][1]["content"]
    assert "TAIL1" in log                            # needed: kept in full, output tail included
    assert "TAIL0" not in log and "step 0 done" in log   # not needed: a one-line note
    last = [d for k, d in events if k == "compact"][-1]
    assert last["full"] == 1 and last["notes"] >= 1 and last["dropped"] == 0


def test_compaction_that_saves_nothing_is_skipped_and_jev_asked_once_per_step(cfg):
    # all steps needed and small: their full form is no smaller than the original (live: 17.9k -> 18.5k)
    events, asked = [], []
    replies = [py(f"print('step {i} done'); print('z' * 700)") for i in range(10)] + [py("final('all done')")]
    h, llm = compact_harness(cfg, replies, needed=lambda s: asked.append(s) or True, events=events,
                             compact_tokens=2500, keep_steps=2)
    assert h.turn("do a long job").answer == "all done"
    assert asked and not [c for k, c in events if k == "compact"]   # scored, then skipped: nothing to gain
    assert len(asked) == len(set(asked))            # a skipped compaction doesn't re-ask about the same steps


def test_compaction_log_stays_within_budget(cfg, monkeypatch):
    import jevh.worker as worker
    monkeypatch.setattr(worker, "PROGRESS_CHARS", 3000)
    events = []
    h, llm = compact_harness(cfg, tail_steps(12), needed=lambda s: True, events=events,
                             compact_tokens=3000, keep_steps=2)
    assert h.turn("do a long job").answer == "all done"
    log = llm.seen[-1][1]["content"].split("[Progress so far", 1)[1]
    assert len(log) < 3000 + 400
    last = [d for k, d in events if k == "compact"][-1]
    assert last["dropped"] > 0 and f"({last['dropped']} earlier steps left out)" in log


def test_compaction_without_jev_falls_back_to_notes(cfg):
    events = []
    h, llm = compact_harness(cfg, tail_steps(8), needed=RuntimeError("jev down"), events=events,
                             compact_tokens=3000, keep_steps=2)
    assert h.turn("do a long job").answer == "all done"
    log = llm.seen[-1][1]["content"]
    assert "step 0 done" in log and "TAIL0" not in log
    assert all(d["full"] == 0 for k, d in events if k == "compact")


def test_compaction_state_survives_a_resume(cfg):
    h, llm = compact_harness(cfg, tail_steps(6)[:-1], needed=lambda s: "step 0 done" in s,
                             compact_tokens=3000, keep_steps=2)
    h.personas["daily_assistant"].max_steps = 5
    r = h.turn("do a long job")
    data = json.loads(Path(r.checkpoint).read_text(encoding="utf-8"))
    kept = [e for e in data["compact"]["entries"] if e["keep"]]
    assert len(kept) == 1 and "TAIL0" in kept[0]["full"]
    assert "TAIL0" in data["progress_log"]


def test_going_in_circles_is_flagged_then_stopped(cfg):
    events = []
    h, llm = harness(cfg, steps(10, final=False), circles=0.95, events=events, progress_every=2)
    r = h.turn("fix the build")
    assert r.answer.startswith("[daily_assistant stopped: its recent steps kept repeating")
    checks = [d for k, d in events if k == "progress_check"]
    assert [c["stuck"] for c in checks] == [True, True] and len(llm.seen) == 4
    assert "The harness's progress check" in llm.seen[2][-1]["content"]   # warned once before stopping
    assert r.checkpoint and Path(r.checkpoint).exists()                  # ...and it can be resumed


def test_healthy_progress_is_left_alone(cfg):
    events = []
    h, _ = harness(cfg, steps(5), circles=0.1, events=events, progress_every=2)
    assert h.turn("do a long job").answer == "all done"
    assert [d["stuck"] for k, d in events if k == "progress_check"] == [False, False]


def test_checkpoint_and_resume_in_the_same_folder(cfg):
    h, llm = harness(cfg, [py("open('draft.txt', 'w').write('half of it')\nprint('saved draft')"),
                           py("print('still going')")])
    h.personas["daily_assistant"].max_steps = 2
    r = h.turn("write the report")
    assert r.checkpoint and "step limit" in r.answer and "resume" in r.answer
    ck = json.loads(Path(r.checkpoint).read_text(encoding="utf-8"))
    assert ck["stopped"] == "max_steps" and ck["persona"] == "daily_assistant"
    assert [x["path"] for x in h.resumable()] == [r.checkpoint]

    h.personas["daily_assistant"].max_steps = 5
    llm.replies = [py("print(open('draft.txt').read())\nfinal('report finished')")]
    r2 = h.resume()
    assert r2.answer == "report finished"
    seen = llm.seen[-1]
    assert any("saved draft" in m["content"] for m in seen) and "Resumed after an interruption" in seen[-1]["content"]
    assert h.resumable() == []                                            # taken over, not offered again


def test_background_jobs(tmp_path):
    with Repl(tmp_path / "wd", tmp_path / "arts") as r:
        out = r.run("import sys\n"
                    "j = start_job(f'{sys.executable} -c \"import time; print(41 + 1); time.sleep(0.5)\"')\n"
                    "print(job_status(j)['running'])\n"
                    "res = job_wait(j, timeout=20)\n"
                    "print(res['running'], res['code'], res['out'].strip())")
    assert "True" in out.output.splitlines()[0] and out.output.splitlines()[-1] == "False 0 42"


def test_resume_brings_back_repl_variables_functions_and_imports(cfg):
    h, llm = harness(cfg, [py("import statistics as st\nrows = [3, 1, 4, 1, 5]\ndef double(x):\n    return 2 * x\n"
                              "print(len(rows))"),
                           py("fh = open('scratch.txt', 'w')\nprint('still going')")])
    h.personas["daily_assistant"].max_steps = 2
    r = h.turn("crunch the numbers")
    assert "step limit" in r.answer

    h.personas["daily_assistant"].max_steps = 5
    llm.replies = [py("final(str(double(st.mean(rows))))")]
    r2 = h.resume()
    assert r2.answer == "5.6"                         # data, the def and the import all came back
    note = llm.seen[-1][-1]["content"]
    assert "variables: rows" in note and "functions/classes: double" in note
    assert "fh" in note                               # an open file can't be saved: named as lost


def test_snapshot_skips_what_cannot_be_saved(tmp_path):
    with Repl(tmp_path / "wd", tmp_path / "arts") as r:
        r.run("import json\ndata = {'a': [1, 2]}\ngen = (i for i in range(3))\nclass P:\n    z = 7")
        info = r.snapshot(tmp_path / "s.pkl")
    assert "data" in info["saved"] and "gen" in info["skipped"] and "P" in info["defs"]
    with Repl(tmp_path / "wd2", tmp_path / "arts") as r:
        back = r.restore(tmp_path / "s.pkl")
        out = r.run("print(data['a'], P.z, json.dumps(1))")
    assert back["restored"] == ["data"] and out.output.strip() == "[1, 2] 7 1"


def test_each_step_reports_what_the_context_is_made_of(cfg):
    events = []
    h, _ = harness(cfg, steps(2, big=True), events=events)
    h.turn("do a long job")
    ctx = [d for k, d in events if k == "context"]
    assert [c["step"] for c in ctx] == [1, 2, 3] and ctx[0]["limit"] == cfg.lim("compact_tokens")
    first, last = ctx[0]["parts"], ctx[-1]["parts"]
    assert set(first) == {"instructions", "skills", "tools", "task", "work", "output"}
    assert first["instructions"] > 0 and first["task"] > 0 and first["work"] == first["output"] == 0
    assert last["work"] > 0 and last["output"] > 500      # two steps printed ~3000 chars each
