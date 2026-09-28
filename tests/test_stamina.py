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
