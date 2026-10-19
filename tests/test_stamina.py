"""Long tasks: compaction, progress checks, checkpoints + resume, background jobs."""
import json
from pathlib import Path

import pytest

from jevh.jev import Jev
from jevh.orchestrator import Harness
from jevh.repl.client import Repl

from conftest import FakeSystemOne, ScriptedLLM, choice_ans, noul_ans, py


@pytest.fixture(autouse=True)
def estimate_sizing(cfg):
    """The compaction mechanics below are tested on the 3.2-chars-a-token estimate, with the four context settings off
    (a scripted model's token count is a placeholder, 10). Tests of a setting turn it on; the shipped values (all on,
    since 2026-10-18) are tested together in test_the_shipped_context_settings_work_together."""
    cfg.raw = {**cfg.raw, "context": {**(cfg.raw.get("context") or {}), "measure": False, "archive": False,
                                      "note_tail": False, "log_share": 0.0}}


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
    assert [c["stuck"] for c in checks] == [True, True] and len(llm.seen) == 5
    assert "The harness's progress check" in llm.seen[2][-1]["content"]   # warned once before stopping
    assert "This is your last step" in llm.seen[4][-1]["content"]         # then one step to report
    assert r.checkpoint and Path(r.checkpoint).exists()                  # ...and it can be resumed


def test_the_quick_reply_can_say_why_the_harness_flagged_the_task(cfg):
    """Notebook: asked "why did the harness flag you?", the task's model had no record of it and guessed."""
    class Watch(list):   # what a quick reply would be given, at each event while the task runs
        digests: list[str] = []

        def append(self, item):
            super().append(item)
            lead = holder[0]._lead
            if lead is not None:
                self.digests.append(lead.digest())
    holder = []
    watch = Watch()
    h, _ = harness(cfg, steps(10, final=False), circles=0.95, events=watch, progress_every=2)
    holder.append(h)
    h.turn("fix the build")
    flagged = [d for d in watch.digests if "What the harness did to this task:" in d]
    assert flagged and "progress check" in flagged[-1] and "score 0.95" in flagged[-1]
    assert not any("What the harness did" in d for d in watch.digests[:2])     # nothing to say before it flagged anything


def test_a_stopped_worker_reports_on_its_last_step(cfg):
    """Live (browser tasks): a stalled worker was cut off and the user got raw REPL output ("105 /
    Are You Human?"); given one step, it says what it found and what blocked it."""
    events = []
    h, llm = harness(cfg, steps(4, final=False) + [py("final('The ticket site shows a bot check; stopped at "
                                                          "https://www.axs.com/events/1')")],
                     circles=0.95, events=events, progress_every=2)
    r = h.turn("get me two tickets")
    assert r.answer.startswith("The ticket site shows a bot check"), r.answer


def test_the_last_step_report_is_not_sent_back(cfg):
    """Live (Target): the report ended with an offer, the final() gate bounced it as "says it will do
    more", and the worker went back to working around the site's bot block."""
    events = []
    cfg.raw = {**cfg.raw, "verify": {**(cfg.raw.get("verify") or {}), "enabled": True}}   # the gate is what's tested
    h, llm = harness(cfg, steps(4, final=False) + [py("final('Target blocks automated browsers. Next I will try "
                                                          "again if you open the window.')"), py("print('more')")],
                     circles=0.95, events=events, progress_every=2)
    r = h.turn("add the coffee to my cart")
    assert r.answer.startswith("Target blocks automated browsers"), r.answer
    assert len(llm.seen) == 5 and not any(d.get("bounced") for k, d in events if k == "verify")


def test_healthy_progress_is_left_alone(cfg):
    events = []
    h, _ = harness(cfg, steps(5), circles=0.1, events=events, progress_every=2)
    assert h.turn("do a long job").answer == "all done"
    assert [d["stuck"] for k, d in events if k == "progress_check"] == [False, False]


def test_progress_every_zero_switches_the_check_off(cfg):
    """(it raised ZeroDivisionError at the first step, failing every code task)"""
    events = []
    h, _ = harness(cfg, steps(3), events=events, progress_every=0)
    assert h.turn("do a long job").answer == "all done"
    assert not [k for k, _ in events if k == "progress_check"]


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


def test_a_resumed_task_that_stops_again_can_be_resumed_from_where_it_got_to(cfg):
    """A resumed run saves its checkpoint over the one it resumed (same folder). resume() then wrote the OLD
    conversation back over it, marked "resumed_by": the resumed run's progress was lost, and a task that hit the
    step limit a second time was no longer offered at all."""
    h, llm = harness(cfg, [py("open('draft.txt', 'w').write('1')\nprint('step one')"), py("print('step two')")])
    h.personas["daily_assistant"].max_steps = 2
    r = h.turn("write the report")
    assert "step limit" in r.answer

    llm.replies = [py("print('step three')"), py("print('step four')")]
    r2 = h.resume()
    assert "step limit" in r2.answer and r2.checkpoint == r.checkpoint
    ck = json.loads(Path(r2.checkpoint).read_text(encoding="utf-8"))
    assert "resumed_by" not in ck and ck["stopped"] == "max_steps"
    assert any("step four" in str(m.get("content")) for m in ck["messages"])    # the second run's own progress
    assert [x["path"] for x in h.resumable()] == [r2.checkpoint]                  # and it can go on

    h.personas["daily_assistant"].max_steps = 5
    llm.replies = [py("final('report finished')")]
    assert h.resume().answer == "report finished"
    assert any("step four" in str(m["content"]) for m in llm.seen[-1])          # it went on from the second run
    assert h.resumable() == []


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


def test_a_timeout_brings_back_the_state_from_the_end_of_the_previous_step(cfg):
    """Live (web.ask, 120s): the block timed out, the REPL restarted and every variable was gone."""
    h, llm = harness(cfg, [py("import statistics as st\nrows = [3, 1, 4, 1, 5]\ndef double(x):\n    return 2 * x\n"
                              "print(len(rows))"),
                           py("made_in_the_stuck_block = 1\nwhile True: pass"),
                           py("final(str(double(st.mean(rows))))")], repl_timeout=2)
    r = h.turn("crunch the numbers")
    assert r.answer == "5.6"                          # data, the def and the import all came back
    note = llm.seen[-1][-1]["content"]
    assert "timed out after 2s" in note and "variables: rows" in note and "functions/classes: double" in note
    assert "previous step" in note and "made_in_the_stuck_block" not in note.split("variables:")[1].split(";")[0]
    assert "all variables were lost" not in note


def test_a_timeout_in_the_first_step_still_says_the_variables_are_gone(cfg):
    h, llm = harness(cfg, [py("x = 1\nwhile True: pass"), py("final('ok')")], repl_timeout=2)
    assert h.turn("go").answer == "ok"
    assert "all variables were lost" in llm.seen[-1][-1]["content"]


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


def test_a_worker_repeating_the_same_result_is_told_to_hand_it_over(cfg):
    """Live (Gemma 4, ARC): a correct grid printed 3 steps in a row, never handed over; the progress
    check said 'try a different approach' and then stopped it, and the grid was lost."""
    same = "print([[8, 8, 9], [8, 2, 9]])"
    events = []
    h, llm = harness(cfg, [py(same)] * 7, circles=0.95, events=events, progress_every=3)
    r = h.turn("solve the puzzle")
    told = llm.seen[3][-1]["content"]
    assert "printed the same result each time" in told and "hand it over now" in told
    assert "genuinely different approach" not in told
    assert r.answer.startswith("[daily_assistant stopped") and "[[8, 8, 9], [8, 2, 9]]" in r.answer


def test_new_pages_count_as_progress_only_when_reached():
    from jevh.worker import new_pages
    earlier = [{"role": "tool", "content": "{'url': 'https://www.booking.com/city/us/denver.html'}"},
               {"role": "assistant", "content": "browser.goto('https://www.booking.com/searchresults.html?x=1')"}]
    recent = [{"role": "assistant", "content": "browser.goto('https://www.booking.com/guessed.html')"},
              {"role": "tool", "content": "{'url': 'https://www.booking.com/city/us/denver.html?d=2', "
                                          "'page': '... https://www.booking.com/searchresults.html ...'}"}]
    # the city page was seen before (query ignored); searchresults was only typed before, now reached;
    # the guessed page was only typed
    assert new_pages(earlier, recent) == ["https://www.booking.com/searchresults.html"]


def test_a_worker_reaching_new_pages_is_not_stopped_as_circling(cfg):
    """Live (Booking.com, BlazeDemo): the first strike came right after real progress - the dates picked, the
    flight chosen - because a browser task's steps look alike to the progress check."""
    events = []
    moving = [py(f"print({{'url': 'https://shop.test/step{i}'}})") for i in range(8)] + [py("final('all done')")]
    h, _ = harness(cfg, moving, circles=0.95, events=events, progress_every=2)
    assert h.turn("go through the checkout").answer == "all done"
    checks = [d for k, d in events if k == "progress_check"]
    assert checks and all(not c["stuck"] and c["new_pages"] for c in checks), checks
    events.clear()
    same = [py("print({'url': 'https://shop.test/cart'})")] * 10
    h, _ = harness(cfg, same, circles=0.95, events=events, progress_every=2)
    assert h.turn("go through the checkout").answer.startswith("[daily_assistant stopped")


def test_a_task_in_the_dashboard_always_gets_the_dock_folder_for_its_browser(cfg, monkeypatch):
    """Live (2026-10-05): Watch browser was off, so the worker had no dock folder; browser_mode() then opened a window
    on the PC's own screen, and the dashboard's Browser panel said "no browser open" to the user on another device."""
    from jevh import orchestrator
    seen = []

    class Spy(orchestrator.Worker):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            seen.append(dict(self.env))
    monkeypatch.setattr(orchestrator, "Worker", Spy)
    for dashboard, watch in ((True, False), (True, True), (False, False)):
        h, _ = harness(cfg, [py("final('ok')")])
        h.browser_dock, h.watch_browser = dashboard, watch
        h.turn("hello")
    assert [("JEVH_BROWSER_DOCK" in e, e["JEVH_BROWSER_HEADED"]) for e in seen] == [(True, "0"), (True, "1"), (False, "0")]
    assert seen[0]["JEVH_BROWSER_DOCK"].endswith("browsers") and seen[0]["JEVH_BROWSER_KEEP"] == "1"
    assert "JEVH_BROWSER_KEEP" not in seen[2]                                 # (outside the dashboard a browser ends with its task)
    # a task of this session asked for a browser to sign in on (nango.browser_mode() left its note): the next turn gets the same
    h, _ = harness(cfg, [py("final('ok')")])
    h.browser_dock, h.watch_browser = True, False
    h.session_dir.mkdir(parents=True, exist_ok=True)
    (h.session_dir / "browser-mode.json").write_text(json.dumps({"visible": True, "profile": "/keep/me"}))
    h.turn("hello again")
    assert seen[-1]["JEVH_BROWSER_HEADED"] == "1" and seen[-1]["JEVH_BROWSER_PROFILE"] == "/keep/me"


def test_bot_check_switch_reaches_the_browser_only_from_config(cfg, monkeypatch):
    """tools.browser_bot_check (default off) is always passed to a task's REPL as JEVH_BROWSER_BOT_CHECK, so a variable left
    in the shell cannot turn the bot-check stop on; only an explicit on in config does."""
    from jevh import orchestrator
    seen = []

    class Spy(orchestrator.Worker):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            seen.append(dict(self.env))
    monkeypatch.setattr(orchestrator, "Worker", Spy)
    monkeypatch.setenv("JEVH_BROWSER_BOT_CHECK", "1")
    h, _ = harness(cfg, [py("final('ok')")])
    h.turn("hello")
    assert seen[-1]["JEVH_BROWSER_BOT_CHECK"] == "0"                 # nothing in config: off, whatever the shell says
    for off in (False, "false", "off", "0", "No"):
        monkeypatch.setitem(cfg.tools, "browser_bot_check", off)
        h, _ = harness(cfg, [py("final('ok')")])
        h.turn("hello")
        assert seen[-1]["JEVH_BROWSER_BOT_CHECK"] == "0", off
    for on in (True, "true", "on", "1"):
        monkeypatch.setitem(cfg.tools, "browser_bot_check", on)
        h, _ = harness(cfg, [py("final('ok')")])
        h.turn("hello")
        assert seen[-1]["JEVH_BROWSER_BOT_CHECK"] == "1", on


# -- context.measure / archive / log_share / note_tail (2026-10-08; evals/bench/long_horizon.py) -----------------------

def with_context(cfg, **settings):
    cfg.raw = {**cfg.raw, "context": {**(cfg.raw.get("context") or {}), **settings}}


def test_archive_keeps_every_condensed_step_in_full_and_the_log_points_to_it(cfg):
    with_context(cfg, archive=True)
    h, llm = compact_harness(cfg, tail_steps(8), events=[], compact_tokens=3000, keep_steps=2)
    r = h.turn("do a long job")
    assert r.answer == "all done"
    log = llm.seen[-1][1]["content"]
    assert "[#1] " in log and "history.md" in log and "TAIL0" not in log      # the note lost the tail ...
    saved = next(cfg.workspace.rglob("history.md")).read_text(encoding="utf-8")
    assert "## [#1]" in saved and "TAIL0" in saved and saved.count("TAIL0") == 1   # ... the file has it, once


def test_archive_off_writes_nothing(cfg):
    h, llm = compact_harness(cfg, tail_steps(8), events=[], compact_tokens=3000, keep_steps=2)
    h.turn("do a long job")
    assert not list(cfg.workspace.rglob("history.md")) and "[#1]" not in llm.seen[-1][1]["content"]


def test_a_note_keeps_the_end_of_a_long_output_when_asked(cfg):
    with_context(cfg, note_tail=True)
    h, llm = compact_harness(cfg, tail_steps(8), events=[], compact_tokens=3000, keep_steps=2)
    assert h.turn("do a long job").answer == "all done"
    log = llm.seen[-1][1]["content"]
    assert "TAIL0" in log and "step 0 done" in log


def test_log_share_lets_the_log_grow_past_the_fixed_budget(cfg, monkeypatch):
    import jevh.worker as worker
    monkeypatch.setattr(worker, "PROGRESS_CHARS", 3000)
    with_context(cfg, log_share=0.9)
    events = []
    h, llm = compact_harness(cfg, tail_steps(12), needed=lambda s: True, events=events, compact_tokens=3000, keep_steps=2)
    assert h.turn("do a long job").answer == "all done"
    log = llm.seen[-1][1]["content"].split("[Progress so far", 1)[1]
    assert len(log) > 3000 + 400                                   # more than the fixed budget allowed above
    assert [d for k, d in events if k == "compact"][-1]["dropped"] == 0


class CountingLLM(ScriptedLLM):
    """A server that counts tokens at `chars_per_token` instead of the harness's 3.2."""
    def __init__(self, replies, chars_per_token):
        super().__init__(replies)
        self.cpt = chars_per_token

    def chat(self, messages):
        r = super().chat(messages)
        r.input_tokens = int(sum(len(str(m.get("content", ""))) for m in messages) / self.cpt)
        return r


def counting_harness(cfg, replies, cpt, **limits):
    cfg.limits.update(limits)
    llm = CountingLLM(replies, cpt)
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules())), llm_factory=lambda p: llm, on_event=lambda k, d: None)
    h.learning = False
    return h, llm


def test_measure_waits_for_text_the_server_counts_as_cheap_and_acts_on_text_it_counts_as_dear(cfg):
    # same conversation, compaction limit 6000 tokens: at 8 chars a token the estimate (3.2) is far too high, so
    # without `measure` the worker condenses early; with it, it waits. At 1 char a token it is the other way round.
    big = steps(10, big=True)
    def condensed(measure, cpt):
        with_context(cfg, measure=measure)
        h, llm = counting_harness(cfg, list(big), cpt, compact_tokens=6000, keep_steps=1)
        h.turn("do a long job")
        return sum("[Progress so far" in s[1]["content"] for s in llm.seen)
    assert condensed(False, 8.0) > condensed(True, 8.0)
    assert condensed(True, 1.0) > condensed(False, 1.0)


def test_the_shipped_context_settings_work_together(cfg):
    # what config.yaml ships since 2026-10-18 (all four on, 80%): on a server that counts 2 chars a token (denser than
    # the 3.2 estimate: code, JSON), the prompts it really receives stay smaller than with the estimate alone, and
    # the archive, the [#n] tags and the notes' tails are all there
    from jevh import context as ctxwin
    from jevh.config import load_config
    shipped = ctxwin.settings(load_config(cfg.source).raw)
    assert (shipped["compact_at"], shipped["measure"], shipped["archive"], shipped["note_tail"], shipped["log_share"]) \
        == (0.8, True, True, True, 0.25)

    def biggest(context):
        cfg.raw = {**cfg.raw, "context": context}
        h, llm = counting_harness(cfg, tail_steps(24), 2.0, compact_tokens=16000, keep_steps=2)
        assert h.turn("do a long job").answer == "all done"
        sent = [sum(len(str(m.get("content", ""))) for m in s) / 2.0 for s in llm.seen]
        return max(sent), llm.seen[-1][1]["content"]
    peak, log = biggest(dict(shipped))
    assert "[Progress so far" in log and "[#1] " in log and "history.md" in log and "TAIL0" in log
    assert next(cfg.workspace.rglob("history.md")).read_text(encoding="utf-8").count("## [#1]") == 1
    peak_estimate, _ = biggest({**shipped, "measure": False, "archive": False, "note_tail": False, "log_share": 0.0})
    assert peak <= 16000 * 1.02 < 16000 * 1.3 < peak_estimate    # (measured here: 15,945 against 24,517)


class _Killed(BaseException):
    """The dashboard's process dying mid-turn (a restart): nothing in the harness catches it."""


def _killed_mid_turn(cfg, events=None):
    """A turn that saves a file and a variable, then dies with the process at its third step."""
    h, llm = harness(cfg, [py("open('notes.txt', 'w').write('node 24 installed')\ninstalled = ['node']\nprint('node ok')"),
                           py("installed.append('rea')\nprint('rea ok')")], events=events)

    def die(messages):
        raise _Killed()
    replies = llm.replies
    llm.chat = lambda messages: _chat(llm, replies, messages, die)
    try:
        h.turn("get REA running on this computer")
    except _Killed:
        pass
    return h, llm


def _chat(llm, replies, messages, die):
    from jevh.llm import LLMResult
    llm.seen.append([dict(m) for m in messages])
    if not replies:
        die(messages)
    return LLMResult(text=replies.pop(0), input_tokens=10, output_tokens=5, model="scripted")


def test_a_turn_the_harness_never_finished_is_remembered_by_the_next_one(cfg):
    """Live 2026-10-12: a turn died with a dashboard restart at step 42 and was never remembered; the user's next
    message was answered as if it had never happened."""
    h, _ = _killed_mid_turn(cfg)
    h2, llm2 = harness(cfg, ["Node and REA are installed; next we pick a target."])
    h2.session, h2.session_dir = h.session, h.session_dir        # the restarted dashboard, same session
    h2.turn("so what did you get done?")
    turns = h2.memory.session_turns(h2.session, 10)
    assert [t.user for t in turns] == ["get REA running on this computer", "so what did you get done?"]
    assert "cut off at step" in turns[0].assistant and "rea ok" in turns[0].notes
    # what routing and the follow-up check read (the fake Jev here never calls a message a follow-up)
    assert "get REA running on this computer" in h2.memory.recent_text(h2.session, 2)
    h3, _ = harness(cfg, ["ok"])
    h3.session, h3.session_dir = h.session, h.session_dir
    h3.turn("thanks")
    assert len(h3.memory.session_turns(h3.session, 10)) == 3      # remembered once, not again


def test_continue_after_a_cut_off_turn_resumes_it_with_its_files_and_variables(cfg):
    h, _ = _killed_mid_turn(cfg)
    h2, llm2 = harness(cfg, [py("print(open('notes.txt').read(), installed)\nfinal('REA is running: ' + ', '.join(installed))")])
    h2.session, h2.session_dir = h.session, h.session_dir
    events = []
    h2.on_event = lambda k, d: events.append((k, d))
    r = h2.turn("ok continue")
    assert r.answer == "REA is running: node, rea"                  # the same task, its REPL variables back
    assert any(k == "resuming" for k, _ in events)
    assert "rea ok" in "\n".join(str(m["content"]) for m in llm2.seen[0])   # its own conversation, not a new turn
    # once it finished, "continue" is an ordinary message again
    h2_llm_replies = ["Nothing is left to continue."]
    llm2.replies = h2_llm_replies
    assert h2.turn("continue").answer == "Nothing is left to continue."


def test_continue_is_only_a_bare_request_to_carry_on():
    from jevh.orchestrator import CONTINUE_RE
    for yes in ["continue", "Continue.", "ok continue", "okay, keep going", "go ahead and finish it", "resume please",
                "carry on where you left off", "pick up where you left off"]:
        assert CONTINUE_RE.fullmatch(yes), yes
    for no in ["continue with the calculator instead", "can you continue the story about Rome?", "go on to the next file",
               "what happened?", "keep going but use npm"]:
        assert not CONTINUE_RE.fullmatch(no), no


def test_say_reaches_the_chat_mid_task_and_is_remembered_with_the_turn(cfg):
    """Live 2026-10-12: "are you running it on bilbo or on the hub?" sent mid-task was answered only in the model's
    thinking and its last reply; say() puts the answer in the chat while the task goes on."""
    events = []
    h, llm = harness(cfg, [py("say('On the hub (DESKTOP-AUIDSM2); Bilbo comes next.')\nprint('working')"),
                           py("final('all set')")], events=events)
    r = h.turn("set it up")
    told = [d for k, d in events if k == "say"]
    assert r.answer == "all set" and [d["text"] for d in told] == ["On the hub (DESKTOP-AUIDSM2); Bilbo comes next."]
    assert told[0]["persona"] == "daily_assistant"
    notes = h.memory.session_turns(h.session, 1)[0].notes
    assert "Told the user while working" in notes and "Bilbo comes next" in notes


def test_a_message_sent_mid_task_tells_the_worker_to_answer_it_with_say():
    from jevh.worker import STEER_NOTE
    assert "say(" in STEER_NOTE


def test_a_second_strike_does_not_stop_a_worker_whose_steps_print_new_things(cfg):
    """Long-horizon bench 2026-10: every run the progress check stopped was reading the task's rule pages, one a step
    (alike to Jev, but each output new); the second strike now stops only when outputs repeat."""
    import random
    random.seed(7)
    words = "rule order total region express bulk flag status cap discount tier customer audit memo".split()
    memos = [py(f"print('Memo R-{i:02d}: ' + {' '.join(random.choice(words) for _ in range(120))!r})") for i in range(8)]
    events = []
    h, llm = harness(cfg, memos + [py("final('policy.py written')")], circles=0.95, events=events, progress_every=2)
    r = h.turn("read the memos and write policy.py")
    assert r.answer == "policy.py written"
    checks = [d for k, d in events if k == "progress_check"]
    assert checks and all(c["stuck"] for c in checks) and any(c.get("held") for c in checks)
    assert max(c["strike"] for c in checks) == 1                       # never a second strike


def test_repeated_outputs_counts_near_repeats_only():
    from jevh.worker import repeated_outputs
    out = lambda t: [{"role": "assistant", "content": py("x")}, {"role": "user", "content": "Output:\n" + t}]   # noqa: E731
    earlier = out("{'status': 'done', 'changed': False, 'url': 'https://developers.facebook.com/apps/1/settings'}")
    recent = (out("{'status': 'done', 'changed': False, 'url': 'https://developers.facebook.com/apps/1/settings'}")
              + out("Memo R-14: the audit team samples about 676 orders a month and flags any bulk discount above cap")
              + out("{'status': 'done', 'changed': False, 'url': 'https://developers.facebook.com/apps/1/settings/'}"))
    assert repeated_outputs(earlier, recent) == 2
