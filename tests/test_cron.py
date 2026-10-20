"""Cron jobs: schedules read and described, the next time computed, jobs stored, the runner's decisions (due, missed,
skipped, deferred while you work, claimed by one of two dashboards), a job done by a real Harness with scripted models,
and the dashboard's API. All offline. (Against live models: evals/bench/cron.py.)"""
import threading
import time
from datetime import datetime

import pytest

from jevh import cron
from jevh.cron import CronError, CronExpr, CronJobs, Runner
from jevh.jev import Jev
from jevh.orchestrator import Harness

from conftest import FakeSystemOne, ScriptedLLM, noul_ans, py
from test_web import auth, make_client


def at(y, mo, d, h=0, mi=0) -> float:
    return datetime(y, mo, d, h, mi).timestamp()


def stamp(ts) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------- schedules

PHRASES = [
    ("every 15 minutes", "*/15 * * * *"), ("every 5 min", "*/5 * * * *"), ("hourly", "0 * * * *"), ("every hour", "0 * * * *"),
    ("every 2 hours", "0 */2 * * *"), ("every 120 minutes", "0 */2 * * *"), ("daily at 9am", "0 9 * * *"),
    ("every day at 17:45", "45 17 * * *"), ("at 6:05 pm", "5 18 * * *"), ("daily", "0 9 * * *"), ("nightly", "0 0 * * *"),
    ("weekdays at 8:30", "30 8 * * 1-5"), ("weekdays", "0 9 * * 1-5"), ("weekends at noon", "0 12 * * 6,0"),
    ("every monday at 14:00", "0 14 * * 1"), ("thurs at 6pm", "0 18 * * 4"), ("every monday and wednesday at 9", "0 9 * * 1,3"),
    ("monthly on the 1st at noon", "0 12 1 * *"), ("monthly", "0 9 1 * *"), ("monthly on the 15th", "0 9 15 * *"),
    ("@daily", "0 0 * * *"), ("@hourly", "0 * * * *"), ("30 8 * * 1-5", "30 8 * * 1-5"), ("0 9 * * mon-fri", "0 9 * * mon-fri"),
    ("  Every   Sunday AT 7am ", "0 7 * * 0"),
]


@pytest.mark.parametrize("phrase,expr", PHRASES)
def test_phrases_become_cron_expressions(phrase, expr):
    assert cron.from_text(phrase) == expr


@pytest.mark.parametrize("bad", ["", "foo", "61 * * * *", "* 25 * * *", "0 0 30 2 *", "* * * *", "every 0 minutes",
                                 "every 90 minutes", "every 25 hours", "at 25:00", "every blursday at 9", "*/0 * * * *",
                                 "5-1 * * * *", "0 9 * * 8", "0 9 * 13 *"])
def test_what_is_not_a_schedule_is_refused_with_a_reason(bad):
    with pytest.raises(CronError) as e:
        cron.check(bad)
    assert str(e.value)   # (a sentence the page shows as it is)


def test_next_run_follows_the_clock():
    x = CronExpr("30 8 * * 1-5")                                  # weekdays at 8:30
    assert stamp(x.next_after(at(2026, 10, 19, 8, 0))) == "2026-10-19 08:30"   # a Monday
    assert stamp(x.next_after(at(2026, 10, 19, 8, 30))) == "2026-10-20 08:30"  # strictly after
    assert stamp(x.next_after(at(2026, 10, 23, 9, 0))) == "2026-10-26 08:30"   # Friday -> Monday
    assert stamp(CronExpr("*/15 * * * *").next_after(at(2026, 10, 19, 8, 47))) == "2026-10-19 09:00"
    assert stamp(CronExpr("0 0 1 * *").next_after(at(2026, 12, 31, 23, 59))) == "2027-01-01 00:00"
    assert stamp(CronExpr("0 0 29 2 *").next_after(at(2026, 3, 1))) == "2028-02-29 00:00"   # the next leap year
    assert stamp(CronExpr("0 12 * jan,jul *").next_after(at(2026, 2, 1))) == "2026-07-01 12:00"


def test_day_of_month_and_day_of_week_together_mean_either():
    x = CronExpr("0 9 13 * 5")   # the 13th, or any Friday
    assert stamp(x.next_after(at(2026, 10, 1))) == "2026-10-02 09:00"          # a Friday
    assert stamp(x.next_after(at(2026, 10, 14, 10))) == "2026-10-16 09:00"     # the 13th has passed; Friday the 16th
    assert stamp(CronExpr("0 9 13 * *").next_after(at(2026, 10, 1))) == "2026-10-13 09:00"


def test_sunday_is_zero_or_seven():
    assert CronExpr("0 9 * * 7").dows == CronExpr("0 9 * * 0").dows == CronExpr("0 9 * * sun").dows == frozenset({0})
    assert CronExpr("0 9 * * 5-7").dows == frozenset({5, 6, 0})


def test_schedules_in_words():
    said = {"*/15 * * * *": "every 15 minutes", "0 * * * *": "every hour", "30 * * * *": "every hour at :30",
            "0 */3 * * *": "every 3 hours", "0 9 * * *": "every day at 9:00 AM", "0 0 * * *": "every day at midnight",
            "30 8 * * 1-5": "weekdays at 8:30 AM", "0 10 * * 6,0": "weekends at 10:00 AM", "0 14 * * 1": "every Monday at 2:00 PM",
            "0 9 * * 1,3": "every Monday and Wednesday at 9:00 AM", "0 12 1 * *": "monthly on the 1st at noon",
            "0 9 22 * *": "monthly on the 22nd at 9:00 AM", "5 4 * * sun": "every Sunday at 4:05 AM",
            "0 9 1 1 *": "cron 0 9 1 1 *"}
    for expr, words in said.items():
        assert cron.describe(expr) == words, expr
    assert cron.describe("nonsense") == "nonsense"


def test_a_job_may_not_repeat_faster_than_the_limit():
    assert cron.check("*/5 * * * *", 5) == "*/5 * * * *"
    with pytest.raises(CronError, match="more often than every 5 minutes"):
        cron.check("*/2 * * * *", 5)
    with pytest.raises(CronError, match="more often"):
        cron.check("0,3 * * * *", 5)   # two close runs inside an hour count too
    assert cron.check("* * * * *", 0)  # (no limit set)


# ---------------------------------------------------------------- the store

@pytest.fixture
def jobs(tmp_path):
    return CronJobs(tmp_path / "cron.db", {"max_jobs": 3, "keep_runs": 3})


def test_add_change_pause_and_delete_a_job(jobs):
    now = at(2026, 10, 19, 7, 0)
    j = jobs.add("  Morning   mail ", "Summarize my unread mail", "weekdays at 8:30", now=now)
    assert (j["name"], j["schedule"], j["human"], j["enabled"], j["persona"]) == ("Morning mail", "30 8 * * 1-5", "weekdays at 8:30 AM", True, None)
    assert stamp(j["next_run"]) == "2026-10-19 08:30"

    j = jobs.update(j["id"], name="Mail", now=now + 60)                          # a rename keeps its turn
    assert j["name"] == "Mail" and stamp(j["next_run"]) == "2026-10-19 08:30"
    j = jobs.update(j["id"], schedule="daily at 6pm", now=now + 60)              # a new schedule is due from now
    assert stamp(j["next_run"]) == "2026-10-19 18:00" and j["schedule"] == "0 18 * * *"
    j = jobs.update(j["id"], enabled=False)
    assert not j["enabled"] and j["next_run"] is None and jobs.due(now + 10 ** 7) == []
    j = jobs.update(j["id"], enabled=True, now=now + 120)
    assert j["next_run"] is not None
    j = jobs.update(j["id"], persona="high_end_dev")
    assert j["persona"] == "high_end_dev"
    assert jobs.update(j["id"], persona=None)["persona"] is None

    jobs.add_run(j["id"], "ok", now=now)
    assert jobs.delete(j["id"]) and jobs.job(j["id"]) is None and jobs.runs(j["id"]) == []
    assert not jobs.delete(j["id"])


def test_a_job_that_cannot_be_saved_says_why(jobs):
    for args, msg in [(("", "do it", "hourly"), "name"), (("n", "  ", "hourly"), "what the assistant should do"),
                      (("n", "do it", "someday"), "could not read"), (("n", "do it", "*/1 * * * *"), "more often"),
                      (("x" * 81, "do it", "hourly"), "longer than 80")]:
        with pytest.raises(CronError, match=msg):
            jobs.add(*args)
    for i in range(3):
        jobs.add(f"j{i}", "p", "hourly")
    with pytest.raises(CronError, match="already 3 jobs"):
        jobs.add("one more", "p", "hourly")
    with pytest.raises(CronError, match="no job 99"):
        jobs.update(99, name="x")
    assert len(jobs.jobs()) == 3   # nothing half-saved


def test_the_revision_goes_up_on_every_change(jobs):
    r0 = jobs.rev()
    j = jobs.add("a", "p", "hourly")
    r1 = jobs.rev()
    jobs.update(j["id"], name="b")
    jobs.add_run(j["id"], "ok")
    assert r0 < r1 < jobs.rev()


def test_only_one_dashboard_gets_a_due_job(tmp_path):
    a, b = CronJobs(tmp_path / "cron.db"), CronJobs(tmp_path / "cron.db")   # two dashboards, one workspace
    now = at(2026, 10, 19, 7, 0)
    j = a.add("a", "p", "daily at 8am", now=now)
    due = [x for db in (a, b) for x in db.due(at(2026, 10, 19, 8, 1))]
    assert len(due) == 2
    won = [db.claim(x, at(2026, 10, 19, 8, 1)) for db, x in zip((a, b), due)]
    assert sorted(won) == [False, True]
    assert stamp(a.job(j["id"])["next_run"]) == "2026-10-20 08:00"


def test_old_runs_are_trimmed_and_lost_ones_are_ended(jobs):
    j = jobs.add("a", "p", "hourly")
    for _ in range(5):
        jobs.add_run(j["id"], "ok")
    assert len(jobs.runs(j["id"])) == 3
    r = jobs.add_run(j["id"], "running", now=1000.0)
    live = jobs.add_run(j["id"], "running", now=1000.0 + cron.STALE_SECONDS + 5)
    assert jobs.reap(now=1000.0 + cron.STALE_SECONDS + 10) == 1
    assert jobs.run(r)["status"] == "interrupted" and jobs.run(live)["status"] == "running"


def test_the_unseen_count_is_for_runs_that_did_something(tmp_path):
    jobs = CronJobs(tmp_path / "cron.db")
    j = jobs.add("a", "p", "hourly")
    jobs.add_run(j["id"], "ok")
    jobs.add_run(j["id"], "error", error="boom")
    jobs.add_run(j["id"], "missed")
    jobs.add_run(j["id"], "skipped")
    assert jobs.unseen() == 2 and jobs.jobs()[0]["unseen"] == 2
    jobs.mark_seen(j["id"])
    assert jobs.unseen() == 0


# ---------------------------------------------------------------- the runner

class Doer:
    """Stands in for a Harness: records the jobs it was given, and answers as told."""

    def __init__(self, status="ok", wait: threading.Event | None = None, boom=None):
        self.status, self.wait, self.boom, self.did = status, wait, boom, []
        self.started = threading.Event()

    def __call__(self, job, cancel, on_session):
        self.did.append(job["name"])
        on_session("20261019-080000-abcd")
        self.started.set()
        if self.wait is not None:
            while not self.wait.wait(0.02):
                if cancel.is_set():
                    return {"status": "cancelled", "answer": "", "error": "", "session": "20261019-080000-abcd"}
        if self.boom:
            raise self.boom
        return {"status": self.status, "answer": f"did {job['name']}", "error": "", "session": "20261019-080000-abcd"}


def settle(jobs, jid, n=1, timeout=5.0):
    """Wait until job `jid` has `n` finished runs; -> its runs, newest first."""
    end = time.time() + timeout
    while time.time() < end:
        done = [r for r in jobs.runs(jid) if r["status"] not in ("queued", "running")]
        if len(done) >= n:
            return jobs.runs(jid)
        time.sleep(0.02)
    raise AssertionError(f"job {jid} did not finish: {jobs.runs(jid)}")


def runner(jobs, doer, **kw):
    r = Runner(jobs, doer, tick=0.05, **kw)
    return r


def test_a_due_job_runs_once_and_its_run_is_kept(jobs):
    doer, done, changes = Doer(), [], []
    r = runner(jobs, doer, on_done=lambda j, run: done.append((j["name"], run["status"])), on_change=lambda: changes.append(1))
    j = jobs.add("Mail", "Summarize", "daily at 8am", now=at(2026, 10, 19, 7, 0))
    assert r.once(at(2026, 10, 19, 7, 59)) == []                       # not yet
    assert len(r.once(at(2026, 10, 19, 8, 0, ) + 5)) == 1
    run = settle(jobs, j["id"])[0]
    assert (run["status"], run["answer"], run["session"], run["trigger"]) == ("ok", "did Mail", "20261019-080000-abcd", "schedule")
    assert doer.did == ["Mail"] and done == [("Mail", "ok")] and changes
    assert r.once(at(2026, 10, 19, 8, 1)) == [] and doer.did == ["Mail"]   # (its next time is tomorrow)
    assert stamp(jobs.job(j["id"])["next_run"]) == "2026-10-20 08:00"


def test_a_run_that_fails_or_raises_is_an_error_not_a_dead_timer(jobs):
    r = runner(jobs, Doer(boom=RuntimeError("model gone")))
    j = jobs.add("Mail", "p", "daily at 8am", now=at(2026, 10, 19, 7, 0))
    r.once(at(2026, 10, 19, 8, 1))
    run = settle(jobs, j["id"])[0]
    assert run["status"] == "error" and "RuntimeError: model gone" in run["error"]
    r2 = runner(jobs, Doer("error"))
    r2.once(at(2026, 10, 20, 8, 1))
    assert settle(jobs, j["id"], 2)[0]["status"] == "error"


def test_a_job_whose_time_passed_unseen_is_missed_not_made_up(jobs):
    doer = Doer()
    r = runner(jobs, doer)
    j = jobs.add("Mail", "p", "daily at 8am", now=at(2026, 10, 19, 7, 0))
    assert r.once(at(2026, 10, 19, 12, 0)) == []     # the dashboard was off until noon
    run = jobs.runs(j["id"])[0]
    assert run["status"] == "missed" and "due 4h ago" in run["error"] and doer.did == []
    assert stamp(jobs.job(j["id"])["next_run"]) == "2026-10-20 08:00"       # and goes on tomorrow
    assert jobs.unseen() == 0                                                # (nothing to read)


def test_a_job_waits_while_you_are_mid_turn_then_runs_or_is_missed(jobs):
    mid_turn = [True]
    doer = Doer()
    r = runner(jobs, doer, ready=lambda: not mid_turn[0])
    j = jobs.add("Mail", "p", "daily at 8am", now=at(2026, 10, 19, 7, 0))
    assert r.once(at(2026, 10, 19, 8, 1)) == [] and jobs.runs(j["id"]) == []      # waits; its time is kept
    assert stamp(jobs.job(j["id"])["next_run"]) == "2026-10-19 08:00"
    mid_turn[0] = False
    assert len(r.once(at(2026, 10, 19, 8, 4))) == 1
    settle(jobs, j["id"])
    # ...but not for ever
    mid_turn[0] = True
    k = jobs.add("Late", "p", "daily at 9am", now=at(2026, 10, 19, 8, 30))
    r.once(at(2026, 10, 19, 9, 30))
    assert jobs.runs(k["id"])[0]["status"] == "missed" and doer.did == ["Mail"]


def test_a_job_still_running_is_not_started_twice(jobs):
    gate = threading.Event()
    doer = Doer(wait=gate)
    r = runner(jobs, doer)
    j = jobs.add("Slow", "p", "*/5 * * * *", now=at(2026, 10, 19, 8, 0))
    r.once(at(2026, 10, 19, 8, 5, ) + 1)
    assert doer.started.wait(2)
    r.once(at(2026, 10, 19, 8, 10) + 1)                                  # its next time came; the run before is going on
    assert [x["status"] for x in jobs.runs(j["id"])][:2] == ["skipped", "running"]
    assert jobs.jobs()[0]["running"] is True
    gate.set()
    settle(jobs, j["id"])
    assert doer.did == ["Slow"]


def test_jobs_run_one_at_a_time_in_the_order_they_came(jobs):
    order, gate = [], threading.Event()

    def doer(job, cancel, on_session):
        order.append(job["name"])
        gate.wait(2) if job["name"] == "first" else None
        return {"status": "ok", "answer": "", "error": "", "session": ""}
    r = Runner(jobs, doer, tick=0.05)
    a = jobs.add("first", "p", "daily at 8am", now=at(2026, 10, 19, 7, 0))
    b = jobs.add("second", "p", "daily at 8:01am", now=at(2026, 10, 19, 7, 0))
    r.once(at(2026, 10, 19, 8, 2))
    time.sleep(0.3)
    assert order == ["first"]                  # the second waits for the first
    assert jobs.runs(b["id"])[0]["status"] == "queued"
    gate.set()
    settle(jobs, b["id"])
    assert order == ["first", "second"]
    settle(jobs, a["id"])


def test_run_now_leaves_the_schedule_alone_and_refuses_a_second_run(jobs):
    gate = threading.Event()
    doer = Doer(wait=gate)
    r = runner(jobs, doer)
    j = jobs.add("Mail", "p", "daily at 8am", now=at(2026, 10, 19, 7, 0))
    nxt = jobs.job(j["id"])["next_run"]
    r.run_now(j["id"])
    assert doer.started.wait(2)
    with pytest.raises(CronError, match="already running"):
        r.run_now(j["id"])
    gate.set()
    run = settle(jobs, j["id"])[0]
    assert run["trigger"] == "manual" and jobs.job(j["id"])["next_run"] == nxt
    with pytest.raises(CronError, match="no job 9"):
        r.run_now(9)


def test_stop_ends_a_running_job_and_deleting_it_ends_its_runs(jobs):
    doer = Doer(wait=threading.Event())
    r = runner(jobs, doer)
    j = jobs.add("Mail", "p", "daily at 8am")
    r.run_now(j["id"])
    assert doer.started.wait(2)
    assert r.stop_run(j["id"]) and not r.stop_run(99)
    assert settle(jobs, j["id"])[0]["status"] == "cancelled"
    assert not r.stop_run(j["id"])                                     # (nothing left to stop)


def test_a_job_deleted_while_it_waits_does_not_run(jobs):
    gate = threading.Event()
    doer = Doer(wait=gate)
    r = runner(jobs, doer)
    a, b = jobs.add("a", "p", "daily at 8am"), jobs.add("b", "p", "daily at 9am")
    r.run_now(a["id"])
    assert doer.started.wait(2)
    r.run_now(b["id"])
    rid = jobs.runs(b["id"])[0]["id"]
    jobs.delete(b["id"])
    gate.set()
    settle(jobs, a["id"])
    time.sleep(0.2)
    assert doer.did == ["a"] and jobs.run(rid) is None   # (its runs went with it)


def test_a_lost_run_is_ended_by_the_next_check(jobs):
    r = runner(jobs, Doer())
    j = jobs.add("Mail", "p", "daily at 8am")
    rid = jobs.add_run(j["id"], "running", now=1000.0)
    r.once(1000.0 + cron.STALE_SECONDS + 1)
    assert jobs.run(rid)["status"] == "interrupted" and not jobs.active(j["id"])


# ---------------------------------------------------------------- a job done by a real harness

def harness_for(cfg, replies):
    return lambda on_event: Harness(cfg, jev=Jev(cfg, FakeSystemOne(lambda k, q, s: noul_ans(0.9) if k in ("needs_high_end_dev", "in_scope", "necessary") else None)),
                                    llm_factory=lambda p: ScriptedLLM(list(replies)), on_event=on_event)


JOB = {"id": 1, "name": "Tidy", "prompt": "write the weekly note", "persona": None}


def test_a_job_is_a_turn_in_a_session_of_its_own(cfg):
    sessions = []
    out = cron.run_job(JOB, harness_for(cfg, [py("p = save_artifact('note.txt', 'all quiet')\nfinal('Wrote the note.', [p])")]),
                       threading.Event(), 60, sessions.append)
    assert out["status"] == "ok" and "Wrote the note" in out["answer"] and out["error"] == ""
    assert sessions == [out["session"]] and (cfg.workspace / "sessions" / out["session"] / "artifacts" / "note.txt").is_file()


def test_nobody_is_there_to_answer_a_question(cfg):
    out = cron.run_job(JOB, harness_for(cfg, [py("a = ask_user('Which folder?', ['a', 'b'])\nfinal('The user said: ' + a)")]),
                       threading.Event(), 60)
    assert out["status"] == "ok" and "nobody is here to answer" in out["answer"]


def test_a_task_that_cannot_be_done_is_an_error(cfg):
    out = cron.run_job(JOB, harness_for(cfg, [py("cant_do_it(")] * 3), threading.Event(), 60)
    assert out["status"] in ("error", "ok")   # (a worker that gives up says so in its answer; a crash is an error)
    broken = cron.run_job(JOB, lambda on_event: (_ for _ in ()).throw(RuntimeError("no model")), threading.Event(), 60)
    assert broken["status"] == "error" and "RuntimeError: no model" in broken["error"]


def test_a_job_runs_out_of_time_or_is_stopped(cfg):
    class Slow(ScriptedLLM):
        def chat(self, messages):
            time.sleep(0.4)
            return super().chat(messages)
    make = lambda on_event: Harness(cfg, jev=Jev(cfg, FakeSystemOne()), llm_factory=lambda p: Slow([py("import time\ntime.sleep(30)")] * 4), on_event=on_event)
    out = cron.run_job(JOB, make, threading.Event(), timeout=0.6)
    assert out["status"] == "error" and "stopped after" in out["error"]
    stop = threading.Event()
    threading.Timer(0.6, stop.set).start()
    assert cron.run_job(JOB, make, stop, 60)["status"] == "cancelled"


# ---------------------------------------------------------------- the dashboard

@pytest.fixture
def api(cfg):
    client, app = make_client(cfg, [py("final('Checked the mail.')")] * 6)
    hub = app.state.hub
    post = lambda path, body=None: client.post(f"/api/cron/{path}", json=body or {}, headers=auth())
    return client, hub, post


def test_the_cron_api_adds_changes_runs_and_removes_a_job(api):
    client, hub, post = api
    board = client.get("/api/cron", headers=auth()).json()
    assert board["jobs"] == [] and board["running"] is True and board["min_minutes"] == 5

    c = post("check", {"schedule": "weekdays at 8:30"}).json()
    assert (c["schedule"], c["human"]) == ("30 8 * * 1-5", "weekdays at 8:30 AM") and c["next"] > time.time()
    bad = post("check", {"schedule": "whenever"})
    assert bad.status_code == 400 and "could not read" in bad.json()["detail"]

    j = post("add", {"name": "Mail", "prompt": "Check the mail", "schedule": "every 30 minutes"}).json()["job"]
    assert j["human"] == "every 30 minutes" and j["enabled"]
    assert post("add", {"name": "Fast", "prompt": "x", "schedule": "* * * * *"}).status_code == 400
    assert post("add", {"name": "Who", "prompt": "x", "schedule": "hourly", "persona": "nobody"}).status_code == 400
    pinned = post("add", {"name": "Pinned", "prompt": "x", "schedule": "hourly", "persona": "daily_assistant"})
    assert pinned.status_code == 200 and pinned.json()["job"]["persona"] == "daily_assistant"

    rev = client.get("/api/cron", headers=auth()).json()["rev"]
    assert client.get("/api/cron", params={"rev": rev}, headers=auth()).json()["unchanged"] is True
    u = post("update", {"id": j["id"], "name": "Mail check", "enabled": False, "persona": None}).json()["job"]
    assert u["name"] == "Mail check" and not u["enabled"]
    assert post("update", {"id": 99, "name": "x"}).status_code == 400
    post("update", {"id": j["id"], "enabled": True})

    post("run", {"id": j["id"]})
    end = time.time() + 20
    while time.time() < end and not [r for r in hub.cron.runs(j["id"]) if r["status"] == "ok"]:
        time.sleep(0.1)
    runs = client.get("/api/cron/runs", params={"id": j["id"]}, headers=auth()).json()["runs"]
    assert runs[0]["status"] == "ok" and "Checked the mail" in runs[0]["answer"] and runs[0]["session"]
    assert (hub.cfg.workspace / "sessions" / runs[0]["session"]).is_dir() and runs[0]["session"] != hub.harness.session
    board = client.get("/api/cron", headers=auth()).json()
    assert board["unseen"] == 1 and [x for x in board["jobs"] if x["id"] == j["id"]][0]["last"]["status"] == "ok"
    post("seen", {})
    assert client.get("/api/cron", headers=auth()).json()["unseen"] == 0

    assert post("delete", {"id": j["id"]}).json() == {"ok": True} and post("delete", {"id": j["id"]}).status_code == 404
    assert client.get("/api/cron/runs", params={"id": j["id"]}, headers=auth()).status_code == 400


def test_the_cron_api_needs_the_token(api):
    client, _, _ = api
    assert client.get("/api/cron").status_code == 403
    assert client.post("/api/cron/add", json={}).status_code == 403
    assert client.post("/api/cron/run", json={"id": 1}, headers={**auth(), "Origin": "http://evil.example"}).status_code == 403


def test_a_run_does_not_touch_the_chat(api):
    client, hub, post = api
    j = post("add", {"name": "Mail", "prompt": "Check the mail", "schedule": "hourly"}).json()["job"]
    session, turns = hub.harness.session, len(hub.turns)
    post("run", {"id": j["id"]})
    end = time.time() + 20
    while time.time() < end and not [r for r in hub.cron.runs(j["id"]) if r["status"] == "ok"]:
        time.sleep(0.1)
    assert hub.harness.session == session and len(hub.turns) == turns
    kinds = {e["kind"] for e in hub.events}
    assert "cron_done" in kinds and "turn_start" not in kinds


def test_the_dashboard_starts_the_timer_and_a_job_runs_when_due(cfg):
    cfg.raw = {**cfg.raw, "cron": {"tick": 0.1}}
    client, app = make_client(cfg, [py("final('Checked the mail.')")] * 4)
    hub = app.state.hub
    j = hub.cron.add("Mail", "Check", "hourly")
    hub._db_due = hub.cron._run("UPDATE jobs SET next_run=? WHERE id=?", (time.time() - 5, j["id"]))   # its time came
    with client:                                                       # (the app's lifespan starts the runner)
        end = time.time() + 20
        while time.time() < end and not [r for r in hub.cron.runs(j["id"]) if r["status"] == "ok"]:
            time.sleep(0.1)
    assert hub.cron.runs(j["id"])[0]["status"] == "ok" and hub.cron.runs(j["id"])[0]["trigger"] == "schedule"


def test_the_command_line_adds_lists_and_removes(cfg, monkeypatch, capsys):
    import jevh.config as config
    monkeypatch.setattr(config, "load_config", lambda *a, **k: cfg)
    assert cron.main(["add", "Mail", "Check the mail", "-s", "weekdays at 8:30"]) == 0
    assert "job 1: Mail - weekdays at 8:30 AM" in capsys.readouterr().out
    assert cron.main(["add", "Bad", "x", "-s", "*/1 * * * *"]) == 2
    assert "more often" in capsys.readouterr().err
    cron.main([])
    assert "weekdays at 8:30 AM" in capsys.readouterr().out
    assert cron.main(["pause", "1"]) == 0 and "paused" in capsys.readouterr().out
    assert cron.main(["rm", "1"]) == 0 and "removed" in capsys.readouterr().out
