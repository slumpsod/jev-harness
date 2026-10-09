"""The History tab: past sessions, each put by Jev in the category of the assistant whose field it is about
(jevh/history.py), and what the dashboard serves for it."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from jevh import history
from jevh.history import EMPTY, MIXED, Categorizer, History, SessionIndex, conversation, pretty, view
from jevh.jev import Jev
from jevh.memory import Turn
from jevh.orchestrator import Harness
from jevh.web.server import create_app

from conftest import FakeSystemOne, ScriptedLLM, choice_ans, py

TOKEN = "test-token"
BASE = "http://127.0.0.1:8740"

DEV, FIT, DAILY, GLACIER = "20261001-100000-aaaa", "20261002-100000-bbbb", "20261003-100000-cccc", "20261004-100000-dddd"
SESSIONS = {   # session id -> [(persona that answered, the user's request, the answer)]
    DEV: [("high_end_dev", "my flask app returns 500 when I delete a job, here is the traceback", "Fixed: the route lost its id."),
          ("high_end_dev", "now add pytest tests for it", "Added 4 tests.")],
    FIT: [("fitness_coach", "plan my workouts for next week, 4 days in the gym", "Mon push, Tue pull..."),
          ("fitness_coach", "swap the squats for leg press, my knees hurt", "Done.")],
    DAILY: [("daily_assistant", "whats on my calendar tomorrow", "Nothing before noon.")],
    GLACIER: [("daily_assistant", "tell me a bit about glaciers", "They are slow rivers of ice.")],
}
# what the fake Jev answers to "which assistant's field is this conversation about": a word in it -> (assistant, how sure)
FIELDS = {"flask": ("high_end_dev", 0.92), "workout": ("fitness_coach", 0.97), "calendar": ("daily_assistant", 0.9),
          "glaciers": ("daily_assistant", 0.3)}


def rules(k, q, s):
    if k == "persona":
        return choice_ans("daily_assistant", 0.9, q["criteria"])
    if k == "category":
        text = s["conversation"].lower()
        for word, (pick, conf) in FIELDS.items():
            if word in text:
                return choice_ans(pick, conf, q["criteria"])


class CountingJev(FakeSystemOne):
    def __init__(self, rules_fn=rules):
        super().__init__(rules_fn)
        self.down = False

    def category_calls(self):
        return [c for c in self.calls if "category" in c[1]]

    def ask(self, state, questions):
        if self.down and "category" in questions:
            raise ConnectionError("Jev is not answering")
        return super().ask(state, questions)


def populate(h, sessions=SESSIONS):
    """Remember these past sessions' turns, a minute apart in the order given (the first session is the oldest)."""
    ts = 1_790_000_000.0
    for session, turns in sessions.items():
        for persona, user, assistant in turns:
            ts += 60
            with h.memory._lock:
                h.memory._insert(Turn(persona, user, assistant, session, ts))
                h.memory.db.commit()


def make(cfg, sessions=SESSIONS, jev=None):
    jev = jev or CountingJev()
    h = Harness(cfg, jev=Jev(cfg, jev), llm_factory=lambda p: ScriptedLLM([]))
    h.learning = False
    populate(h, sessions)
    return h, jev


def settle(hist: History, timeout: float = 10.0) -> None:
    """Wait for the background thread to finish what it was given."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        t = hist.categorizer._thread
        if hist.categorizer.pending() == 0 and not (t and t.is_alive()):
            return
        time.sleep(0.02)
    raise AssertionError("Jev was still being asked")


def by_session(overview):
    return {s["session"]: s for s in overview["sessions"]}


# -- the pure parts -------------------------------------------------------------------------------------------

def test_the_conversation_jev_reads_is_numbered_cut_short_and_a_long_one_shows_its_ends():
    assert conversation(["first", "  second\n thing ", "", "third"]) == "1. first\n2. second thing\n3. third"
    assert conversation(["x" * 500], chars=20) == "1. " + "x" * 20
    long = conversation([f"request {i}" for i in range(1, 13)], limit=6)
    assert long.splitlines() == ["1. request 1", "2. request 2", "3. request 3", "4. request 4",
                                 "[... 6 more requests ...]", "11. request 11", "12. request 12"]
    assert conversation([]) == ""


def test_what_the_tab_shows_follows_the_saved_odds_and_the_current_settings():
    s = {"min_confidence": 0.4, "also_at": 0.25}
    row = {"category": "high_end_dev", "confidence": 0.7, "probs": {"high_end_dev": 0.7, "front_end_designer": 0.28, "daily_assistant": 0.02},
           "turns": 2, "roster": []}
    assert view(row, s) == ("high_end_dev", 0.7, ["front_end_designer"])           # a second tag from 0.25 up
    assert view(row, {**s, "also_at": 0.5})[2] == []                                # (a setting changed: no new question)
    assert view(row, {**s, "min_confidence": 0.8}) == (MIXED, 0.7, ["high_end_dev", "front_end_designer"])
    assert view(None, s) == (None, 0.0, [])                                         # not asked yet
    assert view({"category": "", "confidence": 0.0, "probs": {}, "turns": 0, "roster": []}, s) == (EMPTY, 0.0, [])


def test_a_category_is_named_for_people():
    assert pretty("high_end_dev") == "High end dev" and pretty("daily_assistant") == "Daily assistant"


def test_the_index_is_kept_between_runs_and_can_forget_one_session_or_all(tmp_path):
    ix = SessionIndex(tmp_path / "w" / "history.db")
    ix.put("s1", "fitness_coach", 0.9, {"fitness_coach": 0.9, "daily_assistant": 0.1}, 3, ["daily_assistant", "fitness_coach"])
    ix.put("s2", "daily_assistant", 0.6, {"daily_assistant": 0.6}, 1, ["daily_assistant"])
    ix.close()
    again = SessionIndex(tmp_path / "w" / "history.db").all()
    assert again["s1"]["probs"]["fitness_coach"] == 0.9 and again["s1"]["turns"] == 3 and again["s1"]["roster"] == ["daily_assistant", "fitness_coach"]
    ix = SessionIndex(tmp_path / "w" / "history.db")
    ix.forget("s1")
    assert set(ix.all()) == {"s2"}
    ix.forget()
    assert ix.all() == {}


# -- asking Jev -----------------------------------------------------------------------------------------------

def test_sessions_are_sorted_in_the_background_into_the_assistant_whose_field_they_are_about(cfg):
    h, jev = make(cfg)
    hist = History(lambda: h)
    first = hist.overview()
    assert first["pending"] == 4 and all(s["category"] is None for s in first["sessions"])   # asked for, not waited for
    settle(hist)
    o = hist.overview()
    got = by_session(o)
    assert [got[x]["category"] for x in (DEV, FIT, DAILY, GLACIER)] == ["high_end_dev", "fitness_coach", "daily_assistant", MIXED]
    assert o["pending"] == 0 and o["error"] == "" and o["stale"] == 0
    assert got[GLACIER]["also"] == ["daily_assistant"]                                # (the field it was likeliest to be)
    assert [c["name"] for c in o["categories"]] == ["daily_assistant", "fitness_coach", "high_end_dev", MIXED]   # latest first
    titles = {c["name"]: (c["title"], c["count"]) for c in o["categories"]}
    assert titles["high_end_dev"] == ("High end dev", 1) and titles[MIXED][0] == "No single field"
    assert "software engineer" in next(c for c in o["categories"] if c["name"] == "high_end_dev")["description"].lower()
    assert [s["session"] for s in o["sessions"]] == [GLACIER, DAILY, FIT, DEV]   # newest first


def test_each_session_is_asked_once_and_again_only_when_it_has_more_turns(cfg):
    h, jev = make(cfg)
    hist = History(lambda: h)
    hist.overview()
    settle(hist)
    assert len(jev.category_calls()) == 4
    hist.overview()
    settle(hist)
    assert len(jev.category_calls()) == 4                                            # nothing changed: Jev is not asked
    with h.memory._lock:
        h.memory._insert(Turn("high_end_dev", "and document it in the README", "Done.", DEV, 1_790_100_000.0))
        h.memory.db.commit()                                                         # the session you are in grew
    hist.overview()
    settle(hist)
    assert len(jev.category_calls()) == 5
    assert "document it" in jev.category_calls()[-1][0]["conversation"] and by_session(hist.overview())[DEV]["turns"] == 3


def test_jev_is_asked_about_the_whole_conversation_and_about_the_assistants_as_routing_knows_them(cfg):
    h, jev = make(cfg)
    hist = History(lambda: h)
    hist.overview()
    settle(hist)
    state, questions = next(c for c in jev.category_calls() if "flask" in c[0]["conversation"])
    assert state["conversation"] == ("1. my flask app returns 500 when I delete a job, here is the traceback\n"
                                     "2. now add pytest tests for it")
    q = questions["category"]
    assert "whole conversation" in q["instructions"] and set(q["criteria"]) == set(h.personas)
    assert q["criteria"]["fitness_coach"] == h.personas["fitness_coach"].description


def test_asking_jev_stays_out_of_a_running_turns_trace_and_is_metered_on_no_turn(cfg):
    h, jev = make(cfg)
    h.jev.trace = [{"step": "route"}]
    hist = History(lambda: h)
    hist.overview()
    settle(hist)
    assert h.jev.trace == [{"step": "route"}]                                        # the turn's own Jev calls only
    summary = h.usage.summary(session=h.session)
    assert summary["by_kind"]["jev_history"]["calls"] == 4 and "jev" not in summary["by_kind"]
    assert h.usage.turns(session=h.session)[0]["jev_calls"] == 4 and h.usage.turns(session=h.session)[0]["turn"] == 0


def test_a_jev_that_cannot_be_reached_is_a_message_and_is_not_asked_again_at_once(cfg):
    h, jev = make(cfg)
    jev.down = True
    hist = History(lambda: h)
    hist.overview()
    settle(hist)
    o = hist.overview()
    assert "Jev could not be asked" in o["error"] and "ConnectionError" in o["error"]
    assert o["pending"] == 0 and all(s["category"] is None for s in o["sessions"])
    jev.down = False
    hist.overview()
    settle(hist)
    assert hist.index.all() == {}                                                    # (it waits a little before trying again)
    assert hist.recategorize() == 4                                                  # unless asked to
    settle(hist)
    assert len(hist.index.all()) == 4 and hist.overview()["error"] == ""


def test_history_off_asks_nothing_and_lists_nothing(cfg):
    cfg.raw = {**cfg.raw, "history": {"enabled": False}}
    h, jev = make(cfg)
    o = History(lambda: h).overview()
    assert o == {"enabled": False, "categories": [], "sessions": [], "pending": 0, "error": "", "stale": 0} and jev.category_calls() == []


def test_a_session_whose_turns_all_failed_is_read_from_its_folder(cfg):
    h, jev = make(cfg, sessions={})
    h.session = "20261005-100000-eeee"
    h.session_dir = cfg.workspace / "sessions" / h.session
    h.turn("whats on my calendar tomorrow")                                          # leaves a task folder with its checkpoint
    h.memory.db.execute("DELETE FROM turns WHERE session=?", (h.session,))          # ...but no remembered turn: it failed
    h.memory.db.commit()
    hist = History(lambda: h)
    assert hist.overview()["sessions"][0]["turns"] == 0
    settle(hist)
    row = hist.overview()["sessions"][0]
    assert row["category"] == "daily_assistant" and "calendar" in jev.category_calls()[-1][0]["conversation"]


def test_a_session_with_no_request_in_it_is_listed_as_empty_and_not_asked_about(cfg):
    h, jev = make(cfg, sessions={})
    (cfg.workspace / "sessions" / "20261006-100000-ffff" / "u1t1-daily_assistant").mkdir(parents=True)
    hist = History(lambda: h)
    hist.overview()
    settle(hist)
    o = hist.overview()
    assert o["sessions"][0]["category"] == EMPTY and jev.category_calls() == []
    assert o["categories"][0]["title"] == "Empty sessions"


def test_sessions_sorted_before_the_assistants_changed_are_counted_as_stale(cfg):
    h, jev = make(cfg)
    hist = History(lambda: h)
    hist.overview()
    settle(hist)
    assert hist.overview()["stale"] == 0
    h.personas["steak_chef"] = h.personas["fitness_coach"]                           # a persona created since
    assert hist.overview()["stale"] == 4
    assert hist.recategorize() == 4
    settle(hist)
    assert hist.overview()["stale"] == 0


def test_one_session_can_be_asked_about_again(cfg):
    h, jev = make(cfg)
    hist = History(lambda: h)
    hist.overview()
    settle(hist)
    assert hist.recategorize(FIT) == 1
    settle(hist)
    assert len(jev.category_calls()) == 5


# -- reading one session ----------------------------------------------------------------------------------------

def test_a_session_is_read_with_its_conversation_files_failures_and_jevs_odds(cfg):
    h, jev = make(cfg)
    art = cfg.workspace / "sessions" / DEV / "artifacts"
    art.mkdir(parents=True)
    (art / "notes.txt").write_text("hello")
    hist = History(lambda: h)
    hist.overview()
    settle(hist)
    t = hist.transcript(DEV)
    assert [(x["persona"], x["user"][:12]) for x in t["turns"]] == [("high_end_dev", "my flask app"), ("high_end_dev", "now add pyte")]
    assert t["turns"][0]["assistant"].startswith("Fixed") and t["category"] == "high_end_dev" and t["confidence"] == 0.92
    assert t["probs"]["high_end_dev"] == 0.92 and set(t["probs"]) == set(h.personas)
    assert [(a["name"], a["size"]) for a in t["artifacts"]] == [("notes.txt", 5)] and t["current"] is False
    assert set(t) >= {"tasks", "failures", "usage", "also"}


@pytest.mark.parametrize("bad", ["../../etc/passwd", "nope", "20261001-100000-zzzz", "20269999-100000-aaaa", "", "..", "20261001-100000-aaaa/.."])
def test_only_a_session_this_workspace_has_can_be_read(cfg, bad):
    h, _ = make(cfg)
    assert History(lambda: h).transcript(bad) is None and not History(lambda: h).exists(bad)


# -- what the dashboard serves ------------------------------------------------------------------------------------

@pytest.fixture
def client(cfg):
    box = {}

    def factory(on_event):
        h, jev = make(cfg)
        box["jev"] = jev
        return h
    app = create_app(cfg, port=8740, harness_factory=factory, token=TOKEN)
    return TestClient(app, base_url=BASE), box


def auth():
    return {"X-Jevh-Token": TOKEN}


def sorted_overview(client):
    deadline = time.time() + 10
    while time.time() < deadline:
        o = client.get("/api/history", headers=auth()).json()
        if o["pending"] == 0:
            return o
        time.sleep(0.05)
    raise AssertionError("still sorting")


def test_the_history_endpoints_need_the_token_and_answer_with_the_sorted_sessions(client):
    c, box = client
    for path in ("/api/history", f"/api/history/{DEV}"):
        assert c.get(path).status_code == 403
    assert c.post("/api/history/recategorize", json={}).status_code == 403
    o = sorted_overview(c)
    assert {s["session"]: s["category"] for s in o["sessions"]}[FIT] == "fitness_coach"
    assert o["enabled"] is True and {c_["name"] for c_ in o["categories"]} >= {"high_end_dev", "fitness_coach"}
    one = c.get(f"/api/history/{FIT}", headers=auth()).json()
    assert one["category"] == "fitness_coach" and len(one["turns"]) == 2
    assert c.get("/api/history/nope", headers=auth()).status_code == 404
    assert c.get("/api/history/20261001-100000-zzzz", headers=auth()).status_code == 404


def test_sorting_again_asks_jev_again_for_one_session_or_all_and_refuses_an_unknown_one(client):
    c, box = client
    sorted_overview(c)
    n = len(box["jev"].category_calls())
    assert c.post("/api/history/recategorize", json={"session": DEV}, headers=auth()).json()["pending"] in (0, 1)
    sorted_overview(c)
    assert len(box["jev"].category_calls()) == n + 1
    assert c.post("/api/history/recategorize", json={"session": "nope"}, headers=auth()).status_code == 404
    assert c.post("/api/history/recategorize", json={}, headers=auth()).json()["pending"] in range(0, 5)
    sorted_overview(c)
    assert len(box["jev"].category_calls()) == n + 1 + 4


def test_a_flagged_session_says_so_and_the_one_you_are_in_is_marked_once_it_has_a_turn(client):
    c, box = client
    hub = c.app.state.hub
    assert not any(s["current"] for s in sorted_overview(c)["sessions"])            # (this session has no turn yet)
    hub.harness.review.flag(DEV, "the traceback was ignored")
    flagged = {s["session"]: s["flag"] for s in c.get("/api/history", headers=auth()).json()["sessions"]}
    assert flagged[DEV] and flagged[FIT] is None
    with hub.harness.memory._lock:
        hub.harness.memory._insert(Turn("daily_assistant", "whats on my calendar", "Nothing.", hub.harness.session, 1_790_200_000.0))
        hub.harness.memory.db.commit()
    now = sorted_overview(c)
    assert [s["session"] for s in now["sessions"] if s["current"]] == [hub.harness.session]


# -- resuming a session ---------------------------------------------------------------------------------------------

def test_a_harness_given_a_session_goes_on_numbering_its_turns_and_starts_on_the_assistant_that_answered_last(cfg):
    h, _ = make(cfg)
    assert h.turn_no == 0 and h.current is None                                      # a new session starts clean
    again = Harness(cfg, jev=Jev(cfg, CountingJev()), llm_factory=lambda p: ScriptedLLM([]), session=FIT)
    assert (again.turn_no, again.current, again.session) == (2, "fitness_coach", FIT)
    (cfg.workspace / "sessions" / FIT / "u5t1-fitness_coach").mkdir(parents=True)   # a turn that crashed: a folder, no memory
    assert Harness(cfg, jev=Jev(cfg, CountingJev()), llm_factory=lambda p: ScriptedLLM([]), session=FIT).turn_no == 5


@pytest.fixture
def resume_client(cfg):
    box = {}

    def factory(on_event, session=None):
        if session is None:
            h, box["jev"] = make(cfg)
            return h
        h = Harness(cfg, jev=Jev(cfg, box["jev"]), llm_factory=lambda p: ScriptedLLM([py("final('picked it up')")]),
                    on_event=on_event, session=session)
        h.learning = False
        return h
    app = create_app(cfg, port=8740, harness_factory=factory, token=TOKEN)
    return TestClient(app, base_url=BASE), app.state.hub


def test_resuming_a_session_shows_its_conversation_again_and_the_next_message_joins_it(cfg, resume_client):
    c, hub = resume_client
    assert c.post(f"/api/history/{FIT}/load").status_code == 403                     # needs the token
    assert c.post("/api/history/nope/load", headers=auth()).status_code == 404
    assert c.post("/api/history/20261001-100000-zzzz/load", headers=auth()).status_code == 404
    assert c.post(f"/api/history/{FIT}/load", headers=auth()).json() == {"session": FIT, "turns": 2}
    state = c.get("/api/state", headers=auth()).json()
    assert state["session"] == FIT and [(t["id"], t["status"], t["restored"]) for t in state["turns"]] == [(1, "done", True), (2, "done", True)]
    assert state["turns"][1]["message"].startswith("swap the squats") and state["turns"][0]["persona"] == "fitness_coach"
    kinds = [(e["kind"], e["turn"], e["data"].get("loaded")) for e in hub.events]    # what the chat is rebuilt from
    assert kinds[-3:] == [("reset", None, True), ("turn_done", 1, None), ("turn_done", 2, None)]
    assert c.post(f"/api/history/{FIT}/load", headers=auth()).json()["already"] is True
    assert any(s["current"] for s in c.get("/api/history", headers=auth()).json()["sessions"] if s["session"] == FIT)

    assert c.post("/api/chat", json={"message": "and one more leg day"}, headers=auth()).status_code == 200
    deadline = time.time() + 20
    while time.time() < deadline and not (len(hub.turns) == 3 and hub.turns[2]["status"] != "running"):
        time.sleep(0.05)
    new = hub.turns[2]
    assert new["id"] == 3 and new["status"] == "done" and new["answer"] == "picked it up"
    assert [t.user for t in hub.harness.memory.session_turns(FIT, 10)][-1] == "and one more leg day"   # the same conversation
    assert list((cfg.workspace / "sessions" / FIT).glob("u3t1-*"))                                      # in the same folder, not over turn 1's


def test_a_session_cannot_be_resumed_while_a_turn_runs(cfg, resume_client):
    c, hub = resume_client
    assert hub.busy.acquire(blocking=False)
    try:
        r = c.post(f"/api/history/{DEV}/load", headers=auth())
        assert r.status_code == 409 and hub.harness.session != DEV
    finally:
        hub.busy.release()
    assert c.post(f"/api/history/{DEV}/load", headers=auth()).status_code == 200
