import time

import pytest
from fastapi.testclient import TestClient

from jevh.bus import Task
from jevh.jev import Jev
from jevh.orchestrator import Harness
from jevh.web.server import create_app, jsonable

from conftest import FakeSystemOne, ScriptedLLM, choice_ans, noul_ans, py

TOKEN = "test-token"
BASE = "http://127.0.0.1:8740"


def rules(k, q, s):
    if k == "persona":
        return choice_ans("daily_assistant", 0.9, q["criteria"])
    if k in ("needs_high_end_dev", "in_scope", "necessary"):
        return noul_ans(0.9)


def make_client(cfg, replies=None, delay=0.0):
    replies = replies or [py("p = save_artifact('page.html', '<h1>hi</h1>')\nfinal('built it', [p])")]

    class Slow(ScriptedLLM):
        def chat(self, messages):
            time.sleep(delay)
            return super().chat(messages)

    def factory(on_event):
        return Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: Slow(list(replies)),
                       on_event=on_event)

    app = create_app(cfg, port=8740, harness_factory=factory, token=TOKEN)
    return TestClient(app, base_url=BASE), app


def auth(**extra):
    return {"X-Jevh-Token": TOKEN, **extra}


def wait_done(client, n=1, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        turns = client.get("/api/state", headers=auth()).json()["turns"]
        if len(turns) >= n and turns[n - 1]["status"] != "running":
            return turns[n - 1]
        time.sleep(0.05)
    raise AssertionError("turn did not finish")


def test_index_embeds_token_and_rejects_foreign_host(cfg):
    client, _ = make_client(cfg)
    r = client.get("/")
    assert r.status_code == 200 and TOKEN in r.text
    assert client.get("/", headers={"Host": "evil.example:8740"}).status_code == 403  # DNS rebinding


def test_api_requires_token(cfg):
    client, _ = make_client(cfg)
    assert client.get("/api/state").status_code == 403
    assert client.get("/api/state", headers={"X-Jevh-Token": "wrong"}).status_code == 403
    assert client.get("/api/state", headers=auth()).status_code == 200
    assert client.get(f"/api/state?token={TOKEN}").status_code == 200  # EventSource/iframe form


def test_foreign_origin_post_is_refused(cfg):
    client, _ = make_client(cfg)
    r = client.post("/api/chat", json={"message": "hi"}, headers=auth(Origin="https://evil.example"))
    assert r.status_code == 403


def test_chat_runs_a_turn_and_streams_events(cfg):
    client, app = make_client(cfg)
    r = client.post("/api/chat", json={"message": "build a page"}, headers=auth(Origin=BASE))
    assert r.status_code == 200
    turn = wait_done(client)
    assert turn["status"] == "done" and turn["answer"] == "built it" and turn["persona"] == "daily_assistant"
    kinds = [e["kind"] for e in app.state.hub.events]
    for k in ("turn_start", "route", "tier", "start", "code", "output", "done", "turn_done"):
        assert k in kinds, k
    code = next(e for e in app.state.hub.events if e["kind"] == "code")
    assert code["data"]["task_id"] == 1 and code["turn"] == 1


def test_second_turn_while_busy_is_409(cfg):
    client, _ = make_client(cfg, delay=0.5)
    assert client.post("/api/chat", json={"message": "one"}, headers=auth()).status_code == 200
    assert client.post("/api/chat", json={"message": "two"}, headers=auth()).status_code == 409
    wait_done(client)


def test_artifacts_listed_and_served_sandboxed(cfg):
    client, _ = make_client(cfg)
    client.post("/api/chat", json={"message": "build"}, headers=auth())
    wait_done(client)
    arts = client.get("/api/artifacts", headers=auth()).json()
    assert [a["name"] for a in arts] == ["page.html"]
    r = client.get("/api/file", params={"path": arts[0]["path"], "token": TOKEN})
    assert r.status_code == 200 and "<h1>hi</h1>" in r.text
    assert "sandbox" in r.headers["content-security-policy"]


def test_file_endpoint_blocks_paths_outside_workspace(cfg, tmp_path):
    client, _ = make_client(cfg)
    secret = tmp_path / "secret.txt"
    secret.write_text("nope")
    assert client.get("/api/file", params={"path": str(secret), "token": TOKEN}).status_code == 404
    traversal = str(cfg.workspace / ".." / "secret.txt")
    assert client.get("/api/file", params={"path": traversal, "token": TOKEN}).status_code == 404


def test_settings_validate(cfg):
    client, app = make_client(cfg)
    assert client.post("/api/settings", json={"tier_mode": "fast"}, headers=auth()).json()["tier_mode"] == "fast"
    assert app.state.hub.harness.tier_mode == "fast"
    assert client.post("/api/settings", json={"tier_mode": "turbo"}, headers=auth()).status_code == 400
    assert client.post("/api/settings", json={"pin": "fitness_coach"}, headers=auth()).json()["pin"] == "fitness_coach"
    assert client.post("/api/settings", json={"pin": "nobody"}, headers=auth()).status_code == 400


def test_jsonable_handles_nested_tasks():
    root = Task(1, None, "user", "a", "root task")
    child = Task(2, root, "a", "b", "child task")
    out = jsonable({"task": child, "tasks": [root, child]})
    assert out["task"]["parent"] == 1 and out["task"]["depth"] == 1
    assert out["tasks"][0]["parent"] is None


def test_memory_tab_lists_facts_and_reviews_learned_skills(cfg, tmp_path):
    from jevh.config import ROOT
    cfg.skills_dirs = [tmp_path / "skills", ROOT / "skills"]
    d = tmp_path / "skills" / "learned" / "learned-x"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: learned-x\ndraft: true\ndescription: does x\n---\nsteps", encoding="utf-8")
    client, app = make_client(cfg)
    h = app.state.hub.harness
    fid = h.memory.add_fact("Jed's gym is closed on Sundays.")
    m = client.get("/api/memory", headers=auth()).json()
    assert [f["text"] for f in m["facts"]] == ["Jed's gym is closed on Sundays."]
    assert m["learned"] == [{"name": "learned-x", "description": "does x", "path": str(d / "SKILL.md"), "draft": True}]
    client.post("/api/memory/forget", json={"id": fid}, headers=auth())
    assert client.get("/api/memory", headers=auth()).json()["facts"] == []
    client.post("/api/skills/review", json={"name": "learned-x", "action": "approve"}, headers=auth())
    assert h.roster["learned-x"].draft is False
    assert client.post("/api/skills/review", json={"name": "frontend-design", "action": "delete"},
                       headers=auth()).status_code == 404  # only learned skills can be touched
    client.post("/api/skills/review", json={"name": "learned-x", "action": "delete"}, headers=auth())
    assert not d.exists() and "learned-x" not in h.roster


def test_project_setting_validates_the_folder(cfg, tmp_path):
    client, app = make_client(cfg)
    assert client.post("/api/settings", json={"project": str(tmp_path / "nope")}, headers=auth()).status_code == 400
    r = client.post("/api/settings", json={"project": str(tmp_path)}, headers=auth()).json()
    assert r["project"] == str(tmp_path) and client.get("/api/state", headers=auth()).json()["project"] == str(tmp_path)
