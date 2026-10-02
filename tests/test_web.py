import time

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


def test_an_allowed_remote_host_needs_the_access_key(cfg):
    """`--allow-host` (a tailscale serve front): the page carries the API token, so a host other than this
    machine gets in only with the launch's access key (a link, then a cookie); other hosts stay refused."""
    host = "box.tail1234.ts.net"
    def factory(on_event):
        return Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: ScriptedLLM([]), on_event=on_event)

    app = create_app(cfg, port=8740, harness_factory=factory, token=TOKEN,
                     allow_hosts=[f"{host}:10000"], access_key="k3y")
    client = TestClient(app, base_url=BASE)
    for h in (host, f"{host}:10000"):
        assert client.get("/", headers={"Host": h}).status_code == 403            # no key: the token isn't handed out
        assert client.get("/?key=wrong", headers={"Host": h}).status_code == 403
    assert client.get("/?key=k3y", headers={"Host": "evil.example:10000"}).status_code == 403
    r = client.get("/?key=k3y", headers={"Host": f"{host}:10000"})
    assert r.status_code == 200 and TOKEN in r.text and "jevh_key=k3y" in r.headers["set-cookie"]
    assert "httponly" in r.headers["set-cookie"].lower() and "secure" in r.headers["set-cookie"].lower()
    r = client.get("/api/state", headers={"Host": f"{host}:10000", "Cookie": "jevh_key=k3y", **auth()})
    assert r.status_code == 200                                                    # the cookie carries the key
    assert client.get("/api/state", headers={"Host": f"{host}:10000", **auth()}).status_code == 403
    assert client.get("/").status_code == 200                                      # this machine needs no key


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


def test_created_personas_are_reviewed_through_the_api(cfg, tmp_path):
    import yaml
    learned = cfg.personas_dir / "learned"
    learned.mkdir()
    base = {"description": "Handles tax questions for people.", "system_prompt": "You help with tax.",
            "model": {"provider": "lmstudio", "model": "m"}, "draft": True}
    (learned / "tax_helper.yaml").write_text(yaml.safe_dump({"name": "tax_helper", **base}))
    (learned / "off.yaml").write_text(yaml.safe_dump({"name": "off", **base, "active": False, "source": "gate: no"}))
    client, app = make_client(cfg)
    h = app.state.hub.harness
    assert "tax_helper" in h.personas and "off" not in h.personas
    st = client.get("/api/state", headers=auth()).json()
    assert st["personas"]["tax_helper"]["draft"] is True
    m = client.get("/api/memory", headers=auth()).json()
    assert {(p["name"], p["active"], p["live"]) for p in m["persona_drafts"]} == {("tax_helper", True, True), ("off", False, False)}
    assert client.post("/api/personas/review", json={"name": "daily_assistant", "action": "delete"},
                       headers=auth()).status_code == 404            # a hand-written persona can not be touched
    assert client.post("/api/personas/review", json={"name": "off", "action": "activate"}, headers=auth()).status_code == 200
    assert "off" in h.personas and h.personas["off"].draft
    client.post("/api/personas/review", json={"name": "off", "action": "approve"}, headers=auth())
    assert h.personas["off"].draft is False
    assert [p["name"] for p in client.get("/api/memory", headers=auth()).json()["persona_drafts"]] == ["tax_helper"]
    client.post("/api/personas/review", json={"name": "tax_helper", "action": "delete"}, headers=auth())
    assert "tax_helper" not in h.personas and not (learned / "tax_helper.yaml").exists()
    assert client.post("/api/personas/review", json={"name": "off", "action": "bogus"}, headers=auth()).status_code in (400, 404)


def test_project_setting_validates_the_folder(cfg, tmp_path):
    client, app = make_client(cfg)
    assert client.post("/api/settings", json={"project": str(tmp_path / "nope")}, headers=auth()).status_code == 400
    r = client.post("/api/settings", json={"project": str(tmp_path)}, headers=auth()).json()
    assert r["project"] == str(tmp_path) and client.get("/api/state", headers=auth()).json()["project"] == str(tmp_path)


def test_stop_and_answer_through_the_api(cfg):
    replies = [py("a = ask_user('Which size?', ['S', 'L'])\nfinal('size ' + a)")]
    client, app = make_client(cfg, replies)
    client.post("/api/chat", json={"message": "order a mug"}, headers=auth())
    deadline = time.time() + 20
    while time.time() < deadline and not client.get("/api/state", headers=auth()).json()["questions"]:
        time.sleep(0.05)
    q = client.get("/api/state", headers=auth()).json()["questions"][0]
    assert q["question"] == "Which size?" and q["options"] == ["S", "L"]
    assert client.post("/api/answer", json={"id": "nope", "answer": "L"}, headers=auth()).status_code == 404
    assert client.post("/api/answer", json={"id": q["id"], "answer": "L"}, headers=auth()).json() == {"ok": True}
    assert wait_done(client)["answer"] == "size L"

    client2, _ = make_client(cfg, [py("import time\ntime.sleep(60)\nfinal('never')")])
    client2.post("/api/chat", json={"message": "long job"}, headers=auth())
    time.sleep(1.5)
    assert client2.post("/api/cancel", headers=auth()).json() == {"stopped": True}
    assert wait_done(client2)["status"] == "cancelled"
    assert client2.post("/api/cancel", headers=auth()).json() == {"stopped": False}


def test_usage_api(cfg):
    client, app = make_client(cfg)
    client.post("/api/chat", json={"message": "build a page"}, headers=auth())
    done = wait_done(client)
    assert done["usage"]["by_kind"]["llm"]["calls"] == 1
    u = client.get("/api/usage?range=session", headers=auth()).json()
    assert u["summary"]["turns"] == 1 and u["turns"][0]["turn"] == 1
    assert client.get("/api/usage?range=week", headers=auth()).json()["summary"]["total"]["calls"] >= 1


def test_resume_through_the_api(cfg):
    client, app = make_client(cfg, [py("open('draft.txt','w').write('half')\nprint('saved')"), py("print('more')"),
                                    py("print(open('draft.txt').read())\nfinal('finished')")])
    app.state.hub.harness.personas["daily_assistant"].max_steps = 2
    client.post("/api/chat", json={"message": "write the report"}, headers=auth())
    first = wait_done(client)
    assert first["checkpoint"]
    assert [x["path"] for x in client.get("/api/resumable", headers=auth()).json()] == [first["checkpoint"]]
    assert client.post("/api/resume", json={"path": "nope"}, headers=auth()).status_code == 404
    app.state.hub.harness.personas["daily_assistant"].max_steps = 5
    client.post("/api/resume", json={"path": first["checkpoint"]}, headers=auth())
    assert wait_done(client, 2)["answer"] == "finished"


def test_live_thinking_of_the_running_turn(cfg):
    from pathlib import Path
    client, app = make_client(cfg)
    hub = app.state.hub
    assert client.get("/api/thinking", headers=auth()).json() == {"file": None}      # nothing running
    step = Path(hub.harness.session_dir) / "u1t1-daily_assistant" / "thinking" / "step-03.txt"
    step.parent.mkdir(parents=True)
    step.write_text("=== thinking ===\n" + "x" * 5000 + "so the rule mirrors rows", encoding="utf-8")
    with hub.busy:
        r = client.get("/api/thinking", headers=auth()).json()
    assert r["file"] == "u1t1-daily_assistant/step-03.txt" and r["text"].endswith("mirrors rows")
    assert len(r["text"]) <= 4000 and r["bytes"] > 5000


def test_failures_api_lists_and_triages(cfg):
    client, app = make_client(cfg, [py("print(report_failure('comfy.run hung with no error'))"), py("final('ok')")])
    client.post("/api/chat", json={"message": "make an image"}, headers=auth())
    wait_done(client)
    got = client.get("/api/failures?range=session", headers=auth()).json()
    [f] = got["failures"]
    assert (f["source"], f["kind"], f["status"]) == ("model", "reported", "open") and "work_log" not in f
    assert got["counts"]["by_kind"] == {"reported": 1}
    assert client.post("/api/failures/status", json={"id": f["id"], "status": "bogus"}, headers=auth()).status_code == 400
    assert client.post("/api/failures/status", json={"id": 999, "status": "fixed"}, headers=auth()).status_code == 404
    assert client.post("/api/failures/status", json={"id": f["id"], "status": "fixed", "note": "timeout added"},
                       headers=auth()).json() == {"ok": True}
    assert client.get("/api/failures?status=fixed", headers=auth()).json()["failures"][0]["note"] == "timeout added"


def test_review_flag_notes_model_review_and_verdict(cfg):
    import json as _json
    client, app = make_client(cfg)
    client.post("/api/chat", json={"message": "build a page"}, headers=auth())
    wait_done(client)
    h = app.state.hub.harness
    reply = _json.dumps({"verdict": "failure", "confidence": 0.7, "summary": "the page is a stub",
                         "problems": [{"what": "only a heading", "turn": 1, "note": 1, "severity": "high"}]})
    h.review._factory = lambda block: ScriptedLLM([reply], name="judge")
    assert client.get("/api/reviews", headers=auth()).json()["current"] is None
    f = client.post("/api/reviews/flag", json={"note": "the page is empty", "turn": 1}, headers=auth()).json()["flag"]
    assert (f["session"], f["title"], f["notes"][0]["turn"]) == (h.session, "build a page", 1)
    assert client.post("/api/reviews/flag", json={"session": "nope"}, headers=auth()).status_code == 404
    assert client.post("/api/reviews/note", json={"id": f["id"], "text": " "}, headers=auth()).status_code == 400
    client.post("/api/reviews/note", json={"id": f["id"], "text": "no styling either"}, headers=auth())
    got = client.get("/api/reviews", headers=auth()).json()
    assert got["current"]["id"] == f["id"] and len(got["current"]["notes"]) == 2 and got["counts"] == {"open": 1}
    assert [s["flag"] for s in client.get("/api/sessions", headers=auth()).json()] == [f["id"]]
    detail = client.get(f"/api/reviews/{f['id']}", headers=auth()).json()
    assert detail["record"]["turns"][0]["assistant"] == "built it" and "save_artifact" in detail["record"]["tasks"][0]["work_log"]
    assert client.get("/api/reviews/999", headers=auth()).status_code == 404

    assert client.post("/api/reviews/run", json={"id": f["id"]}, headers=auth()).json() == {"running": True}
    deadline = time.time() + 10
    while time.time() < deadline and not any(e["kind"] == "review_done" for e in app.state.hub.events):
        time.sleep(0.05)
    out = client.get(f"/api/reviews/{f['id']}", headers=auth()).json()["flag"]
    assert (out["verdict"], out["verdict_by"], out["review"]["model"], out["running"]) == ("failure", "model", "judge", False)
    assert h.failures.get(out["failure_id"])["kind"] == "review_failure"

    assert client.post("/api/reviews/verdict", json={"id": f["id"], "verdict": "meh"}, headers=auth()).status_code == 400
    v = client.post("/api/reviews/verdict", json={"id": f["id"], "verdict": "not_failure", "reason": "fine"},
                    headers=auth()).json()["flag"]
    assert (v["verdict"], v["verdict_by"]) == ("not_failure", "user")
    assert h.failures.get(out["failure_id"])["status"] == "ignored"
    assert client.post("/api/reviews/status", json={"id": f["id"], "status": "decided"}, headers=auth()).status_code == 400
    assert client.post("/api/reviews/status", json={"id": f["id"], "status": "dismissed"}, headers=auth()).json()["flag"]["status"] == "dismissed"
    assert client.post("/api/reviews/status", json={"id": f["id"], "status": "open"}, headers=auth()).json()["flag"]["status"] == "decided"


def test_a_failed_model_review_is_reported_as_an_event(cfg):
    client, app = make_client(cfg)
    client.post("/api/chat", json={"message": "build a page"}, headers=auth())
    wait_done(client)
    app.state.hub.harness.review._factory = lambda block: ScriptedLLM(["can't tell"])
    fid = client.post("/api/reviews/flag", json={}, headers=auth()).json()["flag"]["id"]
    client.post("/api/reviews/run", json={"id": fid}, headers=auth())
    deadline = time.time() + 10
    while time.time() < deadline and not any(e["kind"] == "review_error" for e in app.state.hub.events):
        time.sleep(0.05)
    err = next(e for e in app.state.hub.events if e["kind"] == "review_error")
    assert err["data"]["flag"]["id"] == fid and "no JSON" in err["data"]["error"]
    assert client.get(f"/api/reviews/{fid}", headers=auth()).json()["flag"]["status"] == "open"


def test_notebook_api_adds_edits_filters_and_deletes(cfg):
    client, app = make_client(cfg)
    h = app.state.hub.harness
    assert client.get("/api/notebook", headers=auth()).json()["entries"] == []
    a = client.post("/api/notebook", json={"text": "show cost per turn", "kind": "idea", "session": True},
                    headers=auth()).json()["entry"]
    b = client.post("/api/notebook", json={"text": "the chart had no units", "kind": "problem"}, headers=auth()).json()["entry"]
    assert (a["session"], b["session"], a["title"]) == (h.session, "", "show cost per turn")
    assert client.post("/api/notebook", json={"text": "  "}, headers=auth()).status_code == 400
    assert client.post("/api/notebook", json={"text": "x", "kind": "rant"}, headers=auth()).status_code == 400
    got = client.get("/api/notebook", headers=auth()).json()
    assert [e["id"] for e in got["entries"]] == [b["id"], a["id"]]
    assert got["counts"] == {"open": 2, "done": 0, "all": 2, "idea": 1, "problem": 1, "note": 0}
    assert [e["id"] for e in client.get("/api/notebook?kind=idea", headers=auth()).json()["entries"]] == [a["id"]]
    assert [e["id"] for e in client.get("/api/notebook?q=units", headers=auth()).json()["entries"]] == [b["id"]]
    assert client.get("/api/notebook?status=weird", headers=auth()).status_code == 400
    up = client.post("/api/notebook/update", json={"id": a["id"], "pinned": True, "status": "done", "text": "cost per turn"},
                     headers=auth()).json()["entry"]
    assert (up["pinned"], up["status"], up["text"]) == (True, "done", "cost per turn")
    assert client.post("/api/notebook/update", json={"id": 999, "text": "x"}, headers=auth()).status_code == 404
    assert client.post("/api/notebook/update", json={"id": a["id"], "status": "?"}, headers=auth()).status_code == 400
    assert client.post("/api/notebook/delete", json={"id": b["id"]}, headers=auth()).json() == {"ok": True}
    assert client.post("/api/notebook/delete", json={"id": b["id"]}, headers=auth()).status_code == 404
    assert [e["id"] for e in h.notebook.list()] == [a["id"]]
    assert client.post("/api/notebook", json={"text": "x"}, headers=auth(Origin="https://evil.example")).status_code == 403
    assert client.get("/api/notebook").status_code == 403                              # needs the token like the rest
