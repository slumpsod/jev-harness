"""The local Jev servers in openjev/ (issue #6): the bridge's question -> row -> answer mapping (no GPU), and the
adapter forwarding to a bridge, dropping nothing on the way."""
import importlib.util
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    spec = importlib.util.spec_from_file_location(f"openjev_{name}", ROOT / "openjev" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod   # (pydantic resolves the request models' annotations through it)
    spec.loader.exec_module(mod)
    return mod


def test_the_bridge_maps_each_question_type_to_semif_rows_and_back():
    b = load("semif_bridge")
    Q = b.Question
    plan = b._plan({"yes": Q(type="noul", instructions="Is it urgent?"),
                    "who": Q(type="choice", instructions={"question": "Which team?", "ticket": 42},
                             criteria={"billing": "money", "tech": None}),
                    "how": Q(type="score", instructions="How bad?", criteria=["low", "mid", "high"])})
    assert [(k, kind, len(rows)) for k, kind, rows, _ in plan] == [("yes", "noul", 1), ("who", "choice", 1), ("how", "score", 1)]
    assert plan[1][2][0] == ("Which team?\n\nData:\nticket: 42", [("billing", "billing. money"), ("tech", "tech")])
    assert b._answer("noul", [[0.8, 0.2]], {})["noul"] == 0.8
    assert b._answer("choice", [[0.3, 0.7]], plan[1][3])["choice"] == "tech"
    assert b._answer("score", [[0.0, 0.5, 0.5]], plan[2][3])["score"] == 1.5


def test_a_wide_choice_is_read_in_16_way_pieces_and_normalized():
    b = load("semif_bridge")
    names = {f"o{i}": "" for i in range(20)}
    (_, _, rows, extra), = b._plan({"q": b.Question(type="choice", instructions="Pick", criteria=names)})
    assert [len(r[1]) for r in rows] == [16, 4]
    probs = [[0.0] * 15 + [0.5], [0.25, 0.25, 0.0, 0.0]]
    a = b._answer("choice", probs, extra)
    assert a["choice"] == "o15" and abs(sum(a["probabilities"].values()) - 1) < 1e-3


def test_the_bridge_refuses_bad_questions():
    b = load("semif_bridge")
    with pytest.raises(HTTPException):
        b._plan({"q": b.Question(type="choice", instructions="Pick", criteria={})})
    with pytest.raises(HTTPException):
        b._plan({"q": b.Question(type="vote", instructions="?")})


def fake_bridge():
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"status": "up", "model": "Qwen/Qwen3.5-4B"})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(req)
            self._send({"model": "Qwen/Qwen3.5-4B@rev", "usage": {"input_tokens": 12},
                        "answers": {k: {"type": "noul", "noul": 0.9} for k in req["questions"]}})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


def test_the_adapter_forwards_to_the_bridge(monkeypatch):
    from fastapi.testclient import TestClient
    srv, seen = fake_bridge()
    a = load("systemone")
    monkeypatch.setattr(a, "SEMIF_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    with TestClient(a.app) as c:
        r = c.post("/v1/systemone", json={"state": "a ticket", "model": "semif",
                                          "questions": {"urgent": {"type": "noul", "instructions": "Is it urgent?"}}})
        assert r.status_code == 200 and r.json()["answers"]["urgent"]["noul"] == 0.9
        assert r.json()["usage"] == {"input_tokens": 12, "output_tokens": 0}
        assert seen[0]["questions"]["urgent"] == {"type": "noul", "instructions": "Is it urgent?"}   # model not forwarded
        assert c.get("/health").json()["semif"]["status"] == "up"
        assert c.post("/v1/systemone", json={"state": "x", "model": "nope", "questions": {}}).status_code == 422
    srv.shutdown()


def test_the_adapter_sends_laya_requests_to_the_laya_bridge(monkeypatch):
    from fastapi.testclient import TestClient
    semif, seen_semif = fake_bridge()
    laya, seen_laya = fake_bridge()
    a = load("systemone")
    monkeypatch.setattr(a, "SEMIF_URL", f"http://127.0.0.1:{semif.server_address[1]}")
    monkeypatch.setattr(a, "LAYA_URL", f"http://127.0.0.1:{laya.server_address[1]}")
    with TestClient(a.app) as c:
        r = c.post("/v1/systemone", json={"state": "a ticket", "model": "laya",
                                          "questions": {"urgent": {"type": "noul", "instructions": "Is it urgent?"}}})
        assert r.status_code == 200 and r.json()["answers"]["urgent"]["noul"] == 0.9
        assert len(seen_laya) == 1 and not seen_semif
        h = c.get("/health").json()
        assert h["semif"]["status"] == "up" and h["laya"]["status"] == "up"
    semif.shutdown()
    laya.shutdown()


def test_the_adapter_says_when_the_bridge_is_down(monkeypatch):
    from fastapi.testclient import TestClient
    a = load("systemone")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setattr(a, "SEMIF_URL", f"http://127.0.0.1:{port}")
    with TestClient(a.app) as c:
        t = time.time()
        assert c.get("/health").json()["semif"]["status"].startswith("down") and time.time() - t < 5
        r = c.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}})
        assert r.status_code == 503 and "unreachable" in r.json()["detail"]


def test_the_laya_bridge_answers_in_the_harness_shape(monkeypatch):
    """laya_bridge.py, with a stand-in for the Laya agent (no weights): the state goes in as the checkpoint was trained
    on it, and each answer comes back in the shape decide.Answer reads."""
    from fastapi.testclient import TestClient
    b = load("laya_bridge")
    seen = []

    class Agent:
        def predict(self, state, questions):
            seen.append((state, questions))
            return {"answers": {"urgent": {"noul": 0.8, "confidence": 0.3},
                                "team": {"choice": "billing", "probabilities": {"billing": 0.7, "tech": 0.3}},
                                "level": {"score": 1.2, "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}}},
                    "usage": {"input_tokens": 40}}

    monkeypatch.setattr(b, "_load", lambda: setattr(b, "_agent", Agent()))
    with TestClient(b.app) as c:
        for _ in range(50):
            if c.get("/health").json()["status"] == "up":
                break
            time.sleep(0.05)
        r = c.post("/v1/systemone", json={
            "state": {"request": "refund me"}, "model": "laya",
            "questions": {"urgent": {"type": "noul", "instructions": {"question": "Urgent?", "tool": "x"}},
                          "team": {"type": "choice", "instructions": "Which team?",
                                   "criteria": {"billing": "money", "tech": None}},
                          "level": {"type": "score", "instructions": "How hard?", "criteria": ["a", "b", "c"]}}})
        assert r.status_code == 200
        a = r.json()["answers"]
        assert a["urgent"]["noul"] == 0.8 and a["urgent"]["probabilities"] == {"yes": 0.8, "no": 0.2}
        assert a["team"]["choice"] == "billing" and a["team"]["probabilities"]["tech"] == 0.3
        assert a["level"]["score"] == 1.2 and a["level"]["confidence"] == 0.6
        assert r.json()["usage"] == {"input_tokens": 40, "output_tokens": 0}
    state, questions = seen[0]
    assert state == json.dumps({"request": "refund me"}, indent=1)   # as the training data rendered it
    assert questions["team"]["criteria"] == {"billing": "money", "tech": None}   # an option without a description stays
