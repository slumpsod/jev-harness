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


def test_the_laya_bridges_onnx_session_never_pads_a_row_past_the_local_attention_window():
    """laya_bridge.Grouped (LAYA_ONNX): a batch whose rows differ in length by more than PAD_GAP tokens runs in groups,
    each trimmed to its longest row, and the outputs come back in the batch's row order."""
    np = pytest.importorskip("numpy")
    b = load("laya_bridge")
    lens = [300, 20, 290, 60, 25]
    att = np.zeros((5, 300), dtype=np.int64)
    for i, n in enumerate(lens):
        att[i, :n] = 1
    runs = []

    class Session:
        def run(self, names, inputs):
            a = inputs["attention_mask"]
            runs.append((a.shape[1], a.sum(axis=1).tolist()))
            assert all(a.shape[1] - n <= b.PAD_GAP for n in a.sum(axis=1))   # no row padded past the gap
            n = a.sum(axis=1)
            return [np.stack([n, n * 2], axis=1).astype(np.float32), inputs["qtype"][:, None].astype(np.float32)]

    inputs = {"input_ids": att * 7, "attention_mask": att, "marker_pos": np.zeros((5, 3), dtype=np.int64),
              "marker_mask": np.ones((5, 3), dtype=bool), "qtype": np.arange(5)}
    logits, act = b.Grouped(Session()).run(["logits", "act_logits"], inputs)
    assert sorted(runs) == [(60, [60, 25, 20]), (300, [300, 290])]
    assert logits[:, 0].tolist() == lens and act[:, 0].tolist() == [0, 1, 2, 3, 4]   # each row back in its place
    runs.clear()
    b.Grouped(Session()).run(["logits", "act_logits"], {**inputs, "attention_mask": att[[0, 2]],
                                                        "input_ids": att[[0, 2]], "qtype": np.arange(2),
                                                        "marker_pos": inputs["marker_pos"][:2],
                                                        "marker_mask": inputs["marker_mask"][:2]})
    assert runs == [(300, [300, 290])]                                                # close lengths: one run, as is


def test_the_onnx_fix_gives_an_all_padding_attention_row_a_finite_softmax_and_leaves_the_rest(tmp_path):
    """laya_onnx.fix: an attention layer's Where(mask, 0, -inf) mask gets -1e30 for -inf, so a row that is all padding
    (all masked) no longer comes out NaN, every other row is unchanged, and the file is marked so the bridge batches it."""
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    np = pytest.importorskip("numpy")
    from onnx import TensorProto, helper, numpy_helper
    f32 = TensorProto.FLOAT
    const = lambda name, v: helper.make_node("Constant", [], [name], name=name,
                                             value=numpy_helper.from_array(np.array([v], dtype=np.float32), name))
    g = helper.make_graph(
        [const("/attn/zero", 0.0), const("/attn/neginf", -np.inf),
         helper.make_node("Where", ["mask", "/attn/zero", "/attn/neginf"], ["m"], name="/encoder/layers.0/attn/Where"),
         helper.make_node("Add", ["x", "m"], ["a"], name="/encoder/layers.0/attn/Add"),
         helper.make_node("Softmax", ["a"], ["y"], name="/encoder/layers.0/attn/Softmax", axis=-1)],
        "g", [helper.make_tensor_value_info("x", f32, [2, 3]),
              helper.make_tensor_value_info("mask", TensorProto.BOOL, [2, 3])],
        [helper.make_tensor_value_info("y", f32, [2, 3])])
    src, dst = tmp_path / "m.onnx", tmp_path / "fixed.onnx"
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)], ir_version=8), src)
    feed = {"x": np.array([[0.0, 1.0, 5.0], [2.0, 3.0, 4.0]], dtype=np.float32),
            "mask": np.array([[True, True, False], [False, False, False]])}
    before = ort.InferenceSession(str(src)).run(None, feed)[0]
    assert np.isnan(before[1]).all()
    assert load("laya_onnx").fix(str(src), str(dst)) == 1
    s = ort.InferenceSession(str(dst))
    after = s.run(None, feed)[0]
    assert np.isfinite(after).all() and np.array_equal(after[0], before[0]) and after[0][2] == 0.0
    assert s.get_modelmeta().custom_metadata_map["laya_nan_fix"] == "1"
    assert load("laya_onnx").fix(str(dst)) == 0                              # a second run changes nothing
