"""Stopping a turn, and workers asking the user mid-task."""
import threading
import time

from jevh.jev import Jev
from jevh.orchestrator import STOPPED_NOTE, Harness

from conftest import FakeSystemOne, ScriptedLLM, choice_ans, noul_ans, py


def rules(k, q, s):
    if k == "persona":
        return choice_ans("daily_assistant", 0.9, q["criteria"])
    if k in ("needs_high_end_dev", "in_scope", "necessary"):
        return noul_ans(0.9)
    if k == "complexity":
        return {"score": 1.8, "confidence": 0.9}


def harness(cfg, scripts, events=None):
    llms = {n: ScriptedLLM(r, n) for n, r in scripts.items()}
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: llms[p.name],
                on_event=(lambda k, d: events.append((k, d))) if events is not None else None)
    h.learning = False
    return h, llms


def run_async(h, msg, **kw):
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", h.turn(msg, **kw)))
    t.start()
    return t, out


def wait_for(cond, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_cancel_kills_running_code_and_keeps_the_request_as_context(cfg):
    events = []
    h, _ = harness(cfg, {"daily_assistant": [py("import time\nprint('working')\ntime.sleep(60)\nfinal('never')")]},
                   events)
    t, out = run_async(h, "do a long job")
    assert wait_for(lambda: any(k == "code" for k, _ in events))
    time.sleep(0.5)
    started = time.time()
    assert h.cancel() is True
    t.join(15)
    assert not t.is_alive() and time.time() - started < 10
    r = out["r"]
    assert r.cancelled and r.answer == STOPPED_NOTE and r.tasks[0].status == "cancelled"
    last = h.memory.session_turns(h.session, 1)[0]
    assert last.user == "do a long job" and last.assistant == STOPPED_NOTE
    assert h.cancel() is False  # nothing running any more


def test_cancel_reaches_hand_offs(cfg):
    events = []
    h, _ = harness(cfg, {
        "daily_assistant": [py("r = request('high_end_dev', 'long build')\nfinal(r['answer'])")],
        "high_end_dev": [py("import time\ntime.sleep(60)\nfinal('built')")],
    }, events)
    t, out = run_async(h, "build me an app")
    assert wait_for(lambda: any(k == "code" and d.get("persona") == "high_end_dev" for k, d in events))
    h.cancel()
    t.join(15)
    assert not t.is_alive()
    assert {x.to: x.status for x in out["r"].tasks} == {"daily_assistant": "cancelled", "high_end_dev": "cancelled"}


def test_ask_user_waits_for_the_answer_and_keeps_state(cfg):
    events = []
    h, llms = harness(cfg, {"daily_assistant": [py(
        "cart = ['blue mug']\n"
        "a = ask_user('Which size?', ['Small', 'Large'])\n"
        "final(f'{cart[0]} in {a}')")]}, events)
    t, out = run_async(h, "order a mug")
    assert wait_for(lambda: h.pending_questions())
    q = h.pending_questions()[0]
    assert q["question"] == "Which size?" and q["options"] == ["Small", "Large"] and q["persona"] == "daily_assistant"
    assert h.answer(q["id"], "Large") is True
    t.join(15)
    assert out["r"].answer == "blue mug in Large"
    assert [k for k, _ in events if k in ("question", "answered")] == ["question", "answered"]
    assert h.pending_questions() == [] and h.answer(q["id"], "again") is False


def test_cancel_while_waiting_for_an_answer(cfg):
    h, _ = harness(cfg, {"daily_assistant": [py("a = ask_user('Which size?')\nfinal(a)")]})
    t, out = run_async(h, "order a mug")
    assert wait_for(lambda: h.pending_questions())
    h.cancel()
    t.join(15)
    assert out["r"].cancelled and h.pending_questions() == []


def test_unanswered_question_times_out_and_the_worker_carries_on(cfg):
    cfg.limits["ask_timeout"] = 1
    h, _ = harness(cfg, {"daily_assistant": [py("a = ask_user('Which size?')\nfinal('went with Medium: ' + a[:20])")]})
    r = h.turn("order a mug")
    assert r.answer.startswith("went with Medium: (no answer from the")


def test_stop_cuts_a_streaming_reply():
    import http.server
    import json as _json

    import pytest

    from jevh.errors import Cancelled
    from jevh.llm import OpenAICompat

    class Slow(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                for i in range(100):  # 20s of tokens unless the client hangs up
                    chunk = {"choices": [{"delta": {"content": f"tok{i} "}}]}
                    self.wfile.write(f"data: {_json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.2)
            except OSError:
                pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    llm = OpenAICompat(f"http://127.0.0.1:{srv.server_address[1]}/v1", "slow")
    stop = threading.Event()
    threading.Timer(0.6, stop.set).start()
    t = time.time()
    with pytest.raises(Cancelled):
        llm.chat([{"role": "user", "content": "hi"}], cancel=stop)
    assert time.time() - t < 2
    srv.shutdown()


def test_reasoning_text_is_captured_from_the_stream():
    import http.server
    import json as _json

    from jevh.llm import OpenAICompat

    class Thinker(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for delta in ({"reasoning_content": "The rule mirrors "}, {"reasoning_content": "each row."},
                          {"content": "Here is the code."}):
                self.wfile.write(f"data: {_json.dumps({'choices': [{'delta': delta}]})}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Thinker)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    r = OpenAICompat(f"http://127.0.0.1:{srv.server_address[1]}/v1", "t").chat([{"role": "user", "content": "hi"}])
    srv.shutdown()
    assert r.text == "Here is the code." and r.reasoning == "The rule mirrors each row." and r.thinking_tokens == 2
