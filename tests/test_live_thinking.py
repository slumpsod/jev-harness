"""Reading a model's thinking while it happens."""
import http.server
import json
import threading

from jevh.llm import OpenAICompat


def thinker(n_thoughts: int):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                for i in range(n_thoughts):
                    d = {"choices": [{"delta": {"reasoning_content": f"idea {i}. "}}]}
                    self.wfile.write(f"data: {json.dumps(d)}\n\n".encode())
                d = {"choices": [{"delta": {"content": "print(1)"}}]}
                self.wfile.write(f"data: {json.dumps(d)}\n\ndata: [DONE]\n\n".encode())
            except OSError:
                pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_reply_streams_into_a_live_file(tmp_path):
    srv = thinker(5)
    live = tmp_path / "thinking" / "step-01.txt"
    r = OpenAICompat(f"http://127.0.0.1:{srv.server_address[1]}/v1", "m").chat(
        [{"role": "user", "content": "hi"}], live=live)
    srv.shutdown()
    text = live.read_text(encoding="utf-8")
    assert "=== thinking ===\nidea 0. idea 1." in text and "=== reply ===\nprint(1)" in text and "# done" in text
    assert r.text == "print(1)"


def slow_thinker(n_thoughts: int, reply_after: int | None = None, delay: float = 0.002):
    """Streams `n_thoughts` reasoning chunks (~25 chars each); a reply after `reply_after` of them."""
    import time as _t

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                for i in range(n_thoughts):
                    if reply_after is not None and i == reply_after:
                        d = {"choices": [{"delta": {"content": "print(1)"}}]}
                    else:
                        d = {"choices": [{"delta": {"reasoning_content": f"step {i:05d}: the rule is x. "}}]}
                    self.wfile.write(f"data: {json.dumps(d)}\n\n".encode())
                    self.wfile.flush()
                    _t.sleep(delay)
                self.wfile.write(b"data: [DONE]\n\n")
            except OSError:
                pass   # the client hung up: what the watch does
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_watch_pauses_a_long_think_and_keeps_it():
    srv = thinker_srv = slow_thinker(3000)
    seen = []

    def watch(text):
        seen.append(len(text))
        return len(text) > 5000
    r = OpenAICompat(f"http://127.0.0.1:{thinker_srv.server_address[1]}/v1", "m").chat(
        [{"role": "user", "content": "hi"}], watch=watch, watch_every=1000, watch_after=2000)
    srv.shutdown()
    assert r.interrupted and r.text == "" and r.reasoning.startswith("step 00000")
    assert 5000 < len(r.reasoning) < 3000 * 30 and seen[0] >= 2000


def test_watch_never_interrupts_a_reply():
    srv = slow_thinker(400, reply_after=150)
    r = OpenAICompat(f"http://127.0.0.1:{srv.server_address[1]}/v1", "m").chat(
        [{"role": "user", "content": "hi"}], watch=lambda t: True, watch_every=10 ** 9, watch_after=10 ** 9)
    srv.shutdown()
    assert not r.interrupted and "print(1)" in r.text


def test_a_paused_step_is_asked_to_test_its_idea(cfg):
    from jevh.jev import Jev
    from jevh.llm import LLMResult
    from jevh.orchestrator import Harness

    from conftest import FakeSystemOne, ScriptedLLM, py
    llm = ScriptedLLM([LLMResult(text="", reasoning="so the rule is: mirror each row", interrupted=True),
                       py("final('mirrored')")])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne()), llm_factory=lambda p: llm)
    h.learning = False
    assert h.turn("solve it", force="high_end_dev").answer == "mirrored"
    before = llm.seen[1]
    assert before[-2]["reasoning_content"] == "so the rule is: mirror each row"
    assert "paused your thinking" in before[-1]["content"] and "Test that idea now" in before[-1]["content"]
