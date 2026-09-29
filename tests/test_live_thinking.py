"""Reading a model's thinking while it happens, and cutting a runaway think."""
import http.server
import json
import threading

from jevh.jev import Jev
from jevh.llm import LLMResult, OpenAICompat
from jevh.orchestrator import Harness

from conftest import FakeSystemOne, ScriptedLLM, py


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
    assert r.text == "print(1)" and not r.thinking_capped


def test_think_cap_cuts_a_runaway_think_and_keeps_what_it_had(tmp_path):
    srv = thinker(500)
    r = OpenAICompat(f"http://127.0.0.1:{srv.server_address[1]}/v1", "m").chat(
        [{"role": "user", "content": "hi"}], think_cap=10)
    srv.shutdown()
    assert r.thinking_capped and r.text == "" and r.reasoning.startswith("idea 0.") and "idea 9." in r.reasoning


def test_a_capped_step_is_told_to_act_with_its_reasoning_kept(cfg):
    llm = ScriptedLLM([LLMResult(text="", reasoning="maybe the rule mirrors rows", thinking_tokens=30000,
                                 thinking_capped=True),
                       py("final('mirrored')")])
    cfg.limits["think_cap"] = 30000
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne()), llm_factory=lambda p: llm)
    h.learning = False
    assert h.turn("solve it", force="high_end_dev").answer == "mirrored"
    before = llm.seen[1]
    assert before[-2]["reasoning_content"] == "maybe the rule mirrors rows"
    assert "without acting" in before[-1]["content"] and "tests your current best hypothesis" in before[-1]["content"]
