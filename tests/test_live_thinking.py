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
