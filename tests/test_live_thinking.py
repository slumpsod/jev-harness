"""Reading a model's thinking while it happens."""
import http.server
import json
import threading

import pytest

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

    def watch(text, answering=False):
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
        [{"role": "user", "content": "hi"}], watch=lambda t, answering=False: True, watch_every=10 ** 9, watch_after=10 ** 9)
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
    assert "paused your thinking" in before[-1]["content"] and "mirror each row" in before[-1]["content"]
    assert "compute exactly that" in before[-1]["content"] and "print where its output differs" in before[-1]["content"]


def test_jev_watches_only_after_the_task_has_run_code(cfg):
    """Advice, writing and planning never get Jev's thinking watch: no code has run, so there is no data
    in the REPL. (The plain-code loop check watches every step.)"""
    from jevh.jev import Jev
    from jevh.llm import LLMResult
    from jevh.orchestrator import Harness

    from conftest import FakeSystemOne, py
    watched = []

    class Recorder:
        name = "rec"

        def __init__(self, replies):
            self.replies = list(replies)

        def chat(self, messages, watch=None, live=None, watch_every=0, watch_after=0):
            asked = len(jev.calls)
            watch("counting the rows by hand: row 1, row 2, row 3")
            watched.append(len(jev.calls) > asked)   # did the watch ask Jev?
            return LLMResult(text=self.replies.pop(0))
    cfg.limits["thinking_watch"] = 1
    llm = Recorder([py("rows = [1, 2, 3]\nprint(len(rows))"), py("final(str(sum(rows)))")])
    jev = FakeSystemOne()
    h = Harness(cfg, jev=Jev(cfg, jev), llm_factory=lambda p: llm)
    h.learning = False
    assert h.turn("add up the rows", force="high_end_dev").answer == "6"
    assert watched == [False, True]


def test_ignored_pauses_make_the_watch_back_off_but_never_stop(cfg):
    """Live: a model answered two pauses with more hand-work; the watch then switched off and the
    next step thought 30 KB unwatched. Doubling on every ignored pause then let one think 40 KB.
    Now ignored pauses move the check to 2x once, and running code resets it."""
    from jevh.jev import Jev
    from jevh.llm import LLMResult
    from jevh.orchestrator import Harness

    from conftest import FakeSystemOne, py
    after = []   # watch_after per step
    seen = []

    class Recorder:
        name = "rec"

        def __init__(self, replies):
            self.replies = list(replies)

        def chat(self, messages, watch=None, live=None, watch_every=0, watch_after=0):
            after.append(watch_after if watch is not None else None)
            seen.append([dict(m) for m in messages])
            return self.replies.pop(0)
    paused = LLMResult(text="", reasoning="reading sizes by hand", interrupted=True)
    cfg.limits.update(thinking_watch=1, watch_after=6000)
    llm = Recorder([LLMResult(text=py("g = [1, 2]")), paused, paused, paused, LLMResult(text=py("print(g)")),
                    LLMResult(text=py("final('ok')"))])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne()), llm_factory=lambda p: llm)
    h.learning = False
    assert h.turn("go", force="high_end_dev").answer == "ok"
    # 1-3: 6k (one ignored pause changes nothing); 4-5: two or more -> 12k, never more; 6: code ran -> 6k
    assert after == [6000, 6000, 6000, 12000, 12000, 6000]
    assert "pause 2 in a row" in seen[3][-1]["content"] and "pause 2" not in seen[2][-1]["content"]


def test_a_searching_think_is_paused_and_told_to_test_its_best_guess(cfg):
    """Theory after theory (stuck well above checking) pauses even when nothing is 'testable'."""
    from jevh.jev import Jev
    from jevh.llm import LLMResult
    from jevh.orchestrator import Harness

    from conftest import FakeSystemOne, noul_ans, py
    seen, reasons = [], []

    def rules(k, q, s):
        return {"testable": noul_ans(0.1), "stuck": noul_ans(0.95), "checking": noul_ans(0.2)}.get(k)

    class Watching:
        name = "w"

        def __init__(self, replies):
            self.replies = list(replies)

        def chat(self, messages, watch=None, live=None, watch_every=0, watch_after=0):
            seen.append([dict(m) for m in messages])
            if watch is not None:
                reasons.append(watch("maybe it's a spiral... no. Let me try yet another idea: a recursive tree"))
            return self.replies.pop(0)
    cfg.limits["thinking_watch"] = 1
    llm = Watching([LLMResult(text=py("g = [1, 2]")),
                    LLMResult(text="", reasoning="maybe recursion... or a lattice?", interrupted=True),
                    LLMResult(text=py("final('ok')"))])
    events = []
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: llm,
                on_event=lambda k, d: events.append((k, d)))
    h.learning = False
    assert h.turn("go", force="high_end_dev").answer == "ok"
    assert reasons[:2] == [False, True]   # 1: no code has run yet, Jev not asked; 2: searching -> pause
    msg = seen[2][-1]["content"]
    assert "trying one theory after another" in msg and "write your best current guess as a function" in msg
    check = next(d for k, d in events if k == "thinking_watch")
    assert check["search"] == 0.75 and check["p"] == 0.1


def test_checking_is_not_mistaken_for_searching(cfg):
    from jevh.jev import Jev
    from jevh.orchestrator import Harness, Task

    from conftest import FakeSystemOne, noul_ans

    def rules(k, q, s):   # the 8.3-min run's pauses: testable high, stuck barely above checking
        return {"testable": noul_ans(0.2), "stuck": noul_ans(0.7), "checking": noul_ans(0.68)}.get(k)
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: None)
    watch = h._watch_fn(Task(1, None, "user", "high_end_dev", "solve it"))
    assert watch("(3, 4): center (3.5, 9.5): from the seed, -3, +3 ✓") is None


def test_a_looping_reply_is_stopped_and_told_to_act(cfg):
    """Live (Gemma 4, ARC): with the answer computed, it wrote "I'll call final(test_output)" in its reply
    for 108 KB and never called it. The loop check is plain code, runs every step, and reads replies."""
    from jevh.jev import Jev
    from jevh.llm import LLMResult
    from jevh.orchestrator import Harness
    from jevh.worker import repeating

    from conftest import FakeSystemOne, py
    loop = "Wait, let me check the grid once more.\nFinal call.\nI'll call `final(test_output)` now.\n" * 12
    assert repeating(loop) in ("Wait, let me check the grid once more.", "I'll call `final(test_output)` now.")
    assert repeating("0 0 6 0 0 0 0 6\n" * 40) is None                       # printed grids aren't a loop
    assert repeating("Let me check row 3.\nRow 3 has a 6 at column 2.\n" + "The rule holds.\n" * 5) is None

    seen = []

    class Looper:
        name = "g"

        def __init__(self):
            self.n = 0

        def chat(self, messages, watch=None, live=None, watch_every=0, watch_after=0):
            seen.append([dict(m) for m in messages])
            self.n += 1
            if self.n == 1:
                assert watch(loop) is True           # reply text, no reasoning channel, no code yet
                return LLMResult(text=loop, interrupted=True)
            return LLMResult(text=py("test_output = [[1]]\nfinal(test_output)"))
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne()), llm_factory=lambda p: Looper())
    h.learning = False
    assert h.turn("solve it", force="high_end_dev").answer == "[[1]]"
    stopped, told = seen[1][-2]["content"], seen[1][-1]["content"]
    assert stopped.endswith("[... stopped here by the harness]") and len(stopped) < 3100
    assert "you were repeating yourself" in told and "do it now" in told


def test_a_reply_that_never_starts_streaming_is_given_up_on_early():
    """Live (Gemma 4 in LM Studio): generation with nothing streamed for minutes. Until the first
    chunk the limit is short (scaled to the prompt); after it, idle_timeout bounds each gap."""
    import http.server
    import time as _t

    from jevh import llm as L

    class Silent(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                _t.sleep(5)
            except OSError:
                pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Silent)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    old = L.FIRST_CHUNK_BASE
    L.FIRST_CHUNK_BASE = 1.0
    try:
        t = _t.time()
        with pytest.raises(L.LLMError, match="nothing streamed"):
            L.OpenAICompat(f"http://127.0.0.1:{srv.server_address[1]}/v1", "m", idle_timeout=60).chat(
                [{"role": "user", "content": "hi"}])
        assert _t.time() - t < 4
    finally:
        L.FIRST_CHUNK_BASE = old
        srv.shutdown()


def timed_stream(n: int, delay: float, timings: dict | None = None, hold: bool = False):
    """A server that streams `n` one-token chunks `delay` apart (`hold`: all of them at once, after the wait), then its usage."""
    import time as _t

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            if hold:
                _t.sleep(n * delay)
            for i in range(n):
                self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': 'x '}}]})}\n\n".encode())
                self.wfile.flush()
                if not hold:
                    _t.sleep(delay)
            last = {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": n}, **({"timings": timings} if timings else {})}
            self.wfile.write(f"data: {json.dumps(last)}\n\ndata: [DONE]\n\n".encode())
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_decode_time_is_what_the_server_says_else_the_span_the_tokens_streamed_in():
    ask = [{"role": "user", "content": "hi"}]
    srv = timed_stream(20, 0.03, timings={"predicted_ms": 400.0})
    r = OpenAICompat(f"http://127.0.0.1:{srv.server_address[1]}/v1", "m").chat(ask)
    srv.shutdown()
    assert r.decode_seconds == 0.4 and r.output_tokens == 20                       # llama.cpp's own figure wins
    srv = timed_stream(20, 0.03)
    r = OpenAICompat(f"http://127.0.0.1:{srv.server_address[1]}/v1", "m").chat(ask)
    srv.shutdown()
    assert 0.4 < r.decode_seconds < 1.5                                            # ~19 gaps of 30 ms, measured from the stream
    srv = timed_stream(20, 0.03, hold=True)
    r = OpenAICompat(f"http://127.0.0.1:{srv.server_address[1]}/v1", "m").chat(ask)
    srv.shutdown()
    assert r.decode_seconds == 0.0                                                 # sent all at once: it says nothing about speed


def test_progress_reports_the_speed_of_the_last_50_chunks_and_a_zero_when_the_call_ends():
    seen = []
    srv = timed_stream(120, 0.01)
    llm = OpenAICompat(f"http://127.0.0.1:{srv.server_address[1]}/v1", "m")
    llm.on_progress = lambda n, thinking=0, tps=0.0: seen.append((n, round(tps)))
    llm.chat([{"role": "user", "content": "hi"}])
    srv.shutdown()
    assert [n for n, _ in seen][:2] == [50, 100] and len(seen) == 3 and seen[-1][1] == 0
    assert all(10 < tps < 120 for _, tps in seen[:2])                              # 10 ms a token: under 100 a second
    short = []
    srv = timed_stream(20, 0.001)
    llm = OpenAICompat(f"http://127.0.0.1:{srv.server_address[1]}/v1", "m")
    llm.on_progress = lambda *a: short.append(a)
    llm.chat([{"role": "user", "content": "hi"}])
    srv.shutdown()
    assert short == []                                                             # a short reply reports nothing at all


def streams(*chunks: str):
    """A server whose reply streams these `data:` payloads, as written."""
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for c in chunks:
                self.wfile.write(f"data: {c}\n\n".encode())
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_an_error_the_server_streams_mid_reply_is_an_error_not_an_empty_reply():
    """llama.cpp and LM Studio report a failure mid-reply (a context overflow) as a stream event. Ignored, it read as an
    empty reply, and the worker was told to continue, step after step, until its step limit."""
    from jevh.llm import LLMError
    srv = streams(json.dumps({"choices": [{"delta": {"content": "Let me"}}]}),
                  json.dumps({"error": {"code": 400, "message": "the request exceeds the available context size"}}))
    with pytest.raises(LLMError, match="exceeds the available context size"):
        OpenAICompat(f"http://127.0.0.1:{srv.server_address[1]}/v1", "m").chat([{"role": "user", "content": "hi"}])
    srv.shutdown()


def test_a_stream_chunk_that_is_not_json_is_an_llm_error_not_a_crash():
    from jevh.llm import LLMError
    srv = streams("{not json")
    with pytest.raises(LLMError, match="not JSON"):
        OpenAICompat(f"http://127.0.0.1:{srv.server_address[1]}/v1", "m").chat([{"role": "user", "content": "hi"}])
    srv.shutdown()
