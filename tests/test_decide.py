"""The Jev client's one promise: any way Jev fails (unreachable, too slow, a malformed reply) is a SystemOneError, the
only exception the harness's "Jev down: degrade, never crash" handlers (Harness._safe) catch."""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from jevh.decide import Answer, SystemOneClient, SystemOneError, noul
from jevh.jev import Jev
from jevh.orchestrator import Harness

from conftest import ScriptedLLM


class Adapter(BaseHTTPRequestHandler):
    mode = "ok"

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        mode = type(self).mode
        if mode == "drop":            # the adapter restarted mid-request: no answer at all (RemoteDisconnected)
            return
        if mode == "slow":            # nothing until long after the client's timeout (raw TimeoutError before)
            time.sleep(2)
            return
        if mode == "busy":            # answers, but only after 8 s (an adapter queued behind other calls)
            time.sleep(8)
            mode = "ok"
        body = {"html": b"<html>Bad gateway</html>", "list": b"[1, 2]",
                "no_noul": b'{"model": "m", "answers": {"q": {"type": "noul", "error": "busy"}}}',
                "ok": b'{"model": "m", "answers": {"q": {"type": "noul", "noul": 0.7}}}'}[mode]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def adapter():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Adapter)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    Adapter.mode = "ok"
    srv.shutdown()


def client(srv, timeout=0.5):
    return SystemOneClient(f"http://127.0.0.1:{srv.server_address[1]}", backend=None, timeout=timeout, max_retries=0)


def test_a_good_answer_reads_as_before(adapter):
    assert client(adapter).ask({"x": 1}, {"q": noul("yes?")})["q"].noul == 0.7


@pytest.mark.parametrize("mode", ["drop", "slow", "html", "list"])
def test_every_failure_to_answer_is_a_system_one_error(adapter, mode):
    Adapter.mode = mode
    t = time.time()
    with pytest.raises(SystemOneError):
        client(adapter).ask({"x": 1}, {"q": noul("yes?")})
    assert time.time() - t < 5


def test_an_answer_without_its_value_is_a_system_one_error_where_it_is_read(adapter):
    Adapter.mode = "no_noul"
    r = client(adapter).ask({"x": 1}, {"q": noul("yes?")})
    with pytest.raises(SystemOneError, match="no usable 'noul'"):
        r["q"].noul
    with pytest.raises(SystemOneError):
        Answer("p", "choice", {"probabilities": {"a": "high"}}).probabilities


def test_a_turn_degrades_and_still_answers_when_jev_times_out(adapter, cfg):
    """Before: the routing call's timeout came out of urllib raw, Harness._safe (which catches SystemOneError) let it
    through, and the turn crashed. Now every Jev step falls back as designed and the worker still answers."""
    Adapter.mode = "slow"
    events = []
    h = Harness(cfg, jev=Jev(cfg, client(adapter, timeout=0.3)), llm_factory=lambda p: ScriptedLLM(["hi there"]),
                on_event=lambda k, d: events.append((k, d)))
    h.learning = False
    r = h.turn("hello")
    assert r.answer == "hi there" and r.persona == cfg.default_persona
    assert any(k == "jev_error" and d["step"] == "route" for k, d in events)


def test_a_stop_ends_a_jev_call_in_flight(adapter):
    """A Stop used to wait out every Jev call in flight (up to jev.timeout, 300 s)."""
    from jevh.errors import Cancelled
    Adapter.mode = "busy"
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t = time.time()
    with pytest.raises(Cancelled):
        client(adapter, timeout=30).ask({"x": 1}, {"q": noul("yes?")}, cancel=cancel)
    assert time.time() - t < 2


def test_a_stop_while_jev_is_slow_ends_the_turn_at_once(adapter, cfg):
    """Routing, skills, tools and tier each waited for Jev before the turn noticed the Stop. Now the call in flight is
    cut, every later one falls back at once, and the turn ends as stopped."""
    Adapter.mode = "busy"
    h = Harness(cfg, jev=Jev(cfg, client(adapter, timeout=30)), llm_factory=lambda p: ScriptedLLM(["never asked"]))
    h.learning = False
    out = {}
    runner = threading.Thread(target=lambda: out.update(r=h.turn("hello")), daemon=True)
    runner.start()
    while not h._running:
        time.sleep(0.01)
    time.sleep(0.3)
    t = time.time()
    assert h.cancel()
    runner.join(10)
    assert not runner.is_alive() and time.time() - t < 3
    assert out["r"].cancelled
