"""Messages sent while a task is running: redirect (cut the model's reply, ask again with the message), steer
(deliver it with the next step), queue, status. See jevh/interject.py and docs/architecture/16-interjection.md."""
import http.server
import json
import threading
import time

import pytest

from jevh.errors import Cancelled
from jevh.llm import LLMResult, OpenAICompat, WithFallback


def stream_server(handler_body):
    """A local OpenAI-style streaming server. `handler_body(handler)` writes the events."""
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                handler_body(self)
            except OSError:
                pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen, f"http://127.0.0.1:{srv.server_address[1]}/v1"


def chunk(h, delta):
    h.wfile.write(f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode())
    h.wfile.flush()


def slow_tokens(h):
    for i in range(100):   # 20 s of tokens unless the client hangs up
        chunk(h, {"content": f"tok{i} "})
        time.sleep(0.2)


# -- the model call -----------------------------------------------------------------------------------------------

def test_interrupt_cuts_the_reply_and_returns_what_was_streamed():
    srv, seen, url = stream_server(slow_tokens)
    llm = OpenAICompat(url, "slow")
    interrupt = threading.Event()
    threading.Timer(0.7, interrupt.set).start()
    t = time.time()
    out = llm.chat([{"role": "user", "content": "hi"}], interrupt=interrupt)
    srv.shutdown()
    assert time.time() - t < 3
    assert out.redirected and out.interrupted
    assert out.text.startswith("tok0 tok1") and "tok99" not in out.text   # what came before the cut is kept


def test_interrupt_keeps_the_reasoning_streamed_so_far():
    def thinking(h):
        chunk(h, {"reasoning_content": "The user wants the mug. "})
        chunk(h, {"reasoning_content": "Checking the size."})
        time.sleep(30)

    srv, seen, url = stream_server(thinking)
    interrupt = threading.Event()
    threading.Timer(0.6, interrupt.set).start()
    t = time.time()
    out = OpenAICompat(url, "t").chat([{"role": "user", "content": "hi"}], interrupt=interrupt)
    srv.shutdown()
    assert time.time() - t < 5               # cut at once, not when the silent server next speaks
    assert out.redirected and out.reasoning == "The user wants the mug. Checking the size." and out.text == ""


def test_a_silent_server_is_cut_at_once_for_a_redirect_and_for_a_stop():
    """A model still reading its prompt sends nothing for seconds (LM Studio: headers at once, the first chunk
    after the prompt is read). Closing the response does not wake a read waiting for it; shutting the socket does."""
    def quiet(h):
        time.sleep(30)

    srv, seen, url = stream_server(quiet)
    llm = OpenAICompat(url, "quiet")
    for how in ("interrupt", "cancel"):
        ev = threading.Event()
        threading.Timer(0.6, ev.set).start()
        t = time.time()
        if how == "interrupt":
            out = llm.chat([{"role": "user", "content": "hi"}], interrupt=ev)
            assert out.redirected and out.text == ""
        else:
            with pytest.raises(Cancelled):
                llm.chat([{"role": "user", "content": "hi"}], cancel=ev)
        assert time.time() - t < 5, how
    srv.shutdown()


def test_interrupt_set_before_the_request_sends_nothing():
    srv, seen, url = stream_server(slow_tokens)
    interrupt = threading.Event()
    interrupt.set()
    out = OpenAICompat(url, "slow").chat([{"role": "user", "content": "hi"}], interrupt=interrupt)
    srv.shutdown()
    assert out.redirected and out.text == "" and seen == []


def test_a_stop_beats_a_redirect():
    srv, seen, url = stream_server(slow_tokens)
    stop, interrupt = threading.Event(), threading.Event()

    def both():
        interrupt.set()
        stop.set()
    threading.Timer(0.6, both).start()
    with pytest.raises(Cancelled):
        OpenAICompat(url, "slow").chat([{"role": "user", "content": "hi"}], cancel=stop, interrupt=interrupt)
    srv.shutdown()


def test_without_an_interrupt_the_reply_is_whole():
    def short(h):
        chunk(h, {"content": "all of it"})
        h.wfile.write(b"data: [DONE]\n\n")

    srv, seen, url = stream_server(short)
    out = OpenAICompat(url, "s").chat([{"role": "user", "content": "hi"}], interrupt=threading.Event())
    srv.shutdown()
    assert out.text == "all of it" and not out.redirected and not out.interrupted


def test_the_fallback_passes_interrupt_only_to_models_that_take_it():
    class Takes:
        name = "takes"

        def chat(self, messages, cancel=None, interrupt=None):
            return LLMResult(text="took it", redirected=bool(interrupt and interrupt.is_set()))

    class Plain:
        name = "plain"

        def chat(self, messages, cancel=None):
            return LLMResult(text="plain")

    ev = threading.Event()
    ev.set()
    assert WithFallback(lambda: Takes(), Plain()).chat([], interrupt=ev).redirected
    assert WithFallback(lambda: Plain(), Plain()).chat([], interrupt=ev).text == "plain"


# -- the worker ---------------------------------------------------------------------------------------------------

from jevh.personas import Persona  # noqa: E402
from jevh.worker import CUT_REPLY, STEER_HELD, Worker  # noqa: E402

from conftest import py  # noqa: E402


class Model:
    """A model with a script. A reply "SLOW" waits for the interrupt, as a long generation would, and comes back
    cut (what it had written so far); a function reply runs inside the call (a message arriving mid-reply)."""
    name = "model"

    def __init__(self, replies, partial="I will look at the em"):
        self.replies, self.partial = list(replies), partial
        self.seen: list[list[dict]] = []
        self.started = threading.Event()

    def chat(self, messages, cancel=None, interrupt=None):
        self.seen.append([dict(m) for m in messages])
        reply = self.replies.pop(0) if self.replies else "done"
        if reply == "SLOW":
            self.started.set()
            interrupt.wait(10)
            cut = interrupt.is_set()
            return LLMResult(text=self.partial, reasoning="the user wants their email", model=self.name,
                             interrupted=cut, redirected=cut)
        if callable(reply):
            reply = reply()
        return LLMResult(text=reply, input_tokens=10, output_tokens=5, model=self.name)


def make_worker(tmp_path, llm, steps=10, code=True):
    persona = Persona("t", "a test assistant", "You are a test assistant.", {"provider": "x", "model": "y"},
                      max_steps=steps, code=code)
    events = []
    (tmp_path / "w").mkdir(exist_ok=True)
    w = Worker(persona, llm, tmp_path / "w", tmp_path / "art", {}, on_step=lambda k, d: events.append((k, d)))
    return w, events


def run_in_thread(w, task="check my email"):
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", w.run(task, {}, "system")))
    t.start()
    return t, out


def wait_for(cond, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_a_message_sent_while_code_runs_arrives_with_the_next_step(tmp_path):
    llm = Model([py("import time\nprint('searching')\ntime.sleep(1.5)"), py("final('km')")])
    w, events = make_worker(tmp_path, llm)
    t, out = run_in_thread(w)
    assert wait_for(lambda: any(k == "code" for k, _ in events))
    assert w.steer("use km, not miles") is True
    t.join(20)
    assert out["r"].answer == "km"
    second = llm.seen[1]
    assert second[-1]["role"] == "user" and "use km, not miles" in second[-1]["content"]
    assert llm.seen[1][:len(llm.seen[0])] == llm.seen[0]   # everything before it is unchanged: the server's cache holds
    assert w.corrections == ["use km, not miles"] and "use km, not miles" in w._task_text
    assert [d["text"] for k, d in events if k == "steer"] == ["use km, not miles"]


def test_a_message_cuts_the_reply_in_flight_and_the_step_is_not_charged(tmp_path):
    llm = Model(["SLOW", py("final('calendar')")], partial="I will check the email.\n```python\nprint('EMAIL')")
    w, events = make_worker(tmp_path, llm, steps=2)
    t, out = run_in_thread(w)
    assert llm.started.wait(10)
    w.steer("actually check my calendar")
    t.join(20)
    assert out["r"].answer == "calendar"      # two steps allowed: the cut one was not charged
    second = llm.seen[1]
    cut, said = second[-2], second[-1]
    assert cut["role"] == "assistant" and "EMAIL" not in cut["content"] and CUT_REPLY in cut["content"]
    assert cut["content"].startswith("I will check the email.") and "tool_calls" not in cut
    assert cut["reasoning_content"] == "the user wants their email"
    assert said["role"] == "user" and "actually check my calendar" in said["content"]
    assert [k for k, _ in events if k in ("redirect", "steer")] == ["redirect", "steer"]


def test_a_reply_the_message_crossed_is_not_acted_on(tmp_path):
    marker = tmp_path / "ran.txt"
    box = {}

    def reply_while_the_user_speaks():
        box["w"].steer("wait, not that one")
        return py(f"open({str(marker)!r}, 'w').write('ran')\nfinal('old')")

    llm = Model([reply_while_the_user_speaks, py("final('new')")])
    w, _ = make_worker(tmp_path, llm)
    box["w"] = w
    r = w.run("do it", {}, "system")
    assert r.answer == "new" and not marker.exists()
    kept = llm.seen[1][-2]
    assert kept["role"] == "assistant" and CUT_REPLY in kept["content"]


def test_final_is_held_while_a_message_waits(tmp_path):
    llm = Model([py("import time\ntime.sleep(1.5)\nfinal('old answer')"), py("final('new answer')")])
    w, events = make_worker(tmp_path, llm)
    t, out = run_in_thread(w)
    assert wait_for(lambda: any(k == "code" for k, _ in events))
    w.steer("also say hello")
    t.join(20)
    assert out["r"].answer == "new answer"
    held = llm.seen[1]
    assert STEER_HELD.strip() in held[-2]["content"] and "also say hello" in held[-1]["content"]


def test_a_message_that_comes_too_late_is_handed_back(tmp_path):
    llm = Model([py("final('done')")])
    w, _ = make_worker(tmp_path, llm)
    assert w.run("quick", {}, "system").answer == "done"
    assert w.steer("one more thing") is True
    assert w.take_unapplied() == ["one more thing"] and w.take_unapplied() == []
    assert w.steer("   ") is False


def test_a_chat_only_persona_is_asked_again_with_the_message(tmp_path):
    llm = Model(["SLOW", "the calendar answer"], partial="Your emails are")
    w, _ = make_worker(tmp_path, llm, code=False)
    t, out = run_in_thread(w, "check my email")
    assert llm.started.wait(10)
    w.steer("calendar instead")
    t.join(20)
    assert out["r"].answer == "the calendar answer"
    assert "calendar instead" in llm.seen[1][-1]["content"] and llm.seen[1][-2]["role"] == "assistant"


def test_status_says_what_the_worker_is_doing(tmp_path):
    llm = Model([py("import time\ntime.sleep(1.5)\nfinal('x')")])
    w, events = make_worker(tmp_path, llm)
    assert w.status()["doing"] == "getting started"
    t, out = run_in_thread(w)
    assert wait_for(lambda: any(k == "code" for k, _ in events))
    s = w.status()
    assert s["step"] == 1 and s["doing"].startswith("running code: import time") and s["pending"] is False
    assert w.snapshot()[0]["role"] == "system" and w.snapshot()[1]["content"].startswith("check my email")
    t.join(20)


# -- the harness --------------------------------------------------------------------------------------------------

from jevh import interject  # noqa: E402
from jevh.decide import SystemOneError  # noqa: E402
from jevh.jev import Jev, decide_interject  # noqa: E402
from jevh.orchestrator import Harness  # noqa: E402

from conftest import FakeSystemOne, choice_ans, noul_ans  # noqa: E402

SLEEPY = py("import time\ntime.sleep(2)\nprint('checked email')")


def interject_rules(kinds, on_persona=None, fail_kind=False):
    """Jev's answers: the persona is daily_assistant; a message's kind comes from `kinds` (default: a correction)."""
    def rules(k, q, s):
        if k == "persona":
            if on_persona:
                on_persona()
            return choice_ans("daily_assistant", 0.9, q["criteria"])
        if k in ("needs_high_end_dev", "in_scope", "necessary"):
            return noul_ans(0.9)
        if k == "complexity":
            return {"score": 1.8, "confidence": 0.9}
        if k == "kind":
            if fail_kind:
                raise SystemOneError("jev down")
            return choice_ans(kinds.get(s["message"], "correction"), 0.95, q["criteria"])
    return rules


class Smart(Model):
    """A model that also answers a side question (the quick reply) in its own words."""

    def chat(self, messages, cancel=None, interrupt=None):
        if "side message" in str(messages[-1].get("content", "")):
            self.seen.append([dict(m) for m in messages])
            return LLMResult(text="It is checking your email.", input_tokens=50, output_tokens=8, model=self.name)
        return super().chat(messages, cancel, interrupt)


def harness(cfg, replies, kinds=None, model=Model, events=None, **kw):
    llm = model(replies)
    fake = FakeSystemOne(interject_rules(kinds or {}, **kw))
    h = Harness(cfg, jev=Jev(cfg, fake), llm_factory=lambda p: llm,
                on_event=(lambda k, d: events.append((k, d))) if events is not None else None)
    h.learning = False
    return h, llm, fake


def run_turn(h, msg="check my email"):
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", h.turn(msg)))
    t.start()
    return t, out


def test_a_correction_reaches_the_running_task_and_becomes_part_of_its_request(cfg):
    events = []
    h, llm, _ = harness(cfg, [SLEEPY, py("final('calendar done')")], events=events)
    t, out = run_turn(h)
    assert wait_for(lambda: any(k == "code" for k, _ in events))
    res = h.interject("actually check my calendar")
    assert res["outcome"] == "corrected" and res["kind"] == "correction"
    t.join(30)
    r = out["r"]
    assert r.answer == "calendar done"
    assert "actually check my calendar" in llm.seen[1][-1]["content"]
    task = r.tasks[0]
    assert task.text == "check my email" and task.said == [("actually check my calendar", False)]
    assert "actually check my calendar" in task.current and task.current.startswith("check my email")   # what the gate judges
    notes = h.memory.session_turns(h.session, 1)[0].notes
    assert "While this ran, the user also said" in notes and "actually check my calendar" in notes
    kinds = [k for k, _ in events]
    assert "interject" in kinds and "steer" in kinds
    assert [d["outcome"] for k, d in events if k == "interject"] == ["corrected"]


def test_a_status_question_is_answered_without_touching_the_task(cfg):
    cfg.raw = {**cfg.raw, "interject": {"aside_model": False}}
    events = []
    h, llm, _ = harness(cfg, [SLEEPY, py("final('done')")], {"how is it going?": "status"}, events=events)
    t, out = run_turn(h)
    assert wait_for(lambda: any(k == "code" for k, _ in events))
    res = h.interject("how is it going?")
    assert res["outcome"] == "status" and "step 1" in res["answer"] and "running code" in res["answer"]
    assert res["aside"] is False                                   # (aside_model is off: no answer by the model follows)
    t.join(30)
    assert out["r"].answer == "done"
    assert all("how is it going?" not in str(m.get("content")) for call in llm.seen for m in call)
    assert h.memory.session_turns(h.session, 1)[0].notes.count("While this ran") == 0


def test_the_quick_reply_comes_from_the_tasks_model_on_a_digest(cfg):
    events = []
    h, llm, _ = harness(cfg, [SLEEPY, py("final('done')")], {"how is it going?": "status"}, model=Smart,
                        events=events)
    t, out = run_turn(h)
    assert wait_for(lambda: any(k == "code" for k, _ in events))
    h.interject("how is it going?")
    assert wait_for(lambda: any(k == "aside" for k, _ in events))
    t.join(30)
    assert next(d for k, d in events if k == "interject")["aside"] is True       # the dashboard says "answer coming"
    aside = next(d for k, d in events if k == "aside")
    assert aside["source"] == "model" and aside["answer"] == "It is checking your email." and aside["context"] == "digest"
    prompt = next(c for c in llm.seen if "side message" in str(c[-1]["content"]))
    assert prompt[0]["role"] == "system" and "check my email" in prompt[1]["content"]
    assert "time.sleep(2)" in prompt[1]["content"] or "import time" in prompt[1]["content"]   # what it is running
    assert out["r"].answer == "done"                     # the task itself went on untouched
    kinds = h.usage.summary(h.session, 1)["by_kind"]
    assert kinds["aside"]["calls"] == 1 and kinds["aside"]["output"] == 8     # metered as its own kind, not as the task's


def test_the_full_quick_reply_shares_the_workers_own_prompt(cfg):
    cfg.raw = {**cfg.raw, "interject": {"aside_context": "full"}}
    events = []
    h, llm, _ = harness(cfg, [SLEEPY, py("final('done')")], {"how is it going?": "status"}, model=Smart,
                        events=events)
    t, out = run_turn(h)
    assert wait_for(lambda: any(k == "code" for k, _ in events))
    h.interject("how is it going?")
    assert wait_for(lambda: any(k == "aside" for k, _ in events))
    t.join(30)
    first = llm.seen[0]
    prompt = next(c for c in llm.seen if "side message" in str(c[-1]["content"]))
    assert prompt[:len(first)] == first                                  # the worker's own prompt, then one more message
    assert next(d for k, d in events if k == "aside")["context"] == "full"


def test_queue_holds_a_request_for_after(cfg):
    events = []
    h, llm, _ = harness(cfg, [SLEEPY, py("final('done')")], events=events)
    t, out = run_turn(h)
    assert wait_for(lambda: any(k == "code" for k, _ in events))
    res = h.interject("/queue also remind me to call Dave")
    assert res["outcome"] == "queued" and h.queued() == ["also remind me to call Dave"]
    t.join(30)
    assert all("call Dave" not in str(m.get("content")) for call in llm.seen for m in call)     # the task never saw it
    assert h.take_queued() == "also remind me to call Dave" and h.take_queued() is None


def test_a_plain_message_is_added_to_the_prompt_even_if_it_reads_like_another_request(cfg):
    """Jev thinks this is a separate request: it used to be queued. Now only /queue queues."""
    events = []
    h, llm, _ = harness(cfg, [SLEEPY, py("final('done')")], {"also remind me to call Dave": "separate"}, events=events)
    t, out = run_turn(h)
    assert wait_for(lambda: any(k == "code" for k, _ in events))
    res = h.interject("also remind me to call Dave")
    assert res["outcome"] == "corrected" and res["kind"] == "correction" and h.queued() == []
    t.join(30)
    assert "also remind me to call Dave" in llm.seen[1][-1]["content"]                            # the task got it
    assert h.take_queued() is None


def test_a_plain_message_that_reads_like_a_stop_is_added_to_the_prompt_not_obeyed(cfg):
    events = []
    h, llm, _ = harness(cfg, [SLEEPY, py("final('wrapped up')")], {"stop": "stop"}, events=events)
    t, out = run_turn(h)
    assert wait_for(lambda: any(k == "code" for k, _ in events))
    res = h.interject("stop")
    assert res["outcome"] == "corrected"                                                          # only /stop stops
    t.join(30)
    r = out["r"]
    assert not r.cancelled and r.answer == "wrapped up" and "stop" in llm.seen[1][-1]["content"]


def test_stop_stops_the_turn_and_hands_back_what_was_queued(cfg):
    events = []
    h, llm, _ = harness(cfg, [py("import time\ntime.sleep(60)")], events=events)
    t, out = run_turn(h)
    assert wait_for(lambda: any(k == "code" for k, _ in events))
    h.interject("/queue later: milk")
    assert h.interject("/stop")["outcome"] == "stopped"
    t.join(30)
    assert out["r"].cancelled and h.queued() == []
    assert next(d for k, d in events if k == "cancel")["dropped"] == ["later: milk"]


def test_a_command_decides_without_asking_jev(cfg):
    events = []
    h, llm, fake = harness(cfg, [SLEEPY, py("final('done')")], events=events)
    t, out = run_turn(h)
    assert wait_for(lambda: any(k == "code" for k, _ in events))
    assert h.interject("/queue and then the weather")["outcome"] == "queued"
    assert h.interject("/steer use km")["outcome"] == "corrected"
    t.join(30)
    assert h.queued() == ["and then the weather"]
    assert not any("kind" in q for _, q in fake.calls)       # Jev was never asked what the messages were


def test_without_jev_a_message_is_a_correction(cfg):
    events = []
    h, llm, _ = harness(cfg, [SLEEPY, py("final('done')")], events=events, fail_kind=True)
    t, out = run_turn(h)
    assert wait_for(lambda: any(k == "code" for k, _ in events))
    assert h.interject("use km")["outcome"] == "corrected"
    t.join(30)
    assert "use km" in llm.seen[1][-1]["content"]


def test_nothing_running_means_send_it_as_an_ordinary_message(cfg):
    h, _, _ = harness(cfg, [])
    assert h.interject("/steer hello") == {"outcome": "idle", "text": "hello"}
    assert h.status() is None


def test_a_message_sent_while_the_harness_is_still_choosing_waits_for_the_worker(cfg):
    box = {}

    def once():
        if "sent" not in box:
            box["sent"] = h.interject("/steer use km")   # inside routing: the lead worker does not exist yet

    h, llm, _ = harness(cfg, [py("final('km it is')")], on_persona=once)
    r = h.turn("how far is it")
    assert box["sent"]["outcome"] == "corrected" and r.answer == "km it is"
    assert "use km" in llm.seen[0][-1]["content"]


def test_a_message_sent_after_the_worker_is_done_becomes_the_next_request(cfg):
    box = {}
    events = []

    def watch(kind, data):
        events.append((kind, data))
        if kind == "done" and "res" not in box:
            box["res"] = h.interject("/steer one more thing")   # the lead worker has finished: nothing to correct

    h, llm, _ = harness(cfg, [py("final('done')")])
    h.on_event = watch
    r = h.turn("quick one")
    assert box["res"]["outcome"] == "queued" and h.take_queued() == "one more thing"
    assert r.answer == "done"


def test_interjecting_can_be_switched_off(cfg):
    cfg.raw = {**cfg.raw, "interject": {"enabled": False}}
    events = []
    h, llm, _ = harness(cfg, [SLEEPY, py("final('done')")], events=events)
    t, out = run_turn(h)
    assert wait_for(lambda: any(k == "code" for k, _ in events))
    assert h.interject("use km")["outcome"] == "refused"
    t.join(30)


# -- the rules ----------------------------------------------------------------------------------------------------

def test_only_a_clear_question_is_answered_everything_else_is_added_to_the_prompt():
    bar = 0.6
    assert decide_interject({"correction": 0.2, "status": 0.7, "separate": 0.05, "stop": 0.05}, bar) == "status"
    assert decide_interject({"correction": 0.2, "status": 0.5, "separate": 0.2, "stop": 0.1}, bar) == "correction"
    assert decide_interject({"correction": 0.1, "status": 0.1, "separate": 0.1, "stop": 0.7}, bar) == "correction"      # (not obeyed)
    assert decide_interject({"correction": 0.1, "status": 0.1, "separate": 0.65, "stop": 0.15}, bar) == "correction"    # (not queued)
    assert decide_interject({}, bar) == "correction"
    assert decide_interject({"correction": 0.0, "status": 1.0, "separate": 0.0, "stop": 0.0}, 1.0) == "status"
    # "check my calendar instead" drops the old task but asks for a new one: it is added to the prompt, never answered
    assert decide_interject({"correction": 0.02, "replace": 0.9, "status": 0.0, "separate": 0.0, "stop": 0.08}, bar) == "correction"
    assert decide_interject({"correction": 0.1, "replace": 0.4, "status": 0.45, "separate": 0.0, "stop": 0.05}, bar) == "correction"


def test_commands_are_read_from_the_start_of_a_message():
    assert interject.parse("/steer use km") == ("correction", "use km")
    assert interject.parse("/QUEUE  later: milk ") == ("separate", "later: milk")
    assert interject.parse("/btw why that site?") == ("status", "why that site?")
    assert interject.parse("/status") == ("status", "")
    assert interject.parse("/stop") == ("stop", "")
    assert interject.parse("/stopwatch") == (None, "/stopwatch")      # a word that merely starts like one
    assert interject.parse("please /stop") == (None, "please /stop")


def test_the_instant_status_line_says_where_the_task_is():
    line = interject.status_line({"persona": "daily_assistant", "step": 4, "max_steps": 30,
                                  "doing": "running code: web.search('flights')", "seconds": 8.2,
                                  "corrections": ["use km"], "pending": True}, queued=2)
    assert "step 4 of up to 30" in line and "web.search('flights')" in line and "8s" in line
    assert "1 message from you" in line and "waiting for its next step" in line and "2 more requests" in line
    assert "getting ready" in interject.status_line({"persona": None, "doing": "getting ready"})


def test_the_full_quick_reply_leaves_out_a_step_that_is_still_running():
    snap = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]}]
    msgs = interject.aside_messages("full", "t", "task", "running code", "how far?", "", snap)
    assert [m["role"] for m in msgs] == ["system", "user", "user"] and "how far?" in msgs[-1]["content"]
    digest = interject.aside_messages("digest", "t", "the task", "running code", "how far?", "- ran: x", snap)
    assert digest[0]["role"] == "system" and "the task" in digest[1]["content"] and "- ran: x" in digest[1]["content"]


# -- the terminal -------------------------------------------------------------------------------------------------

def test_the_terminal_reader_gives_nothing_without_a_terminal():
    from jevh.cli import LineReader
    assert LineReader().poll() is None        # under pytest stdin is not a terminal: there is never a line


def test_the_terminal_shows_what_became_of_a_message(capsys):
    from jevh.cli import Printer
    p = Printer()
    p("interject", {"outcome": "corrected", "kind": "correction", "text": "use km"})
    p("interject", {"outcome": "queued", "kind": "separate", "text": "milk", "queue": ["milk"]})
    p("interject", {"outcome": "status", "kind": "status", "text": "?", "answer": "daily_assistant is working on it."})
    p("aside", {"source": "model", "answer": "It is reading your inbox.", "seconds": 4.2})
    p("aside", {"source": "state", "answer": "the instant line again"})    # not repeated
    p("steer", {"step": 3, "text": "use km"})
    p("redirect", {"step": 3, "kept": 40})
    p("cancel", {"workers": 1, "dropped": ["milk"]})
    out = " ".join(capsys.readouterr().out.split())
    for want in ("added to the running task", "queued", "working on it", "reading your inbox", "heard you at step 3",
                 "cut the model's reply", "dropped 1 waiting message"):
        assert want in out, want
    assert "the instant line again" not in out


# -- the request as it stands -------------------------------------------------------------------------------------

def test_a_replacement_stands_alone_and_other_messages_add_to_the_request():
    from jevh.bus import Task
    t = Task(1, None, "user", "daily_assistant", "write fib.py")
    assert t.current == "write fib.py"
    t.said.append(("also print it twice", False))
    assert t.current.startswith("write fib.py") and "also print it twice" in t.current and "it wins" in t.current
    t.said.append(("check my calendar instead", True))          # a swap: what came before no longer counts
    assert t.current == "check my calendar instead"
    t.said.append(("only tomorrow", False))
    assert t.current.startswith("check my calendar instead") and "write fib.py" not in t.current


def test_the_gate_asks_for_the_file_the_latest_message_names():
    from jevh.verify import answer_rules
    ok = {"claims": 0.0, "delivers": 1.0, "honest": 0.0, "accurate": 1.0}

    def t(name):
        return 0.8
    req = "Write fib.py and run it.\n\n[The user then said: \"make it primes instead, in primes.py\". It wins.]"
    asked = answer_rules(req, "Done.", ["/x/primes.py"], "", ok, t)
    assert any("fib.py" in p for p in asked)                                  # both names are in the request text...
    latest = answer_rules(req, "Done.", ["/x/primes.py"], "", ok, t, files_from="make it primes instead, in primes.py")
    assert not any("fib.py" in p for p in latest)                             # ...but only the latest message's is due
    assert any("primes.py" in p for p in answer_rules(req, "Done.", ["/x/fib.py"], "", ok, t,
                                                       files_from="make it primes instead, in primes.py"))


def test_a_clear_swap_replaces_the_request_for_the_worker_and_the_gate(cfg):
    events = []
    h, llm, _ = harness(cfg, [SLEEPY, py("final('calendar done')")], {"check my calendar instead": "replace"},
                        events=events)
    t, out = run_turn(h, "check my email")
    assert wait_for(lambda: any(k == "code" for k, _ in events))
    res = h.interject("check my calendar instead")
    assert res["outcome"] == "corrected" and res["replaces"] is True
    t.join(30)
    task = out["r"].tasks[0]
    assert task.current == "check my calendar instead" and task.text == "check my email"
    said = llm.seen[1][-1]["content"]
    assert "replaces the request" in said and "check my calendar instead" in said
    assert next(d for k, d in events if k == "steer")["replaces"] is True


# -- asking again after a cut --------------------------------------------------------------------------------------

class Timed(Model):
    """Records when each call began and when the cut one came back."""

    def __init__(self, replies):
        super().__init__(replies)
        self.began, self.returned = [], []

    def chat(self, messages, cancel=None, interrupt=None):
        self.began.append(time.time())
        out = super().chat(messages, cancel, interrupt)
        self.returned.append(time.time())
        return out


def test_the_retry_waits_a_moment_for_the_server_to_free_the_slot(tmp_path):
    llm = Timed(["SLOW", py("final('done')")])
    w, _ = make_worker(tmp_path, llm)
    w.retry_delay = 0.6
    t, out = run_in_thread(w)
    assert llm.started.wait(10)
    w.steer("change course")
    t.join(20)
    assert out["r"].answer == "done"
    assert llm.began[1] - llm.returned[0] >= 0.55           # the second ask came after the pause, not at once


def test_a_stop_ends_the_wait_before_the_retry(tmp_path):
    llm = Timed(["SLOW", py("final('never')")])
    w, _ = make_worker(tmp_path, llm)
    w.retry_delay = 30
    caught = []

    def go():
        try:
            w.run("check my email", {}, "system")
        except Cancelled as e:
            caught.append(e)
    t = threading.Thread(target=go)
    t.start()
    assert llm.started.wait(10)
    w.steer("change course")
    assert wait_for(lambda: llm.returned)
    stopped = time.time()
    w.cancel.set()
    t.join(10)
    assert caught and time.time() - stopped < 5 and len(llm.began) == 1       # it never asked again
