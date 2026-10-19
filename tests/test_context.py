"""A model's context window: asked of LM Studio, or set in config, and the compaction limit made of it (jevh/context.py).

tests/lmstudio_models.json is a real LM Studio answer (GET /api/v1/models, 2026-10-04, trimmed): Qwen3.8 27B loaded with
100096 tokens of a possible 262144 and 2 parallel slots; Gemma 4 and Ornith not loaded; an embedding model."""
from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from jevh import cli
from jevh import context as ctxwin
from jevh import lmstudio
from jevh.config import ROOT
from jevh.context import Window, budget, compact_tokens
from jevh.jev import Jev
from jevh.llm import OpenAICompat, WithFallback, make_llm
from jevh.lmstudio import ExclusiveModels, info_from
from jevh.orchestrator import Harness

from conftest import FakeSystemOne, ScriptedLLM, choice_ans, py

REAL = json.loads((Path(__file__).parent / "lmstudio_models.json").read_text(encoding="utf-8"))["models"]
QWEN = Window(100096, "lmstudio", True, 262144, 2)


def entry(key="m", instances=(), max_=8192, kind="llm"):
    """A model as /api/v1/models lists it; instances: [(instance id, config)]."""
    return {"type": kind, "key": key, "max_context_length": max_,
            "loaded_instances": [{"id": i, "config": c} for i, c in instances]}


class Server(ExclusiveModels):
    """LM Studio's REST, answered from a list; records every request (method, path, timeout)."""

    def __init__(self, models=REAL, **kw):
        super().__init__("http://fake", **kw)
        self.models, self.requests, self.down = copy.deepcopy(models), [], False

    def _request(self, method, path, body=None, timeout=30.0):
        self.requests.append((method, path, timeout))
        if self.down:
            raise OSError("connection refused")
        return {"models": copy.deepcopy(self.models)}


# -- reading LM Studio's answer ------------------------------------------------------------------------------------

def test_a_loaded_model_reports_what_it_is_loaded_with_not_what_it_supports():
    got = info_from(REAL, "qwen3.8-27b@q6_k")   # how the harness spells the model
    assert got == {"key": "qwen3.8-27b", "loaded": True, "context": 100096, "max": 262144, "parallel": 2}
    assert info_from(REAL, "qwen3.8-27b") == got


def test_a_model_that_is_not_loaded_has_no_context_yet():
    assert info_from(REAL, "ornith-1.5-35b-a3b") == {"key": "ornith-1.5-35b-a3b", "loaded": False, "context": None,
                                                     "max": 262144, "parallel": None}


def test_an_unlisted_model_and_an_embedding_model_are_not_chat_models():
    assert info_from(REAL, "no-such-model") is None
    assert info_from(REAL, "text-embedding-nomic-embed-text-v1.5") is None   # (even loaded, one is never chatted with)
    assert info_from([], "qwen3.8-27b") is None


def test_two_loaded_instances_give_the_smaller_context_and_its_parallel_slots():
    two = [entry(instances=[("m", {"context_length": 32768, "parallel": 2}), ("m:2", {"context_length": 8192, "parallel": 4})])]
    assert info_from(two, "m") == {"key": "m", "loaded": True, "context": 8192, "max": 8192, "parallel": 4}


def test_two_instances_with_the_same_context_but_other_settings_are_not_a_crash():
    # (sorting (context, config) pairs compared the config dicts once the contexts tied: a TypeError)
    same = [entry(instances=[("m", {"context_length": 4096, "parallel": 1}), ("m:2", {"context_length": 4096, "parallel": 2})])]
    got = info_from(same, "m")
    assert got["context"] == 4096 and got["parallel"] == 1   # (the first listed of equals)


def test_a_model_is_found_by_its_instance_id_too():
    assert info_from([entry("some/key", [("my-alias", {"context_length": 5000})])], "my-alias")["context"] == 5000


def test_an_older_lm_studio_without_the_config_is_loaded_but_unknown():
    old = [{"type": "llm", "key": "m", "loaded_instances": [{"id": "m"}]}]
    assert info_from(old, "m") == {"key": "m", "loaded": True, "context": None, "max": None, "parallel": None}


@pytest.mark.parametrize("junk", [True, 0, -5, "100k", None])
def test_a_context_that_is_not_a_positive_number_is_ignored(junk):
    assert info_from([entry(instances=[("m", {"context_length": junk})])], "m")["context"] is None


# -- asking: cached, short, never behind a swap -----------------------------------------------------------------------

def test_lm_studio_is_asked_once_and_briefly_per_model_within_ten_seconds():
    lm = Server()
    assert lm.model_info("qwen3.8-27b@q6_k")["context"] == 100096
    assert lm.model_info("qwen3.8-27b")["context"] == 100096   # the same model, spelled another way
    assert lm.requests == [("GET", "/api/v1/models", lm.info_timeout)] and lm.info_timeout <= 5


def test_a_model_reloaded_with_another_context_is_noticed():
    lm = Server()
    lm.info_ttl = 0.05
    assert lm.model_info("qwen3.8-27b")["context"] == 100096
    next(m for m in lm.models if m["key"] == "qwen3.8-27b")["loaded_instances"][0]["config"]["context_length"] = 50000
    assert lm.model_info("qwen3.8-27b")["context"] == 100096      # (still the answer from a moment ago)
    time.sleep(0.08)
    assert lm.model_info("qwen3.8-27b")["context"] == 50000


def test_when_lm_studio_cannot_be_asked_the_last_answer_stands_for_a_while_then_nothing():
    lm = Server()
    lm.info_ttl, lm.info_stale = 0.03, 0.3
    assert lm.model_info("qwen3.8-27b")["context"] == 100096
    lm.down = True
    time.sleep(0.06)
    assert lm.model_info("qwen3.8-27b")["context"] == 100096      # a hiccup does not change the limit mid-task
    time.sleep(0.35)
    assert lm.model_info("qwen3.8-27b") is None                    # too old to trust


def test_a_server_that_was_never_reachable_gives_none_not_an_error():
    lm = Server()
    lm.down = True
    assert lm.model_info("qwen3.8-27b") is None


def test_detection_off_never_asks():
    lm = Server()
    lm.detect = False
    assert lm.model_info("qwen3.8-27b") is None and lm.requests == []


def test_asking_never_waits_behind_a_model_swap():
    lm, got = Server(), []
    with lm._cond:   # a swap holds the gate's lock, and can for minutes
        t = threading.Thread(target=lambda: got.append(lm.model_info("qwen3.8-27b")))
        t.start()
        t.join(5)
        assert not t.is_alive() and got and got[0]["context"] == 100096


def test_every_llm_is_listed_for_the_status_card_and_jevh_context():
    got = {m["key"]: m for m in Server().llms_info()}
    assert set(got) == {"google/gemma-4-12b-qat", "qwen3.8-27b", "ornith-1.5-35b-a3b"}   # (no embedding model)
    assert got["qwen3.8-27b"]["context"] == 100096 and got["ornith-1.5-35b-a3b"]["loaded"] is False


# -- the model client's window ------------------------------------------------------------------------------------

def client(gate=None, model="qwen3.8-27b@q6_k", **kw):
    return OpenAICompat("http://127.0.0.1:1234/v1", model, gate=gate, **kw)


def test_the_client_reports_the_window_lm_studio_loaded_its_model_with():
    assert client(Server()).window() == QWEN


def test_config_wins_over_asking_and_nothing_is_asked():
    lm = Server()
    assert client(lm, context_length=32768).window() == Window(32768, "config") and lm.requests == []


def test_a_model_lm_studio_has_not_loaded_has_a_pending_window():
    assert client(Server(), model="ornith-1.5-35b-a3b").window() == Window(None, "lmstudio", False, 262144, None)


def test_no_gate_means_no_window_unless_config_says():
    assert client().window() is None
    assert client(context_length=8192).window() == Window(8192, "config")


def test_a_model_lm_studio_does_not_know_and_a_server_that_is_down_have_no_window():
    assert client(Server(), model="typo-model").window() is None
    down = Server()
    down.down = True
    assert client(down).window() is None


class Knows:
    name = "knows"

    def __init__(self, w):
        self.w = w

    def window(self):
        return self.w


class Blind:   # a hosted model nothing is asked about
    name = "blind"


def test_a_fallback_pair_has_the_smaller_window_since_a_failed_call_is_repeated_on_the_other():
    assert WithFallback(lambda: Knows(Window(100096, "lmstudio")), Knows(Window(32768, "config"))).window().tokens == 32768
    assert WithFallback(lambda: Knows(Window(8192, "config")), Knows(Window(32768, "config"))).window().tokens == 8192


def test_a_hosted_fallback_does_not_hide_the_local_models_window():
    assert WithFallback(lambda: Knows(QWEN), Blind()).window() == QWEN
    assert WithFallback(lambda: Blind(), Knows(QWEN)).window() == QWEN
    assert WithFallback(lambda: Blind(), Blind()).window() is None


def test_a_model_not_loaded_yet_is_not_sized_by_its_fallback():
    pending = Window(None, "lmstudio", False, 262144)
    got = WithFallback(lambda: Knows(pending), Knows(Window(200000, "config"))).window()
    assert got == pending   # the fallback's 200000 says nothing about what LM Studio will load it with


def test_a_primary_that_cannot_be_built_leaves_the_fallbacks_window():
    def broken():
        raise RuntimeError("no SDK")
    assert WithFallback(broken, Knows(QWEN)).window() == QWEN


PROVIDERS = {"lm": {"type": "openai", "base_url": "http://127.0.0.1:1234/v1", "exclusive": True},
             "vllm": {"type": "openai", "base_url": "http://127.0.0.1:8000/v1", "context_length": 32768}}


def test_context_length_comes_from_the_model_block_or_its_provider():
    assert make_llm({"provider": "lm", "model": "m"}, PROVIDERS).context_length is None
    assert make_llm({"provider": "lm", "model": "m", "context_length": 50000}, PROVIDERS).context_length == 50000
    assert make_llm({"provider": "vllm", "model": "m"}, PROVIDERS).window() == Window(32768, "config")
    assert make_llm({"provider": "vllm", "model": "m", "context_length": 8192}, PROVIDERS).window() == Window(8192, "config")


# -- what the harness does with a window ------------------------------------------------------------------------------

@pytest.mark.parametrize("window,reply,want", [
    (100096, 8192, 60057),    # 60% of it: limits.compact_tokens (60000) was written for exactly this window
    (262144, 8192, 157286),
    (32768, 8192, 19660),
    (16384, 8192, 8192),      # the reply's room binds before the share does
    (100096, None, 60057),    # may think without limit: a quarter of the window is kept free
    (100096, 65536, 50048),   # a big reply: at most half the window is kept free for it
    (4096, 8192, 2048),       # nothing the harness can run on: the floor
])
def test_older_steps_condense_at_a_share_of_the_window_that_leaves_the_reply_room(window, reply, want):
    assert compact_tokens(window, reply, 0.6) == want


def test_a_known_window_sets_the_limit_and_an_unknown_one_leaves_compact_tokens():
    b = budget(QWEN, 8192, 60000, ctxwin.DEFAULTS)
    assert (b.compact, b.window, b.source, b.max, b.parallel, b.warn) == (80076, 100096, "lmstudio", 262144, 2, "")
    pending = budget(Window(None, "lmstudio", False, 262144), 8192, 60000, ctxwin.DEFAULTS)
    assert (pending.compact, pending.window, pending.loaded) == (60000, None, False)


def test_a_window_too_small_for_the_harness_is_flagged_with_what_to_do():
    small = budget(Window(4096, "lmstudio"), 8192, 60000, ctxwin.DEFAULTS)
    assert small.compact == 2048 and "4,096 tokens" in small.warn and "load it with more in LM Studio" in small.warn
    assert "use a model with more" in budget(Window(4096, "config"), 8192, 60000, ctxwin.DEFAULTS).warn
    assert budget(Window(16384, "config"), 8192, 60000, ctxwin.DEFAULTS).warn == ""


def test_the_settings_are_the_shipped_ones_unless_changed():
    assert ctxwin.settings(None) == ctxwin.DEFAULTS == {
        "detect": True, "compact_at": 0.8, "min": 16384, "measure": True, "archive": True, "log_share": 0.25, "note_tail": True}
    assert ctxwin.settings({"context": {"compact_at": 0.5}})["compact_at"] == 0.5
    assert budget(QWEN, 8192, 60000, ctxwin.settings({"context": {"compact_at": 0.5}})).compact == 50048


# -- the worker follows it ---------------------------------------------------------------------------------------

def rules(k, q, s):
    if k == "persona":
        return choice_ans("daily_assistant", 0.9, q["criteria"])


class WindowLLM(ScriptedLLM):
    """A scripted model that knows its window, as OpenAICompat.window() tells it."""

    def __init__(self, replies, window, max_tokens=8192):
        super().__init__(replies)
        self.win, self.max_tokens = window, max_tokens

    def window(self):
        return self.win(self) if callable(self.win) else self.win


def steps(n):
    return [py(f"print('step {i} done'); print('y' * 3000)") for i in range(n)] + [py("final('all done')")]


def run(cfg, replies, window, **settings):
    events = []
    cfg.limits.update(keep_steps=2)
    # (a 6000-token window is not what is under test; nor is `measure`: a scripted model's token count is a placeholder,
    # so these size the prompt by the estimate)
    cfg.raw = {**cfg.raw, "context": {"min": 1000, "measure": False, **settings}}
    llm = WindowLLM(replies, window)
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: llm,
                on_event=lambda k, d: events.append((k, d)))
    h.learning = False
    res = h.turn("do a long job")
    return [d for k, d in events if k == "context"], [d for k, d in events if k == "compact"], res


def test_compaction_follows_the_window_lm_studio_loaded_the_model_with_not_compact_tokens(cfg):
    ctx, compacted, res = run(cfg, steps(8), Window(6000, "lmstudio", True, 262144, 2))
    assert cfg.lim("compact_tokens") == 60000 and compacted           # 60000 would never have compacted this
    assert {c["limit"] for c in ctx} == {3000} and {c["window"] for c in ctx} == {6000}
    assert ctx[0]["source"] == "lmstudio" and ctx[0]["max"] == 262144 and ctx[0]["parallel"] == 2 and ctx[0]["warn"] == ""
    assert res.answer == "all done"


def test_the_context_event_carries_the_warning_for_a_window_that_is_too_small(cfg):
    ctx, _, _ = run(cfg, steps(1), Window(6000, "lmstudio"), min=16384)
    assert "6,000 tokens of context is little" in ctx[0]["warn"]


def test_an_unknown_window_leaves_compact_tokens_and_the_event_without_one(cfg):
    ctx, compacted, _ = run(cfg, steps(8), None)
    assert not compacted and {c["limit"] for c in ctx} == {60000} and all("window" not in c for c in ctx)


def test_a_model_not_loaded_yet_leaves_compact_tokens_and_says_so(cfg):
    ctx, _, _ = run(cfg, steps(2), Window(None, "lmstudio", False, 262144))
    assert {c["limit"] for c in ctx} == {60000} and ctx[0]["window"] is None and ctx[0]["loaded"] is False


def test_a_model_reloaded_with_a_smaller_context_mid_task_condenses_from_then_on(cfg):
    ctx, compacted, _ = run(cfg, steps(10), lambda llm: Window(100096 if len(llm.seen) < 4 else 6000, "lmstudio"))
    assert [c["limit"] for c in ctx[:3]] == [80076] * 3 and ctx[-1]["limit"] == 3000
    assert compacted and min(c["step"] for c in compacted) > 3        # nothing condensed while the window was big


def test_a_failing_question_about_the_window_never_fails_a_step(cfg):
    def explode(llm):
        raise RuntimeError("LM Studio fell over")
    ctx, _, res = run(cfg, steps(2), explode)
    assert res.answer == "all done" and {c["limit"] for c in ctx} == {60000}


def test_the_limit_follows_the_model_the_worker_uses_now(cfg):
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: ScriptedLLM([]))
    worker = SimpleNamespace(llm=WindowLLM([], QWEN))
    fn = h._budget_fn(worker)
    assert fn().compact == 80076                                       # 80% of 100,096 (context.compact_at)
    worker.llm = WindowLLM([], Window(32768, "lmstudio"))              # the fast model escalated to the strong one
    assert fn().compact == 24576                                       # 80% would leave less than its 8,192 reply
    worker.llm = ScriptedLLM([])                                       # a model nothing can be asked about
    assert fn() is None


# -- jevh context -------------------------------------------------------------------------------------------------

@pytest.fixture
def lm_studio(monkeypatch):
    """The LM Studio the shipped config points at (http://127.0.0.1:1234), answered from the real saved list."""
    lm = Server()
    monkeypatch.setitem(lmstudio._GATES, "http://127.0.0.1:1234", lm)
    return lm


def test_the_report_says_what_each_model_is_loaded_with_and_when_it_condenses(cfg, lm_studio):
    r = ctxwin.report(cfg)
    qwen = next(m for m in r["models"] if m["model"] == "lmstudio:qwen3.8-27b@q6_k")
    assert (qwen["window"], qwen["source"], qwen["compact"], qwen["max"], qwen["parallel"]) == (100096, "lmstudio", 75072, 262144, 2)
    assert "high_end_dev" in qwen["used_by"]
    fast = next(m for m in r["models"] if m["model"] == "lmstudio:ornith-1.5-35b-a3b")
    assert fast["window"] is None and fast["loaded"] is False and fast["compact"] == cfg.lim("compact_tokens")
    assert fast["used_by"] == ["fast tier"]
    assert [m["key"] for m in r["lmstudio"][0]["models"]] == ["google/gemma-4-12b-qat", "qwen3.8-27b", "ornith-1.5-35b-a3b"]


def test_the_report_reads_as_plain_text(cfg, lm_studio):
    text = ctxwin.render(ctxwin.report(cfg))
    assert "qwen3.8-27b" in text and "loaded with   100,096" in text and "supports   262,144" in text
    assert "window 100,096 (asked LM Studio)  -> condenses at 75,072" in text   # (thinks freely: a quarter kept)
    assert "window not known: LM Studio has not loaded it yet  -> condenses at 60,000" in text
    assert "not loaded" in text


def test_a_window_set_in_config_is_reported_as_such_and_beats_lm_studio(cfg, lm_studio):
    cfg.models["fast"] = {**cfg.models["fast"], "context_length": 32768}
    fast = next(m for m in ctxwin.report(cfg)["models"] if m["model"] == "lmstudio:ornith-1.5-35b-a3b")
    assert (fast["window"], fast["source"], fast["compact"]) == (32768, "config", 24576)
    assert "(from config)" in ctxwin.render(ctxwin.report(cfg))


def test_detect_false_never_asks_lm_studio_for_a_window(cfg, lm_studio):
    cfg.raw = {**cfg.raw, "context": {"detect": False}}
    r = ctxwin.report(cfg)
    assert all(m["window"] is None and m["compact"] == cfg.lim("compact_tokens") for m in r["models"])
    assert "not asked" in ctxwin.render(r)
    assert not [q for q in lm_studio.requests if q[2] == lm_studio.info_timeout]   # (the model list itself is still shown)


def test_an_lm_studio_that_is_down_is_a_line_not_a_traceback(cfg, lm_studio):
    lm_studio.down = True
    r = ctxwin.report(cfg)
    assert "connection refused" in r["lmstudio"][0]["error"] and all(m["window"] is None for m in r["models"])
    assert "could not be asked" in ctxwin.render(r)


def test_the_command_prints_text_or_json_and_can_look_at_one_model(lm_studio, capsys):
    assert ctxwin.main(["--config", str(ROOT / "config.yaml")]) == 0
    assert "condenses at 75,072" in capsys.readouterr().out
    assert ctxwin.main(["--config", str(ROOT / "config.yaml"), "--json", "--model", "lmstudio:qwen3.8-27b@q6_k"]) == 0
    got = json.loads(capsys.readouterr().out)
    assert [m["window"] for m in got["models"]] == [100096] and got["models"][0]["used_by"] == ["(asked for)"]
    assert cli.subcommand("context") is ctxwin.main


# -- where it shows ---------------------------------------------------------------------------------------------

def test_the_dashboard_status_lists_each_loaded_model_with_its_context(cfg, monkeypatch):
    from fastapi.testclient import TestClient

    from jevh.web import server
    from test_web import BASE, TOKEN, auth, rules as web_rules

    lm = Server()
    monkeypatch.setattr(server, "gate_for", lambda url: lm)
    monkeypatch.setattr(server, "gpus", lambda: [])
    monkeypatch.setattr(server, "probe", lambda url, timeout=2.0: {"up": False})

    def factory(on_event):
        return Harness(cfg, jev=Jev(cfg, FakeSystemOne(web_rules)), llm_factory=lambda p: ScriptedLLM([]), on_event=on_event)
    client = TestClient(server.create_app(cfg, port=8740, harness_factory=factory, token=TOKEN), base_url=BASE)
    got = client.get("/api/status", headers=auth()).json()["lmstudio"]
    assert got["up"] and got["loaded"] == ["qwen3.8-27b"] and got["contexts"] == {"qwen3.8-27b": 100096}


def test_the_cli_says_a_window_too_small_for_the_harness_once_and_a_good_one_only_when_verbose():
    ev = {"persona": "high_end_dev", "step": 1, "limit": 2048, "window": 4096, "source": "lmstudio",
          "warn": "4,096 tokens of context is little for the harness", "parts": {}}
    good = {"persona": "p", "step": 1, "limit": 60057, "window": 100096, "source": "lmstudio", "warn": ""}
    quiet = cli.Printer()
    with cli.console.capture() as cap:
        quiet("context", ev)
        quiet("context", {**ev, "step": 2})
        quiet("context", good)
    assert cap.get().count("4,096 tokens of context") == 1 and "100,096" not in cap.get()
    loud = cli.Printer(verbose=True)
    with cli.console.capture() as cap:
        loud("context", good)
        loud("context", {**good, "step": 2})
    assert cap.get().count("context window 100,096 tokens (asked LM Studio)") == 1


def test_the_shipped_config_sets_the_context_block_and_keeps_compact_tokens_as_the_fallback(cfg):
    s = ctxwin.settings(cfg.raw)
    assert s == ctxwin.DEFAULTS and 0 < s["compact_at"] < 1
    assert cfg.providers["lmstudio"]["exclusive"] and cfg.lim("compact_tokens") == 60000


# ---- a llama.cpp server is asked what it was started with (GET /props) ----------------------------------------------

@pytest.fixture
def llama_server():
    """A stand-in llama-server: /props as b11451 answers it (trimmed), and /v1 beside it."""
    import http.server
    props = {"default_generation_settings": {"n_ctx": 76032}, "total_slots": 2, "model_alias": "qwen3.8-27b",
             "model_path": "C:\\models\\Qwen3.8-27B-UD-Q5_K_M.gguf", "build_info": "b11451-2207c8e57"}
    asked = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            asked.append(self.path)
            body = json.dumps(props).encode() if self.path == "/props" else b'{"error": "Unexpected endpoint"}'
            self.send_response(200 if self.path == "/props" else 404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    ctxwin._PROPS.clear()
    yield SimpleNamespace(url=f"http://127.0.0.1:{srv.server_address[1]}/v1", asked=asked, props=props)
    srv.shutdown()
    ctxwin._PROPS.clear()


def test_a_llama_cpp_server_says_what_it_was_started_with_and_that_wins_over_config(llama_server):
    """Live (2026-10-07): the harness ran on llama-server -c 76000 (76,032 per slot) while the config said 76000 and
    the dashboard showed only LM Studio's 100,096."""
    info = ctxwin.server_props(llama_server.url)
    assert info == {"context": 76032, "slots": 2, "model": "qwen3.8-27b", "file": "Qwen3.8-27B-UD-Q5_K_M.gguf",
                    "build": "b11451-2207c8e57"}
    llm = OpenAICompat(llama_server.url, "qwen3.8-27b", context_length=76000)
    assert llm.window() == Window(76032, "llama.cpp", True, None, 2)
    llm.window()
    assert llama_server.asked.count("/props") == 1                            # reused for PROPS_TTL seconds
    ctxwin.DETECT["on"] = False                                                # context.detect: false -> config only
    try:
        ctxwin._PROPS.clear()
        assert llm.window() == Window(76000, "config")
    finally:
        ctxwin.DETECT["on"] = True


def test_a_server_that_is_not_llama_cpp_or_is_down_leaves_the_config_value(llama_server):
    llama_server.props["default_generation_settings"] = {}                    # answers, but says nothing useful
    assert OpenAICompat(llama_server.url, "m", context_length=32768).window() == Window(32768, "config")
    ctxwin._PROPS.clear()
    assert ctxwin.server_props("http://127.0.0.1:9/v1", timeout=0.5) is None   # nothing listens there
    assert OpenAICompat("http://127.0.0.1:9/v1", "m").window() is None
