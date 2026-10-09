import threading
import time

from jevh.lmstudio import ExclusiveModels, gate_for


class FakeLMStudio(ExclusiveModels):
    """In-memory LM Studio: records every load/unload and the max models ever resident."""

    def __init__(self, loaded=(), linger=0.0, max_wait=180.0):
        super().__init__("http://fake", linger=linger, max_wait=max_wait)
        self.state = {k: k for k in loaded}  # key -> instance id
        self.log: list[str] = []
        self.max_resident = len(self.state)
        self.lock = threading.Lock()

    def _list(self):
        with self.lock:
            keys = {"ornith", "qwen3.8-27b"} | set(self.state)
            return [{"key": k, "type": "llm", "loaded_instances": [{"id": self.state[k]}] if k in self.state else []}
                    for k in sorted(keys)] + [{"key": "nomic-embed", "type": "embedding",
                                               "loaded_instances": [{"id": "nomic-embed"}]}]

    def _unload(self, instance_id):
        time.sleep(0.02)
        with self.lock:
            self.state = {k: i for k, i in self.state.items() if i != instance_id}
            self.log.append(f"unload {instance_id}")

    def _load(self, model):
        time.sleep(0.05)
        with self.lock:
            self.state[model] = model
            self.max_resident = max(self.max_resident, len(self.state))
            self.log.append(f"load {model}")


def test_swap_unloads_before_loading():
    lm = FakeLMStudio(loaded=["ornith"])
    with lm.use("qwen3.8-27b@q6_k"):
        pass
    assert lm.log == ["unload ornith", "load qwen3.8-27b"]  # variant suffix resolved to the key
    assert lm.max_resident == 1


def test_already_loaded_model_is_a_no_op():
    lm = FakeLMStudio(loaded=["ornith"])
    with lm.use("ornith"):
        pass
    assert lm.log == []


def test_embedding_models_are_left_alone():
    lm = FakeLMStudio(loaded=["ornith"])
    with lm.use("qwen3.8-27b"):
        pass
    assert "unload nomic-embed" not in lm.log


def test_parallel_same_model_requests_overlap():
    lm = FakeLMStudio(loaded=["ornith"])
    inside, peak = [0], [0]

    def work():
        with lm.use("ornith"):
            inside[0] += 1
            peak[0] = max(peak[0], inside[0])
            time.sleep(0.2)
            inside[0] -= 1

    ts = [threading.Thread(target=work) for _ in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert peak[0] == 3 and lm.log == []


def test_mixed_models_never_resident_together_and_never_swap_mid_request():
    lm = FakeLMStudio(loaded=["ornith"])
    active: list[str] = []
    violations = []

    def work(model):
        with lm.use(model):
            with lm.lock:
                resident = set(lm.state) - {"nomic-embed"}
            if resident != {model.split("@")[0]}:
                violations.append((model, resident))
            active.append(model)
            time.sleep(0.05)
            active.remove(model)

    ts = [threading.Thread(target=work, args=(m,)) for m in ["ornith", "qwen3.8-27b", "ornith", "qwen3.8-27b"] * 3]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert violations == []
    assert lm.max_resident == 1


def test_gate_is_shared_per_server():
    assert gate_for("http://127.0.0.1:1234/v1") is gate_for("http://127.0.0.1:1234")


def test_linger_keeps_model_between_a_workers_steps():
    """Worker A: 3 ornith calls with 0.1s REPL gaps. Worker B wants qwen meanwhile.
    With a linger, A finishes all its steps before the single swap - no ping-pong."""
    lm = FakeLMStudio(loaded=["ornith"], linger=0.5)
    order = []

    def worker_a():
        for i in range(3):
            with lm.use("ornith"):
                order.append(f"A{i}")
                time.sleep(0.05)
            time.sleep(0.1)  # REPL step between LLM calls

    def worker_b():
        time.sleep(0.02)
        with lm.use("qwen3.8-27b"):
            order.append("B")

    ta, tb = threading.Thread(target=worker_a), threading.Thread(target=worker_b)
    ta.start(); tb.start(); ta.join(); tb.join()
    assert order == ["A0", "A1", "A2", "B"]
    assert lm.log == ["unload ornith", "load qwen3.8-27b"]  # exactly one swap


def test_max_wait_stops_a_busy_model_starving_others():
    lm = FakeLMStudio(loaded=["ornith"], linger=5.0, max_wait=0.3)
    stop = threading.Event()

    def busy():
        while not stop.is_set():
            with lm.use("ornith"):
                time.sleep(0.02)
            time.sleep(0.01)

    t = threading.Thread(target=busy); t.start()
    time.sleep(0.05)
    started = time.time()
    with lm.use("qwen3.8-27b"):
        waited = time.time() - started
    stop.set(); t.join()
    assert 0.25 < waited < 2.0 and lm.max_resident == 1


def test_an_idle_gate_asks_again_what_another_process_loaded():
    lm = FakeLMStudio(loaded=["qwen3.8-27b"])
    lm.recheck = 0.05
    with lm.use("qwen3.8-27b@q6_k"):
        pass
    lm.state = {"ornith": "ornith"}   # another process swapped (live: refine's test runs)
    with lm.use("qwen3.8-27b@q6_k"):   # still within recheck: trusts its belief
        pass
    assert lm.log == []
    time.sleep(0.06)
    with lm.use("qwen3.8-27b@q6_k"):
        pass
    assert lm.log == ["unload ornith", "load qwen3.8-27b"]
    time.sleep(0.06)
    with lm.use("qwen3.8-27b@q6_k"):   # asked again, and it's there: nothing to do
        pass
    assert lm.log == ["unload ornith", "load qwen3.8-27b"]


def test_a_stop_ends_the_wait_for_another_models_turn():
    """A hand-off waiting for the GPU (another model's request in flight) held a Stop until that request ended - up to
    max_wait, or a whole swap - and the turn could not end before it."""
    import pytest
    from jevh.errors import Cancelled
    lm = FakeLMStudio(loaded=["ornith"])
    holding, release = threading.Event(), threading.Event()

    def busy():
        with lm.use("ornith"):
            holding.set()
            release.wait(10)
    threading.Thread(target=busy, daemon=True).start()
    assert holding.wait(5)
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t = time.time()
    with pytest.raises(Cancelled):
        with lm.use("qwen3.8-27b", cancel):
            pass
    assert time.time() - t < 2
    release.set()
    assert "load qwen3.8-27b" not in lm.log      # it never swapped


def test_a_failed_swap_forgets_which_model_it_thought_was_loaded():
    """The swap unloads first: when the load then failed, the gate still believed the old model loaded, and the next
    request for it went straight through to a model that was gone."""
    import pytest

    class NoLoad(FakeLMStudio):
        def _load(self, model):
            raise RuntimeError("LM Studio /api/v1/models/load: HTTP 404 model not found")
    lm = NoLoad(loaded=["ornith"])
    with lm.use("ornith"):
        pass
    with pytest.raises(RuntimeError):
        with lm.use("qwen3.8-27b"):
            pass
    assert lm.current is None and lm.state == {}   # ornith was unloaded; the gate asks LM Studio next time
