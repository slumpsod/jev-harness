"""config.local.yaml, the hosted Jev and another machine's models (setting the harness up elsewhere)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from jevh.config import load_config, merged
from jevh.jev import Jev
from jevh.orchestrator import Harness

from conftest import FakeSystemOne


def test_local_config_is_laid_over_the_shared_one(tmp_path, monkeypatch):
    monkeypatch.delenv("JEVH_NO_LOCAL_CONFIG", raising=False)
    (tmp_path / "config.yaml").write_text("jev: {url: http://127.0.0.1:33001, backend: semif, backend_health: x}\n"
                                          "tools: {tools_jev: local, browser_watch: false}\n", encoding="utf-8")
    (tmp_path / "config.local.yaml").write_text("jev: {url: https://api.typesafe.ai, backend_health: null}\n"
                                                "tools: {tools_jev: cloud}\n", encoding="utf-8")
    c = load_config(tmp_path / "config.yaml")
    assert c.jev == {"url": "https://api.typesafe.ai", "backend": "semif"}      # merged, null removed
    assert c.tools == {"tools_jev": "cloud", "browser_watch": False}
    monkeypatch.setenv("JEVH_NO_LOCAL_CONFIG", "1")
    assert load_config(tmp_path / "config.yaml").jev["url"] == "http://127.0.0.1:33001"


def test_merged_leaves_its_inputs_alone():
    base = {"a": {"b": 1}}
    assert merged(base, {"a": {"c": 2}}) == {"a": {"b": 1, "c": 2}} and base == {"a": {"b": 1}}


def test_the_hosted_jev_gets_the_key_and_takes_big_choices(cfg, monkeypatch):
    seen = []

    class Api(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.headers.get("Authorization"), body))
            opts = body["questions"]["pick"]["criteria"]
            out = {"answers": {"pick": {"type": "choice", "choice": next(iter(opts)), "confidence": 1.0,
                                        "probabilities": {k: 1.0 / len(opts) for k in opts}}}}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Api)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cfg.jev = {"url": f"http://127.0.0.1:{srv.server_address[1]}", "backend": "jev-latest",
                   "api_key_env": "TEST_JEV_KEY", "choice_max": 240}
        monkeypatch.delenv("TEST_JEV_KEY", raising=False)
        with pytest.raises(RuntimeError, match="TEST_JEV_KEY"):
            Jev(cfg)                                          # a clear message, not a 401 mid-turn
        monkeypatch.setenv("TEST_JEV_KEY", "k-123")
        probs = Jev(cfg).choice_probs("t", {}, "which?", {f"o{i}": f"option {i}" for i in range(40)})
        assert len(seen) == 1                                 # 40 options in one call (semif: rounds of 16)
        assert seen[0][0] == "Bearer k-123" and seen[0][1]["model"] == "jev-latest"
        assert len(probs) == 40
    finally:
        srv.shutdown()


def test_models_strong_puts_every_persona_on_this_machines_model(cfg):
    cfg.models = {"fast": {"provider": "lmstudio", "model": "small-mac-model"},
                  "strong": {"provider": "lmstudio", "model": "big-mac-model"}}
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne()))
    p = h.personas["daily_assistant"]
    strong, fast = h._model_cfg(p, "strong"), h._model_cfg(p, "fast")
    assert strong["model"] == "big-mac-model" and strong.get("think") == p.model.get("think")   # its settings kept
    assert "fallback" not in strong
    assert fast["model"] == "small-mac-model"

