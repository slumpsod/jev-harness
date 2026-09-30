"""The nango tool: the user's connected accounts through a (fake) Nango server, and the change gate."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from jevh.jev import Jev
from jevh.orchestrator import Harness
from jevh.tools import nango

from conftest import FakeSystemOne, ScriptedLLM, noul_ans, py
from test_cancel_ask import run_async, wait_for

CONNECTIONS = [{"id": 1, "connection_id": "conn-gm", "provider": "google-mail", "provider_config_key": "google-mail",
                "created": "2026-09-01T00:00:00Z", "errors": []},
               {"id": 2, "connection_id": "conn-gh", "provider": "github", "provider_config_key": "github-jed",
                "created": "2026-09-01T00:00:00Z", "errors": [{"type": "auth", "log_id": "x"}]}]
INTEGRATIONS = [{"unique_key": "google-mail", "provider": "google-mail", "display_name": "Gmail"},
                {"unique_key": "github-jed", "provider": "github", "display_name": "GitHub"},
                {"unique_key": "notion", "provider": "notion", "display_name": "Notion"}]


class FakeNango(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *a):
        pass

    def _reply(self, status, body):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _any(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n)) if n else None
        FakeNango.seen.append({"method": self.command, "path": self.path, "headers": dict(self.headers), "body": body})
        if self.headers.get("Authorization") != "Bearer sk-test":
            return self._reply(401, {"error": "bad key"})
        if self.path == "/connection":
            return self._reply(200, {"connections": CONNECTIONS})
        if self.path == "/integrations":
            return self._reply(200, {"data": INTEGRATIONS})
        if self.path == "/connect/sessions":
            return self._reply(201, {"data": {"token": "t", "connect_link": "https://connect.nango.dev?session_token=t",
                                              "expires_at": "2026-09-29T12:00:00Z"}})
        if self.path.startswith("/proxy/gmail/v1/users/me/messages/send"):
            return self._reply(200, {"id": "sent-1"})
        if self.path.startswith("/proxy/gmail/v1/users/me/messages"):
            return self._reply(200, {"messages": [{"id": "m1"}, {"id": "m2"}], "resultSizeEstimate": 2})
        if self.path.startswith("/proxy/user"):
            return self._reply(401, {"message": "Bad credentials"})
        return self._reply(404, {"error": "not found"})

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _any


@pytest.fixture
def fake_nango(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeNango)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    FakeNango.seen = []
    monkeypatch.setenv("NANGO_SECRET_KEY", "sk-test")
    monkeypatch.setenv("JEVH_NANGO_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    monkeypatch.delenv("JEVH_NANGO_WRITES", raising=False)
    nango._cache.clear()
    nango._allowed.clear()
    yield FakeNango.seen
    srv.shutdown()
    nango._cache.clear()


def proxied(seen):
    return [s for s in seen if s["path"].startswith("/proxy/")]


def test_a_read_goes_through_the_proxy_as_the_connected_account(fake_nango):
    out = nango.get("google-mail", "/gmail/v1/users/me/messages", {"q": "is:unread", "labelIds": ["INBOX", "UNREAD"]})
    assert out["resultSizeEstimate"] == 2
    call = proxied(fake_nango)[0]
    assert call["path"] == "/proxy/gmail/v1/users/me/messages?q=is%3Aunread&labelIds=INBOX&labelIds=UNREAD"
    h = {k.lower(): v for k, v in call["headers"].items()}
    assert h["connection-id"] == "conn-gm" and h["provider-config-key"] == "google-mail"
    assert h["authorization"] == "Bearer sk-test" and h["retries"] == "2"
    # the provider name finds an integration whose key differs ("github" -> "github-jed")
    with pytest.raises(nango.NangoError, match=r"HTTP 401 \(the github-jed connection may need reconnecting"):
        nango.get("github", "/user")


def test_reads_and_changes_are_told_apart():
    reads = [("GET", "/x", None), ("POST", "/v1/search", None), ("POST", "/v1/databases/abc/query", None),
             ("POST", "/api/conversations.history", None), ("POST", "/api/users.lookupByEmail", None),
             ("POST", "/graphql", {"query": "{ viewer { login } }"})]
    changes = [("POST", "/gmail/v1/users/me/messages/send", None), ("POST", "/api/chat.postMessage", None),
               ("DELETE", "/repos/o/r", None), ("PATCH", "/v1/pages/1", None),
               ("POST", "/graphql", {"query": "mutation { addStar }"}), ("POST", "/graphql", None)]
    assert all(nango.is_read(*r) for r in reads)
    assert not any(nango.is_read(*c) for c in changes)


def test_a_change_with_nobody_to_ask_is_not_sent(fake_nango):
    with pytest.raises(nango.Refused, match="no user here"):
        nango.post("google-mail", "/gmail/v1/users/me/messages/send", json={"raw": "..."})
    assert proxied(fake_nango) == []


def test_a_base_url_override_always_needs_the_user(fake_nango, monkeypatch):
    # Nango attaches the user's token for the override host: a read sent elsewhere could leak it
    monkeypatch.setenv("JEVH_NANGO_WRITES", "allow")
    with pytest.raises(nango.Refused, match=r"sent to https://evil.example"):
        nango.get("google-mail", "/gmail/v1/users/me/messages", base_url="https://evil.example")
    assert proxied(fake_nango) == []


def test_not_connected_and_connect_links(fake_nango):
    with pytest.raises(nango.NotConnected, match=r"nango.connect\('notion'\)"):
        nango.get("notion", "/v1/users")
    # a self-hosted Nango: the link must name it, or the Connect page asks the cloud about the session
    assert nango.connect("notion") == ("https://connect.nango.dev?session_token=t&apiURL="
                                       + nango.urllib.parse.quote(nango._base(), safe=""))
    session = [s for s in fake_nango if s["path"] == "/connect/sessions"][0]
    assert session["body"]["allowed_integrations"] == ["notion"]
    with pytest.raises(nango.NangoError, match="isn't set up in Nango"):
        nango.connect("salesforce")
    assert [i["connected"] for i in nango.integrations()] == [True, True, False]
    assert nango.summary() == "Connected accounts: google-mail, github-jed (needs reconnecting)."


def nango_rules(k, q, s):
    if k == "t_nango":
        return noul_ans(0.9)


def turn_harness(cfg, replies, rules=nango_rules):
    llm = ScriptedLLM(replies)
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: llm)
    h.learning = False
    return h, llm


SEND = py("r = nango.post('google-mail', '/gmail/v1/users/me/messages/send', json={'raw': 'SGk='})\nfinal(r['id'])")


def test_the_user_approves_a_change_before_it_is_sent(cfg, fake_nango):
    h, llm = turn_harness(cfg, [SEND])
    t, out = run_async(h, "email Sam that I'm running late", force="daily_assistant")
    assert wait_for(lambda: h.pending_questions())
    q = h.pending_questions()[0]
    assert "google-mail" in q["question"] and "/gmail/v1/users/me/messages/send" in q["question"] and "SGk=" in q["question"]
    assert proxied(fake_nango) == []          # nothing left before the answer
    h.answer(q["id"], "Allow")
    t.join(30)
    assert out["r"].answer == "sent-1" and len(proxied(fake_nango)) == 1
    system = llm.seen[0][0]["content"]
    assert "nango.get(integration, path" in system
    assert "Connected accounts: google-mail, github-jed (needs reconnecting)." in system


def test_a_refused_change_is_not_sent(cfg, fake_nango):
    code = py("try:\n    nango.post('google-mail', '/gmail/v1/users/me/messages/send', json={'raw': 'x'}, "
              "params={'uploadType': 'q-shown'})\n"
              "except nango.Refused as e:\n    final('refused: ' + str(e))")
    h, _ = turn_harness(cfg, [code])
    t, out = run_async(h, "email Sam", force="daily_assistant")
    assert wait_for(lambda: h.pending_questions())
    assert 'params: {"uploadType": "q-shown"}' in h.pending_questions()[0]["question"]
    h.answer(h.pending_questions()[0]["id"], "Don't allow")
    t.join(30)
    assert out["r"].answer.startswith("refused: POST /gmail/v1/users/me/messages/send on google-mail was not sent")
    assert proxied(fake_nango) == []


def test_nango_is_offered_only_with_a_key_and_names_the_accounts(cfg, fake_nango, monkeypatch):
    seen = {}

    def rules(k, q, s):
        if k.startswith("t_"):
            seen[k] = q["instructions"]["tool"]
    h, llm = turn_harness(cfg, ["ok"] * 2, rules)
    h.turn("any unread email?", force="daily_assistant")
    assert "(Connected accounts: google-mail, github-jed (needs reconnecting))" in seen["t_nango"]
    seen.clear()
    monkeypatch.delenv("NANGO_SECRET_KEY")
    h.turn("any unread email?", force="daily_assistant")
    assert "t_nango" not in seen and "t_web" in seen
