"""A fake Nango server for the tests (HTTP on localhost): the endpoints the `nango` tool uses, with state.

    srv = serve(); srv.url; srv.seen (every request); srv.integrations / srv.connections (dicts and lists
    to look at or change); srv.dashboard_open (the dashboard answers api-keys without a login)
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KEY = "sk-test"

CONNECTIONS = [{"id": 1, "connection_id": "conn-gm", "provider": "google-mail", "provider_config_key": "google-mail",
                "created": "2026-09-01T00:00:00Z", "errors": []},
               {"id": 2, "connection_id": "conn-gh", "provider": "github", "provider_config_key": "github-jed",
                "created": "2026-09-01T00:00:00Z", "errors": [{"type": "auth", "log_id": "x"}]}]
INTEGRATIONS = {"google-mail": {"unique_key": "google-mail", "provider": "google-mail", "display_name": "Gmail"},
                "github-jed": {"unique_key": "github-jed", "provider": "github", "display_name": "GitHub"},
                "notion": {"unique_key": "notion", "provider": "notion", "display_name": "Notion"}}


def _provider(name, display, mode, cats=("productivity",), **extra):
    return {"name": name, "display_name": display, "auth_mode": mode, "categories": list(cats),
            "docs": f"https://nango.dev/docs/api-integrations/{name}",
            "setup_guide_url": f"https://nango.dev/docs/api-integrations/{name}/how-to-register-your-own-{name}-oauth-app",
            **extra}


PROVIDERS = [
    _provider("google-mail", "Gmail", "OAUTH2", ("popular", "productivity")),
    _provider("google-calendar", "Google Calendar", "OAUTH2", ("popular", "productivity")),
    _provider("google-drive", "Google Drive", "OAUTH2", ("popular", "storage")),
    _provider("github", "GitHub (User OAuth)", "OAUTH2", ("popular", "dev-tools")),
    _provider("slack", "Slack", "OAUTH2", ("popular", "communication")),
    _provider("slack-mcp", "Slack (MCP)", "MCP_OAUTH2", ("communication", "mcp")),
    _provider("notion", "Notion", "OAUTH2", ("popular", "productivity")),
    _provider("linear", "Linear", "OAUTH2", ("popular", "ticketing")),
    _provider("linear-mcp", "Linear (MCP)", "MCP_OAUTH2", ("ticketing", "mcp")),
    _provider("twitter", "Twitter (v1)", "OAUTH1", ("social",)),
    _provider("airtable-pat", "Airtable (Personal Access Token)", "API_KEY", ("productivity",),
              credentials={"apiKey": {"type": "string", "title": "Personal Access Token"}},
              proxy={"base_url": "https://api.airtable.com", "verification": {"method": "GET",
                                                                             "endpoints": ["/v0/meta/whoami"]}}),
    _provider("bamboohr", "BambooHR", "API_KEY", ("hr",), connection_config={"subdomain": {"title": "Subdomain"}}),
    _provider("apple-calendar", "Apple Calendar", "BASIC", ("productivity",),
              docs_connect="https://nango.dev/docs/api-integrations/apple-calendar/connect",
              credentials={"username": {"title": "Apple ID", "description": "Your Apple ID email address.",
                                        "example": "you@icloud.com"},
                           "password": {"title": "App-Specific Password",
                                        "description": "An app-specific password for your Apple ID.",
                                        "example": "abcd-****-****-mnop"}}),
]


class FakeNango(BaseHTTPRequestHandler):
    server: "Server"

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
        srv = self.server
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n)) if n else None
        path = self.path
        srv.seen.append({"method": self.command, "path": path, "headers": dict(self.headers), "body": body})
        if path == "/health":
            return self._reply(200, {"result": "ok"})
        if path.startswith("/api/v1/environment/api-keys"):   # the dashboard's own API (no Bearer key)
            return self._reply(200, {"data": [{"secret": KEY}]}) if srv.dashboard_open else \
                self._reply(401, {"error": "login"})
        if self.headers.get("Authorization") != f"Bearer {KEY}":
            return self._reply(401, {"error": "bad key"})
        if path == "/connection":
            return self._reply(200, {"connections": srv.connections})
        if path == "/integrations" and self.command == "GET":
            return self._reply(200, {"data": list(srv.integrations.values())})
        if path == "/integrations" and self.command == "POST":
            if body["unique_key"] in srv.integrations:
                return self._reply(409, {"error": {"code": "duplicate"}})
            srv.integrations[body["unique_key"]] = {k: v for k, v in body.items()}
            return self._reply(200, {"data": {"unique_key": body["unique_key"], "provider": body["provider"]}})
        if path.startswith("/integrations/"):
            key = path.split("/")[2]
            if key not in srv.integrations:
                return self._reply(404, {"error": {"code": "not_found"}})
            if self.command == "PATCH":
                srv.integrations[key].update(body)
            return self._reply(200, {"data": srv.integrations[key]})
        if path == "/providers":
            return self._reply(200, {"data": PROVIDERS})
        if path.startswith("/providers/"):
            hit = next((p for p in PROVIDERS if p["name"] == path.split("/")[2]), None)
            return self._reply(200, {"data": hit}) if hit else self._reply(404, {"error": "unknown provider"})
        if path == "/connect/sessions":
            if srv.on_connect_session:
                srv.on_connect_session(body)
            return self._reply(201, {"data": {"token": "t", "connect_link": "https://connect.nango.dev?session_token=t",
                                              "expires_at": "2026-09-29T12:00:00Z"}})
        if path.startswith("/proxy/gmail/v1/users/me/profile"):
            return self._reply(200, {"emailAddress": "jed@example.com"})
        if path.startswith("/proxy/gmail/v1/users/me/messages/send"):
            return self._reply(200, {"id": "sent-1"})
        if path.startswith("/proxy/gmail/v1/users/me/messages"):
            return self._reply(200, {"messages": [{"id": "m1"}, {"id": "m2"}], "resultSizeEstimate": 2})
        if path.startswith("/proxy/v0/meta/whoami"):
            return self._reply(200, {"id": "usr1"})
        if path.startswith("/proxy/user"):
            return self._reply(401, {"message": "Bad credentials"})
        return self._reply(404, {"error": "not found"})

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _any


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), FakeNango)
        self.seen: list = []
        self.connections = [dict(c) for c in CONNECTIONS]
        self.integrations = {k: dict(v) for k, v in INTEGRATIONS.items()}
        self.dashboard_open = True
        self.on_connect_session = None   # called with the session body when a connect link is made

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def add_connection(self, integration: str, provider: str, connection_id: str) -> None:
        self.connections.append({"id": len(self.connections) + 1, "connection_id": connection_id,
                                 "provider": provider, "provider_config_key": integration,
                                 "created": "2026-09-30T00:00:00Z", "errors": []})

    def proxied(self) -> list:
        return [s for s in self.seen if s["path"].startswith("/proxy/")]


def serve() -> Server:
    srv = Server()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
