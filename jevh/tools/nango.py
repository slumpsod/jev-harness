"""The user's connected accounts through Nango (https://nango.dev), pre-imported as `nango`.

Nango keeps the OAuth tokens and API keys for ~700 APIs (Gmail, Google Calendar and Drive, GitHub,
Slack, Notion, HubSpot, Linear, ...) and refreshes them. Code here calls each API's own REST endpoints
through Nango's proxy, so a token never reaches the worker's prompt.

    nango.connections()        # [{integration, provider, connection_id, ok}] - what the user has connected
    nango.get("google-mail", "/gmail/v1/users/me/messages", params={"q": "is:unread", "maxResults": 5})
    nango.post("github", "/repos/o/r/issues", json={"title": "..."})   # the user approves it first
    nango.connect("notion")    # a link the user opens to connect an account

First-time setup (Docker, the keys, registering an app at the provider, connecting) is in
nango_setup.py and reached through this module: nango.status(), nango.start(), nango.providers(),
nango.guide(), nango.setup(), nango.connect_and_wait(), nango.test(), nango.open_page(),
nango.browser_mode(). The nango-setup skill walks a user through them.

Anything that could change something (a POST, PUT, PATCH or DELETE that is not a search or query) is
shown to the user, who allows or refuses it before it is sent. The harness asks that question itself,
so a request can't approve itself (an email the worker read can't tell it to forward your inbox).

Settings: tools.nango_url in config.yaml (default https://api.nango.dev; a self-hosted server is
usually http://localhost:3003), tools.nango_writes (ask | deny | allow; default ask), and the
secret key in the NANGO_SECRET_KEY environment variable. nango.start() keeps the key and the URL of
the Nango it set up in <workspace>/nango/ (JEVH_NANGO_DIR), which are used when nothing else says.
"""
from __future__ import annotations

import contextvars
import json as jsonlib
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DEFAULT_URL = "https://api.nango.dev"
READ_METHODS = {"GET", "HEAD", "OPTIONS"}
# POSTs that only read: Notion's /v1/search and /databases/{id}/query, Slack's conversations.history,
# users.lookupByEmail, Google's batchGet... (the last path segment, after its last "." or ":")
READ_POST = re.compile(r"^(search|query|list|history|info|replies|members|batchget|lookup\w*)$", re.I)
_cache: dict[str, tuple[float, Any]] = {}
# The harness's settings for this module (JEVH_NANGO_DIR, JEVH_NANGO_URL, JEVH_NANGO_WRITES) when it calls it in its own
# process (nango.summary() for the prompt): set per call with `using`, not in os.environ, which every Harness in the
# process shared - the first one's settings stuck, and its folder replaced the others'. A worker's REPL gets the same
# values as environment variables.
_SETTINGS: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar("nango_settings", default={})


def _setting(name: str, default: str | None = None) -> str | None:
    return _SETTINGS.get().get(name) or os.environ.get(name, default)


@contextmanager
def using(settings: dict[str, str]):
    """Calls made inside use these settings (environment-variable names) over the process environment."""
    token = _SETTINGS.set({**_SETTINGS.get(), **settings})
    try:
        yield
    finally:
        _SETTINGS.reset(token)
_allowed: set[str] = set()   # "all changes to this integration" the user allowed for this task


class NangoError(RuntimeError):
    pass


class NotConnected(NangoError):
    pass


class Refused(NangoError):
    """The user (or tools.nango_writes: deny) did not allow a change."""


def _state_dir() -> Path:
    """Where the Nango this harness set up keeps its key and URL (and the browser's sign-in profile)."""
    d = _setting("JEVH_NANGO_DIR")
    if d:
        return Path(d)
    from jevh.config import load_config
    return load_config().workspace / "nango"


def _state(name: str) -> str:
    try:
        return (_state_dir() / name).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _save_state(name: str, value: str, private: bool = False) -> None:
    d = _state_dir()
    d.mkdir(parents=True, exist_ok=True)
    f = d / name
    f.write_text(value, encoding="utf-8")
    if private:
        try:
            f.chmod(0o600)
        except OSError:
            pass   # Windows: the folder is already the user's own


def _key() -> str:
    """The secret key: the environment first, else the one nango.start() saved."""
    return os.environ.get("NANGO_SECRET_KEY") or _state("secret_key")


def configured() -> bool:
    return bool(_key())


def _base() -> str:
    return (_setting("JEVH_NANGO_URL") or _state("url") or DEFAULT_URL).rstrip("/")


def _ask(question: str, options: list | None = None, secret: bool = False) -> str | None:
    """Ask the user from inside the tool (the worker's code never sees a question it can answer for the
    user, and a secret goes straight to the caller). None: there is no user here (not in a worker's REPL)."""
    main = sys.modules.get("__main__")
    call = getattr(main, "_call", None)
    if call is None or not hasattr(main, "ask_user"):
        return None
    return str(call("ask_user", question, options, **({"secret": True} if secret else {})) or "")


def _http(method: str, path: str, *, params: dict | None = None, body: Any = None,
          headers: dict | None = None, timeout: float = 60) -> tuple[int, str, bytes]:
    key = _key()
    if not key:
        raise NangoError("Nango isn't set up yet: nango.status() says what is missing and nango.start() sets it "
                         "up on this machine (or set NANGO_SECRET_KEY for Nango Cloud; SETUP.md, 'Connected accounts')")
    url = _base() + path
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(
            {k: (jsonlib.dumps(v) if isinstance(v, dict) else v) for k, v in params.items() if v is not None},
            doseq=True)   # a list: repeated keys (Gmail's labelIds=INBOX&labelIds=UNREAD)
    data = None
    hdrs = {"Authorization": f"Bearer {key}", "Accept": "application/json", **(headers or {})}
    if body is not None:
        if isinstance(body, (bytes, str)):
            data = body.encode() if isinstance(body, str) else body
        else:
            data = jsonlib.dumps(body).encode()
            hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", "") if e.headers else "", e.read() or b""
    except urllib.error.URLError as e:
        raise NangoError(f"can't reach Nango at {_base()}: {e.reason}") from None
    except (TimeoutError, OSError) as e:
        raise NangoError(f"Nango at {_base()} didn't answer: {e}") from None


def _decode(ctype: str, raw: bytes) -> Any:
    if "json" in ctype.lower() or raw[:1] in (b"{", b"["):
        try:
            return jsonlib.loads(raw)
        except ValueError:
            pass
    if re.match(r"(text/|application/(xml|x-www-form|javascript))", ctype.lower()) or not ctype:
        return raw.decode("utf-8", "replace")
    if raw[:2048].count(b"\x00"):
        return raw   # an attachment, an image: bytes (write them to a file)
    return raw.decode("utf-8", "replace")


def _nango(method: str, path: str, **kw) -> Any:
    """A call to Nango's own API (not a proxied one)."""
    status, ctype, raw = _http(method, path, **kw)
    out = _decode(ctype, raw)
    if status >= 400:
        msg = out.get("error", out) if isinstance(out, dict) else out
        raise NangoError(f"Nango {method} {path}: HTTP {status}: {str(msg)[:600]}")
    return out


def _cached(name: str, ttl: float, fetch):
    hit = _cache.get(name)
    if hit and time.monotonic() - hit[0] < ttl:
        return hit[1]
    val = fetch()
    _cache[name] = (time.monotonic(), val)
    return val


# -- what is connected -----------------------------------------------------------------------------

def integrations() -> list[dict]:
    """The APIs set up in the Nango dashboard: [{integration, provider, name, connected}]."""
    data = _cached("integrations", 120, lambda: _nango("GET", "/integrations"))
    items = data.get("data", data) if isinstance(data, dict) else data
    conns = connections()
    return [{"integration": i.get("unique_key"), "provider": i.get("provider"),
             "name": i.get("display_name") or i.get("provider"),
             "connected": any(c["integration"] == i.get("unique_key") for c in conns)} for i in items or []]


def connections(refresh: bool = False, timeout: float = 30) -> list[dict]:
    """The user's connected accounts: [{integration, provider, connection_id, ok, created}].
    ok=False: Nango reports an auth error (reconnect with connect())."""
    if refresh:
        _cache.pop("connections", None)
    data = _cached("connections", 60, lambda: _nango("GET", "/connection", timeout=timeout))
    return [{"integration": c.get("provider_config_key"), "provider": c.get("provider"),
             "connection_id": c.get("connection_id"), "ok": not any(e.get("type") == "auth" for e in c.get("errors") or []),
             "created": c.get("created")} for c in (data.get("connections") or [])]


def _resolve(integration: str, connection_id: str | None) -> tuple[str, str]:
    """(provider_config_key, connection_id) for a name the worker used ("github", "google-mail", "gmail")."""
    def find(conns):
        exact = [c for c in conns if c["integration"] == integration]
        return exact or [c for c in conns if c["provider"] == integration]
    hits = find(connections())
    if not hits:
        hits = find(connections(refresh=True))   # connected a minute ago (connect() + ask_user)
    if connection_id:
        hits = [c for c in hits if c["connection_id"] == connection_id]
    if not hits:
        have = ", ".join(sorted({c["integration"] for c in connections()}))
        raise NotConnected(f"no '{integration}' account is connected" + (f" (connected: {have})" if have else "")
                           + f". nango.connect_and_wait('{integration}') opens the sign-in page for the user "
                             "(an app Nango doesn't have yet: nango.guide(app), then nango.setup(app)).")
    if len(hits) > 1:
        raise NangoError(f"{len(hits)} '{integration}' accounts are connected: pass connection_id= one of "
                         f"{[c['connection_id'] for c in hits]}")
    return hits[0]["integration"], hits[0]["connection_id"]


def _integration_key(integration: str) -> str | None:
    """The key of the Nango integration a name means ("github" -> "github-jed"), or None when it isn't set up."""
    keys = {i["integration"]: i for i in integrations()}
    by_provider = {i["provider"]: i["integration"] for i in keys.values()}
    return integration if integration in keys else by_provider.get(integration)


def connect(integration: str) -> str:
    """A Nango Connect link for the user to connect (or reconnect) an account of `integration`.
    Give it to the user with ask_user(f"Open {link} to connect ..., then reply done", ["done", "cancel"]);
    after "done", call the API as usual."""
    key = _integration_key(integration)
    if not key:
        keys = {i["integration"] for i in integrations()}
        raise NangoError(f"'{integration}' isn't set up in Nango yet (set up: {', '.join(sorted(keys)) or 'none'}). "
                         f"nango.guide({integration!r}) says what the user does at the app, and "
                         f"nango.setup({integration!r}) adds it. Don't ask the user to use the dashboard.")
    res = _nango("POST", "/connect/sessions", body={
        "tags": {"end_user_id": os.environ.get("JEVH_NANGO_USER", "jev-harness")},
        "allowed_integrations": [key]})
    data = res.get("data", res) if isinstance(res, dict) else {}
    link = data.get("connect_link")
    if not link:
        raise NangoError(f"Nango made a connect session but returned no link: {str(res)[:300]}")
    if _base() != DEFAULT_URL and "apiURL=" not in link:
        # the Connect page talks to the Nango its link names, else to the cloud: a self-hosted server's
        # links leave it out, and the page then says "Your session has expired" (live, Nango 0.71.10)
        link += ("&" if "?" in link else "?") + urllib.parse.urlencode({"apiURL": _base()})
    _cache.pop("connections", None)
    return link


def summary() -> str:
    """One line for the prompt: which accounts are connected (empty if Nango is down or unset)."""
    if not configured():
        return ("Nango isn't set up yet: nango.status() says what is missing, nango.start() sets it up here, "
                "and the nango-setup skill walks the user through it.")
    down = _cache.get("down")
    if down and time.monotonic() - down[0] < 60:   # every task asks: a dead server costs one short wait a minute
        return ""
    try:
        conns = connections(timeout=5)
    except NangoError:
        _cache["down"] = (time.monotonic(), True)
        return "Nango is set up but not reachable right now (nango.status() says why; nango.start() starts it)."
    if not conns:
        return ("No accounts are connected yet (nango.integrations() lists the apps set up; nango.guide(app) and "
                "nango.setup(app) add one; the nango-setup skill walks the user through it).")
    parts = [f"{c['integration']}{'' if c['ok'] else ' (needs reconnecting)'}" for c in conns]
    return "Connected accounts: " + ", ".join(dict.fromkeys(parts)) + "."


# -- calling the APIs --------------------------------------------------------------------------------

def is_read(method: str, path: str, body: Any = None) -> bool:
    """Plain reads go straight through; anything else is a change the user approves."""
    method = method.upper()
    if method in READ_METHODS:
        return True
    if method != "POST":
        return False
    last = urllib.parse.urlparse(path).path.rstrip("/").rsplit("/", 1)[-1]
    if last.lower() == "graphql":   # a GraphQL query reads; a mutation writes
        q = body.get("query", "") if isinstance(body, dict) else ""
        return bool(q) and not re.match(r"\s*mutation\b", q)
    return bool(READ_POST.match(re.split(r"[.:]", last)[-1]))


def _approve(integration: str, method: str, path: str, body: Any, params: dict | None = None,
             base_url: str | None = None) -> None:
    mode = (_setting("JEVH_NANGO_WRITES") or "ask").lower()
    if base_url is None and (mode == "allow" or integration in _allowed):
        return
    what = f"{method} {path} on {integration}" + (f" (sent to {base_url})" if base_url else "")
    if mode == "deny" and not base_url:
        raise Refused(f"{what} was not sent: changes through Nango are turned off (tools.nango_writes: deny)")
    shown = body if isinstance(body, str) else jsonlib.dumps(body, ensure_ascii=False, default=str) if body is not None else ""
    if params:   # some APIs take the whole message as query params (Slack's chat.postMessage?text=...)
        shown = f"params: {jsonlib.dumps(params, ensure_ascii=False, default=str)}" + (f"\n{shown}" if shown else "")
    q = (f"Send your {integration} login to {base_url}? (a server other than {integration}'s own)\n" if base_url
         else f"Allow this change through your {integration} account?\n") + f"{method} {path}" + (
        f"\n{shown[:800]}{'...' if len(shown) > 800 else ''}" if shown else "")
    everything = f"Allow all {integration} changes this task"
    answer = _ask(q, ["Allow", "Don't allow"] if base_url else ["Allow", everything, "Don't allow"])
    if answer is None:   # not in a worker's REPL: nobody to ask
        raise Refused(f"{what} was not sent: there is no user here to approve a change")
    if answer.strip().lower() == everything.lower():
        _allowed.add(integration)
    elif answer.strip().lower() not in ("allow", "yes", "y"):
        raise Refused(f"{what} was not sent: the user did not allow it (answer: {answer[:200]!r}). "
                      "Don't retry it; say what you would have sent.")


def request(integration: str, method: str, path: str, *, params: dict | None = None, json: Any = None,
            data: Any = None, headers: dict | None = None, connection_id: str | None = None,
            base_url: str | None = None, timeout: float = 60) -> Any:
    """Call `integration`'s own API (its REST path, e.g. "/gmail/v1/users/me/messages") as the user.
    Returns the parsed JSON (or text, or bytes for a file). HTTP errors raise NangoError with the
    API's own message. A change is shown to the user first; a refusal raises Refused."""
    method = method.upper()
    if not path.startswith("/"):
        path = "/" + path
    key, conn = _resolve(integration, connection_id)
    body = json if json is not None else data
    if base_url or not is_read(method, path, body):   # Nango adds the user's token for whatever host base_url names
        _approve(key, method, path, body, params, base_url)
    hdrs = {**(headers or {}), "Connection-Id": conn, "Provider-Config-Key": key}
    if method in READ_METHODS:
        hdrs.setdefault("Retries", "2")   # Nango retries 5xx/429 with backoff; never for a change (no double sends)
    if base_url:
        hdrs["Base-Url-Override"] = base_url
    status, ctype, raw = _http(method, "/proxy" + path, params=params, body=body, headers=hdrs, timeout=timeout)
    out = _decode(ctype, raw)
    if status >= 400:
        text = out if isinstance(out, str) else jsonlib.dumps(out, default=str) if not isinstance(out, bytes) else ""
        hint = (f" (the {key} connection may need reconnecting: nango.connect('{key}'))"
                if status in (401, 403) else "")
        raise NangoError(f"{method} {path} on {key}: HTTP {status}{hint}: {text[:1000]}")
    return out


def get(integration: str, path: str, params: dict | None = None, **kw) -> Any:
    return request(integration, "GET", path, params=params, **kw)


def post(integration: str, path: str, json: Any = None, **kw) -> Any:
    return request(integration, "POST", path, json=json, **kw)


def put(integration: str, path: str, json: Any = None, **kw) -> Any:
    return request(integration, "PUT", path, json=json, **kw)


def patch(integration: str, path: str, json: Any = None, **kw) -> Any:
    return request(integration, "PATCH", path, json=json, **kw)


def delete(integration: str, path: str, **kw) -> Any:
    return request(integration, "DELETE", path, **kw)


# -- first-time setup lives in nango_setup.py; `nango.start()` etc. reach it from here ------------------------
_SETUP = {"status", "start", "providers", "guide", "setup", "connect_and_wait", "test", "open_page",
          "browser_mode", "docker_unstick"}


def __getattr__(name: str):
    if name in _SETUP:
        from jevh.tools import nango_setup
        return getattr(nango_setup, name)
    raise AttributeError(f"module 'nango' has no attribute {name!r}")
