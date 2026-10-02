"""The nango tool's first-time setup: Docker and Nango, finding apps, guides, secrets, connecting, testing.
Against the fake Nango server in tests/fake_nango.py; nothing here touches Docker or a browser."""
import json
import threading
import time

import pytest

from jevh.jev import Jev
from jevh.orchestrator import Harness
from jevh.tools import nango, nango_setup as ns

from conftest import FakeSystemOne, ScriptedLLM, noul_ans, py
from test_cancel_ask import run_async, wait_for

REAL_STALE_SOCKET_DIRS = ns._stale_socket_dirs   # before any test replaces it


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(ns, "POLL", 0.05)
    monkeypatch.setattr(ns, "_say", lambda msg: None)


def asks(monkeypatch, *answers):
    """Stand in for the user: returns the answers in order; the (question, options, secret) asked go in the list."""
    given, log = list(answers), []

    def fake(question, options=None, secret=False):
        log.append((question, options, secret))
        return given.pop(0)
    monkeypatch.setattr(nango, "_ask", fake)
    return log


# -- the key and URL nango.start() keeps ------------------------------------------------------------------

def test_the_saved_key_and_url_are_used_when_nothing_else_says(nango_dir, monkeypatch):
    monkeypatch.delenv("NANGO_SECRET_KEY", raising=False)
    monkeypatch.delenv("JEVH_NANGO_URL", raising=False)
    assert not nango.configured() and nango._base() == nango.DEFAULT_URL
    nango._save_state("secret_key", "saved-key", private=True)
    nango._save_state("url", "http://localhost:3003/")
    assert nango.configured() and nango._key() == "saved-key" and nango._base() == "http://localhost:3003"
    monkeypatch.setenv("NANGO_SECRET_KEY", "env-key")
    monkeypatch.setenv("JEVH_NANGO_URL", "http://elsewhere:1")
    assert nango._key() == "env-key" and nango._base() == "http://elsewhere:1"   # the environment wins


# -- secrets are asked for hidden, and kept out of the events ---------------------------------------------------

def test_a_secret_answer_is_masked_in_events_and_reaches_only_the_code(cfg):
    events = []
    llm = ScriptedLLM([py("k = ask_user('Your API key?', ['a', 'b'], secret=True)\nfinal('len=' + str(len(k)))")])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(lambda k, q, s: noul_ans(0.9) if k == "needs_high_end_dev" else None)),
                llm_factory=lambda p: llm, on_event=lambda k, d: events.append((k, d)))
    h.learning = False
    t, out = run_async(h, "set the key", force="daily_assistant")
    try:
        assert wait_for(lambda: h.pending_questions())
        q = h.pending_questions()[0]
        assert q["secret"] is True and q["options"] == []    # no buttons: a secret isn't one of a few choices
        h.answer(q["id"], "sk-very-secret")
        t.join(30)
    finally:
        h.cancel()
    assert out["r"].answer == "len=14"                       # the code got it
    answered = [d for k, d in events if k == "answered"][0]
    assert answered["answer"] == "(hidden)"
    assert "sk-very-secret" not in json.dumps([(k, str(d)) for k, d in events])
    assert "sk-very-secret" not in json.dumps(h.pending_questions())


def test_a_free_text_question_asking_for_a_password_or_key_is_hidden_whatever_the_worker_passed(cfg):
    asked = {}
    llm = ScriptedLLM([py("a = ask_user('Please paste your API key')\nb = ask_user('What is your password?')\n"
                          "c = ask_user('Which size?')\nd = ask_user('Send the password reset?', ['Yes', 'No'])\n"
                          "final('ok')")])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(lambda k, q, s: noul_ans(0.9) if k == "needs_high_end_dev" else None)),
                llm_factory=lambda p: llm, on_event=lambda k, d: asked.setdefault(d["question"]["question"], d["question"])
                if k == "question" else None)
    h.learning = False
    t, out = run_async(h, "ask me things", force="daily_assistant")
    try:
        for _ in range(4):
            assert wait_for(lambda: h.pending_questions())
            h.answer(h.pending_questions()[0]["id"], "x")
            time.sleep(0.8)
        t.join(30)
    finally:
        h.cancel()
    assert asked["Please paste your API key"]["secret"] and asked["What is your password?"]["secret"]
    assert not asked["Which size?"]["secret"]
    assert not asked["Send the password reset?"]["secret"] and asked["Send the password reset?"]["options"] == ["Yes", "No"]


# -- finding apps ---------------------------------------------------------------------------------------------------

def test_providers_search_ranks_the_plain_app_before_its_variants(fake_nango):
    assert [p["name"] for p in ns.providers("slack")] == ["slack", "slack-mcp"]
    assert ns.providers("calendar")[0]["name"] == "google-calendar"
    popular = {p["name"] for p in ns.providers()}
    assert {"google-mail", "github"} <= popular and "airtable-pat" not in popular
    assert ns.providers("zzz") == []
    assert [p["name"] for p in ns.providers("email")][:1] == ["google-mail"]      # everyday words find the app
    assert "slack" in [p["name"] for p in ns.providers("chat")]


def test_names_people_use_find_the_provider(fake_nango):
    assert ns._provider("Gmail") == "google-mail" and ns._provider("google calendar") == "google-calendar"
    assert ns._provider("github") == "github"
    with pytest.raises(nango.NangoError, match=r"no app called 'googl' .*did you mean: google-"):
        ns._provider("googl")
    with pytest.raises(nango.NangoError, match="providers"):
        ns._provider("nothing-like-it")


# -- what the user does at the app -----------------------------------------------------------------------------------

def test_the_google_guide_has_every_step_with_the_exact_callback(fake_nango, monkeypatch):
    g = ns.guide("gmail")
    assert g["provider"] == "google-mail" and g["auth_mode"] == "OAUTH2"
    assert g["callback_url"] == fake_nango.url + "/oauth/callback"
    assert len(g["steps"]) == 5
    assert "gmail.googleapis.com" in g["steps"][1]["url"] and "Gmail API" in g["steps"][1]["title"]
    assert g["callback_url"] in g["steps"][4]["do"] and "Publish" in g["steps"][3]["title"]
    assert g["scopes"]["read"].endswith("gmail.readonly") and g["scopes"]["write"].endswith("gmail.modify")
    cal = ns.guide("google-calendar")
    assert "calendar-json.googleapis.com" in cal["steps"][1]["url"]
    gh = ns.guide("github")
    assert gh["steps"][0]["url"] == "https://github.com/settings/applications/new" and "scope_note" in gh


def test_other_apps_get_a_generic_guide_from_nangos_own_page(fake_nango):
    sl = ns.guide("slack")
    assert sl["steps"][0]["url"] == sl["setup_guide_url"] and sl["callback_url"] in sl["steps"][0]["do"]
    assert "scopes" not in sl and "nango.setup('slack')" in sl["next"]
    key = ns.guide("airtable-pat")
    assert key["auth_mode"] == "API_KEY" and key["steps"][0]["url"] == key["setup_guide_url"]
    assert "Personal Access Token" in key["steps"][0]["do"] and key["fields"] == {"apiKey": "Personal Access Token"}
    assert ns.guide("bamboohr")["fields"] == {"subdomain": "Subdomain"}


def test_a_login_based_app_says_what_to_find_and_where(fake_nango, monkeypatch):
    monkeypatch.setattr(ns, "_offered", set())
    g = ns.guide("apple-calendar")
    step = g["steps"][0]
    assert g["auth_mode"] == "BASIC" and g["connect_guide_url"].endswith("/apple-calendar/connect")
    assert "Apple ID (Your Apple ID email address) e.g. you@icloud.com" in step["do"]
    assert "App-Specific Password (An app-specific password for your Apple ID) e.g. abcd-****-****-mnop" in step["do"]
    assert "never here" in step["do"] and step["url"] == g["connect_guide_url"]
    monkeypatch.setattr(ns.webbrowser, "open", lambda u: True)
    assert ns.open_page(step["url"])["opened"]       # the page the guide pointed at


def test_open_page_only_opens_local_nango_and_pages_the_guide_gave(fake_nango, monkeypatch):
    opened = []
    monkeypatch.setattr(ns.webbrowser, "open", lambda u: opened.append(u) or True)
    monkeypatch.setattr(ns, "_offered", set())
    with pytest.raises(nango.NangoError, match="show the user the link"):
        ns.open_page("https://console.cloud.google.com/projectcreate")   # no guide asked for it yet
    with pytest.raises(nango.NangoError):
        ns.open_page("https://evil.example/login")
    with pytest.raises(nango.NangoError):
        ns.open_page("file:///C:/Windows/System32/calc.exe")
    ns.guide("google-mail")
    assert ns.open_page("https://console.cloud.google.com/projectcreate")["opened"]
    assert ns.open_page("http://localhost:3003/dev/environment-settings")["opened"]
    with pytest.raises(nango.NangoError):
        ns.open_page("https://console.cloud.google.com/somewhere/else")
    assert opened == ["https://console.cloud.google.com/projectcreate", "http://localhost:3003/dev/environment-settings"]


# -- giving Nango the app's ID and secret ---------------------------------------------------------------------------

GOOD_ID = "1234567890-abcdefg.apps.googleusercontent.com"
GOOD_SECRET = "GOCSPX-abc123_secretvalue"


def test_setup_asks_for_the_id_then_the_secret_hidden_and_stores_them(fake_nango, monkeypatch):
    log = asks(monkeypatch, GOOD_ID, GOOD_SECRET)
    r = ns.setup("google-calendar", access="write")
    assert [(secret, opts) for _, opts, secret in log] == [(False, None), (True, None)]
    assert "Client ID" in log[0][0] and "hidden" in log[1][0]
    stored = fake_nango.integrations["google-calendar"]
    assert stored["credentials"] == {"type": "OAUTH2", "client_id": GOOD_ID, "client_secret": GOOD_SECRET,
                                     "scopes": "https://www.googleapis.com/auth/calendar"}
    assert r["integration"] == "google-calendar" and r["created"] and r["scopes"].endswith("/calendar")
    assert GOOD_SECRET not in json.dumps(r) and "connect_and_wait('google-calendar')" in r["next"]


def test_setup_takes_values_read_off_a_page_and_defaults_to_read_only(fake_nango, monkeypatch):
    log = asks(monkeypatch)       # nobody is asked
    ns.setup("google-drive", client_id=GOOD_ID, client_secret=GOOD_SECRET)
    assert log == []
    assert fake_nango.integrations["google-drive"]["credentials"]["scopes"].endswith("drive.readonly")


def test_setup_finds_the_id_and_secret_on_a_page_without_asking_or_returning_them(fake_nango, monkeypatch):
    log = asks(monkeypatch)
    page = ("OAuth client created\nYour Client ID\n" + GOOD_ID + "\nYour Client secret\n" + GOOD_SECRET
            + "\nDownload JSON  OK  (the secret will not be shown again)")
    r = ns.setup("google-drive", page=page)
    assert log == [] and GOOD_SECRET not in json.dumps(r) and GOOD_ID not in json.dumps(r)
    assert fake_nango.integrations["google-drive"]["credentials"]["client_secret"] == GOOD_SECRET
    # only what the page shows is used: the missing secret is asked for, hidden
    fake_nango.integrations.pop("google-drive")
    log = asks(monkeypatch, GOOD_SECRET)
    ns.setup("google-drive", page="Client ID " + GOOD_ID)
    assert [secret for _, _, secret in log] == [True]
    # GitHub's forms, and a page that shows neither
    gh_secret = "ab12" * 10
    assert ns._find_credentials("github", "Client ID Iv1.0123456789abcdef\nsecret " + gh_secret) == \
        ("Iv1.0123456789abcdef", gh_secret)
    assert ns._find_credentials("github", "Client ID Ov23liAbCdEfGhIjKlMn") == ("Ov23liAbCdEfGhIjKlMn", None)
    assert ns._find_credentials("google-mail", "nothing here") == (None, None)
    assert ns._find_credentials("slack", page) == (None, None)


def test_setup_updates_an_existing_app_and_says_old_accounts_keep_their_scopes(fake_nango, monkeypatch):
    asks(monkeypatch)
    r = ns.setup("google-mail", client_id=GOOD_ID, client_secret=GOOD_SECRET, access="write")
    assert r["updated"] and "connect again" in r["note"]
    assert any(s["method"] == "PATCH" and s["path"] == "/integrations/google-mail" for s in fake_nango.seen)
    assert fake_nango.integrations["google-mail"]["credentials"]["scopes"].endswith("gmail.modify")


def test_setup_catches_a_swapped_or_wrong_paste_before_storing_it(fake_nango, monkeypatch):
    asks(monkeypatch)
    with pytest.raises(nango.NangoError, match="look swapped"):
        ns.setup("google-calendar", client_id=GOOD_SECRET, client_secret=GOOD_ID)
    with pytest.raises(nango.NangoError, match="doesn't look like a Google Client ID"):
        ns.setup("google-calendar", client_id="my-project", client_secret=GOOD_SECRET)
    with pytest.raises(nango.NangoError, match="both needed"):
        ns.setup("google-calendar", client_id=GOOD_ID, client_secret=" ")
    assert "google-calendar" not in fake_nango.integrations


def test_setup_needs_scopes_for_an_app_it_doesnt_know_and_takes_them(fake_nango, monkeypatch):
    asks(monkeypatch)
    with pytest.raises(nango.NangoError, match="which scopes slack needs isn't known"):
        ns.setup("slack", client_id="123.456", client_secret="s3cret")
    ns.setup("slack", client_id="123.456", client_secret="s3cret", scopes=["channels:read", "chat:write"])
    assert fake_nango.integrations["slack"]["credentials"]["scopes"] == "channels:read,chat:write"


def test_setup_for_key_apps_needs_no_credentials_and_odd_modes_point_to_the_dashboard(fake_nango, monkeypatch):
    log = asks(monkeypatch)
    r = ns.setup("airtable-pat")
    assert log == [] and "credentials" not in fake_nango.integrations["airtable-pat"] and r["created"]
    with pytest.raises(nango.NangoError, match="signs in with OAUTH1"):
        ns.setup("twitter")
    with pytest.raises(nango.NangoError, match="signs in with MCP_OAUTH2"):
        ns.setup("linear-mcp")


def test_setup_with_nobody_to_ask_says_so_and_a_failure_never_echoes_the_secret(fake_nango, monkeypatch):
    with pytest.raises(nango.NangoError, match="no user here to ask"):
        ns.setup("google-calendar")       # not in a REPL: no one to ask for the ID
    real = nango._nango

    def leaky(method, path, **kw):
        if method == "POST" and path == "/integrations":
            raise nango.NangoError(f"HTTP 400: bad credentials {GOOD_SECRET} for {GOOD_ID}")
        return real(method, path, **kw)
    monkeypatch.setattr(nango, "_nango", leaky)
    with pytest.raises(nango.NangoError) as e:
        ns.setup("google-calendar", client_id=GOOD_ID, client_secret=GOOD_SECRET)
    assert GOOD_SECRET not in str(e.value) and GOOD_ID not in str(e.value) and "(hidden)" in str(e.value)


def test_the_secret_flows_from_the_users_hidden_answer_to_nango_and_nowhere_else(cfg, fake_nango):
    """A worker runs nango.setup(); the user is asked for the ID, then the secret (hidden). The secret reaches the
    fake Nango and nothing else: not an event, not the worker's answer."""
    events = []
    llm = ScriptedLLM([py("r = nango.setup('google-calendar', access='read')\nfinal('set up ' + r['integration'])")])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(lambda k, q, s: noul_ans(0.9) if k == "t_nango" else None)),
                llm_factory=lambda p: llm, on_event=lambda k, d: events.append((k, d)))
    h.learning = False
    t, out = run_async(h, "connect my calendar", force="daily_assistant")
    try:
        assert wait_for(lambda: h.pending_questions())
        q1 = h.pending_questions()[0]
        assert q1["secret"] is False and "Client ID" in q1["question"]
        h.answer(q1["id"], GOOD_ID)
        assert wait_for(lambda: any(q["id"] != q1["id"] for q in h.pending_questions()))   # q1 is cleared a moment late
        q2 = [q for q in h.pending_questions() if q["id"] != q1["id"]][0]
        assert q2["secret"] is True and q2["options"] == []
        h.answer(q2["id"], GOOD_SECRET)
        t.join(30)
    finally:
        h.cancel()
    assert out["r"].answer == "set up google-calendar"
    assert fake_nango.integrations["google-calendar"]["credentials"]["client_secret"] == GOOD_SECRET
    assert GOOD_SECRET not in json.dumps([(k, str(d)) for k, d in events])
    assert [d["answer"] for k, d in events if k == "answered"] == [GOOD_ID, "(hidden)"]


# -- connecting and checking ---------------------------------------------------------------------------------------

def test_connect_and_wait_opens_the_page_and_returns_when_the_account_appears(fake_nango, monkeypatch):
    opened = []
    monkeypatch.setattr(ns.webbrowser, "open", lambda u: opened.append(u) or True)
    fake_nango.on_connect_session = lambda body: threading.Timer(
        0.3, fake_nango.add_connection, ("notion", "notion", "conn-notion")).start()
    r = ns.connect_and_wait("notion", timeout=10)
    assert r["connected"] and r["connection_id"] == "conn-notion" and "nango.test('notion')" in r["next"]
    assert len(opened) == 1 and opened[0].startswith("https://connect.nango.dev?session_token=t&apiURL=")


def test_connect_and_wait_adds_an_app_that_needs_no_registration_and_says_so_for_one_that_does(fake_nango, monkeypatch):
    monkeypatch.setattr(ns.webbrowser, "open", lambda u: True)
    fake_nango.on_connect_session = lambda body: threading.Timer(
        0.3, fake_nango.add_connection, ("airtable-pat", "airtable-pat", "conn-at")).start()
    r = ns.connect_and_wait("airtable-pat", timeout=10)        # not set up yet; a key app: no app to register
    assert "airtable-pat" in fake_nango.integrations and "credentials" not in fake_nango.integrations["airtable-pat"]
    assert r["connected"] and r["connection_id"] == "conn-at"
    with pytest.raises(nango.NangoError, match=r"needs an app registered at the provider first.*nango.setup\('google-calendar'\)"):
        ns.connect_and_wait("google-calendar", timeout=1)      # OAuth: guide() and setup() first
    assert "google-calendar" not in fake_nango.integrations
    with pytest.raises(nango.NangoError, match="no app called"):
        ns.connect_and_wait("nothing-like-it", timeout=1)


def test_connect_and_wait_follows_a_provider_name_to_its_integration_key(fake_nango, monkeypatch):
    monkeypatch.setattr(ns.webbrowser, "open", lambda u: True)
    fake_nango.on_connect_session = lambda body: threading.Timer(
        0.3, fake_nango.add_connection, (body["allowed_integrations"][0], "github", "conn-gh2")).start()
    r = ns.connect_and_wait("github", timeout=10)               # the integration is called github-jed
    assert r["connected"] and r["integration"] == "github-jed" and r["connection_id"] == "conn-gh2"
    assert [s["body"]["allowed_integrations"] for s in fake_nango.seen if s["path"] == "/connect/sessions"] == [["github-jed"]]


def test_the_not_set_up_message_points_at_setup_not_the_dashboard(fake_nango):
    with pytest.raises(nango.NangoError) as e:
        nango.connect("salesforce")
    assert "nango.setup('salesforce')" in str(e.value) and "dashboard" in str(e.value) and "Don't ask the user" in str(e.value)


def test_connect_and_wait_times_out_with_the_link(fake_nango, monkeypatch):
    monkeypatch.setattr(ns.webbrowser, "open", lambda u: False)
    r = ns.connect_and_wait("notion", timeout=0.3)
    assert r["connected"] is False and "session_token=t" in r["link"]
    # an account that was already connected is not "new"
    r = ns.connect_and_wait("google-mail", timeout=0.3)
    assert r["connected"] is False


def test_test_reads_whose_account_it_is(fake_nango):
    assert ns.test("google-mail") == {"ok": True, "detail": "jed@example.com"}
    r = ns.test("github")           # the fake answers 401 for it
    assert r["ok"] is False and "reconnecting" in r["detail"]
    fake_nango.add_connection("airtable-pat", "airtable-pat", "conn-at")
    assert ns.test("airtable-pat") == {"ok": True, "detail": "the app answered"}   # Nango's own check endpoint


def test_test_for_an_unconnected_app_raises_not_connected(fake_nango):
    with pytest.raises(nango.NotConnected):
        ns.test("slack")


# -- status and start ----------------------------------------------------------------------------------------------

def test_status_says_what_is_missing_and_what_to_do(fake_nango, monkeypatch, nango_dir):
    monkeypatch.setattr(ns, "_docker", lambda: "running")
    s = ns.status()
    assert s["set_up"] and s["nango_running"] and s["where"] == "this machine" and s["url"] == fake_nango.url
    assert s["integrations"] == ["google-mail", "github-jed", "notion"]
    assert s["connected"] == ["google-mail", "github-jed (needs reconnecting)"] and "everything is set up" in s["next"]
    monkeypatch.delenv("NANGO_SECRET_KEY")
    s = ns.status()
    assert s["set_up"] is False and "nango.start()" in s["next"] and "Nango Cloud" in s["next"]
    monkeypatch.setenv("NANGO_SECRET_KEY", "wrong")
    s = ns.status()
    assert "key isn't accepted" in s["next"] and "problem" in s
    monkeypatch.setenv("NANGO_SECRET_KEY", "sk-test")
    monkeypatch.setenv("JEVH_NANGO_URL", "http://127.0.0.1:9")   # nothing there
    s = ns.status()
    assert s["nango_running"] is False and "nango.start()" in s["next"]


@pytest.fixture
def local_nango(fake_nango, tmp_path, monkeypatch):
    """start() against the fake server: no Docker, a compose folder of its own, and no key or URL known yet."""
    compose = tmp_path / "compose"
    compose.mkdir()
    (compose / "docker-compose.yaml").write_text("services: {}\n")
    monkeypatch.setattr(ns, "COMPOSE_DIR", compose)
    monkeypatch.setattr(ns, "LOCAL_URL", fake_nango.url)
    monkeypatch.delenv("NANGO_SECRET_KEY")
    monkeypatch.delenv("JEVH_NANGO_URL")
    calls = {"launched": 0, "run": [], "docker": ["not running", "not running", "running"]}
    monkeypatch.setattr(ns, "_docker", lambda: calls["docker"].pop(0) if len(calls["docker"]) > 1 else calls["docker"][0])
    monkeypatch.setattr(ns, "_docker_running_procs", lambda: False)
    monkeypatch.setattr(ns, "_stale_socket_dirs", lambda: [])
    monkeypatch.setattr(ns, "_launch_docker", lambda: calls.__setitem__("launched", calls["launched"] + 1) or True)
    monkeypatch.setattr(ns, "_run", lambda cmd, timeout, cwd=None: calls["run"].append((cmd, cwd)) or (0, ""))
    monkeypatch.setattr(ns.shutil, "which", lambda name: name)
    nango._cache.clear()
    calls["compose"] = compose
    return calls


def test_start_brings_up_docker_and_nango_and_keeps_the_key(local_nango, fake_nango):
    assert not nango.configured()
    s = ns.start()
    assert local_nango["launched"] == 1
    assert (["docker", "compose", "up", "-d"], local_nango["compose"]) in local_nango["run"]
    env = (local_nango["compose"] / ".env").read_text()
    key = [l.split("=", 1)[1] for l in env.splitlines() if l.startswith("NANGO_ENCRYPTION_KEY=")][0]
    assert len(key) == 44 and "FLAG_AUTH_ENABLED=false" in env          # 32 bytes, base64; dashboard left open
    assert nango.configured() and nango._key() == "sk-test" and nango._base() == fake_nango.url
    assert s["set_up"] and s["nango_running"]
    # the second run changes nothing: same encryption key, no second launch
    local_nango["run"].clear()
    ns.start()
    assert local_nango["run"] == [] and local_nango["launched"] == 1
    assert key in (local_nango["compose"] / ".env").read_text()


def test_start_asks_for_the_key_when_the_dashboard_wants_a_login(local_nango, fake_nango, monkeypatch):
    fake_nango.dashboard_open = False
    monkeypatch.setattr(ns.webbrowser, "open", lambda u: True)
    log = asks(monkeypatch, "sk-test")
    ns.start(dashboard_login=True)
    assert "FLAG_AUTH_ENABLED=true" in (local_nango["compose"] / ".env").read_text()
    assert [(secret, opts) for _, opts, secret in log] == [(True, None)] and "API Keys" in log[0][0]
    assert nango._key() == "sk-test"
    # a wrong key is refused, and not saved
    (nango._state_dir() / "secret_key").unlink()
    asks(monkeypatch, "nope")
    with pytest.raises(nango.NangoError, match="didn't accept that key"):
        ns.start(dashboard_login=True)
    assert not (nango._state_dir() / "secret_key").exists()


def test_start_warns_when_an_old_key_in_the_environment_would_win(local_nango, monkeypatch):
    monkeypatch.setenv("NANGO_SECRET_KEY", "from-another-nango")
    s = ns.start()
    assert "different key and takes priority" in s["warning"]


def test_start_stops_with_install_instructions_when_docker_is_missing(local_nango, monkeypatch):
    monkeypatch.setattr(ns, "_docker", lambda: "not installed")
    with pytest.raises(nango.NangoError, match="Docker isn't installed.*docker.com"):
        ns.start()
    assert local_nango["run"] == []


# -- Docker's leftover files (Windows) -------------------------------------------------------------------------------

@pytest.fixture
def leftovers(tmp_path, monkeypatch):
    base = tmp_path / "Local"
    (base / "Docker" / "run").mkdir(parents=True)
    (base / "Docker" / "run" / "dockerInference").write_text("")
    (base / "docker-secrets-engine").mkdir()
    (base / "docker-secrets-engine" / "engine.sock").write_text("")
    (base / "Docker" / "log").mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(base))
    monkeypatch.setattr(ns, "IS_WINDOWS", True)
    monkeypatch.setattr(ns, "_is_stale_socket", lambda e: e.name in ("dockerInference", "engine.sock"))
    killed = []
    monkeypatch.setattr(ns, "_kill_docker", lambda: killed.append(1))
    return base, killed


def test_docker_unstick_moves_the_leftover_folders_aside_after_asking(leftovers, monkeypatch):
    base, killed = leftovers
    log = asks(monkeypatch, "Move them aside")
    r = ns.docker_unstick()
    assert len(r["moved"]) == 2 and killed == [1]
    assert not (base / "Docker" / "run").exists() and not (base / "docker-secrets-engine").exists()
    assert any(p.name.startswith("run.stale-") and (p / "dockerInference").exists() for p in (base / "Docker").iterdir())
    assert "renamed, not deleted" in log[0][0] and "docker-secrets-engine" in log[0][0]
    assert ns.docker_unstick()["moved"] == []        # nothing left to do


def test_docker_unstick_does_nothing_when_the_user_says_no(leftovers, monkeypatch):
    base, killed = leftovers
    asks(monkeypatch, "Don't")
    with pytest.raises(nango.Refused):
        ns.docker_unstick()
    assert (base / "Docker" / "run" / "dockerInference").exists() and killed == []


def test_start_fixes_leftovers_before_launching_docker(local_nango, leftovers, monkeypatch):
    base, _ = leftovers
    monkeypatch.setattr(ns, "_stale_socket_dirs", REAL_STALE_SOCKET_DIRS)   # the real scan, on the fake folders
    monkeypatch.setattr(ns, "_launch_docker", lambda: local_nango.__setitem__("launched", 1) or True)
    monkeypatch.setattr(ns, "_docker", lambda: "running" if local_nango["launched"] else "not running")
    log = asks(monkeypatch, "Move them aside")
    ns.start()
    assert local_nango["launched"] == 1 and not (base / "docker-secrets-engine").exists()
    assert "stop it from starting" in log[0][0]


# -- a browser the user can sign in to -----------------------------------------------------------------------------

def test_browser_mode_makes_a_visible_window_that_stays_signed_in(nango_dir, monkeypatch):
    monkeypatch.setenv("JEVH_BROWSER_HEADED", "0")
    monkeypatch.delenv("JEVH_BROWSER_DOCK", raising=False)
    monkeypatch.delenv("JEVH_BROWSER_PROFILE", raising=False)
    r = ns.nango_browser = ns.browser_mode()
    import os
    assert os.environ["JEVH_BROWSER_HEADED"] == "1" and r["visible"] and r["persistent"]
    assert os.environ["JEVH_BROWSER_PROFILE"] == str(nango_dir / "browser-profile") and (nango_dir / "browser-profile").is_dir()
    monkeypatch.setenv("JEVH_BROWSER_DOCK", "/dock")      # the dashboard's own browser panel: leave it alone
    monkeypatch.setenv("JEVH_BROWSER_HEADED", "0")
    r = ns.browser_mode(persist=False)
    assert os.environ["JEVH_BROWSER_HEADED"] == "0" and r["visible"] and not r["persistent"]
