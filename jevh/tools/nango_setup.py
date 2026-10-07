"""First-time setup for the `nango` tool: Nango itself, an app at the provider, and connecting an account.

Reached through `nango` (nango.status(), nango.start(), ...); the nango-setup skill walks a user through
them. Everything that can be done by code is done here, so a model only has to order the steps and talk.

    nango.status()                 # what is set up and what the next step is
    nango.start()                  # Docker + Nango on this machine, and the secret key (a few minutes the first time)
    nango.providers("calendar")    # find an app in Nango's catalog (~1000)
    nango.guide("google-mail")     # what the user does at the provider: links, fields, the callback URL
    nango.setup("google-mail")     # create the integration; asks the user for the app's ID and secret (hidden)
    nango.connect_and_wait("google-mail")   # opens the sign-in page in the user's browser and waits for it
    nango.test("google-mail")      # one harmless read to prove it works

Secrets: the client secret is asked for with ask_user(secret=True) from inside setup(), or found by code on a
page (setup(app, page=browser.read())); it is never returned, printed or kept, and an error from Nango is
scrubbed of it.
"""
from __future__ import annotations

import base64
import json
import os
import re
import secrets as _secrets
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

from jevh.config import ROOT
from jevh.tools import nango as core
from jevh.tools.nango import NangoError, Refused

COMPOSE_DIR = ROOT / "nango"
LOCAL_URL = "http://localhost:3003"
DOCKER_DOWNLOAD = "https://www.docker.com/products/docker-desktop/"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
IS_WINDOWS = os.name == "nt"
POLL = 3.0   # seconds between looks at Nango while waiting for a new connection, or for Nango to come up
# sign-in modes with no app to register at the provider: the user types their details on Nango's sign-in page
NO_APP_MODES = ("API_KEY", "BASIC", "NONE", "TWO_STEP", "JWT", "SIGNATURE", "BILL", "AWS_SIGV4")
_offered: set[str] = set()   # pages the tool itself pointed at: the only https pages open_page() opens


# -- what a user does at each provider ----------------------------------------------------------------------

GOOGLE_NAME = {"google-mail": "Gmail", "google-calendar": "Google Calendar", "google-drive": "Google Drive",
               "google-docs": "Google Docs", "google-sheet": "Google Sheets"}
GOOGLE_API = {"google-mail": "gmail.googleapis.com", "google-calendar": "calendar-json.googleapis.com",
              "google-drive": "drive.googleapis.com", "google-docs": "docs.googleapis.com",
              "google-sheet": "sheets.googleapis.com"}
G = "https://www.googleapis.com/auth/"
SCOPES = {   # access -> scopes; what the harness asks Google for. Writes still need the user's OK, call by call.
    "google-mail": {"read": G + "gmail.readonly", "write": G + "gmail.modify"},
    "google-calendar": {"read": G + "calendar.readonly", "write": G + "calendar"},
    "google-drive": {"read": G + "drive.readonly", "write": G + "drive"},
    "google-docs": {"read": G + "documents.readonly", "write": G + "documents"},
    "google-sheet": {"read": G + "spreadsheets.readonly", "write": G + "spreadsheets"},
    "github": {"read": "repo,read:user", "write": "repo,read:user"},   # GitHub has no read-only scope for private repos
}
SCOPE_NOTES = {"github": "GitHub has no read-only scope for private repositories: 'repo' is read and write either way "
                         "(every change still asks you first)."}
ALIASES = {"gmail": "google-mail", "google mail": "google-mail", "email": "google-mail",
           "calendar": "google-calendar", "google calendar": "google-calendar", "gcal": "google-calendar",
           "drive": "google-drive", "google drive": "google-drive", "gdrive": "google-drive",
           "docs": "google-docs", "google docs": "google-docs", "sheets": "google-sheet", "google sheets": "google-sheet",
           "google-sheets": "google-sheet", "sheet": "google-sheet", "git hub": "github"}
# everyday words people search with, and the apps' own names for them
SEARCH_WORDS = {"email": ("gmail", "outlook"), "mail": ("gmail", "outlook"), "files": ("drive", "dropbox", "onedrive", "box"),
                "storage": ("drive", "dropbox", "onedrive", "box"), "chat": ("slack", "discord", "teams"),
                "messages": ("slack", "discord", "teams", "telegram"), "tasks": ("todoist", "ticktick", "asana", "trello"),
                "todo": ("todoist", "ticktick"), "notes": ("notion", "readwise"), "code": ("github", "gitlab"),
                "accounting": ("quickbooks", "xero", "wave", "freshbooks"), "fitness": ("strava", "fitbit", "garmin", "oura", "whoop"),
                "music": ("spotify",), "payments": ("stripe", "paypal", "square")}
# a safe read per provider that shows whose account it is: (path, headers, how to name the answer)
SMOKE = {
    "google-mail": ("/gmail/v1/users/me/profile", {}, lambda r: r.get("emailAddress")),
    "google-calendar": ("/calendar/v3/users/me/calendarList?maxResults=1", {},
                        lambda r: (r.get("items") or [{}])[0].get("summary")),
    "google-drive": ("/drive/v3/about?fields=user", {}, lambda r: (r.get("user") or {}).get("emailAddress")),
    "github": ("/user", {}, lambda r: r.get("login")),
    "slack": ("/api/auth.test", {}, lambda r: f"{r.get('user')} in {r.get('team')}" if r.get("ok") else None),
    "notion": ("/v1/users/me", {"Notion-Version": "2022-06-28"}, lambda r: r.get("name")),
}


def _google_steps(provider: str, callback: str) -> list[dict]:
    api = GOOGLE_API.get(provider)
    return [
        {"title": "Make a Google Cloud project (once; Gmail, Calendar and Drive can share it)",
         "url": "https://console.cloud.google.com/projectcreate",
         "do": "Sign in with the Google account whose data you want to use. Name the project e.g. 'jev-harness' and "
               "Create. (If you already have a project for this, use it.)"},
        {"title": f"Turn on the {GOOGLE_NAME.get(provider, 'Google')} API",
         "url": f"https://console.cloud.google.com/apis/library/{api}" if api else "https://console.cloud.google.com/apis/library",
         "do": "Make sure your project is the one selected at the top, then click Enable."},
        {"title": "Set up the consent screen (once per project)",
         "url": "https://console.cloud.google.com/auth/overview",
         "do": "Get started. App name e.g. 'Jev harness', your email as the support email. Audience: External. Your "
               "email again as the contact, agree to the policy, Create. (It is a four-part form: App information, "
               "Audience, Contact information, Finish; Next moves on, and reloading the page starts it over. "
               "'User support email' is a dropdown menu, not a text box: browser.do(values={'User support email': "
               "'<their email>'}) or browser.select(i, '<their email>') opens it and picks the entry.)"},
        {"title": "Publish the app, or Google signs you out every 7 days",
         "url": "https://console.cloud.google.com/auth/audience",
         "do": "Under Audience, Publishing status 'Testing' -> Publish app -> Confirm. For your own account this needs "
               "no Google review; when you connect, Google warns the app is unverified: Advanced -> continue. "
               "(Staying in Testing also works if you add yourself as a test user, but its logins expire after a week.)"},
        {"title": "Create the OAuth client (this gives the Client ID and secret)",
         "url": "https://console.cloud.google.com/auth/clients/create",
         "do": f"Application type: Web application. Name: Nango. Authorized redirect URIs: add exactly {callback} . "
               "Create. Google then shows the Client ID and Client secret: keep that window open, "
               "the secret may not be shown again."},
    ]


def _github_steps(callback: str) -> list[dict]:
    return [
        {"title": "Register an OAuth app on GitHub",
         "url": "https://github.com/settings/applications/new",
         "do": f"Application name e.g. 'Jev harness'. Homepage URL: http://localhost (anything works). Authorization "
               f"callback URL: exactly {callback} . Leave Device Flow off. Register application."},
        {"title": "Get its Client ID and secret",
         "url": None,
         "do": "On the app's page, copy the Client ID, then 'Generate a new client secret' and copy it (GitHub shows it once)."},
    ]


# -- small helpers -------------------------------------------------------------------------------------------

def _run(cmd: list[str], timeout: float, cwd: Path | None = None) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, creationflags=_NO_WINDOW,
                       encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _is_local(url: str) -> bool:
    return (urllib.parse.urlparse(url).hostname or "") in ("localhost", "127.0.0.1", "::1")


def _health(url: str, timeout: float = 4) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _say(msg: str) -> None:
    print(msg, flush=True)   # a long step (Docker, image downloads) is not silent in the worker's output


def _ask(question: str, options: list | None = None, secret: bool = False) -> str:
    answer = core._ask(question, options, secret)
    if answer is None:
        raise NangoError("there is no user here to ask: " + question.splitlines()[0])
    return answer


# -- Docker (Windows, macOS) -----------------------------------------------------------------------------------

def _docker() -> str:
    """not installed | not running | running."""
    if not shutil.which("docker"):
        return "not installed"
    try:
        rc, _ = _run(["docker", "info", "--format", "{{.ServerVersion}}"], 20)
    except (subprocess.TimeoutExpired, OSError):
        return "not running"
    return "running" if rc == 0 else "not running"


def _docker_exe() -> Path | None:
    if not IS_WINDOWS:
        return None
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramW6432"), r"C:\Program Files"):
        exe = Path(base or "") / "Docker" / "Docker" / "Docker Desktop.exe"
        if exe.is_file():
            return exe
    return None


def _launch_docker() -> bool:
    """Start Docker Desktop. False: nothing here to start (Linux runs Docker as a service)."""
    exe = _docker_exe()
    if exe:
        os.startfile(str(exe))   # noqa: S606 - detached, so it outlives this REPL
        return True
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-a", "Docker"])
        return True
    return False


DOCKER_PROCS = ("Docker Desktop.exe", "com.docker.backend.exe", "docker-ai.exe", "docker-desktop.exe")


def _docker_running_procs() -> bool:
    if not IS_WINDOWS:
        return False
    rc, out = _run(["tasklist", "/FO", "CSV", "/NH"], 20)
    return any(f'"{n.lower()}"' in out.lower() for n in DOCKER_PROCS[:2])


def _kill_docker() -> None:
    for n in DOCKER_PROCS:
        _run(["taskkill", "/F", "/IM", n], 20)
    time.sleep(2)


def _is_stale_socket(entry: os.DirEntry) -> bool:
    """A dead unix-socket file Windows can't open or delete (what a crashed Docker Desktop leaves)."""
    try:
        st = entry.stat(follow_symlinks=False)
    except OSError:
        return True   # "The file cannot be accessed by the system" is the sign
    return bool(getattr(st, "st_reparse_tag", 0) == 0x80000023)   # IO_REPARSE_TAG_AF_UNIX


def _stale_socket_dirs() -> list[Path]:
    """Folders Docker Desktop (Windows) can't start from: they hold socket files left by an unclean exit."""
    base = os.environ.get("LOCALAPPDATA")
    if not IS_WINDOWS or not base:
        return []
    out = []
    for d in (Path(base) / "Docker" / "run", Path(base) / "docker-secrets-engine"):
        try:
            if d.is_dir() and any(_is_stale_socket(e) for e in os.scandir(d)):
                out.append(d)
        except OSError:
            continue
    return out


def docker_unstick(approve: bool = True) -> dict:
    """Docker Desktop on Windows can fail to start with "remove ...engine.sock: The file cannot be accessed by
    the system": a crash left socket files that can't be deleted. Moving their folders aside fixes it (Docker
    makes new ones). Asks the user first; nothing is deleted."""
    dirs = _stale_socket_dirs()
    if not dirs:
        return {"moved": [], "note": "no leftover Docker socket files found"}
    if approve:
        names = "\n".join(f"  {d}" for d in dirs)
        ans = _ask("Docker Desktop left files behind that stop it from starting. Move these folders aside "
                   f"(renamed, not deleted; Docker makes fresh ones)?\n{names}", ["Move them aside", "Don't"])
        if not (ans.lower().startswith("move") or ans.strip().lower() in ("yes", "y", "ok", "okay", "allow", "sure")):
            raise Refused("the user did not allow moving Docker's leftover files aside")
    _kill_docker()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    moved = []
    for d in dirs:
        new = d.with_name(f"{d.name}.stale-{stamp}")
        d.rename(new)
        moved.append(str(new))
    return {"moved": moved, "note": "moved aside; start Docker Desktop again"}


def _docker_log_has_socket_error() -> bool:
    base = os.environ.get("LOCALAPPDATA")
    if not IS_WINDOWS or not base:
        return False
    log = Path(base) / "Docker" / "log" / "host" / "com.docker.backend.exe.log"
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 120_000))
            return b"cannot be accessed by the system" in f.read()
    except OSError:
        return False


def _ensure_docker(wait: float) -> None:
    state = _docker()
    if state == "running":
        return
    if state == "not installed":
        raise NangoError(f"Docker isn't installed. Install Docker Desktop ({DOCKER_DOWNLOAD}), open it once, "
                         "then run nango.start() again.")
    if not _docker_running_procs() and _stale_socket_dirs():
        _say("Docker left files behind from its last shutdown that stop it starting.")
        docker_unstick()
    if not _docker_running_procs():
        _say("Starting Docker Desktop...")
        if not _launch_docker():
            raise NangoError("Docker isn't running and can't be started from here: start it (on Linux: "
                             "`sudo systemctl start docker`), then run nango.start() again.")
    deadline = time.monotonic() + wait
    retried = False
    while time.monotonic() < deadline:
        if _docker() == "running":
            return
        if not retried and _docker_log_has_socket_error() and _stale_socket_dirs():
            _say("Docker Desktop stopped on its leftover files.")
            docker_unstick()
            _launch_docker()
            retried = True
        time.sleep(POLL)
    raise NangoError("Docker Desktop didn't come up in time. Open it yourself and look for an error message; "
                     "then run nango.start() again.")


# -- Nango itself -----------------------------------------------------------------------------------------------

def _env_file() -> Path:
    return COMPOSE_DIR / ".env"


def _ensure_env(dashboard_login: bool) -> None:
    """nango/.env: the key that encrypts the stored logins, and whether the dashboard asks for a login."""
    f = _env_file()
    text = f.read_text(encoding="utf-8") if f.is_file() else ""
    if not re.search(r"^NANGO_ENCRYPTION_KEY=\S+", text, re.M):
        key = base64.b64encode(_secrets.token_bytes(32)).decode()
        text = re.sub(r"^NANGO_ENCRYPTION_KEY=.*\n?", "", text, flags=re.M)
        text = f"NANGO_ENCRYPTION_KEY={key}\n" + text
        _say("Made the key that encrypts your stored logins: nango/.env. Keep a copy somewhere safe, "
             "because it can't be replaced later.")
    if not re.search(r"^FLAG_AUTH_ENABLED=", text, re.M):
        text += f"FLAG_AUTH_ENABLED={'true' if dashboard_login else 'false'}\n"
    f.write_text(text, encoding="utf-8")


def _compose_up(timeout: float = 900) -> None:
    cmds = [["docker", "compose", "up", "-d"], ["docker-compose", "up", "-d"]]
    for cmd in cmds:
        if shutil.which(cmd[0]) is None:
            continue
        rc, out = _run(cmd, timeout, cwd=COMPOSE_DIR)
        if rc == 0:
            return
        if cmd[0] == "docker" and "is not a docker command" not in out:
            raise NangoError(f"`docker compose up -d` failed: {out.strip()[-600:]}")
    raise NangoError("Docker Compose isn't available: update Docker Desktop.")


def _fetch_dashboard_key(url: str) -> str | None:
    """The environment's secret key from the dashboard's own API. Works when the dashboard has no login
    (FLAG_AUTH_ENABLED=false); with a login it answers 401 and the user copies the key instead."""
    try:
        with urllib.request.urlopen(url + "/api/v1/environment/api-keys?env=dev", timeout=10) as r:
            data = json.loads(r.read()).get("data") or []
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return next((k["secret"] for k in data if k.get("secret")), None)


def _key_works(url: str, key: str) -> bool:
    req = urllib.request.Request(url + "/integrations", headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def status() -> dict:
    """What is set up, and the next step. Cheap and safe to call any time."""
    url = core._base()
    out: dict[str, Any] = {"set_up": core.configured(), "url": url,
                           "where": "this machine" if _is_local(url) else
                                    "Nango Cloud" if url == core.DEFAULT_URL else url}
    out["docker"] = _docker() if shutil.which("docker") or not core.configured() else "n/a"
    out["nango_running"] = _health(url) if (core.configured() or _is_local(url)) else False
    if not core.configured():
        out["next"] = ("nango.start() sets Nango up on this machine (needs Docker); or use Nango Cloud: the user "
                       "signs up at https://app.nango.dev and sets NANGO_SECRET_KEY")
        return out
    if not out["nango_running"]:
        out["next"] = "Nango isn't reachable: nango.start() starts it" if _is_local(url) else \
                      f"can't reach Nango at {url}"
        return out
    try:
        conns = core.connections(refresh=True)
        out["integrations"] = [i["integration"] for i in core.integrations()]
        out["connected"] = [c["integration"] + ("" if c["ok"] else " (needs reconnecting)") for c in conns]
    except NangoError as e:
        out["problem"] = str(e)
        out["next"] = "Nango answers but the key isn't accepted: nango.start() fetches the right one"
        return out
    out["next"] = ("everything is set up: use it, or nango.guide(app) + nango.setup(app) to add another app"
                   if out["connected"] else
                   "add an app: nango.guide(app), nango.setup(app), then nango.connect_and_wait(app)")
    return out


def start(dashboard_login: bool = False, wait: float = 240) -> dict:
    """Get Nango running on this machine and this harness pointed at it: Docker, nango/docker-compose.yaml,
    the secret key. Safe to repeat (it only does what's missing). The first run downloads about 400 MB.
    dashboard_login=False leaves Nango's dashboard (http://localhost:3003) open to this machine, which is what
    lets the key be fetched for you; True asks for a Nango account (it stays in the local database) and then
    for you to paste the key."""
    if not (COMPOSE_DIR / "docker-compose.yaml").is_file():
        raise NangoError(f"{COMPOSE_DIR / 'docker-compose.yaml'} is missing: this checkout has no local Nango setup")
    url = LOCAL_URL
    if _health(url) and core.configured() and _key_works(url, core._key()):
        _say("Nango is already running and set up.")
        core._save_state("url", url)
        return status()
    _ensure_docker(wait)
    _ensure_env(dashboard_login)
    _say("Starting Nango (the first time downloads its images: a few minutes)...")
    _compose_up()
    deadline = time.monotonic() + wait
    while not _health(url):
        if time.monotonic() > deadline:
            raise NangoError("Nango didn't answer in time: `docker compose logs nango-server` in the nango/ folder "
                             "shows why")
        time.sleep(POLL)
    key = _fetch_dashboard_key(url)
    if not key:
        page = f"{url}/dev/environment-settings#api-keys"
        _say(f"The dashboard needs a login, so the key can't be fetched. Opening it: {page}")
        open_page(page)
        key = _ask("Nango is running. In the dashboard sign up (or sign in), open Environment Settings > API Keys, "
                   "and paste the secret key here. It is typed hidden and kept only on this machine.",
                   secret=True).strip()
    if not _key_works(url, key):
        raise NangoError("Nango didn't accept that key: it should be the 'secret key' on Environment Settings > API Keys")
    core._save_state("secret_key", key, private=True)
    core._save_state("url", url)
    core._cache.clear()
    _say("Nango is running and the harness is connected to it.")
    out = status()
    env_key = os.environ.get("NANGO_SECRET_KEY")
    if env_key and env_key != key:   # the environment wins over the saved key: say so, or every call 401s
        out["warning"] = ("NANGO_SECRET_KEY in this machine's environment is a different key and takes priority: "
                          "remove it (Windows: Settings > Environment Variables), then restart the harness")
    return out


# -- finding apps and what the user does at each one ----------------------------------------------------------

def _catalog() -> dict[str, dict]:
    def fetch():
        data = core._nango("GET", "/providers")
        items = data.get("data", data) if isinstance(data, dict) else data
        return {i["name"]: i for i in items if i.get("name")}
    return core._cached("providers", 3600, fetch)


def providers(query: str | None = None, limit: int = 12) -> list[dict]:
    """Search Nango's ~1000 supported apps: [{name, display_name, auth_mode, categories}]. `name` is what
    guide() and setup() take. No query: the popular ones."""
    words = [w for w in re.split(r"\W+", (query or "").lower()) if w]
    if len(words) == 1 and words[0] in SEARCH_WORDS:     # "email" -> Gmail, Outlook
        seen: dict[str, dict] = {}
        for alt in SEARCH_WORDS[words[0]]:
            for hit in providers(alt, limit):
                seen.setdefault(hit["name"], hit)
        return list(seen.values())[:limit]
    cat = _catalog()
    rows = []
    for name, p in cat.items():
        hay = f"{name} {p.get('display_name', '')} {' '.join(p.get('categories') or [])}".lower()
        if words and not all(w in hay for w in words):
            continue
        if not words and "popular" not in (p.get("categories") or []):
            continue
        # exact name first, then the plain one (not -sandbox, -mcp, -scim, ...) before its variants, then the
        # popular ones ("calendar" -> Google Calendar before Apple's), then the shorter name
        rank = (name != "-".join(words), bool(re.search(r"sandbox|-mcp|scim|-pat|-api-key|-oauth|-demo", name)),
                "popular" not in (p.get("categories") or []), len(name))
        rows.append((rank, {"name": name, "display_name": p.get("display_name"), "auth_mode": p.get("auth_mode"),
                            "categories": p.get("categories") or []}))
    return [r for _, r in sorted(rows, key=lambda x: x[0])][:limit]


def _provider(name: str) -> str:
    """The provider name for what the user or the model said ("Gmail", "google calendar", "github")."""
    n = name.strip().lower()
    n = ALIASES.get(n, n).replace(" ", "-")
    cat = _catalog()
    if n in cat:
        return n
    hits = providers(name, 5)
    if len(hits) == 1:
        return hits[0]["name"]
    if hits and hits[0]["name"] == n:
        return n
    raise NangoError(f"no app called '{name}' in Nango's catalog" + (
        "; did you mean: " + ", ".join(f"{h['name']} ({h['display_name']})" for h in hits) if hits else
        ": nango.providers('<word>') searches it"))


def guide(provider: str) -> dict:
    """What the user does at `provider` before an account can be connected: {provider, display_name, auth_mode,
    callback_url, steps: [{title, url, do}], scopes, fields, setup_guide_url, next}. Read the steps to the
    user one at a time (open_page(url) opens a step's page); for an app without written steps, read
    setup_guide_url with web.read(url, question) and walk them through that."""
    name = _provider(provider)
    detail = core._nango("GET", f"/providers/{name}")
    detail = detail.get("data", detail)
    mode = detail.get("auth_mode")
    callback = core._base() + "/oauth/callback"
    guide_url = detail.get("setup_guide_url") or detail.get("docs")
    out: dict[str, Any] = {"provider": name, "display_name": detail.get("display_name"), "auth_mode": mode,
                           "callback_url": callback, "setup_guide_url": guide_url, "docs": detail.get("docs")}
    if detail.get("docs_connect"):
        out["connect_guide_url"] = detail["docs_connect"]
    if name in GOOGLE_API:
        steps = _google_steps(name, callback)
    elif name == "github":
        steps = _github_steps(callback)
    elif mode in ("OAUTH2", "OAUTH1"):
        steps = [{"title": f"Register an app at {detail.get('display_name') or name}'s developer site",
                  "url": guide_url,
                  "do": f"Nango's page for it lists the exact clicks (read it with web.read and walk the user through). "
                        f"When it asks for a redirect or callback URL, use exactly {callback} . You end up with a "
                        "Client ID and a Client secret."}]
    elif mode in ("API_KEY", "BASIC"):
        needs = _field_lines(detail)
        steps = [{"title": "Nothing to register: find the login details, and type them on Nango's sign-in page",
                  "url": detail.get("docs_connect") or guide_url,
                  "do": ("You need: " + "; ".join(needs) + ". " if needs else "")
                        + "Nango's page for this app says where to find them (read it with web.read and walk the "
                          "user through it). They type them on Nango's sign-in page (step 6), never here."}]
    else:
        steps = [{"title": f"This app signs in with {mode}", "url": guide_url,
                  "do": "Read Nango's page for it and follow it; connect_and_wait() does the last step."}]
    out["steps"] = steps
    for st in steps:
        if st.get("url"):
            _offered.add(st["url"])
    for u in (guide_url, detail.get("docs"), detail.get("docs_connect")):
        if u:
            _offered.add(u)
    if name in SCOPES:
        out["scopes"] = SCOPES[name]
        if name in SCOPE_NOTES:
            out["scope_note"] = SCOPE_NOTES[name]
    fields = {**{k: v.get("title", k) for k, v in (detail.get("credentials") or {}).items()},
              **{k: v.get("title", k) for k, v in (detail.get("connection_config") or {}).items()}}
    if fields:
        out["fields"] = fields
    out["next"] = ("when the user has the Client ID and secret: nango.setup(%r)" % name) if mode in ("OAUTH2", "OAUTH1") \
        else f"nango.setup({name!r}) then nango.connect_and_wait({name!r})"
    return out


def _field_lines(detail: dict) -> list[str]:
    """"Apple ID (Your Apple ID email address.) e.g. you@icloud.com" for each login detail an app asks for."""
    out = []
    for spec in {**(detail.get("credentials") or {}), **(detail.get("connection_config") or {})}.values():
        line = spec.get("title") or ""
        if spec.get("description"):
            line += f" ({spec['description'].rstrip('.')})"
        if spec.get("example"):
            line += f" e.g. {spec['example']}"
        if line:
            out.append(line)
    return out


def _find_credentials(provider: str, page: str) -> tuple[str | None, str | None]:
    """The OAuth Client ID and secret on a page of text, for the apps whose consoles show them in a known form."""
    if provider.startswith("google-"):
        cid = re.search(r"\b\d+-[a-z0-9_]+\.apps\.googleusercontent\.com\b", page)
        sec = re.search(r"\bGOCSPX-[A-Za-z0-9_-]{10,}", page)
        return (cid.group(0) if cid else None), (sec.group(0) if sec else None)
    if provider == "github":
        cid = re.search(r"\b(?:Iv1\.[0-9a-f]{16}|Ov23[0-9A-Za-z]{16})\b", page)
        sec = re.search(r"\b[0-9a-f]{40}\b", page)
        return (cid.group(0) if cid else None), (sec.group(0) if sec else None)
    return None, None


def _check_ids(provider: str, client_id: str, client_secret: str) -> None:
    """Catch the usual slip (ID and secret swapped, or something else pasted) before Nango stores it."""
    cid, sec = client_id.strip(), client_secret.strip()
    if not cid or not sec:
        raise NangoError("the Client ID and the Client secret are both needed")
    if provider.startswith("google-"):
        if sec.endswith(".apps.googleusercontent.com") or cid.startswith("GOCSPX-"):
            raise NangoError("those look swapped: the Client ID ends in .apps.googleusercontent.com, "
                             "the secret starts with GOCSPX-")
        if not cid.endswith(".apps.googleusercontent.com"):
            raise NangoError("that doesn't look like a Google Client ID (it ends in .apps.googleusercontent.com)")
    if provider == "github" and not re.fullmatch(r"(Iv1\.|Iv23)?[0-9A-Za-z]{16,}", cid):
        raise NangoError("that doesn't look like a GitHub Client ID (20 letters and digits)")


def setup(provider: str, client_id: str | None = None, client_secret: str | None = None,
          access: str = "read", scopes: str | list | None = None, unique_key: str | None = None,
          page: str | None = None) -> dict:
    """Add `provider` to Nango (or update it): the step after the user registered an app at the provider.
    OAuth apps need the app's Client ID and secret: leave them out and the user is asked (the secret typed
    hidden), or pass `page=browser.read()` while the console shows them (found by code, never returned), or
    pass them as variables; never print or save them. `access`
    "read" or "write" picks the scopes for apps nango.guide() lists them for; else pass `scopes`.
    Returns {integration, scopes, next}; the secret is not in it. New scopes apply to accounts connected after."""
    name = _provider(provider)
    detail = core._nango("GET", f"/providers/{name}")
    detail = detail.get("data", detail)
    mode = detail.get("auth_mode")
    key = unique_key or name
    body: dict[str, Any] = {"unique_key": key, "provider": name, "display_name": detail.get("display_name") or name}
    chosen = None
    if mode == "OAUTH2":
        if access not in ("read", "write"):
            raise NangoError("access is 'read' or 'write'")
        chosen = scopes if scopes is not None else SCOPES.get(name, {}).get(access)
        if isinstance(chosen, (list, tuple)):
            chosen = ",".join(chosen)
        if chosen is None:
            raise NangoError(f"which scopes {name} needs isn't known here: read Nango's page for it "
                             f"({detail.get('setup_guide_url') or detail.get('docs')}) with web.read, ask the user "
                             "what they want to do, and pass scopes='a,b'")
        if page and not (client_id and client_secret):
            found_id, found_secret = _find_credentials(name, page)
            client_id, client_secret = client_id or found_id, client_secret or found_secret
        if not client_id:
            client_id = _ask(f"{detail.get('display_name') or name}: what is the app's Client ID? (the secret is next)",
                             None).strip()
        if not client_secret:
            client_secret = _ask(f"{detail.get('display_name') or name}: now the Client secret. It is typed hidden "
                                 "and goes straight to Nango on this machine.", secret=True).strip()
        _check_ids(name, client_id, client_secret)
        body["credentials"] = {"type": "OAUTH2", "client_id": client_id.strip(), "client_secret": client_secret.strip(),
                               "scopes": chosen}
    elif mode in ("OAUTH1", "OAUTH2_CC", "TBA", "APP", "CUSTOM", "INSTALL_PLUGIN", "MCP_OAUTH2", "MCP_OAUTH2_GENERIC"):
        raise NangoError(f"{name} signs in with {mode}, which setup() doesn't fill in: the user enters it in Nango's "
                         f"dashboard (Integrations > {name}); nango.guide({name!r}) has the link")
    exists = True
    try:
        core._nango("GET", f"/integrations/{key}")
    except NangoError as e:
        if "HTTP 404" not in str(e):
            raise
        exists = False
    try:
        if exists:
            core._nango("PATCH", f"/integrations/{key}", body={k: v for k, v in body.items() if k != "unique_key"})
        else:
            core._nango("POST", "/integrations", body=body)
    except NangoError as e:
        msg = str(e)
        for secret in (client_secret, client_id):
            if secret:
                msg = msg.replace(secret, "(hidden)")
        raise NangoError(msg) from None
    core._cache.clear()
    out = {"integration": key, "provider": name, "updated" if exists else "created": True,
           "next": f"nango.connect_and_wait({key!r}) opens the sign-in page for the user"}
    if chosen:
        out["scopes"] = chosen
        if exists:
            out["note"] = "an account connected earlier keeps its old scopes: connect again to apply these"
    return out


# -- connecting an account ----------------------------------------------------------------------------------------

def open_page(url: str) -> dict:
    """Open a page in the user's own browser (where they're already signed in). Only this machine's Nango and
    pages nango.guide() or the tool itself gave you."""
    parsed = urllib.parse.urlparse(url)
    ok = (parsed.scheme == "http" and _is_local(url)) or (parsed.scheme == "https" and url in _offered)
    if not ok:
        raise NangoError("open_page() opens this machine's Nango and the pages nango.guide() gave you; "
                         "show the user the link instead")
    try:
        return {"opened": bool(webbrowser.open(url)), "url": url}
    except webbrowser.Error as e:
        return {"opened": False, "url": url, "note": str(e)}


def connect_and_wait(integration: str, timeout: float | None = None, open_browser: bool = True) -> dict:
    """Open Nango's sign-in page for `integration` in the user's browser and wait until the account is
    connected. Returns {connected, connection_id}. The user signs in and allows access there; nothing is typed
    here. Waits `timeout` seconds (default 300, or JEVH_NANGO_CONNECT_TIMEOUT); on time out connected is False
    (the link is in the result, still valid for a while)."""
    timeout = float(timeout if timeout is not None else os.environ.get("JEVH_NANGO_CONNECT_TIMEOUT") or 300)
    if core._integration_key(integration) is None:
        name = _provider(integration)     # raises if Nango has no such app
        mode = _catalog()[name].get("auth_mode")
        if mode not in NO_APP_MODES:
            raise NangoError(f"{name} isn't set up in Nango yet, and it needs an app registered at the provider first: "
                             f"nango.guide({name!r}) says how, then nango.setup({name!r}).")
        _say(f"Adding {name} to Nango first (it needs no app registered)...")
        setup(name)
        integration = name
    integration = core._integration_key(integration) or integration      # "github" -> "github-jed"
    before = {c["connection_id"] for c in core.connections(refresh=True) if c["integration"] == integration}
    link = core.connect(integration)
    opened = False
    if open_browser:
        _offered.add(link)
        try:
            opened = bool(webbrowser.open(link))
        except webbrowser.Error:
            opened = False
    _say(("Opened the sign-in page in your browser" if opened else "Open this link in your browser")
         + f" (waiting up to {int(timeout // 60) or 1} min): {link}")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(POLL)
        try:
            new = [c for c in core.connections(refresh=True) if c["integration"] == integration
                   and c["connection_id"] not in before]
        except NangoError:
            continue
        if new:
            return {"connected": True, "integration": integration, "connection_id": new[0]["connection_id"],
                    "next": f"nango.test({integration!r}) checks it works"}
    return {"connected": False, "integration": integration, "link": link,
            "note": "no new account appeared: the user may not have finished; ask, then call this again"}


def test(integration: str) -> dict:
    """One harmless read that shows the connection works and whose account it is: {ok, detail}."""
    key, conn = core._resolve(integration, None)
    prov = next((c["provider"] for c in core.connections() if c["integration"] == key), key)
    spec = SMOKE.get(prov)
    path = headers = who = None
    if spec:
        path, headers, who = spec
    else:
        try:
            ver = (core._nango("GET", f"/providers/{prov}").get("data") or {}).get("proxy", {}).get("verification") or {}
        except NangoError:
            ver = {}
        if ver.get("endpoints"):
            path, headers, who = ver["endpoints"][0], {}, None
    if not path:
        return {"ok": None, "note": f"no ready-made check for {prov}, and none is needed: the connection exists "
                                    "(nango.status() lists it). Say it's connected; only call the app's API when the "
                                    "user asks for something specific."}
    try:
        res = core.get(key, path, headers=headers)
    except NangoError as e:
        return {"ok": False, "detail": str(e)[:400]}
    detail = who(res) if (who and isinstance(res, dict)) else None
    return {"ok": True, "detail": detail or "the app answered"}


# -- a browser the user can sign in to, for the pages a model should drive ------------------------------------------

def browser_mode(visible: bool = True, persist: bool = True) -> dict:
    """Prepare the `browser` tool for a provider's website: a visible window the user can sign in on (the model
    never types a password), kept signed in between tasks. Call it BEFORE the first browser.goto() of the task.
    The watched browser is the user's own Chrome (jevh/tools/browser.py), which Google and Facebook take for an
    ordinary browser. If one still refuses ("this browser or app may not be secure"): don't work around it; do the
    steps in the user's own browser instead (open_page(url))."""
    mod = sys.modules.get("jevh.tools.browser")
    if mod is not None and getattr(mod, "_session", None) is not None:
        mod.close()   # a browser already started this task: restart it with the new settings
    notes = []
    prof = None
    if persist:
        prof = core._state_dir() / "browser-profile"
        prof.mkdir(parents=True, exist_ok=True)
        os.environ["JEVH_BROWSER_PROFILE"] = str(prof)
        notes.append("stays signed in between tasks (a profile on this machine)")
    if visible:   # (with the dashboard's dock folder set, a watched browser is shown in its Browser panel: no window)
        os.environ["JEVH_BROWSER_HEADED"] = "1"
    window = os.environ.get("JEVH_BROWSER_HEADED") == "1"
    docked = window and bool(os.environ.get("JEVH_BROWSER_DOCK"))
    if docked:   # the next turn of this session starts its browser the same way (live: it came up signed out, on a blank page)
        try:
            (Path(os.environ["JEVH_BROWSER_DOCK"]).parent / "browser-mode.json").write_text(
                json.dumps({"visible": True, "profile": str(prof) if prof else ""}), encoding="utf-8")
        except OSError:
            pass
    notes.append("shown in the dashboard's Browser panel (no separate window): the user signs in there, with the "
                 "mouse and keyboard on the picture; it stays open after this task, on the same page, until the session "
                 "ends - the user can finish a step in it, and your next turn finds it where it was" if docked
                 else "a window the user can see and sign in on" if window
                 else "hidden: the user can't sign in on it")
    return {"visible": window, "docked": docked, "persistent": persist, "notes": notes}
