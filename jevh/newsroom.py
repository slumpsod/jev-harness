"""Newsroom: a dashboard of widgets that keep an eye on things for you (posts from an X account, news on a topic
or about a company, an RSS feed, a subreddit, new releases of a project...) and tell you when something new
turns up. You set it up by asking in the chat ("keep an eye on Nvidia news and tell me when they mention
earnings"): the worker adds the widgets with the `newsroom` tool (jevh/tools/newsroom.py).

workspace/newsroom.db
  widgets   one per tile: its kind, the kind's settings (`params`), how often to look (`every`, minutes), what
            counts as worth a notification (`alert`), size and place on the board, and how the last look went
  items     what each widget found (newest first; each one once: a story seen again is not new)
  notes     the notifications: an alert from a widget, or a message a worker sent with newsroom.notify()

Two processes use it: the worker's REPL (adds and changes widgets, takes a first look so it can say what it
found) and the dashboard (shows the board, and its `Runner` looks again every widget's `every` minutes).
SQLite in WAL mode lets both write. Without the dashboard running, widgets keep what they have but are not
refreshed (`jevh newsroom refresh` does one pass by hand).

A KIND is the template for a widget: a name, the settings it takes, and one function that fetches items. To
add a source, add a kind (a few lines; `feed_kind` makes one from a feed's URL). The dashboard's "Add widget"
form, the worker's guide and `jevh newsroom kinds` are all generated from the registry, so nothing else changes.

    jevh newsroom                              # the board: widgets and what is new
    jevh newsroom kinds                        # what a widget can follow, and the settings each takes
    jevh newsroom add news topic="solid-state batteries" --alert "breakthrough, recall"
    jevh newsroom refresh [id]                 # look now (all widgets, or one)
    jevh newsroom notes                        # the notifications
    jevh newsroom rm 3
"""
from __future__ import annotations

import argparse
import hashlib
import html
import ipaddress
import json
import os
import re
import socket
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable

ABOUT_Q = ("Is the news item or post in `item` about what the user asked to be told about in `interest`? Say yes only if "
           "it clearly is about that, not just the same company or a similar word.")
UA = "Mozilla/5.0 (compatible; jev-harness-newsroom/1.0)"
MAX_FEED_BYTES = 2_000_000
MAX_NOTES_PER_LOOK = 5          # one look at one widget notifies at most this many times (then one "and N more")
SIZES = ("s", "m", "l")         # a tile: one column (headlines only), one column, two columns
FIELD_CAPS = {"title": 300, "url": 1000, "source": 120, "summary": 600, "author": 120}

DEFAULTS = {"about_threshold": 0.7, "enabled": True, "tick": 30, "min_minutes": 5, "default_minutes": 30, "keep_items": 100,
            "max_widgets": 40, "x_feed": "", "x_integration": "", "allow_local": False,
            "summary": True, "summary_items": 10, "summary_tokens": 700}


def settings(raw: dict | None) -> dict:
    """The `newsroom:` block of config.yaml over the defaults."""
    out = dict(DEFAULTS)
    out.update({k: v for k, v in ((raw or {}).get("newsroom") or {}).items() if k in DEFAULTS and v is not None})
    return out


class NewsroomError(ValueError):
    """Something the caller (a worker, the dashboard) can fix: the message says what."""


class SourceUnavailable(NewsroomError):
    """A source that can't be read until the user sets something up (X needs a token or a feed)."""


# ---------------------------------------------------------------- fetching

def _blocked_host(host: str) -> bool:
    """A host that is this machine or its network (or a cloud metadata address), unless allow_local is set.
    Feeds are fetched by the dashboard, with whatever a worker (reading the web) put in a widget."""
    if os.environ.get("JEVH_NEWSROOM_ALLOW_LOCAL") == "1":
        return False
    try:
        addrs = {ai[4][0] for ai in socket.getaddrinfo(host, None)}
    except OSError:
        return False   # not resolvable: the fetch itself will fail with a plain message
    for a in addrs:
        ip = ipaddress.ip_address(a.split("%")[0])
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            return True
    return False


class _Redirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def check_url(url: str) -> str:
    u = urllib.parse.urlparse(str(url).strip())
    if u.scheme not in ("http", "https") or not u.hostname:
        raise NewsroomError(f"not a web address (http:// or https://): {str(url)[:120]!r}")
    if _blocked_host(u.hostname):
        raise NewsroomError(f"{u.hostname} is on this machine or its network; a widget only follows the public web")
    return u.geturl()


def http_get(url: str, timeout: float = 15, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(check_url(url), headers={"User-Agent": UA, "Accept": "*/*", **(headers or {})})
    opener = urllib.request.build_opener(_Redirects)
    try:
        with opener.open(req, timeout=timeout) as r:
            return r.read(MAX_FEED_BYTES)
    except urllib.error.HTTPError as e:
        raise NewsroomError(f"{urllib.parse.urlparse(url).hostname} answered HTTP {e.code}") from None
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        raise NewsroomError(f"could not reach {urllib.parse.urlparse(url).hostname}: {getattr(e, 'reason', e)}") from None


def _text(s: str | None) -> str:
    s = html.unescape(re.sub(r"<[^>]+>", " ", s or ""))
    return re.sub(r"\s+", " ", s).strip()


def _when(s: str | None) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return parsedate_to_datetime(s).timestamp()
    except (TypeError, ValueError):
        pass
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp()
    except ValueError:
        return None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


class Items(list):
    """What parse_feed found, with the feed's own name in `.title`."""
    title = ""


def parse_feed(data: bytes, limit: int = 50) -> list[dict]:
    """RSS 2.0, RSS 1.0 (RDF) and Atom -> items [{title, url, source, published, summary, author, uid}], in feed order."""
    head = data[:4096].lower()
    if b"<!doctype" in head or b"<!entity" in head:
        raise NewsroomError("the feed uses an XML feature that is not accepted (DOCTYPE/ENTITY)")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        raise NewsroomError(f"not a feed (RSS or Atom): {e}") from None
    top = next((c for c in root if _local(c.tag) == "channel"), root)
    feed_title = _text(next((c.text for c in top if _local(c.tag) == "title"), ""))
    out = Items()
    out.title = feed_title
    for el in root.iter():
        if _local(el.tag) not in ("item", "entry"):
            continue
        f: dict[str, str] = {}
        link = ""
        for c in el:
            n = _local(c.tag)
            if n == "link":
                href = c.get("href")
                if href and (c.get("rel") in (None, "alternate")):
                    link = link or href
                elif c.text and c.text.strip():
                    link = link or c.text.strip()
            elif n == "source":   # Google News: <source url="...">Reuters</source>
                f["source"] = _text(c.text)
            elif n in ("title", "description", "summary", "content", "encoded", "pubdate", "published", "updated", "date",
                       "guid", "id", "creator"):
                f.setdefault(n, (c.text or "") if n != "content" else ET.tostring(c, encoding="unicode", method="text"))
            elif n == "author":
                nm = next((x.text for x in c if _local(x.tag) == "name"), None)
                f["author"] = _text(nm or c.text)
        if not link and f.get("guid", "").startswith("http"):
            link = f["guid"]
        title = _text(f.get("title"))
        if not title and not link:
            continue
        summary = _text(f.get("description") or f.get("summary") or f.get("content") or f.get("encoded"))
        source = f.get("source") or ""
        if source and title.endswith(f" - {source}"):   # Google News headlines end with the outlet: it is shown beside them
            title = title[:-len(source) - 3]
        if title and summary.startswith(title) and len(summary) - len(title) < 60:   # ...and its description is the headline again
            summary = ""
        out.append({"title": title or link, "url": link.strip(), "source": source,
                    "published": _when(f.get("pubdate") or f.get("published") or f.get("updated") or f.get("date")),
                    "summary": summary, "author": _text(f.get("author") or f.get("creator")),
                    "uid": (f.get("guid") or f.get("id") or "").strip()})
        if len(out) >= limit:
            break
    for it in out:
        it["source"] = it["source"] or feed_title or urllib.parse.urlparse(it["url"]).netloc
    return out


def _clean_item(it: dict) -> dict:
    out = {k: str(it.get(k) or "")[:cap] for k, cap in FIELD_CAPS.items()}
    if out["url"] and not out["url"].lower().startswith(("http://", "https://")):
        out["url"] = ""   # (never a javascript: link on the board)
    out["published"] = float(it["published"]) if it.get("published") else None
    out["uid"] = str(it.get("uid") or out["url"] or hashlib.sha1(out["title"].encode()).hexdigest())[:400]
    return out


# ---------------------------------------------------------------- kinds: the template for a widget

@dataclass
class Param:
    name: str
    help: str
    required: bool = False
    default: Any = None
    example: str = ""
    main: bool = False     # shown up front in the Add form even though it isn't required (the rest sit under "more")


@dataclass
class Kind:
    name: str
    label: str
    description: str                                  # what the worker (and the Add form) read
    params: list[Param]
    fetch: Callable[[dict], list[dict]]               # params -> items, newest first
    title: Callable[[dict], str]                      # a name for a widget with these params
    ready: Callable[[], str | None] = lambda: None    # None, or what the user must set up first
    icon: str = "news"
    # How many looks in a row an item the widget has never held must be in the results before it counts as new. 1: at
    # once (a feed is stable). More, for a source whose results shuffle between two identical requests (a search
    # engine's bottom places): there, a page that appears for one look is churn, not news.
    settle: int = 1

    def public(self) -> dict:
        return {"name": self.name, "label": self.label, "description": self.description, "icon": self.icon,
                "params": [asdict(p) for p in self.params], "needs": self.ready()}


KINDS: dict[str, Kind] = {}


def register(kind: Kind) -> Kind:
    KINDS[kind.name] = kind
    return kind


def feed_kind(name: str, label: str, description: str, url: str, params: list[Param], title: str,
              icon: str = "news", safe: dict[str, str] | None = None) -> Kind:
    """A kind that is one feed: `url` has {param} holes (values are URL-encoded), `title` the widget's default name.
    `safe`: per parameter, characters left as they are when it is encoded - the "/" in a GitHub project's owner/name,
    which encoded to %2F made GitHub answer 404 for every project."""
    def fetch(p: dict) -> list[dict]:
        quoted = {k: urllib.parse.quote(str(v), safe=(safe or {}).get(k, "")) for k, v in p.items()}
        return parse_feed(http_get(url.format_map(_Safe(quoted))))
    return register(Kind(name, label, description, params, fetch, lambda p: title.format_map(_Safe(p)), icon=icon))


class _Safe(dict):
    def __missing__(self, key):
        return ""


# news: Google News search as RSS (no key)
def _news(p: dict) -> list[dict]:
    q = p["topic"]
    if p.get("site"):
        q += f" site:{p['site']}"
    q += f" when:{int(p.get('days') or 7)}d"
    region = str(p.get("region") or "US").upper()
    lang = str(p.get("language") or "en").lower()
    url = ("https://news.google.com/rss/search?" +
           urllib.parse.urlencode({"q": q, "hl": f"{lang}-{region}", "gl": region, "ceid": f"{region}:{lang}"}))
    return parse_feed(http_get(url))


register(Kind(
    "news", "News", "News stories about a topic, a company, a person or a place (Google News; no account needed). "
                    "The topic is a search: quotes make a phrase (\"Nvidia\" \"earnings\"), OR widens it.",
    [Param("topic", "what to follow: a company, topic or search", True, example="Nvidia earnings"),
     Param("site", "only stories from this site", example="reuters.com"),
     Param("days", "how far back to look", default=7), Param("region", "country code", default="US"),
     Param("language", "language code", default="en")],
    _news, lambda p: f"News: {p['topic']}", icon="news"))


# web: a web search, for what the news doesn't cover (blogs, forums, product pages)
def _web(p: dict) -> list[dict]:
    from jevh.tools import web
    return [{"title": h["title"], "url": h["url"], "summary": h["snippet"], "source": urllib.parse.urlparse(h["url"]).netloc}
            for h in web.search(p["query"], int(p.get("n") or 10))]


register(Kind(
    "web", "Web search", "Pages from a web search (the engine chosen in Settings), for what news sites don't cover: blogs, forums, "
                         "product or status pages. No dates; a page is new when it is in the results on two looks in a row "
                         "(a search engine's bottom places shuffle between identical searches), so a new page is "
                         "noticed one look late.",
    [Param("query", "the search", True, example='"open source" LLM inference benchmark'),
     Param("n", "results to keep", default=10)],
    _web, lambda p: f"Web: {p['query']}", icon="globe", settle=2))


def _rss(p: dict) -> list[dict]:
    return parse_feed(http_get(p["url"]))


register(Kind(
    "rss", "RSS / Atom feed", "Any RSS or Atom feed: a blog, a newsroom page, a podcast, a status page, a changelog.",
    [Param("url", "the feed's address", True, example="https://example.com/feed.xml")],
    _rss, lambda p: f"Feed: {urllib.parse.urlparse(p['url']).netloc}{urllib.parse.urlparse(p['url']).path}"[:80],
    icon="rss"))

feed_kind("reddit", "Reddit", "New posts in a subreddit.",
          "https://www.reddit.com/r/{subreddit}/new/.rss",
          [Param("subreddit", "its name, without r/", True, example="LocalLLaMA")], "Reddit: r/{subreddit}", icon="rss")
feed_kind("hn", "Hacker News", "Hacker News stories that match a search (hnrss.org).",
          "https://hnrss.org/newest?q={query}",
          [Param("query", "words to match", True, example="llama.cpp")], "Hacker News: {query}", icon="rss")
feed_kind("github", "GitHub releases", "New releases of a GitHub project.",
          "https://github.com/{repo}/releases.atom",
          [Param("repo", "owner/name", True, example="ggml-org/llama.cpp")], "Releases: {repo}", icon="box",
          safe={"repo": "/"})
feed_kind("youtube", "YouTube channel", "New videos on a YouTube channel (by channel id, UC...).",
          "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}",
          [Param("channel_id", "the channel's id (starts with UC)", True, example="UCXXXXXXXXXXXXXXXXXXXXXX")],
          "YouTube: {channel_id}", icon="rss")
feed_kind("arxiv", "arXiv papers", "New arXiv papers that match a search.",
          "https://export.arxiv.org/api/query?search_query=all:{query}&sortBy=submittedDate&sortOrder=descending&max_results=25",
          [Param("query", "words to match", True, example="speculative decoding")], "arXiv: {query}", icon="file")


# x: posts from an account, or matching a search. X has no free feed: it needs one of three things set up.
X_API = "/2/"
_x_ids: dict[str, str] = {}


_x_cache: dict[str, Any] = {"ts": 0.0, "value": None}


def _x_setup() -> dict:
    """What is set up for reading X: {feed, bearer, nango}; each a value or ''. (Asking Nango costs a request:
    the answer is kept a minute.)"""
    if _x_cache["value"] is not None and time.time() - _x_cache["ts"] < 60:
        return _x_cache["value"]
    value = _x_look()
    _x_cache.update(ts=time.time(), value=value)
    return value


def _x_look() -> dict:
    from_cfg = os.environ.get("JEVH_NEWSROOM_X_FEED", "")
    nango = ""
    if not os.environ.get("X_BEARER_TOKEN"):
        try:
            from jevh.tools import nango as nango_tool
            if nango_tool.configured():
                want = os.environ.get("JEVH_NEWSROOM_X_INTEGRATION", "").strip().lower()
                for c in nango_tool.connections():
                    names = {str(c.get("integration", "")).lower(), str(c.get("provider", "")).lower()}
                    if (want and want in names) or (not want and names & {"x", "twitter", "twitter-v2", "x-twitter"}):
                        nango = str(c.get("integration"))
                        break
        except Exception:  # noqa: BLE001 - Nango not set up or not reachable: just not an option
            pass
    return {"feed": from_cfg, "bearer": os.environ.get("X_BEARER_TOKEN", ""), "nango": nango}


def _x_ready() -> str | None:
    s = _x_setup()
    if s["feed"] or s["bearer"] or s["nango"]:
        return None
    return ("X can't be read until one of these is set up: (1) connect your X account in Nango (nango-setup), "
            "(2) set the X_BEARER_TOKEN environment variable to an X API token that can read posts, or (3) point "
            "newsroom.x_feed in config at an RSS bridge you run (RSSHub, Nitter), e.g. https://rsshub.example/twitter/user/{user}")


def _x_api(path: str, params: dict, setup: dict) -> dict:
    if setup["bearer"]:
        url = "https://api.x.com" + path + "?" + urllib.parse.urlencode(params)
        raw = http_get(url, headers={"Authorization": f"Bearer {setup['bearer']}"})
        return json.loads(raw.decode("utf-8", "replace"))
    from jevh.tools import nango as nango_tool
    return nango_tool.get(setup["nango"], path, params)


def _x(p: dict) -> list[dict]:
    setup = _x_setup()
    user, query = p.get("user", ""), p.get("query", "")
    if setup["feed"]:
        tpl = setup["feed"]
        if ("{user}" in tpl and not user) or ("{query}" in tpl and not query):
            raise NewsroomError(f"the configured X feed ({tpl}) needs {'a user' if '{user}' in tpl else 'a query'}")
        q = urllib.parse.quote
        return parse_feed(http_get(tpl.replace("{user}", q(user, safe="")).replace("{query}", q(query, safe=""))))
    if not (setup["bearer"] or setup["nango"]):
        raise SourceUnavailable(_x_ready() or "X is not set up")
    fields = {"tweet.fields": "created_at,author_id,public_metrics", "max_results": 20}
    if user:
        if user.lower() not in _x_ids:
            who = _x_api(f"{X_API}users/by/username/{urllib.parse.quote(user)}", {}, setup)
            if not (who.get("data") or {}).get("id"):
                raise NewsroomError(f"X has no account @{user}")
            _x_ids[user.lower()] = who["data"]["id"]
        data = _x_api(f"{X_API}users/{_x_ids[user.lower()]}/tweets",
                      {**fields, "exclude": "retweets,replies" if not p.get("replies") else "retweets"}, setup)
    else:
        data = _x_api(f"{X_API}tweets/search/recent", {**fields, "query": query + " -is:retweet", "max_results": 20}, setup)
    out = []
    for t in data.get("data") or []:
        handle = user or "i"
        out.append({"title": t.get("text", ""), "url": f"https://x.com/{handle}/status/{t['id']}", "uid": str(t["id"]),
                    "source": f"@{user}" if user else "X", "published": _when(t.get("created_at")),
                    "author": f"@{user}" if user else ""})
    return out


def _x_params(p: dict) -> dict:
    if bool(p.get("user")) == bool(p.get("query")):
        raise NewsroomError("an X widget follows either an account (user=) or a search (query=), not both or neither")
    if p.get("user"):
        p["user"] = str(p["user"]).strip().lstrip("@")
        if not re.fullmatch(r"\w{1,15}", p["user"]):
            raise NewsroomError(f"{p['user']!r} is not an X handle (letters, digits, underscore; up to 15)")
    return p


register(Kind(
    "x", "X posts", "Posts on X (Twitter): from one account (user), or every recent post that matches a search (query). "
                    "Needs X set up first (see `needs`).",
    [Param("user", "the account's handle", example="OpenAI", main=True),
     Param("query", "or a search instead of an account", example='"GPT-6" OR "o5"', main=True),
     Param("replies", "include the account's replies", default=False)],
    _x, lambda p: f"X: @{p['user']}" if p.get("user") else f"X: {p['query']}", ready=_x_ready, icon="x"))

KIND_CHECKS: dict[str, Callable[[dict], dict]] = {"x": _x_params}


def kinds() -> list[dict]:
    return [k.public() for k in KINDS.values()]


def clean_params(kind: str, raw: dict) -> dict:
    """The settings for a widget of this kind, checked: unknown names and missing required ones are errors that say
    what the valid ones are (a worker reads them and fixes the call)."""
    if kind not in KINDS:
        raise NewsroomError(f"unknown kind {kind!r}; kinds: {', '.join(KINDS)}")
    k = KINDS[kind]
    names = {p.name for p in k.params}
    unknown = set(raw) - names
    if unknown:
        raise NewsroomError(f"{kind} takes {', '.join(sorted(names))}; not {', '.join(sorted(unknown))}")
    out = {}
    for p in k.params:
        v = raw.get(p.name)
        if isinstance(v, str):
            v = v.strip()
        if v in (None, ""):
            if p.required:
                raise NewsroomError(f"{kind} needs {p.name} ({p.help}), e.g. {p.example!r}")
            continue
        out[p.name] = v if not isinstance(v, str) else v[:300]
    if kind == "rss":
        check_url(out["url"])
    if kind in KIND_CHECKS:
        out = KIND_CHECKS[kind](out)
    return out


# ---------------------------------------------------------------- alerts

def clean_alert(value: Any) -> dict:
    """What counts as worth a notification, in any of the forms a worker or a form gives it:
    None/False/"" (never), True (every new item), "a, b" or ["a", "b"] (a new item that mentions any),
    or {"keywords": [...], "exclude": [...], "all": bool, "about": "..."}. `about` is a sentence on what the user
    wants to hear about ("results of their quarterly earnings"): Jev judges each new item against it, which
    catches what no keyword names ("Nvidia reports record quarter")."""
    about = ""
    if isinstance(value, dict):
        kw, ex, every, about = value.get("keywords"), value.get("exclude"), bool(value.get("all")), str(value.get("about") or "")
    elif value is True:
        kw, ex, every = None, None, True
    elif isinstance(value, (list, tuple, str)):
        kw, ex, every = value, None, False
    else:
        kw, ex, every = None, None, False

    def words(v) -> list[str]:
        if isinstance(v, str):
            v = re.split(r"[,\n;]", v)
        return [w.strip() for w in (v or []) if str(w).strip()][:20]
    return {"all": every, "keywords": [str(w)[:80] for w in words(kw)], "exclude": [str(w)[:80] for w in words(ex)],
            "about": about.strip()[:200]}


def _pattern(word: str) -> re.Pattern:
    # a whole word at its start; a word of 4+ letters may go on (launch -> launches), so "AI" never matches "said"
    return re.compile(r"(?<!\w)" + re.escape(word) + (r"\w*" if len(word) >= 4 else r"(?!\w)"), re.I)


def alert_matches(alert: dict, item: dict, judge: Callable[[str, str], float] | None = None,
                  threshold: float = 0.7) -> str | None:
    """Why this new item notifies (every new item, the keywords it mentions, or what it is about), or None.
    `judge(text, about) -> P(the item is about that)` is Jev's; without one an `about` alert says nothing."""
    text = " ".join(str(item.get(k) or "") for k in ("title", "summary", "author"))
    if any(_pattern(w).search(text) for w in alert.get("exclude") or []):
        return None
    if alert.get("all"):
        return "new item"
    hit = [w for w in alert.get("keywords") or [] if _pattern(w).search(text)]
    if hit:
        return "mentions " + ", ".join(hit)
    if alert.get("about") and judge:
        try:
            if judge(text, alert["about"]) >= threshold:
                return f"about {alert['about'][:60]}"
        except Exception:  # noqa: BLE001 - Jev down: the keywords still work, this item is not judged
            pass
    return None


def ago(ts: float | None, now: float | None = None) -> str:
    if not ts:
        return ""
    d = max(0, (now or time.time()) - ts)
    for n, unit in ((86400 * 7, "w"), (86400, "d"), (3600, "h"), (60, "m")):
        if d >= n:
            return f"{int(d // n)}{unit} ago"
    return "just now"


# ---------------------------------------------------------------- the store

class Newsroom:
    """The widgets, what they found and the notifications, in SQLite (safe across threads and across processes)."""

    def __init__(self, path: Path, cfg: dict | None = None, judge: Callable[[str, str], float] | None = None,
                 tool_env: Callable[[], dict[str, str]] | None = None):
        self.judge = judge   # (item text, what the user wants to hear about) -> P(it is): Jev's, given by the dashboard
        # () -> the harness's tool settings (Harness.tool_env): a look made here, in the dashboard's own process, searches
        # the web with the engine chosen in Settings, as a worker's REPL does through its environment
        self.tool_env = tool_env
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS widgets(id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, title TEXT, params TEXT,
                every INTEGER, alert TEXT, size TEXT DEFAULT 'm', pos REAL, paused INTEGER DEFAULT 0, created REAL,
                last_look REAL, last_ok REAL, error TEXT DEFAULT '', fails INTEGER DEFAULT 0, session TEXT DEFAULT '');
            CREATE TABLE IF NOT EXISTS items(id INTEGER PRIMARY KEY AUTOINCREMENT, widget_id INTEGER, uid TEXT, title TEXT,
                url TEXT, source TEXT, published REAL, summary TEXT, author TEXT, first_seen REAL, read INTEGER DEFAULT 0,
                UNIQUE(widget_id, uid));
            CREATE INDEX IF NOT EXISTS items_widget ON items(widget_id, read);
            CREATE TABLE IF NOT EXISTS pending(widget_id INTEGER, uid TEXT, title TEXT, url TEXT, source TEXT, published REAL,
                summary TEXT, author TEXT, looks INTEGER DEFAULT 1, PRIMARY KEY(widget_id, uid));
            CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, widget_id INTEGER, item_id INTEGER,
                title TEXT, body TEXT, url TEXT, why TEXT DEFAULT '', read INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value INTEGER);
            INSERT OR IGNORE INTO meta VALUES('rev', 0);
        """)
        # (added 2026-10-24: a board made before has no summary columns)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(widgets)")}
        for col, decl in (("summary", "TEXT DEFAULT ''"), ("summary_wanted", "INTEGER DEFAULT 0")):
            if col not in cols:
                self.db.execute(f"ALTER TABLE widgets ADD COLUMN {col} {decl}")
        self.db.commit()

    # -- plumbing ----------------------------------------------------------------------------
    def _run(self, sql: str, args=(), commit: bool = True, touch: bool = True) -> sqlite3.Cursor:
        with self._lock:
            cur = self.db.execute(sql, args)
            if commit:
                if touch:
                    self.db.execute("UPDATE meta SET value=value+1 WHERE key='rev'")
                self.db.commit()
            return cur

    def _all(self, sql: str, args=()) -> list[sqlite3.Row]:
        with self._lock:
            return self.db.execute(sql, args).fetchall()

    def rev(self) -> int:
        """Goes up on every change, by any process: a page that holds the last one knows nothing changed."""
        return self._all("SELECT value FROM meta WHERE key='rev'")[0][0]

    @staticmethod
    def _widget(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["params"] = json.loads(d["params"] or "{}")
        d["alert"] = json.loads(d["alert"] or "{}")
        d["paused"] = bool(d["paused"])
        try:
            d["summary"] = json.loads(d.get("summary") or "null")
        except ValueError:
            d["summary"] = None
        d["summary_wanted"] = bool(d.get("summary_wanted"))
        return d

    # -- widgets -----------------------------------------------------------------------------
    def add_widget(self, kind: str, params: dict | None = None, *, title: str | None = None, every: int | None = None,
                   alert: Any = None, size: str = "m", session: str = "") -> dict:
        """Add a widget. One with the same kind and settings already on the board is returned instead (`existing`)."""
        params = clean_params(kind, dict(params or {}))
        minutes = self._every(every)
        for w in self.widgets():
            if w["kind"] == kind and {k: str(v).lower() for k, v in w["params"].items()} == \
                    {k: str(v).lower() for k, v in params.items()}:
                return {**w, "existing": True}
        if len(self._all("SELECT id FROM widgets")) >= int(self.cfg["max_widgets"]):
            raise NewsroomError(f"the board holds {self.cfg['max_widgets']} widgets; remove one first")
        if size not in SIZES:
            raise NewsroomError(f"size must be one of {', '.join(SIZES)}")
        now = time.time()
        pos = (self._all("SELECT MAX(pos) FROM widgets")[0][0] or 0) + 1
        wid = self._run("INSERT INTO widgets(kind, title, params, every, alert, size, pos, created, session) VALUES (?,?,?,?,?,?,?,?,?)",
                        (kind, (title or KINDS[kind].title(params))[:120], json.dumps(params), minutes,
                         json.dumps(clean_alert(alert)), size, pos, now, session)).lastrowid
        return self.widget(wid)

    def _every(self, every: Any) -> int:
        try:
            n = int(every) if every not in (None, "") else int(self.cfg["default_minutes"])
        except (TypeError, ValueError):
            raise NewsroomError(f"every is a number of minutes, not {every!r}") from None
        return max(int(self.cfg["min_minutes"]), min(n, 60 * 24 * 7))

    def widget(self, wid: int) -> dict | None:
        rows = self._all("SELECT * FROM widgets WHERE id=?", (wid,))
        return self._widget(rows[0]) if rows else None

    def widgets(self) -> list[dict]:
        """In board order, each with its unread count and item count."""
        out = []
        for r in self._all("SELECT * FROM widgets ORDER BY pos, id"):
            w = self._widget(r)
            n, unread = self._all("SELECT COUNT(*), COALESCE(SUM(1-read),0) FROM items WHERE widget_id=?", (w["id"],))[0]
            w["items_total"], w["unread"] = n, int(unread)
            out.append(w)
        return out

    def update_widget(self, wid: int, *, title: str | None = None, every: Any = None, alert: Any = "keep",
                      paused: bool | None = None, size: str | None = None, params: dict | None = None) -> dict:
        w = self.widget(wid)
        if not w:
            raise NewsroomError(f"no widget {wid}")
        sets: dict[str, Any] = {}
        if title is not None:
            sets["title"] = str(title).strip()[:120] or w["title"]
        if every is not None:
            sets["every"] = self._every(every)
        if not (isinstance(alert, str) and alert == "keep"):
            sets["alert"] = json.dumps(clean_alert(alert))
        if paused is not None:
            sets["paused"] = int(bool(paused))
        if size is not None:
            if size not in SIZES:
                raise NewsroomError(f"size must be one of {', '.join(SIZES)}")
            sets["size"] = size
        if params:   # follow something else: the old items belong to the old settings
            new = clean_params(w["kind"], {**w["params"], **params})
            if new != w["params"]:
                sets["params"] = json.dumps(new)
                sets.update(last_look=None, last_ok=None, error="", fails=0, summary="", summary_wanted=0)
                if title is None and w["title"] == KINDS[w["kind"]].title(w["params"]):
                    sets["title"] = KINDS[w["kind"]].title(new)[:120]
                self._run("DELETE FROM items WHERE widget_id=?", (wid,), commit=False)
                self._run("DELETE FROM pending WHERE widget_id=?", (wid,), commit=False)
        if sets:
            self._run(f"UPDATE widgets SET {', '.join(k + '=?' for k in sets)} WHERE id=?", [*sets.values(), wid])
        return self.widget(wid)

    def delete_widget(self, wid: int) -> bool:
        with self._lock:
            self.db.execute("DELETE FROM items WHERE widget_id=?", (wid,))
            self.db.execute("DELETE FROM pending WHERE widget_id=?", (wid,))
            self.db.execute("DELETE FROM notes WHERE widget_id=?", (wid,))
            n = self.db.execute("DELETE FROM widgets WHERE id=?", (wid,)).rowcount
            self.db.execute("UPDATE meta SET value=value+1 WHERE key='rev'")
            self.db.commit()
        return n > 0

    def move(self, wid: int, step: int) -> None:
        """Move a widget one place earlier (-1) or later (+1) on the board."""
        ids = [r["id"] for r in self._all("SELECT id FROM widgets ORDER BY pos, id")]
        if wid not in ids:
            raise NewsroomError(f"no widget {wid}")
        i, j = ids.index(wid), max(0, min(len(ids) - 1, ids.index(wid) + (1 if step > 0 else -1)))
        ids.insert(j, ids.pop(i))
        with self._lock:
            for n, w in enumerate(ids):
                self.db.execute("UPDATE widgets SET pos=? WHERE id=?", (n + 1, w))
            self.db.execute("UPDATE meta SET value=value+1 WHERE key='rev'")
            self.db.commit()

    # -- items -------------------------------------------------------------------------------
    def items(self, wid: int, limit: int = 30, unread: bool = False) -> list[dict]:
        rows = self._all(f"SELECT * FROM items WHERE widget_id=? {'AND read=0' if unread else ''} "
                         "ORDER BY COALESCE(published, first_seen) DESC, id DESC LIMIT ?", (wid, limit))
        return [{**dict(r), "read": bool(r["read"])} for r in rows]

    def mark_read(self, *, item: int | None = None, widget: int | None = None) -> None:
        if item is not None:
            self._run("UPDATE items SET read=1 WHERE id=?", (item,))
        elif widget is not None:
            self._run("UPDATE items SET read=1 WHERE widget_id=?", (widget,))
        else:
            self._run("UPDATE items SET read=1")

    # -- notifications -----------------------------------------------------------------------
    def notify(self, title: str, body: str = "", url: str = "", widget_id: int | None = None, item_id: int | None = None,
               why: str = "") -> dict:
        """A notification for the user (the bell on the Newsroom page, a toast, and a desktop notification if allowed)."""
        title = str(title).strip()[:200]
        if not title:
            raise NewsroomError("a notification needs a title")
        url = str(url or "")
        if url and not url.lower().startswith(("http://", "https://")):
            url = ""
        nid = self._run("INSERT INTO notes(ts, widget_id, item_id, title, body, url, why) VALUES (?,?,?,?,?,?,?)",
                        (time.time(), widget_id, item_id, title, str(body)[:400], url[:1000], why[:200])).lastrowid
        return self._note(self._all("SELECT * FROM notes WHERE id=?", (nid,))[0])

    @staticmethod
    def _note(r: sqlite3.Row) -> dict:
        return {**dict(r), "read": bool(r["read"])}

    def notes(self, unread: bool = False, limit: int = 50) -> list[dict]:
        return [self._note(r) for r in self._all(
            f"SELECT * FROM notes {'WHERE read=0' if unread else ''} ORDER BY id DESC LIMIT ?", (limit,))]

    def unread_notes(self) -> int:
        return self._all("SELECT COUNT(*) FROM notes WHERE read=0")[0][0]

    def read_notes(self, nid: int | None = None) -> None:
        self._run("UPDATE notes SET read=1" + (" WHERE id=?" if nid is not None else ""), (nid,) if nid is not None else ())

    def clear_notes(self) -> None:
        self._run("DELETE FROM notes")

    # -- looking -----------------------------------------------------------------------------
    def due(self, now: float | None = None) -> list[int]:
        """Widgets whose time has come. A widget that keeps failing is tried less and less often (up to 6 hours)."""
        now = now or time.time()
        out = []
        for r in self._all("SELECT id, every, last_look, fails, paused FROM widgets ORDER BY pos, id"):
            if r["paused"]:
                continue
            gap = min(r["every"] * 60 * (2 ** min(r["fails"], 5)), 6 * 3600) if r["fails"] else r["every"] * 60
            if r["last_look"] is None or now - r["last_look"] >= gap:
                out.append(r["id"])
        return out

    def refresh(self, wid: int, now: float | None = None) -> dict:
        """Look at one widget's source now. -> {widget, new, alerts: [notification], error}.
        The first look only fills the widget (what is already there is not news, so it can't notify)."""
        w = self.widget(wid)
        if not w:
            raise NewsroomError(f"no widget {wid}")
        now = now or time.time()
        try:
            with self._looking():
                raw = KINDS[w["kind"]].fetch(w["params"])
            found = [_clean_item(i) for i in raw][:50]
        except Exception as e:  # noqa: BLE001 - one source failing is a state of the widget, not a crash
            msg = str(e) if isinstance(e, NewsroomError) else f"{type(e).__name__}: {e}"
            self._run("UPDATE widgets SET last_look=?, error=?, fails=fails+1 WHERE id=?", (now, msg[:400], wid))
            return {"widget": wid, "new": 0, "alerts": [], "error": msg[:400]}
        first = w["last_ok"] is None
        name = getattr(raw, "title", "")
        if first and w["kind"] == "rss" and name and w["title"] == KINDS["rss"].title(w["params"]):
            self._run("UPDATE widgets SET title=? WHERE id=?", (f"Feed: {name}"[:120], wid), touch=False)   # (a feed has a name of its own)
        fresh: list[dict] = []
        with self._lock:
            settle = KINDS[w["kind"]].settle
            if settle > 1 and not first:
                found = self._settled(wid, found, settle)
            for it in reversed(found):   # (oldest first, so the feed's order is the id order when times tie)
                cur = self.db.execute(
                    "INSERT OR IGNORE INTO items(widget_id, uid, title, url, source, published, summary, author, first_seen, read) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (wid, it["uid"], it["title"], it["url"], it["source"], it["published"], it["summary"], it["author"],
                     now, int(first)))
                if cur.rowcount:
                    fresh.append({**it, "id": cur.lastrowid})
            keep = int(self.cfg["keep_items"])
            self.db.execute("DELETE FROM items WHERE widget_id=? AND id NOT IN (SELECT id FROM items WHERE widget_id=? "
                            "ORDER BY COALESCE(published, first_seen) DESC, id DESC LIMIT ?)", (wid, wid, keep))
            self.db.execute("UPDATE widgets SET last_look=?, last_ok=?, error='', fails=0 WHERE id=?", (now, now, wid))
            # the summary at the top of the tile is written again when the look found something (or there is none yet);
            # the dashboard's Summarizer writes it, between turns (a worker's first look, in its REPL, only asks for it)
            if self.cfg["summary"] and (fresh or (found and not w.get("summary"))):
                self.db.execute("UPDATE widgets SET summary_wanted=1 WHERE id=?", (wid,))
            self.db.execute("UPDATE meta SET value=value+1 WHERE key='rev'")
            self.db.commit()
        fresh.reverse()
        alerts: list[dict] = []
        if not first:
            hits = [(it, why) for it in fresh[:40] if (why := alert_matches(w["alert"], it, self.judge, float(self.cfg["about_threshold"])))]
            for it, why in hits[:MAX_NOTES_PER_LOOK]:
                alerts.append(self.notify(w["title"], it["title"], it["url"], wid, it["id"], why))
            if len(hits) > MAX_NOTES_PER_LOOK:
                alerts.append(self.notify(w["title"], f"and {len(hits) - MAX_NOTES_PER_LOOK} more new items", "", wid,
                                          None, "more"))
        return {"widget": wid, "new": len(fresh) if not first else 0, "first": first, "alerts": alerts, "error": ""}

    def _looking(self):
        """The settings a look is made under: the harness's web search choice, when the dashboard gave a way to ask."""
        from contextlib import nullcontext
        from jevh.tools import web
        env = self.tool_env() if self.tool_env else None
        return web.using({k: v for k, v in env.items() if k.startswith(("JEVH_SEARCH", "JEVH_SEARXNG", "JEVH_BRAVE"))})             if env else nullcontext()

    def _settled(self, wid: int, found: list[dict], settle: int) -> list[dict]:
        """`found` without the items this widget has never held that have not yet been in the results `settle` looks in a
        row; those wait in `pending` (an item in the results on one look only is dropped from it). An empty answer says
        nothing - a search engine that is rate limiting sends one - so it leaves `pending` as it was. Call with the lock."""
        if not found:
            return found
        held = {r["uid"] for r in self.db.execute("SELECT uid FROM items WHERE widget_id=?", (wid,))}
        looks = {r["uid"]: r["looks"] for r in self.db.execute("SELECT uid, looks FROM pending WHERE widget_id=?", (wid,))}
        keep, wait = [], []
        for it in found:
            n = looks.get(it["uid"], 0) + 1
            (keep if it["uid"] in held or n >= settle else wait).append((it, n))
        self.db.execute("DELETE FROM pending WHERE widget_id=?", (wid,))   # (what the results no longer show is dropped)
        for it, n in wait:
            self.db.execute("INSERT OR REPLACE INTO pending(widget_id, uid, title, url, source, published, summary, author, looks) "
                            "VALUES (?,?,?,?,?,?,?,?,?)", (wid, it["uid"], it["title"], it["url"], it["source"], it["published"],
                                                           it["summary"], it["author"], n))
        return [it for it, _ in keep]

    # -- summaries ---------------------------------------------------------------------------
    def summaries_wanted(self) -> list[int]:
        """Tiles whose summary is to be written (again), in board order."""
        return [r["id"] for r in self._all("SELECT id FROM widgets WHERE summary_wanted=1 AND paused=0 ORDER BY pos, id")]

    def write_summary(self, wid: int, write: Callable[[dict, list[dict]], dict]) -> dict | None:
        """Write one tile's summary with `write(widget, items) -> {text, sources}` (summarize_with, given a model) from its
        newest `summary_items` items, and keep it. None: nothing to summarize (the request is dropped)."""
        w = self.widget(wid)
        items = self.items(wid, int(self.cfg["summary_items"])) if w else []
        if not w or not items:
            self._run("UPDATE widgets SET summary_wanted=0 WHERE id=?", (wid,), touch=False)
            return None
        out = {**write(w, items), "ts": time.time(), "items": len(items)}
        # (a look that came in meanwhile asked again: the next pass writes it from the newer items)
        self._run("UPDATE widgets SET summary=?, summary_wanted=CASE WHEN last_ok>? THEN 1 ELSE 0 END WHERE id=?",
                  (json.dumps(out), w["last_ok"] or 0, wid))
        return out

    def refresh_due(self, now: float | None = None, workers: int = 3) -> list[dict]:
        ids = self.due(now)
        if not ids:
            return []
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(ids)))) as ex:
            return list(ex.map(lambda i: self.refresh(i, now), ids))

    def close(self) -> None:
        with self._lock:
            self.db.close()


class Runner:
    """Looks at the widgets whose time has come, every `tick` seconds, on a thread of its own.
    `on_alerts(notes)` is called when a look makes notifications, `on_change()` when anything was fetched."""

    def __init__(self, room: Newsroom, on_alerts: Callable[[list[dict]], None] | None = None,
                 on_change: Callable[[], None] | None = None, tick: float | None = None):
        self.room, self.on_alerts, self.on_change = room, on_alerts, on_change
        self.tick = float(tick if tick is not None else room.cfg["tick"])
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def once(self, now: float | None = None) -> list[dict]:
        results = self.room.refresh_due(now)
        notes = [n for r in results for n in r["alerts"]]
        if notes and self.on_alerts:
            self.on_alerts(notes)
        if results and self.on_change:
            self.on_change()
        return results

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    self.once()
                except Exception:  # noqa: BLE001 - never let the loop die
                    pass
                self._stop.wait(self.tick)
        self._thread = threading.Thread(target=loop, name="newsroom", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------- the summary at the top of a tile
SUMMARY_PROMPT = """You write the short summary at the top of a news tile on the user's dashboard.
The tile follows: {follows}
Below are the newest items its last look found, numbered.

Say in 2 to 4 short sentences what they report: the main developments, plainly, so the user knows what happened without
opening every link. Use only what the titles and snippets say: no facts from elsewhere, no guesses, no advice. After each
sentence put the numbers of the items it comes from in brackets, like [1] or [2][5]. If the items have little in common,
cover the two or three most notable. Reply with the summary only.

{items}"""
CITE = re.compile(r"\[(\d{1,2})\]")


def summary_prompt(w: dict, items: list[dict]) -> str:
    follows = f"{w['title']} ({KINDS[w['kind']].label if w['kind'] in KINDS else w['kind']}: " + \
        ", ".join(f"{k}={v}" for k, v in (w.get("params") or {}).items()) + ")"
    lines = []
    for n, it in enumerate(items, 1):
        when = ago(it.get("published") or it.get("first_seen"))
        head = " · ".join(x for x in (it.get("source") or "", when) if x)
        snippet = (it.get("summary") or "").strip()
        lines.append(f"[{n}] {it['title']}" + (f" ({head})" if head else "") + (f"\n    {snippet[:400]}" if snippet else ""))
    return SUMMARY_PROMPT.format(follows=follows, items="\n".join(lines))


def parse_summary(text: str, items: list[dict]) -> dict:
    """{text, sources}: the summary without a model's leftovers, and the items it cites, numbered as in the text (an
    answer that cites nothing lists every item it was given). A number that names no item is dropped from the text."""
    text = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.S).strip()
    text = re.sub(r"^(?:summary|here(?:'s| is) the summary)\s*[:\-]\s*", "", text, flags=re.I).strip().strip('"')
    if not text:
        raise NewsroomError("the model wrote no summary")
    text = CITE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(items) else "", text)
    cited = sorted({int(n) for n in CITE.findall(text)})
    keep = cited or list(range(1, len(items) + 1))
    return {"text": text[:1500],
            "sources": [{"n": n, "title": items[n - 1]["title"], "url": items[n - 1].get("url") or "",
                         "source": items[n - 1].get("source") or ""} for n in keep]}


def summarize_with(llm: Any, w: dict, items: list[dict], max_tokens: int = 700) -> dict:
    """A tile's summary from `llm` (any model with .chat(messages)): one short call, thinking off on a local server."""
    import inspect
    kwargs: dict[str, Any] = {}
    try:
        params = inspect.signature(llm.chat).parameters
    except (TypeError, ValueError):
        params = {}
    if "body_extra" in params or any(p.kind == p.VAR_KEYWORD for p in params.values()):
        from .light import _no_thinking
        kwargs["body_extra"] = {"max_tokens": int(max_tokens), **_no_thinking(llm)}
    out = llm.chat([{"role": "user", "content": summary_prompt(w, items)}], **kwargs)
    return parse_summary(getattr(out, "text", "") or "", items)


class Summarizer:
    """Writes the tiles' summaries, one at a time, on a thread of its own - only while `ready()` (the dashboard: no turn of
    the user's is running). A local model serves one request at a time: live 2026-10-21, a check that ran after a turn
    held it, and the user's next message waited 6 minutes for its first token. `write(widget, items) -> {text, sources}`."""

    def __init__(self, room: Newsroom, write: Callable[[dict, list[dict]], dict], ready: Callable[[], bool] | None = None,
                 on_change: Callable[[], None] | None = None, tick: float = 5.0):
        self.room, self.write, self.ready, self.on_change, self.tick = room, write, ready or (lambda: True), on_change, tick
        self._stop, self._wake = threading.Event(), threading.Event()
        self._thread: threading.Thread | None = None
        self.errors: dict[int, str] = {}   # tile -> why its last summary failed (it is tried again after its next look)

    def once(self) -> int:
        """Write the summaries that are wanted while ready() holds; -> how many were written."""
        done = 0
        for wid in self.room.summaries_wanted():
            if self._stop.is_set() or not self.ready():
                break
            try:
                if self.room.write_summary(wid, self.write) is not None:
                    done += 1
                self.errors.pop(wid, None)
            except Exception as e:  # noqa: BLE001 - the model down or failing: no summary this time, the items still show
                self.errors[wid] = f"{type(e).__name__}: {e}"[:300]
                self.room._run("UPDATE widgets SET summary_wanted=0 WHERE id=?", (wid,), touch=False)
            if self.on_change:
                self.on_change()
        return done

    def wake(self) -> None:
        self._wake.set()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    self.once()
                except Exception:  # noqa: BLE001 - never let the loop die
                    pass
                self._wake.wait(self.tick)
                self._wake.clear()
        self._thread = threading.Thread(target=loop, name="newsroom-summaries", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()


def default_path() -> Path:
    """The store for this process: JEVH_NEWSROOM_DB (set by the harness for worker REPLs), else the workspace's."""
    if os.environ.get("JEVH_NEWSROOM_DB"):
        return Path(os.environ["JEVH_NEWSROOM_DB"])
    from .config import load_config
    return load_config().workspace / "newsroom.db"


def worker_env(cfg) -> dict[str, str]:
    """The store's place and the X settings, as environment variables for a worker's REPL (the harness adds them to
    each worker's own, not to the process's: Harness.tool_env). An X setting already in the environment wins."""
    s = settings(cfg.raw)
    env = {"JEVH_NEWSROOM_DB": str(cfg.workspace / "newsroom.db")}
    for name, value in (("JEVH_NEWSROOM_X_FEED", s["x_feed"]), ("JEVH_NEWSROOM_X_INTEGRATION", s["x_integration"])):
        if value and name not in os.environ:
            env[name] = str(value)
    if s["allow_local"]:
        env["JEVH_NEWSROOM_ALLOW_LOCAL"] = "1"
    return env


def export_env(cfg) -> None:
    """The same settings for a process that uses the store itself (the dashboard's runner, `jevh newsroom`)."""
    os.environ.update(worker_env(cfg))


# ---------------------------------------------------------------- the `jevh newsroom` command

def describe(w: dict) -> str:
    state = "paused" if w["paused"] else (f"failing: {w['error']}" if w["error"] else f"looked {ago(w['last_ok'])}" if w["last_ok"] else "not looked at yet")
    alert = w["alert"]
    watch = "every new item" if alert.get("all") else ", ".join((alert.get("keywords") or []) + ([f"about: {alert['about']}"] if alert.get("about") else [])) or "no alerts"
    return f"#{w['id']:<3} {w['kind']:<8} {w['title'][:48]:<48} {w['unread']:>3} new  every {w['every']}m  alerts: {watch}  [{state}]"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="jevh newsroom", description="The Newsroom board: widgets that follow news and posts.")
    ap.add_argument("action", nargs="?", default="list", choices=["list", "kinds", "add", "refresh", "notes", "rm", "items"])
    ap.add_argument("rest", nargs="*", help="add: KIND key=value ...; refresh/rm/items: the widget id")
    ap.add_argument("--alert", help='keywords that notify, comma separated; "all" for every new item')
    ap.add_argument("--every", type=int, help="minutes between looks")
    ap.add_argument("--title")
    a = ap.parse_args(argv)
    from .config import load_config
    cfg = load_config()
    room = Newsroom(cfg.workspace / "newsroom.db", settings(cfg.raw))
    export_env(cfg)
    try:
        if a.action == "kinds":
            for k in kinds():
                print(f"{k['name']:<8} {k['label']}: {k['description']}")
                for p in k["params"]:
                    print(f"           {p['name']}{'*' if p['required'] else ''}: {p['help']}"
                          + (f" (default {p['default']})" if p["default"] is not None else ""))
                if k["needs"]:
                    print(f"           NOT SET UP: {k['needs']}")
        elif a.action == "add":
            if not a.rest:
                ap.error("add needs a kind (see `jevh newsroom kinds`)")
            params = dict(kv.split("=", 1) for kv in a.rest[1:] if "=" in kv)
            alert = True if a.alert == "all" else a.alert
            w = room.add_widget(a.rest[0], params, title=a.title, every=a.every, alert=alert)
            if not w.get("existing"):
                r = room.refresh(w["id"])
                print(f"added #{w['id']} {w['title']}" + (f" (it failed: {r['error']})" if r["error"] else f": {len(room.items(w['id']))} items"))
            else:
                print(f"already on the board as #{w['id']}")
        elif a.action == "refresh":
            ids = [int(a.rest[0])] if a.rest else [w["id"] for w in room.widgets() if not w["paused"]]
            for i in ids:
                r = room.refresh(i)
                print(f"#{i}: " + (f"failed: {r['error']}" if r["error"] else f"{r['new']} new, {len(r['alerts'])} notifications"))
        elif a.action == "notes":
            for n in room.notes(limit=30):
                print(f"{'*' if not n['read'] else ' '} {time.strftime('%m-%d %H:%M', time.localtime(n['ts']))}  {n['title']}: {n['body']}  {n['url']}")
        elif a.action == "rm":
            print("removed" if a.rest and room.delete_widget(int(a.rest[0])) else "no such widget")
        elif a.action == "items":
            w = room.widget(int(a.rest[0]))
            if w and (w.get("summary") or {}).get("text"):   # the summary of the last look, then its sources
                s = w["summary"]
                print(f"Summary ({ago(s['ts'])}): {s['text']}\nSources:")
                print("\n".join(f"  [{x['n']}] {x['title'][:80]}  {x['url']}" for x in s.get("sources") or []) + "\n")
            for it in room.items(int(a.rest[0]), 20):
                print(f"{ago(it['published'] or it['first_seen']):>9}  {it['source'][:22]:<22} {it['title'][:90]}")
        else:
            ws = room.widgets()
            print("\n".join(describe(w) for w in ws) if ws else
                  "The board is empty. Ask in the chat (\"keep an eye on ...\") or: jevh newsroom add news topic=\"...\"")
            if room.unread_notes():
                print(f"\n{room.unread_notes()} unread notifications (jevh newsroom notes)")
    except NewsroomError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
