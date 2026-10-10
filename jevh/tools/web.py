"""Web search and reading, pre-imported as `web`. Jev keeps pages out of your context:
you get the passages that answer your question, not the whole page.

    web.search("espresso machine inventor")          # [{title, url, snippet}, ...] (the engine set in Settings; DuckDuckGo by default)
    web.read(url, "who patented it and when?")       # only the relevant passages of that page
    web.read(url)                                    # the page as plain text (first 4000 chars)
    web.ask("who patented the first espresso machine?")
        # search -> Jev ranks the results -> reads the best pages in parallel -> relevant
        # passages with their sources: {"passages": [{title, url, text}], "results": [...]}

For pages that need clicking, forms or JavaScript, use the `browser` tool instead.
"""
from __future__ import annotations

import contextvars
import html
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from html.parser import HTMLParser

from jevh.tools._pick import choose, client_from_env, pick_passages

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"
MAX_BYTES = 3_000_000
_jev = None


def _client():
    global _jev
    if _jev is None:
        _jev = client_from_env()
    return _jev


def _strip(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


class SearchError(RuntimeError):
    """A web search that could not be made, with what to do about it in its message (web.ask passes it on in `note`)."""


class SearchBlocked(SearchError):
    """The search engine would not answer: it asked for proof of a human (a CAPTCHA). Nothing here gets past that; it
    clears with time, or from another network. Before this was told apart, the harness took the challenge page for a
    search with no results - every web search of every worker came back empty, silently, for hours (2026-10-09)."""


CHALLENGE = re.compile(r"bots use DuckDuckGo|anomaly-modal|Select all squares", re.I)

# Which engine `search` asks is a setting (the dashboard's Settings tab; jevh/settings.py). It reaches a worker's REPL as
# environment variables and the dashboard's own Newsroom as `using(...)`; DuckDuckGo when nothing is set.
PROVIDERS = {
    "duckduckgo": {"label": "DuckDuckGo", "needs": "",
                   "help": "No account. It reads DuckDuckGo's web page, which can ask for a human check (a CAPTCHA)."},
    "searxng": {"label": "SearXNG", "needs": "url",
                "help": "Your own SearXNG server (or one you trust). Its settings.yml must allow the json format."},
    "brave": {"label": "Brave Search", "needs": "key",
              "help": "Brave's search API. Needs an API key (api.search.brave.com; there is a free plan)."},
}
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
_SETTINGS: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar("web_search_settings", default={})


def _setting(name: str, default: str = "") -> str:
    return _SETTINGS.get().get(name) or os.environ.get(name, default)


@contextmanager
def using(settings: dict[str, str]):
    """Searches made inside use these settings (environment-variable names: JEVH_SEARCH_PROVIDER, JEVH_SEARXNG_URL,
    JEVH_BRAVE_KEY) over the process environment."""
    token = _SETTINGS.set({**_SETTINGS.get(), **settings})
    try:
        yield
    finally:
        _SETTINGS.reset(token)


def search_env(cfg: dict | None) -> dict[str, str]:
    """The `search:` block of the config as those environment variables (always names the provider)."""
    c = cfg or {}
    env = {"JEVH_SEARCH_PROVIDER": str(c.get("provider") or "duckduckgo")}
    if c.get("searxng_url"):
        env["JEVH_SEARXNG_URL"] = str(c["searxng_url"])
    if c.get("brave_api_key"):
        env["JEVH_BRAVE_KEY"] = str(c["brave_api_key"])
    return env


def search_with(settings: dict[str, str], query: str, n: int = 8) -> list[dict]:
    """`search` under these settings (the Settings tab's Test button tries a choice before it is saved)."""
    with using(settings):
        return search(query, n)


def search(query: str, n: int = 8) -> list[dict]:
    """[{title, url, snippet}, ...] from the engine chosen in Settings (DuckDuckGo by default); SearchError when it cannot."""
    provider = _setting("JEVH_SEARCH_PROVIDER", "duckduckgo").strip().lower() or "duckduckgo"
    if provider == "duckduckgo":
        return _search_ddg(query, n)
    if provider == "searxng":
        return _search_searxng(query, n)
    if provider == "brave":
        return _search_brave(query, n)
    raise SearchError(f"unknown web search provider {provider!r}: choose one of {', '.join(PROVIDERS)} in Settings")


def _get_json(req: urllib.request.Request, who: str, hints: dict[int, str]) -> dict:
    """The JSON a search API answers with; any failure is a SearchError that says what to do (`hints`: by HTTP status)."""
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        raise SearchError(f"{who} answered HTTP {e.code}. " + hints.get(e.code, "")) from None
    except urllib.error.URLError as e:
        raise SearchError(f"{who} is not reachable ({getattr(e, 'reason', e)}).") from None
    except (TimeoutError, OSError) as e:
        raise SearchError(f"{who} did not answer ({type(e).__name__}).") from None
    except ValueError:
        raise SearchError(f"{who} answered, but not with JSON: is the address right?") from None
    if not isinstance(data, dict):
        raise SearchError(f"{who} answered with something that is not a search result.")
    return data


def _search_searxng(query: str, n: int) -> list[dict]:
    base = _setting("JEVH_SEARXNG_URL").strip().rstrip("/")
    if not base.lower().startswith(("http://", "https://")):
        raise SearchError("SearXNG is chosen but its address is not set: give the server's URL (http://...) in Settings.")
    req = urllib.request.Request(f"{base}/search?" + urllib.parse.urlencode({"q": query, "format": "json"}),
                                 headers={"User-Agent": UA, "Accept": "application/json"})
    data = _get_json(req, f"SearXNG at {base}", {
        403: "That server does not allow the json format: add it under `search: formats:` in its settings.yml.",
        404: "Is the address right? It should be the server's own address, without /search."})
    out = []
    for r in data.get("results") or []:
        if isinstance(r, dict) and r.get("url"):
            out.append({"title": _strip(str(r.get("title") or "")), "url": str(r["url"]),
                        "snippet": _strip(str(r.get("content") or ""))})
        if len(out) >= n:
            break
    return out


def _search_brave(query: str, n: int) -> list[dict]:
    key = (_setting("JEVH_BRAVE_KEY") or os.environ.get("BRAVE_SEARCH_API_KEY", "")).strip()
    if not key:
        raise SearchError("Brave Search is chosen but there is no API key: paste it in Settings "
                          "(or set BRAVE_SEARCH_API_KEY).")
    req = urllib.request.Request(f"{BRAVE_URL}?" + urllib.parse.urlencode({"q": query, "count": max(1, min(int(n), 20))}),
                                 headers={"User-Agent": UA, "Accept": "application/json", "X-Subscription-Token": key})
    data = _get_json(req, "Brave Search", {
        401: "The API key was refused: check it in Settings.", 403: "The API key was refused: check it in Settings.",
        422: "Brave did not accept the search.", 429: "Brave's rate or monthly limit for this key is used up."})
    out = []
    for r in ((data.get("web") or {}).get("results") or []):
        if isinstance(r, dict) and r.get("url"):
            out.append({"title": _strip(str(r.get("title") or "")), "url": str(r["url"]),
                        "snippet": _strip(str(r.get("description") or ""))})
        if len(out) >= n:
            break
    return out


def _search_ddg(query: str, n: int = 8) -> list[dict]:
    body = urllib.parse.urlencode({"q": query}).encode()
    req = urllib.request.Request("https://html.duckduckgo.com/html/", data=body, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        page = r.read().decode("utf-8", "replace")
    blocks = re.split(r'<div class="result results_links', page)[1:]
    if not blocks and CHALLENGE.search(page):
        raise SearchBlocked("DuckDuckGo is asking for a human check (a CAPTCHA) and is not returning results to this "
                            "machine right now. Try another source (the browser on a specific site, a feed), or "
                            "search again later.")
    out = []
    for block in blocks:
        a = re.search(r'class="result__a" href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not a:
            continue
        url = html.unescape(a.group(1))
        if "duckduckgo.com/l/?" in url:  # redirect wrapper -> real target
            url = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("uddg", [url])[0]
        if "duckduckgo.com/y.js" in url:  # ad
            continue
        snip = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', block, re.S)
        out.append({"title": _strip(a.group(2)), "url": url, "snippet": _strip(snip.group(1) if snip else "")})
        if len(out) >= n:
            break
    return out


class _Text(HTMLParser):
    # MathML keeps each formula twice: shown (<mn>28.4</mn>) and as TeX in <annotation>. Both as text made
    # "28.428.4" (live: an ar5iv paper), so the number the answer gave was in no output it could be matched to,
    # and the final() gate sent a correct 28.4 back as made up.
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "aside", "template",
            "annotation", "annotation-xml"}
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article",
             "blockquote", "pre", "table", "ul", "ol", "dd", "dt"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip, self.title, self._in_title = [], 0, "", False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n")
        self._in_title = tag == "title"

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif tag in self.BLOCK:
            self.out.append("\n")
        self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.out.append(data)


def fetch(url: str, timeout: float = 25) -> dict:
    """{url, title, text} for a page (HTML converted to plain text)."""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html,text/plain,*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        ctype = r.headers.get("Content-Type", "").lower()
        raw = r.read(MAX_BYTES)
        final = r.geturl()
    if raw[:5] == b"%PDF-" or "application/pdf" in ctype:
        return {"url": final, "title": "", "text": _pdf_text(raw)}
    if raw[:2048].count(b"\x00") or re.match(r"(image|audio|video|font)/|application/(octet-stream|zip|x-)", ctype):
        # seen live: a PDF decoded as text became 359k chars of junk -> 608 "passages" for Jev to rank
        raise ValueError(f"not a text page ({ctype or 'binary content'})")
    charset = re.search(r"charset=([\w-]+)", ctype)
    body = raw.decode(charset.group(1) if charset else "utf-8", "replace")
    if "html" not in ctype and not body.lstrip().lower().startswith(("<!doctype", "<html")):
        return {"url": final, "title": "", "text": body}
    p = _Text()
    p.feed(body)
    text = re.sub(r"[ \t\r\f\v]+", " ", "".join(p.out))
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return {"url": final, "title": p.title.strip(), "text": text}


def _pdf_text(raw: bytes) -> str:
    try:
        import io

        from pypdf import PdfReader
    except ImportError:
        raise ValueError("a PDF, and no PDF reader is installed (pip install pypdf)") from None
    reader = PdfReader(io.BytesIO(raw))
    return "\n".join((page.extract_text() or "") for page in reader.pages[:40])


def read(url: str, question: str | None = None, max_chars: int = 4000) -> str:
    page = fetch(url)
    if not question:
        return page["text"][:max_chars]
    jev, cap = _client()
    return "\n---\n".join(pick_passages(jev, cap, question, page["text"]))[:max_chars]


def ask(question: str, pages: int = 2, results: int = 8) -> dict:
    """Search, let Jev pick the most promising results, read them in parallel, and return
    the passages that help answer `question` with their sources."""
    try:
        hits = search(question, results)
    except SearchError as e:   # (the answer has its usual shape, and says why it is empty: blocked, no key, server down)
        return {"passages": [], "results": [], "note": str(e)}
    if not hits:
        return {"passages": [], "results": [], "note": "no search results"}
    jev, cap = _client()
    opts = {f"r{i}": f"{h['title']} ({urllib.parse.urlparse(h['url']).netloc}): {h['snippet'][:200]}"
            for i, h in enumerate(hits)}
    probs, _ = choose(jev, {"question": question}, "Which search result is most likely to answer `question`?",
                      opts, cap)
    ranked, domains = [], set()
    for k in sorted(probs, key=probs.get, reverse=True):   # best first, one page per site
        h = hits[int(k[1:])]
        dom = urllib.parse.urlparse(h["url"]).netloc
        if dom not in domains:
            domains.add(dom)
            ranked.append(h)
    spare = ranked[:pages + 2]                              # a slow or blocked site shouldn't cost the answer

    def one(h):
        try:
            text = fetch(h["url"], timeout=10)["text"]
        except (urllib.error.URLError, TimeoutError, ValueError, ConnectionError) as e:
            return {"error": str(e)}
        return {"title": h["title"], "url": h["url"],
                "text": "\n".join(pick_passages(jev, cap, question, text))[:2500]}

    with ThreadPoolExecutor(max_workers=len(spare)) as ex:
        got = list(ex.map(one, spare))
    passages = [g for g in got if "error" not in g][:pages]
    failed = [h["url"] for h, g in zip(spare, got) if "error" in g]
    return {"passages": passages, "results": hits, **({"unreadable": failed} if failed else {})}
