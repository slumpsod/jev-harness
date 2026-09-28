"""Web search and reading, pre-imported as `web`. Jev keeps pages out of your context:
you get the passages that answer your question, not the whole page.

    web.search("espresso machine inventor")          # [{title, url, snippet}, ...] (DuckDuckGo)
    web.read(url, "who patented it and when?")       # only the relevant passages of that page
    web.read(url)                                    # the page as plain text (first 4000 chars)
    web.ask("who patented the first espresso machine?")
        # search -> Jev ranks the results -> reads the best pages in parallel -> relevant
        # passages with their sources: {"passages": [{title, url, text}], "results": [...]}

For pages that need clicking, forms or JavaScript, use the `browser` tool instead.
"""
from __future__ import annotations

import html
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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


def search(query: str, n: int = 8) -> list[dict]:
    body = urllib.parse.urlencode({"q": query}).encode()
    req = urllib.request.Request("https://html.duckduckgo.com/html/", data=body, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        page = r.read().decode("utf-8", "replace")
    out = []
    for block in re.split(r'<div class="result results_links', page)[1:]:
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
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "aside", "template"}
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
        ctype = r.headers.get("Content-Type", "")
        raw = r.read(MAX_BYTES)
        final = r.geturl()
    charset = re.search(r"charset=([\w-]+)", ctype)
    body = raw.decode(charset.group(1) if charset else "utf-8", "replace")
    if "html" not in ctype and not body.lstrip().lower().startswith(("<!doctype", "<html")):
        return {"url": final, "title": "", "text": body}
    p = _Text()
    p.feed(body)
    text = re.sub(r"[ \t\r\f\v]+", " ", "".join(p.out))
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return {"url": final, "title": p.title.strip(), "text": text}


def read(url: str, question: str | None = None, max_chars: int = 4000) -> str:
    page = fetch(url)
    if not question:
        return page["text"][:max_chars]
    jev, cap = _client()
    return "\n---\n".join(pick_passages(jev, cap, question, page["text"]))[:max_chars]


def ask(question: str, pages: int = 2, results: int = 8) -> dict:
    """Search, let Jev pick the most promising results, read them in parallel, and return
    the passages that help answer `question` with their sources."""
    hits = search(question, results)
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
