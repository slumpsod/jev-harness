"""The Newsroom: feeds parsed, widgets stored, alerts that fire and ones that must not, the worker's tool, the
dashboard's API, and a full turn where a chat request puts a widget on the board. All offline: a feed served from
localhost stands in for the web. (Against real sources and real models: evals/bench/newsroom.py.)"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from jevh import newsroom as nr
from jevh.jev import Jev
from jevh.orchestrator import Harness
from jevh.tools import TOOLS
from jevh.tools import newsroom as tool

from conftest import FakeSystemOne, ScriptedLLM, noul_ans, py
from test_web import BASE, TOKEN, auth, make_client


def rss(*items, title="Wire"):
    body = "".join(f"<item><title>{t}</title><link>{l}</link><guid>{l}</guid><pubDate>Tue, 06 Oct 2026 10:00:00 GMT</pubDate>"
                   f"<description>{d}</description></item>" for t, l, d in items)
    return f'<?xml version="1.0"?><rss version="2.0"><channel><title>{title}</title>{body}</channel></rss>'.encode()


GOOGLE_NEWS = b"""<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>"nvidia" - Google News</title>
<item><title>Nvidia beats earnings estimates - Reuters</title><link>https://news.google.com/rss/articles/CBMi1</link>
<guid isPermaLink="false">CBMi1</guid><pubDate>Tue, 06 Oct 2026 08:30:00 GMT</pubDate>
<description>&lt;a href="https://news.google.com/x"&gt;Nvidia beats earnings estimates&lt;/a&gt;&amp;nbsp;&amp;nbsp;&lt;font color="#6f6f6f"&gt;Reuters&lt;/font&gt;</description>
<source url="https://www.reuters.com">Reuters</source></item></channel></rss>"""

ATOM = b"""<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom"><title>Release notes from llama.cpp</title>
<entry><id>tag:github.com,2008:Repository/1/b6500</id><updated>2026-10-05T12:00:00Z</updated>
<link rel="alternate" type="text/html" href="https://github.com/ggml-org/llama.cpp/releases/tag/b6500"/>
<title>b6500</title><author><name>ggerganov</name></author><content type="html">&lt;p&gt;Faster &lt;b&gt;prompt&lt;/b&gt; processing&lt;/p&gt;</content></entry></feed>"""


class Feed:
    """A feed on localhost whose body a test can change."""

    def __init__(self):
        feed = self
        self.body, self.hits, self.status = rss(), 0, 200

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                feed.hits += 1
                self.send_response(feed.status)
                self.send_header("Content-Type", "application/rss+xml")
                self.end_headers()
                self.wfile.write(feed.body)

            def log_message(self, *a):
                pass
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/feed.xml"

    def set(self, *items):
        self.body = rss(*items)


@pytest.fixture
def feed(monkeypatch):
    monkeypatch.setenv("JEVH_NEWSROOM_ALLOW_LOCAL", "1")
    f = Feed()
    yield f
    f.srv.shutdown()


@pytest.fixture
def room(tmp_path):
    return nr.Newsroom(tmp_path / "newsroom.db")


@pytest.fixture
def board(tmp_path, monkeypatch):
    """The worker's tool pointed at a store of its own."""
    monkeypatch.setenv("JEVH_NEWSROOM_DB", str(tmp_path / "tool.db"))
    tool._room = None
    yield tool
    tool._room = None


A = ("Nvidia beats earnings estimates", "https://ex.test/a", "Revenue up")
B = ("Chip export rules change", "https://ex.test/b", "Policy")
C = ("Nvidia unveils a new GPU", "https://ex.test/c", "Launch event")


# ---------------------------------------------------------------- reading feeds

def test_google_news_headlines_lose_the_outlet_suffix_and_the_repeated_description():
    (it,) = nr.parse_feed(GOOGLE_NEWS)
    assert it["title"] == "Nvidia beats earnings estimates" and it["source"] == "Reuters"
    assert it["summary"] == "" and it["url"].startswith("https://news.google.com/")
    assert it["uid"] == "CBMi1" and it["published"] > 1.7e9


def test_a_feed_widget_takes_the_feeds_own_name_unless_it_was_named(room, feed):
    feed.set(A)
    a = room.add_widget("rss", {"url": feed.url})
    b = room.add_widget("rss", {"url": feed.url + "?b"}, title="Mine")
    room.refresh(a["id"])
    room.refresh(b["id"])
    assert room.widget(a["id"])["title"] == "Feed: Wire" and room.widget(b["id"])["title"] == "Mine"


def test_atom_entries_have_links_authors_and_plain_text_summaries():
    (it,) = nr.parse_feed(ATOM)
    assert it["url"] == "https://github.com/ggml-org/llama.cpp/releases/tag/b6500" and it["author"] == "ggerganov"
    assert it["summary"] == "Faster prompt processing" and it["source"] == "Release notes from llama.cpp"
    assert it["published"] is not None


def test_a_feed_with_entity_tricks_or_no_feed_at_all_is_an_error_not_a_hang():
    with pytest.raises(nr.NewsroomError, match="DOCTYPE"):
        nr.parse_feed(b'<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY a "aaaa">]><rss><channel><item><title>&a;</title></item></channel></rss>')
    with pytest.raises(nr.NewsroomError, match="not a feed"):
        nr.parse_feed(b"<html><body>Not a feed</body>")


def test_links_that_are_not_web_addresses_never_reach_the_board():
    it = nr._clean_item({"title": "x", "url": "javascript:alert(1)", "uid": "u"})
    assert it["url"] == ""


def test_only_the_public_web_is_followed(monkeypatch):
    monkeypatch.delenv("JEVH_NEWSROOM_ALLOW_LOCAL", raising=False)
    for url in ("http://127.0.0.1:9/feed", "http://localhost/feed", "http://169.254.169.254/latest/meta-data", "http://192.168.1.5/rss"):
        with pytest.raises(nr.NewsroomError, match="on this machine or its network"):
            nr.check_url(url)
    for url in ("file:///etc/passwd", "ftp://example.com/x", "javascript:alert(1)"):
        with pytest.raises(nr.NewsroomError, match="not a web address"):
            nr.check_url(url)
    with pytest.raises(nr.NewsroomError, match="on this machine"):   # a redirect to the inside is refused too
        nr._Redirects().redirect_request(nr.urllib.request.Request("https://example.com/"), None, 302, "Found", {}, "http://127.0.0.1/admin")


# ---------------------------------------------------------------- widgets

def test_a_widget_needs_what_its_kind_needs_and_the_error_says_what_that_is(room):
    with pytest.raises(nr.NewsroomError, match=r"news needs topic .*e\.g\. 'Nvidia earnings'"):
        room.add_widget("news", {})
    with pytest.raises(nr.NewsroomError, match="news takes days, language, region, site, topic; not subject"):
        room.add_widget("news", {"topic": "x", "subject": "y"})
    with pytest.raises(nr.NewsroomError, match="unknown kind 'tiktok'"):
        room.add_widget("tiktok", {})
    with pytest.raises(nr.NewsroomError, match="either an account"):
        room.add_widget("x", {})
    with pytest.raises(nr.NewsroomError, match="not a web address"):
        room.add_widget("rss", {"url": "file:///etc/passwd"})
    assert room.add_widget("x", {"user": "@OpenAI"})["params"] == {"user": "OpenAI"}


def test_adding_the_same_widget_twice_returns_the_first(room):
    a = room.add_widget("news", {"topic": "Nvidia"})
    b = room.add_widget("news", {"topic": "nvidia"})
    assert b["id"] == a["id"] and b["existing"] and len(room.widgets()) == 1


def test_every_is_kept_between_the_minimum_and_a_week_and_the_board_has_a_limit(tmp_path):
    room = nr.Newsroom(tmp_path / "n.db", {"max_widgets": 2, "min_minutes": 5})
    assert room.add_widget("news", {"topic": "a"}, every=1)["every"] == 5
    assert room.add_widget("news", {"topic": "b"}, every=10 ** 9)["every"] == 60 * 24 * 7
    with pytest.raises(nr.NewsroomError, match="holds 2 widgets"):
        room.add_widget("news", {"topic": "c"})
    with pytest.raises(nr.NewsroomError, match="minutes"):
        room.add_widget("news", {"topic": "d"}, every="often")


def test_widgets_can_be_moved_renamed_paused_and_pointed_at_something_else(room, feed):
    a, b, c = (room.add_widget("news", {"topic": t}) for t in "abc")
    room.move(c["id"], -1)
    assert [w["title"] for w in room.widgets()] == ["News: a", "News: c", "News: b"]
    room.move(a["id"], -1)   # already first: stays
    assert room.widgets()[0]["id"] == a["id"]
    room.update_widget(a["id"], title="Alpha", paused=True, size="l", alert="x, y")
    w = room.widget(a["id"])
    assert (w["title"], w["paused"], w["size"], w["alert"]["keywords"]) == ("Alpha", True, "l", ["x", "y"])
    assert a["id"] not in room.due()
    feed.set(A)
    r = room.add_widget("rss", {"url": feed.url})
    room.refresh(r["id"])
    assert room.items(r["id"])
    room.update_widget(r["id"], params={"url": feed.url + "?other=1"})
    assert room.items(r["id"]) == [] and room.widget(r["id"])["last_ok"] is None   # what it found belonged to the old address
    with pytest.raises(nr.NewsroomError, match="size must be"):
        room.update_widget(a["id"], size="xl")
    assert room.delete_widget(a["id"]) and room.widget(a["id"]) is None


def test_the_revision_goes_up_on_every_change(room):
    r0 = room.rev()
    w = room.add_widget("news", {"topic": "x"})
    r1 = room.rev()
    room.update_widget(w["id"], title="y")
    room.notify("t")
    assert r0 < r1 < room.rev()


# ---------------------------------------------------------------- looking, and what notifies

def test_the_first_look_is_the_baseline_and_only_later_items_notify(room, feed):
    feed.set(A, B)
    w = room.add_widget("rss", {"url": feed.url}, alert="earnings")
    first = room.refresh(w["id"])
    assert first["first"] and first["new"] == 0 and first["alerts"] == []        # "Nvidia beats earnings" was already there
    assert len(room.items(w["id"])) == 2 and room.widgets()[0]["unread"] == 0
    feed.set(C, A, B)
    again = room.refresh(w["id"])
    assert again["new"] == 1 and again["alerts"] == []                            # new, but says nothing about earnings
    feed.set(("Nvidia earnings call moved", "https://ex.test/d", "Date change"), C, A, B)
    third = room.refresh(w["id"])
    (note,) = third["alerts"]
    assert note["body"] == "Nvidia earnings call moved" and note["why"] == "mentions earnings" and note["url"] == "https://ex.test/d"
    assert room.refresh(w["id"])["new"] == 0 and room.unread_notes() == 1          # seen again is not new
    assert room.widgets()[0]["unread"] == 2


def test_a_github_project_keeps_its_slash_and_other_values_are_still_encoded(monkeypatch):
    """The feed URL had the project's owner/name encoded to owner%2Fname, and GitHub answered 404 for every project."""
    asked = []
    monkeypatch.setattr(nr, "http_get", lambda url, *a, **k: asked.append(url) or b"<rss><channel/></rss>")
    nr.KINDS["github"].fetch({"repo": "ggml-org/llama.cpp"})
    nr.KINDS["hn"].fetch({"query": "llama.cpp & vision/audio"})
    assert asked[0] == "https://github.com/ggml-org/llama.cpp/releases.atom"
    assert asked[1] == "https://hnrss.org/newest?q=llama.cpp%20%26%20vision%2Faudio"      # a search is encoded whole


@pytest.fixture
def shuffling(monkeypatch):
    """A `web`-like kind whose results the test sets: what a search engine does, unlike a feed."""
    results = {"now": []}
    kind = nr.Kind("shuffle", "Shuffle", "test", [nr.Param("q", "x", True)], lambda p: list(results["now"]),
                   lambda p: "Shuffle", settle=2)
    monkeypatch.setitem(nr.KINDS, "shuffle", kind)
    monkeypatch.setattr(nr, "clean_params", lambda k, p: p)
    return results


def hit(n):
    return {"title": f"Page {n}", "url": f"https://ex.test/{n}", "summary": "", "source": "ex.test"}


def test_a_page_is_new_only_when_it_stays_in_the_results_for_two_looks(room, shuffling):
    """DuckDuckGo moved 2 of its 10 results between two identical searches a few seconds apart: each swap was "new"."""
    shuffling["now"] = [hit(1), hit(2), hit(3)]
    w = room.add_widget("shuffle", {"q": "x"}, alert=True)
    assert room.refresh(w["id"])["first"]                                              # the baseline holds all three
    shuffling["now"] = [hit(1), hit(2), hit(4)]                                         # page 3 slid out, page 4 slid in
    one = room.refresh(w["id"])
    assert one["new"] == 0 and one["alerts"] == []                                      # once is churn, not news
    shuffling["now"] = [hit(1), hit(2), hit(3)]                                         # ...and it slid back: nothing happened
    assert room.refresh(w["id"])["new"] == 0
    shuffling["now"] = [hit(5), hit(1), hit(2)]
    assert room.refresh(w["id"])["new"] == 0
    shuffling["now"] = [hit(5), hit(1), hit(2)]                                         # page 5 is still there: it is news
    two = room.refresh(w["id"])
    assert two["new"] == 1 and [n["body"] for n in two["alerts"]] == ["Page 5"]
    assert room.refresh(w["id"])["new"] == 0                                            # and only once
    assert {i["title"] for i in room.items(w["id"])} == {"Page 1", "Page 2", "Page 3", "Page 5"}


def test_an_empty_answer_from_a_rate_limited_search_changes_nothing(room, shuffling):
    shuffling["now"] = [hit(1)]
    w = room.add_widget("shuffle", {"q": "x"}, alert=True)
    room.refresh(w["id"])
    shuffling["now"] = [hit(1), hit(2)]
    assert room.refresh(w["id"])["new"] == 0
    shuffling["now"] = []                                                               # DuckDuckGo says nothing for a while
    assert room.refresh(w["id"])["new"] == 0
    shuffling["now"] = [hit(1), hit(2)]
    assert room.refresh(w["id"])["new"] == 1                                            # page 2 had been seen before the gap


def test_pending_items_go_with_a_widget_and_with_a_changed_search(room, shuffling):
    shuffling["now"] = [hit(1)]
    w = room.add_widget("shuffle", {"q": "x"})
    room.refresh(w["id"])
    shuffling["now"] = [hit(1), hit(2)]
    room.refresh(w["id"])
    assert room.db.execute("SELECT COUNT(*) FROM pending").fetchone()[0] == 1
    room.update_widget(w["id"], params={"q": "y"})
    assert room.db.execute("SELECT COUNT(*) FROM pending").fetchone()[0] == 0
    room.refresh(w["id"])
    room.refresh(w["id"])
    room.delete_widget(w["id"])
    assert room.db.execute("SELECT COUNT(*) FROM pending").fetchone()[0] == 0


def test_a_feed_still_counts_an_item_as_new_at_once(room, feed):
    feed.set(A, B)
    w = room.add_widget("rss", {"url": feed.url})
    room.refresh(w["id"])
    feed.set(C, A, B)
    assert room.refresh(w["id"])["new"] == 1                                            # settle is 1 for a feed


def test_alert_words_match_whole_words_so_ai_is_not_said():
    one = lambda kw, title: nr.alert_matches(nr.clean_alert(kw), {"title": title})
    assert one("AI", "OpenAI said no") is None and one("AI", "New AI chip") == "mentions AI"
    assert one("launch", "Company launches product") and one("launch", "A prelaunch party") is None
    assert one(["earnings", "guidance"], "Guidance raised") == "mentions guidance"
    assert one({"keywords": ["nvidia"], "exclude": ["rumor"]}, "Nvidia rumor mill") is None
    assert one(True, "anything") == "new item" and one(None, "anything") is None and one("", "anything") is None
    assert nr.clean_alert("a, b;c\nd")["keywords"] == ["a", "b", "c", "d"]


def test_an_about_alert_is_judged_by_jev_and_a_keyword_still_needs_no_judge(room, feed):
    asked = []

    def judge(text, about):
        asked.append((text, about))
        return 0.9 if "quarter" in text else 0.1
    room.judge = judge
    feed.set(A)
    w = room.add_widget("rss", {"url": feed.url}, alert={"keywords": ["lawsuit"], "about": "their quarterly results"})
    room.refresh(w["id"])
    assert asked == []                                                         # the first look judges nothing
    feed.set(("Nvidia reports record quarter", "https://ex.test/q", "Revenue"), ("Nvidia sued in lawsuit", "https://ex.test/l", ""),
             ("Nvidia sponsors a race team", "https://ex.test/r", ""), A)
    got = {n["body"]: n["why"] for n in room.refresh(w["id"])["alerts"]}
    assert got == {"Nvidia reports record quarter": "about their quarterly results", "Nvidia sued in lawsuit": "mentions lawsuit"}
    assert len(asked) == 2 and all(a[1] == "their quarterly results" for a in asked)   # the keyword hit was not sent to Jev
    room.judge = None                                                          # no Jev: keywords only, nothing breaks
    feed.set(("Another record quarter", "https://ex.test/q2", ""), *[(t, u, d) for t, u, d in (A,)])
    assert room.refresh(w["id"])["alerts"] == []
    room.judge = lambda t, a: 1 / 0                                            # Jev failing on an item is not a crash
    feed.set(("Yet another quarter", "https://ex.test/q3", ""), A)
    assert room.refresh(w["id"])["alerts"] == []
    assert nr.clean_alert({"about": "  results  "})["about"] == "results" and nr.clean_alert("a")["about"] == ""


def test_a_flood_of_matches_is_a_handful_of_notifications_and_one_summary(room, feed):
    feed.set(A)
    w = room.add_widget("rss", {"url": feed.url}, alert=True)
    room.refresh(w["id"])
    feed.set(*[(f"Story {i}", f"https://ex.test/s{i}", "") for i in range(12)], A)
    r = room.refresh(w["id"])
    assert r["new"] == 12 and len(r["alerts"]) == nr.MAX_NOTES_PER_LOOK + 1
    assert r["alerts"][-1]["body"] == "and 7 more new items"


def test_a_source_that_fails_is_a_state_of_the_widget_and_is_tried_less_and_less_often(room, feed):
    w = room.add_widget("rss", {"url": feed.url}, every=60)
    feed.status = 503
    t = 1_000_000.0
    r = room.refresh(w["id"], now=t)
    assert "HTTP 503" in r["error"] and room.widget(w["id"])["fails"] == 1
    assert room.due(t + 3600 + 1) == []             # failed once: waits twice as long
    assert room.due(t + 7200 + 1) == [w["id"]]
    for k in range(6):
        room.refresh(w["id"], now=t + k)
    assert room.due(t + 5 + 6 * 3600 - 10) == [] and room.due(t + 5 + 6 * 3600 + 10) == [w["id"]]   # never longer than 6 hours
    feed.status = 200
    feed.set(A)
    assert room.refresh(w["id"], now=t + 7 * 3600)["error"] == "" and room.widget(w["id"])["fails"] == 0


def test_only_the_newest_items_are_kept(tmp_path, feed):
    room = nr.Newsroom(tmp_path / "n.db", {"keep_items": 5})
    feed.set(*[(f"Story {i}", f"https://ex.test/s{i}", "") for i in range(9)])
    w = room.add_widget("rss", {"url": feed.url})
    room.refresh(w["id"])
    assert len(room.items(w["id"], 50)) == 5


def test_the_runner_looks_at_what_is_due_and_says_when_it_notified(room, feed):
    feed.set(A)
    w = room.add_widget("rss", {"url": feed.url}, alert=True, every=5)
    said, changed = [], []
    runner = nr.Runner(room, on_alerts=said.extend, on_change=lambda: changed.append(1), tick=0.01)
    assert runner.once(now=100.0)[0]["first"] and not said
    assert runner.once(now=101.0) == []                          # not due yet
    feed.set(B, A)
    results = runner.once(now=100.0 + 5 * 60 + 1)
    assert results[0]["new"] == 1 and said[0]["body"] == B[0] and len(changed) == 2


def test_the_runner_survives_a_widget_whose_kind_is_gone(room):
    w = room.add_widget("news", {"topic": "x"})
    room._run("UPDATE widgets SET kind='removed' WHERE id=?", (w["id"],))
    (r,) = nr.Runner(room).once()
    assert r["error"] and room.widget(w["id"])["fails"] == 1


# ---------------------------------------------------------------- X, which needs setting up

def test_x_says_what_to_set_up_and_keeps_the_widget(room, monkeypatch):
    for var in ("X_BEARER_TOKEN", "JEVH_NEWSROOM_X_FEED"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(nr, "_x_cache", {"ts": 0.0, "value": None})
    monkeypatch.setattr(nr, "_x_look", lambda: {"feed": "", "bearer": "", "nango": ""})
    assert "X_BEARER_TOKEN" in nr.KINDS["x"].public()["needs"]
    w = room.add_widget("x", {"user": "OpenAI"})
    r = room.refresh(w["id"])
    assert "newsroom.x_feed" in r["error"] and room.widget(w["id"])["fails"] == 1


def test_x_through_an_account_connected_in_nango_or_an_rss_bridge(room, feed, monkeypatch):
    monkeypatch.setattr(nr, "_x_cache", {"ts": 0.0, "value": None})
    calls = []

    def api(path, params, setup):
        calls.append(path)
        if path.endswith("/users/by/username/OpenAI"):
            return {"data": {"id": "42"}}
        return {"data": [{"id": "9001", "text": "We shipped a new model", "created_at": "2026-10-06T10:00:00.000Z"}]}
    monkeypatch.setattr(nr, "_x_look", lambda: {"feed": "", "bearer": "", "nango": "twitter"})
    monkeypatch.setattr(nr, "_x_api", api)
    monkeypatch.setattr(nr, "_x_ids", {})
    w = room.add_widget("x", {"user": "OpenAI"})
    assert room.refresh(w["id"])["error"] == ""
    assert calls == ["/2/users/by/username/OpenAI", "/2/users/42/tweets"]
    (it,) = room.items(w["id"])
    assert it["title"] == "We shipped a new model" and it["url"] == "https://x.com/OpenAI/status/9001" and it["source"] == "@OpenAI"
    # a bridge you run: {user} in its address
    feed.set(("OpenAI: hello", "https://x.com/OpenAI/status/1", ""))
    monkeypatch.setattr(nr, "_x_cache", {"ts": 0.0, "value": None})
    monkeypatch.setattr(nr, "_x_look", lambda: {"feed": feed.url + "?u={user}", "bearer": "", "nango": ""})
    w2 = room.add_widget("x", {"user": "Anthropic"})
    assert room.refresh(w2["id"])["error"] == "" and room.items(w2["id"])[0]["title"] == "OpenAI: hello"


# ---------------------------------------------------------------- the worker's tool

def test_the_tool_adds_a_widget_looks_once_and_says_what_it_found(board, feed):
    feed.set(A, B)
    r = board.add("rss", url=feed.url, alert="earnings", title="Wire")
    assert r["title"] == "Wire" and r["found"] == 2 and r["alerts"] == ["earnings"] and r["every_minutes"] == 30
    assert [x["title"] for x in r["latest"]] == [A[0], B[0]] and "baseline" in r["note"]
    again = board.add("rss", url=feed.url)
    assert again["existing"] and len(board.widgets()) == 1


def test_the_tool_finds_a_widget_by_id_or_by_part_of_its_title(board, feed):
    feed.set(A)
    board.add("rss", url=feed.url, title="Nvidia wire", look=False)
    board.add("news", topic="Nvidia", look=False)
    assert board.edit("wire", every=15)["every_minutes"] == 15
    with pytest.raises(nr.NewsroomError, match=r"matches 2 widgets: 1 Nvidia wire; 2 News: Nvidia"):
        board.edit("nvidia", every=15)
    with pytest.raises(nr.NewsroomError, match="no widget 'zzz'; on the board: 1 Nvidia wire; 2 News: Nvidia"):
        board.remove("zzz")
    assert board.edit(2, alert=True)["alerts"] == "every new item"
    assert board.edit("#2", alert=None)["alerts"] == "none"
    assert board.remove("wire") == {"removed": "Nvidia wire"} and [w["id"] for w in board.widgets()] == [2]


def test_the_tool_sets_an_about_alert_beside_keywords(board, feed):
    feed.set(A)
    r = board.add("rss", url=feed.url, title="Wire", alert="lawsuit", about="their quarterly results", look=False)
    assert r["alerts"] == ["lawsuit", "about: their quarterly results"]
    assert board.edit("Wire", about="product launches")["alerts"] == ["lawsuit", "about: product launches"]
    assert board.edit("Wire", alert=True)["alerts"] == "every new item"
    assert board.edit("Wire", alert=None)["alerts"] == "none"


def test_the_tool_edits_what_a_widget_follows_and_refreshes(board, feed):
    feed.set(A)
    board.add("rss", url=feed.url, title="Wire")
    other = Feed()
    other.set(B)
    try:
        assert board.edit("Wire", url=other.url)["params"] == {"url": other.url}
        assert board.refresh("Wire") == [{"widget": 1, "new": 0, "alerts": 0, "error": ""}]
        assert board.items("Wire")[0]["title"] == B[0]
    finally:
        other.srv.shutdown()


def test_the_tool_tells_a_failed_first_look_and_what_to_set_up(board, monkeypatch):
    monkeypatch.setattr(nr, "_x_look", lambda: {"feed": "", "bearer": "", "nango": ""})
    monkeypatch.setattr(nr, "_x_cache", {"ts": 0.0, "value": None})
    r = board.add("x", user="OpenAI", alert=True)
    assert "X_BEARER_TOKEN" in r["needs"] and "newsroom.x_feed" in r["error"] and r["id"] == 1
    assert board.widgets()[0]["error"]


def test_the_tool_can_notify_on_its_own_and_make_a_company_pair(board, feed, monkeypatch):
    assert board.notify("Report is out", "Q3 summary", "https://ex.test/q3")["sent"] == "Report is out"
    assert board._board().notes()[0]["why"] == "from the assistant"
    assert board._board().notify("bad link", url="javascript:alert(1)")["url"] == ""
    monkeypatch.setattr(nr, "_x_look", lambda: {"feed": "", "bearer": "", "nango": ""})
    monkeypatch.setattr(nr, "_x_cache", {"ts": 0.0, "value": None})
    monkeypatch.setattr(nr.KINDS["news"], "fetch", lambda p: nr.parse_feed(GOOGLE_NEWS))
    pair = board.company("Nvidia", x="nvidia", alert="earnings")
    assert [w["kind"] for w in pair] == ["news", "x"] and pair[0]["found"] == 1
    assert pair[0]["alerts"] == ["earnings"] == pair[1]["alerts"]


def test_the_guide_covers_every_kind_and_the_persona_has_the_tool():
    from jevh.personas import load_personas
    guide = TOOLS["newsroom"].guide
    for name in nr.KINDS:
        assert f"{name} (" in guide or f"{name}," in guide or f"{name}=" in guide or f"{name})" in guide, f"the guide doesn't mention the {name} kind"
    from jevh.config import ROOT
    assert "newsroom" in load_personas(ROOT / "personas")["daily_assistant"].tools
    assert "alert" in TOOLS["newsroom"].description and "one-off" in TOOLS["newsroom"].description


# ---------------------------------------------------------------- the dashboard's API

def newsroom_client(cfg, monkeypatch):
    monkeypatch.setenv("JEVH_NEWSROOM_ALLOW_LOCAL", "1")
    client, app = make_client(cfg)
    return client, app.state.hub


def test_the_api_adds_changes_and_removes_widgets(cfg, feed, monkeypatch):
    client, hub = newsroom_client(cfg, monkeypatch)
    get = lambda **q: client.get("/api/newsroom", params=q, headers=auth()).json()
    post = lambda path, body=None: client.post(f"/api/newsroom/{path}", json=body or {}, headers=auth())
    d = get()
    assert d["widgets"] == [] and [k["name"] for k in d["kinds"]][:3] == ["news", "web", "rss"] and d["running"]
    feed.set(A, B)
    r = post("add", {"kind": "rss", "params": {"url": feed.url}, "title": "Wire", "every": 10, "alert": {"keywords": ["earnings"]}}).json()
    assert r["widget"]["title"] == "Wire" and not r["existing"]
    wid = r["widget"]["id"]
    d = get()
    assert len(d["widgets"][0]["items"]) == 2 and d["widgets"][0]["every"] == 10 and d["widgets"][0]["alert"]["keywords"] == ["earnings"]
    assert get(rev=d["rev"]) == {"rev": d["rev"], "unchanged": True, "unread_notes": 0}
    assert post("add", {"kind": "rss", "params": {"url": feed.url}}).json()["existing"]
    assert post("add", {"kind": "news", "params": {}}).status_code == 400
    assert "needs topic" in post("add", {"kind": "news", "params": {}}).json()["detail"]
    assert post("update", {"id": wid, "title": "Renamed", "paused": True, "alert": None}).json()["widget"]["paused"]
    assert post("update", {"id": 999, "title": "x"}).status_code == 400
    assert post("delete", {"id": "abc"}).status_code == 400 and post("move", {"id": wid, "step": "up"}).status_code == 400
    feed.set(C, A, B)
    assert post("refresh", {"id": wid}).json()["results"][0]["new"] == 1
    assert get()["widgets"][0]["unread"] == 1
    post("read", {"widget": wid})
    assert get()["widgets"][0]["unread"] == 0
    other = post("add", {"kind": "news", "params": {"topic": "x"}}).json()["widget"]["id"]
    post("move", {"id": other, "step": -1})
    assert [w["id"] for w in get()["widgets"]] == [other, wid]
    assert post("delete", {"id": wid}).status_code == 200 and post("delete", {"id": wid}).status_code == 404


def test_api_calls_need_the_token_like_the_rest(cfg, monkeypatch):
    client, _ = newsroom_client(cfg, monkeypatch)
    assert client.get("/api/newsroom").status_code == 403
    assert client.post("/api/newsroom/delete", json={"id": 1}).status_code == 403
    assert client.post("/api/newsroom/add", json={}, headers={**auth(), "Origin": "http://evil.example"}).status_code == 403


def test_alerts_reach_the_page_as_events_and_the_bell(cfg, feed, monkeypatch):
    client, hub = newsroom_client(cfg, monkeypatch)
    feed.set(A)
    post = lambda path, body=None: client.post(f"/api/newsroom/{path}", json=body or {}, headers=auth()).json()
    wid = post("add", {"kind": "rss", "params": {"url": feed.url}, "alert": True})["widget"]["id"]
    feed.set(B, A)
    post("refresh", {"id": wid})
    ev = [e for e in hub.events if e["kind"] == "newsroom_alert"]
    assert ev and ev[-1]["data"]["notes"][0]["body"] == B[0]
    d = client.get("/api/newsroom", headers=auth()).json()
    assert d["unread_notes"] == 1 and d["notes"][0]["title"] == "News: rss" or d["notes"][0]["body"] == B[0]
    post("notes/read", {"id": d["notes"][0]["id"]})
    assert client.get("/api/newsroom", headers=auth()).json()["unread_notes"] == 0
    post("notes/clear")
    assert client.get("/api/newsroom", headers=auth()).json()["notes"] == []


def test_the_dashboard_asks_jev_whether_an_item_is_about_what_the_user_wants(cfg, feed, monkeypatch):
    monkeypatch.setenv("JEVH_NEWSROOM_ALLOW_LOCAL", "1")
    asked = []

    def rules(k, q, s):
        if k == "match":
            asked.append((s, q["instructions"]))
            return noul_ans(0.95 if "record" in s["item"] else 0.05)
    from test_web import rules as web_rules
    from conftest import FakeSystemOne as Fake
    client, app = make_client(cfg)
    hub = app.state.hub
    hub.harness.jev.client = Fake(lambda k, q, s: rules(k, q, s) or web_rules(k, q, s))
    assert hub.newsroom.judge("Nvidia posts record quarter", "their results") >= cfg.t("newsroom_about")
    assert hub.newsroom.judge("Nvidia sponsors a race", "their results") < cfg.t("newsroom_about")
    assert asked[0][0] == {"item": "Nvidia posts record quarter", "interest": "their results"}
    assert hub.newsroom.cfg["about_threshold"] == cfg.t("newsroom_about") == 0.7


def test_the_runner_starts_with_the_app_and_looks_by_itself(cfg, feed, monkeypatch):
    monkeypatch.setenv("JEVH_NEWSROOM_ALLOW_LOCAL", "1")
    cfg.raw = {**cfg.raw, "newsroom": {**cfg.raw["newsroom"], "tick": 0.05}}
    client, app = make_client(cfg)
    feed.set(A)
    with client:   # (runs the app's startup)
        hub = app.state.hub
        w = hub.newsroom.add_widget("rss", {"url": feed.url}, alert=True)
        deadline = time.time() + 10
        while time.time() < deadline and not hub.newsroom.widget(w["id"])["last_ok"]:
            time.sleep(0.05)
        assert hub.newsroom.widget(w["id"])["last_ok"], "the runner never looked"
        feed.set(B, A)
        hub.newsroom._run("UPDATE widgets SET last_look=0 WHERE id=?", (w["id"],))   # (its time has come)
        while time.time() < deadline and not hub.newsroom.notes():
            time.sleep(0.05)
        assert hub.newsroom.notes()[0]["body"] == B[0]
    assert hub.newsroom_runner._stop.is_set()


# ---------------------------------------------------------------- asking in the chat

def test_asking_in_the_chat_puts_a_widget_on_the_board(cfg, feed, monkeypatch):
    """A request goes through Jev's tool pick to the worker, whose code calls newsroom.add; the board has the widget."""
    monkeypatch.setenv("JEVH_NEWSROOM_ALLOW_LOCAL", "1")
    feed.set(A, B)
    seen = {}

    def rules(k, q, s):
        if k.startswith("t_"):
            seen[k] = q["instructions"]["tool"]
        if k == "t_newsroom":
            return noul_ans(0.9)
    code = py(f"r = newsroom.add('rss', url={feed.url!r}, title='Wire', alert='earnings, guidance')\n"
              "final(f\"Added {r['title']}: {r['found']} items; I will notify you about {r['alerts']}.\")")
    llm = ScriptedLLM([code])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules)), llm_factory=lambda p: llm)
    h.learning = False
    res = h.turn("keep an eye on the wire and tell me when they mention earnings or guidance", force="daily_assistant")
    assert res.answer == "Added Wire: 2 items; I will notify you about ['earnings', 'guidance']."
    assert "newsroom.add(kind" in llm.seen[0][0]["content"]                   # the guide reached the worker's prompt
    assert "Newsroom dashboard" in seen["t_newsroom"] and "t_web" in seen      # Jev was offered it beside the others
    room = nr.Newsroom(cfg.workspace / "newsroom.db")
    (w,) = room.widgets()
    assert w["title"] == "Wire" and w["alert"]["keywords"] == ["earnings", "guidance"] and len(room.items(w["id"])) == 2


# ---------------------------------------------------------------- jevh newsroom

def test_the_command_line_adds_looks_lists_and_removes(cfg, feed, monkeypatch, capsys):
    from jevh import cli
    import jevh.config
    monkeypatch.setattr(jevh.config, "load_config", lambda *a, **k: cfg)
    main = cli.subcommand("newsroom")
    feed.set(A, B)
    assert main(["kinds"]) == 0
    assert "news     News:" in capsys.readouterr().out
    assert main(["add", "rss", f"url={feed.url}", "--alert", "earnings", "--title", "Wire"]) == 0
    assert "added #1 Wire: 2 items" in capsys.readouterr().out
    assert main(["add", "rss", f"url={feed.url}"]) == 0 and "already on the board as #1" in capsys.readouterr().out
    assert main([]) == 0
    listing = capsys.readouterr().out
    assert "Wire" in listing and "alerts: earnings" in listing
    assert main(["items", "1"]) == 0 and "Nvidia beats earnings estimates" in capsys.readouterr().out
    assert main(["refresh"]) == 0 and "#1: 0 new, 0 notifications" in capsys.readouterr().out
    assert main(["add", "news"]) == 1 and "news needs topic" in capsys.readouterr().err
    assert main(["rm", "1"]) == 0 and "removed" in capsys.readouterr().out
