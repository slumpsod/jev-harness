"""The Newsroom's tile summaries (notebook #17, 2026-10-23: "a summary of what it found from its last search, and then at
the bottom a spot that says sources that then leads me to the links"): what is written from what, when it is written
again, that it waits for the user's turn to end, and the tile in a real browser. Live: evals/bench/newsroom_summary.py."""
import pytest

from jevh import newsroom as nr

from conftest import ScriptedLLM
from test_newsroom import A, B, C, Feed, feed  # noqa: F401

ITEMS = [{"title": "Nvidia beats earnings estimates", "url": "https://ex.test/a", "source": "Reuters", "summary": "Revenue up"},
         {"title": "Chip export rules change", "url": "https://ex.test/b", "source": "AP", "summary": "Policy"},
         {"title": "Nvidia unveils a new GPU", "url": "https://ex.test/c", "source": "", "summary": ""}]


@pytest.fixture
def room(tmp_path):
    return nr.Newsroom(tmp_path / "newsroom.db")


def test_a_summary_cites_its_items_and_lists_them_as_sources():
    s = nr.parse_summary("Nvidia beat estimates [1]. Export rules changed [2][9].", ITEMS)
    assert s["text"] == "Nvidia beat estimates [1]. Export rules changed [2]."   # [9] names no item: dropped
    assert [(x["n"], x["url"], x["source"]) for x in s["sources"]] == [(1, "https://ex.test/a", "Reuters"),
                                                                      (2, "https://ex.test/b", "AP")]


def test_a_summary_that_cites_nothing_lists_every_item_and_loses_a_models_leftovers():
    s = nr.parse_summary("<think>hmm</think>Summary: Nvidia had a busy week.", ITEMS)
    assert s["text"] == "Nvidia had a busy week." and [x["n"] for x in s["sources"]] == [1, 2, 3]
    with pytest.raises(nr.NewsroomError):
        nr.parse_summary("<think>only thinking</think>", ITEMS)


def test_the_prompt_numbers_the_items_and_keeps_to_what_they_say(room, feed):
    feed.set(A, B)
    w = room.add_widget("rss", {"url": feed.url})
    room.refresh(w["id"])
    llm = ScriptedLLM(["Nvidia beat estimates [2]."])
    s = nr.summarize_with(llm, room.widget(w["id"]), room.items(w["id"]))
    prompt = llm.seen[0][0]["content"]
    assert "[1] Chip export rules change" in prompt or "[1] Nvidia beats earnings estimates" in prompt
    assert "Use only what the titles and snippets say" in prompt and feed.url in prompt
    assert s["sources"][0]["n"] == 2


def test_a_look_that_finds_something_asks_for_a_summary_and_one_that_finds_nothing_does_not(room, feed):
    feed.set(A, B)
    w = room.add_widget("rss", {"url": feed.url})
    room.refresh(w["id"])
    assert room.summaries_wanted() == [w["id"]]          # the first look: there is none yet
    seen = []
    out = room.write_summary(w["id"], lambda widget, items: seen.append(items) or nr.parse_summary("Two stories [1][2].", items))
    assert len(seen[0]) == 2 and out["items"] == 2 and out["ts"] > 0
    got = room.widget(w["id"])
    assert got["summary"]["text"] == "Two stories [1][2]." and not got["summary_wanted"] and room.summaries_wanted() == []
    room.refresh(w["id"])                                  # nothing new: the summary stands
    assert room.summaries_wanted() == []
    feed.set(C, A, B)
    room.refresh(w["id"])                                  # something new: written again, from the newest items
    assert room.summaries_wanted() == [w["id"]]


def test_following_something_else_drops_the_old_summary(room, feed):
    feed.set(A)
    w = room.add_widget("rss", {"url": feed.url})
    room.refresh(w["id"])
    room.write_summary(w["id"], lambda widget, items: nr.parse_summary("One story [1].", items))
    room.update_widget(w["id"], params={"url": feed.url + "?other"})
    assert room.widget(w["id"])["summary"] is None


def test_a_board_made_before_summaries_gets_the_columns(tmp_path):
    import sqlite3
    db = sqlite3.connect(tmp_path / "old.db")
    db.execute("CREATE TABLE widgets(id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, title TEXT, params TEXT, every INTEGER, "
               "alert TEXT, size TEXT DEFAULT 'm', pos REAL, paused INTEGER DEFAULT 0, created REAL, last_look REAL, "
               "last_ok REAL, error TEXT DEFAULT '', fails INTEGER DEFAULT 0, session TEXT DEFAULT '')")
    db.execute("INSERT INTO widgets(kind, title, params, every, alert, pos) VALUES ('rss', 'Old', '{}', 30, '{}', 1)")
    db.commit()
    db.close()
    w = nr.Newsroom(tmp_path / "old.db").widgets()[0]
    assert w["title"] == "Old" and w["summary"] is None and not w["summary_wanted"]


def test_summaries_wait_for_the_users_turn_and_a_failing_model_costs_only_the_summary(room, feed):
    feed.set(A, B)
    w = room.add_widget("rss", {"url": feed.url})
    room.refresh(w["id"])
    busy = {"turn": True}
    s = nr.Summarizer(room, lambda widget, items: nr.parse_summary("Two [1][2].", items), ready=lambda: not busy["turn"])
    assert s.once() == 0 and room.summaries_wanted() == [w["id"]]   # a turn runs: the model is the user's
    busy["turn"] = False
    assert s.once() == 1 and room.widget(w["id"])["summary"]["text"] == "Two [1][2]."

    feed.set(C, A, B)
    room.refresh(w["id"])

    def down(widget, items):
        raise nr.NewsroomError("the model is down")
    bad = nr.Summarizer(room, down)
    assert bad.once() == 0 and "down" in bad.errors[w["id"]]
    got = room.widget(w["id"])
    assert got["summary"]["text"] == "Two [1][2]." and not got["summary_wanted"]   # the old one stays; tried again next look
    assert len(room.items(w["id"])) == 3


# ---------------------------------------------------------------- the tile, in a real browser
pw = pytest.importorskip("playwright.sync_api")
pytest.importorskip("uvicorn")
from test_dashboard_ui import browse, serve  # noqa: E402,F401


@pytest.fixture
def dashboard(cfg, feed):
    feed.set(A, B)
    cfg.raw = {**cfg.raw, "newsroom": {**(cfg.raw.get("newsroom") or {}), "summary": True}}
    # the harness's model answers the summary request (no turn is run in this test)
    yield from serve(cfg, ["Nvidia beat earnings estimates [2], and chip export rules changed [1]."])


def test_the_tile_shows_the_summary_on_top_and_its_sources_at_the_bottom(dashboard, feed, browse):
    def flow(page):
        page.goto(dashboard)
        page.wait_for_selector(".hello")
        page.click('.rail [data-page="newsroom"]')
        page.click('#nracts [data-nract="add"]')
        page.click('#nrkinds [data-nrkind="rss"]')
        page.fill('#nrfields [name="p_url"]', feed.url)
        page.click("#nrok")
        page.wait_for_selector("#nr .nrw .nrsumm:not(.wait) p", timeout=20000)
        tile = page.locator("#nr .nrw").first
        assert "Nvidia beat earnings estimates" in tile.locator(".nrsumm p").inner_text()
        cites = tile.locator(".nrsumm a.cite")
        assert cites.count() == 2 and cites.first.get_attribute("rel") == "noopener noreferrer"
        assert cites.first.get_attribute("href") in ("https://ex.test/a", "https://ex.test/b")
        # the summary is above the items, the sources below them
        order = page.evaluate("""() => { const t = document.querySelector('#nr .nrw');
            const at = (s) => [...t.querySelectorAll('*')].indexOf(t.querySelector(s));
            return [at('.nrsumm'), at('.nrlist'), at('.nrsrc')]; }""")
        assert order[0] < order[1] < order[2]
        src = tile.locator(".nrsrc")
        assert "Sources (2)" in src.locator("summary").inner_text()
        src.locator("summary").click()
        links = src.locator("li a")
        assert links.count() == 2 and {links.nth(i).get_attribute("href") for i in range(2)} == {"https://ex.test/a",
                                                                                                "https://ex.test/b"}
    browse(flow)
