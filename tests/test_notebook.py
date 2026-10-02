"""The notebook: entries and their states, search, the `jevh notebook` command and the chat commands."""
import pytest

from jevh.cli import notebook_chat
from jevh.jev import Jev
from jevh.notebook import Notebook, main, title_of
from jevh.orchestrator import Harness

from conftest import FakeSystemOne, ScriptedLLM


def test_entries_are_added_listed_changed_and_deleted(tmp_path):
    nb = Notebook(tmp_path / "n.db")
    a = nb.add("# Show cost per turn\nin the header", "idea", session="s1")
    b = nb.add("The chart axis had no units", "problem")
    c = nb.add("remember to rotate the keys")
    assert (a["kind"], a["status"], a["pinned"], a["session"], a["title"]) == ("idea", "open", False, "s1",
                                                                              "Show cost per turn")
    assert [e["id"] for e in nb.list()] == [c["id"], b["id"], a["id"]]            # newest first
    nb.update(a["id"], pinned=True)
    assert [e["id"] for e in nb.list()][0] == a["id"]                              # pinned first
    assert [e["id"] for e in nb.list(kind="problem")] == [b["id"]]
    assert nb.update(b["id"], status="done")["status"] == "done"
    assert [e["id"] for e in nb.list(status="open")] == [a["id"], c["id"]]
    assert nb.counts() == {"open": 2, "done": 1, "all": 3, "idea": 1, "problem": 0, "note": 1}
    edited = nb.update(c["id"], text="rotate the API keys monthly", kind="idea")
    assert (edited["kind"], edited["text"]) == ("idea", "rotate the API keys monthly")
    assert nb.update(999, text="x") is None and nb.get(999) is None
    assert nb.delete(a["id"]) and not nb.delete(a["id"]) and nb.get(a["id"]) is None
    nb2 = Notebook(tmp_path / "n.db")                                              # it is on disk
    assert [e["id"] for e in nb2.list()] == [c["id"], b["id"]]


def test_bad_input_is_refused(tmp_path):
    nb = Notebook(tmp_path / "n.db")
    e = nb.add("one")
    for bad in (lambda: nb.add("  "), lambda: nb.add("x", "rant"), lambda: nb.add("x" * 20001),
                lambda: nb.update(e["id"], status="maybe"), lambda: nb.update(e["id"], text=""),
                lambda: nb.list(kind="rant"), lambda: nb.list(status="x")):
        with pytest.raises(ValueError):
            bad()
    assert nb.get(e["id"])["text"] == "one"


def test_search_needs_every_word_and_takes_percent_signs_literally(tmp_path):
    nb = Notebook(tmp_path / "n.db")
    nb.add("the chart had the wrong units")
    nb.add("units are fine but the table is 100% wrong")
    nb.add("snake_case names")
    assert len(nb.list(query="WRONG units")) == 2
    assert len(nb.list(query="chart units")) == 1
    assert [e["text"] for e in nb.list(query="100%")] == ["units are fine but the table is 100% wrong"]
    assert len(nb.list(query="%")) == 1 and len(nb.list(query="_")) == 1           # not wildcards
    assert nb.list(query="nothing like this") == []


def test_title_is_the_first_line_without_markdown_marks():
    assert title_of("\n\n## A heading\nbody") == "A heading"
    assert title_of("- a bullet") == "a bullet" and title_of("   ") == ""
    assert len(title_of("word " * 80)) <= 100


def test_the_command_line(cfg, capsys, monkeypatch):
    monkeypatch.setattr("jevh.config.load_config", lambda path=None: cfg)
    assert main([]) == 0 and "the notebook is empty" in capsys.readouterr().out
    assert main(["idea", "a", "second", "model", "as", "checker"]) == 0
    assert main(["problem", "the", "chart", "had", "no", "units"]) == 0
    assert main(["add", "--kind", "note", "buy", "milk"]) == 0
    assert main(["add", "   "]) == 1
    capsys.readouterr()
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "idea" in out and "the chart had no units" in out and "open: 1 ideas, 1 problems, 1 notes; 0 done" in out
    assert main(["--kind", "problem", "-q", "units"]) == 0
    out = capsys.readouterr().out
    assert "chart had no units" in out and "milk" not in out
    assert main(["done", "2"]) == 0 and main(["pin", "1"]) == 0 and main(["edit", "3", "buy", "oat", "milk"]) == 0
    capsys.readouterr()
    assert main(["--status", "done"]) == 0
    assert "chart had no units" in capsys.readouterr().out
    assert main(["show", "3"]) == 0 and "buy oat milk" in capsys.readouterr().out
    assert main(["reopen", "2"]) == 0 and main(["rm", "2"]) == 0
    assert main(["show", "2"]) == 1 and "no entry #2" in capsys.readouterr().out


def harness(cfg):
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne()), llm_factory=lambda p: ScriptedLLM([]))
    h.learning = False
    return h


def said(capsys) -> str:
    """What was printed, with the line wrapping undone: the chat commands print through rich, which wraps at the
    terminal's width (80 columns under pytest on Windows), so "cookie banners" can end up across a line break."""
    return " ".join(capsys.readouterr().out.split())


def test_chat_commands_write_and_manage_entries(cfg, capsys):
    h = harness(cfg)
    notebook_chat(h, "idea", "cache the router decisions")
    notebook_chat(h, "problem", "the browser tool hangs on cookie banners")
    notebook_chat(h, "jot", "ask about the GPU")
    notebook_chat(h, "jot", "")
    e = h.notebook.list(kind="idea")[0]
    assert e["session"] == h.session and e["text"] == "cache the router decisions"
    assert "usage: /jot" in said(capsys)
    notebook_chat(h, "notebook", "")
    out = said(capsys)
    assert "cache the router decisions" in out and "cookie banners" in out and "open: 1 ideas, 1 problems, 1 notes" in out
    notebook_chat(h, "notebook", "problems")
    out = said(capsys)
    assert "cookie banners" in out and "router decisions" not in out
    notebook_chat(h, "notebook", "GPU")                                            # any other words search
    assert "ask about the GPU" in said(capsys)
    notebook_chat(h, "notebook", f"done {e['id']}")
    notebook_chat(h, "notebook", "done")
    assert "router decisions" in said(capsys)
    notebook_chat(h, "notebook", f"pin {e['id']}")
    notebook_chat(h, "notebook", f"show {e['id']}")
    assert "cache the router decisions" in said(capsys)
    notebook_chat(h, "notebook", f"rm {e['id']}")
    notebook_chat(h, "notebook", "show 999")
    notebook_chat(h, "notebook", "rm nope")
    out = said(capsys)
    assert "no entry #999" in out and "usage: /notebook rm <id>" in out and h.notebook.get(e["id"]) is None
