"""Tools (project, web parsing, Jev choice rounds, MCP) and how the orchestrator wires them."""
import sys
from pathlib import Path

import pytest

from jevh.jev import Jev
from jevh.mcp import McpError, McpHub
from jevh.orchestrator import Harness
from jevh.repl.client import Repl
from jevh.tools import TOOLS, _pick, web

from conftest import FakeSystemOne, ScriptedLLM, choice_ans, noul_ans, py

FAKE_MCP = {"fake": {"command": sys.executable, "args": [str(Path(__file__).with_name("fake_mcp_server.py"))],
                     "timeout": 20}}


def test_choose_runs_rounds_over_the_cap():
    fake = FakeSystemOne(lambda k, q, s: choice_ans("o37", 0.9, q["criteria"]) if "o37" in q.get("criteria", {})
                         else None)
    opts = {f"o{i}": f"option {i}" for i in range(40)}
    probs, _ = _pick.choose(fake, {"q": "x"}, "pick", opts, cap=16)
    assert max(probs, key=probs.get) == "o37"
    assert set(probs) == set(opts)
    assert all(len(q["_pick" if "_pick" in q else next(iter(q))]["criteria"]) <= 16
               for _, q in fake.calls for q in [q])
    assert len(fake.calls) == 2  # all rounds in one request, then the final


class _Reply:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self.text.encode("utf-8")


CHALLENGE_PAGE = ("<html><head><title>DuckDuckGo</title></head><body><div class='anomaly-modal__title'>Unfortunately, bots use "
                  "DuckDuckGo too.</div>Please complete the following challenge to confirm this search was made by a human. "
                  "Select all squares containing a duck.</body></html>")
RESULT_PAGE = ('<div class="result results_links results_links_deep"><a class="result__a" href="https://ex.test/a">First</a>'
               '<a class="result__snippet">a snippet</a></div>')


def test_a_search_engine_asking_for_a_human_is_an_error_not_an_empty_search(monkeypatch):
    """DuckDuckGo answered this machine with a CAPTCHA page for hours: it was read as a search with no results, so every
    worker's web search came back empty and nothing said why. A search that really has no results is still empty."""
    page = {"html": CHALLENGE_PAGE}
    monkeypatch.setattr(web.urllib.request, "urlopen", lambda *a, **k: _Reply(page["html"]))
    with pytest.raises(web.SearchBlocked, match="human check"):
        web.search("llama.cpp release notes")
    out = web.ask("what is new in llama.cpp?")                   # keeps its shape, and says why it is empty
    assert out["passages"] == [] and out["results"] == [] and "human check" in out["note"]
    page["html"] = "<html><body>No  results.</body></html>"
    assert web.search("zzzzqqqq nothing") == []
    assert web.ask("zzzzqqqq nothing")["note"] == "no search results"
    page["html"] = RESULT_PAGE
    assert web.search("anything") == [{"title": "First", "url": "https://ex.test/a", "snippet": "a snippet"}]


def test_the_newsroom_guide_adds_an_x_tile_only_when_asked():
    """The guide's example added a company's X tile every time: asked to keep an eye on Nvidia news, the worker also made an
    X tile (the bench's one miss in nvidia_earnings), and told the user so."""
    guide = TOOLS["newsroom"].guide
    assert 'newsroom.company("Nvidia", alert="earnings")' in guide and "only when the user asked" in guide
    assert 'company("Nvidia", x="nvidia"' not in guide


def test_web_html_to_text_skips_chrome():
    p = web._Text()
    p.feed("<html><head><title>T</title><style>x{}</style></head><body><nav>menu</nav>"
           "<h1>Hello</h1><p>First <b>para</b>.</p><script>evil()</script><footer>foot</footer></body></html>")
    text = "".join(p.out)
    assert "Hello" in text and "First para." in text
    assert "menu" not in text and "evil" not in text and "foot" not in text and p.title == "T"


def test_web_html_to_text_keeps_a_formula_once():
    """Live (ar5iv): MathML's TeX annotation doubled every number - "28.4" read as "28.428.4"."""
    p = web._Text()
    p.feed('<p>a new state-of-the-art BLEU score of <math alttext="28.4"><semantics><mn>28.4</mn>'
           '<annotation encoding="application/x-tex">28.4</annotation></semantics></math>. Training took '
           '<math><semantics><mn>3.5</mn><annotation-xml encoding="MathML-Content"><cn>3.5</cn></annotation-xml>'
           '</semantics></math> days.</p>')
    text = "".join(p.out)
    assert "score of 28.4." in text and "took 3.5 days" in text and "28.428.4" not in text


def test_project_tool_edits_inside_the_folder_and_undoes(tmp_path):
    proj = tmp_path / "proj"
    (proj / "src").mkdir(parents=True)
    (proj / "src" / "app.py").write_text("def hello():\n    return 'hi'\n", encoding="utf-8")
    with Repl(tmp_path / "wd", tmp_path / "arts", tools={"project": "jevh.tools.project"},
              env={"JEVH_PROJECT": str(proj)}) as r:
        out = r.run("print(project.grep(r'def \\w+', '**/*.py'))\n"
                    "print(project.edit('src/app.py', \"'hi'\", \"'hello'\"))\n"
                    "print(project.read('src/app.py'))\n"
                    "print(project.sh('python -c \"print(40+2)\"')['out'])")
        assert "src/app.py:1: def hello():" in out.output
        assert "return 'hello'" in out.output and "42" in out.output
        bad = r.run("project.write('../escape.txt', 'x')")
        assert bad.error and "outside the project" in bad.output
        r.run("project.undo()")
    assert (proj / "src" / "app.py").read_text(encoding="utf-8") == "def hello():\n    return 'hi'\n"
    assert not (tmp_path / "escape.txt").exists()


def test_project_undo_restores_the_original_whatever_way_the_path_was_written(tmp_path):
    """Backups were keyed by the path as written: editing "./src/app.py" after "src/app.py" copied the edited file over
    the original's backup (undo then restored the edit), and an absolute path was copied onto itself and failed."""
    proj = tmp_path / "proj"
    (proj / "src").mkdir(parents=True)
    app = proj / "src" / "app.py"
    app.write_text("v0\n", encoding="utf-8")
    with Repl(tmp_path / "wd", tmp_path / "arts", tools={"project": "jevh.tools.project"},
              env={"JEVH_PROJECT": str(proj)}) as r:
        out = r.run("print(project.edit('src/app.py', 'v0', 'v1'))\n"
                    "print(project.edit('./src/app.py', 'v1', 'v2'))\n"
                    f"print(project.edit({str(app)!r}, 'v2', 'v3'))\n"
                    "print(project.changed())")
        assert not out.error, out.output
        assert "['src/app.py']" in out.output
        assert r.run(f"project.undo({str(app)!r})").output.strip() == "['src/app.py']"
    assert app.read_text(encoding="utf-8") == "v0\n"


def test_mcp_hub_lists_and_calls_tools():
    hub = McpHub(FAKE_MCP)
    try:
        cat = hub.catalog()
        assert set(cat) == {"fake.echo", "fake.add", "fake.boom"}
        assert cat["fake.add"].signature() == 'mcp("fake.add", a: number, b: number)'
        assert hub.call("fake.echo", {"text": "hi"}) == "hi"
        assert hub.call("fake.add", {"a": 2, "b": 3}) == "5"
        with pytest.raises(McpError, match="it broke"):
            hub.call("fake.boom", {})
    finally:
        hub.close()


def test_an_mcp_server_logging_odd_lines_to_stdout_keeps_working():
    """A log line that was JSON but not an object (a bare number), or not UTF-8, ended the reader thread: every call
    after it waited its whole timeout. And a call that got no answer raised a bare TimeoutError, not McpError."""
    noisy = {"fake": {**FAKE_MCP["fake"], "env": {"FAKE_MCP_NOISE": "1"}, "timeout": 3}}
    hub = McpHub(noisy)
    try:
        assert hub.call("fake.echo", {"text": "one"}) == "one"
        assert hub.call("fake.add", {"a": 2, "b": 3}) == "5"      # the reader survived the noise
        with pytest.raises(McpError, match="no answer within"):
            hub.call("fake.hang", {})
        assert hub.call("fake.echo", {"text": "after"}) == "after"
    finally:
        hub.close()


def test_broken_mcp_server_does_not_break_the_catalog():
    hub = McpHub({"nope": {"command": "definitely-not-a-real-binary-xyz"}, **FAKE_MCP})
    try:
        assert set(hub.catalog()) == {"fake.echo", "fake.add", "fake.boom"}
        assert "nope" in hub.errors
    finally:
        hub.close()


def tool_rules(tools_yes=(), mcp_pick=None):
    def rules(k, q, s):
        if k == "persona":
            return choice_ans("daily_assistant", 0.9, q["criteria"])
        if k.startswith("t_"):
            return noul_ans(0.95 if k[2:] in tools_yes else 0.02)
        if mcp_pick and q.get("type") == "choice" and mcp_pick in q.get("criteria", {}):
            return choice_ans(mcp_pick, 0.9, q["criteria"])
        if mcp_pick and k.startswith("m") and q.get("type") == "noul" and mcp_pick in str(q["instructions"]):
            return noul_ans(0.9)
    return rules


def test_only_picked_tool_guides_reach_the_prompt(cfg):
    llm = ScriptedLLM(["The weather is sunny."])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(tool_rules(tools_yes={"web"}))), llm_factory=lambda p: llm)
    h.learning = False
    h.turn("what's the weather in Seattle?", force="daily_assistant")
    system = llm.seen[0][0]["content"]
    assert "web.ask(question)" in system
    # (the browser guide's own text: a learned skill in the roster may mention browser.do too)
    assert "you state outcomes, Jev picks each click" not in system and "Also importable in your REPL (help(name) shows how): browser" in system


def test_worker_calls_an_mcp_tool_jev_picked(cfg):
    cfg.mcp_servers = FAKE_MCP
    llm = ScriptedLLM([py("final(mcp('fake.add', a=2, b=40))")])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(tool_rules(mcp_pick="fake.add"))), llm_factory=lambda p: llm)
    h.learning = False
    try:
        r = h.turn("add 2 and 40 with the calculator tool", force="daily_assistant")
        assert r.answer == "42"
        assert 'mcp("fake.add", a: number, b: number)' in llm.seen[0][0]["content"]
        assert "fake.echo" not in llm.seen[0][0]["content"]
    finally:
        h.mcp.close()


def test_project_is_visible_to_dev_only_when_open(cfg, tmp_path):
    (tmp_path / "repo").mkdir()
    llm = ScriptedLLM([py("final(str(ctx.get('project')) + '|' + str(project.root))")] * 2)
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(tool_rules(tools_yes={"project"}))), llm_factory=lambda p: llm)
    h.learning = False
    assert h.turn("what's in my repo?", force="high_end_dev").answer == "None|None"
    h.set_project(str(tmp_path / "repo"))
    r = h.turn("what's in my repo?", force="high_end_dev")
    assert r.answer == f"{tmp_path / 'repo'}|{tmp_path / 'repo'}"
    assert "project.find(question)" in llm.seen[1][0]["content"]


def test_pages_jev_ranks_are_bounded():
    passages = [f"filler paragraph number {i} about nothing" for i in range(600)]
    passages[437] = "To cancel a gym membership after moving, send written notice with your new address."
    kept = _pick.prefilter("cancel gym membership new address", passages)
    assert len(kept) == _pick.MAX_PASSAGES and passages[437] in kept


def test_binary_downloads_are_refused_and_pdfs_read(tmp_path, monkeypatch):
    import io
    import urllib.request

    class Resp(io.BytesIO):
        def __init__(self, body, ctype):
            super().__init__(body)
            self.headers = {"Content-Type": ctype}

        def geturl(self):
            return "http://x/"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    bodies = {"img": (b"\x89PNG\r\n\x1a\n\x00\x00\x00" + b"\x00" * 100, "image/png"),
              "zip": (b"PK\x03\x04" + b"\x00" * 50, "application/zip"),
              "pdf-nolib": (b"%PDF-1.7 not really a pdf", "application/octet-stream")}
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=0: Resp(*bodies[req.full_url.rsplit("/", 1)[-1]]))
    for name in ("img", "zip"):
        with pytest.raises(ValueError, match="not a text page"):
            web.fetch(f"http://x/{name}")
    with pytest.raises(Exception):  # a broken PDF must fail loudly, never turn into junk text
        web.fetch("http://x/pdf-nolib")


def test_native_tool_calls_are_translated():
    """Live 2026-09-28: ornith and Qwen both wrote <tool_call><function=web>... instead of a python
    block, three turns running; each was taken as a plain answer, bounced, and shipped."""
    from jevh.worker import translate_tool_calls
    tools = {n: f"jevh.tools.{n}" for n in ("web", "browser")}
    call = "<tool_call>\n<function=web>\n<parameter={k}>\n{v}\n</parameter>\n</function>\n</tool_call>"
    assert translate_tool_calls(call.format(k="search", v="porch and patio lindon"), tools) == \
        (["print(web.search('porch and patio lindon'))"], [])
    assert translate_tool_calls(call.format(k="question", v="Fairview price?"), tools) == \
        (["print(web.ask(question='Fairview price?'))"], [])
    assert translate_tool_calls("<function=weather><parameter=city>SLC</parameter></function>", tools) == \
        ([], ["weather"])


def test_native_final_call_ends_the_task_and_unknown_calls_get_a_nudge(cfg):
    llm = ScriptedLLM(["<tool_call><function=weather><parameter=city>SLC</parameter></function></tool_call>",
                       "<tool_call><function=final><parameter=answer>Sunny, 72F.</parameter></function></tool_call>"])
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(tool_rules(tools_yes={"web"}))), llm_factory=lambda p: llm)
    h.learning = False
    assert h.turn("weather in Salt Lake?", force="daily_assistant").answer == "Sunny, 72F."
    assert "<tool_call> text (weather) isn't run here" in llm.seen[1][-1]["content"]


# ---- the browser's bot-check stop is off by default (no Playwright needed: runs on CI too) ----

def test_bot_checks_are_off_by_default():
    """Shipped config: the switch is off. Loaded without the setting, the tool stops nowhere."""
    import os

    import yaml
    from jevh.config import ROOT
    from jevh.tools import browser
    assert yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["tools"]["browser_bot_check"] is False
    if not os.environ.get("JEVH_BROWSER_BOT_CHECK"):
        assert browser.BOT_CHECK is False


def test_bot_check_switch_defaults_off_and_only_an_explicit_on_turns_it_on():
    from jevh.tools import browser
    for on in ("1", "true", "on", "yes", "On", " TRUE "):
        assert browser._bot_check(on) is True, on
    for off in (None, "", "0", "false", "False", " off ", "no", "NO", "maybe", "2"):
        assert browser._bot_check(off) is False, off


def test_bot_check_exempt_is_everywhere_when_off_and_nowhere_when_on(monkeypatch):
    from jevh.tools import browser
    monkeypatch.setattr(browser, "BOT_CHECK", True)
    assert not browser.bot_check_exempt("https://www.mysite.dev/x") and not browser.bot_check_exempt("https://friend.com/")
    monkeypatch.setattr(browser, "BOT_CHECK", False)
    assert browser.bot_check_exempt("https://friend.com/") and browser.bot_check_exempt("https://anything.example/a?b=1")
