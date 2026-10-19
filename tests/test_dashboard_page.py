"""The dashboard page (jevh/web/static/index.html), checked without a browser: every theme it offers has
styles, every element the script looks up exists, and the background stays free of a grid."""
import re
from pathlib import Path

PAGE = (Path(__file__).resolve().parent.parent / "jevh" / "web" / "static" / "index.html").read_text(encoding="utf-8")
CSS = PAGE[PAGE.index("<style>"):PAGE.index("</style>")]
SCRIPT = PAGE[PAGE.index("<script>", PAGE.index("</style>")):]
THEME_IDS = re.findall(r'\{id: "(\w+)",\s+name: "[^"]+",\s+note:', PAGE)


def test_every_theme_in_the_picker_has_styles():
    assert len(THEME_IDS) >= 6 and len(set(THEME_IDS)) == len(THEME_IDS)
    for t in THEME_IDS:
        assert f':root[data-theme="{t}"]' in CSS, f"theme {t} is offered but has no CSS block"


def test_every_theme_fills_the_same_slots():
    slots = ["--bg:", "--surface:", "--surface2:", "--text:", "--muted:", "--line:", "--accent:", "--on-accent:", "--grad:",
             "--user-bg:", "--shadow:"]
    for t in THEME_IDS:
        # a block per theme (aurora has one per scheme): the slots must all be set somewhere in the theme's blocks
        blocks = re.findall(r':root\[data-theme="%s"\][^{]*\{(.*?)\n\}' % t, CSS, re.S)
        body = "\n".join(blocks)
        for s in slots:
            assert s in body, f"theme {t} does not set {s}"


def test_no_grid_background():
    assert "--grid" not in PAGE
    line = r"(?:[^()]|\([^()]*\))*"   # a gradient's arguments, allowing var(...) inside
    assert not re.search(rf"linear-gradient\({line}1px{line}\)\s*0 0\s*/\s*\d+px", CSS), "a repeating line grid"


def test_script_only_looks_up_elements_that_exist():
    ids = set(re.findall(r'\bid="([\w-]+)"', PAGE.split("<script>")[0] + PAGE))   # markup, plus ids built by the script
    wanted = set(re.findall(r'\$\("([\w-]+)"\)', SCRIPT))
    missing = {w for w in wanted if w not in ids}
    assert not missing, f"the script looks up ids that are not in the page: {sorted(missing)}"


def test_every_icon_the_script_uses_is_defined():
    defined = set(re.findall(r'<symbol id="i-([\w-]+)"', PAGE))
    used = set(re.findall(r'href="#i-([\w-]+)"', PAGE)) | set(re.findall(r'\bico\("([\w-]+)"\)', SCRIPT))
    assert len(used) > 15, "the icon scan found almost nothing: the patterns need updating"
    missing = {u for u in used if u not in defined}
    assert not missing, f"icons used but not defined: {sorted(missing)}"


def test_tool_calls_are_summarised_not_dumped():
    """The chat says what was done in one plain sentence; the code is behind a click (and a setting)."""
    for part in ("parseOps", "const CATS", 'activity: "summary"', '"verbatim"', '"hidden"'):
        assert part in PAGE, part
    assert "wchip" not in PAGE and "livelabel" not in PAGE, "the chip/shimmer styling should be gone"


def test_thinking_is_prose_not_a_terminal_log():
    css = re.search(r"\.answer\.thought \{[^}]*\}", CSS).group(0)
    assert "font-family" not in css and "mono" not in css, "reasoning should use the reading font"
    assert "thoughtOf" in SCRIPT and "md(text)" in SCRIPT


def test_the_page_is_made_for_phones():
    assert "viewport-fit=cover" in PAGE                                      # so the safe-area insets (notch, home bar) apply
    assert "@media (max-width: 900px)" in CSS and "@media (pointer: coarse)" in CSS
    assert "env(safe-area-inset-bottom" in CSS
    assert "font-size: 16px !important" in CSS                               # smaller fields make iOS zoom the page on tap
    assert re.search(r'\.app\[data-panel="open"\] \.side \{[^}]*background: var\(--bg\)', CSS)   # the panel covers the chat: opaque


def test_the_panel_starts_closed():
    assert '<div class="app" id="app" data-panel="closed" data-view="chat">' in PAGE
    assert "$(\"side\").inert = true" in PAGE


def test_the_newsroom_page_is_wired_and_its_icons_exist():
    for part in ('data-page="newsroom"', 'id="page-newsroom"', 'id="nrdlg"', 'id="toasts"', "/api/newsroom", 'e.kind === "newsroom_alert"'):
        assert part in PAGE, part
    icons = set(re.findall(r'<symbol id="i-([\w-]+)"', PAGE))
    mapped = set(re.findall(r'(?:^|[{,])\s*(?:\w+): "(\w+)"', re.search(r"const NR_ICON = \{(.*?)\};", SCRIPT, re.S).group(1)))
    assert mapped and mapped <= icons, f"newsroom kind icons with no symbol: {sorted(mapped - icons)}"
    from jevh import newsroom
    wanted = {k.icon for k in newsroom.KINDS.values()}
    assert wanted <= set(re.findall(r"(\w+): \"\w+\"", re.search(r"const NR_ICON = \{(.*?)\};", SCRIPT, re.S).group(1))), \
        "a kind uses an icon name the page has no mapping for"
