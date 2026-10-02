# The dashboard's look

`jevh dashboard` is one page, `jevh/web/static/index.html`: markup, styles and script together. This page covers
what you can change from the browser, and how to add a theme for everyone. (What each panel shows is in the
[README](../README.md#dashboard).)

## Appearance

The palette at the bottom of the left rail opens **Appearance**. Choices apply as you click, and are kept in
this browser (`localStorage`, key `jevh.ui`), not in `config.yaml`: they are about one screen, not the harness.
**Reset** puts everything back.

| Setting | Choices |
|---|---|
| Theme | Aurora (frosted glass, follows your system's light/dark), Sumi-e (ink and a vermilion seal), Paper (a typewritten ledger), Clay (warm daylight, serif answers), Obsidian (graphite and gold), Nocturne (indigo and violet), Phosphor (a green terminal) |
| Accent | the theme's own, eight presets, or any colour |
| Type | the theme's own, Sans, Serif or Mono; text size Small, Medium or Large |
| Layout | chat width Narrow, Medium or Wide; Cozy or Compact spacing; Sharp, Soft or Round corners (default: the theme's) |
| Tool calls in the chat | **Compact** (chips, one click for steps), **Steps** (steps listed), **Verbatim** (every step opens with its code and output), **Hidden** (only a live status line) |
| Open live thinking | show the model's reasoning while it streams, instead of a collapsed line |
| Open the panel for you | open the side panel when the browser goes live or a turn makes a file |
| Visual effects | glass blur, glow and paper grain; off is flatter and lighter on a weak GPU |

The panel's width is remembered when you drag its edge. The expand button next to the X makes it nearly full width.

## How tool calls become chips

The page never receives a "tool call" event: the worker's `code` event carries the Python it ran, and the
`output` event what it printed. `parseOps()` reads the code for the calls the REPL offers (`project.read`,
`web.search`, `browser.goto`, `request(...)`, `save_artifact(...)`, a shell command, a file write, ...) and gives
each a category, an icon, a past and a present tense, and a short detail (the path, the query, the host). A step
with none of them shows as "Ran code" with its first line. To teach it a new tool, add a row to `OPT` (and a
category to `CATS` if it needs its own chip) in the page's script.

## Adding a theme

A theme is one CSS block and one entry in the picker, so it is a small change.

1. In `index.html`, add a block that fills every colour slot (copy `Obsidian` and change the values):
   `:root[data-theme="mine"] { --bg: ...; --surface: ...; --surface2: ...; --surface3: ...; --text: ...; --muted: ...;
   --faint: ...; --line: ...; --line2: ...; --accent: ...; --accent2: ...; --on-accent: ...; --grad: ...; --user-bg: ...;
   --user-text: ...; --user-border: ...; --shadow: ...; }`. Optional slots: `--ambient` (background glows),
   `--grain` and `--grain-o` (paper noise), `--blur` (frosted glass), `--glow`, `--av-r` (avatar shape), and the
   three font slots `--font-body`, `--font-display`, `--font-answer`.
2. Add `{id: "mine", name: "Mine", note: "One line", scheme: "dark", corners: "soft", pv: [bg, surface, text, accent]}`
   to `THEMES` in the `<head>` script (`scheme` is `"light"`, `"dark"` or `"auto"`; `pv` colours the picker's preview).
3. For anything a theme can't do with colours alone (Paper's hard shadows, Phosphor's prompt character), add rules
   under `:root[data-theme="mine"]` in the "Theme details" section.

`tests/test_dashboard_page.py` fails if a theme is in the picker without a CSS block or leaves a slot empty.
Run `python -m pytest -q tests/test_dashboard_page.py tests/test_dashboard_ui.py`; the second drives the page in
Chromium (skipped when Playwright's Chromium isn't installed).

Nothing here is loaded from the internet: fonts are system stacks, so the dashboard looks the same offline.
