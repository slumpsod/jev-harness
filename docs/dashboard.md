# The dashboard's look

`jevh dashboard` is one page, `jevh/web/static/index.html`: markup, styles and script together. This page covers
how it is laid out, what you can change from the browser, and how to add a theme for everyone. (What each panel shows is in
the [README](../README.md#dashboard).)

## Panels and pages

The left rail (a bar along the bottom on a phone) opens two kinds of view:

- **Panels** (Inspector, Artifacts, Memory, Browser, History, Review, Notebook) open beside the chat, which stays in view.
  The same tab again, Esc or the X closes one.
- **Pages** (Newsroom, below the rail's divider, and Settings, the sliders icon above Appearance) take the place of the chat
  and its panel, so they can use the whole width: the Newsroom's tiles fill as many columns as fit (a Large tile takes two),
  and Settings has its sections in a column of their own. **Chat** at the top of a page, Esc (outside a text field) or the
  page's rail button again goes back to the chat as you left it, scrolled where you were, with the panel that was open. A
  panel's tab also leaves the page. The chat is only hidden meanwhile: a running turn carries on.

A turn that adds a Newsroom widget opens the Newsroom page with the new tile marked, as a turn that makes a file opens the
Artifacts panel (switch both off with *Open panels and pages for you* in Appearance). In the page's script, `openPanel` and
`closePanel` drive the panels and `showPage` and `showChat` the pages (`S.tab` is the open panel, `S.view` is `"chat"` or
the page shown).

## Sending files

The paperclip at the left of the message box attaches photos, screenshots, PDFs and other files (images, PDFs, text, CSV
and Office files; up to 25 MB each, 10 with a message). Pasting a screenshot into the box or dropping files on the chat
does the same. Each file uploads at once (`POST /api/upload?name=...`, the body is the file, behind the same token,
origin and sign-in checks as every other `/api/` call) into the session's folder, `workspace/sessions/<id>/uploads/`;
the tray above the text shows them, and Send waits until they are up. The message then goes with their paths
(`/api/chat` or `/api/answer` take `files: [...]`, which must be this session's uploads), and the request the harness runs
is your text followed by a note naming each file (`jevh/web/uploads.py`): routing, Jev's tool pick, the worker and the
session's memory all see that a file came with it. The worker opens it from there, and looks at a picture or a scanned
PDF with `look_at(path, question)` (its own model, when that model takes images). In the chat a picture shows as a
thumbnail and anything else as a chip; the page reads the note back out of the message. Files sent while a task runs are
added to that task (never answered aside as a question about its progress); `/queue` with files queues them with the
request. On a phone, the button opens the photo library or the camera. The orb's small chat has no attach button.

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
| Tool calls in the chat | **Compact** (one sentence, one click for steps), **Steps** (steps listed), **Verbatim** (every step opens with its code and output), **Hidden** (only a live status line) |
| Open live thinking | show the model's reasoning, as plain text, while it streams, instead of a folded "Thinking" line with its last sentence |
| Open panels and pages for you | open the Browser panel when the browser goes live, Artifacts when a turn makes a file, and the Newsroom page when it adds a widget |
| Visual effects | glass blur, glow and paper grain; off is flatter and lighter on a weak GPU |

The panel's width is remembered when you drag its edge. The expand button next to the X makes it nearly full width.

## Settings

The sliders icon at the bottom of the rail opens the **Settings** page. Its sections (a list on the left; a row of buttons at
the top on a phone; the arrow keys move between them) are:

| Section | What is in it |
|---|---|
| Models and search | the choices below |
| Devices | your other computers the assistant can use (pair one with a code, rename it, remove it), and signing in: accounts, invite links, passwords. In the desktop app on another computer, a "This computer" card. See [devices.md](devices.md). |
| Status | each model server (LM Studio, a llama.cpp server), the Jev adapter and backend, ComfyUI, the GPUs, add-ons that were skipped, and LM Studio's model swaps. The three dots at the top of the chat show the same at a glance and open this section. |
| Updates | the harness version on disk, the update channel, new commits on GitHub, **Update now** and **Undo update**. A count on the Settings icon (and on this section) says how many commits are new. |

Settings opens on the section it was last on. **Models and search** holds the few choices that are easy to get wrong in YAML.
Appearance is about this screen and stays in your browser; these belong to the harness, so they are saved on this machine and
apply to every browser, the phone page and the command line.

| Setting | What it does |
|---|---|
| Model | The model every assistant runs on (`models.strong`). "Each assistant's own" leaves each one on the model in its persona file. The list is what your local servers say they serve (LM Studio's models, loaded or not; the model a llama.cpp server has loaded); "Other" takes any `provider:model`, such as `anthropic:claude-opus-5-5`. |
| Quick model | Where Jev sends easy tasks (`models.fast`). Off: every task uses the main model. |
| Chat glass | How see-through the desktop app's orb chat is (`appearance.orb_glass`, 0 to 100). Unlike the rest, it is saved as soon as you let go of the slider, even while a turn runs, and every signed-in app's orb picks it up. The strip under it previews the look. |
| Web search | The engine behind `web.search` and `web.ask` (workers) and the Newsroom's web widgets. **DuckDuckGo** needs nothing (it reads a web page, which can ask for a human check). **SearXNG** takes your server's address; its `settings.yml` must allow the `json` format. **Brave Search** takes an API key (api.search.brave.com has a free plan). **Test search** runs one real search with what is on the page, before you save. |

Save sends only what you changed, and refuses a choice that cannot work in plain words (an unknown provider, a missing address
or key). It is refused while a turn is running: the turn keeps the model it has. What you save applies from your next message;
the first one may wait while a model loads. **Reset** forgets everything saved here, so the config files decide again.

They are saved in `workspace/settings.yaml`, which is read last: a choice made here wins over `config.yaml`,
`~/.jevh/config.yaml` and `config.local.yaml`, and those files (and their comments) are never rewritten. Only `models`,
`search` and `appearance` are read from it. It can hold a Brave key, so it is created readable by your user account only, and the page is never
sent the key back (it says "saved"). A dashboard started with `--model` ignores the two model choices, and says so.

## How tool calls become a sentence

The page never receives a "tool call" event: the worker's `code` event carries the Python it ran, and the
`output` event what it printed. `parseOps()` reads the code for the calls the REPL offers (`project.read`,
`web.search`, `browser.goto`, `request(...)`, `save_artifact(...)`, a shell command, a file write, ...) and gives
each a category, a past and a present tense, and a short detail (the path, the query, the host, a command's first
words). A step with none of them shows as "Ran some code". The turn's sentence counts the categories ("looked at 3
files, edited 2 files"). To teach it a new tool, add a row to `OPT` (and a phrase to `CATS` if it needs its own
words) in the page's script.

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
