"""Jev-driven desktop, pre-imported as `desktop` for personas that list it in `tools` (Windows).

The worker says what outcome it wants in one app, and Jev picks the window, every control, every value and checks the
result; the worker's model never reads a UI tree. It is the `browser` tool's design moved to native and Electron apps
(UI Automation instead of Playwright):

    desktop.windows()                                   # what is open: app, title, in front?, browser tabs
    desktop.where("the claude code tab")                # Jev picks the window (and browser tab) that matches
    desktop.open("notepad")                             # start an app, wait for its window
    r = desktop.do("press the New button", window="Claude")        # -> {"status": "done", ...}
    r = desktop.do("type the note into the text area", window="Notepad", values={"note": "hello"})
    desktop.read("which session is open?", window="Claude")        # only the passages that answer it
    desktop.key("ctrl+n", window="Claude")              # a shortcut, sent only once that window is in front
    desktop.elements("send"); desktop.click(12); desktop.fill(3, "text", submit=True)   # when Jev is stuck
    desktop.look("what does the chart show?")           # your own model looks at a screenshot

do() statuses: done, likely_done, needs_confirmation (send/delete/pay/close-without-saving is next: repeat with
allow_irreversible=True only if the user asked for exactly that), needs_value, needs_user (the user did not allow a password to be typed, an
administrator prompt or a window that will not come to the front: the user does that part), ambiguous, error, stuck,
no_window, max_actions.

What the structure gives over writing UI Automation by hand (the session this came from spent 37 steps listing windows):
one cached query reads a whole window in ~50 ms; the action goes through the control's own pattern first (no focus
stolen) and by a real mouse click only when that did nothing; text goes in as real key events, or by the clipboard when
it is long or has line breaks, and is read back; Enter is checked (an autocomplete that takes the Enter is closed and the
text sent again); keys are only sent to a window that is verifiably in front; dialogs and menus of the app are part of
what Jev sees, and a modal dialog hides the window behind it. A password is typed only after the tool itself asks the user and they
allow it (the worker cannot answer for them), and it is never shown, kept or read back.

Environment: JEVH_TOOLS_JEV=local|cloud (see _pick.py), JEVH_ARTIFACTS for screenshots.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import time
from pathlib import Path

from jevh.decide import noul
from jevh.tools._pick import choose, client_from_env, pick_passages, prefilter
from jevh.tools.browser import QUOTED, SECRET, SET_TO, SUBMIT_WORDS, _same_text, _words, click_label, field_matches

ARTIFACTS = Path(os.environ.get("JEVH_ARTIFACTS", "artifacts"))
TEXT_CHARS = 2000           # window text per decision; semif's latency grows with state size
MAX_ELEMENTS = 250
WANTED_MIN = float(os.environ.get("JEVH_DESKTOP_WANTED", "0.4"))

# ------------------------------------------------------------------ what a control is
KINDS = {50000: "button", 50002: "checkbox", 50003: "dropdown", 50005: "link", 50007: "list item", 50011: "menu item",
         50013: "radio", 50015: "slider", 50016: "spinner", 50019: "tab", 50024: "tree item", 50029: "list item",
         50031: "button"}
CONTAINERS = {50008, 50009, 50010, 50018, 50021, 50023, 50026, 50028, 50033, 50036}   # groups that name a region
GENERIC = {50006, 50020, 50025, 50026, 50033}   # an image/text/group/pane with an action is a button
FIELDS = {"text field", "text area", "search box", "password"}
TOGGLES = {"checkbox", "radio", "dropdown"}
PSEUDO = {"scroll": "scroll down in the window to reveal more controls"}
PATTERN_FIRST = {"tab", "list item", "radio", "checkbox", "tree item", "dropdown"}   # acted on through UIA, not the mouse
NAV_KINDS = ("tab", "menu item", "tree item", "list item", "link", "button")
DONE_AFTER_ACTING = 0.7   # Jev's "fully achieved" after at least one action
BROWSERS = {"chrome", "msedge", "brave", "firefox", "vivaldi", "opera", "arc"}
HELPERS = {"TextInputHost", "cua-driver", "NVIDIA Overlay", "SearchHost", "ShellExperienceHost", "StartMenuExperienceHost"}
POPUP_CLASSES = ("#32768", "ComboLBox", "Chrome_WidgetWin", "Xaml_WindowedPopupClass", "DropDown")

# a control whose name says what clicking it does: it needs a yes from the user first (unless they asked for it)
IRREVERSIBLE = re.compile(r"\b(send|post|publish|delete|remove|erase|wipe|format|uninstall|empty (the )?recycle|shut ?down|"
                          r"restart|sign out|log ?out|don'?t save|discard|overwrite|replace all|purchase|buy|pay|transfer|"
                          r"donate|subscribe|submit order|place order|close all|quit)\b", re.I)
RISKY_KEYS = {"alt+f4", "ctrl+w", "ctrl+q", "ctrl+shift+delete", "shift+delete", "win+l", "ctrl+alt+delete"}
KEY_NAMES = r"enter|return|escape|esc|tab|backspace|delete|space|f\d{1,2}"
KEY_GOAL = re.compile(r"^\s*(?:press|hit|use)\s+(?:the\s+)?(?:keys?\s+|keyboard shortcut\s+|shortcut\s+)?"
                      r"((?:(?:ctrl|control|shift|alt|win|cmd)\s*\+\s*)+[\w\[\]/,.;'\`=-]+|"
                      r"(?:" + KEY_NAMES + r")(?!\s*(?:button|link|menu|item|tab\b)))\s*(?:key\b)?", re.I)
OPEN_GOAL = re.compile(r"^\s*(?:open|launch|start|run)\s+(?:the\s+)?(?:windows\s+)?(.{2,40}?)(?:\s+(?:app|application|program))?\s*\.?$", re.I)
KNOWN_APPS = {"notepad": "notepad", "calculator": "calc", "calc": "calc", "paint": "mspaint", "file explorer": "explorer",
              "explorer": "explorer", "settings": "ms-settings:", "task manager": "taskmgr", "command prompt": "cmd",
              "cmd": "cmd", "powershell": "powershell", "wordpad": "write", "snipping tool": "snippingtool",
              "character map": "charmap", "control panel": "control"}
DOUBLE_GOAL = re.compile(r"\bdouble[- ]?click", re.I)   # opens a file or folder in a list
ENTER_GOAL = re.compile(r"\b(send|search|press enter|hit enter|and enter|then enter)\b", re.I)   # goals that end a text with Enter
CHROME_GOAL = re.compile(r"\b(close|minimi[sz]e|maximi[sz]e|restore|exit|quit|system menu)\b", re.I)   # goals about the window itself
SCROLL_BUTTON = r"^(Line|Page|Scroll)\s+(up|down|left|right)\b"   # a scroll bar's own buttons
WINDOW_BUTTONS = {"Minimize", "Maximize", "Restore", "Restore Down", "Close", "System"}
NAV_GOAL = re.compile(r"^\s*(?:(?:go|navigate|switch|jump)\s+to|open|show|select|click(?:\s+on)?)\s+(?:the\s+)?['\"“]?(.{2,40}?)['\"”]?"
                      r"(?:\s+(?:page|tab|section|screen|panel|view))?\s*\.?$", re.I)


REMOVE_GOAL = re.compile(r"^\s*(?:please\s+)?(?:remove|delete|close|dismiss|clear|get rid of)\b", re.I)
SET_CLAUSE = re.compile(r"\b(?:set|change|enter|type|put|write)\s+(?:the\s+|my\s+)?([A-Za-z][\w ]{1,28}?)\s+(?:to|as|=)\s+"
                        r"(\"[^\"]{1,80}\"|'[^']{1,80}'|\S+?)(?=\s*(?:,|;|\.|$|\s+and\b|\s+then\b))", re.I)


def goal_values(goal: str) -> dict[str, str]:
    """'set Display name to Ana, turn on Dark mode' -> {'Display name': 'Ana'}: values written into the goal itself."""
    out = {}
    for m in SET_CLAUSE.finditer(goal or ""):
        key, val = m.group(1).strip(), m.group(2).strip().strip("\"'")
        if val and not re.match(r"(?:on|off|true|false|the|a|an)$", val, re.I):
            out.setdefault(key, val)
    return out


def named_gone(goal: str, before: set[str], obs: dict) -> bool:
    """A remove/delete/close goal that names something the window showed at the start ("remove Charlie from the list"):
    it is done once that is no longer there, whatever Jev makes of the rest."""
    if not REMOVE_GOAL.match(goal or ""):
        return False
    names = [n for n in before if len(n) >= 3 and re.search(r"(?<![\w])" + re.escape(n) + r"(?![\w])", goal, re.I)]
    if not names:
        return False
    now = {e["label"].lower() for e in obs["elements"]} | {ln.lower() for ln in obs["text"].split("\n")}
    return any(n not in now for n in names)


FINAL_CLICK = re.compile(r"(?:,|\band\b|\bthen\b)\s*(?:then\s+)?(?:press|click|hit|tap|choose|select)\s+(?:on\s+)?(?:the\s+)?['\"“]?"
                         r"([A-Za-z][\w .-]{0,24}?)['\"”]?(?:\s+button)?\s*\.?\s*$", re.I)


def final_click(goal: str) -> str | None:
    """'..., turn on Dark mode and press OK' -> 'OK': the control that ends the goal. Until it has been clicked the goal
    is not done, whatever the window looks like (live: Jev called it done with the dialog still open and OK untouched)."""
    m = FINAL_CLICK.search(goal or "")
    return m.group(1).strip() if m and len(m.group(1).split()) <= 3 else None


def _label_is(label: str, name: str) -> bool:
    """A control's label against the name a goal gave it ('Export...' is 'click Export'; 'Save' is 'click the Save button')."""
    norm = lambda x: re.sub(r"\s+", " ", x or "").strip().lower().rstrip(". " + chr(0x2026))
    return bool(norm(name)) and norm(label) == norm(name)


def nav_target(goal: str) -> str | None:
    """'open the List tab', 'switch to Settings' -> the label that goal names (a tab, a menu entry, a sidebar item)."""
    m = NAV_GOAL.match(goal or "")
    return m.group(1).strip() if m and len(m.group(1).split()) <= 4 else None


Q_WINDOW = "Which window should `task.goal` be carried out in?"
Q_DONE = ("Has `task.goal` been fully achieved in `window`, including every part it asks for (such as saving, "
          "sending or opening a particular screen)? Judge from `window` and `task.history`.")
Q_ERROR = "Does `window` show an error or rejection message or dialog caused by the actions in `task.history`?"
Q_BLOCKED = ("Is there something in `window` that stops progress on `task.goal` and cannot be handled by clicking or "
             "typing (an administrator-permission prompt, a licence or sign-in wall, or a window that is not responding)?")
Q_LOGIN = "Is `window` a sign-in or sign-up screen, or asking the user to log in, before `task.goal` can continue?"
Q_IRREV = ("Would clicking `element`, by itself and with no further confirmation step, send a message, delete or "
           "overwrite something, pay, publish, or close something with unsaved work?")
Q_PENDING = "Does `controls` show a setting that still differs from what `goal` asks for?"
Q_WANTED = "Does `goal` ask for what clicking `element` would do?"
Q_TARGET = "Which control of `window` should the next action toward `task.goal` act on?"
Q_GROUP = "Which group of `window` controls contains the one the next action toward `task.goal` should act on?"
Q_VALUE = "Which of `values` should be typed into `field` to make progress on `goal`?"
Q_MATCH = "Which window or browser tab best matches `question`?"


def _backend(ui):
    if ui is not None:
        return ui
    from jevh.tools._winui import WinUI
    return WinUI()


def norm_text(s: str | None) -> str:
    return (s or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n").strip()


def label_of(r: dict) -> str:
    return re.sub(r"\s+", " ", r.get("name") or r.get("help") or "").strip()


def kind_of(r: dict) -> str | None:
    """The tool's word for a raw UIA element, or None when it is not something to act on."""
    ct, av = r["ct"], r["avail"]
    if ct == 50004:
        if r["pw"]:
            return "password"
        if r["readonly"] and not av["invoke"]:
            return None
        return "search box" if re.search(r"search", r["name"] or "", re.I) else "text field"
    if ct == 50030:   # a document: an editor when it takes text, the page of a browser or reader when it does not
        if (av["value"] and not r["readonly"]) or re.search(r"edit|scintilla|textbox", r["cls"] or "", re.I):
            return "text area"
        return None
    if ct in KINDS:
        return KINDS[ct]
    if ct in GENERIC and label_of(r) and (av["invoke"] or av["select"] or av["toggle"]):
        return "checkbox" if av["toggle"] else "button"
    return None


def region_word(rect, win) -> str:
    cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
    w, h = max(1, win[2] - win[0]), max(1, win[3] - win[1])
    col = "left" if cx < win[0] + w / 3 else "right" if cx > win[0] + 2 * w / 3 else "middle"
    row = "top" if cy < win[1] + h / 3 else "bottom" if cy > win[1] + 2 * h / 3 else "middle"
    return f"{row}-{col}" if (row, col) != ("middle", "middle") else "centre"


def build_elements(raws: list[dict], win_rect: tuple, layer: str = "") -> tuple[list[dict], list[str]]:
    """Raw UIA elements -> (the controls Jev chooses between, the plain text lines on screen)."""
    els: list[dict] = []
    texts: list[str] = []
    containers: list[tuple[str, tuple]] = []
    labels = [(label_of(r), r["rect"]) for r in raws if r["ct"] == 50020 and label_of(r) and not r["off"]]
    last_key = None
    combo = None   # a dropdown's own edit box and open button are parts of it, not controls
    for idx, r in enumerate(raws):
        if r["off"]:
            continue
        if combo and r["ct"] != 50003 and combo[0] <= r["rect"][0] and combo[1] <= r["rect"][1]                 and combo[2] >= r["rect"][2] and combo[3] >= r["rect"][3] and r["ct"] in (50004, 50000, 50020):
            continue
        rect = r["rect"]
        if rect[2] <= rect[0] or rect[3] <= rect[1] or rect[2] < win_rect[0] or rect[0] > win_rect[2] \
                or rect[3] < win_rect[1] or rect[1] > win_rect[3]:
            continue
        label = label_of(r)
        if r["ct"] in CONTAINERS and label:
            containers.append((label, rect))
        if r["ct"] == 50020 and label:
            if not texts or texts[-1] != label:
                texts.append(label)
            continue
        kind = kind_of(r)
        if kind is None:
            if r["ct"] in (50030, 50004) and r.get("value"):
                texts.append(f"[{label or 'text'}] {norm_text(r['value'])[:600]}")
            continue
        if not label:   # an unnamed field: the text beside or above it names it
            near = [(abs(rect[0] - lr[2]) + abs(rect[1] - lr[1]), name) for name, lr in labels
                    if (lr[2] <= rect[0] + 6 and lr[0] > rect[0] - 400 and min(lr[3], rect[3]) - max(lr[1], rect[1]) > 4)
                    or (lr[3] <= rect[1] + 6 and lr[1] > rect[1] - 50 and lr[0] < rect[2] and lr[2] > rect[0])]
            label = min(near)[1] if near else (r["aid"] if re.search(r"[a-zA-Z]{3}", r["aid"] or "") else "")
        if not label and kind not in FIELDS:
            continue
        key = (kind, label, tuple(round(v / 4) for v in rect))
        if key == last_key:
            continue   # (Chromium repeats a control as its own child)
        last_key = key
        e = {"i": len(els), "kind": kind, "label": label[:90], "rect": rect, "raw": r, "layer": layer,
             "disabled": not r["enabled"]}
        if kind in FIELDS | {"dropdown"}:
            e["value"] = norm_text(r.get("value"))[:80] if not r["pw"] else ("(hidden)" if r.get("value") else "")
        elif kind == "slider" or kind == "spinner":
            e["value"] = norm_text(r.get("value"))[:40]
        av = r["avail"]   # (a state the control does not have comes back as 0: only believe the patterns it has)
        if av["toggle"] and r["toggle"] is not None and kind not in FIELDS:
            e["checked"] = r["toggle"] == 1
        elif kind == "radio":
            e["checked"] = bool(r["selected"])
        if av["select"] and r["selected"] and kind in ("tab", "list item", "tree item", "menu item"):
            e["selected"] = True
        if av["expand"] and r["expand"] == 1:
            e["expanded"] = True
        inside = [(c, cr) for c, cr in containers if c != label and cr[0] <= rect[0] + 2 and cr[1] <= rect[1] + 2
                  and cr[2] >= rect[2] - 2 and cr[3] >= rect[3] - 2 and (cr[2] - cr[0]) * (cr[3] - cr[1]) > 1]
        if inside:
            e["context"] = min(inside, key=lambda c: (c[1][2] - c[1][0]) * (c[1][3] - c[1][1]))[0][:40]
        e["nav"] = False
        if (kind == "button" and re.match(SCROLL_BUTTON, label)) or (label in WINDOW_BUTTONS and rect[1] - win_rect[1] < 48):
            e["chrome"] = True   # the window's own buttons: only offered when the goal is about them
        if r["ct"] == 50003:
            combo = rect
            e["kind"] = "dropdown"
            e["value"] = norm_text(r.get("value"))[:80] or e.get("value", "")
            # a combo with a writable edit box inside it takes typed text (a Save As dialog's File name); one without
            # (a drop-down list) takes a choice
            if any(c["ct"] == 50004 and not c["readonly"] and rect[0] <= c["rect"][0] and rect[1] <= c["rect"][1]
                   and rect[2] >= c["rect"][2] and rect[3] >= c["rect"][3] for c in raws[idx + 1: idx + 6]):
                e["kind"] = "text field"
        els.append(e)
        if len(els) >= MAX_ELEMENTS:
            break
    total: dict[tuple, int] = {}
    for e in els:
        total[(e["kind"], e["label"])] = total.get((e["kind"], e["label"]), 0) + 1
    nth: dict[tuple, int] = {}
    for e in els:   # the same control twice: say which and where, so a choice can tell them apart
        key = (e["kind"], e["label"])
        if total[key] > 1:
            nth[key] = nth.get(key, 0) + 1
            e["context"] = (e.get("context", "") + f" {region_word(e['rect'], win_rect)}, {nth[key]} of {total[key]}").strip()
    return els, texts


def describe(e: dict) -> str:
    s = f"{e['kind']} '{e['label']}'" if e.get("label") else e["kind"]
    if e.get("layer"):
        s += f" [{e['layer']}]"
    if e.get("context"):
        s += f" (in: {e['context']})"
    if e.get("value"):
        s += f" = '{e['value']}'"
    if "checked" in e:
        s += " (checked)" if e["checked"] else " (unchecked)"
    if e.get("selected"):
        s += " (selected)"
    if e.get("expanded"):
        s += " (open)"
    if e.get("disabled"):
        s += " (disabled)"
    return s


def window_line(w: dict) -> str:
    s = f"{w['app']} '{w['title']}'"
    if w.get("tabs"):
        s += " tabs: " + " | ".join(t[:40] for t in w["tabs"][:10])
    if w.get("foreground"):
        s += " (in front)"
    if w.get("parked"):
        s += " (parked off-screen)"
    if w.get("minimized"):
        s += " (minimized)"
    return s


def signature(obs: dict) -> str:
    import hashlib
    controls = [(e["kind"], e["label"], e.get("value"), e.get("checked"), e.get("selected"), e.get("expanded"),
                 e["disabled"]) for e in obs["elements"]]
    return hashlib.sha1(repr((obs["title"], obs["layers"], obs["text"], controls)).encode()).hexdigest()


class SecretDeclined(RuntimeError):
    """The user did not allow a password to be typed (or there was nobody to ask)."""


SECRET_OPTIONS = ["Allow", "Allow for this task", "Don't allow - I'll type it myself"]


def _ask_user(question: str, options: list | None = None) -> str | None:
    """Ask the user from inside the tool, as the nango tool does: the worker's own code never sees a question it could answer
    for the user. None: there is no user here (not in a worker's REPL)."""
    main = sys.modules.get("__main__")
    call = getattr(main, "_call", None)
    if call is None or not hasattr(main, "ask_user"):
        return None
    return str(call("ask_user", question, options) or "")


class Overwrite(RuntimeError):
    """Replacing text the tool did not write (a document, a long field) needs a yes first."""


class Session:
    """The Jev loop over one app at a time. Module-level functions use a shared session on its own thread."""

    def __init__(self, ui=None, jev=None, cap: int | None = None):
        if jev is None:
            jev, default_cap = client_from_env()
            cap = cap or default_cap
        self.jev, self.cap = jev, cap or 16
        self.ui = _backend(ui)
        self.jev_calls = 0
        self.jev_secs = 0.0
        self.pause = float(os.environ.get("JEVH_DESKTOP_PAUSE", "0.1"))
        self._elements: list[dict] = []   # the numbering click()/fill()/select() refer to
        self._hwnd = 0
        self.button_log: list[dict] = []
        self.shots = 0
        self.ask_user = _ask_user
        self.secret_all = False   # the user said "Allow for this task"
        self.written: dict[tuple, str] = {}   # (window, field) -> the text this tool last put there

    # ------------------------------------------------------------------ plumbing
    def ask(self, state, questions):
        """Every Jev request goes through here (also from choose()/pick_passages(), which get the session as their
        client), so calls and time are counted per do()."""
        t = time.monotonic()
        self.jev_calls += 1
        try:
            return self.jev.ask(state, questions)
        finally:
            self.jev_secs += time.monotonic() - t

    _ask = ask

    def _choose(self, state, instructions, options, extra=None):
        return choose(self, state, instructions, options, self.cap, extra)

    # ------------------------------------------------------------------ windows
    def windows(self, tabs: bool = False) -> list[dict]:
        """The app windows open now, front to back. `parked` windows sit off-screen (a docked browser the harness
        keeps for a session); they are listed after the others and never chosen unless named."""
        raw = self.ui.windows()
        out = []
        for w in raw:
            if w["owner"] or w["tool"] or not w["title"] or w["exe"] in HELPERS or w["title"] in ("Program Manager",):
                continue
            r = w["rect"]
            parked = not w["minimized"] and (r[0] <= -30000 or r[0] >= 30000)
            app = w["exe"] or w["cls"]
            if app == "ApplicationFrameHost":
                app = w["title"]
            d = {"id": w["hwnd"], "app": app, "title": w["title"], "pid": w["pid"], "foreground": w["foreground"],
                 "minimized": w["minimized"], "parked": parked, "rect": w["rect"], "cls": w["cls"]}
            if tabs and app.lower() in BROWSERS and not parked:
                d["tabs"] = self._tabs(w["hwnd"])
            out.append(d)
        return sorted(out, key=lambda d: d["parked"])

    def _tabs(self, hwnd: int) -> list[str]:
        try:
            return [label_of(r) for r in self.ui.snapshot(hwnd) if r["ct"] == 50019 and label_of(r)]
        except Exception:  # noqa: BLE001
            return []

    def resolve(self, window, goal: str = "") -> dict | None:
        """The window meant by `window` (an id, a window dict, an app name or part of a title), best first: in front,
        then visible, then the rest. None if nothing matches."""
        wins = self.windows()
        if isinstance(window, dict):
            window = window.get("id", window.get("title"))
        if isinstance(window, int) or (isinstance(window, str) and window.isdigit()):
            return next((w for w in wins if w["id"] == int(window)), None)
        want = (window or "").strip().lower()
        if not want:
            return None

        def rank(w):
            return (not w["foreground"], w["parked"], w["minimized"])
        exact = [w for w in wins if w["title"].lower() == want or w["app"].lower() == want]
        part = [w for w in wins if want in w["title"].lower() or want in w["app"].lower()]
        words = [w for w in wins if _words(want) and _words(want) <= (_words(w["title"]) | _words(w["app"]))]
        for group in (exact, part, words):
            if group:
                return sorted(group, key=rank)[0]
        return None

    def _app_in_goal(self, goal: str) -> dict | None:
        """The one open window whose app or title the goal names ("in the Claude app"); None if none or several."""
        gw = _words(goal) - {"the", "app", "window", "new", "open", "click", "press", "button", "type", "tab", "in", "on"}
        hits = []
        for w in self.windows():
            if w["parked"]:
                continue
            names = _words(w["app"]) | (_words(w["title"]) if len(w["title"]) < 30 else set())
            if names & gw and (_words(w["app"]) & gw):
                hits.append(w)
        return hits[0] if len(hits) == 1 else None

    def where(self, question: str) -> dict:
        """Jev picks the window (and the browser tab) that matches `question` from everything open."""
        wins = [w for w in self.windows(tabs=True) if not w["parked"]]
        if not wins:
            return {"status": "no_window", "note": "no app windows are open"}
        opts: dict[str, str] = {}
        for w in wins:
            opts[f"w{w['id']}"] = f"{w['app']} window '{w['title']}'" + (" (in front)" if w["foreground"] else "")
            for n, t in enumerate(w.get("tabs") or []):
                opts[f"w{w['id']}.t{n}"] = f"tab '{t}' of the {w['app']} window '{w['title']}'"
        opts["none"] = "none of these"
        probs, _ = self._choose({"question": question}, Q_MATCH, opts)
        top = sorted(probs, key=probs.get, reverse=True)
        best = top[0]
        if best == "none" or probs[best] < 0.2:
            return {"status": "none", "note": "no open window or tab matches", "p": round(probs[best], 2),
                    "windows": [window_line(w) for w in wins]}
        wid = int(best[1:].split(".")[0])
        w = next(x for x in wins if x["id"] == wid)
        tab = w["tabs"][int(best.split(".t")[1])] if ".t" in best else None
        return {"status": "found", "window": w, "tab": tab, "p": round(probs[best], 2),
                "other": [opts[k] for k in top[1:3] if k != "none"]}

    def focus(self, window) -> dict:
        w = self.resolve(window)
        if not w:
            return {"status": "no_window", "note": f"no window matches {window!r}", "windows": [window_line(x) for x in self.windows()]}
        ok = self.ui.focus(w["id"])
        return {"status": "done" if ok else "needs_user", "window": window_line({**w, "foreground": ok}),
                "note": "" if ok else "Windows would not bring it to the front (another program holds the focus)"}

    def open(self, app: str, wait: float = 10.0) -> dict:
        """Start an app by name ('notepad', 'calculator', a path, 'ms-settings:') and wait for its new window."""
        before = {w["id"] for w in self.windows()}
        target = KNOWN_APPS.get(app.strip().lower()) or (app if os.path.exists(app) or ":" in app else shutil.which(app) or app)
        try:
            self.ui.open(target)
        except OSError as e:
            return {"status": "error", "note": f"could not start {app!r}: {e}"}
        end = time.monotonic() + wait
        while time.monotonic() < end:
            time.sleep(0.25)
            new = [w for w in self.windows() if w["id"] not in before and not w["parked"]]
            if new:
                time.sleep(0.4)
                return {"status": "done", "window": window_line(new[0]), "id": new[0]["id"]}
        return {"status": "stuck", "note": f"started {target!r} but no new window appeared in {wait:.0f}s "
                f"(it may have opened in a window that was already there: desktop.windows())"}

    # ------------------------------------------------------------------ looking
    def _layers(self, w: dict) -> tuple[dict | None, list[dict], bool]:
        """The window itself, the dialogs/menus/popups of the same app on top of it, and modal? (it is disabled)."""
        allw = self.ui.windows()
        main = next((x for x in allw if x["hwnd"] == w["id"]), None)
        if main is None:
            return None, [], False
        extra = []
        owner_of = {x["hwnd"]: x["owner"] for x in allw}

        def owned_by_main(h: int) -> bool:   # (a message box over a dialog is owned by the dialog, not by the window)
            for _ in range(5):
                h = owner_of.get(h, 0)
                if h == main["hwnd"]:
                    return True
                if not h:
                    return False
            return False
        for x in allw:
            if x["hwnd"] == main["hwnd"]:
                continue
            owned = owned_by_main(x["hwnd"])
            popup = x["pid"] == main["pid"] and any(x["cls"].startswith(c) for c in POPUP_CLASSES) and not x["title"] \
                and x["rect"][0] < main["rect"][2] and x["rect"][2] > main["rect"][0]
            if owned or popup:
                extra.append(x)
        # a dialog with another dialog over it is disabled, and a UIA call into a disabled window can hang for minutes
        # (live, 2026-10-08: one elements() call took 7 minutes with an error box over a Save As dialog): read the top one
        live = [x for x in extra if x["enabled"]]
        extra = live or extra
        modal = bool(extra) and not main["enabled"] and any(owned_by_main(x["hwnd"]) for x in extra)
        return main, extra, modal

    def _snap(self, hwnd: int) -> list[dict] | None:
        for _ in range(2):
            try:
                return self.ui.snapshot(hwnd)
            except Exception:  # noqa: BLE001 - a window still being built, or just closed
                time.sleep(0.2)
        return None

    def _look(self, w: dict) -> dict:
        main, extra, modal = self._layers(w)
        if main is None:
            raise LookupError("the window is gone")
        parts = []   # (hwnd, label, rect)
        if not modal:
            parts.append((w["id"], "", main["rect"]))
        for x in extra:
            parts.append((x["hwnd"], f"dialog '{x['title']}'" if x["title"] else "popup", x["rect"]))
        elements, texts, scrollable = [], [], False
        for hwnd, layer, rect in parts:
            raws = self._snap(hwnd)
            if raws is None:
                if hwnd == w["id"]:
                    raise LookupError("the window cannot be read right now")
                continue   # a popup that went away while it was being read
            scrollable = scrollable or any(r["avail"]["scroll"] and not r["off"] for r in raws)
            if hwnd == w["id"] and len(raws) < 12 and main["cls"].startswith("Chrome_WidgetWin"):
                for _ in range(3):   # Chromium builds its accessibility tree when the first client asks
                    time.sleep(0.4)
                    raws = self.ui.snapshot(hwnd)
                    if len(raws) >= 12:
                        break
            es, ts = build_elements(raws, rect if hwnd == w["id"] else (rect[0] - 5, rect[1] - 5, rect[2] + 5, rect[3] + 5), layer)
            for e in es:
                e["i"] = len(elements)
                e["hwnd"] = hwnd
                elements.append(e)
            texts += ([f"-- {layer} --"] if layer else []) + ts
        notices = [layer for _, layer, _ in parts if layer]
        text = "\n".join(texts)
        obs = {"hwnd": w["id"], "title": main["title"], "scrollable": scrollable, "app": w["app"], "elements": elements, "text": text,
               "layers": notices, "modal": modal, "rect": main["rect"], "foreground": main["foreground"]}
        obs["sig"] = signature(obs)
        return obs

    def observe(self, w: dict) -> dict:
        """The window as it stands once it has stopped changing (an animation or a page still loading)."""
        obs = self._look(w)
        for _ in range(4):
            time.sleep(0.12)
            again = self._look(w)
            same = again["sig"] == obs["sig"]
            obs = again
            if same:
                break
        return obs

    def _remember(self, obs: dict) -> None:
        """The numbering click()/fill()/select() refer to: only ever a list the worker was shown (elements(), a do()'s
        candidates). Looking again after an action must not renumber it (live: after fill() opened an autocomplete list,
        click(3) - Send in the list the worker had - hit 'Line up' in the new one)."""
        self._elements, self._hwnd = obs["elements"], obs["hwnd"]

    def _view(self, obs: dict, n: int = TEXT_CHARS, goal: str = "") -> str:
        text = f"{obs['title']}\n{obs['text']}"
        if len(text) <= n:
            return text
        lines = prefilter(goal, text.split("\n"), 60) if goal else text.split("\n")
        out = "\n".join(lines)
        return out[:n]

    # ------------------------------------------------------------------ acting
    def _front(self, hwnd: int) -> bool:
        """The window is in front, bringing it there if need be. One more try after a second: when the front window just
        closed, Windows can leave the foreground with a window nobody can see for a moment (seen live; NOT tested against
        that state, which would not come back on demand)."""
        if self.ui.in_front(hwnd) or self.ui.focus(hwnd):
            return True
        time.sleep(1.0)
        return self.ui.focus(hwnd)

    def _press(self, e: dict, hwnd: int, mouse: bool = False, double: bool = False, right: bool = False,
               other: bool = False) -> str:
        """Act on a control. Tabs, list items, radios, checkboxes: their own UIA pattern (no focus needed). Buttons, menu
        entries, links: a real click, because Invoke on a button that opens a modal dialog does not return until the dialog
        closes and leaves later UIA calls into that app hanging (measured on the fixture: 5 s lost, then every read of the
        dialog failed; a real click: the dialog read in 10 ms). `other` swaps the method (used after the first did nothing)."""
        what = f"{e['kind']} '{e['label']}'"
        use_mouse = mouse or double or right or ((e["kind"] not in PATTERN_FIRST) != other)
        if not use_mouse:
            try:
                used = self.ui.invoke(e["raw"], e["kind"] in PATTERN_FIRST)
                return f"{'toggled' if used == 'toggle' else 'selected' if used == 'select' else 'clicked'} {what}"
            except Exception:  # noqa: BLE001 - no pattern, or the app refused it: click it
                pass
        if not self._front(hwnd):
            raise RuntimeError("the window would not come to the front, so nothing was clicked")
        x, y = (e["rect"][0] + e["rect"][2]) // 2, (e["rect"][1] + e["rect"][3]) // 2
        self.ui.click_at(x, y, "right" if right else "left", double)
        return f"{'right-' if right else 'double-' if double else ''}clicked {what}" + (" with the mouse" if other else "")

    def _may_type_secret(self, e: dict, hwnd: int) -> None:
        """A password goes into a field only if the user says so: the tool asks them itself ("Allow" = this once). There is no
        argument a worker can pass to skip it. Raises SecretDeclined when they do not allow it, or nobody can be asked."""
        if self.secret_all:
            return
        title = next((w["title"] for w in self.windows() if w["id"] == hwnd), "a window")
        answer = self.ask_user(f"May I type a password into the '{e['label']}' field of '{title}'? It goes straight into the "
                               "window: it is not shown, kept or read back.", SECRET_OPTIONS)
        a = (answer or "").strip().lower()
        if answer is None:
            raise SecretDeclined("nobody can be asked: a password is typed only with the user's permission. Ask them to type it "
                                 "in the window themselves.")
        if a.startswith("allow for this task"):
            self.secret_all = True
        elif not a.startswith("allow"):
            raise SecretDeclined("the user did not allow typing the password: ask them to type it in the window themselves "
                                 "(do not work around this)")

    def _guard_overwrite(self, e: dict, hwnd: int) -> None:
        """Typing over an editor's content, or a long field, loses the user's text (live, 2026-10-08: a test replaced the
        unsaved note in a restored Notepad tab). Only text this tool wrote itself may be replaced without a yes."""
        have = norm_text(self.ui.get_value(e["raw"]))
        if have and (e["kind"] == "text area" or len(have) > 200) and have != self.written.get((hwnd, e["label"])):
            raise Overwrite(f"{describe(e)} already holds {len(have)} characters of someone's text ({have[:50]!r}...): "
                            "typing would replace it. Use replace=False to add at the end, or ask_user first and call again "
                            "with allow_irreversible=True")

    def _type_into(self, e: dict, hwnd: int, text: str, replace: bool = True, allow_overwrite: bool = False) -> dict:
        """Put `text` in a field as real keystrokes (or the clipboard), then read it back."""
        raw, ui = e["raw"], self.ui
        secret = e["kind"] == "password"
        if secret:
            self._may_type_secret(e, hwnd)
        elif replace and not allow_overwrite:
            self._guard_overwrite(e, hwnd)
        if not self._front(hwnd):
            return {"ok": False, "note": "the window would not come to the front, so nothing was typed"}
        try:
            ui.set_focus(raw)
        except Exception:  # noqa: BLE001
            x, y = (e["rect"][0] + e["rect"][2]) // 2, (e["rect"][1] + e["rect"][3]) // 2
            ui.click_at(x, y)
        time.sleep(0.08)
        before = ui.get_value(raw)
        ui.keys("ctrl+a" if replace else "ctrl+end")
        # (a password is always typed key by key: never through the clipboard, and it cannot be read back)
        (ui.paste if (not secret and ("\n" in text or len(text) > 120)) else ui.type_text)(text)
        time.sleep(0.2)
        if secret:
            return {"ok": True, "verified": None, "method": "keys", "secret": True}
        got, method = ui.get_value(raw), "keys"
        want = norm_text(text)
        good = lambda g: g is not None and (norm_text(g) == want if replace else norm_text(g).endswith(want))
        if got is not None and not good(got):
            try:   # the keys did not land as typed (a field that eats characters): set the value directly
                ui.set_value(raw, text if replace else (before or "") + text)
                time.sleep(0.15)
                got, method = ui.get_value(raw), "value"
            except Exception:  # noqa: BLE001
                pass
        if got is not None:
            self.written[(hwnd, e["label"])] = norm_text(got)
        return {"ok": True, "verified": None if got is None else good(got), "method": method, "value": norm_text(got)[:120]}

    def _submit(self, e: dict, hwnd: int, text: str) -> str:
        """Enter in a field, checked: sent when the field emptied; an autocomplete that took the Enter (it rewrote the
        text instead - live: '/rc' became 'remote-control ') is closed and the text sent again."""
        ui, raw = self.ui, e["raw"]
        for attempt in range(2):
            before = norm_text(ui.get_value(raw))
            ui.keys("enter")
            time.sleep(0.45)
            try:
                after = ui.get_value(raw)
            except Exception:  # noqa: BLE001 - the control went away with the message
                after = ""
            if after is None:
                return "pressed Enter (the field cannot be read back, so not confirmed)"
            a = norm_text(after)
            if a == "":
                return "pressed Enter; the field emptied (sent)"
            if a == before:
                return "pressed Enter but the text is still there (this field may need another key, or the send button)"
            if attempt == 0:   # rewritten: close the suggestions, put the text back, Enter again
                ui.keys("ctrl+a")
                ui.keys("delete")
                (ui.paste if "\n" in text else ui.type_text)(text)
                time.sleep(0.15)
                ui.keys("escape")
                time.sleep(0.1)
        return f"pressed Enter twice but the field shows {a[:40]!r}: an autocomplete is still in the way"

    def _pick_option(self, state, e: dict, hwnd: int, wants: list[str]) -> str:
        """Open a dropdown, let Jev choose among its items (or take the one that matches `want`), click it."""
        ui = self.ui
        try:
            if not ui.expand(e["raw"]):
                self._press(e, hwnd, mouse=True)
        except Exception:  # noqa: BLE001
            self._press(e, hwnd, mouse=True)
        time.sleep(0.35)
        w = {"id": hwnd, "app": ""}
        opts = [x for x in self._look(w)["elements"] if x["kind"] in ("list item", "menu item")]
        if not opts:
            return f"opened {describe(e)} but found no options"
        pick = next((o for o in opts if any(_same_text(o["label"], w) for w in wants)), None)
        if pick is None:
            probs, _ = self._choose(state, f"Which option of dropdown '{e['label']}' does `task.goal` call for?",
                                    {str(n): o["label"] or "(blank)" for n, o in enumerate(opts)})
            pick = opts[int(max(probs, key=probs.get))]
        self._press(pick, hwnd)
        return f"chose '{pick['label']}' in dropdown '{e['label']}'"

    def _keys(self, combo: str, hwnd: int, times: int = 1) -> str:
        if not self._front(hwnd):
            raise RuntimeError("the window would not come to the front, so no keys were sent")
        self.ui.keys(combo, times)
        return f"pressed {combo}" + (f" x{times}" if times > 1 else "")

    # ------------------------------------------------------------------ the loop
    def _target(self, goal: str, window) -> tuple[dict | None, str]:
        if window is not None:
            w = self.resolve(window)
            return w, "" if w else f"no window matches {window!r}"
        w = self._app_in_goal(goal)
        if w:
            return w, ""
        wins = [x for x in self.windows(tabs=True) if not x["parked"] and not x["minimized"]] or \
               [x for x in self.windows() if not x["parked"]]
        if not wins:
            return None, "no app windows are open"
        if len(wins) == 1:
            return wins[0], ""
        opts = {str(x["id"]): window_line(x) for x in wins}
        probs, _ = self._choose({"task": {"goal": goal}, "windows": list(opts.values())}, Q_WINDOW, opts)
        best = max(probs, key=probs.get)
        return next(x for x in wins if str(x["id"]) == best), ""

    def do(self, goal: str, window=None, values: dict | None = None, max_actions: int = 8,
           allow_irreversible: bool = False) -> dict:
        """Reach one observable outcome in one app. See the module docstring."""
        started, calls0, secs0 = time.monotonic(), self.jev_calls, self.jev_secs
        values = {str(k): str(v) for k, v in (values or {}).items()}
        for k, v in goal_values(goal).items():   # "set Display name to Ana": the goal holds its own values
            if not any(_same_text(k, g) for g in values):
                values[k] = v
        used: set[str] = set()
        history: list[str] = []
        banned: set[int] = set()
        deferred: set[int] = set()
        escalate: set[tuple] = set()   # controls whose pattern did nothing: next time by mouse
        no_effect = 0
        last_change = "nothing yet"
        choice: tuple[dict, dict] | None = None
        win_label = ""

        def result(status: str, obs: dict | None, note: str = "", probs: dict | None = None) -> dict:
            out = {"status": status, "goal": goal, "window": win_label, "actions": history, "note": note,
                   "page": self._view(obs, 1200, goal) if obs else "",
                   "p": {k: round(v, 2) for k, v in (probs or {}).items()},
                   "jev_calls": self.jev_calls - calls0, "jev_seconds": round(self.jev_secs - secs0, 1),
                   "seconds": round(time.monotonic() - started, 1)}
            if obs:
                out["title"] = obs["title"]
            if status in ("stuck", "ambiguous", "max_actions", "error") and obs and choice and choice[0] is obs:
                by_i = {str(e["i"]): e for e in obs["elements"]}
                top = [k for k in sorted(choice[1], key=choice[1].get, reverse=True) if k in by_i][:6]
                if top:
                    self._remember(obs)
                    out["candidates"] = [f"[{k}] {describe(by_i[k])}" for k in top]
                    out["note"] += ("; " if out["note"] else "") + \
                        "act yourself if you see the right one: desktop.click(i) / fill(i, text) / select(i, option)"
            return out

        # starting an app, and a bare key press, are settled before any window is chosen
        km = KEY_GOAL.match(goal)
        om = OPEN_GOAL.match(goal) if window is None and not km else None
        app = om.group(1).strip().lower() if om else ""
        if app and (app in KNOWN_APPS or shutil.which(app)):
            have = self.resolve(app)   # already open: bring it forward instead of starting another
            if have and not have["parked"]:
                r = self.focus(have["id"])
                history.append(f"{app} was already open: brought it to the front")
                return result("done" if r["status"] == "done" else r["status"], None, r.get("note", ""))
            r = self.open(app)
            history.append(f"started {app}")
            win_label = r.get("window", "")
            return result(r["status"], None, r.get("note", ""))
        if km and window is None:
            w = next((x for x in self.windows() if x["foreground"]), None)
            why = "" if w else "no window is in front"
        else:
            w, why = self._target(goal, window)
        if w is None:
            return result("no_window", None, why or "no window to work in: desktop.windows() lists what is open")
        win_label = window_line(w)
        if km and not values:
            combo = re.sub(r"\s*\+\s*", "+", km.group(1).lower())
            if combo in RISKY_KEYS and not allow_irreversible:
                return result("needs_confirmation", None, f"{combo} can close or lose something: call again with "
                              "allow_irreversible=True if the user asked for it")
            try:
                before = self.observe(w)
                history.append(self._keys(combo, w["id"]))
            except Exception as ex:  # noqa: BLE001
                return result("needs_user", None, str(ex))
            time.sleep(0.3)
            try:
                after = self.observe(self.resolve(w["id"]) or w)
            except LookupError:
                return result("done", None, "the window closed")
            same = after["sig"] == before["sig"]
            return result("likely_done" if same else "done", after,
                          "the window looks the same after the keys: check `page`" if same else "")
        named = click_label(goal)
        nav = nav_target(goal)
        final = final_click(goal)
        if final and re.fullmatch(KEY_NAMES, final, re.I):
            final = None   # (a key to press, not a control to click)
        try:
            obs = self.observe(w)
            before_names = {e["label"].lower() for e in obs["elements"]} | {ln.lower() for ln in obs["text"].split("\n")}
        except Exception as ex:  # noqa: BLE001
            return result("blocked", None, f"cannot read that window ({type(ex).__name__}): an administrator-level or "
                          "not-responding window cannot be controlled")
        for _ in range(2 * max_actions + 2):
            # values that obviously belong in an empty field go in without asking Jev
            try:
                filled = self._autofill(obs, values, used, w["id"], allow_irreversible)
            except Overwrite as ex:
                return result("needs_confirmation", obs, str(ex))
            except SecretDeclined as ex:
                return result("needs_user", obs, str(ex))
            if filled:
                history.extend(filled)
                obs = self._refresh(w, obs)
                if obs is None:
                    return result("done", None, "the window closed")
                continue
            els = [e for e in obs["elements"] if e["i"] not in banned | deferred and not e["disabled"]
                   and not (e["kind"] == "password") and (not e.get("chrome") or CHROME_GOAL.search(goal))]
            if nav and not values:   # "open the List tab": the tab strip names it; no need to ask
                hit = [e for e in obs["elements"] if e["kind"] in NAV_KINDS and _same_text(e["label"], nav)]
                if len(hit) == 1 and (hit[0].get("selected") or hit[0].get("checked")):
                    return result("done", obs, f"'{nav}' is already open")
                if len(hit) == 1 and not hit[0]["disabled"]:
                    named = nav
            if named and not values:
                exact = [e for e in els if _label_is(e["label"], named) or
                         (e["kind"] in ("button", "link", "tab", "menu item", "list item") and e["label"].lower().startswith(named.lower() + " "))]
                loose = [e for e in els if named.lower() in e["label"].lower() and e["kind"] in NAV_KINDS]
                if exact:
                    els = exact
                elif any(_label_is(e["label"], named) for e in obs["elements"]):
                    return result("stuck", obs, f"'{named}' was clicked and nothing changed: read `page` for what it is "
                                  "waiting for (a required field, a dialog)")
                elif loose and len(loose) <= 4:
                    els = loose
                elif not history:
                    # a goal that names a control the window does not have must not become a click on something else (live:
                    # "click the Cancel button of the Settings dialog" with no such dialog selected a radio button)
                    cands = [f"[{e['i']}] {describe(e)}" for e in obs["elements"] if e["kind"] in NAV_KINDS][:8]
                    return result("stuck", obs, f"no control named '{named}' in this window (the goal may be about another window "
                                  "or screen): desktop.elements() lists what there is", None) | {"candidates": cands}
            dec = self._decide(obs, els, goal, history, last_change, bool(values))
            if dec is None:
                return result("stuck", obs, "no control in this window can be acted on", None)
            target_p, probs, state = dec
            best = max(target_p, key=target_p.get)
            choice = (obs, target_p)
            if SUBMIT_WORDS.search(goal) and not any(re.match(r"(clicked|toggled|selected) |.*pressed Enter", h) for h in history):
                probs["done"] = min(probs["done"], 0.29)
            if final and not any(h.startswith(("clicked", "selected")) and f"'{final.lower()}'" in h.lower() for h in history):
                probs["done"] = min(probs["done"], 0.29)   # the goal ends with a click that has not happened yet
            if probs["done"] >= 0.9:
                return result("done", obs, probs=probs)
            if probs.get("error", 0) >= 0.7:
                return result("error", obs, "the window shows an error after the last action", probs)
            if probs.get("blocked", 0) >= 0.85:
                return result("blocked", obs, "something in the window blocks progress", probs)
            if probs.get("login", 0) >= 0.7:
                return result("needs_user", obs, "sign-in required: call again with values={'Username': ..., 'Password': ...} (a password is typed only if the user allows "
                              "it when the tool asks), or have the user sign in themselves", probs)
            if history and probs["done"] >= DONE_AFTER_ACTING:
                return result("done", obs, probs=probs)
            if values and all(k in used for k in values) and not SUBMIT_WORDS.search(goal) and probs["done"] >= 0.5:
                return result("done", obs, probs=probs)
            if sum(1 for h in history if re.match(r"(clicked|toggled|selected|filled|typed|chose|pressed|scrolled)", h)) >= max_actions:
                return result("max_actions", obs, f"stopped after {max_actions} actions", probs)
            if target_p[best] < 0.2:
                by_i = {str(e["i"]): e for e in obs["elements"]}
                top = sorted(target_p, key=target_p.get, reverse=True)[:3]
                return result("ambiguous", obs, "unsure which control: " + "; ".join(
                    f"{describe(by_i[k]) if k in by_i else k} ({target_p[k]:.2f})" for k in top), probs)

            target = None if best in PSEUDO else next(e for e in obs["elements"] if str(e["i"]) == best)
            tool = "scroll" if best == "scroll" else self._tool_for(target)
            value = pick = None
            if tool == "type":
                pick = self._pick_value(goal, target, values, used)
                value = values.get(pick) if pick else None
                quoted = QUOTED.findall(goal) or [m.group(1) for m in [SET_TO.match(goal)] if m]
                if value is None and not values and len(quoted) == 1:
                    value, pick = quoted[0], None
                if value is not None and (target.get("value") or "") == norm_text(value)[:80] and "\n" not in value:
                    banned.add(target["i"])
                    last_change = f"{describe(target)} already holds it"
                    continue
                if value is None:
                    if target.get("value"):
                        banned.add(target["i"])
                        last_change = f"{describe(target)} is already filled in"
                        continue
                    return result("needs_value", obs, f"{describe(target)} needs text: call again with "
                                  f"values={{'{target.get('label') or 'text'}': '<the text>'}}", probs)
            elif tool == "click" and target["kind"] in ("button", "menu item", "link", "list item", "tab"):
                irreversible, ready, wanted = self._check_button(goal, target, obs, allow_irreversible)
                if named and _label_is(target["label"], named):
                    ready = wanted = True
                if not wanted:
                    deferred.add(target["i"])
                    last_change = f"not clicking {describe(target)}: it isn't part of the goal"
                    continue
                if not ready:
                    deferred.add(target["i"])
                    last_change = f"not clicking {describe(target)} yet: the form doesn't match the goal"
                    continue
                if irreversible:
                    return result("needs_confirmation", obs, f"next: click {describe(target)}, which looks hard to undo "
                                  "(sends, deletes, pays or closes something); call again with allow_irreversible=True if "
                                  "the user asked for exactly that", probs)

            before = obs["sig"]
            try:
                label = self._act(tool, target, value, w["id"], escalate, goal, state, values, allow_irreversible)
            except Overwrite as ex:
                return result("needs_confirmation", obs, str(ex), probs)
            except Exception as ex:  # noqa: BLE001 - a failed action is information, not a crash
                label = f"tried to {tool} {describe(target) if target else ''} but it failed ({str(ex).splitlines()[0][:100]})"
                if "front" in str(ex):
                    history.append(label)
                    return result("needs_user", obs, str(ex), probs)
            if pick:
                used.add(pick)
                label += f" (value '{pick}')"
            history.append(label)
            if "(sent)" in label:   # the field emptied on Enter: the message went; typing it again would send it twice
                obs = self._refresh(w, obs)
                return result("done", obs, "the message was sent", probs)
            if target is not None and target["kind"] in TOGGLES - {"dropdown"}:
                banned.add(target["i"])
            obs = self._refresh(w, obs)
            if obs is None:
                return result("done", None, "the window closed after the last action", probs)
            if obs["sig"] != before and named_gone(goal, before_names, obs):
                return result("done", obs, "what the goal names is gone from the window")
            if obs["sig"] == before:
                no_effect += 1
                last_change = "the last action changed nothing in the window"
                if target is not None:
                    key = (target["kind"], target["label"])
                    if tool == "click" and key not in escalate:
                        escalate.add(key)   # that way of acting did nothing: the other one next (mouse <-> UIA pattern)
                        last_change += " (will act on it the other way)"
                    else:
                        deferred.add(target["i"])   # tried again whenever the window changes (selecting an item enables Remove)
                if no_effect >= 3:
                    return result("stuck", obs, "the last actions had no visible effect", probs)
            else:
                if final and tool == "click" and target is not None and _same_text(target["label"], final):
                    return result("done", obs, f"clicked {describe(target)}, the last step of the goal", probs)
                if named and tool == "click" and target is not None and _label_is(target["label"], named):
                    return result("done", obs, f"clicked {describe(target)}, as the goal says: read `page` for what it led to", probs)
                no_effect = 0
                deferred.clear()
                last_change = f"after '{label}' the window changed" + (
                    f" and now has {', '.join(obs['layers'])}" if obs["layers"] else "")
        return result("max_actions", obs, f"stopped after {max_actions} actions")

    def _decide(self, obs: dict, els: list[dict], goal: str, history: list[str], last_change: str, has_values: bool):
        """Jev's view of one step: (distribution over `els`, {done/error/blocked/login: P}, the state it was shown), or None
        when there is nothing to choose between. Shared by do() and the decision eval (evals/desktop_cases.py)."""
        controls = [describe(e) for e in obs["elements"] if e["kind"] in FIELDS | TOGGLES][:30]
        state = {"task": {"goal": goal, "history": history[-6:] or ["(no actions yet)"], "last_change": last_change},
                 "window": {"app": obs["app"], "title": obs["title"], "text": self._view(obs, TEXT_CHARS, goal),
                            "selected": [e["label"] for e in obs["elements"] if e.get("selected")][:8],
                            "controls": controls, "dialogs": obs["layers"]}}
        extra = {"done": noul(Q_DONE)}
        if history:
            extra["error"] = noul(Q_ERROR)
        else:
            extra["blocked"] = noul(Q_BLOCKED)
            if not has_values:
                extra["login"] = noul(Q_LOGIN)
        options = {str(e["i"]): describe(e)[:160] for e in els}
        # no "none" option: measured on the fixture, Jev took it for "open the settings dialog" with the Open settings
        # button on screen (0.7 vs 0.07). "Is it done?" is its own question; the choice is always a control (or a scroll,
        # offered only where something scrolls)
        pseudo = PSEUDO if obs["scrollable"] else {}
        if not options and not pseudo:
            return None
        if len(options) + len(pseudo) <= self.cap:
            target_p, ans = self._choose(state, Q_TARGET, {**options, **pseudo}, extra)
        else:
            target_p, ans = self._pick_target(state, els, options, extra, pseudo)
        return target_p, {k: ans[k].noul for k in ("done", "error", "blocked", "login") if k in ans}, state

    def _refresh(self, w: dict, old: dict) -> dict | None:
        """Look again after an action. The window may have been renamed (a saved file) or closed (None)."""
        time.sleep(self.pause)
        cur = self.resolve(w["id"])
        if cur is None:
            return None
        try:
            return self.observe(cur)
        except LookupError:
            return None

    @staticmethod
    def _tool_for(target: dict | None) -> str:
        if target is None:
            return "scroll"
        if target["kind"] in FIELDS:
            return "type"
        if target["kind"] == "dropdown":
            return "select"
        return "click"

    def _act(self, tool: str, target: dict | None, value: str | None, hwnd: int, escalate: set, goal: str, state,
             values: dict, allow: bool = False) -> str:
        if tool == "scroll":
            ok = self.ui.scroll(None, hwnd, 1)
            return "scrolled down" if ok else "could not scroll"
        if tool == "type":
            r = self._type_into(target, hwnd, value, allow_overwrite=allow)
            if not r["ok"]:
                raise RuntimeError(r["note"])
            label = f"typed into {describe(target)}" + (" (read back OK)" if r["verified"] else
                                                        " (could NOT confirm it landed)" if r["verified"] is False else "")
            # Enter ends a search box or a message ("type hello and send it"), never one field of a form
            if len(values) <= 1 and (target["kind"] == "search box" or ENTER_GOAL.search(goal)):
                label += "; " + self._submit(target, hwnd, value)
            return label
        if tool == "select":
            return self._pick_option(state, target, hwnd, list(values.values()))
        key = (target["kind"], target["label"])
        if DOUBLE_GOAL.search(goal) and target["kind"] in ("list item", "tree item", "button"):
            return self._press(target, hwnd, double=True)
        return self._press(target, hwnd, other=key in escalate)

    def _check_button(self, goal: str, target: dict, obs: dict, allow_irreversible: bool) -> tuple[bool, bool, bool]:
        """(irreversible, ready, wanted) for a click, in one small call (the browser tool's button check)."""
        controls = [describe(e) for e in obs["elements"] if e["kind"] in FIELDS | TOGGLES][:30]
        qs = {"wanted": noul(Q_WANTED)}
        label = target.get("label", "")
        if not allow_irreversible and not IRREVERSIBLE.search(label):
            qs["irreversible"] = noul(Q_IRREV)
        if controls:
            qs["pending"] = noul(Q_PENDING)
        # the goal itself asks for this ("remove Charlie" and a "Remove selected" button): that is the user's yes
        asked = bool({m.group(1).lower() for m in IRREVERSIBLE.finditer(label)} & _words(goal))
        irreversible = not allow_irreversible and not asked and bool(IRREVERSIBLE.search(label))
        res = self._ask({"goal": goal, "element": describe(target), "controls": controls,
                         "window": {"app": obs["app"], "title": obs["title"]}}, qs)
        if "irreversible" in qs:
            irreversible = res["irreversible"].noul >= 0.6 and not asked
        self.button_log.append({"goal": goal, "element": describe(target), **{k: round(res[k].noul, 3) for k in qs}})
        wanted = res["wanted"].noul >= WANTED_MIN
        return irreversible, ("pending" not in qs or res["pending"].noul < 0.8), wanted

    def _pick_target(self, state, els, options, extra, pseudo=None):
        """Too many controls for one Choice: pick 1-2 groups by their label summaries, then the control."""
        pseudo = PSEUDO if pseudo is None else pseudo
        size = -(-len(els) // (self.cap - len(pseudo)))
        groups = [els[i:i + size] for i in range(0, len(els), size)]
        gopts = {f"g{n}": "controls: " + " | ".join((e["label"] or e["kind"])[:24] for e in g)[:600] for n, g in enumerate(groups)}
        gp, ans = self._choose(state, Q_GROUP, {**gopts, **pseudo}, extra)
        top = max(gp, key=gp.get)
        if top in PSEUDO:
            return gp, ans
        picked, mass = [], 0.0
        for g in sorted((g for g in gp if g not in PSEUDO), key=gp.get, reverse=True):
            picked.append(int(g[1:]))
            mass += gp[g]
            if mass >= 0.85 or len(picked) == 2:
                break
        cand = {str(e["i"]): options[str(e["i"])] for n in picked for e in groups[n]}
        slim = {"task": state["task"], "window": {k: state["window"][k] for k in ("app", "title", "dialogs")}}
        target_p, _ = self._choose(slim, Q_TARGET, cand)
        return target_p, ans

    def _pick_value(self, goal: str, field: dict, values: dict, used: set[str]) -> str | None:
        # A value whose key says it is secret (a password, a token, a PIN) goes only into a password field, and that is
        # _autofill's job after the user's yes: never into a field Jev picks (live, 2026-10-08: with every value used, the
        # next field Jev picked, the user name, was given the password). A value already used goes again only into a field
        # whose label names it ("Confirm email").
        pool = {k: v for k, v in values.items() if not SECRET.search(k)}
        left = [k for k in pool if k not in used] or [k for k in pool if field_matches(k, field)]
        if not left:
            return None
        if len(left) == 1:
            return left[0]
        shown = {k: "(secret)" if SECRET.search(k) else values[k] for k in left}
        probs, _ = self._choose({"goal": goal, "field": describe(field), "values": shown}, Q_VALUE,
                                {k: f"value '{k}'" for k in left})
        return max(probs, key=probs.get)

    def _autofill(self, obs: dict, values: dict, used: set[str], hwnd: int, allow: bool = False) -> list[str]:
        """Values that obviously belong in a field (its label says so) go in without asking Jev."""
        filled = []
        for key, val in values.items():
            if key in used:
                continue
            fields = [e for e in obs["elements"] if e["kind"] in FIELDS and not e["disabled"]
                      and field_matches(key, e) and (e.get("value") or "").lower() != norm_text(val).lower()[:80]]
            if fields:
                e = fields[0]
                r = self._type_into(e, hwnd, val, allow_overwrite=allow)
                used.add(key)
                if r.get("secret"):   # (the text itself is never written anywhere the worker or the log could read it)
                    filled.append(f"typed a password into {describe(e)} (value '{key}'; not shown, cannot be read back)")
                    continue
                filled.append(f"typed into {describe(e)} (value '{key}')" + (
                    " (read back OK)" if r.get("verified") else " (could NOT confirm it landed)" if r.get("ok") else
                    f" failed: {r.get('note')}"))
                continue
            menus = [e for e in obs["elements"] if e["kind"] == "dropdown" and not e["disabled"] and field_matches(key, e)
                     and not _same_text(e.get("value", ""), val)]
            if len(menus) == 1 and not SECRET.search(key):
                try:
                    filled.append(self._pick_option({"task": {"goal": f"set {key} to {val}"}}, menus[0], hwnd, [val]) + f" (value '{key}')")
                except Exception as ex:  # noqa: BLE001
                    filled.append(f"could not set dropdown '{menus[0]['label']}': {str(ex).splitlines()[0][:80]}")
                used.add(key)
                continue
            toggles = [e for e in obs["elements"] if e["kind"] in ("radio", "checkbox", "list item") and _same_text(e["label"], val)]
            if len(toggles) == 1 and not SECRET.search(key):
                if not toggles[0].get("checked") and not toggles[0].get("selected"):
                    try:
                        filled.append(self._press(toggles[0], hwnd) + f" (value '{key}')")
                    except Exception as ex:  # noqa: BLE001
                        filled.append(f"could not set {describe(toggles[0])}: {str(ex)[:80]}")
                used.add(key)
        return filled

    # ------------------------------------------------------------------ direct actions (when Jev is stuck)
    def _window_for_direct(self, window) -> dict:
        if window is not None:
            w = self.resolve(window)
        else:
            w = next((x for x in self.windows() if x["id"] == self._hwnd), None) or \
                next((x for x in self.windows() if x["foreground"]), None)
        if w is None:
            raise LookupError(f"no window matches {window!r}: desktop.windows() lists what is open")
        return w

    def elements(self, match: str | None = None, window=None) -> list[str]:
        w = self._window_for_direct(window)
        obs = self.observe(w)
        self._remember(obs)
        rows = [f"[{e['i']}] {describe(e)}" for e in obs["elements"]]
        if match:
            words = [x for x in re.split(r"\W+", match.lower()) if x]
            rows = [r for r in rows if all(x in r.lower() for x in words)] or [r for r in rows if any(x in r.lower() for x in words)]
        return rows[:120]

    def _find(self, i: int, window=None) -> tuple[dict, dict]:
        """Control number i of the last look, found again in the window as it is now (numbers change with it)."""
        if window is not None:   # the worker names the window it means: that is the one whose numbering it saw
            w = self.resolve(window)
            if w is not None and w["id"] != self._hwnd:
                raise LookupError(f"the numbers you have are from another window: call desktop.elements(window={window!r}) first")
        old = next((e for e in self._elements if e["i"] == i), None)
        if old is None:
            raise LookupError("no control with that number: call desktop.elements() again")
        w = self.resolve(self._hwnd)
        if w is None:
            raise LookupError("that window is gone: call desktop.windows()")
        obs = self.observe(w)
        cand = [e for e in obs["elements"] if e["kind"] == old["kind"] and e["label"] == old["label"]]
        if not cand:
            raise LookupError(f"the {describe(old)} is no longer there: call desktop.elements() again")
        e = min(cand, key=lambda c: abs(c["rect"][0] - old["rect"][0]) + abs(c["rect"][1] - old["rect"][1]))
        return e, obs

    def _direct(self, status_obs: dict, label: str, before: str, w: dict, note: str = "") -> dict:
        cur = self.resolve(w["id"])
        try:
            obs = self.observe(cur) if cur else None
        except LookupError:
            obs = None
        return {"status": "done", "changed": obs is None or obs["sig"] != before, "action": label, "note": note,
                "page": self._view(obs, 1200) if obs else "(the window closed)"}

    def click(self, i: int, allow_irreversible: bool = False, double: bool = False, right: bool = False, window=None) -> dict:
        e, obs = self._find(i, window)
        if not allow_irreversible and IRREVERSIBLE.search(e["label"]):
            return {"status": "needs_confirmation", "note": f"{describe(e)} looks hard to undo: ask_user first, then "
                    "click(i, allow_irreversible=True) if the user asked for it"}
        w = self.resolve(self._hwnd)
        label = self._press(e, w["id"], mouse=double or right, double=double, right=right)
        return self._direct(obs, label, obs["sig"], w)

    def fill(self, i: int, text: str, submit: bool = False, replace: bool = True, allow_irreversible: bool = False,
             window=None) -> dict:
        e, obs = self._find(i, window)
        if e["kind"] not in FIELDS:
            return {"status": "error", "note": f"{describe(e)} is not a text field"}
        w = self.resolve(self._hwnd)
        try:
            r = self._type_into(e, w["id"], text, replace, allow_irreversible)
        except Overwrite as ex:
            return {"status": "needs_confirmation", "note": str(ex)}
        except SecretDeclined as ex:
            return {"status": "needs_user", "note": str(ex)}
        if not r["ok"]:
            return {"status": "needs_user", "note": r["note"]}
        label = (f"typed a password into {describe(e)} (not shown, cannot be read back)" if r.get("secret") else
                 f"typed into {describe(e)}" + (" (read back OK)" if r["verified"] else
                                                " (could NOT confirm it landed)" if r["verified"] is False else ""))
        if submit:
            label += "; " + self._submit(e, w["id"], text)
        out = self._direct(obs, label, obs["sig"], w)
        out["verified"] = r["verified"]
        return out

    def select(self, i: int, option: str, window=None) -> dict:
        e, obs = self._find(i, window)
        w = self.resolve(self._hwnd)
        label = self._pick_option({"task": {"goal": f"choose {option}"}}, e, w["id"], [option])
        return self._direct(obs, label, obs["sig"], w)

    def key(self, combo: str, window=None, times: int = 1, allow_irreversible: bool = False) -> dict:
        combo = re.sub(r"\s*\+\s*", "+", combo.strip().lower())
        if combo in RISKY_KEYS and not allow_irreversible:
            return {"status": "needs_confirmation", "note": f"{combo} can close or lose something: ask_user first"}
        w = self._window_for_direct(window)
        obs = self.observe(w)
        try:
            label = self._keys(combo, w["id"], times)
        except RuntimeError as ex:
            return {"status": "needs_user", "note": str(ex)}
        time.sleep(0.3)
        return self._direct(obs, label, obs["sig"], w)

    def read(self, question: str | None = None, window=None, max_chars: int = 4000) -> str:
        w = self._window_for_direct(window)
        obs = self.observe(w)
        fields = [describe(e) for e in obs["elements"] if e["kind"] in FIELDS | TOGGLES]
        text = f"{obs['title']}\n{obs['text']}"
        if question and len(text) > 700:
            text = "\n".join(pick_passages(self, self.cap, question, text))
        out = text[:max_chars]
        if not question and fields:
            out += "\n\nControls:\n" + "\n".join(fields[:40])
        return out

    def screenshot(self, name: str = "window.png", window=None) -> str:
        w = self._window_for_direct(window) if window is not None or self._hwnd else None
        rect = None
        if w is not None:
            self.ui.focus(w["id"])
            time.sleep(0.2)
            rect = next((x["rect"] for x in self.ui.windows() if x["hwnd"] == w["id"]), None)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        return self.ui.screenshot(str(ARTIFACTS / name), rect)

    def close(self) -> None:
        pass


# ------------------------------------------------------------------ module API (one session, one thread)
_session: Session | None = None
_thread = None
_host = None   # set by the worker's REPL: calls into the harness (here: its model, to look at a screenshot)
_looks = 0


def _run(fn, *args, **kwargs):
    """Run fn(session, ...) on the tool's own thread: COM objects belong to the thread that made them."""
    global _thread
    if _thread is None:
        from concurrent.futures import ThreadPoolExecutor
        _thread = ThreadPoolExecutor(1, thread_name_prefix="jevh-desktop")
    return _thread.submit(lambda: fn(session(), *args, **kwargs)).result()


def session() -> Session:
    global _session
    if _session is None:
        _session = Session()
    return _session


def windows(tabs: bool = False) -> list[dict]:
    """What is open, front to back: [{id, app, title, foreground, minimized, parked, tabs?}]. tabs=True lists each
    browser window's tabs. `parked` windows are off-screen (a docked browser kept for a session)."""
    return _run(Session.windows, tabs)


def where(question: str) -> dict:
    """Jev picks the open window (or browser tab) that matches `question` -> {status, window, tab, p}."""
    return _run(Session.where, question)


def focus(window) -> dict:
    return _run(Session.focus, window)


def open(app: str, wait: float = 10.0) -> dict:   # noqa: A001 - the tool's own verb
    return _run(Session.open, app, wait)


def do(goal: str, window=None, values: dict | None = None, max_actions: int = 8, allow_irreversible: bool = False) -> dict:
    return _run(Session.do, goal, window, values, max_actions, allow_irreversible)


def read(question: str | None = None, window=None, max_chars: int = 4000) -> str:
    return _run(Session.read, question, window, max_chars)


def elements(match: str | None = None, window=None) -> list[str]:
    return _run(Session.elements, match, window)


def click(i: int, allow_irreversible: bool = False, double: bool = False, right: bool = False, window=None) -> dict:
    return _run(Session.click, i, allow_irreversible, double, right, window)


def fill(i: int, text: str, submit: bool = False, replace: bool = True, allow_irreversible: bool = False, window=None) -> dict:
    return _run(Session.fill, i, text, submit, replace, allow_irreversible, window)


def select(i: int, option: str, window=None) -> dict:
    return _run(Session.select, i, option, window)


def key(combo: str, window=None, times: int = 1, allow_irreversible: bool = False) -> dict:
    return _run(Session.key, combo, window, times, allow_irreversible)


def screenshot(name: str = "window.png", window=None) -> str:
    return _run(Session.screenshot, name, window)


def look(question: str, window=None) -> dict:
    """What the window looks like, answered by your own model from a screenshot (a chart, a canvas, an image).
    -> {answer, screenshot}. A model call with an image: use read()/elements() when the text holds the answer."""
    global _looks
    if _host is None:
        raise RuntimeError("desktop.look() works inside a worker's REPL (it asks the worker's model)")
    _looks += 1
    shot = screenshot(f"desktop-look-{_looks:02d}.png", window)
    answer = _host("look", shot, "Look at the image and answer right away - no deliberation or double-checking, just read it "
                   "off. It is a screenshot of a window on the desktop: answer only from what it shows, exactly (numbers, "
                   "labels, positions), and say so if something can't be seen.\n\n" + question)
    return {"answer": answer, "screenshot": shot}


def close() -> None:
    global _session
    _session = None


def _snapshot_cli(argv: list[str]) -> None:
    """python -m jevh.tools.desktop snapshot <window> [out.json]: save a window's controls as an eval case source."""
    import json
    s = Session(jev=object())
    w = s.resolve(argv[0])
    if not w:
        raise SystemExit(f"no window matches {argv[0]!r}: " + "; ".join(window_line(x) for x in s.windows()))
    obs = s.observe(w)
    snap = {"app": obs["app"], "title": obs["title"], "layers": obs["layers"], "text": obs["text"],
            "elements": [{k: v for k, v in e.items() if k not in ("raw", "hwnd", "rect")} for e in obs["elements"]]}
    out = json.dumps(snap, indent=1, ensure_ascii=False)
    if len(argv) > 1:
        Path(argv[1]).write_text(out, encoding="utf-8")
        print(f"saved {len(snap['elements'])} controls of {obs['title']!r} to {argv[1]}")
    else:
        print(out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else "windows"), sys.argv[2:]
    if cmd == "snapshot" and args:
        _snapshot_cli(args)
    else:
        for w in Session(jev=object()).windows(tabs=True):
            print(window_line(w))
