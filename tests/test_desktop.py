"""The desktop tool without Windows: a fake UI backend (a tiny app model that reacts to clicks, typing and keys) and a
scripted Jev. These prove the code paths and the guards; whether Jev picks the right control in real apps is what
evals/desktop_cases.py and evals/bench/desktop_tasks.py measure live."""
import pytest

from conftest import FakeSystemOne, choice_ans, noul_ans
from jevh.tools import desktop as D
from jevh.tools._winui import parse_combo

AV = {"invoke": False, "value": False, "toggle": False, "select": False, "expand": False, "scroll": False, "text": False}


def R(ct, name, rect, **kw):
    """A raw UIA element, as _winui.snapshot returns it."""
    r = {"el": None, "hwnd": 1, "ct": ct, "name": name, "rect": rect, "off": False, "enabled": True, "cls": "", "aid": "",
         "help": "", "pw": False, "value": None, "readonly": False, "toggle": None, "selected": None, "expand": None,
         "avail": dict(AV)}
    av = kw.pop("avail", {})
    r.update(kw)
    r["avail"].update(av)
    return r


WIN = (0, 0, 800, 600)


# ------------------------------------------------------------------ pure pieces
def test_parse_combo():
    assert parse_combo("ctrl+n") == [0x11, 0x4E]
    assert parse_combo("Ctrl + Shift + Enter") == [0x11, 0x10, 0x0D]
    assert parse_combo("f5") == [0x74]
    with pytest.raises(ValueError):
        parse_combo("a+b")


def test_build_elements_kinds_labels_and_noise():
    raws = [
        R(50020, "Email address", (20, 60, 120, 80)),
        R(50004, "", (140, 60, 400, 80), value="a@b.c", avail={"value": True}),                   # named by the text beside it
        R(50003, "Country", (140, 100, 340, 120), value="Japan", avail={"expand": True}),
        R(50004, "Country", (141, 101, 300, 119), value="Japan", readonly=True),                    # the dropdown's own edit box
        R(50000, "Open", (310, 101, 338, 119)),                                                      # ...and its button
        R(50000, "Submit", (140, 200, 240, 230), avail={"invoke": True}),
        R(50000, "Close", (760, 2, 800, 30), avail={"invoke": True}),                                # the window's own button
        R(50000, "Line down", (780, 400, 800, 420), enabled=False),                                  # a scroll bar's
        R(50004, "Hidden", (0, 0, 10, 10), pw=False, **{"off": True}),
        R(50004, "Password", (140, 140, 400, 160), pw=True, value="x", avail={"value": True}),
        R(50002, "Subscribe", (140, 170, 300, 190), toggle=1, avail={"toggle": True}),
        R(50000, "Same", (10, 300, 90, 320)), R(50000, "Same", (10, 340, 90, 360)),
    ]
    els, texts = D.build_elements(raws, WIN)
    by = {(e["kind"], e["label"]): e for e in els}
    assert ("text field", "Email address") in by and by[("text field", "Email address")]["value"] == "a@b.c"
    assert ("dropdown", "Country") in by and by[("dropdown", "Country")]["value"] == "Japan"
    assert sum(1 for e in els if e["label"] in ("Country", "Open")) == 1       # parts of the dropdown are not controls
    assert by[("button", "Close")].get("chrome") and by[("button", "Line down")].get("chrome")
    assert by[("password", "Password")]["value"] == "(hidden)"
    assert by[("checkbox", "Subscribe")]["checked"] is True
    assert not any(e["label"] == "Hidden" for e in els)                        # off-screen
    same = [e for e in els if e["label"] == "Same"]
    assert len(same) == 2 and same[0]["context"] != same[1]["context"]          # told apart by where they are
    assert "Email address" in texts


def test_unsupported_states_are_not_believed():
    r = R(50000, "Go", (0, 0, 50, 20), toggle=0, selected=False, expand=0, avail={"invoke": True})
    e = D.build_elements([r], WIN)[0][0]
    assert "checked" not in e and not e.get("selected")


def test_goal_helpers():
    assert D.goal_values("open the dialog, set Display name to Ana, turn on Dark mode and press OK") == {"Display name": "Ana"}
    assert D.goal_values("type hello into Notes") == {}
    assert D.final_click("turn on Dark mode and press OK") == "OK"
    assert D.final_click("fill the form, then click the Submit button") == "Submit"
    assert D.final_click("click Increment") is None
    assert D.nav_target("open the List tab") == "List" and D.nav_target("switch to Composer") == "Composer"
    assert D.nav_target("open the settings dialog, set x to y and press OK") is None
    assert D.KEY_GOAL.match("press Ctrl+N in the Claude app").group(1) == "Ctrl+N"
    assert D.KEY_GOAL.match("press the Delete button") is None and D.KEY_GOAL.match("hit enter").group(1) == "enter"
    assert D.OPEN_GOAL.match("open the Notepad app").group(1) == "Notepad"


def test_named_gone():
    obs = {"elements": [{"label": "Alpha"}, {"label": "Delta"}], "text": "Items: 2"}
    assert D.named_gone("remove Charlie from the list", {"charlie", "alpha"}, obs)
    assert not D.named_gone("remove Alpha from the list", {"charlie", "alpha"}, obs)
    assert not D.named_gone("open Charlie", {"charlie"}, obs)


# ------------------------------------------------------------------ a fake desktop
class FakeUI:
    """One app window ('Lab') whose controls are the app's own dict; clicks and keys act on it like a real UI would."""

    def __init__(self, app):
        self.app, self.front, self.focused = app, True, None
        self.log: list[str] = []
        self.clipboard = None
        self.selected_all = False
        self.windows_list = [self._win(1, "Lab", "lab", True)]

    @staticmethod
    def _win(hwnd, title, exe, fg, rect=(0, 0, 800, 600), owner=0, **kw):
        return {"hwnd": hwnd, "pid": hwnd, "exe": exe, "title": title, "cls": "Fake", "rect": rect, "minimized": False,
                "foreground": fg, "owner": owner, "tool": False, "enabled": True, **kw}

    def windows(self):
        return self.windows_list

    def snapshot(self, hwnd):
        return self.app.raws()

    def in_front(self, hwnd):
        return self.front

    def focus(self, hwnd):
        self.log.append("focus")
        return self.front

    def foreground(self):
        return 1

    def _find(self, raw):
        return raw["name"]

    def invoke(self, raw, select_first=False):
        self.log.append(f"invoke {raw['name']}")
        self.app.click(raw["name"])
        return "invoke"

    def click_at(self, x, y, button="left", double=False, move_only=False):
        self.log.append(f"click {x},{y}")
        for r in self.app.raws():
            if r["rect"][0] <= x <= r["rect"][2] and r["rect"][1] <= y <= r["rect"][3] and r["ct"] != 50033:
                self.app.click(r["name"])
                return

    def set_focus(self, raw):
        self.focused = raw["name"]

    def get_value(self, raw):
        return self.app.fields.get(raw["name"])

    def set_value(self, raw, text):
        self.log.append("set_value")
        self.app.fields[raw["name"]] = text

    def keys(self, combo, times=1, delay=0):
        self.log.append(f"keys {combo}")
        if combo == "ctrl+a":
            self.selected_all = True
        elif combo == "delete":
            self.app.fields[self.focused] = ""
        elif combo == "enter":
            self.app.enter(self.focused)
        elif combo == "escape":
            self.app.escape()

    def type_text(self, text, delay=0):
        self.log.append(f"type {text!r}")
        cur = "" if self.selected_all else self.app.fields.get(self.focused, "")
        self.selected_all = False
        self.app.fields[self.focused] = self.app.accept(self.focused, cur + text)

    def paste(self, text):
        self.log.append("paste")
        self.type_text(text)

    def expand(self, raw):
        return False

    def scroll(self, raw, hwnd, dy):
        return True

    def screenshot(self, path, rect=None):
        return path


class FormApp:
    """Fields Name and Notes (a document), a Send composer with a slash-autocomplete, buttons Submit / Delete all."""

    def __init__(self, autocomplete=False):
        self.fields = {"Name": "", "Composer": "", "Notes": ""}
        self.clicks, self.sent, self.suggest, self.autocomplete = [], [], False, autocomplete
        self.password = False
        self.fields["Secret"] = ""

    def raws(self):
        extra = [R(50004, "Password", (100, 260, 400, 280), pw=True, value=self.fields["Secret"], avail={"value": True})]             if self.password else []
        return extra + [
            R(50004, "Name", (100, 20, 400, 40), value=self.fields["Name"], avail={"value": True}),
            R(50004, "Composer", (100, 60, 400, 80), value=self.fields["Composer"], avail={"value": True}),
            R(50030, "Notes", (100, 100, 400, 200), value=self.fields["Notes"], avail={"value": True, "text": True}),
            R(50000, "Submit", (100, 220, 200, 250), avail={"invoke": True}),
            R(50000, "Delete all", (220, 220, 320, 250), avail={"invoke": True}),
            R(50020, "Status: " + ("submitted" if "Submit" in self.clicks else "idle"), (100, 300, 300, 320)),
        ]

    def click(self, name):
        self.clicks.append(name)

    def accept(self, field, text):
        if field == "Composer" and self.autocomplete and text.startswith("/"):
            self.suggest = True
        return text

    def enter(self, field):
        if field != "Composer":
            return
        if self.suggest:   # the autocomplete takes the Enter
            self.fields["Composer"] = "remote-control "
            self.suggest = False
        else:
            self.sent.append(self.fields["Composer"])
            self.fields["Composer"] = ""

    def escape(self):
        self.suggest = False


def jev_for(*picks, done_when=None, irreversible=False):
    """Picks the control whose description contains the next fragment in `picks` that has not been acted on yet."""
    def rules(k, q, s):
        if k == "done":
            return noul_ans(0.97 if done_when and done_when(s) else 0.02)
        if k == "irreversible":
            return noul_ans(0.9 if irreversible else 0.02)
        if k == "wanted":
            return noul_ans(0.95)
        if q["type"] == "choice":
            hist = " ".join(s.get("task", {}).get("history", [])) if isinstance(s, dict) else ""
            for want in picks:
                hit = next((key for key, d in q["criteria"].items() if d and want in d), None)
                if hit and want not in hist:
                    return choice_ans(hit, 0.9, q["criteria"])
            return choice_ans(next(iter(q["criteria"])), 0.5, q["criteria"])
        return noul_ans(0.02)
    return FakeSystemOne(rules)


def session(app=None, jev=None):
    app = app or FormApp()
    ui = FakeUI(app)
    s = D.Session(ui=ui, jev=jev or FakeSystemOne(), cap=16)
    s.pause = 0
    return s, ui, app


# ------------------------------------------------------------------ windows
def test_windows_hides_noise_and_parks_offscreen():
    s, ui, _ = session()
    ui.windows_list += [
        FakeUI._win(2, "Dialog", "lab", False, owner=1),                                    # owned: not a window of its own
        FakeUI._win(3, "Program Manager", "explorer", False),
        FakeUI._win(4, "Docked", "chrome", False, rect=(-32768, -32768, -31000, -31000)),
        FakeUI._win(5, "Calculator", "ApplicationFrameHost", False),
        FakeUI._win(6, "Cursor", "cua-driver", False),
    ]
    ws = s.windows()
    assert [w["title"] for w in ws] == ["Lab", "Calculator", "Docked"]                      # parked last
    assert ws[1]["app"] == "Calculator" and ws[2]["parked"]
    assert s.resolve("lab")["id"] == 1 and s.resolve("calc")["title"] == "Calculator" and s.resolve("nothing") is None


# ------------------------------------------------------------------ typing
def test_fill_is_read_back_and_never_types_into_a_password():
    s, ui, app = session()
    s.elements()
    i = next(e["i"] for e in s._elements if e["label"] == "Name")
    out = s.fill(i, "Ana Lopez")
    assert out["verified"] is True and app.fields["Name"] == "Ana Lopez" and "ctrl+a" in " ".join(ui.log)
    app.password = True
    s.elements()
    j = next(e["i"] for e in s._elements if e["kind"] == "password")
    assert s.fill(j, "hunter2")["status"] == "needs_user" and app.fields["Secret"] == ""


def test_text_the_tool_did_not_write_is_not_replaced_without_a_yes():
    s, ui, app = session()
    app.fields["Notes"] = "my unsaved note"
    s.elements()
    i = next(e["i"] for e in s._elements if e["label"] == "Notes")
    out = s.fill(i, "overwritten")
    assert out["status"] == "needs_confirmation" and app.fields["Notes"] == "my unsaved note"
    assert "typed" not in " ".join(ui.log) and "paste" not in ui.log
    assert s.fill(i, " more", replace=False, allow_irreversible=True)["status"] == "done"   # (append is still a write)
    s.elements()
    j = next(e["i"] for e in s._elements if e["label"] == "Notes")
    assert s.fill(j, "rewritten", allow_irreversible=True)["verified"] is True
    s.elements()
    k = next(e["i"] for e in s._elements if e["label"] == "Notes")
    assert s.fill(k, "again")["status"] == "done"      # its own text may be replaced


def test_front_tries_again_once_after_a_moment():
    s, ui, app = session()
    ui.front = False
    answers = iter([False, True])
    ui.focus = lambda hwnd: next(answers)
    assert s._front(1) is True


def test_a_window_that_will_not_come_to_front_gets_no_keys():
    s, ui, app = session()
    ui.front = False
    s.elements()
    i = next(e["i"] for e in s._elements if e["label"] == "Name")
    assert s.fill(i, "x")["status"] == "needs_user" and app.fields["Name"] == ""
    assert s.key("ctrl+n", window="Lab")["status"] == "needs_user" and not any(l.startswith("keys") for l in ui.log)


def test_enter_taken_by_an_autocomplete_is_closed_and_the_text_sent_again():
    s, ui, app = session(FormApp(autocomplete=True))
    s.elements()
    i = next(e["i"] for e in s._elements if e["label"] == "Composer")
    out = s.fill(i, "/rc", submit=True)
    assert app.sent == ["/rc"], (app.sent, ui.log)
    assert "emptied (sent)" in out["action"]


def test_risky_keys_and_irreversible_clicks_ask_first():
    s, ui, app = session()
    assert s.key("alt+f4", window="Lab")["status"] == "needs_confirmation"
    s.elements()
    i = next(e["i"] for e in s._elements if e["label"] == "Delete all")
    assert s.click(i)["status"] == "needs_confirmation" and "Delete all" not in app.clicks
    assert s.click(i, allow_irreversible=True)["status"] == "done" and "Delete all" in app.clicks


# ------------------------------------------------------------------ do()
def test_do_fills_by_label_and_clicks_what_jev_picks():
    jev = jev_for("button 'Submit'", done_when=lambda s: "submitted" in str(s.get("window", {}).get("text", "")))
    s, ui, app = session(jev=jev)
    r = s.do("fill in the name and submit", window="Lab", values={"Name": "Ana"})
    assert r["status"] == "done", r
    assert app.fields["Name"] == "Ana" and app.clicks == ["Submit"]
    assert r["actions"][0].startswith("typed into text field 'Name'") and r["jev_calls"] >= 1


def test_do_stops_before_a_hard_to_undo_click_unless_the_goal_asks_for_it():
    s, ui, app = session(jev=jev_for("button 'Delete all'"))
    r = s.do("tidy up", window="Lab")
    assert r["status"] == "needs_confirmation" and "Delete all" in r["note"] and app.clicks == []
    s2, ui2, app2 = session(jev=jev_for("button 'Delete all'"))
    r2 = s2.do("delete all the entries", window="Lab")      # the goal says "delete": that is the user's yes
    assert "Delete all" in app2.clicks and r2["status"] != "needs_confirmation"


def test_a_goal_that_ends_with_a_click_is_not_done_before_it():
    jev = jev_for("button 'Submit'", done_when=lambda s: True)   # Jev calls everything done, always
    s, ui, app = session(jev=jev)
    r = s.do("fill in the name, then press Submit", window="Lab", values={"Name": "Ana"})
    assert app.fields["Name"] == "Ana" and app.clicks == ["Submit"], (r, ui.log)
    assert r["status"] == "done" and "last step" in r["note"]


def test_key_goal_goes_to_the_front_window_and_is_gated():
    s, ui, app = session()
    r = s.do("press Ctrl+N")
    assert any(l == "keys ctrl+n" for l in ui.log) and r["status"] in ("done", "likely_done")
    assert s.do("press alt+f4")["status"] == "needs_confirmation"


def test_no_window_says_so():
    s, ui, app = session()
    r = s.do("click the Submit button", window="Nonexistent")
    assert r["status"] == "no_window" and "Nonexistent" in r["note"]


def test_where_picks_a_window_or_a_tab():
    s, ui, app = session()
    ui.windows_list.append(FakeUI._win(7, "Docs - Chrome", "chrome", False))
    s._tabs = lambda hwnd: ["Mail", "Claude Code"]
    def rules(k, q, st):
        if q["type"] == "choice":
            hit = next((key for key, d in q["criteria"].items() if d and "Claude Code" in d), None)
            return choice_ans(hit or "none", 0.9, q["criteria"])
    s.jev = FakeSystemOne(rules)
    r = s.where("the claude code tab")
    assert r["status"] == "found" and r["tab"] == "Claude Code" and r["window"]["id"] == 7


def test_a_combo_with_a_writable_edit_box_is_text_and_one_without_is_a_dropdown():
    inside = lambda r, d: (r[0] + d, r[1] + d, r[2] - d, r[3] - d)
    raws = [
        R(50003, "File name:", (100, 100, 400, 120), value="notes", avail={"value": True, "expand": True}),
        R(50004, "File name:", inside((100, 100, 400, 120), 2), value="notes", avail={"value": True}),   # writable: typed text
        R(50000, "Open", (380, 102, 398, 118)),
        R(50003, "Save as type:", (100, 140, 400, 160), value="Text files", avail={"value": True, "expand": True}),
        R(50000, "Open", (380, 142, 398, 158)),                                                           # no edit box: a list
    ]
    els = D.build_elements(raws, WIN)[0]
    assert [(e["kind"], e["label"]) for e in els] == [("text field", "File name:"), ("dropdown", "Save as type:")]


def test_a_disabled_dialog_under_another_is_not_read():
    s, ui, app = session()
    ui.windows_list += [FakeUI._win(2, "Save As", "lab", False, owner=1, enabled=False),
                        FakeUI._win(3, "Error", "lab", False, owner=2)]          # owned by the dialog, not by the window
    ui.windows_list[0]["enabled"] = False
    main, extra, modal = s._layers(s.windows()[0])
    assert [x["title"] for x in extra] == ["Error"] and modal


def test_numbers_the_worker_was_shown_survive_the_window_changing_under_them():
    s, ui, app = session(FormApp(autocomplete=True))
    shown = s.elements()
    send_i = next(e["i"] for e in s._elements if e["label"] == "Submit")
    comp_i = next(e["i"] for e in s._elements if e["label"] == "Composer")
    app.raws_orig = app.raws
    # typing "/rc" makes a suggestion list appear in front of every other control (renumbering a fresh look)
    app.raws = lambda: ([R(50007, "remote-control", (100, 90, 300, 110), avail={"select": True})] if app.suggest else []) + app.raws_orig()
    s.fill(comp_i, "/rc")
    assert app.suggest
    out = s.click(send_i)           # the number from the list the worker saw: still the Submit button
    assert "Submit" in out["action"] and app.clicks == ["Submit"], out


# ------------------------------------------------------------------ passwords: only with the user's yes
def _pw_session(answers):
    """A session whose user answers each question with the next of `answers` (None: nobody to ask)."""
    s, ui, app = session()
    app.password = True
    asked = []
    it = iter(answers)

    def ask(question, options=None):
        asked.append((question, options))
        return next(it)
    s.ask_user = ask
    return s, ui, app, asked


def _pw_index(s):
    s.elements()
    return next(e["i"] for e in s._elements if e["kind"] == "password")


def test_a_password_is_typed_when_the_user_allows_it_and_not_through_the_clipboard():
    s, ui, app, asked = _pw_session(["Allow"])
    out = s.fill(_pw_index(s), "hunter2")
    assert out["status"] == "done" and app.fields.get("Password") == "hunter2"
    assert "paste" not in ui.log                                  # (the clipboard would hold it)
    assert "hunter2" not in out["action"] and "hunter2" not in str(out["page"])
    assert len(asked) == 1 and "password" in asked[0][0] and asked[0][1][0] == "Allow"
    assert "hunter2" not in asked[0][0]                           # (the question does not repeat it)


def test_a_password_the_user_declines_is_not_typed_and_the_worker_is_told_not_to_work_around_it():
    s, ui, app, asked = _pw_session(["Don't allow - I'll type it myself"])
    out = s.fill(_pw_index(s), "hunter2")
    assert out["status"] == "needs_user" and "do not work around" in out["note"]
    assert not app.fields.get("Password") and not any(l.startswith("type") for l in ui.log)
    s2, ui2, app2, _ = _pw_session(["No - don't type the password for me, I'll enter it myself."])   # free text is a no too
    assert s2.fill(_pw_index(s2), "hunter2")["status"] == "needs_user" and not app2.fields.get("Password")


def test_a_password_with_nobody_to_ask_is_not_typed():
    s, ui, app, _ = _pw_session([None])
    out = s.fill(_pw_index(s), "hunter2")
    assert out["status"] == "needs_user" and "nobody can be asked" in out["note"] and not app.fields.get("Password")


def test_allow_for_this_task_asks_once():
    s, ui, app, asked = _pw_session(["Allow for this task"])
    i = _pw_index(s)
    assert s.fill(i, "one")["status"] == "done"
    s.elements()
    assert s.fill(_pw_index(s), "two")["status"] == "done" and app.fields["Password"] == "two" and len(asked) == 1
    s3, _, _, asked3 = _pw_session(["Allow", "Allow"])           # plain Allow is once: it asks again
    s3.fill(_pw_index(s3), "one")
    s3.fill(_pw_index(s3), "two")
    assert len(asked3) == 2


def test_do_types_a_password_value_only_after_the_user_allows_and_keeps_it_out_of_the_history():
    s, ui, app, asked = _pw_session(["Allow"])
    r = s.do("sign in", window="Lab", values={"Password": "hunter2"})
    assert app.fields.get("Password") == "hunter2" and asked
    assert "hunter2" not in str(r["actions"]) and "hunter2" not in str(r["page"]) and "hunter2" not in r["note"]
    assert any("typed a password" in a for a in r["actions"])
    s2, ui2, app2, _ = _pw_session(["Don't allow - I'll type it myself"])
    r2 = s2.do("sign in", window="Lab", values={"Password": "hunter2"})
    assert r2["status"] == "needs_user" and not app2.fields.get("Password")


def test_a_secret_value_never_goes_into_a_field_jev_picks():
    """Live: with the password typed, Jev picked the user name field next and the used password value went into it."""
    s, ui, app, asked = _pw_session(["Allow"])
    app.fields["Plain"] = ""
    raws_orig = app.raws
    app.raws = lambda: raws_orig() + [R(50004, "Username", (100, 300, 400, 320), value="", avail={"value": True})]
    s.jev = jev_for("text field 'Username'", done_when=lambda st: False)
    r = s.do("type the password into the Password field", window="Lab", values={"password": "hunter2"}, max_actions=3)
    assert app.fields.get("Password") == "hunter2"
    assert app.fields.get("Username", "") == ""                       # (the typed password did not reach another field)
    assert "hunter2" not in str(r["actions"]) and "hunter2" not in str(r["page"])


def test_a_used_value_is_not_typed_again_into_a_field_that_does_not_name_it():
    s, ui, app = session()
    field = {"kind": "text field", "label": "Notes", "value": ""}
    assert s._pick_value("g", field, {"Name": "Ana"}, used={"Name"}) is None
    assert s._pick_value("g", {"kind": "text field", "label": "Name"}, {"Name": "Ana"}, used={"Name"}) == "Name"
    assert s._pick_value("g", {"kind": "text field", "label": "Name"}, {"Password": "x"}, used=set()) is None
