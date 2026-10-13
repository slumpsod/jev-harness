"""What the orb's chat shows and does, without a screen: the text being typed in the bar, and the speech bubble's content
(waiting, a reply, a question) as the conversation changes. jevh/orb.py draws it (jevh/orb_chat_art.py) and feeds it keys,
clicks and the conversation's state, which the app's hidden chat page reports (jevh/web/static/orb.html, `update`).
"""
from __future__ import annotations

ENTER, BACKSPACE, ESC, CTRL_V, CTRL_A, CTRL_BACKSPACE = 13, 8, 27, 22, 1, 127
VK_LEFT, VK_RIGHT, VK_HOME, VK_END, VK_DELETE = 0x25, 0x27, 0x24, 0x23, 0x2E


class ChatState:
    def __init__(self):
        self.text, self.caret = "", 0
        self.all_selected = False
        self.question: dict | None = None     # the assistant is asking: {id, question, options, secret}
        self.awaiting: str | None = None      # what was just sent, until its answer comes
        self.bubble: dict | None = None       # what the speech bubble says (orb_chat_art.draw_bubble's model)
        self.bubble_open = False
        self.shown_id: int | None = None      # the newest turn whose answer was shown (None: not known yet)
        self.busy = False

    # ------------------------------------------------------------------ typing
    @property
    def secret(self) -> bool:
        return bool(self.question and self.question.get("secret"))

    def insert(self, s: str) -> None:
        s = s.replace("\r", "").replace("\n", " ")
        if self.all_selected:
            self.text, self.caret, self.all_selected = "", 0, False
        self.text = self.text[:self.caret] + s + self.text[self.caret:]
        self.caret += len(s)

    def char(self, ch: int):
        """A WM_CHAR. -> an action for the window: ("send", text) | ("answer", id, text) | ("close",) | ("paste",) | None."""
        if ch == ENTER:
            text = self.text.strip()
            if not text:
                return None
            self.text, self.caret, self.all_selected = "", 0, False
            if self.question:
                return ("answer", self.question["id"], text)
            return ("send", text)
        if ch == ESC:
            return ("close",)
        if ch == CTRL_V:
            return ("paste",)
        if ch == CTRL_A:
            self.all_selected = bool(self.text)
            return None
        if ch == BACKSPACE:
            if self.all_selected:
                self.text, self.caret, self.all_selected = "", 0, False
            elif self.caret:
                self.text, self.caret = self.text[:self.caret - 1] + self.text[self.caret:], self.caret - 1
            return None
        if ch == CTRL_BACKSPACE:   # a word
            left = self.text[:self.caret].rstrip()
            cut = len(left) - len(left.split(" ")[-1]) if left else 0
            self.text, self.caret = self.text[:cut] + self.text[self.caret:], cut
            return None
        if ch >= 32:
            self.insert(chr(ch))
        return None

    def key(self, vk: int) -> None:
        self.all_selected = False
        if vk == VK_LEFT:
            self.caret = max(0, self.caret - 1)
        elif vk == VK_RIGHT:
            self.caret = min(len(self.text), self.caret + 1)
        elif vk == VK_HOME:
            self.caret = 0
        elif vk == VK_END:
            self.caret = len(self.text)
        elif vk == VK_DELETE:
            self.text = self.text[:self.caret] + self.text[self.caret + 1:]

    # ------------------------------------------------------------------ the bubble
    def sent(self, text: str) -> None:
        """A message went: the bubble says it is being worked on, at once."""
        self.awaiting = text
        self.bubble = {"kind": "waiting", "status": "Working", "you": text}
        self.bubble_open = True

    def answered(self) -> None:
        self.question = None
        self.bubble = {"kind": "waiting", "status": "Working", "you": self.awaiting}

    def update(self, state: dict) -> bool:
        """The conversation's state (orb.html): {busy, persona, last: {id, message, answer, status, persona}, question}.
        -> True when the bubble should pop out (a new reply or a question), even with the bar closed."""
        self.busy = bool(state.get("busy"))
        q = state.get("question")
        last = state.get("last") or {}
        if self.shown_id is None:   # the first look: what is already answered is not news
            self.shown_id = last.get("id") or 0 if last.get("status") != "running" else (last.get("id") or 1) - 1
        if q:
            new = not self.question or self.question.get("id") != q.get("id")
            self.question = q
            self.bubble = {"kind": "question", "text": q.get("question", ""), "options": q.get("options") or [],
                           "id": q.get("id")}
            if new:
                self.text, self.caret = "", 0
                self.bubble_open = True
            return new
        self.question = None
        if last.get("status") == "running" or self.busy:
            if self.awaiting is not None or (self.bubble and self.bubble.get("kind") == "waiting"):
                who = last.get("persona") or ""
                self.bubble = {"kind": "waiting", "status": f"Working ({who.replace('_', ' ')})" if who else "Working",
                               "you": self.awaiting or last.get("message")}
            return False
        lid = last.get("id") or 0
        if last.get("answer") and lid > (self.shown_id or 0):
            self.bubble = {"kind": "reply", "text": last["answer"], "you": last.get("message"),
                           "persona": last.get("persona"), "id": lid}
            self.shown_id, self.awaiting = lid, None
            self.bubble_open = True
            return True
        return False

    def dismiss(self) -> None:
        self.bubble_open = False
