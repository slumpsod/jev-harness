"""Jev-driven browser, pre-imported as `browser` for personas that list it in `tools`.

The worker says what outcome it wants, one step at a time; Jev picks every click and field,
so the worker's model never reads a page snapshot:

    browser.goto("https://en.wikipedia.org/wiki/Coffee")
    r = browser.do("open the article about espresso")          # -> {"status": "done", ...}
    r = browser.do("sign in", values={"email": e, "password": pw})
    browser.read("what year was the espresso machine patented?")  # only the passages that answer it
    browser.screenshot("page.png")                               # saved as an artifact

do() statuses: done, likely_done, needs_confirmation (an order/payment/send/delete is next:
repeat with allow_irreversible=True only if the user asked for it), needs_value (a field
needs text you didn't give), needs_login, ambiguous, error (the page shows an error),
blocked, stuck, max_actions. Anything but done/likely_done: look at `page` in the result,
then rephrase the goal, split it into smaller steps, or report back.

Design adapted from jev-browser (MIT, Ying-Kai Liao): the planner states outcomes, Jev
decides, Playwright acts. Differences, measured on evals/browser_cases.py with local semif
(whose cost grows with every question asked): the action kind follows from the element
(field -> type, dropdown -> select, else click) instead of a separate question; values are
typed into fields whose label matches them by plain code; the irreversible check runs only
before clicking a button. Environment: JEVH_TOOLS_JEV=local|cloud (see _pick.py),
JEVH_BROWSER_HEADED=1 to watch it (the element Jev picked is outlined and held for
JEVH_BROWSER_PAUSE seconds, default 0.8), JEVH_BROWSER_PROFILE=<dir> to keep logins.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path

from jevh.decide import noul
from jevh.tools._pick import choose, client_from_env, pick_passages

ARTIFACTS = Path(os.environ.get("JEVH_ARTIFACTS", "artifacts"))
TEXT_CHARS = 2000          # page text per decision; semif's latency grows with state size
MAX_ELEMENTS = 300
SECRET = re.compile(r"pass|secret|token|pin|cvv|cvc|card|ssn|otp|code", re.I)
IRREVERSIBLE = re.compile(r"\b(place|confirm|submit)\s+(order|payment|purchase)|\b(buy|pay|purchase|"
                          r"delete|remove account|send|publish|post|transfer|subscribe|donate)\b", re.I)
# putting something in a cart or on a list charges and sends nothing: never ask about undoing it
REVERSIBLE = re.compile(r"\s*(add\s+(it\s+|this\s+|all\s+\d+\s+)?to\s+(the\s+|your\s+|my\s+)?(shopping\s+)?"
                        r"(cart|bag|basket|trolley|list|wish\s*list|registry|favou?rites|compare)|save\s+for\s+later)\b",
                        re.I)
FIELDS = {"text field", "search box", "password"}
TOGGLES = {"checkbox", "radio", "dropdown"}
WANTED_MIN = float(os.environ.get("JEVH_BROWSER_WANTED", "0.4"))
SUBMIT_WORDS = re.compile(r"\b(save|submit|continue|apply|confirm|send|proceed|update|place)\b", re.I)
SEARCH_KEYS = {"query", "q", "search", "term", "terms", "keywords", "keyword"}

PAGE_JS = r"""() => {
  const SEL = 'a[href],button,input,select,textarea,summary,[role=button],[role=link],[role=tab],' +
    '[role=menuitem],[role=checkbox],[role=radio],[role=switch],[role=option],[role=combobox],' +
    '[role=searchbox],[role=textbox],[contenteditable=""],[contenteditable=true],[onclick],label[for]';
  const clean = t => (t || '').replace(/\s+/g, ' ').trim();
  const pageW = document.documentElement.scrollWidth;
  // parked off-page (skip links, keyboard-shortcut panels): not something a person can see or click
  const offPage = r => r.right <= 0 || r.left >= pageW || r.bottom + scrollY <= 0;
  const vis = e => {
    const r = e.getBoundingClientRect();
    if (r.width < 2 || r.height < 2 || offPage(r)) return false;
    const s = getComputedStyle(e);
    if (s.visibility === 'hidden' || s.display === 'none') return false;
    if (parseFloat(s.opacity || '1') > 0.05) return true;
    // a transparent input/button laid over styled text is the real control (Amazon's a-button)
    const p = e.parentElement;
    return ['INPUT', 'BUTTON'].includes(e.tagName) && !!p && parseFloat(getComputedStyle(p).opacity || '1') > 0.05 &&
      clean(p.innerText) !== '';
  };
  const ownText = l => { const c = l.cloneNode(true); c.querySelectorAll('select,input,textarea').forEach(x => x.remove()); return c.textContent; };
  const labelOf = e => {
    let t = e.getAttribute('aria-label');
    const by = e.getAttribute('aria-labelledby');
    if (!t && by) t = by.split(' ').map(id => clean(document.getElementById(id)?.innerText)).join(' ');
    if (!t && e.labels && e.labels.length) t = [...e.labels].map(ownText).join(' ');
    if (!t && e.tagName === 'LABEL') t = ownText(e);
    if (!t && !['INPUT', 'SELECT', 'TEXTAREA'].includes(e.tagName)) t = e.innerText;
    if (!t) t = e.getAttribute('placeholder') || e.getAttribute('title') || e.querySelector?.('img[alt]')?.alt;
    if (!t && e.tagName === 'INPUT' && ['submit', 'button'].includes(e.type)) t = e.value;
    if (!t) t = e.getAttribute('name') || e.id;
    return clean(t).slice(0, 80);
  };
  const kindOf = e => {
    const tag = e.tagName.toLowerCase(), type = (e.getAttribute('type') || '').toLowerCase();
    const role = e.getAttribute('role');
    if (tag === 'select' || (role === 'combobox' && tag !== 'input')) return 'dropdown';
    if (tag === 'textarea' || e.isContentEditable) return 'text field';
    if (tag === 'input') {
      if (['checkbox', 'radio', 'password'].includes(type)) return type;
      if (type === 'file') return 'file input';
      if (['submit', 'button', 'reset', 'image'].includes(type)) return 'button';
      if (type === 'search' || role === 'searchbox' || /search|query|^q$/i.test(e.name || '')) return 'search box';
      return 'text field';
    }
    if (role === 'searchbox') return 'search box';
    if (role === 'textbox') return 'text field';
    if (role === 'checkbox' || role === 'switch') return 'checkbox';
    if (role === 'radio') return 'radio';
    if (role === 'tab') return 'tab';
    if (tag === 'a' || role === 'link') return 'link';
    return 'button';
  };
  document.querySelectorAll('[data-jevh-i]').forEach(e => e.removeAttribute('data-jevh-i'));
  const out = [], seen = new Set();
  for (const e of document.querySelectorAll(SEL)) {
    if (seen.has(e)) continue;
    seen.add(e);
    if (e.tagName === 'INPUT' && e.type === 'hidden') continue;
    if (e.tagName === 'LABEL') { const c = e.control; if (!c || vis(c)) continue; }
    if (!vis(e)) continue;
    const kind = e.tagName === 'LABEL' ? kindOf(e.control) : kindOf(e);
    const r = e.getBoundingClientRect();
    const el = {i: out.length, kind, label: labelOf(e), below: r.top > innerHeight};
    if (e.disabled) el.disabled = true;
    if (['text field', 'search box'].includes(kind)) el.value = clean(e.value ?? e.innerText).slice(0, 60);
    if (kind === 'password') el.value = e.value ? '(filled)' : '';
    if (['checkbox', 'radio'].includes(kind)) el.checked = !!(e.checked ?? e.control?.checked ?? e.getAttribute('aria-checked') === 'true');
    if (kind === 'dropdown' && e.options) {
      el.value = clean(e.options[e.selectedIndex]?.text);
      el.options = [...e.options].slice(0, 40).map(o => clean(o.text));
    }
    if (kind === 'link') el.href = (e.getAttribute('href') || '').slice(0, 60);
    if (e.name) el.name = e.name;
    e.setAttribute('data-jevh-i', out.length);
    out.push(el);
    if (out.length >= %d) break;
  }
  const notices = [...document.querySelectorAll('dialog[open],[role=dialog],[role=alertdialog],[role=alert],[role=status],[aria-live]')]
    .filter(vis).map(d => clean(d.innerText).slice(0, 300)).filter(Boolean);
  // Page text as a person sees it: not dropdown option lists (a store's 60 search departments
  // used to fill the whole decision window), hidden or off-page text, or the watch overlay.
  // One line per block element; view_start marks where the visible part of the page begins.
  const SKIP = 'select,script,style,noscript,template,[aria-hidden="true"],.jevh-watch';
  const blockOf = new Map(), okOf = new Map(), lines = [];
  const block = el => {
    if (blockOf.has(el)) return blockOf.get(el);
    let x = el;
    while (x && x !== document.body && getComputedStyle(x).display.startsWith('inline')) x = x.parentElement;
    blockOf.set(el, x || document.body);
    return x || document.body;
  };
  const shown = el => {
    if (!okOf.has(el)) {
      const r = el.getBoundingClientRect();
      okOf.set(el, !el.closest(SKIP) && !offPage(r) && (r.width > 0 || r.height > 0) &&
        el.checkVisibility({visibilityProperty: true, checkVisibilityCSS: true}));
    }
    return okOf.get(el);
  };
  let cur = null;
  if (document.body) {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const t = clean(n.data), p = n.parentElement;
      if (!t || !p || !shown(p)) continue;
      const b = block(p);
      if (cur && cur.b === b) { cur.t += ' ' + t; continue; }
      cur = {b, t, bottom: b.getBoundingClientRect().bottom};
      lines.push(cur);
    }
  }
  let text = '', view_start = -1;
  for (const l of lines) {
    if (view_start < 0 && l.bottom > 0) view_start = text.length;
    text += l.t + '\n';
  }
  return {url: location.href, title: document.title, elements: out, text, view_start: Math.max(view_start, 0), notices};
}""" % MAX_ELEMENTS

PSEUDO = {"scroll": "scroll down to reveal more of the page",
          "back": "go back to the previous page",
          "none": "no action: the goal is already achieved on this page, or nothing on it can help"}

Q_DONE = ("Has `task.goal` been fully achieved on `page`, including every part it asks for (such as saving, "
          "submitting or opening a particular page)? Judge from `page` and `task.history`.")
Q_ERROR = ("Does `page` show an error or rejection message (e.g. invalid credentials, a validation error, "
           "not found) caused by the actions in `task.history`?")
Q_BLOCKED = ("Is there something on `page` that stops progress on `task.goal` and cannot be handled by "
             "clicking or typing (a captcha, a paywall, an access-denied page)?")
Q_LOGIN = "Is `page` a sign-in or sign-up screen, or asking the user to log in, before `task.goal` can continue?"
Q_IRREV = ("Would clicking `element`, by itself and with no further confirmation step, place an order, take a "
           "payment, send a message, publish, or delete something?")
Q_PENDING = "Does `controls` show a setting that still differs from what `goal` asks for?"
# evals/button_intent_cases.py: 9/11 unwanted clicks caught on tuning, 4/6 held out, no wanted click
# blocked (plateau 0.3-0.6). Misses share a word with the goal ("open the cart" vs "Add to cart").
Q_WANTED = "Does `goal` ask for what clicking `element` would do?"
Q_TARGET = "Which element of `page` should the next action toward `task.goal` act on?"
Q_GROUP = "Which group of `page` elements contains the one the next action toward `task.goal` should act on?"
Q_VALUE = "Which of `values` should be typed into `field` to make progress on `goal`?"

# watch mode only: a caption bar with the goal and Jev's pick, and an outline on the element
SHOW_JS = r"""([sel, caption]) => {
  document.querySelectorAll('.jevh-watch').forEach(x => x.remove());
  const bar = document.createElement('div');
  bar.className = 'jevh-watch';
  bar.textContent = caption;
  bar.style.cssText = 'position:fixed;left:0;right:0;bottom:0;z-index:2147483647;padding:8px 14px;' +
    'background:rgba(20,20,30,.88);color:#fff;font:14px/1.4 system-ui,sans-serif;pointer-events:none';
  document.documentElement.appendChild(bar);
  const e = sel && document.querySelector(sel);
  if (!e) return;
  const r = e.getBoundingClientRect();
  const box = document.createElement('div');
  box.className = 'jevh-watch';
  box.style.cssText = `position:fixed;left:${r.left - 4}px;top:${r.top - 4}px;width:${r.width + 8}px;` +
    `height:${r.height + 8}px;border:3px solid #ff3d7f;border-radius:6px;z-index:2147483646;` +
    'pointer-events:none;box-shadow:0 0 0 4px rgba(255,61,127,.25)';
  document.documentElement.appendChild(box);
}"""


def describe(e: dict) -> str:
    s = f"{e['kind']} '{e['label']}'" if e.get("label") else e["kind"]
    if e.get("href") and not e["href"].startswith(("javascript", "#")):
        s += f" (opens {e['href']})"
    if e.get("value"):
        s += f" = '{e['value']}'"
    if "checked" in e:
        s += " (checked)" if e["checked"] else " (unchecked)"
    if e.get("options"):
        s += " options: " + " | ".join(e["options"][:12])
    if e.get("disabled"):
        s += " (disabled)"
    if e.get("below"):
        s += " (below the fold)"
    return s


def _words(s: str) -> set[str]:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s or "")
    return {w for w in re.split(r"[^a-z0-9]+", s.lower()) if len(w) >= 2}


def field_matches(key: str, e: dict) -> bool:
    """Does value `key` obviously belong in field `e`? (email -> 'Email', password -> the password box)."""
    if e["kind"] == "password":
        return bool(SECRET.search(key))
    if e["kind"] == "search box":
        return key.lower() in SEARCH_KEYS
    k = _words(key) - {"the", "your"}
    names = _words(e.get("label", "")) | _words(e.get("name", ""))
    return bool(k) and k <= names


class Session:
    """One Chromium page plus the Jev loop. Module-level functions use a shared session."""

    def __init__(self, headed: bool | None = None, profile: str | None = None, jev=None, cap: int | None = None):
        from playwright.sync_api import sync_playwright
        if jev is None:
            jev, default_cap = client_from_env()
            cap = cap or default_cap
        self.jev, self.cap = jev, cap or 16
        headed = os.environ.get("JEVH_BROWSER_HEADED") == "1" if headed is None else headed
        profile = profile or os.environ.get("JEVH_BROWSER_PROFILE")
        self._pw = sync_playwright().start()
        viewport = {"width": 1280, "height": 900}
        if profile:
            self.context = self._pw.chromium.launch_persistent_context(profile, headless=not headed, viewport=viewport)
            self._browser = None
        else:
            self._browser = self._pw.chromium.launch(headless=not headed)
            self.context = self._browser.new_context(viewport=viewport)
        self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
        # watching: outline what Jev picked and hold it on screen before acting
        self.headed = headed
        self.pause = float(os.environ.get("JEVH_BROWSER_PAUSE", "0.8" if headed else "0"))
        self.jev_calls = 0
        self.button_log: list[dict] = []   # every button check's scores, for evals and debugging
        self.jev_secs = 0.0

    # ------------------------------------------------------------------ plumbing
    def ask(self, state, questions):
        """Every Jev request goes through here (also from choose()/pick_passages(), which get
        the session as their client), so calls and time are counted per do()."""
        t = time.monotonic()
        self.jev_calls += 1
        try:
            return self.jev.ask(state, questions)
        finally:
            self.jev_secs += time.monotonic() - t

    _ask = ask

    def _choose(self, state, instructions, options, extra=None):
        return choose(self, state, instructions, options, self.cap, extra)

    def _settle(self) -> None:
        if len(self.context.pages) > 1 and self.context.pages[-1] is not self.page:
            self.page = self.context.pages[-1]  # a link opened a new tab: follow it
        for state, ms in (("domcontentloaded", 5000), ("networkidle", 800)):
            try:
                self.page.wait_for_load_state(state, timeout=ms)
            except Exception:  # noqa: BLE001 - a busy page is still usable
                pass
        time.sleep(0.1)

    def _show(self, i: int | None, caption: str) -> None:
        """Watch mode: outline element `i` (None: just the caption) and pause so a person can follow."""
        if not self.pause:
            return
        try:
            self.page.evaluate(SHOW_JS, [f'[data-jevh-i="{i}"]' if i is not None else "", caption])
        except Exception:  # noqa: BLE001 - cosmetic; never let it break an action
            pass
        time.sleep(self.pause)

    def observe(self) -> dict:
        self._settle()
        try:
            return self.page.evaluate(PAGE_JS)
        except Exception:  # noqa: BLE001 - navigation raced the script; try once more
            time.sleep(0.5)
            return self.page.evaluate(PAGE_JS)

    @staticmethod
    def _signature(obs: dict) -> str:
        # the whole page: a store's cart sheet or "1 item in cart" can change far below the top
        controls = [(e.get("label"), e.get("value"), e.get("checked")) for e in obs["elements"]]
        return hashlib.sha1(repr((obs["url"], obs["text"], controls, obs["notices"])).encode()).hexdigest()

    @staticmethod
    def _view(obs: dict, n: int = TEXT_CHARS) -> str:
        """The part of the page text a person is looking at: from the top of the viewport on
        (after a scroll, not the page header again), filled up from above near the page end."""
        text = obs["text"]
        start = max(0, min(obs.get("view_start", 0), len(text) - n))
        return text[start:start + n]

    # ------------------------------------------------------------------ public API
    def goto(self, url: str) -> dict:
        if not re.match(r"^[a-z]+://", url):
            url = "https://" + url
        self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        obs = self.observe()
        return {"url": obs["url"], "title": obs["title"], "page": self._view(obs, 1200)}

    def read(self, question: str | None = None, max_chars: int = 4000) -> str:
        """Page text. With a question, only the passages Jev finds relevant to it."""
        text = self.observe()["text"]
        if not question:
            return text[:max_chars]
        return "\n---\n".join(pick_passages(self, self.cap, question, text))[:max_chars]

    def elements(self) -> list[str]:
        return [f"[{e['i']}] {describe(e)}" for e in self.observe()["elements"]]

    def screenshot(self, name: str = "page.png", full_page: bool = False) -> str:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        dest = ARTIFACTS / Path(name).name
        if dest.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            dest = dest.with_name(dest.name + ".png")  # Playwright picks the format from the extension
        if self.pause:  # keep the watch-mode outline out of saved screenshots
            self.page.evaluate("() => document.querySelectorAll('.jevh-watch').forEach(x => x.remove())")
        self.page.screenshot(path=str(dest), full_page=full_page)
        return str(dest)

    def close(self) -> None:
        try:
            self.context.close()
            if self._browser:
                self._browser.close()
        finally:
            self._pw.stop()

    def do(self, goal: str, values: dict | None = None, max_actions: int = 8,
           allow_irreversible: bool = False) -> dict:
        """Reach one observable outcome on the current page. See the module docstring."""
        started, calls0, secs0 = time.monotonic(), self.jev_calls, self.jev_secs
        values = {str(k): str(v) for k, v in (values or {}).items()}
        used: set[str] = set()
        history: list[str] = []
        banned: set[int] = set()      # no effect, or a toggle already set during this goal
        deferred: set[int] = set()    # a submit button held back until the form matches the goal
        no_effect = 0
        last_change = "nothing yet"

        def result(status: str, obs: dict, note: str = "", probs: dict | None = None) -> dict:
            return {"status": status, "goal": goal, "url": obs["url"], "title": obs["title"],
                    "actions": history, "note": note, "page": self._view(obs, 1200),
                    "p": {k: round(v, 2) for k, v in (probs or {}).items()},
                    "jev_calls": self.jev_calls - calls0, "jev_seconds": round(self.jev_secs - secs0, 1),
                    "seconds": round(time.monotonic() - started, 1)}

        obs = self.observe()
        actions = 0
        for _ in range(2 * max_actions + 2):
            # values that obviously belong in an empty field go in without asking Jev
            filled = self._autofill(obs, values, used)
            if filled:
                history.append("filled " + ", ".join(filled))
                actions += 1
                obs = self.observe()
                continue

            els = [e for e in obs["elements"] if e["i"] not in banned | deferred and not e.get("disabled")]
            controls = [describe(e) for e in obs["elements"] if e["kind"] in FIELDS | TOGGLES][:30]
            state = {"task": {"goal": goal, "history": history[-6:] or ["(no actions yet)"], "last_change": last_change},
                     "page": {"url": obs["url"], "title": obs["title"], "text": self._view(obs),
                              "controls": controls, "notices": obs["notices"]}}
            extra = {"done": noul(Q_DONE)}
            if history:
                extra["error"] = noul(Q_ERROR)
            else:
                extra["blocked"] = noul(Q_BLOCKED)
                if not values:
                    extra["login"] = noul(Q_LOGIN)
            options = {str(e["i"]): describe(e) for e in els}
            if len(options) + len(PSEUDO) <= self.cap:
                target_p, ans = self._choose(state, Q_TARGET, {**options, **PSEUDO}, extra)
            else:
                target_p, ans = self._pick_target(state, els, options, extra)
            probs = {k: ans[k].noul for k in ("done", "error", "blocked", "login") if k in ans}
            best = max(target_p, key=target_p.get)

            # "... and save/continue": setting the control isn't the end; semif's done check misses that
            if SUBMIT_WORDS.search(goal) and not any(re.match(r"clicked (button|link)|.*pressed Enter", h)
                                                     for h in history):
                probs["done"] = min(probs["done"], 0.29)
            if probs["done"] >= 0.9:
                return result("done", obs, probs=probs)
            if probs.get("error", 0) >= 0.7:
                return result("error", obs, "the page shows an error after the last action", probs)
            if probs.get("blocked", 0) >= 0.85:
                return result("blocked", obs, "something on the page blocks progress", probs)
            if probs.get("login", 0) >= 0.7:
                return result("needs_login", obs, "sign-in required: call again with values= for the form", probs)
            if best == "none":
                if probs["done"] >= 0.5:
                    return result("done", obs, probs=probs)
                if probs["done"] >= 0.3:
                    return result("likely_done", obs, "probably achieved; check `page`", probs)
                return result("stuck", obs, "no action on this page seems to help", probs)
            if actions >= max_actions:
                return result("max_actions", obs, f"stopped after {max_actions} actions", probs)
            if target_p[best] < 0.2:
                by_i = {str(e["i"]): e for e in obs["elements"]}
                top = sorted(target_p, key=target_p.get, reverse=True)[:3]
                return result("ambiguous", obs, "unsure which element: " + "; ".join(
                    f"{describe(by_i[k]) if k in by_i else k} ({target_p[k]:.2f})" for k in top), probs)

            target = None if best in PSEUDO else next(e for e in obs["elements"] if str(e["i"]) == best)
            tool, value, pick = self._tool_for(target, best), None, None
            if tool in ("type", "press_enter"):
                pick = self._pick_value(goal, target, values, used)
                value = values.get(pick) if pick else None
                if value is None and not target.get("value"):
                    return result("needs_value", obs, f"{describe(target)} needs text: call again with "
                                  f"values={{'<name>': '<text>'}}", probs)
                if value is None:
                    tool = "press_enter"   # already filled: submit it
            elif tool == "select":
                value = next((v for v in values.values() if any(v.lower() == o.lower() for o in target.get("options", []))),
                             None) or self._pick_option(state, target)
            elif tool == "click" and target["kind"] == "button":
                irreversible, ready, wanted = self._check_button(goal, target, obs, allow_irreversible)
                if not wanted:
                    deferred.add(target["i"])   # a side effect nobody asked for (add to cart, share): pick again
                    last_change = f"not clicking {describe(target)}: it isn't part of the goal"
                    continue
                if not ready:
                    deferred.add(target["i"])   # the form isn't filled in as asked yet: pick something else
                    last_change = f"not clicking {describe(target)} yet: the form doesn't match the goal"
                    continue
                if irreversible:
                    return result("needs_confirmation", obs, f"next: click {describe(target)}, which looks hard "
                                  f"to undo; call again with allow_irreversible=True if the user wants it", probs)
            elif tool == "click" and not allow_irreversible and IRREVERSIBLE.search(target.get("label", "")):
                return result("needs_confirmation", obs, f"next: click {describe(target)}, which looks hard "
                              f"to undo; call again with allow_irreversible=True if the user wants it", probs)

            before = self._signature(obs)
            self._caption = (f"Jev ({target_p[best]:.2f}): {tool.replace('_', ' ')} "
                             f"{describe(target) if target else ''}   |   goal: {goal}")
            try:
                label = self._act(tool, target, value)
            except Exception as e:  # noqa: BLE001 - a failed action is information, not a crash
                label = f"tried to {tool} {describe(target) if target else ''} but it failed ({str(e).splitlines()[0][:80]})"
            if pick:
                used.add(pick)
                label += f" (value '{pick}')"
            history.append(label)
            actions += 1
            if target is not None and target["kind"] in TOGGLES:
                banned.add(target["i"])     # set it once; don't flip it back
            obs = self.observe()
            if self._signature(obs) == before:
                no_effect += 1
                last_change = "the last action changed nothing on the page"
                if target is not None:
                    banned.add(target["i"])
                if no_effect >= 3:
                    return result("stuck", obs, "the last actions had no visible effect", probs)
            else:
                no_effect = 0
                deferred.clear()
                last_change = f"after '{label}' the page changed" + (
                    f" to {obs['url']}" if obs["url"] != state["page"]["url"] else "")
        return result("max_actions", obs, f"stopped after {max_actions} actions")

    # ------------------------------------------------------------------ decision helpers
    def _pick_target(self, state, els, options, extra):
        """Too many elements for one Choice: pick 1-2 groups by their label summaries, then the element."""
        size = -(-len(els) // (self.cap - len(PSEUDO)))
        groups = [els[i:i + size] for i in range(0, len(els), size)]
        gopts = {f"g{n}": "elements: " + " | ".join((e["label"] or e["kind"])[:24] for e in g)[:600]
                 for n, g in enumerate(groups)}
        gp, ans = self._choose(state, Q_GROUP, {**gopts, **PSEUDO}, extra)
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
        slim = {"task": state["task"], "page": {k: state["page"][k] for k in ("url", "title", "notices")}}
        target_p, _ = self._choose(slim, Q_TARGET, cand)   # the group already used the page text
        return target_p, ans

    @staticmethod
    def _tool_for(target: dict | None, best: str) -> str:
        if target is None:
            return best                       # scroll / back
        if target["kind"] in FIELDS:
            return "press_enter" if target["kind"] == "search box" else "type"
        if target["kind"] == "dropdown" and target.get("options"):
            return "select"
        return "click"

    def _autofill(self, obs: dict, values: dict, used: set[str]) -> list[str]:
        filled = []
        for key, val in values.items():
            if key in used:
                continue
            fields = [e for e in obs["elements"] if e["kind"] in FIELDS and not e.get("value") and field_matches(key, e)]
            if not fields:
                continue
            for e in fields if fields[0]["kind"] == "password" else fields[:1]:
                self._show(e["i"], f"Jev: fill {describe(e)} with '{key}' (label match)")
                self.page.locator(f'[data-jevh-i="{e["i"]}"]').first.fill(val, timeout=5000)
                filled.append(f"{e['kind']} '{e['label']}' (value '{key}')")
            used.add(key)
        return filled

    def _pick_value(self, goal: str, field: dict, values: dict, used: set[str]) -> str | None:
        left = [k for k in values if k not in used] or list(values)
        if not left:
            return None
        if len(left) == 1:
            return left[0]
        shown = {k: "(secret)" if SECRET.search(k) else values[k] for k in left}
        probs, _ = self._choose({"goal": goal, "field": describe(field), "values": shown}, Q_VALUE,
                                {k: f"value '{k}'" for k in left})
        return max(probs, key=probs.get)

    def _pick_option(self, state, target):
        opts = target.get("options") or []
        if not opts:
            return None
        probs, _ = self._choose(state, f"Which option of dropdown '{target['label']}' does `task.goal` call for?",
                                {f"o{i}": o or "(blank)" for i, o in enumerate(opts)})
        return opts[int(max(probs, key=probs.get)[1:])]

    def _check_button(self, goal: str, target: dict, obs: dict, allow_irreversible: bool) -> tuple[bool, bool, bool]:
        """(irreversible, ready, wanted) for clicking a button, in one small call. `ready`: a form
        on the page already holds what the goal asks for, so a submit/continue isn't premature.
        `wanted`: the click serves the goal (not "Add to cart" when asked to open the cart)."""
        controls = [describe(e) for e in obs["elements"] if e["kind"] in FIELDS | TOGGLES][:30]
        qs = {"wanted": noul(Q_WANTED)}
        label = target.get("label", "")
        if not allow_irreversible and not IRREVERSIBLE.search(label) and not REVERSIBLE.match(label):
            qs["irreversible"] = noul(Q_IRREV)   # semif gave Amazon's "Add to cart" 0.73: those skip it
        if controls:
            qs["pending"] = noul(Q_PENDING)
        irreversible = not allow_irreversible and bool(IRREVERSIBLE.search(target.get("label", "")))
        res = self._ask({"goal": goal, "element": describe(target), "controls": controls,
                         "page": {"title": obs["title"], "url": obs["url"]}}, qs)
        if "irreversible" in qs:
            irreversible = res["irreversible"].noul >= 0.6
        self.button_log.append({"goal": goal, "element": describe(target),
                                **{k: round(res[k].noul, 3) for k in qs}})
        # measured: 0.97-0.99 when a requested setting is still unset, 0.32-0.44 once it is
        return (irreversible, ("pending" not in qs or res["pending"].noul < 0.8),
                res["wanted"].noul >= WANTED_MIN)

    def _act(self, tool: str, target: dict | None, value: str | None) -> str:
        page = self.page
        caption = getattr(self, "_caption", f"Jev: {tool}")
        if target is None:
            self._show(None, caption)
        if tool == "scroll":
            page.mouse.wheel(0, 700)
            return "scrolled down"
        if tool == "back":
            page.go_back(timeout=15000)
            return "went back"
        loc = page.locator(f'[data-jevh-i="{target["i"]}"]').first
        what = f"{target['kind']} '{target['label']}'"
        if loc.count() == 0:  # the page re-rendered while Jev was deciding: find the same element again
            fresh = next((e for e in self.observe()["elements"]
                          if e["kind"] == target["kind"] and e["label"] == target["label"]), None)
            if fresh is None:
                raise RuntimeError("the element disappeared")
            loc = page.locator(f'[data-jevh-i="{fresh["i"]}"]').first
        try:
            loc.scroll_into_view_if_needed(timeout=2000)
        except Exception:  # noqa: BLE001
            pass
        if self.pause:
            self._show(int(loc.get_attribute("data-jevh-i", timeout=2000) or target["i"]), caption)
        if tool == "select":
            opts = target.get("options", [])
            match = next((o for o in opts if o.lower() == str(value).lower()), None) or \
                next((o for o in opts if str(value).lower() in o.lower()), value)
            loc.select_option(label=match, timeout=5000)
            return f"selected '{match}' in {what}"
        if tool in ("type", "press_enter"):
            if value is not None:
                loc.fill(value, timeout=5000)
            if tool == "press_enter":
                loc.press("Enter", timeout=5000)
                return f"typed into {what} and pressed Enter" if value is not None else f"pressed Enter in {what}"
            return f"typed into {what}"
        try:
            loc.click(timeout=3000)
        except Exception:  # noqa: BLE001 - covered or animating: a DOM click still reaches the handler
            loc.evaluate("el => el.click()", timeout=3000)
        return f"clicked {what}"


_session: Session | None = None


def session() -> Session:
    global _session
    if _session is None:
        _session = Session()
    return _session


def goto(url: str) -> dict:
    return session().goto(url)


def do(goal: str, values: dict | None = None, max_actions: int = 8, allow_irreversible: bool = False) -> dict:
    return session().do(goal, values, max_actions, allow_irreversible)


def read(question: str | None = None, max_chars: int = 4000) -> str:
    return session().read(question, max_chars)


def elements() -> list[str]:
    return session().elements()


def screenshot(name: str = "page.png", full_page: bool = False) -> str:
    return session().screenshot(name, full_page)


def url() -> str:
    return session().page.url


def visible() -> bool:
    """Is the browser a window the user can see (Watch browser on)? Then they can take over,
    e.g. to enter payment details the worker must never type."""
    return _session.headed if _session is not None else os.environ.get("JEVH_BROWSER_HEADED") == "1"


def close() -> None:
    global _session
    if _session is not None:
        _session.close()
        _session = None
