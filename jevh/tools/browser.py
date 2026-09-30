"""Jev-driven browser, pre-imported as `browser` for personas that list it in `tools`.

The worker says what outcome it wants, one step at a time; Jev picks every click and field,
so the worker's model never reads a page snapshot:

    browser.goto("https://en.wikipedia.org/wiki/Coffee")
    r = browser.do("open the article about espresso")          # -> {"status": "done", ...}
    r = browser.do("sign in", values={"email": e, "password": pw})
    browser.read("what year was the espresso machine patented?")  # only the passages that answer it
    browser.screenshot("page.png")                               # saved as an artifact
    browser.elements("october"); browser.click(12)               # when Jev is stuck: the worker acts itself

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
JEVH_BROWSER_PAUSE seconds, default 0.8), JEVH_BROWSER_PROFILE=<dir> to keep logins. With
JEVH_BROWSER_DOCK=<dir> (the dashboard sets it) a watched browser has no window: it is shown,
and can be taken over, in the dashboard's Browser panel.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import socket
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
ADD_GOAL = re.compile(r"\b(add|adding|put|place|save|get|buy|order|purchase|grab|want|include|choose|select|"
                      r"pick|keep|book|reserve|\d+\s+(of|x)\b)", re.I)
FIELDS = {"text field", "search box", "password"}
TOGGLES = {"checkbox", "radio", "dropdown"}
WANTED_MIN = float(os.environ.get("JEVH_BROWSER_WANTED", "0.4"))
# browser.test's final check: both must hold (evals/expect_cases.py: 21/22 true results passed,
# 0/16 false ones; either question alone passed 1-2 false ones - "the message was sent" next to
# "Email is required", "the page shows the order history" on an order confirmation)
EXPECT_SHOWS_Q = "Does `page` show that `expect` is true?"
EXPECT_HOLDS_Q = ("Going only by what `page` shows, is `expect` true - with every detail in `expect` (numbers, "
                  "names, states) matching the page, and nothing on the page saying otherwise?")
EXPECT_SHOWS, EXPECT_HOLDS = 0.55, 0.6
EXPECT_PAGE_CHARS = 4000   # page text the check reads; longer pages: the passages about `expect`
# the user's own app: a test may submit, order or delete there without asking (that is what's tested)
LOCAL_URL = re.compile(r"^(?:file:|https?://(?:localhost|127\.0\.0\.1|\[::1\]|[^/:]+\.localhost|[^/:]+\.test)"
                       r"(?::\d+)?(?:[/?#]|$))", re.I)
# bot checks: plain code catches them on any page (Jev's "blocked" question only runs before the first action)
BOT_WALL = re.compile(r"just a moment\.\.\.|checking (if the site connection is secure|your browser)|verify(ing)? "
                      r"(you are|you're|that you are) (a )?human|are you a (human|robot)|ensuring a fair fan experience|"
                      r"pardon our interruption|press (&|and) hold|unusual traffic from your computer|"
                      r"access denied|request unsuccessful\. incapsula|captcha", re.I)
CARD_FIELD = re.compile(r"card\s*(number|no\b|#)|\bcvv|\bcvc|security code|\bexpir|\biban\b|routing|"
                        r"account number|\bssn\b|social security", re.I)
# query keys that aren't the search itself: tracking, and the hand-offs of a sign-in flow (their loss is normal)
NOT_SEARCH_KEYS = re.compile(r"^(utm_|_ga|gclid$|fbclid$|label$|aid$|sid$|srpvid$|chal_t$|force_referer$|efdco$|ssne|"
                             r"ref$|src$|source$|next$|return|redirect|continue$|token$|code$|state$|session|lang$)", re.I)
STRIPPED_NOTE = ("the site threw the search away: it sent {asked} on to {final} without {keys}. Sites do this to "
                 "turn away automated browsers (Booking.com does). Don't retry or rebuild the URL: tell the user, "
                 "say what is filled in and where, and let them finish the search in their own browser")
# links that would open a new tab open in this one: one page for the agent, and one for the dashboard's dock
SAME_TAB_JS = """document.addEventListener('click', e => {
  const a = e.target && e.target.closest && e.target.closest('a[target]');
  if (a && !['_self', '_top', '_parent'].includes(a.target) && a.href && !/^javascript:/i.test(a.getAttribute('href') || ''))
    a.target = '_self';
}, true);"""
CANVAS_SHARE = 0.12   # a canvas covering this much of the screen holds the page's real content
CANVAS_NOTE = ("the page's main content is drawn on a canvas (a game board, chart or map): it is not in the text "
               "or HTML, so don't search for it there. See it with browser.look(question); play or act with "
               "browser.press(keys) / click(i)")
WALL_WAIT = 2.0   # seconds between looks while a bot check might clear by itself (4 looks)
REFUSED = {403, 429, 435, 456}   # what bot managers answer a flagged client's API calls with
BLOCKED_NOTE = ("the site shows a bot check ({}), which the agent does not get past - not by retrying, calling "
                "the site's API or faking browser headers either. Tell the user; if browser.visible(), they can "
                "complete it in the browser window and reply when done")
FORM_SEND = re.compile(r"\b(submit|send|save|confirm|continue|proceed|place)\b", re.I)   # buttons that send a form
SEND_GOAL = re.compile(r"\b(finish|complete|book|order|register|sign\s*up|next|go on|move on)\b", re.I)
SUBMIT_WORDS = re.compile(r"\b(save|submit|continue|apply|confirm|send|proceed|update|place)\b", re.I)
QUOTED = re.compile(r"""['"“‘]([^'"”’]{1,80})['"”’]""")
SET_TO = re.compile(r"^\s*(?:set|change|update)\s+(?:the\s+|my\s+)?[\w\s-]{2,40}?\s+to\s+(\S.{0,59}?)\s*\.?\s*$", re.I)
SEARCH_KEYS = {"query", "q", "search", "term", "terms", "keywords", "keyword"}

PAGE_JS = r"""() => {
  const MAX = %d;
  const OPT = '[id*="-option-"],[class*="__option"],[role=listbox] li,li[class*="suggestion" i],' +
    '[class*="autocomplete" i] li';
  const SEL = 'a[href],button,input,select,textarea,summary,[role=button],[role=link],[role=tab],' +
    '[role=menuitem],[role=checkbox],[role=radio],[role=switch],[role=option],[role=combobox],' +
    '[role=searchbox],[role=textbox],[contenteditable=""],[contenteditable=true],[onclick],label[for],' +
    // autocomplete suggestions are often plain divs/lis (react-select, airport pickers)
    OPT;
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
    if (!t && ['INPUT', 'SELECT', 'TEXTAREA'].includes(e.tagName)) {
      // a label beside the field but not tied to it (React forms put <label>Date of Birth</label> in the
      // next column): the closest label before it within a few levels up; with a placeholder, both
      // ("Email (name@example.com)" - the placeholder alone is an example, not a name)
      let near = '';
      for (let a = e.parentElement, d = 0; a && d < 6 && !near; a = a.parentElement, d++) {
        const ls = [...a.querySelectorAll('label')].filter(l => !l.contains(e) && !l.control && clean(l.innerText) &&
          (l.compareDocumentPosition(e) & Node.DOCUMENT_POSITION_FOLLOWING));
        if (ls.length) near = clean(ls[ls.length - 1].innerText);
      }
      const ph = clean(e.getAttribute('placeholder'));
      t = near && ph && !ph.toLowerCase().includes(near.toLowerCase()) ? `${near} (${ph})` : (ph || near);
    }
    if (!t) t = e.getAttribute('placeholder') || e.getAttribute('title') || e.querySelector?.('img[alt]')?.alt;
    if (!t && e.tagName === 'INPUT' && ['submit', 'button'].includes(e.type)) t = e.value;
    if (!t) t = e.getAttribute('name') || e.id;
    // a hover-effect copy of the text ("Buy Tickets Buy Tickets") is one label
    return clean(t).replace(/^(.+) \1$/i, '$1').slice(0, 80);
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
    if (role === 'option' || e.matches(OPT)) return 'option';
    if (tag === 'a' || role === 'link') return 'link';
    return 'button';
  };
  document.querySelectorAll('[data-jevh-i]').forEach(e => e.removeAttribute('data-jevh-i'));
  const out = [], seen = new Set(), nodes = [];
  for (const e of document.querySelectorAll(SEL)) {
    if (seen.has(e)) continue;
    seen.add(e);
    if (e.tagName === 'INPUT' && e.type === 'hidden') continue;
    if (e.tagName === 'LABEL') { const c = e.control; if (!c || vis(c)) continue; }
    if (!vis(e)) continue;
    const kind = e.tagName === 'LABEL' ? kindOf(e.control) : kindOf(e);
    const r = e.getBoundingClientRect();
    const el = {i: out.length, kind, label: labelOf(e), below: r.top > innerHeight, _v: r.bottom > 0 && r.top < innerHeight};
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
    out.push(el);
    nodes.push(e);
    if (out.length >= MAX * 4) break;
  }
  // over the cap: what's on screen first (live, Booking.com: an open calendar sits at the end of the page and
  // its days were cut off by the first 300 controls in page order), then page order; numbered after
  if (out.length > MAX) {
    const keep = new Set(out.map((o, j) => j).sort((a, b) => (out[b]._v - out[a]._v) || (a - b)).slice(0, MAX));
    const o2 = out.filter((o, j) => keep.has(j)), n2 = nodes.filter((n, j) => keep.has(j));
    out.length = 0; nodes.length = 0; out.push(...o2); nodes.push(...n2);
  }
  out.forEach((o, j) => { o.i = j; delete o._v; nodes[j].setAttribute('data-jevh-i', j); });
  // identical controls (a "Choose This Flight" per table row, an "Add to cart" per product card) are
  // told apart by their own row's text: the nearest container holding only this one of them
  const key = o => o.kind + '|' + o.label, count = {};
  out.forEach(o => count[key(o)] = (count[key(o)] || 0) + 1);
  out.forEach((o, j) => {
    if (count[key(o)] < 2 || o.kind === 'dropdown') return;   // a dropdown's options already describe it
    const twins = nodes.filter((n, k) => k !== j && key(out[k]) === key(o));
    for (let a = nodes[j].parentElement, d = 0; a && a !== document.body && d < 6; a = a.parentElement, d++) {
      if (twins.some(n => a.contains(n))) break;
      const t = clean(clean(a.innerText).split(o.label).join(' '));
      if (t.length > 3) { o.context = t.slice(0, 90); break; }
    }
  });
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
  // how much of the screen a canvas covers: a game board or chart drawn there is in no text or HTML
  const shareOf = c => { const r = c.getBoundingClientRect();
    return Math.max(0, Math.min(r.right, innerWidth) - Math.max(r.left, 0)) *
           Math.max(0, Math.min(r.bottom, innerHeight) - Math.max(r.top, 0)) / (innerWidth * innerHeight); };
  const canvas = Math.max(0, ...[...document.querySelectorAll('canvas')].filter(vis).map(shareOf));
  return {url: location.href, title: document.title, elements: out, text, view_start: Math.max(view_start, 0), notices,
          canvas: Math.round(canvas * 100) / 100};
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
    if e.get("context"):
        s += f" (in: {e['context']})"
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


def corrected_url(url: str) -> str | None:
    """`url` with its host swapped for the near-identical one the request names, if there is one.
    Live, a worker typed "blavedemo.com" for the requested blazedemo.com and spent 16 steps
    debugging DNS without noticing."""
    import difflib
    host = re.sub(r"^[a-z]+://", "", url).split("/")[0].split(":")[0].lower()
    named = os.environ.get("JEVH_TASK_HOSTS", "").split()
    close = difflib.get_close_matches(host.removeprefix("www."), [h.removeprefix("www.") for h in named], 1, 0.75)
    if not close:
        return None
    return re.sub(re.escape(host), next(h for h in named if h.removeprefix("www.") == close[0]), url, count=1, flags=re.I)


def _same_text(a: str, b: str) -> bool:
    return re.sub(r"\s+", " ", a or "").strip().lower() == re.sub(r"\s+", " ", b or "").strip().lower()


DATE_IN = ["%d %B %Y", "%B %d, %Y", "%B %d %Y", "%d %b %Y", "%b %d, %Y", "%b %d %Y", "%Y-%m-%d", "%m/%d/%Y",
           "%d %B, %Y", "%A, %B %d, %Y"]
DATE_OUT = ["%Y-%m-%d", "%m/%d/%Y", "%d %b %Y", "%b %d, %Y", "%B %d, %Y", "%d/%m/%Y"]


def _date_forms(text: str) -> list[str]:
    """`text` as a date written the common ways (ISO first); [] if it isn't a date."""
    from datetime import datetime
    t = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", (text or "").strip())
    for f in DATE_IN:
        try:
            d = datetime.strptime(t, f)
        except ValueError:
            continue
        return [d.strftime(o) for o in DATE_OUT]
    return []


def dropped_search(asked: str, final: str) -> list[str]:
    """The search terms of `asked` that the site threw away on the way to `final` ([] if it kept any, or
    `asked` had fewer than two real ones). Booking.com sends a headless browser's searchresults.html?ss=
    Denver&checkin=... back to a page with no query at all - no dates, no hotels, no error text."""
    from urllib.parse import parse_qsl, urlsplit
    if _site(asked) != _site(final):
        return []
    keys = sorted({k for k, v in parse_qsl(urlsplit(asked).query) if v and not NOT_SEARCH_KEYS.match(k)})
    if len(keys) < 2:
        return []
    kept = {k for k, _ in parse_qsl(urlsplit(final).query)}
    return [] if kept & set(keys) else keys


def _site(url: str) -> str:
    """The site a URL belongs to, roughly: its last two host labels (redsky.target.com -> target.com)."""
    host = re.sub(r"^[a-z]+://", "", url or "").split("/")[0].split(":")[0].lower()
    return host if re.fullmatch(r"[\d.]+", host) else ".".join(host.split(".")[-2:])


def unresolved_hint(url: str) -> str:
    """Why goto() failed on a host with no DNS entry."""
    host = re.sub(r"^[a-z]+://", "", url).split("/")[0].split(":")[0].lower()
    return f"{host} does not exist (no such domain). Check the spelling against the URL you were given."


def _words(s: str) -> set[str]:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s or "")
    return {w for w in re.split(r"[^a-z0-9]+", s.lower()) if len(w) >= 2}


def field_matches(key: str, e: dict) -> bool:
    """Does value `key` obviously belong in field `e`? (email -> 'Email', password -> the password box)."""
    if e["kind"] == "password":
        return bool(SECRET.search(key))
    if e["kind"] == "search box":
        return key.lower() in SEARCH_KEYS
    k = _words(key) - {"the", "your", "my", "select", "choose", "enter", "pick"}   # "Select State" -> state
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
        # docked (the dashboard's Browser panel): no window; the dashboard attaches over CDP on a local
        # port to mirror the page and pass on the user's clicks and typing
        dock, args, port = os.environ.get("JEVH_BROWSER_DOCK") if headed else None, [], 0
        if dock:
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            args = [f"--remote-debugging-port={port}"]
        window = headed and not dock
        if profile:
            self.context = self._pw.chromium.launch_persistent_context(profile, headless=not window, viewport=viewport,
                                                                       args=args)
            self._browser = None
        else:
            self._browser = self._pw.chromium.launch(headless=not window, args=args)
            self.context = self._browser.new_context(viewport=viewport)
        self._dock_file = Path(dock) / f"{os.getpid()}.json" if dock else None
        if self._dock_file:
            self._dock_file.parent.mkdir(parents=True, exist_ok=True)
            self._dock_file.write_text(json.dumps({"port": port, "pid": os.getpid(), "started": time.time()}))
        self.context.add_init_script(SAME_TAB_JS)
        self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
        self.navs: list[str] = []   # the pages (main-frame documents) asked for since the last action
        # script errors on every page (test() reports them): uncaught exceptions and console.error
        self.js_errors: list[str] = []
        self.console_errors: list[str] = []
        self.refused: list[tuple[str, int]] = []   # (site, HTTP status) of each site's refused data requests
        self.context.on("page", self._listen)
        for p in self.context.pages:
            self._listen(p)
        # watching (a window, or docked in the dashboard - both let the user take over): outline what
        # Jev picked and hold it on screen before acting
        self.headed = headed
        self.pause = float(os.environ.get("JEVH_BROWSER_PAUSE", "0.8" if headed else "0"))
        self.jev_calls = 0
        self.button_log: list[dict] = []   # every button check's scores, for evals and debugging
        self.jev_secs = 0.0

    # ------------------------------------------------------------------ plumbing
    def _listen(self, page) -> None:
        page.on("pageerror", lambda e: self.js_errors.append(str(e).splitlines()[0][:300]))
        page.on("console", lambda m: self.console_errors.append(m.text[:300])
                if m.type == "error" and "favicon" not in m.text.lower() else None)
        page.on("response", self._on_response)
        page.on("request", lambda r: self.navs.append(r.url)
                if r.is_navigation_request() and r.frame == r.frame.page.main_frame else None)

    def _dropped(self, final: str, also: str | None = None) -> str:
        """STRIPPED_NOTE if a page asked for since the last action (or `also`) had its search thrown away."""
        for asked in ([also] if also else []) + self.navs:
            keys = dropped_search(asked, final)
            if keys:
                return STRIPPED_NOTE.format(asked=asked[:160], final=final[:120], keys=", ".join(keys))
        return ""

    def _on_response(self, r) -> None:
        """The page's own data requests turned away (live: Target answered its product search API with
        HTTP 435, and the page said "We couldn't find a match" - which reads like a real empty result)."""
        try:
            if r.status in REFUSED and r.request.resource_type in ("xhr", "fetch") and \
                    _site(r.url) == _site(r.frame.page.url):   # its own APIs; not a refused ad tracker
                self.refused.append((_site(r.url), r.status))
        except Exception:  # noqa: BLE001 - a listener never breaks the page
            pass

    def _refusing(self) -> bool:
        """The site turns away enough of its own data requests to call it blocked (live: Target's search
        said "We couldn't find a match" and the worker, told of the refusals, tried 5 more searches)."""
        return len(self._refused_here()) >= 3

    def _refused_here(self) -> list[int]:
        """Refusals by the site now open (a blocked site doesn't taint the next one)."""
        here = _site(self.page.url)
        return [code for site, code in self.refused if site == here]

    def _refused_note(self) -> str:
        codes = self._refused_here()
        if len(codes) < 2:
            return ""
        return (f"the site refused {len(codes)} of its own data requests (HTTP {', '.join(str(c) for c in sorted(set(codes)))}): it is likely "
                f"blocking automated browsers, so parts of the page (results, prices) may be missing - "
                f"don't report them as absent, and don't work around the block (its API, faked headers): "
                f"tell the user")

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

    def _look(self) -> dict:
        try:
            return self.page.evaluate(PAGE_JS)
        except Exception:  # noqa: BLE001 - navigation raced the script; try once more
            time.sleep(0.5)
            self._settle()
            return self.page.evaluate(PAGE_JS)

    def observe(self) -> dict:
        self._settle()
        obs = self._look()
        # a page just loaded may still be building itself (Booking.com: 150 elements, then 219 with the
        # search form 1.5 s later - a destination value had no field yet, and Jev clicked a promo link)
        if obs["url"] != getattr(self, "_last_url", None):
            for _ in range(8):
                self.page.wait_for_timeout(400)
                again = self._look()
                grown = len(again["elements"]) > len(obs["elements"]) or len(again["text"]) > len(obs["text"]) + 200
                obs = again
                if not grown:
                    break
        self._last_url = obs["url"]
        self._last_obs = obs   # the numbering click()/fill()/select() refer to
        return obs

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
        self.refused, self.navs = [], []
        fixed, resp = None, None
        try:
            try:
                resp = self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            except Exception as e:  # noqa: BLE001
                # the page we are leaving was still redirecting itself and cut this navigation off (live:
                # Booking.com's results page, "net::ERR_ABORTED"): once more, after it settles
                if "ERR_ABORTED" not in str(e) and "interrupted by another navigation" not in str(e):
                    raise
                self.page.wait_for_timeout(800)
                resp = self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        except Exception as e:  # noqa: BLE001
            if "ERR_NAME_NOT_RESOLVED" not in str(e):
                raise
            # a near miss of a site the request names is a copying slip: go there (live, told the right
            # spelling, the worker typed "blaxedemo" and then "blazemo")
            fixed = corrected_url(url)
            if not fixed:
                raise RuntimeError(unresolved_hint(url)) from None
            # Chromium moves on to its own error page after a failed lookup, and that navigation cut off
            # an immediate retry ("interrupted by another navigation to chrome-error://") - live, BlazeDemo
            for attempt in range(3):
                try:
                    self.page.wait_for_timeout(300)
                    resp = self.page.goto(fixed, wait_until="domcontentloaded", timeout=30000)
                    break
                except Exception as e2:  # noqa: BLE001
                    if "interrupted by another navigation" not in str(e2) or attempt == 2:
                        raise
        obs = self.observe()
        if resp is not None and resp.status >= 400 and resp.status not in REFUSED:
            # a guessed URL that doesn't exist (live: books.toscrape's "travel_17" - "done", then 20 steps lost)
            return {"status": "error", "url": obs["url"], "title": obs["title"], "page": self._view(obs, 600),
                    "note": f"HTTP {resp.status}: this page doesn't exist. Don't guess URLs: open the site's home "
                            f"page and do() your way there (\"open the Travel category\")."}
        # refusals trickle in after the page loads (Target: 1-2 by now, 4-5 within two seconds)
        deadline = time.monotonic() + 2.0
        while 0 < len(self._refused_here()) < 3 and time.monotonic() < deadline:
            self.page.wait_for_timeout(250)
        out = {"status": "done", "url": obs["url"], "title": obs["title"], "page": self._view(obs, 1200),
               "note": CANVAS_NOTE if obs.get("canvas", 0) >= CANVAS_SHARE else ""}
        if fixed:
            out["note"] = f"{url} does not exist; opened {fixed} (the site the request names) instead"
        wall = self._bot_wall(obs)
        if wall:
            out.update(status="blocked", page=self._view(wall, 1200), note=BLOCKED_NOTE.format(repr(wall["wall"])))
        elif self._refused_note():
            out["note"] = self._refused_note()
            if self._refusing():
                out["status"] = "blocked"
        elif self._dropped(obs["url"], url):
            out.update(status="blocked", note=self._dropped(obs["url"], url))
        return out

    def read(self, question: str | None = None, max_chars: int = 4000) -> str:
        """Page text. With a question, only the passages Jev finds relevant to it."""
        obs = self.observe()
        text = obs["text"]
        note = f"[note: {self._refused_note()}]\n" if self._refused_note() else ""
        if obs.get("canvas", 0) >= CANVAS_SHARE:
            note += f"[note: {CANVAS_NOTE}]\n"
        # what is typed into the fields isn't page text: without it a worker couldn't check its own form
        # (live, demoqa: it reloaded the page to "check" and lost what it had entered; asked as a question,
        # "what are the field values?" got the page's menu back and it retried fields that were right)
        fields = [describe(e) for e in obs["elements"] if e["kind"] in FIELDS | TOGGLES][:40]
        tail = "\n[form fields]\n" + "\n".join(fields) if fields else ""
        if not question:
            return note + text[:max(max_chars - len(tail), 0)] + tail
        return note + "\n---\n".join(pick_passages(self, self.cap, question, text))[:max(max_chars - len(tail), 0)] + tail

    def elements(self, match: str | None = None) -> list[str]:
        """The page's controls, numbered for click()/fill()/select(). `match` keeps the ones whose
        description contains it (case-insensitive): elements("october"), elements("add to cart")."""
        out = [f"[{e['i']}] {describe(e)}" for e in self.observe()["elements"]]
        return [s for s in out if match.lower() in s.lower()] if match else out

    # ---------------------------------------------------------------- the worker acting itself
    # do() is the default: Jev picks each click and the worker never reads the page. When Jev gets
    # stuck (a big app-like site, a calendar), the worker picks the element itself, by the number
    # elements() or a do() result's `candidates` shows. The same guards apply as in do().
    def _element(self, i: int) -> dict:
        els = (getattr(self, "_last_obs", None) or {}).get("elements", [])
        e = next((e for e in els if e["i"] == int(i)), None)
        if e is None:
            raise ValueError(f"no element [{i}] in the last listing: call browser.elements() again")
        return e

    def _direct(self, tool: str, e: dict, value: str | None = None) -> dict:
        before = self._signature(self._last_obs)
        self._caption = f"worker (not Jev): {tool.replace('_', ' ')} {describe(e)}"
        self.navs = []
        try:
            label = self._act(tool, e, value)
        except Exception as ex:  # noqa: BLE001 - reported, like a failed action in do()
            obs = self.observe()
            return {"status": "error", "action": f"{tool} {describe(e)}", "changed": False, "url": obs["url"],
                    "note": f"it failed: {str(ex).splitlines()[0][:160]}", "page": self._view(obs, 1200)}
        obs = self.observe()
        out = {"status": "done", "action": label, "changed": self._signature(obs) != before, "url": obs["url"],
               "note": "", "page": self._view(obs, 1200)}
        wall = self._bot_wall(obs)
        if wall:
            out.update(status="blocked", note=BLOCKED_NOTE.format(repr(wall["wall"])), page=self._view(wall, 1200))
        elif self._refusing():
            out.update(status="blocked", note=self._refused_note())
        elif self._dropped(obs["url"]):
            out.update(status="blocked", note=self._dropped(obs["url"]))
        elif "didn't keep it" in label:
            out.update(status="error", note="the field didn't keep the text")
        opts = [f"[{x['i']}] {describe(x)}" for x in obs["elements"] if x["kind"] == "option"]
        if opts and tool == "type":
            out["suggestions"] = opts[:10]
            out["note"] = (out["note"] + "; " if out["note"] else "") + "pick one with browser.click(i)"
        return out

    def click(self, i: int, allow_irreversible: bool = False) -> dict:
        """Click element [i] (from elements() or a do() result's candidates) -> {status, action, changed,
        url, page, note}. An order, payment, send or delete returns needs_confirmation unless allowed."""
        e = self._element(i)
        if e["kind"] in FIELDS:
            return {"status": "error", "note": f"{describe(e)} is a field: use browser.fill({i}, text)"}
        if e["kind"] == "dropdown":
            return {"status": "error", "note": f"{describe(e)} is a dropdown: use browser.select({i}, option)"}
        label = e.get("label", "")
        if not allow_irreversible and not self._navigates(e) and not REVERSIBLE.match(label):
            risky = bool(IRREVERSIBLE.search(label))
            if not risky and e["kind"] == "button":   # the same Jev check do() makes before a button
                risky = self._ask({"element": describe(e), "page": {"title": self._last_obs["title"],
                                                                     "url": self._last_obs["url"]}},
                                  {"irreversible": noul(Q_IRREV)})["irreversible"].noul >= 0.6
            if risky:
                return {"status": "needs_confirmation", "note": f"clicking {describe(e)} looks hard to undo: "
                        f"confirm with the user, then browser.click({i}, allow_irreversible=True)"}
        return self._direct("click", e)

    def fill(self, i: int, text: str, enter: bool = False) -> dict:
        """Type `text` into field [i] (replacing what is there), read back; enter=True presses Enter after.
        -> {status, action, changed, url, page, note, suggestions (when the field opens a list)}."""
        e = self._element(i)
        if e["kind"] not in FIELDS:
            return {"status": "error", "note": f"{describe(e)} is not a text field"
                    + (f": use browser.select({i}, option)" if e["kind"] == "dropdown" else "")}
        if e["kind"] == "password" or CARD_FIELD.search(e.get("label", "") + " " + e.get("name", "")):
            return {"status": "refused", "note": "passwords and card details aren't typed by hand: a sign-in "
                    "the user gave you goes through do('sign in', values={...}); payment is the user's"}
        return self._direct("press_enter" if enter else "type", e, str(text))

    def select(self, i: int, option: str) -> dict:
        """Choose `option` in dropdown [i] (exact text, else the option containing it)."""
        e = self._element(i)
        if e["kind"] != "dropdown":
            return {"status": "error", "note": f"{describe(e)} is not a dropdown"
                    + (": its choices are separate elements - browser.click(i)" if e["kind"] in TOGGLES else "")}
        return self._direct("select", e, str(option))

    def html(self, css: str | None = None, max_chars: int = 30000) -> str:
        """The page's HTML (or of each element matching `css`), for scraping what text doesn't show:
        ratings drawn as star icons (books.toscrape: class="star-rating Four"), links, attributes."""
        obs = self.observe()
        head = f"<!-- jevh note: {CANVAS_NOTE} -->\n" if obs.get("canvas", 0) >= CANVAS_SHARE else ""
        if css:
            parts = self.page.eval_on_selector_all(css, "els => els.map(e => e.outerHTML)")
            return head + "\n".join(parts)[:max_chars]
        return head + self.page.content()[:max_chars]

    def press(self, keys: str | list[str], times: int = 1, delay: float = 0.15) -> dict:
        """Press keys on the page: arrow keys for a game, Escape to close a popup, Tab, Enter.
        `keys` is one key ("ArrowLeft") or a list pressed in turn; the whole list repeats `times`."""
        keys = [keys] if isinstance(keys, str) else list(keys)
        before = self.observe()
        # a canvas repaints without any text changing (live, 2048: a move that merged nothing left the score
        # alone, "changed" said False, and the worker decided the keys did nothing): compare what it shows
        canvas = before.get("canvas", 0) >= CANVAS_SHARE
        pic = hashlib.sha1(self.page.screenshot()).hexdigest() if canvas else ""
        for _ in range(max(1, times)):
            for k in keys:
                self.page.keyboard.press(k)
                if delay:
                    self.page.wait_for_timeout(int(delay * 1000))
        if canvas:
            self.page.wait_for_timeout(250)   # let the move's animation finish before comparing
        obs = self.observe()
        changed = self._signature(obs) != self._signature(before) or (
            canvas and hashlib.sha1(self.page.screenshot()).hexdigest() != pic)
        return {"status": "done", "pressed": keys * max(1, times) if len(keys) * times <= 20 else f"{keys} x{times}",
                "changed": changed, "url": obs["url"], "page": self._view(obs, 1200),
                "note": CANVAS_NOTE if canvas else ""}

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
        if self._dock_file:
            self._dock_file.unlink(missing_ok=True)
        try:
            self.context.close()
            if self._browser:
                self._browser.close()
        finally:
            self._pw.stop()

    def test(self, url: str, steps: str | list[str], expect: str, values: dict | None = None,
             max_actions: int = 8, allow_irreversible: bool | None = None) -> dict:
        """Test a flow in a web app: open `url`, reach each step with do(), then check the page
        shows `expect`. -> {passed, why, steps, expect, js_errors, console_errors, url, page,
        screenshot (on failure)}. Uncaught script errors fail the test; console errors are reported.
        On the user's own local app (localhost, *.test, file://) steps may submit, order or delete
        without asking - that's what a test exercises; elsewhere as allow_irreversible says."""
        started, calls0 = time.monotonic(), self.jev_calls
        steps = [steps] if isinstance(steps, str) else list(steps)
        js0, con0 = len(self.js_errors), len(self.console_errors)
        self.goto(url)
        allow = bool(LOCAL_URL.match(self.page.url)) if allow_irreversible is None else allow_irreversible
        done, why = [], ""
        for goal in steps:
            r = self.do(goal, values, max_actions, allow, act_first=True)
            done.append({"goal": goal, "status": r["status"], "actions": r["actions"], "note": r["note"]})
            if r["status"] not in ("done", "likely_done"):
                why = f"step {goal!r} ended {r['status']}" + (f" ({r['note']})" if r["note"] else "")
                break
        obs = self.observe()
        check = {"text": expect}
        if not why:
            text = obs["text"]
            if len(text) > EXPECT_PAGE_CHARS:
                text = "\n---\n".join(pick_passages(self, self.cap, expect, text))[:EXPECT_PAGE_CHARS]
            a = self.ask({"expect": expect, "page": text},
                         {"shows": noul(EXPECT_SHOWS_Q), "holds": noul(EXPECT_HOLDS_Q)})
            check.update(shows=round(a["shows"].noul, 2), holds=round(a["holds"].noul, 2))
            if not (check["shows"] >= EXPECT_SHOWS and check["holds"] >= EXPECT_HOLDS):
                why = f"the page doesn't show that {expect}"
        js, console = self.js_errors[js0:], self.console_errors[con0:]
        if js and not why:
            why = f"the page threw {len(js)} script error(s): {js[0]}"
        shot = None
        if why:
            slug = re.sub(r"[^a-z0-9]+", "-", expect.lower()).strip("-")[:40] or "test"
            try:
                shot = self.screenshot(f"test-failed-{slug}.png")
            except Exception:  # noqa: BLE001 - the report matters more than the picture
                pass
        return {"passed": not why, "why": why, "steps": done, "expect": check, "js_errors": js[:10],
                "console_errors": console[:10], "url": obs["url"], "page": self._view(obs, 1200),
                "screenshot": shot, "jev_calls": self.jev_calls - calls0,
                "seconds": round(time.monotonic() - started, 1)}

    def do(self, goal: str, values: dict | None = None, max_actions: int = 8,
           allow_irreversible: bool = False, act_first: bool = False) -> dict:
        """Reach one observable outcome on the current page. See the module docstring.
        `act_first`: don't accept "already done" before taking an action (test() steps)."""
        started, calls0, secs0 = time.monotonic(), self.jev_calls, self.jev_secs
        values = {str(k): str(v) for k, v in (values or {}).items()}
        used: set[str] = set()
        history: list[str] = []
        banned: set[int] = set()      # no effect, or a toggle already set during this goal
        deferred: set[int] = set()    # a submit button held back until the form matches the goal
        no_effect = 0
        suggesting = False            # the last typing opened autocomplete suggestions
        last_change = "nothing yet"
        choice: tuple[dict, dict] | None = None   # (page, Jev's scores) of the last pick

        def result(status: str, obs: dict, note: str = "", probs: dict | None = None) -> dict:
            if status not in ("done", "needs_confirmation") and self._refusing():
                status = "blocked"
            if self._refused_note():
                note = (note + "; " if note else "") + self._refused_note()
            out = {"status": status, "goal": goal, "url": obs["url"], "title": obs["title"],
                   "actions": history, "note": note, "page": self._view(obs, 1200),
                   "p": {k: round(v, 2) for k, v in (probs or {}).items()},
                   "jev_calls": self.jev_calls - calls0, "jev_seconds": round(self.jev_secs - secs0, 1),
                   "seconds": round(time.monotonic() - started, 1)}
            # Jev couldn't finish: what it was choosing between, numbered, so the worker can act itself
            # (live, Booking.com: the worker saw the date it needed and could only reword the goal)
            if status in ("stuck", "ambiguous", "max_actions", "error") and choice and choice[0] is obs:
                by_i = {str(e["i"]): e for e in obs["elements"]}
                top = [k for k in sorted(choice[1], key=choice[1].get, reverse=True) if k in by_i][:6]
                if top:
                    out["candidates"] = [f"[{k}] {describe(by_i[k])}" for k in top]
                    out["note"] += ("; " if out["note"] else "") + ("act yourself if you see the right one: "
                                                                    "browser.click(i) / fill(i, text) / select(i, option)")
            return out

        obs = self.observe()
        start_url = obs["url"]
        actions = 0
        for _ in range(2 * max_actions + 2):
            wall = self._bot_wall(obs)
            if wall:
                return result("blocked", wall, BLOCKED_NOTE.format(repr(wall["wall"])))
            # values that obviously belong in an empty field go in without asking Jev
            filled = self._autofill(obs, values, used)
            if filled:
                history.append("filled " + ", ".join(filled))
                actions += 1
                obs = self.observe()
                suggesting = any(e["kind"] == "option" for e in obs["elements"])
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
            choice = (obs, target_p)

            # "... and save/continue": setting the control isn't the end; semif's done check misses that
            if SUBMIT_WORDS.search(goal) and not any(re.match(r"clicked (button|link)|.*pressed Enter", h)
                                                     for h in history):
                probs["done"] = min(probs["done"], 0.29)
            # a test's step is something to do: "already done" before any action isn't a pass (live:
            # "open the cart" was judged done on the product page's "Added ... to your cart" note, so
            # the test never saw the broken cart page)
            if act_first and not history:
                probs["done"] = min(probs["done"], 0.29)
            # typing into an autocomplete selects nothing until a suggestion is picked (live: demoqa's
            # "Subjects" was judged done with "Maths" typed and its suggestion still open)
            if suggesting and any(e["kind"] == "option" for e in obs["elements"]):
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
                # every value given went into its field and nothing asks to submit: that was the goal
                # (live, demoqa: all four fields filled, reported "stuck")
                if values and all(k in used for k in values) and not SUBMIT_WORDS.search(goal) and \
                        not any("didn't keep it" in h for h in history) and \
                        not (suggesting and any(e["kind"] == "option" for e in obs["elements"])):
                    return result("done", obs, probs=probs)
                if probs["done"] >= 0.5:
                    return result("done", obs, probs=probs)
                if probs["done"] >= 0.3:
                    return result("likely_done", obs, "probably achieved; check `page`", probs)
                if history and obs["url"] != start_url:
                    # it acted and the page moved on, but doesn't look like the goal (live: BlazeDemo's purchase
                    # page names another flight than the one chosen; "stuck" alone hid that the click worked)
                    return result("stuck", obs, f"the actions took the page from {start_url} to {obs['url']}, but "
                                  "it doesn't look like the goal: read `page` - it may be done with the site showing "
                                  "something unexpected", probs)
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
                quoted = QUOTED.findall(goal) or [m.group(1) for m in [SET_TO.match(goal)] if m]
                if value is None and not values and len(quoted) == 1 and target.get("value") != quoted[0]:
                    # the text is in the goal: do("type 'Sam' into First Name"), do("set the date of birth to 12 Mar 1990")
                    value, pick = quoted[0], None
                if value is None and not target.get("value"):
                    return result("needs_value", obs, f"{describe(target)} needs text: call again with "
                                  f"values={{'{target.get('label') or 'text'}': '<the text>'}}", probs)
                if value is None and target["kind"] != "search box":
                    # already filled: Enter in a form field submits the whole form (live, demoqa)
                    banned.add(target["i"])
                    last_change = f"{describe(target)} is already filled in"
                    continue
                if value is None:
                    tool = "press_enter"   # a filled search box: run the search
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
            elif tool == "click" and not allow_irreversible and IRREVERSIBLE.search(target.get("label", "")) \
                    and not self._navigates(target):
                return result("needs_confirmation", obs, f"next: click {describe(target)}, which looks hard "
                              f"to undo; call again with allow_irreversible=True if the user wants it", probs)

            before = self._signature(obs)
            self._caption = (f"Jev ({target_p[best]:.2f}): {tool.replace('_', ' ')} "
                             f"{describe(target) if target else ''}   |   goal: {goal}")
            self.navs = []
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
            if self._dropped(obs["url"]):   # the site threw the search away (Booking.com's Search button)
                return result("blocked", obs, self._dropped(obs["url"]), probs)
            suggesting = tool == "type" and any(e["kind"] == "option" for e in obs["elements"])
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
                    f" to {obs['url']}" if obs["url"] != state["page"]["url"] else "") + (
                    "; suggestions opened under the field: pick the one the goal means" if suggesting else "")
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
    def _navigates(target: dict) -> bool:
        """A link to another page: following it only loads that page ("Buy Tickets" to the ticket
        site), so it is never the hard-to-undo step itself."""
        href = target.get("href") or ""
        return target["kind"] == "link" and bool(href) and not href.startswith(("#", "javascript"))

    def _bot_wall(self, obs: dict) -> dict | None:
        """A bot check (Cloudflare, Akamai, captcha) on the page: give it a few seconds to clear by
        itself, then report it. The agent doesn't try to get past it; a person can, in a visible browser."""
        for _ in range(4):
            m = BOT_WALL.search(obs["title"] + "\n" + obs["text"][:1500])
            if not m:
                return None
            time.sleep(WALL_WAIT)
            obs = self.observe()
        return {**obs, "wall": m.group(0)}

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
            # empty, or holding something else (a date field showing today, an old search): replace it
            fields = [e for e in obs["elements"] if e["kind"] in FIELDS and field_matches(key, e)
                      and (e.get("value") or "").lower() != val.lower()[:60]]
            if not fields:
                # a choice, not text: {"Gender": "Male"} -> radio 'Male', {"Hobbies": "Reading"} -> checkbox
                # 'Reading' (live, demoqa: "Reading" was typed into the Subjects box)
                toggles = [e for e in obs["elements"] if e["kind"] in ("radio", "checkbox") and _same_text(e["label"], val)]
                if len(toggles) == 1 and not SECRET.search(key):
                    if not toggles[0].get("checked"):
                        self._show(toggles[0]["i"], f"Jev: {describe(toggles[0])} for '{key}' (label match)")
                        loc = self.page.locator(f'[data-jevh-i="{toggles[0]["i"]}"]').first
                        try:
                            loc.click(timeout=3000)
                        except Exception:  # noqa: BLE001 - a styled control over the input: click it the DOM way
                            loc.evaluate("el => el.click()", timeout=3000)
                        filled.append(f"clicked {toggles[0]['kind']} '{toggles[0]['label']}' (value '{key}')")
                    used.add(key)
                    obs = self.observe()
                continue
            for e in fields if fields[0]["kind"] == "password" else fields[:1]:
                self._show(e["i"], f"Jev: fill {describe(e)} with '{key}' (label match)")
                loc = self.page.locator(f'[data-jevh-i="{e["i"]}"]').first
                filled.append(f"{e['kind']} '{e['label']}' (value '{key}')" + self._fill(loc, val, e["kind"]))
                picked = self._pick_suggestion(val)
                if picked:
                    filled.append(picked)
            used.add(key)
            obs = self.observe()   # a fill can open a picker or re-render the form: fresh element numbers
        # one value whose name matches no field, and one empty field in view: it goes there (live, Booking.com:
        # values={"From": "Denver"} for the one box, "Enter destination" - Jev clicked a promo link instead)
        left = [k for k in values if k not in used and not SECRET.search(k)]
        empty = [e for e in obs["elements"] if e["kind"] in ("text field", "search box") and not e.get("value")
                 and not e.get("below") and not e.get("disabled")]
        if len(left) == 1 and len(values) == 1 and len(empty) == 1 and \
                not any(field_matches(left[0], e) for e in obs["elements"] if e["kind"] in FIELDS):
            e, val = empty[0], values[left[0]]
            loc = self.page.locator(f'[data-jevh-i="{e["i"]}"]').first
            filled.append(f"{e['kind']} '{e['label']}' (value '{left[0]}', the only empty field)" + self._fill(loc, val, e["kind"]))
            picked = self._pick_suggestion(val)
            if picked:
                filled.append(picked)
            used.add(left[0])
        return filled

    def _fill(self, loc, val: str, kind: str = "text field") -> str:
        """Fill a field and read it back. A date the field rejects (demoqa's picker turned "12 March 1990"
        back into today while "12 Mar 1990" stuck) is retried in common formats. -> "" when the field
        holds the value, else a note for the history."""
        date = _date_forms(val)[:1]

        def put(text: str) -> str:
            loc.fill(text, timeout=5000)
            if date:   # a date picker shows what was typed until the field loses focus, then keeps or reverts it
                loc.evaluate("el => el.blur()", timeout=2000)
                self.page.wait_for_timeout(150)
            return loc.input_value(timeout=2000)

        if kind == "password":
            loc.fill(val, timeout=5000)
            return ""
        try:
            kept = put(val)
        except Exception:  # noqa: BLE001 - a contenteditable has no input value: trust the fill
            return ""
        if date:
            self.page.keyboard.press("Escape")   # close the calendar the field opened: it covers the next fields
        if not kept and loc.get_attribute("aria-autocomplete", timeout=2000):
            return ""   # a combobox (react-select) empties its box once a choice is made; not a rejection
        same = lambda shown: _same_text(shown, val) or bool(date) and _date_forms(shown)[:1] == date
        if same(kept) or re.sub(r"\D", "", kept) == re.sub(r"\D", "", val) != "":   # kept, maybe reformatted
            return ""
        for alt in _date_forms(val):
            kept = put(alt)
            if same(kept):
                return f" [as '{kept}']"
        return f" [the field didn't keep it: it shows '{kept[:40]}']"

    def _pick_suggestion(self, val: str) -> str | None:
        """Typing opened suggestions and exactly one is what was typed ("Maths" -> option 'Maths'): pick it,
        without asking Jev. Several or none: Jev chooses in the loop."""
        self.page.wait_for_timeout(300)
        opts = [e for e in self.observe()["elements"] if e["kind"] == "option"]
        hits = [e for e in opts if _same_text(e["label"], val)] or \
               [e for e in opts if e["label"].lower().startswith(val.lower())]
        if len(hits) != 1:
            return None
        self.page.locator(f'[data-jevh-i="{hits[0]["i"]}"]').first.click(timeout=3000)
        return f"picked suggestion '{hits[0]['label']}'"

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
        # "Add to cart" only serves a goal that adds, buys or gets something: semif scored it 0.47-0.5
        # "wanted" for "open the shopping cart" (they share "cart"), just over the line
        wanted = res["wanted"].noul >= WANTED_MIN and not (REVERSIBLE.match(label) and not ADD_GOAL.search(goal))
        # a form is sent only when the goal says so (live, demoqa: "fill name, email, gender and mobile"
        # filled them, then clicked Submit)
        if FORM_SEND.search(label) and not (SUBMIT_WORDS.search(goal) or SEND_GOAL.search(goal)):
            wanted = False
        return irreversible, ("pending" not in qs or res["pending"].noul < 0.8), wanted

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
                          if e["kind"] == target["kind"] and e["label"] == target["label"]
                          and e.get("context") == target.get("context")), None)
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
            kept = self._fill(loc, value, target["kind"]) if value is not None else ""
            if tool == "press_enter":
                loc.press("Enter", timeout=5000)
                return f"typed into {what} and pressed Enter" if value is not None else f"pressed Enter in {what}"
            picked = self._pick_suggestion(value) if value else None
            return f"typed into {what}{kept}" + (f"; {picked}" if picked else "")
        tabs = len(self.context.pages)
        try:
            loc.click(timeout=3000)
        except Exception:  # noqa: BLE001 - covered or animating: a DOM click still reaches the handler
            loc.evaluate("el => el.click()", timeout=3000)
        # links open in this tab (SAME_TAB_JS); a script's pop-up still opens a tab, a moment after the click:
        # without waiting, the old page looked unchanged and the click was banned as "no effect" (live: Red
        # Rocks' "Buy Tickets" opened three different concerts on AXS)
        deadline = time.monotonic() + 0.4
        while len(self.context.pages) <= tabs and time.monotonic() < deadline:
            page.wait_for_timeout(100)
        if len(self.context.pages) > tabs:
            self.page = self.context.pages[-1]
            self.page.bring_to_front()
            return f"clicked {what}; it opened in a new tab"
        return f"clicked {what}"


_session: Session | None = None
_thread = None   # the one thread that owns the shared session (Playwright's sync API is per thread)


def _run(fn, *args):
    """Run fn(session, *args) on the browser's own thread. Playwright's sync API refuses a thread
    with a running asyncio loop, and a worker that starts Playwright itself in the REPL leaves one
    behind (live: every browser.goto() after that raised "Sync API inside the asyncio loop")."""
    global _thread
    if _thread is None:
        from concurrent.futures import ThreadPoolExecutor
        _thread = ThreadPoolExecutor(1, thread_name_prefix="jevh-browser")
    return _thread.submit(lambda: fn(session(), *args)).result()


def session() -> Session:
    """The shared session. Call its methods through the module functions: they run on its thread."""
    global _session
    if _session is None:
        _session = Session()
    return _session


def goto(url: str) -> dict:
    return _run(Session.goto, url)


def do(goal: str, values: dict | None = None, max_actions: int = 8, allow_irreversible: bool = False) -> dict:
    return _run(Session.do, goal, values, max_actions, allow_irreversible)


def test(url: str, steps: str | list[str], expect: str, values: dict | None = None, max_actions: int = 8,
         allow_irreversible: bool | None = None) -> dict:
    return _run(Session.test, url, steps, expect, values, max_actions, allow_irreversible)


def read(question: str | None = None, max_chars: int = 4000) -> str:
    return _run(Session.read, question, max_chars)


def elements(match: str | None = None) -> list[str]:
    return _run(Session.elements, match)


def click(i: int, allow_irreversible: bool = False) -> dict:
    return _run(Session.click, i, allow_irreversible)


def fill(i: int, text: str, enter: bool = False) -> dict:
    return _run(Session.fill, i, text, enter)


def select(i: int, option: str) -> dict:
    return _run(Session.select, i, option)


def html(css: str | None = None, max_chars: int = 30000) -> str:
    return _run(Session.html, css, max_chars)


def press(keys: str | list[str], times: int = 1, delay: float = 0.15) -> dict:
    return _run(Session.press, keys, times, delay)


_host = None   # set by the worker's REPL: calls into the harness (here: its model, to look at a screenshot)
_looks = 0


def look(question: str, full_page: bool = False) -> dict:
    """What the page looks like, answered by your own model from a screenshot: a canvas game board, a
    chart, a seat map, anything drawn rather than written. -> {answer, screenshot}. Costs a model call
    with an image, so use read()/elements() when the text holds it."""
    global _looks
    if _host is None:
        raise RuntimeError("browser.look() works inside a worker's REPL (it asks the worker's model)")
    _looks += 1
    shot = screenshot(f"look-{_looks:02d}.png", full_page)
    # "right away": asked plainly, Qwen 3.8 thought for 8k tokens (5 min, cut off) over a 2048 board; told
    # to read it off, 586 tokens in 42 s, every tile right
    answer = _host("look", shot, "Look at the image and answer right away - no deliberation or double-checking, "
                   "just read it off. It is a screenshot of the web page open in the browser: answer only from what "
                   "it shows, exactly (numbers, positions, labels), and say so if something is hidden or can't be "
                   "seen.\n\n" + question)
    return {"answer": answer, "screenshot": shot}


def screenshot(name: str = "page.png", full_page: bool = False) -> str:
    return _run(Session.screenshot, name, full_page)


def url() -> str:
    return _run(lambda s: s.page.url)


def visible() -> bool:
    """Is the browser a window the user can see (Watch browser on)? Then they can take over,
    e.g. to enter payment details the worker must never type."""
    return _session.headed if _session is not None else os.environ.get("JEVH_BROWSER_HEADED") == "1"


def close() -> None:
    global _session
    if _session is not None:
        _run(Session.close)
        _session = None
