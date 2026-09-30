"""Tool modules a persona can list under `tools:`. Each is pre-imported into the worker's
REPL under its name. Jev decides per task which tools' guides go into the prompt
(`description` is what it judges on), so a question never carries browser docs.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolInfo:
    description: str     # what Jev sees when deciding if the tool helps
    guide: str           # what the worker's prompt gets when it does


TOOLS: dict[str, ToolInfo] = {
    "web": ToolInfo(
        "search the internet and read web pages to look up current facts, docs, prices, news or how-tos",
        """web (search + reading; you get the passages that answer your question, not whole pages):
- web.ask(question) -> {"passages": [{title, url, text}], "results": [...]}: search, pick the best
  pages, return the relevant passages with sources. Start here for any lookup.
- web.search(query, n=8) -> [{title, url, snippet}]; web.read(url, question=None) -> text."""),
    "browser": ToolInfo(
        "operate a real web browser: click through websites, fill in and submit forms, sign in, "
        "or test a web page or app",
        """browser (a real Chromium; you state outcomes, Jev picks each click in ~1-2s). It is already
imported: use it, don't start Playwright/Selenium yourself.
- browser.goto(url); r = browser.do("one observable outcome", values={"email": ...}) -> {status, page, actions, note}
  status: done | likely_done | needs_confirmation | needs_value | needs_login | ambiguous | error | stuck | ...
  One outcome per do() ("open the pricing page", "add the blue mug in size L to the cart"). If it is not
  done, read `note`/`page`, then rephrase or split the step. On needs_confirmation (an order, payment,
  sending, deleting) confirm first: ask_user("Place the order for $51.25?", ["Yes", "No"]), and pass
  allow_irreversible=True only after a yes. needs_value / needs_login: ask_user for the details.
  blocked (goto too): the site is refusing automated browsers or shows a bot check. Don't retry, rephrase
  or work around it: if browser.visible(), ask_user to take over in the window; otherwise final() saying
  which site blocked you and what you had found before it.
- Fields: pass the text as values={"First name": "Sam", "Email": ...} (keys like the field labels). A field
  that suggests options as you type (airports, cities, subjects): do("choose Boston Logan in From",
  values={"From": "Boston"}) - it types, then picks the matching suggestion.
- Many identical buttons ("Select", "Add to cart" per row): name the row: do("choose the Aer Lingus flight").
- browser.read() (no question) ends with the form fields and what they hold: check a form with it.
  browser.goto() reloads the page and loses what was typed: don't use it to "refresh" a form.
- Keys: browser.press("ArrowLeft", times=1) or press(["ArrowUp", "ArrowLeft"], times=20) -> {changed, page}
  for games and keyboard-driven pages; press("Escape") closes a popup. A game drawn on a canvas can't be
  read (not in text or HTML): play by keys, and go by the page text (score) and `changed` (False: that key
  did nothing - try another).
- Buying or booking something the user asked for is the job, not something to decline: find it, choose
  it, fill in what you know (ask_user for names, dates or choices you don't have) and go up to payment.
  Never type card numbers or passwords yourself. If browser.visible(): ask_user("It's at the payment
  step in the browser window - pay there, then reply done", ["done", "cancel"]) and confirm the result
  after; if not, say where you stopped and give the page URL.
- Scraping: browser.html(css=None) -> the HTML (of the elements matching `css`), for what the text doesn't
  show (star ratings as icons, links, attributes); parse it in Python. Navigate by do(), not guessed URLs.
- browser.read(question=None) -> text (only the passages that answer `question`);
  browser.screenshot(name) -> artifact path; browser.elements() -> list; browser.url().
- Local files: browser.goto("file:///C:/path/page.html") to check a page you built.
- Testing an app you built or changed: t = browser.test(url, ["add the blue mug to the cart", "open the
  cart"], expect="the cart has 1 blue mug") -> {passed, why, steps, js_errors, console_errors,
  screenshot}. One flow per test; if it fails, read `why` and fix the app (or the test), then rerun."""),
    "project": ToolInfo(
        "read, search, edit and run code or files in the user's project folder",
        """project (the user's project folder; ctx['project'] is its path):
- project.tree(".", depth=2); project.find(question) -> ["path:line: text", ...] where it's answered, best
  first (project.read(hit) opens the lines around one); project.grep(regex, glob)
- project.read(path, start, end) -> numbered lines; project.edit(path, old, new) (exact, unique text);
  project.write(path, text); project.sh(cmd, timeout=120) -> {code, out}
- project.changed() / project.undo(path=None). Read before editing; run the tests after.
- An edit that breaks the project's own rules (RULES.md / AGENTS.md / CLAUDE.md) raises RuleViolation
  quoting the rule: fix the edit. Only if the rule truly doesn't fit, pass rule_ok="why" and say so
  in your final answer."""),
    "comfy": ToolInfo(
        "generate or edit images and video with ComfyUI workflows",
        """comfy (ComfyUI): wf = comfy.load(name); comfy.find(wf, class_type); comfy.set(wf, {"node.input": v});
comfy.seed(); paths = comfy.run(wf) -> saved artifacts. comfy.workflows() lists the available files."""),
}
