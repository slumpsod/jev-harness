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
        """browser (a real Chromium; you state outcomes, Jev picks each click in ~1-2s):
- browser.goto(url); r = browser.do("one observable outcome", values={"email": ...}) -> {status, page, actions, note}
  status: done | likely_done | needs_confirmation | needs_value | needs_login | ambiguous | error | stuck | ...
  One outcome per do() ("open the pricing page", "add the blue mug in size L to the cart"). If it is not
  done, read `note`/`page`, then rephrase or split the step. On needs_confirmation (an order, payment,
  sending, deleting) confirm first: ask_user("Place the order for $51.25?", ["Yes", "No"]), and pass
  allow_irreversible=True only after a yes. needs_value / needs_login: ask_user for the details.
- Buying or booking something the user asked for is the job, not something to decline: find it, choose
  it, fill in what you know (ask_user for names, dates or choices you don't have) and go up to payment.
  Never type card numbers or passwords yourself. If browser.visible(): ask_user("It's at the payment
  step in the browser window - pay there, then reply done", ["done", "cancel"]) and confirm the result
  after; if not, say where you stopped and give the page URL.
- browser.read(question=None) -> text (only the passages that answer `question`);
  browser.screenshot(name) -> artifact path; browser.elements() -> list; browser.url().
- Local files: browser.goto("file:///C:/path/page.html") to check a page you built."""),
    "project": ToolInfo(
        "read, search, edit and run code or files in the user's project folder",
        """project (the user's project folder; ctx['project'] is its path):
- project.tree(".", depth=2); project.find(question) -> likely files; project.grep(regex, glob)
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
