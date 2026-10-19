"""Worker: one persona doing one task, acting by writing Python in its own REPL.

The prompt holds the persona, the REPL API, and only the skills Jev picked. The
request is the user message; everything bigger (memory, other assistants' outputs)
sits in the REPL as `ctx[...]` and is only listed by name and size.
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import platform
import re
from collections import Counter
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .errors import Cancelled
from .interject import UPDATED
from .llm import LLM, LLMError
from .personas import Persona
from .repl.client import Repl
from .skills import Skill

# Thinking models (Qwen3.8 in LM Studio) otherwise draft the whole solution in hidden
# reasoning before replying - measured 16k+ thinking tokens and zero code on a web-app task,
# vs 127 thinking tokens with this nudge. enable_thinking / reasoning_effort / /no_think
# had no effect through LM Studio's API.
FIRST_STEP_NUDGE = ("\n\nIf this needs code: reply now with a short plan (at most 5 bullets) and a first "
                    "python block for step one, and build incrementally across replies - do not design "
                    "the whole solution before replying. If it needs no code, reply with just the answer.")
HISTORY_CHARS = 3000       # per earlier assistant reply carried into a follow-up
WATCH_WHY = {"testable": "you were working out by hand something code does exactly on the data",
             "searching": "you have been trying one theory after another in your head",
             "repeating": "you were repeating yourself"}
SILENT_RETRIES = 2          # more tries for a step the server answers with silence
REPLY_KEPT = 3000          # chars of a stopped reply kept in the conversation (a model that reasons in its reply)
WORDS_RE = re.compile(r"[A-Za-z][A-Za-z'_]+")


def _step_outputs(messages: list[dict]) -> list[str]:
    """What each step's code printed (tool results / result messages), oldest first, harness notes cut."""
    outs, current = [], None
    for m in messages:
        if m["role"] == "assistant":
            if current is not None:
                outs.append(current)
            current = "" if (m.get("tool_calls") or code_blocks(str(m.get("content", "")))) else None
        elif current is not None and m["role"] in ("tool", "user"):
            text = str(m.get("content", "")).split("\n\n[", 1)[0].replace("Output:\n", "", 1).strip()
            current += ("\n" if current else "") + text
    if current is not None:
        outs.append(current)
    return outs


PAGE_URL = re.compile(r"https?://[^\s'\"<>)\]\\]+", re.I)


def new_pages(earlier: list[dict], recent: list[dict]) -> list[str]:
    """Pages (host + path) the tool outputs of `recent` show that those of `earlier` never did. Only
    outputs count, not the worker's code: a URL it typed or guessed hasn't been reached."""
    def pages(msgs: list[dict]) -> set[str]:
        out = set()
        for m in msgs:
            if m.get("role") in ("tool", "user") and isinstance(m.get("content"), str):
                for u in PAGE_URL.findall(m["content"]):
                    out.add(re.sub(r"[?#].*$", "", u).rstrip("/.,;:").lower())
        return out
    before = pages(earlier)
    return sorted(pages(recent) - before)


STOP_REPEATS = 2   # recent outputs that repeat an earlier one, for a second strike to stop the task (untuned: n = 2 loops)


def repeated_outputs(earlier: list[dict], recent: list[dict]) -> int:
    """How many of `recent`'s step outputs nearly repeat an output from `earlier` or from before them in `recent`."""
    import difflib
    pool = [o[:1500] for o in _step_outputs(earlier) if o]
    hits = 0
    for o in (o[:1500] for o in _step_outputs(recent) if o and o != "[final answer recorded]"):
        if any(difflib.SequenceMatcher(None, o, p).quick_ratio() > 0.9 and difflib.SequenceMatcher(None, o, p).ratio() > 0.85
               for p in pool):
            hits += 1
        pool.append(o)
    return hits


def repeated_result(messages: list[dict], n: int = 3) -> str | None:
    """The result the last `n` code steps all printed, if identical and not an error - a worker that
    has its answer and keeps re-checking it instead of handing it over."""
    outs = [o for o in _step_outputs(messages) if o][-n:]
    if len(outs) < n or len(set(outs)) != 1:
        return None
    out = outs[0]
    return None if re.search(r"Traceback|Error\b|^\(no output\)$", out) or len(out) < 2 else out


def _last_output(messages: list[dict]) -> str:
    outs = [o for o in _step_outputs(messages) if o]
    return outs[-1] if outs else ""


def repeating(text: str, window: int = 4000, times: int = 8) -> str | None:
    """The line or sentence a model keeps writing, or None. Plain code, any model: in the last `window`
    chars, a prose line (3+ words: printed grids and tables don't count) written `times`+ times. Tuned on
    380 real thinking/reply files from the ARC runs: it flags exactly the two real loops (Gemma 4 writing
    "I'll call final(test_output)" for 108 KB, Qwen "Let me do that." for 37 KB) and nothing else."""
    units = [u.strip() for u in re.split(r"\n|(?<=[.!?])\s+", text[-window:])]
    units = [u for u in units if len(WORDS_RE.findall(u)) >= 3]
    if not units:
        return None
    line, n = Counter(units).most_common(1)[0]
    return line if n >= times else None
REASONING_CHARS = 40000    # of each step's hidden reasoning sent back with it (its end if longer; 0 = none)
NOTES_CHARS = 2500         # plus that turn's work notes (the code it ran and what came back)
NOTES_HEADER = ("[Harness note, not part of the reply the user saw: the work this reply was based on - "
                "the code you ran and what came back. Use it to answer questions about sources or steps.]")
CHARS_PER_TOKEN = 3.2      # conservative for code-heavy transcripts (context-size estimate)
# A message the user sent while the task ran (jevh/interject.py), as the model reads it. Appended at the end of the
# conversation, never spliced in: everything before it stays byte for byte what the model server already holds
# in its cache, so the redirect costs the new tokens only.
STEER_NOTE = ('[Message from the user, sent while you were working: "{text}"\n'
              "Take it into account from here on: if it changes what they want, follow it instead of the earlier "
              "request. If it asks you something, answer it now with say(\"...\") - they are waiting and see nothing "
              "of your thinking - then carry on. Your REPL variables and files are as you left them.]")
STEER_REPLACE = ('[Message from the user, sent while you were working: "{text}"\n'
                 "This replaces the request you were given: drop what you were doing for it, do this instead, and "
                 "hand over only what this asks for. Your REPL variables and files are as you left them.]")
STEER_HELD = ("\n[final() was not accepted yet: the user sent a message while you were finishing. It follows; "
              "answer it, then call final() again.]")
CUT_REPLY = "[stopped before it ran: the user sent a message]"


def _msg_chars(m: dict) -> int:
    return len(str(m.get("content", ""))) + len(str(m.get("reasoning_content", "")))


def _steps(messages: list[dict]) -> list[list[dict]]:
    """Group a stretch of conversation into steps: each assistant reply with what came back."""
    groups: list[list[dict]] = []
    for m in messages:
        if m["role"] == "assistant" or not groups:
            groups.append([m])
        else:
            groups[-1].append(m)
    return groups


def _render(entries: list[dict], dropped: int, legacy: str = "") -> str:
    """The progress log: kept steps in full, the rest as one-line notes."""
    head = f"({dropped} earlier steps left out)\n" if dropped else ""
    return legacy + head + "".join(e["full"] + "\n" if e["keep"] else e["brief"] for e in entries)


def _clip_mid(text: str, n: int) -> str:
    """Head and tail of a long text: the start says what it is, the end what came of it."""
    if len(text) <= n:
        return text
    return text[:n * 2 // 3] + f"\n[... {len(text) - n} chars ...]\n" + text[-(n // 3):]

PROGRESS_CHARS = 16000     # the progress log's budget: past it, full steps shrink to notes, then the least useful go
FULL_CODE, FULL_OUT, FULL_THOUGHT = 1500, 2000, 400   # an old step Jev keeps in full, capped at these (chars)
COMPACT_TARGET = 0.6       # compaction aims the context at this share of compact_tokens
MIN_FULL_LOG = 8000        # ...but steps still needed always get this much room in the log (chars)
RESUME_NOTE = ("[Resumed after an interruption. Your REPL was restarted, so variables are gone; files in your "
               "working directory, the artifact store and the project are as you left them. Continue the task "
               "from where you were.]")
RESTORED_NOTE = ("[Resumed after an interruption. Your REPL was restarted and its saved state brought back - "
                 "variables: {vars}; functions/classes: {defs}.{lost} Files in your working directory, the artifact "
                 "store and the project are as you left them. Continue the task from where you were.]")
# A follow-up turn by the same assistant starts where the previous one left off (Worker.carry_from). Live 2026-10-12, a
# chess game played over several turns: every move began "the helper module was lost - let me rebuild it", re-reading
# the game's source and the board before it could play.
CARRIED_NOTE = ("[This follows your previous turn. Its working folder is {dir} - the files you wrote there are still "
                "there (use that path).{state}]")
CARRIED_STATE = " Your REPL state from the end of that turn is back: variables: {vars}; functions/classes: {defs}.{lost}"
TIMEOUT_RESTORED_NOTE = ("[timed out after {secs:.0f}s; the REPL was restarted and its state as of the end of your "
                         "previous step was brought back - variables: {vars}; functions/classes: {defs}.{lost} "
                         "Anything this block created or changed in memory is gone; files it wrote are still there.]")
SKILL_PROMPT_CHARS = 6000   # longer skills: head in the prompt, full text in ctx['skills']
HARVEST_MAX = 20            # files handed over from the worker's folder when it ends without naming any
# installed packages, never the worker's own deliverable: not walked when harvesting (dist/ and build/ are, since a
# web app's built page can be the deliverable; the HARVEST_MAX cap bounds what is copied from them)
HARVEST_SKIP = {"node_modules", "bower_components", "site-packages", "venv", "env"}
CODE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
FENCE_RE = re.compile(r"```(python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def code_blocks(text: str, native: bool = False) -> list[str]:
    """The code a reply's text asks to run: its ```python blocks, and a plain ``` block only when it is Python.
    Live (2026-10-12, Qwen with native tools, a chess game): the reply "Made and verified:\n```\n1. b3 e5\n2. Bb2 Nf6..."
    was run as code; its SyntaxError came back as the next user turn, and the model spent the rest of the session
    answering "echo pastes" it took for the user's. With native tools the model acts by calling the python tool, so a
    plain block in its text is part of what it says (a move list, a table, a command for the user); in text mode it
    is code when it compiles and is more than a bare value."""
    out = []
    for tag, body in FENCE_RE.findall(text or ""):
        if tag or (not native and _is_code(body)):
            out.append(body)
    return out


def _is_code(body: str) -> bool:
    try:
        tree = ast.parse(body)
    except (SyntaxError, ValueError):
        return False
    # "1, 2, 3", "Total: 5", a bare name: data that happens to parse, not something to run
    return any(not (isinstance(n, ast.Expr) and isinstance(n.value, (ast.Constant, ast.Tuple, ast.Name)))
               and not (isinstance(n, ast.AnnAssign) and n.value is None) for n in tree.body)


# Some local models (seen: ornith in LM Studio) sometimes answer in their native tool-call
# template instead of a fenced block: <tool_call><function=python><parameter=content>...
TOOL_CALL_RE = re.compile(r"<function=python[^>]*>\s*<parameter=\w+>\s*(.*?)\s*(?:</parameter>|</function>|</tool_call>|\Z)",
                          re.DOTALL)
OPEN_FENCE_RE = re.compile(r"```(?:python|py)[ \t]*\n")
# ...and for the REPL's tools too (live 2026-09-28, ornith and Qwen alike, three turns running):
# <tool_call><function=web><parameter=search>...  They are translated into the call they mean.
XML_CALL_RE = re.compile(r"<function=(\w+)>(.*?)(?:</function>|</tool_call>|\Z)", re.DOTALL)
XML_PARAM_RE = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*(?:</parameter>|(?=<parameter=)|\Z)", re.DOTALL)
PRIMARY = {"web": "ask", "browser": "do", "project": "find", "comfy": "run"}   # a bare "question=" etc.
REPL_CALLS = {"final", "ask_user", "say", "recall", "remember", "request", "save_artifact", "load_artifact", "peek",
              "report_failure"}


FINAL_KEYS = {"answer", "answer_var", "artifacts"}


def final_params(params: dict[str, Any]) -> dict[str, Any]:
    """The final tool's arguments, read leniently. Live (Gemma 4 in LM Studio), the answer arrived packed
    into a key - {"answer_var='test_output',artifacts=[],answer": "[[8, 8, 9, 8], ...]"} - or as
    raw text "answer='[[...]]',artifacts=[]"; final(**params) then failed on a solved task."""
    if not params or set(params) <= FINAL_KEYS:
        return params
    blob = " ".join(f"{k} = {v}" for k, v in params.items())
    m = re.search(r"answer_var\W{0,4}([A-Za-z_]\w*)", blob)
    if m:
        return {"answer_var": m.group(1)}
    m = re.search(r"answer\W{0,4}?(?=[\[{])", blob)
    if m:   # the JSON value right after "answer": decoded, so a trailing ",artifacts=[]" isn't swallowed
        try:
            value, end = json.JSONDecoder().raw_decode(blob[m.end():])
            return {"answer": blob[m.end():m.end() + end]}
        except ValueError:
            pass
    values = [v for v in params.values() if isinstance(v, str)]
    return {"answer": max(values, key=len)} if values else params


def call_to_code(name: str, params: dict[str, Any], tools: dict[str, str]) -> str | None:
    """The Python a tool call means, or None. python(code=...) -> the code; web(search=q) ->
    web.search(q); web(question=q) -> web.ask(question=q); final(answer=...) -> final(...)."""
    import importlib
    if name in PYTHON_TOOL_NAMES:
        code = params.get("code")
        if not isinstance(code, str):
            code = next((v for v in params.values() if isinstance(v, str)), None)
        return code
    if name == "final":
        params = final_params(params)
    if name == "final" and isinstance(params.get("answer_var"), str) and params["answer_var"].strip().isidentifier():
        rest = {k: v for k, v in params.items() if k not in ("answer_var", "answer")}
        return f"print(final({params['answer_var'].strip()}{', **' + repr(rest) if rest else ''}))"
    if name in REPL_CALLS and params:
        return f"print({name}(**{params!r}))"
    if name not in tools or not params:
        return None
    mod = importlib.import_module(tools[name])
    funcs = {n: f for n, f in vars(mod).items() if callable(f) and not n.startswith("_")
             and getattr(f, "__module__", "") == mod.__name__}

    def accepts(fn):
        try:
            return set(inspect.signature(funcs[fn]).parameters)
        except (TypeError, ValueError):
            return set()
    called = next((k for k in params if k in funcs), None)
    if called:   # <parameter=search>q: the parameter names the function
        rest = {k: v for k, v in params.items() if k != called and k in accepts(called)}
        args = [repr(params[called])] + [f"{k}={v!r}" for k, v in rest.items()]
    else:        # <parameter=question>q: the function whose arguments these are
        fits = [f for f in funcs if params.keys() <= accepts(f)] or \
               [f for f in funcs if next(iter(params)) in accepts(f)]
        if not fits:
            return None
        called = PRIMARY.get(name) if PRIMARY.get(name) in fits else fits[0]
        args = [f"{k}={v!r}" for k, v in params.items() if k in accepts(called)]
    return f"print({name}.{called}({', '.join(args)}))"


def translate_tool_calls(text: str, tools: dict[str, str]) -> tuple[list[str], list[str]]:
    """(code blocks, calls it couldn't map) for native-template tool calls written as text.
    <function=web><parameter=search>q -> web.search('q'); <parameter=question>q -> web.ask(question='q')."""
    codes, failed = [], []
    for name, body in XML_CALL_RE.findall(text):
        if name == "python":
            continue   # TOOL_CALL_RE's job
        code = call_to_code(name, {k: v for k, v in XML_PARAM_RE.findall(body)}, tools)
        (codes.append(code) if code else failed.append(name))
    return codes, failed


def call_args(arguments: str) -> dict[str, Any]:
    """A native call's arguments. Some servers send bad escapes in long code: fall back to
    pulling the code string out by hand, then to the raw text as code."""
    try:
        args = json.loads(arguments or "{}")
        return args if isinstance(args, dict) else {"code": str(args)}
    except ValueError:
        m = (re.search(r'"code"\s*:\s*"(.*)"\s*}\s*$', arguments or "", re.DOTALL)
             # cut off mid-string (a stopped runaway call): what came before the cut
             or re.search(r'^\s*\{\s*"code"\s*:\s*"(.*?)(?:"\s*\}?\s*)?$', arguments or "", re.DOTALL))
        if m:
            s = m.group(1).rstrip("\\")
            try:
                return {"code": json.loads('"' + s + '"')}
            except ValueError:
                import warnings
                with warnings.catch_warnings():   # bad escapes are the reason we're here
                    warnings.simplefilter("ignore")
                    return {"code": s.encode().decode("unicode_escape", "ignore")}
        return {"code": arguments}


def valid_args(arguments: str) -> str:
    """A tool call's arguments as valid JSON, for sending the conversation back. Live (Gemma 4 in LM
    Studio): a call cut off mid-string was kept as is, and the server then answered every later request
    with HTTP 500 (it renders past calls through the model's chat template). Invalid: what was run."""
    try:
        json.loads(arguments or "{}")
        return arguments or "{}"
    except ValueError:
        return json.dumps(call_args(arguments))


def msg_code(m: dict) -> list[str]:
    """The code an assistant message ran, whatever form it came in."""
    if m.get("role") != "assistant":
        return []
    calls = [call_args(c.get("function", {}).get("arguments", "")) for c in m.get("tool_calls") or []]
    native = [a["code"] for a in calls if isinstance(a.get("code"), str)]
    text = str(m.get("content", ""))
    return native + (code_blocks(text, bool(native)) or TOOL_CALL_RE.findall(text))


def is_output(m: dict) -> bool:
    """A tool/code result: a tool message, or our 'Output:' user message."""
    return m.get("role") == "tool" or (m.get("role") == "user" and str(m.get("content", "")).startswith("Output:"))


PYTHON_TOOL_NAMES = {"python", "ipython", "code", "run_python", "execute_python", "python_repl"}
PYTHON_TOOL = {"type": "function", "function": {
    "name": "python",
    "description": ("Run Python code in your persistent REPL and see its output (stdout, errors, and the value "
                    "of a trailing expression). Variables, imports and functions persist between calls. The "
                    "REPL's helpers (final, ask_user, save_artifact, peek, the tool modules...) are pre-imported."),
    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to run"}},
                   "required": ["code"]}}}
# Live (ornith, native tools): with `python` as the only tool, it wrote report.md at step 6, then ran
# print("done") for 34 steps saying "I need to call the final tool" - it looked for final() as a tool.
# A tool call can't refer to a REPL variable, so a computed answer passed as `answer` gets retyped by the
# model (live, ARC: a correct 10x10 grid came back as strings "0","2"...; another had a row wrong) -
# `answer_var` hands over the variable itself.
FINAL_TOOL = {"type": "function", "function": {
    "name": "final",
    "description": ("Finish the task and hand over the answer. For anything you computed (data, a list, a table, "
                    "a grid, a long text), pass answer_var: the name of the REPL variable that holds it - it is sent "
                    "exactly, never retyped. Use answer only for a short answer you write yourself. artifacts: paths "
                    "of files to hand over. The same as calling final() in Python."),
    "parameters": {"type": "object", "properties": {
        "answer_var": {"type": "string", "description": "name of the REPL variable holding the answer"},
        "answer": {"type": "string", "description": "a short answer written out"},
        "artifacts": {"type": "array", "items": {"type": "string"}, "description": "file paths to hand over"}}}}}

WORKDIR_MARK = "<your working folder>"   # the worker fills in the real path
PACKAGES = ["numpy", "pandas", "scipy", "PIL", "matplotlib", "requests", "bs4", "sklearn", "networkx",
            "sympy", "openpyxl", "yaml", "playwright"]


def importable(names: list[str] = PACKAGES) -> list[str]:
    """The common third-party packages the REPL can import (it runs this same Python)."""
    import importlib.util
    out = []
    for n in names:
        try:
            if importlib.util.find_spec(n) is not None:
                out.append(n)
        except (ImportError, ValueError):
            pass
    return out


# Adapted from Prime Agent's REPL prompt: the REPL is where the model thinks, not just where it
# runs a final program. ARC bench (live thinking files): Qwen misread a printed grid row and argued
# about a "gap" for 18 KB, and spent 80 KB comparing two hypotheses it never ran.
HOW_YOU_WORK = """\
## How you work
You solve tasks with a persistent Python REPL: {act_intro} You see what it prints, errors, and the
value of a last expression. Everything you define - variables, functions, imports - persists.
- Work one step at a time: break the problem down, run code, look at the result, iterate.
- The REPL is where you think. Keep what you learn in named variables, write small helper
  functions, and test each idea by running it on the data. Don't work through data by hand in
  your head: a line of code is faster, and it doesn't misread.
- Assign what you read or compute to named variables, so you can come back to it without re-reading.
- Print what reveals the pattern, not raw dumps: differences between inputs and outputs, positions
  or values relative to each other, counts per group, a short summary table.
- When you have a candidate rule, answer or fix, write it as a function and run it on every example
  you have, printing only where it's wrong - before thinking further. The mismatches show what to fix.
- To see how a module or function works: help(x), dir(x), inspect.signature(f).
- When you are done, call final(answer, artifacts=[...]) {final_where}: `answer` is what the user
  (or the assistant who asked you) receives. Hand over what you computed from its variable{final_var}
  - never retype data. If the request needs no code, just reply in plain text{plain}.

In the REPL:
- ctx: what you were given (the task lists its keys, if any); look with peek(ctx['key']).
- peek(x, n=2000) -> str: preview any value.
- save_artifact(name, data=None, path=None) -> path: a deliverable for the user. data= is the
    file's content (str/bytes/JSON); to hand over a file you wrote, pass its name as path=
    (save_artifact('report.csv', path='report.csv')) - never as data. load_artifact(ref).
- ask_user(question, options=None) -> str: ask the user and wait (details only they know; confirm
  anything hard to undo, like an order, a payment, sending or deleting).
- say(text): tell the user something now and keep working - above all the answer to a question they
  send while you work (they see nothing of your thinking or code until you finish).
- final(answer, artifacts=None): finish.
- Also there (help(name) shows how): start_job / job_status / job_wait (long commands in the
  background), recall / remember (past conversations; lasting facts about the user), llm_query
  (a sub-question to a model), jev (fast typed yes/no and choice decisions), report_failure(what,
  detail) (log a tool, helper or instruction that misbehaved so the harness can be fixed; then carry on).
Your working folder is private scratch space. Keep printed output short; write large files in
several steps (open(path, 'a'))."""
NATIVE_ACT = dict(act_intro="the `python` tool. Call it with code to run.",
                  final_where="(the `final` tool, or final(...) in your code)", plain=" without calling a tool",
                  final_var=" (the final tool's answer_var='result', or final(result) in code)")
TEXT_ACT = dict(act_intro="to act, reply with ONE ```python block; it runs.",
                final_where="in a python block", plain="", final_var=" (final(result))")

REQUEST_GUIDE = """\
- request(assistant, task, inputs=[artifact paths]) -> {{status, answer, artifacts}}: hand a
  self-contained sub-task to another assistant and wait. They see ONLY the task text and
  the inputs you pass, so write the task completely.
- Independent sub-tasks run IN PARALLEL - prefer this when you need more than one:
  results = request_all([(assistant, task), (assistant2, task2, [inputs])])
  or h = request_async(assistant, task) ... keep working ... r = wait(h)
  Only chain requests when one needs another's output.
Available assistants:
{roster}"""


@dataclass
class WorkerResult:
    answer: str
    artifacts: list[str] = field(default_factory=list)
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    stopped: str = "final"            # final | plain | max_steps | error
    transcript: list[dict[str, str]] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)   # set by the orchestrator (for learning)
    llm: Any = None                                    # the model that ran it (learning reuses it)


def describe_ctx(ctx: dict[str, Any]) -> str:
    """The non-empty ctx keys and their sizes ("" when there is nothing to say)."""
    lines = []
    for k, v in ctx.items():
        if k == "request" or (isinstance(v, (list, tuple, dict, str)) and not v):
            continue
        if isinstance(v, (list, tuple)):
            size = f"list of {len(v)}"
        elif isinstance(v, str):
            size = f"str, {len(v)} chars"
        else:
            size = type(v).__name__
        lines.append(f"- ctx[{k!r}]: {size}")
    return "\n".join(lines)


def build_system(persona: Persona, skills: list[Skill], requestable: dict[str, str],
                 suggested: list[str], tools: list[str] | None = None,
                 mcp_tools: list | None = None, native: bool = False, tool_notes: dict[str, str] | None = None,
                 where: str | None = None) -> str:
    """`tools`: the tool guides Jev picked for this task; the persona's other tools are still
    importable and named in one line. `mcp_tools`: McpTool objects Jev picked. `tool_notes`: a line of
    live state after a picked tool's guide (nango: which accounts are connected)."""
    from .tools import TOOLS
    # the date only (not the time), so the prompt - and the local model's prompt cache - holds all day;
    # without it "next week on Tuesday" was searched as tomorrow (live, 2026-09-28)
    today = time.strftime("%A, %B %d, %Y").replace(" 0", " ")
    parts = [persona.system_prompt.strip() + f"\n\nToday is {today}. Work out relative dates "
             "(\"next Tuesday\", \"in two weeks\") from it."]
    if persona.code:
        # (ARC bench, native tool: step 1 assumed Linux and walked '/' looking for a Windows path)
        pkgs = importable()
        # (the computer's name: live 2026-10-12, asked "my laptop or the hub computer?" mid-task, workers answered "on your
        # laptop" and "on your own computer" from nothing)
        parts.append(f"## Environment\nYour REPL runs Python {sys.version.split()[0]} on {platform.system()} "
                     f"{platform.release()}, on the computer named {platform.node() or 'unknown'}"
                     + ("" if where else " (unless told, you don't know which computer the user is on: don't guess)")
                     + f". It starts in your private working folder, {WORKDIR_MARK}. Paths you "
                     f"are given are {'Windows' if os.name == 'nt' else 'POSIX'} paths you can open as-is."
                     + (f" Installed packages you can import: {', '.join(pkgs)} (and the standard library)."
                        if pkgs else "")
                     # where the user is, when that is not the computer this REPL runs on (Harness.origin): said here,
                     # beside "your REPL runs on", since that is what a worker reads before it opens a file
                     + (f"\n{where}" if where else ""))
        # native: acting is a `python` tool call, as in Prime Agent; text: a fenced block
        guide = HOW_YOU_WORK.format(**(NATIVE_ACT if native else TEXT_ACT))
        if requestable:
            roster = "\n".join(f"    - {n}: {d}" for n, d in requestable.items())
            guide += "\n" + REQUEST_GUIDE.format(roster=roster)
        parts.append(guide)
        picked = [t for t in tools or [] if t in TOOLS]
        if picked:
            notes = tool_notes or {}
            parts.append("## Tools in your REPL\n" + "\n".join(
                TOOLS[t].guide + (f"\n  {notes[t]}" if notes.get(t) else "") for t in picked))
        rest = [t for t in persona.tools if t not in picked]
        if rest:
            parts.append("Also importable in your REPL (help(name) shows how): " + ", ".join(rest))
            notes = tool_notes or {}
            pointers = [TOOLS[t].pointer + (f" {notes[t]}" if notes.get(t) else "") for t in rest
                        if t in TOOLS and TOOLS[t].pointer]
            if pointers:   # (Jev's pick misses some requests: a tool the worker has must not be a tool it cannot find)
                parts[-1] += "\n" + " ".join(pointers)
        if mcp_tools:
            parts.append("## MCP tools (call from the REPL; returns text)\n" + "\n".join(
                f"- {t.signature()}: {t.description[:300]}" for t in mcp_tools))
    if suggested:
        parts.append("Jev (the router) suggests this request likely needs help from: "
                     + ", ".join(suggested) + ". Delegate the parts in their expertise with request().")
    for s in skills:
        body = s.body
        if persona.code and len(body) > SKILL_PROMPT_CHARS:
            body = (body[:SKILL_PROMPT_CHARS] + f"\n\n[... {len(s.body) - SKILL_PROMPT_CHARS} more chars: "
                    f"full text in ctx['skills'][{s.name!r}] - search it rather than printing it all]")
        if s.draft:
            body = ("(Learned automatically from one earlier run and not yet reviewed: treat it as hints, "
                    "and check its claims - e.g. that an API returns what you need - before relying on it.)\n" + body)
        parts.append(f"## Skill: {s.name}\n{body}")
    return "\n\n".join(parts)


class Worker:
    def __init__(self, persona: Persona, llm: LLM, workdir: Path, artifacts: Path,
                 handlers: dict[str, Callable[..., Any]], tools: dict[str, str] | None = None,
                 repl_timeout: float = 120.0, output_cap: int = 4000,
                 on_step: Callable[[str, dict], None] | None = None, env: dict[str, str] | None = None,
                 cancel: threading.Event | None = None, verify: Callable | None = None, max_bounces: int = 2,
                 progress: Callable[[str], float] | None = None, stall_at: float = 0.6, progress_every: int = 8,
                 compact_tokens: int = 60000, keep_steps: int = 3, checkpoint: Path | None = None):
        self.persona = persona
        self.llm = llm
        self.workdir = workdir
        self.artifacts = artifacts
        self.handlers = handlers
        self.tools = tools or {}
        self.repl_timeout = repl_timeout
        self.output_cap = output_cap
        self.on_step = on_step or (lambda kind, data: None)
        self.env = env or {}
        self.cancel = cancel or threading.Event()
        self.verify = verify            # (answer, artifacts, transcript, changed) -> Verdict; None = trust final()
        self.max_bounces = max_bounces
        self._bounces = 0
        self._verdict = None
        self.progress = progress        # (recent work log) -> P(going in circles); None = no check
        self.stall_at, self.progress_every = stall_at, progress_every
        self._stalls = 0
        self._wrap_step: int | None = None   # the step told to report and stop (after two stall strikes)
        self.compact_tokens, self.keep_steps = compact_tokens, keep_steps
        # () -> context.Budget | None, set by the harness: where compaction starts for the model in use, from its context
        # window (jevh/context.py). None, or a model nothing can be asked about: compact_tokens stands.
        self.budget: Callable[[], Any] | None = None
        self.checkpoint = checkpoint    # json of the conversation after every step (for resume)
        self.carry_from: Path | None = None   # the previous turn's folder, when this turn follows up on it (_carry)
        self._task_idx = 0              # index of the task message (compaction rewrites everything after it)
        self._task_text = ""
        self._progress_log = ""         # a pre-Jev checkpoint's log (resume only); new logs are self._entries
        self.harness_notes: list[str] = []   # what the harness did to this worker (progress-check flags): a quick reply to "why was I flagged?" reads these
        self._entries: list[dict] = []  # compacted steps: {brief, full, score, keep}
        self._dropped = 0               # compacted steps left out of the log to fit PROGRESS_CHARS
        self._scores: dict[str, float] = {}   # Jev's score per step text (a skipped compaction isn't re-asked)
        # (old step texts, where the work is now) -> P(the rest of the task needs each step); None = notes only
        self.step_value: Callable[[list[str], str], list[float]] | None = None
        self.keep_at = 0.8
        # context settings (jevh/context.py; off = the behaviour before 2026-10-08), set by the harness
        self.measure = False            # context.measure: size the prompt from the server's own token count, not 3.2 chars a token
        self.archive = False            # context.archive: condensed steps are also saved in full in history.md
        self.log_share = 0.0            # context.log_share: the progress log may use this share of the limit (0: PROGRESS_CHARS)
        self.note_tail = False          # context.note_tail: a one-line note keeps the end of the output as well as its start
        self._cal: tuple[int, int] | None = None   # (prompt tokens the server counted, chars of messages sent) for the last call
        self._seq = 0                   # steps archived so far (the [#n] in the log and history.md)
        self.repl: Repl | None = None      # the orchestrator kills it when the user stops the turn
        # fast tier only: () -> the persona's strong model, swapped in when the gate sends work back
        self.escalate: Callable[[], LLM] | None = None
        self.escalated = False
        self.keep_reasoning = REASONING_CHARS
        self.thinking_watch: Callable[[str], str | None] | None = None   # (reasoning so far) -> why pause, or None
        self._watch_reason: str | None = None
        self._loop_line = ""
        self.watch_every, self.watch_after = 4000, 6000
        self._ran_code = False
        self._pauses = 0   # watch pauses since code last ran: from the 2nd, checks come at 2x watch_after
        self._step = 0
        # messages the user sends while the task runs (steer(); docs/architecture/16-interjection.md)
        self._steers: list[tuple[str, bool]] = []   # (text, replaces the request): join the conversation before the next call
        self._steer_lock = threading.Lock()
        self.interrupt = threading.Event()    # set by steer(): cuts the model's reply in flight (llm.chat interrupt=)
        self.corrections: list[str] = []      # the messages delivered so far
        self._free = 0                        # steps spent hearing the user out: not charged to the step limit
        self.retry_delay = 0.0                # seconds to wait before asking again after a cut reply (interject.retry_delay)
        self._live: list[dict] | None = None  # the conversation the model is given (snapshot() copies it)
        self._doing = ("getting started", time.time())
        self._use(llm)

    def _assistant(self, out) -> dict:
        """The reply as it stays in the conversation, with its hidden reasoning in
        `reasoning_content` (as Prime Agent does). Without it each step re-derived everything:
        ARC bench, one task took 56k thinking tokens over 5 harness steps vs 25k in a single call,
        and a reply that was all thinking lost its progress entirely. LM Studio's Qwen template
        shows it back to the model (measured: a word that existed only in an earlier step's
        reasoning was recalled; without it the model invented one). Kept out of `content`, so
        the gate's work log never counts reasoning as work done."""
        msg = {"role": "assistant", "content": out.text}
        thought = (getattr(out, "reasoning", "") or "").strip()
        if thought and self.keep_reasoning:
            msg["reasoning_content"] = thought[-self.keep_reasoning:]
        return msg

    def _use(self, llm: LLM) -> None:
        self.llm = llm
        try:
            self._llm_cancel = "cancel" in inspect.signature(llm.chat).parameters
        except (TypeError, ValueError):
            self._llm_cancel = False

    # -- messages from the user while the task runs ------------------------------------------------------------
    @property
    def limit(self) -> int:
        """The step limit: the persona's, plus the steps spent hearing the user out (those are not charged)."""
        return self.persona.max_steps + self._free

    def steer(self, text: str, abort: bool = True, replaces: bool = False) -> bool:
        """The user sent a message while this task runs: it joins the conversation before the next model call.
        `abort`: a reply the model is writing right now is cut and asked for again with the message. Code that
        is running is never killed: the message arrives with its output. `replaces`: it swaps the request for
        another (the task, and what the final() gate expects, are then that alone). Safe to call from any thread."""
        text = (text or "").strip()
        if not text:
            return False
        with self._steer_lock:   # (set under the lock: set after it, it could land just after _deliver_steers had
            self._steers.append((text, replaces))   # taken the message and cleared it, cutting the next reply for
            if abort:                                # nothing - a wasted model call)
                self.interrupt.set()
        return True

    def _steers_pending(self) -> bool:
        with self._steer_lock:
            return bool(self._steers)

    def take_unapplied(self) -> list[str]:
        """Messages that never reached the model (the task ended first); the caller makes them the next request."""
        with self._steer_lock:
            left, self._steers = self._steers, []
            self.interrupt.clear()
        return [text for text, _ in left]

    def _deliver_steers(self, messages: list[dict]) -> None:
        """Before a model call: each pending message joins the conversation as a user message (at the end)."""
        with self._steer_lock:
            pending, self._steers = self._steers, []
            self.interrupt.clear()
        for text, replaces in pending:
            messages.append({"role": "user", "content": (STEER_REPLACE if replaces else STEER_NOTE).format(text=text)})
            self.corrections.append(text)
            if self._task_text:   # compaction rebuilds the task message from this: the message survives it
                self._task_text += "\n\n" + UPDATED.format(text=text)
            self.on_step("steer", {"persona": self.persona.name, "step": self._step, "text": text,
                                   "replaces": replaces})

    def _redirect(self, messages: list[dict], out, step: int) -> None:
        """A model reply the user's message cut off, or crossed: what it had written stays as context, but
        nothing in it runs (the user is changing the task) and the step it cost is not charged."""
        text = (out.text or "").strip()
        if text.count("```") % 2:   # cut inside a code block: the half block goes
            text = text[:text.rfind("```")].rstrip()
        thought = (getattr(out, "reasoning", "") or "").strip()
        if text or thought:
            msg = self._assistant(out)
            msg["content"] = f"{text}\n{CUT_REPLY}" if text else CUT_REPLY
            messages.append(msg)
        self._free += 1
        self.on_step("redirect", {"persona": self.persona.name, "step": step, "kept": len(text) + len(thought)})

    def _after_cut(self) -> None:
        """Before the model is asked again with the user's message. llama.cpp frees the cancelled slot a moment after
        the hang-up; a retry that arrives first starts on the other slot with nothing cached and the whole prompt is
        read again (measured: 4 of 8 retries). A Stop still ends the wait at once."""
        if self.retry_delay > 0:
            self.cancel.wait(self.retry_delay)

    def _note(self, doing: str) -> None:
        self._doing = (doing, time.time())

    def status(self) -> dict[str, Any]:
        """What the worker is doing right now (safe to read from another thread): a status answer is made of this."""
        doing, since = self._doing
        return {"persona": self.persona.name, "step": self._step, "max_steps": self.limit, "doing": doing,
                "seconds": round(time.time() - since, 1), "corrections": list(self.corrections),
                "pending": self._steers_pending()}

    def snapshot(self) -> list[dict]:
        """A copy of the conversation as the model has it so far, system prompt first."""
        return [dict(m) for m in list(self._live or [])]

    def digest(self, last: int = 6) -> str:
        """The task so far in a few lines, for a quick answer about it: steps compaction already condensed, then
        the last `last` steps (the code that ran, what came back)."""
        msgs = self.snapshot()
        body = msgs[self._task_idx + 1:] if 0 < self._task_idx < len(msgs) else msgs[1:]
        parts = [self._render_log()[-1500:]] if (self._entries or self._progress_log) else []
        older = len(_steps(body)) - last
        if older > 0:
            parts.append(f"({older} earlier steps not shown)")
        parts.append(self._condense(body[-2 * last:]).strip())
        if self.harness_notes:
            parts.append("What the harness did to this task:\n" + "\n".join(f"- {n}" for n in self.harness_notes[-4:]))
        return "\n".join(p for p in parts if p).strip()

    def tool_schemas(self) -> list[dict] | None:
        """The tools the model is offered right now: a prompt that shares this worker's start must offer the same."""
        if not self.native:
            return None
        return [FINAL_TOOL] if self._wrap_step is not None else [PYTHON_TOOL, FINAL_TOOL]

    def run(self, task: str, ctx: dict[str, Any], system: str,
            history: list[tuple[str, str]] | None = None, resume: dict | None = None) -> WorkerResult:
        """`history`: the (user, assistant) turns this message follows up on, as real chat turns.
        `resume`: a checkpoint of an interrupted run of this task: its conversation continues."""
        system = system.replace(WORKDIR_MARK, str(self.workdir))
        prior = []
        for u, a, *notes in history or []:
            a = a[:HISTORY_CHARS]
            if notes and notes[0]:   # so "where did you get that?" is answered from what happened
                a += "\n\n" + NOTES_HEADER + "\n" + notes[0][:NOTES_CHARS]
            prior += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        if not self.persona.code:
            return self._chat(task, ctx, system, prior)
        if resume:
            messages = [{"role": "system", "content": system}, *[m for m in resume["messages"] if m["role"] != "system"]]
            self._task_idx, self._task_text = resume.get("task_idx", 1), resume.get("task_text", task)
            comp = resume.get("compact")
            if comp:
                self._entries, self._dropped = comp.get("entries", []), comp.get("dropped", 0)
                self._seq = comp.get("seq", 0)
            else:
                self._progress_log = resume.get("progress_log", "")
            # (the resume note goes in once the REPL state is restored: it says what came back)
        else:
            listed = describe_ctx(ctx)
            self._task_text = (task + (f"\n\n(In your REPL:\n{listed})" if listed else "")
                               + ("" if getattr(self.llm, "unlimited_thinking", False) else FIRST_STEP_NUDGE))
            messages = [{"role": "system", "content": system}, *prior, {"role": "user", "content": self._task_text}]
            self._task_idx = len(messages) - 1
        res = WorkerResult(answer="")
        self._last_err, self._err_streak = None, 0
        with Repl(self.workdir, self.artifacts, self.handlers, self.tools,
                  self.repl_timeout, self.output_cap, self.env) as repl:
            self.repl = repl
            repl.set("ctx", ctx)
            if resume:
                self._add_resume_note(messages, repl)
            elif self.carry_from is not None:
                self._carry(messages, repl)
            self._live = messages
            step = 0
            while step < self.limit:   # (a step spent hearing the user out raises the limit by one)
                step += 1
                self._check()
                self._deliver_steers(messages)   # the user's messages go in last: the prompt before them is unchanged
                self._compact(messages, step)
                self._report_context(messages, step)
                self._step = step
                self._note("the model is writing its next step")
                out = self._llm(messages)
                self._calibrate(messages, out)
                res.steps, res.input_tokens, res.output_tokens = (
                    step, res.input_tokens + out.input_tokens, res.output_tokens + out.output_tokens)
                if getattr(out, "redirected", False) or self._steers_pending():
                    # the user sent a message while the model was replying: that reply is not acted on
                    self._redirect(messages, out, step)
                    self._after_cut()
                    continue
                messages.append(self._assistant(out))
                # (a stream stopped inside a runaway tool call still carries the call: it is acted on below)
                if getattr(out, "interrupted", False) and not getattr(out, "tool_calls", None):
                    # the thinking watch paused a long think: hand-work code does exactly, or theory after theory.
                    # Either way the way forward is code (reasoning kept). Live: a bare "test it now" was answered
                    # with more thinking, so the message quotes what it was doing; and when searching, "compute
                    # this" had nothing to compute (ignored 4 in a row) - so it also says what to do when unsure.
                    self._pauses += 1
                    reason = self._watch_reason or "testable"
                    why = WATCH_WHY.get(reason, WATCH_WHY["testable"])
                    self.on_step("output", {"persona": self.persona.name, "step": step, "error": False,
                                            "output": f"(paused a long think: {why} - asked to test it in code)"})
                    reasoning = (getattr(out, "reasoning", "") or "").strip()
                    # a model that reasons in its reply keeps what it wrote (its reasoning); a loop, only its start
                    messages[-1]["content"] = (out.text[:REPLY_KEPT] + "\n[... stopped here by the harness]"
                                               if out.text.strip() and not reasoning else "(paused to test an idea)")
                    tail = (reasoning or out.text).strip()[-600:].replace("```", "")
                    again = (f" This is pause {self._pauses} in a row: after the last one you went back to "
                             "reasoning in your head instead of running code." if self._pauses > 1 else "")
                    if reason == "repeating":
                        messages.append({"role": "user", "content": (
                            f"[The harness stopped your reply: you were repeating yourself - \"{self._loop_line[:160]}"
                            f"\" came up again and again. Whatever you keep saying you will do next, do it now: your "
                            f"next reply should {self._act.replace('reply with', 'be')} that does it (or the final call, "
                            "if you already have the answer). No more checking in your head.]"
                            + self._steps_left_note(step))})
                        continue
                    messages.append({"role": "user", "content": (
                        f"[The harness paused your thinking: {why}.{again} Your reasoning so far is kept. It ended "
                        f"with:\n...{tail}\n\nYour next reply should {self._act.replace('reply with', 'be')}:\n"
                        "- if you were checking one idea against the data, compute exactly that - the values, "
                        "positions, sizes, counts or rule - and print the result;\n"
                        "- if you are not sure what the rule is, write your best current guess as a function, run "
                        "it on the examples and print where its output differs from the expected one - the "
                        "mismatches show what to fix.\nKeep your thinking short: decide what to run, then run it.]"
                        + self._steps_left_note(step))})
                    continue
                calls = [c for c in (getattr(out, "tool_calls", None) or []) if c.get("name")]
                call_ids: list[str] = []
                note = ""
                if calls:   # native tool calls: the python tool (or a REPL module called by name)
                    messages[-1]["tool_calls"] = [{"id": c["id"], "type": "function",
                                                   "function": {"name": c["name"], "arguments": valid_args(c["arguments"])}}
                                                  for c in calls]
                    call_ids = [c["id"] for c in calls]
                    blocks, failed = [], []
                    for c in calls:
                        code = call_to_code(c["name"], call_args(c["arguments"]), self.tools)
                        (blocks.append(code) if code else failed.append(c["name"]))
                    if failed:
                        note = (f"\n(Not run: there is no `{'`, `'.join(failed)}` tool. The tools are `python` and "
                                "`final`; the REPL's modules and helpers are pre-imported inside `python`.)")
                    if not blocks:
                        messages += self._replies(call_ids, note.strip("\n()") + self._steps_left_note(step))
                        continue
                else:
                    blocks = code_blocks(out.text, self.native) or TOOL_CALL_RE.findall(out.text)
                    if not blocks and "<function=" in out.text:
                        blocks, failed = translate_tool_calls(out.text, self.tools)
                        if blocks:
                            note = ("\n(Your <tool_call> text ran as Python: " + "; ".join(blocks) +
                                    f". Next time {self._act}.)")
                        else:
                            messages.append({"role": "user", "content": (
                                f"Nothing ran: <tool_call> text ({', '.join(failed)}) isn't run here. To act, "
                                f"{self._act}; the tools are modules already imported in your REPL, e.g. "
                                "print(web.ask(\"the question\"))." + self._steps_left_note(step))})
                            continue
                    if "[output cut off at max_tokens" in out.text[-120:] or (
                            not blocks and OPEN_FENCE_RE.search(out.text)):
                        self.on_step("output", {"persona": self.persona.name, "step": step,
                                                "output": "(reply cut off - asked to use smaller steps)",
                                                "error": True})
                        messages.append({"role": "user", "content": (
                            "Your reply was cut off before the code was complete, so nothing ran. "
                            "Work in smaller steps: write big files in parts (open(..., 'a')) across "
                            "several steps, and keep each reply short." + self._steps_left_note(step))})
                        continue
                    if not blocks and not out.text.strip():
                        # all thinking, no reply (Qwen, ARC bench: ~10 min of reasoning then nothing): not an
                        # answer to check, just a turn to continue
                        self.on_step("output", {"persona": self.persona.name, "step": step,
                                                "output": "(empty reply - asked to continue)", "error": True})
                        messages[-1]["content"] = "(no reply text)"
                        messages.append({"role": "user", "content": (
                            f"Your last reply was empty (only thinking, no text). Continue from where your "
                            f"reasoning got to: {self._act}, or call final() if you are done."
                            + self._steps_left_note(step))})
                        continue
                    if not blocks:
                        if self._bounce(out.text, [], [], messages, step, "(no code was run)\n\n"):
                            continue
                        res.answer, res.stopped = self._with_unresolved(out.text), "plain"
                        break
                code = "\n\n".join(blocks)
                self._ran_code, self._pauses = True, 0
                self.on_step("code", {"persona": self.persona.name, "step": step, "code": code})
                self._note("running code: " + ([ln.strip() for ln in code.splitlines() if ln.strip()] or [""])[0][:100])
                ex = repl.run(code)
                if ex.timed_out:
                    self._restore_after_timeout(repl, ex)
                if ex.usage:
                    self.on_step("tool_usage", {"persona": self.persona.name, **ex.usage})
                self._check()  # a stop kills the REPL mid-run: don't read that as the code failing
                self._note("reading the output of that code")
                self.on_step("output", {"persona": self.persona.name, "step": step,
                                        "output": ex.output, "error": ex.error})
                if ex.final is not None and self._steers_pending():
                    # the user sent a message while the code ran: nothing is handed over before they are heard
                    messages += self._replies(call_ids, (ex.output or "(no output)") + STEER_HELD)
                    self._free += 1
                    self._save(messages, step, False)
                    continue
                if ex.final is not None:
                    artifacts = ex.final["artifacts"] or self._harvest()
                    # (no artifacts but files written in private scratch: ship them, or whoever asked
                    # never sees the deliverable - seen live: a designer's index.html)
                    prefix = f"Output:\n{ex.output or '(no output)'}\n\n"
                    if call_ids:   # a tool call gets its tool result first; a bounce follows as a user turn
                        messages += self._replies(call_ids, ex.output or "(no output)")
                        prefix = ""
                    if self._bounce(ex.final["answer"], artifacts, ex.final.get("changed") or [], messages, step,
                                    prefix):
                        continue
                    res.answer, res.artifacts, res.stopped = (self._with_unresolved(ex.final["answer"]),
                                                              artifacts, "final")
                    break
                messages += self._replies(call_ids, (ex.output or "(no output)") + note
                                          + self._stuck_note(ex.output, ex.error) + self._steps_left_note(step))
                self._save(messages, step, False)
                if (self._wrap_step is not None and step > self._wrap_step) or self._stalled(messages, step):
                    res.artifacts = self._harvest()
                    same = repeated_result(messages)
                    res.answer = (f"[{self.persona.name} stopped: its recent steps kept repeating the same attempt "
                                  f"with the same result, even after being told. Files it wrote were collected; "
                                  f"you can resume it with more guidance.]\n"
                                  + (f"Its last result, printed in each of its last steps (not handed over as its "
                                     f"answer):\n{same[:2000]}" if same else
                                     _last_output(messages)[-2000:]))
                    res.stopped = "stalled"
                    break
            else:
                res.artifacts = self._harvest()
                res.answer = (f"[{self.persona.name} reached its step limit ({self.persona.max_steps}) without "
                              f"calling final(); files it wrote were collected as artifacts, and you can resume "
                              f"it to continue where it stopped]\n"
                              + messages[-2]["content"][-2000:])
                res.stopped = "max_steps"
            if res.stopped in ("final", "plain"):   # the state a follow-up turn starts from (carry_from)
                self._snapshot()
        res.transcript = messages
        self._save(messages, res.steps, True, res.stopped)
        return res

    # -- stamina ---------------------------------------------------------------------
    def _limit(self) -> tuple[int, dict[str, Any]]:
        """(the estimated prompt size past which older steps condense, what is known of the model's window). Asked
        each step: the model may have been loaded with another context, or swapped for the strong one."""
        if self.budget is not None:
            try:
                b = self.budget()
            except Exception:  # noqa: BLE001 - never let a question about the window fail a step
                b = None
            if b is not None:
                return b.compact, {k: v for k, v in b.public().items() if k != "compact"}
        return self.compact_tokens, {}

    def _report_context(self, messages: list[dict], step: int) -> None:
        """What the prompt about to be sent is made of (estimated tokens), for the dashboard's
        context ring: out of the compaction limit, where older steps get condensed."""
        system = str(messages[0].get("content", "")) if messages and messages[0]["role"] == "system" else ""
        skills_at = system.find("\n\n## Skill: ")
        skills_at = len(system) if skills_at < 0 else skills_at
        tools = 0
        for mark in ("\n\n## Tools in your REPL", "\n\n## MCP tools"):
            a = system.find(mark, 0, skills_at)
            if a >= 0:   # up to the next section of another kind
                ends = [system.find(m, a + len(mark), skills_at) for m in ("\n\nAlso importable", "\n\n## MCP tools",
                                                                            "\n\nJev (the router) suggests")]
                tools += min([e for e in ends if e >= 0] + [skills_at]) - a
        task = self._task_text if self._task_idx < len(messages) else ""
        task_msg = _msg_chars(messages[self._task_idx]) if self._task_idx < len(messages) else 0
        work = output = 0
        for m in messages[self._task_idx + 1:]:
            if m["role"] == "assistant":
                work += _msg_chars(m) + len(str(m.get("tool_calls") or ""))
            else:
                output += _msg_chars(m)
        chars = {"instructions": skills_at - tools, "skills": len(system) - skills_at, "tools": tools,
                 "task": len(task) + sum(_msg_chars(m) for m in messages[1:self._task_idx]),
                 "work": work + max(0, task_msg - len(task)),   # the progress log is condensed past work
                 "output": output}
        limit, window = self._limit()
        self.on_step("context", {"persona": self.persona.name, "step": step, "limit": limit,
                                 "parts": {k: round(v * self._per_char()) for k, v in chars.items()}, **window})

    def _compact(self, messages: list[dict], step: int) -> None:
        """Keep the conversation inside the model's context: past the compaction limit (a share of the
        model's context window, or compact_tokens), older steps
        become a progress log in the task message; the last `keep_steps` steps stay verbatim.
        Jev reads each old step: the ones the rest of the task still needs (values, names, sources,
        what failed and why) stay in full, the rest become one-line notes. No model call: a summary
        by the worker's model cost minutes on a thinking model and lost exact values."""
        size = self._tokens(messages)
        limit, _ = self._limit()
        if size <= limit:
            return
        body = messages[self._task_idx + 1:]
        keep = 2 * self.keep_steps
        if len(body) <= keep:
            return
        cut = len(body) - keep
        while cut < len(body) and body[cut]["role"] != "assistant":   # the tail starts with the model's turn
            cut += 1   # (and never splits a tool call from its tool results)
        old, recent = body[:cut], body[cut:]
        if not old:
            return
        per = self._per_char()
        old_chars = sum(_msg_chars(m) for m in old)
        if old_chars * per < limit * 0.1:
            return  # not enough old material to be worth condensing (e.g. a big system prompt)
        now = self._condense(recent)[-2000:]
        groups = _steps(old)
        new = [{"brief": self._condense(g), "full": self._full(g), "score": None, "keep": False} for g in groups]
        if self.archive:   # [#n]: the step's number in history.md, where it is kept in full
            for i, e in enumerate(new):
                e["n"] = self._seq + i + 1
                e["brief"] = f"[#{e['n']}] " + e["brief"]
                e["full"] = f"[#{e['n']}] " + e["full"]
        self._score(new, now)
        # Aim well under the limit (live: compacting only down to it re-triggered every step, and each
        # compaction rewrites the prompt's start, so the model server re-reads all of it). What the log
        # may use is what the target leaves after the system prompt, task and recent steps.
        rest = sum(_msg_chars(m) for m in messages[:self._task_idx] + recent) + len(self._task_text)
        soft = max(MIN_FULL_LOG, int(limit * COMPACT_TARGET / per) - rest)
        entries = [dict(e) for e in self._entries] + new
        dropped = self._fit(entries, now, soft, self._log_cap(limit))
        log_chars = len(_render(entries, self._dropped + dropped, self._progress_log))
        if log_chars > 0.9 * (old_chars + len(self._render_log())):
            return  # would barely save anything (live: all-needed steps in full grew the context)
        self._entries, self._dropped = entries, self._dropped + dropped
        if self.archive:
            self._save_history(groups)
        task_msg = {"role": "user", "content": self._task_text + "\n\n[Progress so far - earlier steps condensed to "
                    "save context (the ones still needed are kept in full); your REPL variables and files are "
                    "intact." + self._history_note() + "\n" + self._render_log() + "]"}
        before = int(size)
        messages[:] = messages[:self._task_idx] + [task_msg] + recent
        after = int(self._tokens(messages))
        self.on_step("compact", {"persona": self.persona.name, "step": step, "tokens_before": before,
                                 "tokens_after": after, "full": sum(e["keep"] for e in self._entries),
                                 "notes": sum(not e["keep"] for e in self._entries), "dropped": self._dropped})

    # -- sizing the prompt ---------------------------------------------------------------------------------
    def _calibrate(self, messages: list[dict], out) -> None:
        """After a call: remember how many tokens the server counted for the messages just sent."""
        n = int(getattr(out, "input_tokens", 0) or 0)
        if self.measure and n > 0:
            self._cal = (n, sum(_msg_chars(m) for m in messages))

    def _per_char(self) -> float:
        """Tokens per character of conversation: the server's own count from the last call where it gave one
        (`measure`), else the fixed estimate. Text, code and numbers differ by a factor of three or more (measured
        2026-10-08 on Qwen: prose 4.5 chars a token, code 4.0, JSON 3.1, digit grids 1.0)."""
        if self._cal and self._cal[1] > 0:
            return min(max(self._cal[0] / self._cal[1], 0.1), 1.5)
        return 1 / CHARS_PER_TOKEN

    def _tokens(self, messages: list[dict]) -> float:
        """The prompt's size in tokens: what the server counted last call, plus the messages added or removed since."""
        chars = sum(_msg_chars(m) for m in messages)
        if self._cal and self._cal[1] > 0:
            return max(0.0, self._cal[0] + (chars - self._cal[1]) * self._per_char())
        return chars / CHARS_PER_TOKEN

    def _log_cap(self, limit: float) -> int:
        """Chars the progress log may use: PROGRESS_CHARS, or (`log_share`) a share of the compaction limit."""
        if self.log_share > 0:
            return max(PROGRESS_CHARS, int(limit * self.log_share / self._per_char()))
        return PROGRESS_CHARS

    # -- the archive ---------------------------------------------------------------------------------------
    def _history_path(self) -> Path:
        return Path(self.workdir) / "history.md"

    def _history_note(self) -> str:
        if not self.archive:
            return ""
        return (f" Every earlier step is saved in full, in order, in {self._history_path()} (numbers match the [#n] "
                "tags below): search or read that file with Python for an exact value, file content or error "
                "before reading a file again or redoing work.")

    def _save_history(self, groups: list[list[dict]]) -> None:
        """Append the steps just condensed to history.md in full: the log keeps notes, this keeps everything."""
        lines = []
        for i, g in enumerate(groups):
            thought, codes, outs = "", [], []
            for m in g:
                if m["role"] == "assistant":
                    thought = (str(m.get("reasoning_content", "")) or CODE_RE.sub("", str(m.get("content", "")))).strip()
                    codes += msg_code(m)
                else:
                    outs.append(str(m.get("content", "")).split("[Not accepted yet", 1)[0].replace("Output:\n", "", 1).strip())
            lines.append(f"\n## [#{self._seq + i + 1}]\n"
                         + (f"THOUGHT (end of it): ...{thought[-1500:]}\n" if thought else "")
                         + "CODE:\n" + "\n".join(codes).strip() + "\nOUTPUT:\n" + "\n".join(outs).strip() + "\n")
        try:
            Path(self.workdir).mkdir(parents=True, exist_ok=True)
            with open(self._history_path(), "a", encoding="utf-8") as f:
                f.write("".join(lines))
        except OSError:
            pass   # a convenience: the log still has the notes
        self._seq += len(groups)

    def _score(self, entries: list[dict], now: str, cached: bool = True) -> None:
        """Jev: does the rest of the task still need this step? (evals/compact_cases.py). A step
        scored before (a compaction that was skipped) keeps its score unless `cached` is False."""
        todo = [e for e in entries if not (cached and e["full"] in self._scores)]
        if self.step_value is not None and todo:
            try:
                for e, s in zip(todo, self.step_value([e["full"] for e in todo], now)):
                    self._scores[e["full"]] = s
            except Cancelled:
                raise
            except Exception:  # noqa: BLE001 - Jev down: notes only this time
                pass
        for e in entries:
            s = self._scores.get(e["full"])
            e["score"], e["keep"] = s, s is not None and s >= self.keep_at

    def _fit(self, entries: list[dict], now: str, soft: int, cap: int = 0) -> int:
        """Shrink the log in place: past `soft` chars, re-check the full steps against where the
        work is now and turn the least needed into notes; past PROGRESS_CHARS, also leave out the
        least needed notes (oldest first on ties). -> how many were left out."""
        def size():
            return sum(len(e["full"] if e["keep"] else e["brief"]) for e in entries)
        cap = cap or PROGRESS_CHARS
        limit = min(soft, cap)
        if size() > limit:
            full = [e for e in entries if e["keep"]]
            self._score(full, now, cached=False)
            for e in sorted((e for e in full if e["keep"]), key=lambda e: e["score"] or 0.0):
                if size() <= limit:
                    break
                e["keep"] = False
        dropped = 0
        for e in sorted(entries, key=lambda e: e["score"] or 0.0):
            if size() <= cap:
                break
            entries.remove(e)
            dropped += 1
        return dropped

    def _render_log(self) -> str:
        return _render(self._entries, self._dropped, self._progress_log)

    @staticmethod
    def _full(group: list[dict]) -> str:
        """One old step as Jev reads it, and as the log keeps it when it is still needed."""
        thought, codes, outs = "", [], []
        for m in group:
            text = str(m.get("content", ""))
            if m["role"] == "assistant":
                thought = (str(m.get("reasoning_content", "")) or CODE_RE.sub("", text)).strip()
                codes += msg_code(m)
            else:
                outs.append(text.split("[Not accepted yet", 1)[0].replace("Output:\n", "", 1).strip())
        code, out = "\n".join(codes).strip(), "\n".join(outs).strip()
        return ((f"THOUGHT: ...{thought[-FULL_THOUGHT:]}\n" if thought else "")
                + f"CODE:\n{_clip_mid(code, FULL_CODE)}\nOUTPUT:\n{_clip_mid(out, FULL_OUT)}")

    def _condense(self, old: list[dict]) -> str:
        lines = []
        for m in old:
            text = str(m.get("content", ""))
            if m["role"] == "assistant":
                codes = msg_code(m)
                prose = CODE_RE.sub("", text).strip()
                if prose:
                    lines.append("- note: " + prose[:300].replace("\n", " "))
                for c in codes:
                    head = c.strip().splitlines()
                    lines.append("- ran: " + " | ".join(head[:8])[:600] + (" ..." if len(head) > 8 else ""))
            else:
                out = text.split("[Not accepted yet", 1)[0].replace("Output:\n", "", 1).strip()
                if self.note_tail and len(out) > 400:   # the start says what it is, the end what came of it
                    out = out[:260] + " [...] " + out[-140:]
                    lines.append("  -> " + out.replace("\n", " "))
                else:
                    lines.append("  -> " + out[:400].replace("\n", " ") + (" ..." if len(out) > 400 else ""))
        return "\n".join(lines) + "\n"

    def _stalled(self, messages: list[dict], step: int) -> bool:
        """Every `progress_every` steps, Jev looks at the recent steps: going in circles? The first
        time the worker is told to change approach; a second time in a row stops it."""
        if self.progress is None or self.progress_every <= 0 or step % self.progress_every:
            return False   # (progress_every: 0 switches the check off; it raised ZeroDivisionError at step 1)
        recent = messages[-2 * self.progress_every:]
        # a page it hadn't reached before is progress, however alike the steps look (live: Booking.com's first
        # strike came right after the dates were picked by hand, BlazeDemo's right after the flight was chosen)
        new = new_pages(messages[self._task_idx + 1:-len(recent)], recent)
        if new:
            self._stalls = 0
            self.on_step("progress_check", {"persona": self.persona.name, "step": step, "circles": None,
                                            "stuck": False, "strike": 0, "new_pages": new[:3]})
            return False
        log = self._condense(recent)
        try:
            circles = self.progress(log)
        except Exception:  # noqa: BLE001 - Jev down: no check this time
            return False
        stuck = circles >= self.stall_at
        # a second strike stops the task only when its steps also printed what earlier ones had: reading a new memo each
        # step is not a loop, however alike the steps look (long-horizon bench 2026-10: all 5 runs the check stopped had 0
        # repeated outputs and were reading the task's pages; the two real loops in the failure log, clicks on another
        # computer that kept timing out, had 3 and 2 of 8). Held, it is told to change approach again.
        held = stuck and self._stalls >= 1 and \
            repeated_outputs(messages[self._task_idx + 1:-len(recent)], recent) < STOP_REPEATS
        strike = self._stalls + (1 if stuck and not held else 0)
        self.on_step("progress_check", {"persona": self.persona.name, "step": step, "circles": round(circles, 3),
                                        "stuck": stuck, "strike": strike, **({"held": True} if held else {})})
        if stuck:
            self.harness_notes.append(
                f"step {step}: the progress check (Jev, every {self.progress_every} steps) judged the last "
                f"{self.progress_every} steps to be the same attempt repeating with the same result "
                f"(score {circles:.2f}, flagged from {self.stall_at:.2f}); strike {strike} of 2, "
                + ("not stopped: its outputs were new. " if held else
                   "the task is stopped on the second. " if strike == 1 else "the task is being stopped. ")
                + "A check is not a judgement that the steps were wasted: re-running a test to read its output "
                "can look like a repeat.")
        if not stuck:
            self._stalls = 0
            return False
        self._stalls = strike
        if self._stalls >= 2:
            if self._wrap_step is not None:
                return True
            # one last step to report: stopping outright handed the user raw REPL output ("105 / Are You
            # Human?") where the worker knew what it had found and what blocked it (live, browser tasks)
            self._wrap_step = step
            messages[-1] = {**messages[-1], "content": messages[-1]["content"] + (
                "\n\n[The harness is stopping this task: your steps keep repeating without progress. This is "
                "your last step. Do not try anything else: call final() now with what you found, what blocked "
                "you, and exactly where you stopped (the page URL, if you were in a browser).]")}
            return False
        same = repeated_result(messages)
        if same:   # done but not handing it over (live, Gemma 4: a correct grid printed 3 times, then stopped)
            note = (f"\n\n[The harness's progress check: your last steps printed the same result each time:\n"
                    f"{same[:600]}\nIf that is your answer, stop checking and hand it over now - "
                    f"final(...) with it (the final tool's answer_var, or final(result) in code). If it is not, "
                    "change the approach.]")
        else:
            note = ("\n\n[The harness's progress check: your recent steps look like the same attempt repeating with "
                    "the same result. Step back: find the root cause (read the error, inspect the data or the docs), "
                    "try a genuinely different approach, or ask_user() for what you need. If it can't be done, call "
                    "final() and say what blocks it.]")
        messages[-1] = {**messages[-1], "content": messages[-1]["content"] + note}
        return False

    @property
    def _state_path(self) -> Path | None:
        return self.checkpoint.with_name("repl_state.pkl") if self.checkpoint else None

    def _restore_after_timeout(self, repl, ex) -> None:
        """A timed-out block restarted the REPL empty: put back the snapshot `_save` took after the previous
        step, and say so (nothing to restore, e.g. in the first step: the REPL's own message stands)."""
        info = repl.restore(self._state_path) if self._state_path else {}
        if not (info.get("restored") or info.get("defs")):
            return
        lost = info.get("failed") or []
        ex.output = TIMEOUT_RESTORED_NOTE.format(
            secs=repl.timeout, vars=", ".join(info.get("restored") or []) or "none",
            defs=", ".join(info.get("defs") or []) or "none",
            lost=f" Not restored (recreate them if needed): {', '.join(lost)}." if lost else "")

    def _carry(self, messages: list[dict], repl) -> None:
        """A follow-up turn: the previous turn's REPL state comes back, and the task message names its folder (the note
        goes in the task text, so compaction keeps it)."""
        prev = Path(self.carry_from)
        state = prev / "repl_state.pkl"
        info = repl.restore(state) if state.is_file() else {}
        extra = ""
        if info.get("restored") or info.get("defs"):
            lost = info.get("failed") or []
            extra = CARRIED_STATE.format(vars=", ".join(info.get("restored") or []) or "none",
                                         defs=", ".join(info.get("defs") or []) or "none",
                                         lost=f" Not brought back: {', '.join(lost)}." if lost else "")
        note = "\n\n" + CARRIED_NOTE.format(dir=prev, state=extra)
        self._task_text += note
        if 0 < self._task_idx < len(messages):
            messages[self._task_idx] = {**messages[self._task_idx],
                                        "content": str(messages[self._task_idx]["content"]) + note}

    def _add_resume_note(self, messages: list[dict], repl) -> None:
        info = repl.restore(self._state_path) if self._state_path else {}
        if info.get("restored") or info.get("defs"):
            lost = info.get("failed") or []
            note = RESTORED_NOTE.format(vars=", ".join(info.get("restored") or []) or "none",
                                        defs=", ".join(info.get("defs") or []) or "none",
                                        lost=f" Not restored (recreate them if needed): {', '.join(lost)}." if lost else "")
        else:
            note = RESUME_NOTE
        if messages[-1]["role"] == "user":
            messages[-1] = {"role": "user", "content": messages[-1]["content"] + "\n\n" + note}
        else:
            messages.append({"role": "user", "content": note})

    def _snapshot(self) -> None:
        if self.repl is None or self.checkpoint is None:
            return
        try:
            self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            self.repl.snapshot(self._state_path)
        except Exception:  # noqa: BLE001 - a convenience; never fail the task over it
            pass

    def _save(self, messages: list[dict], step: int, finished: bool, stopped: str | None = None) -> None:
        if self.checkpoint is None:
            return
        if not finished:   # REPL variables too, so a resume keeps them
            self._snapshot()
        data = {"version": 1, "persona": self.persona.name, "task_text": self._task_text, "task_idx": self._task_idx,
                "progress_log": self._render_log(), "compact": {"entries": self._entries, "dropped": self._dropped, "seq": self._seq},
                "step": step, "finished": finished, "stopped": stopped,
                "workdir": str(self.workdir), "saved": time.time(),
                "messages": [m for m in messages if m["role"] != "system"]}
        try:
            self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.checkpoint.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.checkpoint)
        except OSError:
            pass  # a checkpoint is a convenience; never fail the task over it

    def _bounce(self, answer: str, artifacts: list[str], changed: list[str], messages: list[dict],
                step: int, prefix: str) -> bool:
        """The final() gate. True = problems found and sent back to the worker (keep going)."""
        self._verdict = None
        if self.verify is None:
            return False
        self._note("checking the answer before handing it over")
        # what the final() block itself printed counts as seen (print(total); final(...) is common)
        seen = messages + ([{"role": "user", "content": prefix.strip()}] if prefix.startswith("Output:") else [])
        gate_failed = False
        try:
            if self._verify_takes_task_idx():
                verdict = self.verify(answer, artifacts, seen, changed, task_idx=self._task_idx)
            else:   # a verify callable from before task_idx
                verdict = self.verify(answer, artifacts, seen, changed)
        except Cancelled:
            raise
        except Exception as e:  # noqa: BLE001 - the gate's own failure must not cost the user a finished answer
            # (it ended the task as "failed" before, the answer lost; now it is handed over, marked as not checked. And
            # a TypeError from inside the gate no longer runs the whole gate a second time, as `except TypeError` did)
            from .verify import Verdict
            gate_failed = True
            verdict = Verdict(ok=False, problems=[f"the harness could not check this answer: {type(e).__name__}: {e}"[:300]],
                              checks={"gate": f"error: {type(e).__name__}"})
        self._verdict = verdict
        # the last-step report of a stalled worker is taken as it is, problems listed under it: sent back,
        # it went on trying (live: a report on Target's bot block bounced for "says it will do more", and the
        # worker then tried Target's API with a faked browser header)
        can_retry = (self._bounces < self.max_bounces and step < self.limit and self._wrap_step is None
                     and not gate_failed)
        self.on_step("verify", {"persona": self.persona.name, "step": step, "ok": verdict.ok,
                                "problems": verdict.problems, "checks": verdict.checks, "scores": verdict.scores,
                                "bounced": (not verdict.ok) and can_retry, "answer": answer[:4000]})
        if verdict.ok or not can_retry:
            return False
        self._bounces += 1
        if self.escalate is not None:   # the fast model's work didn't hold up: the strong one fixes it
            strong, self.escalate, self.escalated = self.escalate(), None, True
            self._use(strong)
            self.on_step("escalate", {"persona": self.persona.name, "step": step,
                                      "model": getattr(strong, "_model", "") or getattr(strong, "model", "")
                                      or "the strong tier"})
        listed = "\n".join(f"- {p}" for p in verdict.problems)
        messages.append({"role": "user", "content": (
            f"{prefix}[Not accepted yet: the harness checked your work before handing it over and found:\n"
            f"{listed}\nFix these and call final() again. If something can't be fixed, call final() and "
            f"say plainly what is not done.]" + self._steps_left_note(step))})
        return True

    def _verify_takes_task_idx(self) -> bool:
        try:
            params = inspect.signature(self.verify).parameters
        except (TypeError, ValueError):
            return False
        return "task_idx" in params or any(p.kind == p.VAR_KEYWORD for p in params.values())

    def _with_unresolved(self, answer: str) -> str:
        v = getattr(self, "_verdict", None)
        if v is None or v.ok:
            return answer
        # ((p.splitlines() or [""]): an empty problem string raised IndexError here, after all the work was done)
        return answer + "\n\n---\nNot verified - the harness's checks still found:\n" + "\n".join(
            f"- {(p.splitlines() or [''])[0]}" for p in v.problems)

    def _check(self) -> None:
        if self.cancel.is_set():
            raise Cancelled("stopped by the user")

    @property
    def native(self) -> bool:
        return bool(getattr(self.llm, "native_tools", False))

    @property
    def _act(self) -> str:
        return "call the python tool with your code" if self.native else "reply with ONE ```python block"

    def _llm(self, messages):
        # the last step of a stalled worker offers only `final`: told to report, the fast model ran more
        # code instead and the user got raw output (live, Booking.com)
        kw = {"tools": self.tool_schemas()} if self.native else {}
        if self._llm_cancel:
            kw["cancel"] = self.cancel
        try:
            params = inspect.signature(self.llm.chat).parameters
            takes_all = any(p.kind == p.VAR_KEYWORD for p in params.values())
        except (TypeError, ValueError):
            params, takes_all = {}, False
        if takes_all or "live" in params:   # the reply streams to thinking/step-NN.txt as it is generated
            kw["live"] = self.workdir / "thinking" / f"step-{self._step:02d}.txt"
        if takes_all or "interrupt" in params:   # a message from the user cuts the reply in flight (steer())
            kw["interrupt"] = self.interrupt
        # A long reply is watched until the model acts. Always, in plain code: is it repeating itself (see
        # repeating())? And the thinking watch (Jev), only once this task has run code - there is data in the
        # REPL that code could read exactly - and only on reasoning, or on the reply of a model that reasons
        # in its reply (no separate reasoning in this step): a long final answer is never paused.
        # After an ignored pause the next check comes later (2x), never later than that: switching off let
        # a step read by hand for 30 KB, doubling each time let one think 40 KB / 10 minutes unchecked.
        if takes_all or "watch" in params:
            jev = self.thinking_watch if self._ran_code else None

            def watch(text: str, answering: bool = False) -> bool:
                loop = repeating(text)
                if loop:
                    self._watch_reason, self._loop_line = "repeating", loop
                    return True
                if jev is None or answering:
                    return False
                self._watch_reason = jev(text)
                return bool(self._watch_reason)
            kw.update(watch=watch, watch_every=self.watch_every,
                      watch_after=self.watch_after * (2 if self._pauses >= 2 else 1))
        try:
            return self.llm.chat(messages, **kw)
        except LLMError as e:
            # the server went silent. Live (Gemma 4 in LM Studio): it sometimes generates for minutes without
            # streaming anything - replaying one such request, 5 of 9 tries went silent, the rest answered in
            # seconds - and the whole task failed after computing the right answer. Sampling differs on a
            # retry: up to SILENT_RETRIES more tries.
            if "no tokens for" not in str(e):
                raise
            for attempt in range(1, SILENT_RETRIES + 1):
                self.on_step("output", {"persona": self.persona.name, "step": self._step, "error": True,
                                        "output": f"({e} - retrying the step, try {attempt + 1})"})
                try:
                    return self.llm.chat(messages, **kw)
                except LLMError as again:
                    if "no tokens for" not in str(again) or attempt == SILENT_RETRIES:
                        raise
                    e = again

    @staticmethod
    def _replies(call_ids: list[str], text: str) -> list[dict]:
        """What a step's code produced, as the model expects it back: a tool result per native call
        (the first carries the output), or one 'Output:' user message for code written as text."""
        if not call_ids:
            return [{"role": "user", "content": f"Output:\n{text}"}]
        return [{"role": "tool", "tool_call_id": call_ids[0], "content": text}] + [
            {"role": "tool", "tool_call_id": i, "content": "(ran together with the first call: its output is above)"}
            for i in call_ids[1:]]

    def _steps_left_note(self, step: int) -> str:
        left = self.limit - step
        where = "with your next python tool call" if self.native else "in your next block"
        if left == 2:
            return (f"\n\n[2 steps left: save your deliverables with save_artifact() and call "
                    f"final() {where}.]")
        if left == 1:
            return (f"\n\n[LAST STEP: only call final(answer, artifacts) {where}, with what you have "
                    "now. Anything else is lost.]")
        return ""

    def _stuck_note(self, output: str, error: bool) -> str:
        """Weak models retry the same failing call until the step limit. After the same
        error line 3 times in a row, tell them to stop and report the blocker."""
        sig = output.strip().splitlines()[-1][:200] if (error and output.strip()) else None
        self._err_streak = self._err_streak + 1 if (sig and sig == self._last_err) else (1 if sig else 0)
        self._last_err = sig
        if self._err_streak >= 3:
            return ("\n\n[The same error has now happened 3 times in a row. Stop retrying it. Either "
                    "try a genuinely different approach, or call final() explaining the blocker.]")
        return ""

    def _harvest(self) -> list[str]:
        """Step limit hit: keep the work instead of losing it - copy files the worker wrote in its scratch dir into the
        artifact store: at most HARVEST_MAX, the folder's own files before those in its subfolders. Only those are
        copied: every file used to be, while only the first 20 by name were handed over - after an npm install, all of
        node_modules was copied, and its files sorted ahead of (and pushed out) the page the worker had written."""
        own = {self.checkpoint, self._state_path}   # the harness's files, and their half-written copies
        own |= {self.checkpoint.with_suffix(".tmp"), Path(f"{self._state_path}.tmp")} if self.checkpoint else set()
        found: list[tuple[int, str, Path]] = []
        for dirpath, dirnames, filenames in os.walk(self.workdir):
            here = Path(dirpath)
            depth = len(here.relative_to(self.workdir).parts)
            # thinking/ is the harness's own log of each step, not the worker's work: shipped as artifacts, the
            # final() gate checked them for placeholders ("example" in sam@example.com) and the worker then
            # edited its own logs to get past the check (live, the demoqa form)
            dirnames[:] = [d for d in dirnames if not d.startswith((".", "__")) and d not in HARVEST_SKIP
                           and not (depth == 0 and d == "thinking")]
            for f in filenames:
                p = here / f
                try:
                    if f.startswith((".", "__")) or p in own or not p.is_file() or not p.stat().st_size:
                        continue
                except OSError:
                    continue
                found.append((depth, p.relative_to(self.workdir).as_posix(), p))
        out = []
        self.artifacts.mkdir(parents=True, exist_ok=True)
        for _, rel, p in sorted(found)[:HARVEST_MAX]:
            dest = self.artifacts / f"{self.persona.name}_{rel.replace('/', '_')}"
            try:
                shutil.copyfile(p, dest)
            except OSError:   # (locked by another program on Windows, or gone): the rest are still kept
                continue
            out.append(str(dest))
        return out

    def _chat(self, task: str, ctx: dict[str, Any], system: str, prior: list | None = None) -> WorkerResult:
        memory = ctx.get("memory") or []
        extra = ""
        if memory:
            extra += "\n\nRelevant earlier conversation:\n" + "\n---\n".join(memory)
        if ctx.get("inputs"):
            extra += "\n\nInputs: " + ", ".join(map(str, ctx["inputs"]))
        messages = [{"role": "system", "content": system}, *(prior or []),
                    {"role": "user", "content": task + extra}]
        self._live = messages
        while True:   # (asked again, with the user's message, each time one cuts the reply)
            self._check()
            self._deliver_steers(messages)
            self._note("the model is writing its reply")
            out = self._llm(messages)
            if not (getattr(out, "redirected", False) or self._steers_pending()):
                break
            self._redirect(messages, out, 1)
            self._after_cut()
        messages.append({"role": "assistant", "content": out.text})
        return WorkerResult(answer=out.text, steps=1, input_tokens=out.input_tokens,
                            output_tokens=out.output_tokens, stopped="plain", transcript=messages)
