"""Worker: one persona doing one task, acting by writing Python in its own REPL.

The prompt holds the persona, the REPL API, and only the skills Jev picked. The
request is the user message; everything bigger (memory, other assistants' outputs)
sits in the REPL as `ctx[...]` and is only listed by name and size.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .llm import LLM
from .personas import Persona
from .repl.client import Repl
from .skills import Skill

# Thinking models (Qwen3.8 in LM Studio) otherwise draft the whole solution in hidden
# reasoning before replying - measured 16k+ thinking tokens and zero code on a web-app task,
# vs 127 thinking tokens with this nudge. enable_thinking / reasoning_effort / /no_think
# had no effect through LM Studio's API.
FIRST_STEP_NUDGE = ("\n\nReply now with a short plan (at most 5 bullets) and a first python block "
                    "for step one. Build incrementally across replies - do not design the whole "
                    "solution before replying.")
HISTORY_CHARS = 3000       # per earlier assistant reply carried into a follow-up
SKILL_PROMPT_CHARS = 6000   # longer skills: head in the prompt, full text in ctx['skills']
CODE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
# Some local models (seen: ornith in LM Studio) sometimes answer in their native tool-call
# template instead of a fenced block: <tool_call><function=python><parameter=content>...
TOOL_CALL_RE = re.compile(r"<function=python[^>]*>\s*<parameter=\w+>\s*(.*?)\s*(?:</parameter>|</function>|</tool_call>|\Z)",
                          re.DOTALL)
OPEN_FENCE_RE = re.compile(r"```(?:python|py)[ \t]*\n")

REPL_GUIDE = """\
## How you work
You have a persistent Python REPL. To act, reply with ONE ```python block; it runs and
you see its output. Variables persist between blocks. When you are done, call
final(answer, artifacts=[...]) in a python block - `answer` is what the user (or the
assistant who asked you) receives. If the request needs no code, just reply in plain text.

Available in the REPL:
- ctx: dict of what you were given (see the key list in the task). Big values live
  here instead of in your prompt - look with peek(ctx['key']) or slice them.
- peek(x, n=2000) -> str: preview any value.
- llm_query(prompt, data=None) -> str: ask a focused sub-question about a slice of data.
- jev(state, questions) -> dict: fast typed decisions; questions like
  {"is_x": {"type": "noul", "instructions": "Is ...?"}}.
- save_artifact(name, data=None, path=None) -> path; load_artifact(ref); list_artifacts().
- final(answer, artifacts=None).
Your working directory is private scratch space. Keep printed output short, and keep
each reply short: write large files across several blocks (open(path, 'a'))."""

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
    lines = []
    for k, v in ctx.items():
        if k == "request":
            continue
        if isinstance(v, (list, tuple)):
            size = f"list of {len(v)}"
        elif isinstance(v, str):
            size = f"str, {len(v)} chars"
        else:
            size = type(v).__name__
        lines.append(f"- ctx[{k!r}]: {size}")
    return "\n".join(lines) or "- (nothing else)"


MEMORY_GUIDE = """\
- recall(query) -> [snippets]: search ALL past conversations and remembered facts (ctx['memory']
  only holds what Jev judged relevant up front). remember(fact): save a lasting fact about the user."""


def build_system(persona: Persona, skills: list[Skill], requestable: dict[str, str],
                 suggested: list[str], tools: list[str] | None = None,
                 mcp_tools: list | None = None) -> str:
    """`tools`: the tool guides Jev picked for this task; the persona's other tools are still
    importable and named in one line. `mcp_tools`: McpTool objects Jev picked."""
    from .tools import TOOLS
    parts = [persona.system_prompt.strip()]
    if persona.code:
        guide = REPL_GUIDE + "\n" + MEMORY_GUIDE
        if requestable:
            roster = "\n".join(f"    - {n}: {d}" for n, d in requestable.items())
            guide += "\n" + REQUEST_GUIDE.format(roster=roster)
        parts.append(guide)
        picked = [t for t in tools or [] if t in TOOLS]
        if picked:
            parts.append("## Tools in your REPL\n" + "\n".join(TOOLS[t].guide for t in picked))
        rest = [t for t in persona.tools if t not in picked]
        if rest:
            parts.append("Also importable in your REPL (help(name) shows how): " + ", ".join(rest))
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
                 on_step: Callable[[str, dict], None] | None = None, env: dict[str, str] | None = None):
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

    def run(self, task: str, ctx: dict[str, Any], system: str,
            history: list[tuple[str, str]] | None = None) -> WorkerResult:
        """`history`: the (user, assistant) turns this message follows up on, as real chat turns."""
        prior = []
        for u, a in history or []:
            prior += [{"role": "user", "content": u}, {"role": "assistant", "content": a[:HISTORY_CHARS]}]
        if not self.persona.code:
            return self._chat(task, ctx, system, prior)
        messages = [{"role": "system", "content": system}, *prior,
                    {"role": "user", "content": f"{task}\n\n(In your REPL:\n{describe_ctx(ctx)})"
                                                + FIRST_STEP_NUDGE}]
        res = WorkerResult(answer="")
        self._last_err, self._err_streak = None, 0
        with Repl(self.workdir, self.artifacts, self.handlers, self.tools,
                  self.repl_timeout, self.output_cap, self.env) as repl:
            repl.set("ctx", ctx)
            for step in range(1, self.persona.max_steps + 1):
                out = self.llm.chat(messages)
                res.steps, res.input_tokens, res.output_tokens = (
                    step, res.input_tokens + out.input_tokens, res.output_tokens + out.output_tokens)
                messages.append({"role": "assistant", "content": out.text})
                blocks = CODE_RE.findall(out.text) or TOOL_CALL_RE.findall(out.text)
                if "[output cut off at max_tokens" in out.text[-120:] or (
                        not blocks and OPEN_FENCE_RE.search(out.text)):
                    self.on_step("output", {"persona": self.persona.name, "step": step,
                                            "output": "(reply cut off - asked to use smaller steps)",
                                            "error": True})
                    messages.append({"role": "user", "content": (
                        "Your reply was cut off before the code block closed, so nothing ran. "
                        "Work in smaller steps: write big files in parts (open(..., 'a')) across "
                        "several python blocks, and keep each reply short." + self._steps_left_note(step))})
                    continue
                if not blocks:
                    res.answer, res.stopped = out.text, "plain"
                    break
                code = "\n\n".join(blocks)
                self.on_step("code", {"persona": self.persona.name, "step": step, "code": code})
                ex = repl.run(code)
                self.on_step("output", {"persona": self.persona.name, "step": step,
                                        "output": ex.output, "error": ex.error})
                if ex.final is not None:
                    res.answer, res.artifacts, res.stopped = ex.final["answer"], ex.final["artifacts"], "final"
                    if not res.artifacts:
                        # final() without artifacts but files written in private scratch: ship them,
                        # or whoever asked never sees the deliverable (seen live: a designer's index.html)
                        res.artifacts = self._harvest()
                    break
                messages.append({"role": "user", "content": f"Output:\n{ex.output or '(no output)'}"
                                                            + self._stuck_note(ex.output, ex.error)
                                                            + self._steps_left_note(step)})
            else:
                res.artifacts = self._harvest()
                res.answer = (f"[{self.persona.name} hit its {self.persona.max_steps}-step limit without "
                              f"calling final(); files it wrote were collected as artifacts]\n"
                              + messages[-2]["content"][-2000:])
                res.stopped = "max_steps"
        res.transcript = messages
        return res

    def _steps_left_note(self, step: int) -> str:
        left = self.persona.max_steps - step
        if left == 2:
            return ("\n\n[2 steps left: save your deliverables with save_artifact() and call "
                    "final() in your next block.]")
        if left == 1:
            return ("\n\n[LAST STEP: reply with only a python block that calls final(answer, "
                    "artifacts) with what you have now. Anything else is lost.]")
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
        """Step limit hit: keep the work instead of losing it - copy files the worker
        wrote in its scratch dir into the artifact store."""
        out = []
        self.artifacts.mkdir(parents=True, exist_ok=True)
        for p in sorted(self.workdir.rglob("*")):
            rel = p.relative_to(self.workdir)
            if p.is_file() and p.stat().st_size and not any(
                    part.startswith((".", "__")) for part in rel.parts):
                dest = self.artifacts / f"{self.persona.name}_{'_'.join(rel.parts)}"
                shutil.copyfile(p, dest)
                out.append(str(dest))
        return out[:20]

    def _chat(self, task: str, ctx: dict[str, Any], system: str, prior: list | None = None) -> WorkerResult:
        memory = ctx.get("memory") or []
        extra = ""
        if memory:
            extra += "\n\nRelevant earlier conversation:\n" + "\n---\n".join(memory)
        if ctx.get("inputs"):
            extra += "\n\nInputs: " + ", ".join(map(str, ctx["inputs"]))
        messages = [{"role": "system", "content": system}, *(prior or []),
                    {"role": "user", "content": task + extra}]
        out = self.llm.chat(messages)
        messages.append({"role": "assistant", "content": out.text})
        return WorkerResult(answer=out.text, steps=1, input_tokens=out.input_tokens,
                            output_tokens=out.output_tokens, stopped="plain", transcript=messages)
