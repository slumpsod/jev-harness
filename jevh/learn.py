"""Learning after a turn: remembered facts and learned skills.

Runs after a successful turn, off the critical path (a background thread), and asks Jev
first so the model is only called when there is something to keep:

- Facts: "does the user state a lasting fact about themselves or their world?" -> the task's
  own model writes 1-3 short standalone facts -> an older fact that says the same thing (or
  an outdated version of it) is retired, so memory stays current instead of piling up.
- Skills: after a multi-step task, "did this work out a reusable procedure?" -> if a skill was
  loaded, its lessons are appended (to the SKILL.md for skills the harness owns, to
  skills/_lessons/<name>.md for others, e.g. Hermes skills, so their files are never edited);
  otherwise a new skills/learned/learned-<slug>/SKILL.md is written. Skills are plain files:
  review them with git, delete the ones you don't want.

The learning calls use the model that just ran the task, which is already loaded: with
LM Studio's one-model rule, a different model would cost a 12-23s swap.
"""
from __future__ import annotations

import inspect
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .decide import noul
from .errors import Cancelled
from .memory import Memory
from .skills import Skill, parse_skill

# Phrasings chosen on evals/learn_cases.py (semif). FACT_Q: 12/12 facts, 0 false at 0.9 (a
# reminder request scored 0.56, "what's on my calendar" 0.82). "Does new say the same as or update
# old" caught 1/5 replacements; "made redundant or outdated" 5/5 (>=0.65) vs <=0.08. "Worked out a
# reusable procedure" caught 1/3; "non-obvious steps, fixes or settings" 3/3 (>=0.78) vs <=0.18.
FACT_Q = ("Does the user's message state a lasting fact about the user or their world (their preferences, "
          "people, pets, home, work, projects, equipment, goals or schedule) that would help in future "
          "conversations?")
SAME_FACT_Q = "Is `old_fact` made redundant or outdated by `new_fact`?"
SKILL_Q = "Did `work_log` involve figuring out non-obvious steps, fixes or settings (not just a single simple step)?"

FACT_PROMPT = """From the exchange below, write the lasting facts about the user worth remembering in \
future conversations: preferences, people, pets, home, work, projects, equipment, goals, schedule. \
Each fact is one short standalone sentence in third person ("Jed's gym is closed on Sundays."). \
Skip anything only relevant to this request. Reply with a JSON list of 0-3 strings and nothing else.\
{direct}

User: {user}
Assistant: {assistant}"""

SKILL_PROMPT = """You just finished the task below. Write a short how-to guide (a skill) so that next \
time a similar task comes up, it can be done quickly and without the mistakes made this time.

Task: {task}

Work log (code that ran and its output):
{log}

Reply with ONLY a SKILL.md file:
---
name: <3-5 word kebab-case name>
description: <one sentence: when to use this skill>
---
# <title>
## Steps
<numbered steps with the exact commands, functions, settings or URLs that worked>
## Pitfalls
<what went wrong this time and how to avoid it>
Keep it under 60 lines.{direct}"""

LESSON_PROMPT = """You just finished the task below with the skill guide "{name}" loaded. If the guide \
wasn't really about this kind of task, reply NONE. Otherwise: what should the next person following \
this guide do differently, based on what happened in this run (exact commands, settings, pitfalls)? \
Write 1-4 short bullet points starting with "- ", each an instruction for next time ("Check X before \
Y"), not a critique of the guide. Reply NONE if the guide already covered everything.{direct}

Task: {task}

Work log:
{log}

The guide:
{body}"""


# No "reply right away" line: it cut Qwen's thinking ~25% (1400 -> 1090 tokens) on skill writing, but
# quality comes first here, and a new turn drops a learning call still in progress anyway.
DIRECT = ""


@dataclass
class Learned:
    facts: list[str] = field(default_factory=list)
    retired: list[str] = field(default_factory=list)
    skill: str | None = None          # path written or updated
    skill_action: str | None = None   # "created" | "lessons"
    scores: dict[str, float] = field(default_factory=dict)


def work_log(transcript: list[dict[str, str]], cap: int = 5000) -> str:
    """The code a worker ran and what came back, compact, most recent last."""
    parts = []
    for m in transcript[2:]:
        text = m.get("content", "")
        if m.get("role") == "assistant":
            for code in re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S):
                parts.append("CODE:\n" + code.strip()[:700])
        elif text.startswith("Output:"):
            parts.append("OUTPUT: " + text[7:].strip()[:350])
    log = "\n".join(parts)
    return log if len(log) <= cap else "[...]\n" + log[-cap:]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "skill"


class Learner:
    def __init__(self, jev, memory: Memory, skills_root: Path, thresholds: Callable[[str], float],
                 on_event: Callable[[str, dict], None] | None = None):
        self.cancel = threading.Event()   # the harness sets it when the user starts a new turn
        self._run = threading.local()     # each learning run keeps the flag it started with
        self.jev, self.memory, self.root = jev, memory, Path(skills_root)
        self.t = thresholds
        self.on_event = on_event or (lambda kind, data: None)

    # -- facts -------------------------------------------------------------------
    def facts(self, user: str, assistant: str, llm, source: str = "") -> tuple[list[str], list[str], float]:
        state = {"user_message": user[:2000], "assistant_reply": assistant[:600]}
        p = self.jev._ask("learn_fact", state, {"fact": noul(FACT_Q)})["fact"].noul
        if p < self.t("fact"):
            return [], [], p
        raw = self._chat(llm, FACT_PROMPT.format(user=user[:3000], assistant=assistant[:1500], direct=DIRECT))
        new = [f.strip() for f in _json_list(raw) if isinstance(f, str) and 8 <= len(f.strip()) <= 300][:3]
        return new, self._store_facts(new, source), p

    def remember(self, fact: str, source: str = "") -> str:
        retired = self._store_facts([fact.strip()], source)
        return f"remembered: {fact.strip()}" + (f" (replaces: {'; '.join(retired)})" if retired else "")

    def _store_facts(self, new: list[str], source: str) -> list[str]:
        retired = []
        for fact in new:
            old = self.memory.facts(fact, 4)
            if old:
                r = self.jev._ask("learn_same_fact", {"new_fact": fact, "old_facts": {f"f{o.id}": o.text for o in old}},
                                  {f"f{o.id}": noul(SAME_FACT_Q.replace("`old_fact`", f"`old_facts.f{o.id}`"))
                                   for o in old})
                for o in old:
                    if r[f"f{o.id}"].noul >= self.t("same_fact"):
                        self.memory.retire_fact(o.id)
                        retired.append(o.text)
            self.memory.add_fact(fact, source)
        return retired

    # -- skills ------------------------------------------------------------------
    def skill(self, task: str, transcript: list[dict[str, str]], used: list[Skill], llm) -> tuple[str | None, str | None, float]:
        log = work_log(transcript)
        if not log:
            return None, None, 0.0
        p = self.jev._ask("learn_skill", {"task": task[:1500], "work_log": log[-3500:]}, {"skill": noul(SKILL_Q)})["skill"].noul
        if p < self.t("learn_skill"):
            return None, None, p
        if used:
            return self._lessons(task, log, used[0], llm), "lessons", p
        return self._create(task, log, llm), "created", p

    def _create(self, task: str, log: str, llm) -> str | None:
        text = self._chat(llm, SKILL_PROMPT.format(task=task[:2000], log=log, direct=DIRECT))
        text = re.sub(r"^```(?:markdown|md)?\s*\n|\n```\s*$", "", text.strip())
        m = re.search(r"^---\s*\n(.*?)\n---", text, re.S | re.M)
        if not m:
            return None
        text = text[m.start():]
        name_m = re.search(r"^name:\s*(.+)$", m.group(1), re.M)
        base = name = "learned-" + slug(name_m.group(1) if name_m else task)
        n = 2
        while (self.root / "learned" / name).exists():
            name, n = f"{base}-{n}", n + 1
        # draft until the user approves it (dashboard Memory tab): one run's result is unverified
        text = re.sub(r"^name:\s*.+$", f"name: {name}\ndraft: true", text, count=1, flags=re.M)
        dest = self.root / "learned" / name / "SKILL.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text.strip() + f"\n\n<!-- learned {time.strftime('%Y-%m-%d')} from: {task[:200]} -->\n",
                        encoding="utf-8")
        if parse_skill(dest) is None:  # unusable frontmatter: don't leave junk in the roster
            dest.unlink()
            return None
        return str(dest)

    def _lessons(self, task: str, log: str, skill: Skill, llm) -> str | None:
        text = self._chat(llm, LESSON_PROMPT.format(
            name=skill.name, task=task[:1500], log=log, body=skill.body[:4000], direct=DIRECT)).strip()
        bullets = [l for l in text.splitlines() if l.strip().startswith("- ")][:4]
        if not bullets or text.upper().startswith("NONE"):
            return None
        entry = f"\n<!-- {time.strftime('%Y-%m-%d')}: {task[:120]} -->\n" + "\n".join(bullets) + "\n"
        path = Path(skill.path)
        if self.root.resolve() in path.resolve().parents:  # ours: improve the guide itself
            body = path.read_text(encoding="utf-8")
            if "## Lessons from past runs" not in body:
                body = body.rstrip() + "\n\n## Lessons from past runs\n"
            path.write_text(body + entry, encoding="utf-8")
            return str(path)
        note = self.root / "_lessons" / f"{skill.name}.md"  # not ours (e.g. Hermes): overlay file
        note.parent.mkdir(parents=True, exist_ok=True)
        with note.open("a", encoding="utf-8") as f:
            f.write(entry)
        return str(note)

    def _chat(self, llm, prompt: str) -> str:
        """The learning model call; dropped (Cancelled) when the user starts a new turn."""
        cancel = getattr(self._run, "cancel", self.cancel)
        if cancel.is_set():
            raise Cancelled("a new turn started")
        messages = [{"role": "user", "content": prompt}]
        try:
            takes_cancel = "cancel" in inspect.signature(llm.chat).parameters
        except (TypeError, ValueError):
            takes_cancel = False
        return (llm.chat(messages, cancel=cancel) if takes_cancel else llm.chat(messages)).text

    # -- one turn ----------------------------------------------------------------
    def after_turn(self, persona: str, user: str, answer: str, transcript: list[dict[str, str]], steps: int,
                   used: list[Skill], llm, session: str) -> Learned:
        out = Learned()
        self._run.cancel = self.cancel
        try:
            out.facts, out.retired, out.scores["fact"] = self.facts(user, answer, llm, source=f"{session}/{persona}")
            if steps >= 3:
                out.skill, out.skill_action, out.scores["skill"] = self.skill(user, transcript, used, llm)
            if not out.skill:
                out.skill_action = None
        except Cancelled:  # the user moved on: their turn gets the model, this learning is dropped
            self.on_event("learn_error", {"error": "skipped: a new turn started"})
            return out
        except Exception as e:  # noqa: BLE001 - learning is best effort; never break a turn
            self.on_event("learn_error", {"error": f"{type(e).__name__}: {e}"})
        self.on_event("learned", {"persona": persona, "learned": out})
        return out


def _json_list(text: str) -> list[Any]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        v = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return v if isinstance(v, list) else []
