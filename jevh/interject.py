"""Messages sent while a task is running.

A running task used to have one control: Stop. Now a message sent meanwhile is simply added to the task's prompt:

  (a message)         The lead worker gets it as a user message at the end of its conversation. A reply the model
  /steer              is writing is cut and asked for again with the message (a redirect); code that is running is
                      never killed: the message arrives with its output (a steer). Nothing the cut reply said is
                      acted on. /steer skips the check below and always adds.

Two things are told apart first (jev.interject_kind: one Choice call, ~0.4 s) unless a command says what it is:

  a question          "how's it going?" asks for nothing new. Answered at once from what the worker is doing, then in
  /btw, /status       the task's own voice by its model, on a copy of what it knows, while the task carries on (the
                      quick reply). Added to the prompt instead, a question could end the task with a status report.
  a swap              "check my calendar instead" replaces the request, not adds to it: the message is added to the
                      prompt all the same, but it stands alone as the request the final() gate judges.

Queueing and stopping are never guessed from the words, only commands:

  /queue <text>       another request. Queued: it runs as its own turn when this one ends.
  /stop               stops the turn, like the Stop button (which also drops what was queued).

The model server matters for how cheap a redirect and a quick reply are: both are one more request whose start is
the conversation the server already holds, so the harness only ever appends to a conversation and builds the
quick reply's prompt to share that start (docs/architecture/16-interjection.md).

Settings (config.yaml `interject:`):

    interject:
      enabled: true          # false: a message sent while a task runs is refused, as before
      classify: true         # Jev tells a question and a swap of the request; false: every message without a command is
                             # just added to the prompt
      redirect: true         # a correction cuts the model's reply in flight; false: it waits for the next step
      aside_model: true      # a status question is also answered by the task's model; false: the instant line only
      aside_context: digest  # what the quick reply is given: digest (the task and its last steps, ~1k tokens) or
                             # full (the worker's whole conversation: for servers that share a prompt cache, vLLM)
      aside_max_tokens: 1500 # (a thinking model spends some of it thinking)
      retry_delay: 0.5       # seconds to wait before asking the model again after a cut: llama.cpp frees the cancelled
                             # slot a moment after the hang-up, and a retry that arrives first starts on the other slot
                             # with nothing cached (a whole prompt read again); 0 for a server without slots (vLLM)
"""
from __future__ import annotations

import re
from typing import Any

DEFAULTS = {"enabled": True, "classify": True, "redirect": True, "aside_model": True,
            "aside_context": "digest", "aside_max_tokens": 1500, "retry_delay": 0.5}
KINDS = ("correction", "status", "separate", "stop")
COMMANDS = {"steer": "correction", "queue": "separate", "btw": "status", "status": "status", "stop": "stop"}
_COMMAND = re.compile(r"^/(steer|queue|btw|status|stop)\b\s*(.*)$", re.DOTALL | re.IGNORECASE)

ASIDE_SYSTEM = ("You are {persona}, the assistant doing the task below for the user. The user sent a side message "
                "while you work. Answer it in one to three plain sentences, from the notes: what you are doing, what "
                "you have found or done so far, what comes next. Do not continue the task, do not call tools and do "
                "not claim anything the notes do not show. If the notes do not say, say you do not know yet.")
ASIDE_DIGEST = ("Task, as the user gave it:\n{task}\n\nWhat has happened so far (newest last):\n{steps}\n\n"
                "Right now: {doing}\n\nThe user's side message: {question}")
ASIDE_FULL = ("[The user sent a side message while you work: \"{question}\"\nRight now: {doing}.\n"
              "Answer it in one to three plain sentences from what you know. Do not continue the task and do not "
              "call tools: the task carries on separately.]")


# The request as it stands once the user has said something while it was worked on. A message that replaced the
# request stands alone (Task.current); any other is added under the original, and where the two disagree the
# message wins - the final() gate and the progress checks judge this text, not the original.
UPDATED = ('[The user then said, while this was being worked on: "{text}". This is their latest word on it: where it '
           'conflicts with the request above, it wins.]')


def settings(raw: dict[str, Any] | None) -> dict[str, Any]:
    return {**DEFAULTS, **((raw or {}).get("interject") or {})}


def parse(text: str) -> tuple[str | None, str]:
    """A message and the command it starts with: ("correction" | "separate" | "status" | "stop" | None, the text
    without its /command). None: no command, Jev decides."""
    text = (text or "").strip()
    m = _COMMAND.match(text)
    if not m:
        return None, text
    return COMMANDS[m.group(1).lower()], m.group(2).strip()


def status_line(st: dict[str, Any], queued: int = 0) -> str:
    """The answer to "how is it going?" that needs no model: where the task is and what it is doing this moment."""
    who = st.get("persona") or "The assistant"
    step, limit = st.get("step") or 0, st.get("max_steps") or 0
    where = f" (step {step} of up to {limit})" if step else ""
    doing = str(st.get("doing") or "").strip()
    line = f"{who} is working on it{where}"
    line += f": {doing}." if doing else "."
    secs = float(st.get("seconds") or 0)
    if secs >= 5 and doing:
        line += f" That has taken {secs:.0f}s so far."
    n = len(st.get("corrections") or [])
    if n:
        line += f" It has taken in {n} message{'s' if n != 1 else ''} from you since it started."
    if st.get("pending"):
        line += " One of your messages is waiting for its next step."
    if queued:
        line += f" {queued} more request{'s' if queued != 1 else ''} of yours {'are' if queued != 1 else 'is'} queued for after."
    return line


def aside_messages(mode: str, persona: str, task: str, doing: str, question: str, steps: str,
                   snapshot: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The quick reply's prompt.
    digest: the task and its last steps in a few lines: small, and it shares nothing with the task's own prompt,
            so it never disturbs the model server's cache of that.
    full:   the worker's own conversation as it stands, plus one more message at the end: the same start as the
            worker's prompt, so a server that keeps one prompt cache for all requests (vLLM) reads none of it again.
            A step whose code is still running (an assistant message whose tool calls have no results yet) is left
            out: a conversation must not end on unanswered tool calls."""
    if mode == "full" and snapshot:
        msgs = list(snapshot)
        while msgs and msgs[-1].get("role") == "assistant" and msgs[-1].get("tool_calls"):
            msgs.pop()
        return msgs + [{"role": "user", "content": ASIDE_FULL.format(question=question, doing=doing)}]
    return [{"role": "system", "content": ASIDE_SYSTEM.format(persona=persona)},
            {"role": "user", "content": ASIDE_DIGEST.format(task=task[:2500], steps=steps or "(nothing yet)",
                                                            doing=doing or "starting", question=question)}]
