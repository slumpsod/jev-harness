"""Conversation memory on disk. Nothing here reaches a worker unless Jev's
gate_context() passes it.

workspace/memory/timeline.jsonl   every turn, all personas (shared pool)
workspace/memory/<persona>.jsonl  that persona's own turns
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

CHUNK_CHARS = 900


@dataclass
class Turn:
    persona: str
    user: str
    assistant: str
    session: str
    ts: float

    def chunk(self) -> str:
        text = f"[{self.persona}] User: {self.user}\nAssistant: {self.assistant}"
        return text if len(text) <= CHUNK_CHARS else text[: CHUNK_CHARS - 3] + "..."


class Memory:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _read(self, name: str, n: int) -> list[Turn]:
        path = self.root / f"{name}.jsonl"
        if not path.exists() or n <= 0:
            return []
        lines = path.read_text(encoding="utf-8").splitlines()[-n:]
        return [Turn(**json.loads(l)) for l in lines if l.strip()]

    def append(self, persona: str, user: str, assistant: str, session: str) -> None:
        turn = Turn(persona, user, assistant, session, time.time())
        line = json.dumps(asdict(turn), ensure_ascii=False) + "\n"
        for name in (persona, "timeline"):
            with open(self.root / f"{name}.jsonl", "a", encoding="utf-8") as f:
                f.write(line)

    def own(self, persona: str, n: int) -> list[Turn]:
        return self._read(persona, n)

    def shared(self, exclude: str, n: int) -> list[Turn]:
        """Recent turns from *other* personas."""
        out = [t for t in self._read("timeline", n * 6) if t.persona != exclude]
        return out[-n:]

    def recent_text(self, session: str, n: int = 2) -> str:
        turns = [t for t in self._read("timeline", 20) if t.session == session][-n:]
        return "\n\n".join(t.chunk() for t in turns)

    def candidates(self, persona: str, n_own: int, n_shared: int) -> list[str]:
        """Chunks Jev may choose from for this persona, oldest first, de-duplicated."""
        seen, out = set(), []
        turns = sorted(self.own(persona, n_own) + self.shared(persona, n_shared), key=lambda t: t.ts)
        for t in turns:
            c = t.chunk()
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out
