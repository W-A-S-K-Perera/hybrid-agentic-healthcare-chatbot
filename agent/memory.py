"""
memory.py
Manages short-term conversation history and rolling summaries to maintain context across multiple turns.
"""

from dataclasses import dataclass, field

BUFFER_WINDOW = 6  # number of raw turns (user+assistant pairs count as 2) to keep verbatim


@dataclass
class ConversationMemory:
    summary: str = ""
    buffer: list = field(default_factory=list)  # list of {"role": ..., "content": ...}

    def add_turn(self, role: str, content: str):
        self.buffer.append({"role": role, "content": content})

    def needs_compaction(self) -> bool:
        return len(self.buffer) > BUFFER_WINDOW

    def compact(self, summarize_fn):
        """
        Compress the oldest turns into `self.summary` using the provided
        `summarize_fn(existing_summary: str, turns: list[dict]) -> str`
        callback (kept as a callback so this module has no direct LLM
        dependency -- router.py supplies the actual LLM call).
        """
        overflow = len(self.buffer) - BUFFER_WINDOW
        if overflow <= 0:
            return
        to_summarize = self.buffer[:overflow]
        self.buffer = self.buffer[overflow:]
        self.summary = summarize_fn(self.summary, to_summarize)

    def as_context_string(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"Summary of earlier conversation:\n{self.summary}")
        if self.buffer:
            recent = "\n".join(f"{t['role']}: {t['content']}" for t in self.buffer)
            parts.append(f"Recent turns:\n{recent}")
        return "\n\n".join(parts)

    def to_dict(self) -> dict:
        return {"summary": self.summary, "buffer": self.buffer}

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationMemory":
        return cls(summary=data.get("summary", ""), buffer=data.get("buffer", []))
