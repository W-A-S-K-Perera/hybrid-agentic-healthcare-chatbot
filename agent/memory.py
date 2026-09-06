"""
memory.py
---------
Simple agentic memory for multi-turn context.

Two layers, kept intentionally simple so it's easy to explain in an
interview/demo:

1. Short-term buffer: the last N raw turns are kept verbatim and sent
   with every request, so follow-ups like "what about on Saturdays?"
   or "how much does that cost?" resolve correctly against the prior
   turn.

2. Rolling summary: once the conversation exceeds the buffer window,
   older turns are compressed into a running natural-language summary
   (via an LLM call) instead of being dropped, so long conversations
   don't lose earlier context (e.g. "earlier you said you were looking
   for a cardiologist for your father") while keeping the prompt small.

This is deliberately in-process/session-scoped (Streamlit session
state holds the ConversationMemory instance) rather than persisted to
disk/DB, but `to_dict`/`from_dict` are provided so it's trivial to
persist per-user if you want true cross-session memory later.
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
