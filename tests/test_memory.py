"""
tests/test_memory.py
---------------------
Unit tests for agent/memory.py's buffer/compaction logic. These don't
touch any LLM or embedding model -- `compact()` takes a plain callback,
so we can test the compaction *trigger* logic in isolation with a fake
summarizer.

Run with:
    pytest tests/test_memory.py -v
"""

from agent.memory import ConversationMemory, BUFFER_WINDOW


def fake_summarize(existing_summary, turns):
    """A deterministic stand-in for the real LLM-backed summarizer."""
    new_facts = " ".join(t["content"] for t in turns)
    return f"{existing_summary} | {new_facts}".strip(" |")


def test_no_compaction_needed_below_window():
    memory = ConversationMemory()
    for i in range(BUFFER_WINDOW):
        memory.add_turn("user", f"message {i}")
    assert memory.needs_compaction() is False


def test_compaction_triggers_above_window():
    memory = ConversationMemory()
    for i in range(BUFFER_WINDOW + 2):
        memory.add_turn("user", f"message {i}")
    assert memory.needs_compaction() is True


def test_compact_moves_overflow_into_summary():
    memory = ConversationMemory()
    for i in range(BUFFER_WINDOW + 3):
        memory.add_turn("user", f"message {i}")

    memory.compact(fake_summarize)

    assert len(memory.buffer) == BUFFER_WINDOW
    assert memory.summary != ""
    assert "message 0" in memory.summary  # oldest turn got folded into summary
    assert memory.buffer[0]["content"] == "message 3"  # newest BUFFER_WINDOW retained verbatim


def test_context_string_includes_summary_and_buffer():
    memory = ConversationMemory(summary="Patient asked about cardiology.")
    memory.add_turn("user", "What's the fee?")
    context = memory.as_context_string()
    assert "cardiology" in context
    assert "What's the fee?" in context


def test_to_dict_from_dict_roundtrip():
    memory = ConversationMemory(summary="test summary")
    memory.add_turn("user", "hello")
    memory.add_turn("assistant", "hi there")

    restored = ConversationMemory.from_dict(memory.to_dict())

    assert restored.summary == memory.summary
    assert restored.buffer == memory.buffer
