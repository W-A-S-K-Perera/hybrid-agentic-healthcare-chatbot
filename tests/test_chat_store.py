"""
tests/test_chat_store.py
--------------------------
Unit tests for agent/chat_store.py (SQLite-backed chat thread storage
powering the sidebar's chat history).

Run with:
    pytest tests/test_chat_store.py -v
"""

from agent.chat_store import (
    create_thread,
    list_threads,
    load_thread,
    save_thread,
    delete_thread,
    auto_title,
)
from agent.memory import ConversationMemory


def test_create_thread_appears_in_list():
    thread_id = create_thread()
    ids = [t["thread_id"] for t in list_threads()]
    assert thread_id in ids
    delete_thread(thread_id)


def test_new_thread_loads_empty():
    thread_id = create_thread()
    loaded = load_thread(thread_id)
    assert loaded["messages"] == []
    assert loaded["title"] == "New chat"
    delete_thread(thread_id)


def test_save_and_reload_thread_roundtrips():
    thread_id = create_thread()
    messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    memory = ConversationMemory(summary="greeted the patient")

    save_thread(thread_id, "Greeting chat", messages, memory)
    loaded = load_thread(thread_id)

    assert loaded["title"] == "Greeting chat"
    assert loaded["messages"] == messages
    assert loaded["memory"].summary == "greeted the patient"
    delete_thread(thread_id)


def test_delete_thread_removes_it():
    thread_id = create_thread()
    delete_thread(thread_id)
    ids = [t["thread_id"] for t in list_threads()]
    assert thread_id not in ids


def test_auto_title_truncates_long_messages():
    long_message = "a" * 100
    title = auto_title(long_message)
    assert len(title) <= 41  # 40 chars + ellipsis


def test_auto_title_keeps_short_messages():
    title = auto_title("Hi there")
    assert title == "Hi there"
