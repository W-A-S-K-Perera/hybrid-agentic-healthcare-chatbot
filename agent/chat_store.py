"""
chat_store.py
-------------
Stores chat history and conversation memory in SQLite for persistence across app restarts.
"""

import json
import sqlite3
import uuid
from pathlib import Path

from agent.memory import ConversationMemory

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "chat_history.db"

NEW_CHAT_TITLE = "New chat"
TITLE_MAX_LEN = 40


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_threads (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            messages_json TEXT NOT NULL,
            memory_json TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    return conn


def create_thread() -> str:
    """Create a new, empty chat thread and return its id."""
    thread_id = str(uuid.uuid4())
    conn = _get_conn()
    conn.execute(
        "INSERT INTO chat_threads (thread_id, title, messages_json, memory_json) VALUES (?, ?, ?, ?)",
        (thread_id, NEW_CHAT_TITLE, json.dumps([]), json.dumps(ConversationMemory().to_dict())),
    )
    conn.commit()
    conn.close()
    return thread_id


def list_threads() -> list[dict]:
    """Return all threads, most recently updated first (for the sidebar list)."""
    conn = _get_conn()
    cur = conn.execute(
        "SELECT thread_id, title, updated_at FROM chat_threads ORDER BY updated_at DESC"
    )
    rows = [{"thread_id": r[0], "title": r[1], "updated_at": r[2]} for r in cur.fetchall()]
    conn.close()
    return rows


def load_thread(thread_id: str) -> dict:
    """Load a thread's messages + memory. Returns an empty shell if not found."""
    conn = _get_conn()
    cur = conn.execute(
        "SELECT title, messages_json, memory_json FROM chat_threads WHERE thread_id = ?",
        (thread_id,),
    )
    row = cur.fetchone()
    conn.close()

    if row is None:
        return {"title": NEW_CHAT_TITLE, "messages": [], "memory": ConversationMemory()}

    title, messages_json, memory_json = row
    return {
        "title": title,
        "messages": json.loads(messages_json),
        "memory": ConversationMemory.from_dict(json.loads(memory_json)),
    }


def save_thread(thread_id: str, title: str, messages: list, memory: ConversationMemory):
    """Persist a thread's current title, messages, and memory."""
    conn = _get_conn()
    conn.execute(
        """
        UPDATE chat_threads
        SET title = ?, messages_json = ?, memory_json = ?, updated_at = CURRENT_TIMESTAMP
        WHERE thread_id = ?
        """,
        (title, json.dumps(messages), json.dumps(memory.to_dict()), thread_id),
    )
    conn.commit()
    conn.close()


def delete_thread(thread_id: str):
    conn = _get_conn()
    conn.execute("DELETE FROM chat_threads WHERE thread_id = ?", (thread_id,))
    conn.commit()
    conn.close()


def auto_title(first_message: str) -> str:
    """Derive a short sidebar title from the first user message, ChatGPT-style."""
    title = first_message.strip().split("\n")[0]
    if len(title) > TITLE_MAX_LEN:
        return title[:TITLE_MAX_LEN] + "…"
    return title
