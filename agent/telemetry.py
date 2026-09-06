"""
telemetry.py
------------
Structured logging of every agent interaction, so "agent performance"
isn't just a claim -- you have a log file (and a couple of derived
stats) to show evaluators. Each turn appends one JSON line to
logs/interactions.jsonl with:

    timestamp, question, route (faq/sql/vector/both/none),
    tool_calls (names + args), latency_sec, success, error (if any)

This is intentionally file-based JSONL (not a DB) so it's trivial to
`tail -f` during a live demo, or load into pandas for a quick chart.

Usage: call `log_interaction(...)` from agent/router.py after each turn.
Call `summary_stats()` any time (e.g. from a small "Metrics" tab in the
UI) to get aggregate numbers.
"""

import json
import time
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_PATH = LOG_DIR / "interactions.jsonl"


def _route_from_trace(trace: dict) -> str:
    if trace.get("used_faq_cache"):
        return "faq"
    tools = {c["tool"] for c in trace.get("tool_calls", [])}
    has_sql = "query_hospital_database" in tools
    has_vector = "search_hospital_website" in tools
    if has_sql and has_vector:
        return "both"
    if has_sql:
        return "sql"
    if has_vector:
        return "vector"
    return "none"


def log_interaction(question: str, trace: dict, latency_sec: float, error: str | None = None):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": question,
        "route": _route_from_trace(trace),
        "tool_calls": [{"tool": c["tool"], "args": c.get("args", {})} for c in trace.get("tool_calls", [])],
        "used_faq_cache": trace.get("used_faq_cache", False),
        "latency_sec": round(latency_sec, 3),
        "success": error is None,
        "error": error,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def summary_stats() -> dict:
    """Aggregate stats across all logged interactions (for a metrics view)."""
    if not LOG_PATH.exists():
        return {"total_interactions": 0}

    records = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return {"total_interactions": 0}

    total = len(records)
    successes = sum(1 for r in records if r["success"])
    avg_latency = sum(r["latency_sec"] for r in records) / total
    route_counts = {}
    for r in records:
        route_counts[r["route"]] = route_counts.get(r["route"], 0) + 1

    return {
        "total_interactions": total,
        "success_rate": round(successes / total, 3),
        "avg_latency_sec": round(avg_latency, 2),
        "route_breakdown": route_counts,
    }
