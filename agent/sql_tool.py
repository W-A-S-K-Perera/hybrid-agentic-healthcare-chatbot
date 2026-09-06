"""
sql_tool.py
Provides a safe Text-to-SQL tool that executes read-only SELECT queries and returns limited, JSON-friendly results.
"""

import re
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "hospital.db"
MAX_ROWS = 50

SCHEMA_DESCRIPTION = """
TABLE specialties(id, name, department)
TABLE doctors(id, name, specialty_id -> specialties.id, qualifications, consultation_fee)
TABLE channeling_sessions(id, doctor_id -> doctors.id, day_of_week, start_time, end_time, room_number, max_patients)
TABLE lab_tests(id, test_code, test_name, category, price, fasting_required_hours, preparation_instructions, report_delivery_hours)
TABLE health_packages(id, package_name, category, price, target_audience, included_tests_and_services)
""".strip()


class UnsafeQueryError(ValueError):
    pass


def _is_select_only(sql: str) -> bool:
    """Allow only a single read-only SELECT/WITH statement."""
    stripped = sql.strip().rstrip(";")
    if ";" in stripped:  # block statement stacking
        return False
    forbidden = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|PRAGMA)\b",
        re.IGNORECASE,
    )
    if forbidden.search(stripped):
        return False
    return bool(re.match(r"^\s*(SELECT|WITH)\b", stripped, re.IGNORECASE))


def run_sql_query(sql: str) -> dict:
    """
    Execute a read-only SQL query against hospital.db.

    Returns:
        {"success": True, "rows": [...], "row_count": N} on success, or
        {"success": False, "error": "..."} on failure.
    """
    if not _is_select_only(sql):
        return {
            "success": False,
            "error": "Only single, read-only SELECT/WITH queries are permitted.",
        }

    if not DB_PATH.exists():
        return {
            "success": False,
            "error": "hospital.db not found. Run scripts/setup_database.py first.",
        }

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchmany(MAX_ROWS)
        result = [dict(row) for row in rows]
        conn.close()
        return {"success": True, "rows": result, "row_count": len(result)}
    except sqlite3.Error as exc:
        return {"success": False, "error": str(exc)}


# Tool schema in the format the LLM function-calling API expects.
SQL_TOOL_SPEC = {
    "name": "query_hospital_database",
    "description": (
        "Run a read-only SQL SELECT query against the hospital's structured database "
        "to answer questions about doctors, specialties, channeling/appointment schedules, "
        "lab test prices/prep instructions, or health package prices/inclusions. "
        f"Schema:\n{SCHEMA_DESCRIPTION}"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "A single valid SQLite SELECT statement answering the user's question.",
            }
        },
        "required": ["sql"],
    },
}
