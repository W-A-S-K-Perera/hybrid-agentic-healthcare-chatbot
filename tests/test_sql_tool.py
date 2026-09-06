"""
tests/test_sql_tool.py
-----------------------
Unit tests for agent/sql_tool.py -- primarily the read-only safety
guardrail, since a hallucinated destructive query is the single
highest-risk failure mode for a hospital-facing text-to-SQL tool.

Run with:
    pytest tests/test_sql_tool.py -v
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module", autouse=True)
def ensure_test_db():
    """Build hospital.db from the real data.sql once before these tests run."""
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "setup_database.py")],
        check=True,
        cwd=ROOT,
    )


def test_valid_select_returns_rows():
    from agent.sql_tool import run_sql_query
    result = run_sql_query("SELECT name FROM doctors WHERE specialty_id = 7")
    assert result["success"] is True
    assert result["row_count"] >= 1
    assert any("Priyadarshan" in row["name"] for row in result["rows"])


def test_query_with_no_matches_succeeds_with_empty_rows():
    from agent.sql_tool import run_sql_query
    result = run_sql_query("SELECT * FROM doctors WHERE name = 'Dr. Nobody'")
    assert result["success"] is True
    assert result["row_count"] == 0


@pytest.mark.parametrize(
    "malicious_sql",
    [
        "DROP TABLE doctors",
        "DELETE FROM doctors",
        "UPDATE doctors SET consultation_fee = 0",
        "INSERT INTO doctors (name) VALUES ('hacked')",
        "SELECT * FROM doctors; DROP TABLE doctors;",
        "ATTACH DATABASE '/etc/passwd' AS pwn",
        "PRAGMA writable_schema = 1",
    ],
)
def test_destructive_queries_are_blocked(malicious_sql):
    from agent.sql_tool import run_sql_query
    result = run_sql_query(malicious_sql)
    assert result["success"] is False
    assert "read-only" in result["error"].lower()


def test_row_cap_is_enforced():
    from agent.sql_tool import run_sql_query, MAX_ROWS
    result = run_sql_query("SELECT * FROM lab_tests")
    assert result["row_count"] <= MAX_ROWS


def test_invalid_sql_syntax_fails_gracefully():
    from agent.sql_tool import run_sql_query
    result = run_sql_query("SELECT FROM WHERE garbage")
    assert result["success"] is False
    assert "error" in result
