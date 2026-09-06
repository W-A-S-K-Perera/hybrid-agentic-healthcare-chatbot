#import libaries 
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SQL_SOURCE = ROOT / "data" / "data.sql"
DB_PATH = ROOT / "data" / "hospital.db"


def sanitize_for_sqlite(sql_text: str) -> str:
    """Strip/rewrite the handful of MySQL-only syntax bits SQLite chokes on."""
    text = sql_text

    # SQLite auto-increments INTEGER PRIMARY KEY columns natively.
    text = re.sub(r"\bINT PRIMARY KEY AUTO_INCREMENT\b", "INTEGER PRIMARY KEY AUTOINCREMENT", text)
    text = re.sub(r"\bAUTO_INCREMENT\b", "", text)

    # SQLite has no native DECIMAL/VARCHAR(n)/TIME types, but it accepts
    # them syntactically (type affinity), so we leave those as-is on
    # purpose -- no functional change needed.

    return text


def build_database():
    if not SQL_SOURCE.exists():
        raise FileNotFoundError(f"Could not find {SQL_SOURCE}. Place data.sql in the data/ folder.")

    raw_sql = SQL_SOURCE.read_text(encoding="utf-8")
    clean_sql = sanitize_for_sqlite(raw_sql)

    # Fresh start every time this script runs.
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(clean_sql)
        conn.commit()
    finally:
        conn.close()

    print(f"Database created at: {DB_PATH}")
    _print_summary()


def _print_summary():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for table in ["specialties", "doctors", "channeling_sessions", "lab_tests", "health_packages"]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"   - {table}: {count} rows")
    conn.close()


if __name__ == "__main__":
    build_database()
