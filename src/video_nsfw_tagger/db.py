"""SQLite index for scanned video results."""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    duration_s REAL,
    frames_total INTEGER,
    nsfw_percent REAL,
    max_score REAL,
    verdict TEXT,
    threshold REAL,
    model TEXT,
    vlm_model TEXT,
    act_tags TEXT,
    sidecar_path TEXT,
    scanned_at TEXT
);
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    """Open or create the SQLite index and initialise the schema.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        A ``sqlite3.Connection`` in the default commit/rollback mode.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def upsert_video(conn: sqlite3.Connection, record: Dict[str, Any]) -> None:
    """Insert or update a video row by ``path``.

    Args:
        conn: SQLite connection.
        record: Column-value mapping. Must include ``path``.
    """
    fields = list(record.keys())
    placeholders = ", ".join(f":{f}" for f in fields)
    updates = ", ".join(f"{f} = :{f}" for f in fields if f != "path")
    sql = (
        f"INSERT INTO videos ({', '.join(fields)}) VALUES ({placeholders}) "
        f"ON CONFLICT(path) DO UPDATE SET {updates}"
    )
    conn.execute(sql, record)
    conn.commit()


def query_videos(
    conn: sqlite3.Connection,
    verdict: Optional[str] = None,
    min_percent: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Query the index, optionally filtering by verdict and minimum NSFW %.

    Args:
        conn: SQLite connection.
        verdict: Filter by ``verdict`` column.
        min_percent: Minimum ``nsfw_percent`` filter.

    Returns:
        Matching rows as dictionaries.
    """
    where: List[str] = []
    params: Dict[str, Any] = {}
    if verdict:
        where.append("verdict = :verdict")
        params["verdict"] = verdict
    if min_percent is not None:
        where.append("nsfw_percent >= :min_percent")
        params["min_percent"] = min_percent

    clause = f"WHERE {' AND '.join(where)}" if where else ""
    sql = f"SELECT * FROM videos {clause} ORDER BY id"
    conn.row_factory = sqlite3.Row
    cur = conn.execute(sql, params)
    return [dict(r) for r in cur.fetchall()]
