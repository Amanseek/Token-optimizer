"""SQLite storage for request logs and sticky routing decisions."""

from __future__ import annotations

import sqlite3
import threading
import time
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    engineer TEXT NOT NULL,
    team TEXT,
    conversation TEXT,
    model_requested TEXT,
    model_used TEXT,
    route_decision TEXT,          -- 'none' | 'shadow' | 'routed' | 'cascaded'
    stream INTEGER,
    status INTEGER,
    latency_ms INTEGER,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_creation_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0,
    -- What the same request would have cost with no optimization (estimate).
    baseline_cost_usd REAL DEFAULT 0,
    trimmed_chars INTEGER DEFAULT 0,
    deduped_chars INTEGER DEFAULT 0,
    cache_breakpoints_added INTEGER DEFAULT 0,
    shadow_model TEXT,
    shadow_cost_usd REAL
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
CREATE INDEX IF NOT EXISTS idx_requests_engineer ON requests(engineer);

CREATE TABLE IF NOT EXISTS route_decisions (
    conversation TEXT PRIMARY KEY,
    target_model TEXT,           -- NULL means 'keep original model'
    reason TEXT,
    ts REAL
);
"""


class Store:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def log_request(self, row: dict[str, Any]) -> None:
        row = {"ts": time.time(), **row}
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        with self._lock:
            self._conn.execute(f"INSERT INTO requests ({cols}) VALUES ({marks})", list(row.values()))
            self._conn.commit()

    def get_route(self, conversation: str) -> tuple[bool, str | None, str | None]:
        """Returns (found, target_model, reason)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT target_model, reason FROM route_decisions WHERE conversation = ?", (conversation,)
            )
            r = cur.fetchone()
        if r is None:
            return False, None, None
        return True, r["target_model"], r["reason"]

    def save_route(self, conversation: str, target_model: str | None, reason: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO route_decisions (conversation, target_model, reason, ts) VALUES (?, ?, ?, ?)",
                (conversation, target_model, reason, time.time()),
            )
            self._conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
