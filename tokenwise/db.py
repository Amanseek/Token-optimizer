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

CREATE TABLE IF NOT EXISTS response_cache (
    key TEXT PRIMARY KEY,
    created REAL NOT NULL,
    content_type TEXT,
    body BLOB NOT NULL,
    model TEXT,
    cost_usd REAL DEFAULT 0,
    hits INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS clients (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    key_hash TEXT UNIQUE NOT NULL,     -- sha256 of the tw- gateway key (the key itself is never stored)
    key_prefix TEXT,                   -- first characters, to recognise a key in lists
    created REAL,
    active INTEGER DEFAULT 1,
    capture INTEGER DEFAULT 1,         -- store traces for this client
    mask INTEGER DEFAULT 1,            -- mask personal data in stored traces
    openai_base_url TEXT,              -- any OpenAI-compatible host
    openai_key_enc TEXT,               -- upstream keys, encrypted with TOKENWISE_SECRET
    gemini_key_enc TEXT,
    anthropic_key_enc TEXT
);

CREATE TABLE IF NOT EXISTS traces (
    id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    client_id TEXT,
    provider TEXT,                     -- openai | gemini | anthropic
    upstream_host TEXT,
    engineer TEXT,
    feature TEXT,
    workflow_key TEXT,                 -- feature label, or a hash of the system prompt
    workflow_label TEXT,
    model TEXT,
    status INTEGER,
    stream INTEGER,
    system TEXT,
    messages TEXT,                     -- conversation sent, normalised (JSON)
    tools TEXT,                        -- declared tool names (JSON list)
    last_user TEXT,
    input_hash TEXT,
    output_text TEXT,
    tool_calls TEXT,                   -- tool calls the model made (JSON list)
    finish TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0,
    latency_ms INTEGER,
    success INTEGER,                   -- automatic: 1 ok; 0 error/blocked/cut off/empty/retried
    reason TEXT,
    feedback INTEGER,                  -- from the app: 1 good, 0 bad, NULL none
    feedback_note TEXT,
    corrected_output TEXT              -- a better answer supplied by the app or a reviewer
);
CREATE INDEX IF NOT EXISTS idx_traces_wf ON traces(client_id, workflow_key, ts);
CREATE INDEX IF NOT EXISTS idx_traces_retry ON traces(client_id, engineer, workflow_key, input_hash, ts);

CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    actor TEXT,                        -- admin (dashboard/API), cli, or client:<name>
    action TEXT,                       -- view_dashboard, export, delete, create_client, ...
    target TEXT,
    detail TEXT
);

CREATE TABLE IF NOT EXISTS route_decisions (
    conversation TEXT PRIMARY KEY,
    target_model TEXT,           -- NULL means 'keep original model'
    reason TEXT,
    ts REAL
);
"""


CLIENT_NEW_COLUMNS = {
    "capture_rate": "REAL DEFAULT 1.0",    # share of requests to capture (0-1)
    "retention_days": "INTEGER DEFAULT 90", # traces older than this are deleted automatically
    "policy": "TEXT",                       # fallback / shadow settings (policies.py), keys encrypted
}

NEW_COLUMNS = {
    "provider": "TEXT DEFAULT 'anthropic'",
    "feature": "TEXT",
    "audio_tokens": "INTEGER DEFAULT 0",
}


class Store:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        # Columns added after v0.1; add them to older databases.
        have = {r["name"] for r in self._conn.execute("PRAGMA table_info(requests)")}
        for col, decl in NEW_COLUMNS.items():
            if col not in have:
                self._conn.execute(f"ALTER TABLE requests ADD COLUMN {col} {decl}")
        have = {r["name"] for r in self._conn.execute("PRAGMA table_info(clients)")}
        for col, decl in CLIENT_NEW_COLUMNS.items():
            if col not in have:
                self._conn.execute(f"ALTER TABLE clients ADD COLUMN {col} {decl}")
        self._conn.commit()

    def audit(self, actor: str, action: str, target: str = "", detail: str = "") -> None:
        self.insert("audit", {"ts": time.time(), "actor": actor, "action": action, "target": target,
                              "detail": detail})

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

    # ---------------- response cache
    def cache_get(self, key: str, ttl: int) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM response_cache WHERE key = ?", (key,))
            r = cur.fetchone()
            if r is None:
                return None
            if time.time() - r["created"] > ttl:
                self._conn.execute("DELETE FROM response_cache WHERE key = ?", (key,))
                self._conn.commit()
                return None
            self._conn.execute("UPDATE response_cache SET hits = hits + 1 WHERE key = ?", (key,))
            self._conn.commit()
            return dict(r)

    def cache_put(self, key: str, body: bytes, content_type: str, model: str, cost_usd: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO response_cache (key, created, content_type, body, model, cost_usd, hits) "
                "VALUES (?, ?, ?, ?, ?, ?, 0)",
                (key, time.time(), content_type, body, model, cost_usd),
            )
            self._conn.commit()

    # ---------------- generic helpers (clients, traces)
    def insert(self, table: str, row: dict[str, Any]) -> None:
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        with self._lock:
            self._conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(row.values()))
            self._conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> int:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur.rowcount

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
