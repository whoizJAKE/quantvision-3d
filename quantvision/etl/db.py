from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    market_id TEXT PRIMARY KEY,
    venue TEXT NOT NULL,
    venue_market_id TEXT NOT NULL,
    event_id TEXT,
    ticker TEXT,
    title TEXT,
    question TEXT,
    category TEXT,
    status TEXT,
    yes_token_id TEXT,
    no_token_id TEXT,
    close_time TEXT,
    last_price REAL,
    best_bid REAL,
    best_ask REAL,
    mid_price REAL,
    volume REAL,
    volume_24h REAL,
    open_interest REAL,
    liquidity REAL,
    is_major INTEGER DEFAULT 0,
    extra_json TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    venue TEXT NOT NULL,
    traded_at TEXT NOT NULL,
    traded_at_unix INTEGER,
    price REAL NOT NULL,
    size REAL NOT NULL,
    notional REAL,
    side TEXT,
    outcome TEXT,
    is_block INTEGER DEFAULT 0,
    extra_json TEXT,
    ingested_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_market_time ON trades(market_id, traded_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_venue_time ON trades(venue, traded_at DESC);

CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    venue TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    best_bid REAL,
    best_ask REAL,
    mid REAL,
    microprice REAL,
    spread REAL,
    bid_depth REAL,
    ask_depth REAL,
    imbalance REAL,
    bids_json TEXT,
    asks_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_book_market_time ON orderbook_snapshots(market_id, captured_at DESC);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    predicted_prob REAL NOT NULL,
    market_prob REAL,
    deviation REAL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pred_market_time ON predictions(market_id, created_at DESC);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    title TEXT,
    predicted_prob REAL,
    market_prob REAL,
    deviation REAL,
    threshold REAL,
    severity TEXT,
    message TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS performance_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    predicted_prob REAL,
    market_prob REAL,
    deviation REAL,
    abs_deviation REAL,
    is_major INTEGER,
    settled_result INTEGER,
    brier REAL,
    log_loss REAL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_perf_market_time ON performance_log(market_id, created_at DESC);

CREATE TABLE IF NOT EXISTS etl_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT,
    finished_at TEXT,
    status TEXT,
    markets_upserted INTEGER DEFAULT 0,
    trades_upserted INTEGER DEFAULT 0,
    books_upserted INTEGER DEFAULT 0,
    error TEXT
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def session(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _placeholders(columns: Iterable[str]) -> str:
    cols = list(columns)
    return ", ".join(cols), ", ".join("?" for _ in cols)


def upsert_many(conn: sqlite3.Connection, table: str, rows: list[dict[str, Any]], key: str) -> int:
    if not rows:
        return 0
    columns = list(rows[0].keys())
    col_sql, placeholders = _placeholders(columns)
    updates = ", ".join(f"{c}=excluded.{c}" for c in columns if c != key)
    sql = f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders}) ON CONFLICT({key}) DO UPDATE SET {updates}"
    conn.executemany(sql, [tuple(row.get(c) for c in columns) for row in rows])
    return len(rows)


def insert_many(conn: sqlite3.Connection, table: str, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    columns = list(rows[0].keys())
    col_sql, placeholders = _placeholders(columns)
    sql = f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders})"
    conn.executemany(sql, [tuple(row.get(c) for c in columns) for row in rows])
    return len(rows)


def fetchall(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def latest_books(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    sql = """
    SELECT * FROM orderbook_snapshots
    WHERE snapshot_id IN (
        SELECT MAX(snapshot_id) FROM orderbook_snapshots GROUP BY market_id
    )
    """
    return {row["market_id"]: row for row in fetchall(conn, sql)}
