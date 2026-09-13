from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from quantvision.config import Settings, load_settings
from quantvision.etl import kalshi, polymarket
from quantvision.etl.db import insert_many, session, upsert_many
from quantvision.etl.normalize import NormalizedMarket, utc_now


def _mark_major(markets: list[NormalizedMarket], min_volume: float) -> None:
    for market in markets:
        volume = market.volume_24h or market.volume or market.liquidity or 0.0
        market.is_major = 1 if volume >= min_volume else 0


def _ingest_venue(settings: Settings, venue: str, conn, fetch_markets, fetch_trades, fetch_book) -> dict[str, int]:
    markets = fetch_markets(settings)
    _mark_major(markets, settings.major_min_volume)
    market_rows = [m.as_row() for m in markets]
    upsert_many(conn, "markets", market_rows, "market_id")
    trades_n = 0
    books_n = 0
    for market in markets:
        try:
            trades = fetch_trades(settings, market)
            trades_n += upsert_many(conn, "trades", [t.as_row() for t in trades], "trade_id")
        except Exception as exc:
            insert_many(
                conn,
                "etl_runs",
                [
                    {
                        "started_at": utc_now(),
                        "finished_at": utc_now(),
                        "status": "partial",
                        "error": f"{venue} trades {market.market_id}: {exc}",
                    }
                ],
            )
        try:
            book = fetch_book(settings, market)
            insert_many(conn, "orderbook_snapshots", [book.as_row()])
            books_n += 1
            conn.execute(
                """
                UPDATE markets
                SET best_bid=?, best_ask=?, mid_price=?, last_price=COALESCE(?, last_price), updated_at=?
                WHERE market_id=?
                """,
                (
                    book.best_bid,
                    book.best_ask,
                    book.mid,
                    book.microprice or book.mid,
                    utc_now(),
                    market.market_id,
                ),
            )
        except Exception as exc:
            insert_many(
                conn,
                "etl_runs",
                [
                    {
                        "started_at": utc_now(),
                        "finished_at": utc_now(),
                        "status": "partial",
                        "error": f"{venue} book {market.market_id}: {exc}",
                    }
                ],
            )
        time.sleep(0.05)
    return {"markets": len(markets), "trades": trades_n, "books": books_n}


def export_snapshot(settings: Settings, conn) -> Path:
    markets = [dict(row) for row in conn.execute("SELECT * FROM markets ORDER BY COALESCE(volume_24h, volume, 0) DESC")]
    trades = [
        dict(row)
        for row in conn.execute("SELECT * FROM trades ORDER BY traded_at DESC LIMIT 400")
    ]
    books = [
        dict(row)
        for row in conn.execute(
            """
            SELECT * FROM orderbook_snapshots
            WHERE snapshot_id IN (SELECT MAX(snapshot_id) FROM orderbook_snapshots GROUP BY market_id)
            """
        )
    ]
    alerts = [dict(row) for row in conn.execute("SELECT * FROM alerts ORDER BY created_at DESC LIMIT 50")]
    deviations = [
        dict(row)
        for row in conn.execute("SELECT * FROM performance_log ORDER BY created_at DESC LIMIT 200")
    ]
    payload: dict[str, Any] = {
        "generated_at": utc_now(),
        "markets": markets,
        "trades": trades,
        "orderbooks": books,
        "alerts": alerts,
        "performance": deviations,
    }
    path = settings.snapshot_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=str))
    if settings.website_snapshot_path:
        try:
            settings.website_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            settings.website_snapshot_path.write_text(json.dumps(payload, default=str))
        except OSError:
            pass
    return path


def run_etl(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    started = utc_now()
    totals = {"markets": 0, "trades": 0, "books": 0}
    error = None
    with session(settings.db_path) as conn:
        try:
            if settings.kalshi.enabled:
                stats = _ingest_venue(
                    settings,
                    "kalshi",
                    conn,
                    kalshi.fetch_markets,
                    kalshi.fetch_trades,
                    kalshi.fetch_orderbook,
                )
                for key, value in stats.items():
                    totals[key] += value
            if settings.polymarket.enabled:
                stats = _ingest_venue(
                    settings,
                    "polymarket",
                    conn,
                    polymarket.fetch_markets,
                    polymarket.fetch_trades,
                    polymarket.fetch_orderbook,
                )
                for key, value in stats.items():
                    totals[key] += value
            snapshot = str(export_snapshot(settings, conn))
            status = "ok"
        except Exception as exc:
            error = str(exc)
            snapshot = None
            status = "error"
            raise
        finally:
            insert_many(
                conn,
                "etl_runs",
                [
                    {
                        "started_at": started,
                        "finished_at": utc_now(),
                        "status": status if error is None else "error",
                        "markets_upserted": totals["markets"],
                        "trades_upserted": totals["trades"],
                        "books_upserted": totals["books"],
                        "error": error,
                    }
                ],
            )
    return {**totals, "status": "ok", "snapshot": snapshot, "db": str(settings.db_path)}
