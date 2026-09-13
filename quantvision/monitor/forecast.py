from __future__ import annotations

import math
from typing import Any

import pandas as pd

from quantvision.config import Mapping, Settings
from quantvision.etl.db import fetchall


def _haystack(market: dict[str, Any]) -> str:
    parts = [
        market.get("title") or "",
        market.get("question") or "",
        market.get("ticker") or "",
        market.get("category") or "",
    ]
    return " ".join(parts).lower()


def match_ticker(market: dict[str, Any], mappings: list[Mapping]) -> str | None:
    text = _haystack(market)
    for mapping in mappings:
        if any(keyword in text for keyword in mapping.keywords):
            return mapping.ticker
    return None


def signed_flow(conn, market_id: str, limit: int = 80) -> float:
    rows = fetchall(
        conn,
        "SELECT price, size, side, outcome FROM trades WHERE market_id=? ORDER BY traded_at DESC LIMIT ?",
        (market_id, limit),
    )
    if not rows:
        return 0.0
    score = 0.0
    total = 0.0
    for row in rows:
        size = float(row["size"] or 0.0)
        side = str(row.get("side") or "").lower()
        outcome = str(row.get("outcome") or "").lower()
        direction = 0.0
        if side in {"buy", "yes"} or outcome in {"yes", "up"}:
            direction = 1.0
        elif side in {"sell", "no"} or outcome in {"no", "down"}:
            direction = -1.0
        score += direction * size
        total += size
    if total <= 0:
        return 0.0
    return max(-1.0, min(1.0, score / total))


def market_consensus(market: dict[str, Any], book: dict[str, Any] | None) -> float | None:
    if book:
        for key in ("mid", "best_bid", "best_ask"):
            value = book.get(key)
            if value is not None:
                return float(value)
    for key in ("mid_price", "last_price", "best_bid", "best_ask"):
        value = market.get(key)
        if value is not None:
            return float(value)
    return None


def app_probability(
    conn,
    market: dict[str, Any],
    book: dict[str, Any] | None,
    settings: Settings,
) -> tuple[float | None, str]:
    ticker = match_ticker(market, settings.mappings)
    if ticker:
        from quantvision.artifacts import load_latest_ensemble

        artifact = load_latest_ensemble(settings.artifacts_dir, ticker)
        if artifact:
            return float(artifact["latest_ensemble"]), f"ensemble:{ticker}"

    stored = fetchall(
        conn,
        """
        SELECT predicted_prob, model_name FROM predictions
        WHERE market_id=? AND model_name NOT LIKE 'auto:%'
        ORDER BY created_at DESC LIMIT 1
        """,
        (market["market_id"],),
    )
    if stored:
        return float(stored[0]["predicted_prob"]), stored[0]["model_name"]

    if book and book.get("microprice") is not None:
        return float(book["microprice"]), "microprice"
    consensus = market_consensus(market, book)
    if consensus is None:
        return None, "none"
    return float(consensus), "mid"


def log_loss(y_true: int, y_prob: float) -> float:
    p = min(1 - 1e-9, max(1e-9, y_prob))
    return - (y_true * math.log(p) + (1 - y_true) * math.log(1 - p))


def performance_row(
    market: dict[str, Any],
    model_name: str,
    predicted: float,
    consensus: float,
    created_at: str,
) -> dict[str, Any]:
    deviation = predicted - consensus
    settled = None
    status = (market.get("status") or "").lower()
    extra = market.get("extra_json")
    result = None
    if status in {"settled", "determined", "finalized", "closed"}:
        result = market.get("last_price")
        if result is not None:
            settled = 1 if float(result) >= 0.5 else 0
    brier = None
    ll = None
    if settled is not None:
        brier = (predicted - settled) ** 2
        ll = log_loss(settled, predicted)
    return {
        "market_id": market["market_id"],
        "model_name": model_name,
        "predicted_prob": predicted,
        "market_prob": consensus,
        "deviation": deviation,
        "abs_deviation": abs(deviation),
        "is_major": int(market.get("is_major") or 0),
        "settled_result": settled,
        "brier": brier,
        "log_loss": ll,
        "created_at": created_at,
    }


def trades_frame(conn, market_id: str) -> pd.DataFrame:
    rows = fetchall(
        conn,
        "SELECT traded_at, price, size FROM trades WHERE market_id=? ORDER BY traded_at",
        (market_id,),
    )
    return pd.DataFrame(rows)
