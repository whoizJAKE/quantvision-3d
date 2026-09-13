from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def parse_time(value: Any) -> tuple[str | None, int | None]:
    if value is None or value == "":
        return None, None
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:
            ts /= 1000.0
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z"), int(ts)
    text = str(value)
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"), int(dt.timestamp())
    except ValueError:
        return text, None


def clip01(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, value))


@dataclass
class NormalizedMarket:
    market_id: str
    venue: str
    venue_market_id: str
    event_id: str | None = None
    ticker: str | None = None
    title: str | None = None
    question: str | None = None
    category: str | None = None
    status: str | None = None
    yes_token_id: str | None = None
    no_token_id: str | None = None
    close_time: str | None = None
    last_price: float | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    mid_price: float | None = None
    volume: float | None = None
    volume_24h: float | None = None
    open_interest: float | None = None
    liquidity: float | None = None
    is_major: int = 0
    extra_json: str | None = None
    updated_at: str = field(default_factory=utc_now)

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NormalizedTrade:
    trade_id: str
    market_id: str
    venue: str
    traded_at: str
    traded_at_unix: int | None
    price: float
    size: float
    notional: float | None = None
    side: str | None = None
    outcome: str | None = None
    is_block: int = 0
    extra_json: str | None = None
    ingested_at: str = field(default_factory=utc_now)

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NormalizedBook:
    market_id: str
    venue: str
    captured_at: str
    best_bid: float | None
    best_ask: float | None
    mid: float | None
    microprice: float | None
    spread: float | None
    bid_depth: float | None
    ask_depth: float | None
    imbalance: float | None
    bids_json: str
    asks_json: str

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def book_from_levels(
    market_id: str,
    venue: str,
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    captured_at: str | None = None,
) -> NormalizedBook:
    bids = sorted([(p, s) for p, s in bids if s > 0], key=lambda x: -x[0])
    asks = sorted([(p, s) for p, s in asks if s > 0], key=lambda x: x[0])
    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None
    bid_sz = bids[0][1] if bids else 0.0
    ask_sz = asks[0][1] if asks else 0.0
    mid = None
    if best_bid is not None and best_ask is not None:
        mid = (best_bid + best_ask) / 2
    elif best_bid is not None:
        mid = best_bid
    elif best_ask is not None:
        mid = best_ask
    micro = None
    depth = bid_sz + ask_sz
    if best_bid is not None and best_ask is not None and depth > 0:
        micro = (best_ask * bid_sz + best_bid * ask_sz) / depth
    spread = (best_ask - best_bid) if best_bid is not None and best_ask is not None else None
    imbalance = ((bid_sz - ask_sz) / depth) if depth > 0 else None
    return NormalizedBook(
        market_id=market_id,
        venue=venue,
        captured_at=captured_at or utc_now(),
        best_bid=best_bid,
        best_ask=best_ask,
        mid=clip01(mid),
        microprice=clip01(micro),
        spread=spread,
        bid_depth=sum(s for _, s in bids[:10]),
        ask_depth=sum(s for _, s in asks[:10]),
        imbalance=imbalance,
        bids_json=json.dumps(bids[:25]),
        asks_json=json.dumps(asks[:25]),
    )


def kalshi_yes_book(orderbook_fp: dict[str, Any]) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    yes_raw = orderbook_fp.get("yes_dollars") or orderbook_fp.get("yes") or []
    no_raw = orderbook_fp.get("no_dollars") or orderbook_fp.get("no") or []
    bids = []
    for level in yes_raw:
        if len(level) < 2:
            continue
        price, size = to_float(level[0]), to_float(level[1])
        if price is not None and size is not None:
            bids.append((price, size))
    asks = []
    for level in no_raw:
        if len(level) < 2:
            continue
        price, size = to_float(level[0]), to_float(level[1])
        if price is not None and size is not None:
            asks.append((1.0 - price, size))
    return bids, asks
