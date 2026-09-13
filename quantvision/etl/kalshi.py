from __future__ import annotations

import json
from typing import Any

from quantvision.config import Settings
from quantvision.etl.http import get_json
from quantvision.etl.normalize import (
    NormalizedMarket,
    NormalizedTrade,
    book_from_levels,
    clip01,
    kalshi_yes_book,
    parse_time,
    to_float,
    utc_now,
)


def _get(settings: Settings, path: str, params: dict[str, Any] | None = None) -> Any:
    return get_json(f"{settings.kalshi_base}{path}", params=params, timeout=settings.request_timeout)


def fetch_events(settings: Settings, tickers: list[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    wanted = {t for t in tickers if t}
    try:
        data = _get(settings, "/events", {"limit": 200, "status": "open", "with_nested_markets": False})
        for event in data.get("events") or []:
            ticker = event.get("event_ticker")
            if ticker:
                found[ticker] = event
    except Exception:
        found = {}
    missing = [t for t in wanted if t not in found][:8]
    for ticker in missing:
        try:
            data = _get(settings, f"/events/{ticker}")
            event = data.get("event") or data
            if event.get("event_ticker"):
                found[event["event_ticker"]] = event
        except Exception:
            continue
    return found


def fetch_markets(settings: Settings) -> list[NormalizedMarket]:
    markets: list[dict[str, Any]] = []
    cursor = None
    pages = 0
    while pages < 6:
        params = {
            "limit": 200,
            "status": "open",
            "cursor": cursor,
        }
        if settings.kalshi.extra.get("exclude_mve", True):
            params["mve_filter"] = "exclude"
        data = _get(settings, "/markets", params)
        batch = data.get("markets") or []
        markets.extend(batch)
        cursor = data.get("cursor")
        pages += 1
        if not cursor or not batch:
            break

    def volume_key(raw: dict[str, Any]) -> float:
        return to_float(raw.get("volume_24h_fp")) or to_float(raw.get("volume_fp")) or 0.0

    markets.sort(key=volume_key, reverse=True)
    min_vol = settings.kalshi.min_volume_24h
    filtered = [m for m in markets if volume_key(m) >= min_vol][: settings.kalshi.max_markets]
    if not filtered:
        filtered = markets[: settings.kalshi.max_markets]

    events = fetch_events(settings, [m.get("event_ticker") for m in filtered])
    out: list[NormalizedMarket] = []
    for raw in filtered:
        ticker = raw.get("ticker")
        if not ticker:
            continue
        event = events.get(raw.get("event_ticker") or "", {})
        last = clip01(to_float(raw.get("last_price_dollars")))
        bid = clip01(to_float(raw.get("yes_bid_dollars")))
        ask = clip01(to_float(raw.get("yes_ask_dollars")))
        mid = clip01((bid + ask) / 2) if bid is not None and ask is not None else last
        volume_24h = to_float(raw.get("volume_24h_fp"))
        volume = to_float(raw.get("volume_fp"))
        title = event.get("title") or raw.get("title") or raw.get("yes_sub_title") or ticker
        question = raw.get("yes_sub_title") or event.get("sub_title") or title
        close_iso, _ = parse_time(raw.get("close_time"))
        extra = {
            "event_ticker": raw.get("event_ticker"),
            "series_ticker": event.get("series_ticker"),
            "rules_primary": raw.get("rules_primary"),
        }
        out.append(
            NormalizedMarket(
                market_id=f"kalshi:{ticker}",
                venue="kalshi",
                venue_market_id=ticker,
                event_id=raw.get("event_ticker"),
                ticker=ticker,
                title=title,
                question=question,
                category=event.get("category"),
                status=raw.get("status") or "open",
                close_time=close_iso,
                last_price=last,
                best_bid=bid,
                best_ask=ask,
                mid_price=mid,
                volume=volume,
                volume_24h=volume_24h,
                open_interest=to_float(raw.get("open_interest_fp")),
                extra_json=json.dumps(extra),
            )
        )
    return out


def fetch_trades(settings: Settings, market: NormalizedMarket, limit: int = 200) -> list[NormalizedTrade]:
    data = _get(settings, "/markets/trades", {"ticker": market.venue_market_id, "limit": min(limit, 1000)})
    trades: list[NormalizedTrade] = []
    for raw in data.get("trades") or []:
        trade_id = raw.get("trade_id")
        price = clip01(to_float(raw.get("yes_price_dollars")))
        size = to_float(raw.get("count_fp")) or 0.0
        traded_at, unix = parse_time(raw.get("created_time"))
        if not trade_id or price is None or not traded_at:
            continue
        trades.append(
            NormalizedTrade(
                trade_id=f"kalshi:{trade_id}",
                market_id=market.market_id,
                venue="kalshi",
                traded_at=traded_at,
                traded_at_unix=unix,
                price=price,
                size=size,
                notional=price * size,
                side=raw.get("taker_outcome_side") or raw.get("taker_side"),
                outcome=raw.get("taker_outcome_side") or raw.get("taker_side"),
                is_block=1 if raw.get("is_block_trade") else 0,
                extra_json=json.dumps(
                    {
                        "taker_book_side": raw.get("taker_book_side"),
                        "no_price_dollars": raw.get("no_price_dollars"),
                    }
                ),
            )
        )
    return trades


def fetch_orderbook(settings: Settings, market: NormalizedMarket):
    data = _get(settings, f"/markets/{market.venue_market_id}/orderbook", {"depth": 25})
    book = data.get("orderbook_fp") or data.get("orderbook") or {}
    bids, asks = kalshi_yes_book(book)
    return book_from_levels(market.market_id, "kalshi", bids, asks, utc_now())
