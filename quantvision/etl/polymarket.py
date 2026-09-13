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
    parse_jsonish,
    parse_time,
    to_float,
    utc_now,
)


def _levels(raw_levels: Any) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for level in raw_levels or []:
        if isinstance(level, dict):
            price, size = to_float(level.get("price")), to_float(level.get("size"))
        elif isinstance(level, (list, tuple)) and len(level) >= 2:
            price, size = to_float(level[0]), to_float(level[1])
        else:
            continue
        if price is not None and size is not None:
            out.append((price, size))
    return out


def fetch_markets(settings: Settings) -> list[NormalizedMarket]:
    data = get_json(
        f"{settings.poly_gamma}/markets",
        params={
            "closed": "false",
            "active": "true",
            "limit": max(settings.polymarket.max_markets * 2, 50),
            "order": "volume24hr",
            "ascending": "false",
        },
        timeout=settings.request_timeout,
    )
    if isinstance(data, dict):
        raw_markets = data.get("markets") or data.get("data") or []
    else:
        raw_markets = data or []

    def vol(raw: dict[str, Any]) -> float:
        return to_float(raw.get("volume24hr")) or to_float(raw.get("volumeNum")) or to_float(raw.get("volume")) or 0.0

    qualified = [m for m in raw_markets if vol(m) >= settings.polymarket.min_volume_24h]
    selected = (qualified or raw_markets)[: settings.polymarket.max_markets]
    out: list[NormalizedMarket] = []
    for raw in selected:
        condition_id = raw.get("conditionId") or raw.get("condition_id")
        if not condition_id:
            continue
        tokens = parse_jsonish(raw.get("clobTokenIds")) or []
        if isinstance(tokens, str):
            tokens = parse_jsonish(tokens) or []
        yes_token = str(tokens[0]) if len(tokens) > 0 else None
        no_token = str(tokens[1]) if len(tokens) > 1 else None
        prices = parse_jsonish(raw.get("outcomePrices")) or []
        last = clip01(to_float(prices[0])) if prices else None
        bid = clip01(to_float(raw.get("bestBid")))
        ask = clip01(to_float(raw.get("bestAsk")))
        mid = clip01((bid + ask) / 2) if bid is not None and ask is not None else last
        close_iso, _ = parse_time(raw.get("endDate") or raw.get("endDateIso"))
        extra = {
            "slug": raw.get("slug"),
            "outcomes": parse_jsonish(raw.get("outcomes")),
            "event_slug": raw.get("eventSlug"),
        }
        volume_24h = to_float(raw.get("volume24hr"))
        volume = to_float(raw.get("volumeNum")) or to_float(raw.get("volume"))
        out.append(
            NormalizedMarket(
                market_id=f"polymarket:{condition_id}",
                venue="polymarket",
                venue_market_id=str(condition_id),
                event_id=str(raw.get("questionID") or raw.get("id") or ""),
                ticker=raw.get("slug"),
                title=raw.get("question") or raw.get("groupItemTitle") or raw.get("slug"),
                question=raw.get("question"),
                category=raw.get("groupItemTitle"),
                status="open" if raw.get("active") and not raw.get("closed") else "closed",
                yes_token_id=yes_token,
                no_token_id=no_token,
                close_time=close_iso,
                last_price=last,
                best_bid=bid,
                best_ask=ask,
                mid_price=mid,
                volume=volume,
                volume_24h=volume_24h,
                liquidity=to_float(raw.get("liquidityNum")) or to_float(raw.get("liquidity")),
                extra_json=json.dumps(extra),
            )
        )
    return out


def _yes_price(raw: dict[str, Any], market: NormalizedMarket) -> float | None:
    price = to_float(raw.get("price"))
    if price is None:
        return None
    asset = str(raw.get("asset") or raw.get("asset_id") or "")
    outcome = str(raw.get("outcome") or "").strip().lower()
    if market.no_token_id and asset == str(market.no_token_id):
        return clip01(1.0 - price)
    if outcome in {"no", "down", "false"}:
        return clip01(1.0 - price)
    return clip01(price)


def fetch_trades(settings: Settings, market: NormalizedMarket, limit: int = 100) -> list[NormalizedTrade]:
    data = get_json(
        f"{settings.poly_data}/trades",
        params={
            "market": market.venue_market_id,
            "limit": min(limit, 1000),
            "takerOnly": "true",
        },
        timeout=settings.request_timeout,
    )
    raw_trades = data if isinstance(data, list) else (data.get("data") or data.get("trades") or [])
    trades: list[NormalizedTrade] = []
    for raw in raw_trades:
        price = _yes_price(raw, market)
        size = to_float(raw.get("size")) or 0.0
        traded_at, unix = parse_time(raw.get("timestamp") or raw.get("match_time"))
        trade_key = raw.get("transactionHash") or raw.get("id") or raw.get("transaction_hash")
        if price is None or not traded_at or not trade_key:
            continue
        trade_id = f"polymarket:{trade_key}:{raw.get('asset', '')}:{unix}"
        trades.append(
            NormalizedTrade(
                trade_id=trade_id[:180],
                market_id=market.market_id,
                venue="polymarket",
                traded_at=traded_at,
                traded_at_unix=unix,
                price=price,
                size=size,
                notional=(price * size) if price is not None else None,
                side=raw.get("side"),
                outcome=raw.get("outcome"),
                extra_json=json.dumps(
                    {
                        "proxyWallet": raw.get("proxyWallet"),
                        "slug": raw.get("slug"),
                        "asset": raw.get("asset"),
                    }
                ),
            )
        )
    return trades


def fetch_orderbook(settings: Settings, market: NormalizedMarket):
    if not market.yes_token_id:
        return book_from_levels(market.market_id, "polymarket", [], [], utc_now())
    data = get_json(
        f"{settings.poly_clob}/book",
        params={"token_id": market.yes_token_id},
        timeout=settings.request_timeout,
    )
    bids = _levels(data.get("bids"))
    asks = _levels(data.get("asks"))
    captured, _ = parse_time(data.get("timestamp"))
    return book_from_levels(market.market_id, "polymarket", bids, asks, captured or utc_now())
