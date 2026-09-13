from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

from quantvision.config import Settings, load_settings
from quantvision.etl.db import fetchall, insert_many, latest_books, session
from quantvision.etl.normalize import utc_now
from quantvision.monitor.forecast import app_probability, market_consensus, performance_row


def threshold_for(market: dict[str, Any], settings: Settings) -> float:
    if int(market.get("is_major") or 0):
        return settings.major_threshold
    volume = float(market.get("volume_24h") or market.get("volume") or 0.0)
    if volume >= settings.major_min_volume:
        return settings.major_threshold
    return settings.default_threshold


def _notify_macos(title: str, message: str) -> None:
    script = f'display notification {json.dumps(message)} with title {json.dumps(title)}'
    try:
        subprocess.run(["osascript", "-e", script], check=False, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        return


def _notify_webhook(url: str, payload: dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "QuantVision-3D/0.2"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:
        return


def emit_alert(settings: Settings, alert: dict[str, Any]) -> None:
    line = (
        f"{alert['created_at']} | {alert['severity']} | {alert['title']} | "
        f"model={alert['predicted_prob']:.3f} market={alert['market_prob']:.3f} "
        f"dev={alert['deviation']:+.3f} thr={alert['threshold']:.3f}"
    )
    path: Path = settings.alert_log
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line)
    _notify_macos("QuantVision alert", alert["message"])
    if settings.webhook_url:
        _notify_webhook(settings.webhook_url, alert)


def run_monitor(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    created = utc_now()
    alerts: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    perf: list[dict[str, Any]] = []

    with session(settings.db_path) as conn:
        markets = fetchall(conn, "SELECT * FROM markets")
        books = latest_books(conn)
        recent_alerts = {
            (row["market_id"], round(float(row["deviation"]), 3))
            for row in fetchall(conn, "SELECT market_id, deviation FROM alerts ORDER BY created_at DESC LIMIT 200")
        }
        for market in markets:
            book = books.get(market["market_id"])
            predicted, model_name = app_probability(conn, market, book, settings)
            consensus = market_consensus(market, book)
            if predicted is None or consensus is None:
                continue
            deviation = predicted - consensus
            predictions.append(
                {
                    "market_id": market["market_id"],
                    "model_name": f"auto:{model_name}",
                    "predicted_prob": predicted,
                    "market_prob": consensus,
                    "deviation": deviation,
                    "created_at": created,
                }
            )
            perf.append(performance_row(market, model_name, predicted, consensus, created))
            limit = threshold_for(market, settings)
            if abs(deviation) < limit:
                continue
            trained = model_name.startswith("ensemble:") or model_name not in {"microprice", "mid", "flow_adjusted", "none"}
            two_sided = bool(book and book.get("best_bid") is not None and book.get("best_ask") is not None)
            spread = abs(float(book["best_ask"]) - float(book["best_bid"])) if two_sided else 1.0
            if not trained:
                if not two_sided or spread > 0.15 or abs(deviation) < max(limit, 0.05):
                    continue
            fingerprint = (market["market_id"], round(deviation, 3))
            if fingerprint in recent_alerts:
                continue
            severity = "major" if int(market.get("is_major") or 0) or abs(deviation) >= 0.10 else "watch"
            title = market.get("title") or market.get("question") or market["market_id"]
            message = (
                f"{title}: app {predicted:.1%} vs market {consensus:.1%} "
                f"({deviation:+.1%}, threshold {limit:.0%})"
            )
            alert = {
                "market_id": market["market_id"],
                "title": title,
                "predicted_prob": predicted,
                "market_prob": consensus,
                "deviation": deviation,
                "threshold": limit,
                "severity": severity,
                "message": message,
                "created_at": created,
            }
            alerts.append(alert)
            emit_alert(settings, alert)

        insert_many(conn, "predictions", predictions)
        insert_many(conn, "performance_log", perf)
        insert_many(conn, "alerts", alerts)

    return {
        "checked": len(predictions),
        "alerts": len(alerts),
        "logged": len(perf),
        "db": str(settings.db_path),
        "alert_log": str(settings.alert_log),
    }
