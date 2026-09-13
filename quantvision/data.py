from __future__ import annotations

import pandas as pd
import yfinance as yf


def load_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Download OHLCV bars and return a flat Open/High/Low/Close/Volume frame."""
    frame = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if frame is None or frame.empty:
        raise ValueError(
            "No data returned. Try a supported ticker, a shorter intraday period, or another interval."
        )
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.rename(columns=str.title)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    available = [c for c in needed if c in frame.columns]
    frame = frame[available].copy()
    for col in available:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "Volume" not in frame.columns:
        frame["Volume"] = 0.0
    frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
    frame["Volume"] = frame["Volume"].fillna(0.0)
    if frame.empty:
        raise ValueError("OHLCV frame is empty after cleaning.")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame
