from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "ret_1",
    "ret_5",
    "ret_20",
    "vol_20",
    "range_pct",
    "volume_z",
    "ma_gap_10",
    "ma_gap_30",
    "rsi_14",
    "atr_pct",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return num / den


def build_features(ohlcv: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """Causal technical features. Target is 1 iff close-to-close return over `horizon` is positive."""
    df = ohlcv.copy().sort_index()
    close, high, low = df["Close"], df["High"], df["Low"]
    log_close = np.log(close.replace(0, np.nan))

    df["ret_1"] = log_close.diff()
    df["ret_5"] = log_close.diff(5)
    df["ret_20"] = log_close.diff(20)
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df["range_pct"] = _safe_div(high - low, close)

    vol_mean = df["Volume"].rolling(30).mean()
    vol_std = df["Volume"].rolling(30).std().replace(0, np.nan)
    df["volume_z"] = (df["Volume"] - vol_mean) / vol_std
    df["ma_gap_10"] = _safe_div(close, close.rolling(10).mean()) - 1
    df["ma_gap_30"] = _safe_div(close, close.rolling(30).mean()) - 1

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    prev = close.shift()
    true_range = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    df["atr_pct"] = _safe_div(true_range.rolling(14).mean(), close)

    hour = df.index.hour if hasattr(df.index, "hour") else pd.Index([0] * len(df))
    dow = df.index.dayofweek if hasattr(df.index, "dayofweek") else pd.Index([0] * len(df))
    df["hour_sin"] = np.sin(2 * np.pi * np.asarray(hour) / 24)
    df["hour_cos"] = np.cos(2 * np.pi * np.asarray(hour) / 24)
    df["dow_sin"] = np.sin(2 * np.pi * np.asarray(dow) / 7)
    df["dow_cos"] = np.cos(2 * np.pi * np.asarray(dow) / 7)

    df["future_return"] = close.shift(-horizon) / close - 1
    df["target"] = (df["future_return"] > 0).astype(int)
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=FEATURES + ["target"]).copy()


def make_sequences(df: pd.DataFrame, features: list[str] | None = None, lookback: int = 64):
    features = features or FEATURES
    values = df[features].to_numpy(dtype=np.float32)
    target = df["target"].to_numpy(dtype=np.float32)
    if len(df) < lookback:
        empty_x = np.empty((0, lookback, len(features)), dtype=np.float32)
        return empty_x, target[:0], df.index[:0]
    windows = np.lib.stride_tricks.sliding_window_view(values, (lookback, values.shape[1]))
    sequences = np.asarray(windows[:, 0], dtype=np.float32)
    return sequences, target[lookback - 1 :], df.index[lookback - 1 :]


def trades_to_ohlcv(trades: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """Build OHLCV candles from a stream of yes-price trades."""
    if trades.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    frame = trades.copy()
    if "traded_at" in frame.columns:
        frame["traded_at"] = pd.to_datetime(frame["traded_at"], utc=True)
        frame = frame.set_index("traded_at")
    price = pd.to_numeric(frame["price"], errors="coerce")
    size = pd.to_numeric(frame.get("size", 1.0), errors="coerce").fillna(0.0)
    ohlc = price.resample(freq).ohlc()
    volume = size.resample(freq).sum()
    candles = ohlc.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    candles["Volume"] = volume
    candles = candles.dropna(subset=["Close"])
    candles[["Open", "High", "Low"]] = candles[["Open", "High", "Low"]].fillna(candles["Close"])
    return candles
