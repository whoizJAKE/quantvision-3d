from __future__ import annotations

import numpy as np
import pandas as pd


def _periods_per_year(index: pd.Index) -> float:
    if len(index) < 3:
        return 252.0
    delta = pd.Series(index).diff().median()
    if not isinstance(delta, pd.Timedelta) or delta <= pd.Timedelta(0):
        return 252.0
    return float(pd.Timedelta(days=365) / delta)


def long_flat_backtest(
    frame: pd.DataFrame,
    probability: pd.Series,
    threshold: float = 0.55,
    cost_bps: float = 10,
):
    aligned = frame.loc[probability.index, ["Close"]].copy()
    aligned["probability"] = probability.astype(float)
    aligned["position"] = (aligned["probability"] >= threshold).astype(float)
    aligned["asset_return"] = aligned["Close"].pct_change().fillna(0.0)
    aligned["turnover"] = aligned["position"].diff().abs()
    aligned.loc[aligned.index[0], "turnover"] = aligned["position"].iloc[0]
    aligned["strategy_return"] = (
        aligned["position"].shift(1).fillna(0.0) * aligned["asset_return"]
        - aligned["turnover"] * cost_bps / 10_000
    )
    aligned["equity"] = (1 + aligned["strategy_return"]).cumprod()
    aligned["drawdown"] = aligned["equity"] / aligned["equity"].cummax() - 1
    scale = np.sqrt(_periods_per_year(aligned.index))
    denom = float(aligned["strategy_return"].std() or 0.0) + 1e-12
    sharpe = float(scale * aligned["strategy_return"].mean() / denom)
    stats = {
        "Total return": float(aligned["equity"].iloc[-1] - 1) if len(aligned) else 0.0,
        "Max drawdown": float(aligned["drawdown"].min()) if len(aligned) else 0.0,
        "Sharpe (ann.)": sharpe,
        "Trades": int((aligned["turnover"] > 0).sum()),
        "Exposure": float(aligned["position"].mean()) if len(aligned) else 0.0,
    }
    return aligned, stats
