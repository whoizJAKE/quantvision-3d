import numpy as np
import pandas as pd

def long_flat_backtest(frame: pd.DataFrame, probability: pd.Series, threshold=0.55, cost_bps=10):
    df = frame.loc[probability.index, ['Close']].copy()
    df['probability'] = probability
    df['position'] = (df['probability'] >= threshold).astype(float)
    df['asset_return'] = df['Close'].pct_change().fillna(0)
    df['turnover'] = df['position'].diff().abs().fillna(df['position'])
    df['strategy_return'] = df['position'].shift(1).fillna(0) * df['asset_return'] - df['turnover'] * cost_bps / 10000
    df['equity'] = (1 + df['strategy_return']).cumprod()
    df['drawdown'] = df['equity'] / df['equity'].cummax() - 1
    sharpe = np.sqrt(252) * df['strategy_return'].mean() / (df['strategy_return'].std() + 1e-12)
    stats = {'Total return': df['equity'].iloc[-1]-1, 'Max drawdown': df['drawdown'].min(), 'Sharpe (period-scaled)': sharpe, 'Trades': int((df['turnover'] > 0).sum()), 'Exposure': df['position'].mean()}
    return df, stats
