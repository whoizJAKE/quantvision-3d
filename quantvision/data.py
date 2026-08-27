import pandas as pd
import yfinance as yf

def load_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    frame = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if frame.empty:
        raise ValueError('No data returned. Try a supported ticker, shorter intraday period, or another interval.')
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.rename(columns=str.title)
    needed = ['Open', 'High', 'Low', 'Close', 'Volume']
    frame = frame[[c for c in needed if c in frame.columns]].dropna()
    if 'Volume' not in frame:
        frame['Volume'] = 0.0
    return frame
