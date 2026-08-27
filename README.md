# QuantVision 3D

A local quantitative-research dashboard for **research and paper trading only**. It compares three direction models on market candles:

1. Logistic Regression baseline
2. HistGradientBoosting (tree ensemble)
3. PyTorch Transformer with sinusoidal positional encoding

## Features
- Downloads OHLCV market data with `yfinance`
- Builds causally available technical and calendar features
- Uses chronological train/validation/test splits only
- Produces probability forecasts, ensemble signals, and a transaction-cost-aware long/flat backtest
- Stores trained artifacts in `artifacts/`
- Provides a dark Streamlit dashboard

## Quick start

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Use the sidebar to download data, train models, and run a backtest. Start with `BTC-USD`, `SPY`, or `NVDA`, and use hourly or daily data. Intraday history availability is limited by the data provider.

## Training from terminal

```bash
python train.py --ticker BTC-USD --period 730d --interval 1h --epochs 20
```

## Risk and methodology

This is not financial advice and does not place orders. Historical accuracy or backtest performance does not establish future profitability. A model is evaluated only after its training period; the backtest deducts a configurable cost whenever position exposure changes. Do not deploy to a broker before extensive out-of-sample and paper-trading evaluation.

`target=1` means the close-to-close return at the selected horizon was positive. Features use only the current and past bars. The final test partition is intentionally held out of model fitting.
