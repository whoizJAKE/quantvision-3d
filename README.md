# QuantVision 3D

Local quantitative-research dashboard **for research and paper trading only**. It compares three direction models on market candles, then mirrors live Kalshi and Polymarket books into SQLite so you can watch model probabilities against market consensus.

1. Logistic Regression baseline
2. HistGradientBoosting (tree ensemble)
3. PyTorch Transformer with sinusoidal positional encoding

## Features

- Downloads OHLCV with `yfinance` and builds causally available technical features
- Chronological train / validation / test splits, artifact saving, interval-aware Sharpe
- Automated ETL: Kalshi + Polymarket trades and order books → normalized schema → local SQLite
- Monitor that alerts when app probability diverges from market consensus (5% on major events)
- Historical `performance_log` for backtesting those gaps
- Live web UI (FastAPI) and a standalone `quantvisionUI/` page you can later publish at `https://jacobhmaiddout.com/quantvisionUI`

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Use the sidebar to download data, train models, or open the **Live markets** workspace.

```bash
python train.py --ticker BTC-USD --period 730d --interval 1h --epochs 20
```

## Live prediction-market pipeline

```bash
# one-shot ingest into data/quantvision.db
python scripts/run_etl.py

# compare model / flow-adjusted probabilities to market mids
python scripts/run_monitor.py

# loop ETL + monitor and serve the dashboard
python scripts/run_live.py
```

Then open [http://127.0.0.1:8787](http://127.0.0.1:8787).

| Script | Role |
| --- | --- |
| `scripts/run_etl.py` | Pulls open markets, public trades, and order books from Kalshi and Polymarket, normalizes them, upserts SQLite, writes `quantvisionUI/snapshot.json` |
| `scripts/run_monitor.py` | Loads QuantVision ensemble artifacts when a market maps to `BTC-USD` / `SPY` / etc., otherwise a trade-flow adjustment. Alerts if \|app − market\| exceeds the threshold |
| `scripts/run_live.py` | Background poll + FastAPI UI |

Public Kalshi market/trade/orderbook endpoints and Polymarket Gamma / CLOB / Data APIs do not require keys for this read path. Optional `QUANTVISION_WEBHOOK_URL` posts alerts. On macOS, alerts also fire a Notification Center banner.

### Schema (SQLite)

Normalized yes-probability in `[0, 1]` for both venues.

- `markets` — venue, title, mid, volume, major flag
- `trades` — tape (`trade_id` primary key, idempotent upserts)
- `orderbook_snapshots` — bids/asks, microprice, imbalance
- `predictions` — app probability vs market consensus
- `alerts` — threshold breaches
- `performance_log` — every check, including non-alerts, for backtests
- `etl_runs` — ingest health

Thresholds live in `config.yaml`: **5%** for major events (`volume_24h` ≥ `major_min_volume`), **10%** otherwise.

## Live UI document (`quantvisionUI/`)

The dashboard lives in this repo only — it is **not** on the homepage.

- Local: run `python scripts/run_live.py` and open [http://127.0.0.1:8787](http://127.0.0.1:8787)
- Later, copy the folder `quantvisionUI/` onto the site as `/quantvisionUI` so it is served at `https://jacobhmaiddout.com/quantvisionUI`

See `quantvisionUI/README.md` for the publish steps. The page can stream public Polymarket data in the browser. Kalshi blocks browser CORS, so paste `http://127.0.0.1:8787` into **Use API** when the local server is running.

## Risk and methodology

This is not financial advice and does not place orders. Historical accuracy or backtest performance does not establish future profitability. A model is evaluated only after its training period; the backtest deducts a configurable cost whenever position exposure changes.

`target=1` means the close-to-close return at the selected horizon was positive. Features use only the current and past bars. The final test partition is held out of model fitting.
