from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantvision.backtest import long_flat_backtest
from quantvision.config import load_settings
from quantvision.data import load_ohlcv
from quantvision.etl.db import fetchall, session
from quantvision.features import build_features
from train import run_training

st.set_page_config(page_title="QuantVision 3D", page_icon="◈", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background:#07111f; color:#d7e6ef; }
      .stButton>button { background:#12b8a6; color:#06111f; border-radius:8px; border:0; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio("Workspace", ["Research", "Live markets"])
st.title("◈ QuantVision 3D")
st.caption("Local quantitative research • Kalshi/Polymarket live tape • Paper trading only")


def _fmt_pct(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{value:.1%}"


if page == "Research":
    with st.sidebar:
        st.header("Research controls")
        ticker = st.text_input("Ticker", "BTC-USD").upper()
        period = st.selectbox("History", ["180d", "365d", "730d", "max"], index=2)
        interval = st.selectbox("Interval", ["1h", "1d", "15m", "5m"])
        horizon = st.slider("Prediction horizon (bars)", 1, 48, 12)
        threshold = st.slider("Long threshold", 0.50, 0.80, 0.55, 0.01)
        cost = st.slider("Cost per exposure change (bps)", 0, 100, 10)
        epochs = st.slider("Transformer epochs", 5, 40, 20)
        train = st.button("Train all 3 models", use_container_width=True)

    try:
        raw = load_ohlcv(ticker, period, interval)
        data = build_features(raw, horizon)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if train:
        with st.spinner("Training three models on a chronological split..."):
            st.session_state.result = run_training(
                ticker, period, interval, horizon, epochs=epochs
            )

    if "result" not in st.session_state:
        st.info("Download complete. Click Train all 3 models to fit Logistic, Boosting, and Transformer.")
        st.line_chart(raw["Close"])
        st.stop()

    result = st.session_state.result
    preds = result["predictions"]
    latest = preds.iloc[-1]
    cols = st.columns(4)
    for col, label, value in zip(
        cols,
        ["Logistic", "Boosting", "Transformer", "Ensemble"],
        [latest.logistic, latest.boosting, latest.transformer, latest.ensemble],
    ):
        col.metric(label, f"{value:.1%}", "bullish probability")

    metric_cols = st.columns(4)
    for col, name in zip(metric_cols, ["logistic", "boosting", "transformer", "ensemble"]):
        stats = result["metrics"][name]
        col.write(f"**{name}**")
        col.write(f"acc {_fmt_pct(stats['accuracy'])} · auc {stats['auc']:.3f} · brier {stats['brier']:.3f}")

    st.subheader(f"{ticker} • held-out test predictions")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close", line=dict(color="#60a5fa")))
    fig.add_trace(
        go.Scatter(
            x=preds.index,
            y=preds["ensemble"],
            name="Ensemble P(up)",
            yaxis="y2",
            line=dict(color="#12b8a6"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=460,
        paper_bgcolor="#07111f",
        plot_bgcolor="#07111f",
        yaxis2=dict(overlaying="y", side="right", range=[0, 1], title="probability"),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

    backtest, stats = long_flat_backtest(data, preds["ensemble"], threshold, cost)
    for col, (name, value) in zip(st.columns(5), stats.items()):
        col.metric(name, str(value) if name == "Trades" else f"{value:.2%}")
    st.subheader("Equity curve after estimated costs")
    st.line_chart(backtest["equity"])
    st.subheader("Model predictions")
    st.dataframe(preds.tail(250).sort_index(ascending=False), use_container_width=True)
    st.caption(f"Artifacts saved to {result.get('artifact_dir', 'artifacts/')}")
    st.warning("Research and paper trading only. Do not treat results as a promise or instruction to trade.")

else:
    settings = load_settings()
    st.sidebar.write(f"SQLite: `{settings.db_path}`")
    if not Path(settings.db_path).exists():
        st.info("No live database yet. Run `python scripts/run_etl.py` then `python scripts/run_monitor.py`.")
        st.code("python scripts/run_live.py")
        st.stop()
    with session(settings.db_path) as conn:
        markets = fetchall(
            conn,
            "SELECT * FROM markets ORDER BY COALESCE(volume_24h, volume, 0) DESC LIMIT 80",
        )
        trades = fetchall(conn, "SELECT * FROM trades ORDER BY traded_at DESC LIMIT 80")
        alerts = fetchall(conn, "SELECT * FROM alerts ORDER BY created_at DESC LIMIT 30")
        perf = fetchall(conn, "SELECT * FROM performance_log ORDER BY created_at DESC LIMIT 80")
    if not markets:
        st.info("Database is empty. Run the ETL pipeline to pull Kalshi and Polymarket data.")
        st.stop()
    kpis = st.columns(4)
    kpis[0].metric("Markets", len(markets))
    kpis[1].metric("Recent trades", len(trades))
    kpis[2].metric("Open alerts", len(alerts))
    mean_dev = pd.DataFrame(perf)["abs_deviation"].mean() if perf else None
    kpis[3].metric("Mean |Δ|", _fmt_pct(mean_dev))
    st.subheader("Normalized live markets")
    table = pd.DataFrame(markets)[
        [
            "venue",
            "title",
            "mid_price",
            "last_price",
            "volume_24h",
            "is_major",
            "status",
            "market_id",
        ]
    ]
    st.dataframe(table, use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    with left:
        st.subheader("Trade tape")
        st.dataframe(pd.DataFrame(trades), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Deviation alerts")
        st.dataframe(pd.DataFrame(alerts), use_container_width=True, hide_index=True)
    st.subheader("Historical model vs market")
    st.dataframe(pd.DataFrame(perf), use_container_width=True, hide_index=True)
    st.caption("Live HTML dashboard: run `python scripts/run_live.py` and open http://127.0.0.1:8787")
