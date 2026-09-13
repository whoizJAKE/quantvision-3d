import pandas as pd

from quantvision.etl.db import connect, fetchall, upsert_many
from quantvision.etl.normalize import NormalizedTrade, utc_now
from quantvision.features import FEATURES, build_features, make_sequences, trades_to_ohlcv


def test_sqlite_trade_upsert(tmp_path):
    conn = connect(tmp_path / "qv.db")
    trade = NormalizedTrade(
        trade_id="kalshi:1",
        market_id="kalshi:TEST",
        venue="kalshi",
        traded_at=utc_now(),
        traded_at_unix=1,
        price=0.55,
        size=10,
    )
    upsert_many(conn, "trades", [trade.as_row(), trade.as_row()], "trade_id")
    conn.commit()
    rows = fetchall(conn, "SELECT * FROM trades")
    assert len(rows) == 1
    assert rows[0]["price"] == 0.55
    conn.close()


def test_feature_pipeline_is_finite():
    idx = pd.date_range("2024-01-01", periods=80, freq="h")
    close = pd.Series(range(80), index=idx).astype(float) + 100
    raw = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1000,
        }
    )
    frame = build_features(raw, horizon=3)
    assert set(FEATURES).issubset(frame.columns)
    assert frame[FEATURES].isna().sum().sum() == 0
    seq, y, index = make_sequences(frame, lookback=16)
    assert seq.shape[1:] == (16, len(FEATURES))
    assert len(y) == len(index) == len(seq)


def test_trades_to_ohlcv():
    trades = pd.DataFrame(
        {
            "traded_at": pd.to_datetime(
                ["2026-09-12T00:00:01Z", "2026-09-12T00:00:30Z", "2026-09-12T00:01:10Z"]
            ),
            "price": [0.40, 0.42, 0.41],
            "size": [2, 3, 1],
        }
    )
    candles = trades_to_ohlcv(trades, "1min")
    assert list(candles.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(candles) == 2
    assert candles.iloc[0]["High"] == 0.42
