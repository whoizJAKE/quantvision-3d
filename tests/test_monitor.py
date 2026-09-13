from quantvision.config import Mapping, Settings
from quantvision.monitor.alerts import threshold_for
from quantvision.monitor.forecast import market_consensus, match_ticker, performance_row


def test_consensus_uses_mid_not_microprice():
    market = {"mid_price": 0.40, "last_price": 0.41}
    book = {"microprice": 0.47, "mid": 0.42, "best_bid": 0.41, "best_ask": 0.43}
    assert market_consensus(market, book) == 0.42
    assert market_consensus(market, None) == 0.40


def test_keyword_mapping():
    mappings = [Mapping("BTC-USD", ["bitcoin", "btc"])]
    market = {"title": "Bitcoin Up or Down this hour", "question": "", "ticker": "btc-updown", "category": ""}
    assert match_ticker(market, mappings) == "BTC-USD"
    assert match_ticker({"title": "Will it rain in Paris?", "question": "", "ticker": "", "category": ""}, mappings) is None


def test_major_threshold():
    settings = Settings(major_threshold=0.05, default_threshold=0.10, major_min_volume=100000)
    assert threshold_for({"is_major": 1}, settings) == 0.05
    assert threshold_for({"is_major": 0, "volume_24h": 50}, settings) == 0.10
    assert threshold_for({"is_major": 0, "volume_24h": 500000}, settings) == 0.05


def test_performance_row_unsettled():
    row = performance_row(
        {"market_id": "polymarket:abc", "is_major": 1, "status": "open"},
        "ensemble:BTC-USD",
        0.62,
        0.55,
        "2026-09-12T00:00:00Z",
    )
    assert round(row["deviation"], 2) == 0.07
    assert round(row["abs_deviation"], 2) == 0.07
    assert row["settled_result"] is None
    assert row["brier"] is None
