from quantvision.etl.normalize import book_from_levels, kalshi_yes_book, parse_jsonish, parse_time, to_float


def test_kalshi_yes_book_converts_no_bids_to_yes_asks():
    bids, asks = kalshi_yes_book(
        {
            "yes_dollars": [["0.40", "10"], ["0.39", "4"]],
            "no_dollars": [["0.55", "8"]],
        }
    )
    assert bids[0] == (0.40, 10.0)
    assert round(asks[0][0], 2) == 0.45
    assert asks[0][1] == 8.0


def test_book_microprice_and_imbalance():
    book = book_from_levels(
        "kalshi:TEST",
        "kalshi",
        bids=[(0.48, 20), (0.47, 5)],
        asks=[(0.52, 10), (0.53, 9)],
    )
    assert book.best_bid == 0.48
    assert book.best_ask == 0.52
    assert round(book.mid, 4) == 0.50
    assert book.imbalance > 0
    assert 0.48 < book.microprice < 0.52


def test_parsers():
    assert to_float("0.5600") == 0.56
    assert to_float("nope") is None
    iso, unix = parse_time("2026-09-12T12:00:00Z")
    assert iso.startswith("2026-09-12")
    assert unix > 0
    assert parse_jsonish('["a","b"]') == ["a", "b"]
