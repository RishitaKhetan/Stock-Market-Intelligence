"""Tests for src/data.py (and a sanity check on the config)."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import PRICE_COLUMNS, flatten_columns, load_prices, normalize_ticker


def test_config_values():
    assert config.HORIZON_DAYS == 5


def test_normalize_ticker():
    assert normalize_ticker("  tcs.ns ") == "TCS.NS"


def test_flatten_columns_multiindex():
    # Mimic what new yfinance returns: ("Price", "Ticker") headers,
    # in a different order than we want.
    dates = pd.date_range("2024-01-01", periods=3)
    columns = pd.MultiIndex.from_tuples(
        [("Close", "AAPL"), ("High", "AAPL"), ("Low", "AAPL"),
         ("Open", "AAPL"), ("Volume", "AAPL")],
        names=["Price", "Ticker"],
    )
    raw = pd.DataFrame(np.arange(15).reshape(3, 5), index=dates, columns=columns)

    flat = flatten_columns(raw)

    assert list(flat.columns) == PRICE_COLUMNS
    # The values must stay attached to the right column.
    assert list(flat["Close"]) == [0, 5, 10]
    assert list(flat["Volume"]) == [4, 9, 14]


def test_flatten_columns_ticker_first():
    # Some versions put the ticker on the first level instead.
    dates = pd.date_range("2024-01-01", periods=2)
    columns = pd.MultiIndex.from_product([["AAPL"], PRICE_COLUMNS])
    raw = pd.DataFrame(np.ones((2, 5)), index=dates, columns=columns)

    assert list(flatten_columns(raw).columns) == PRICE_COLUMNS


def test_flatten_columns_already_flat():
    dates = pd.date_range("2024-01-01", periods=2)
    raw = pd.DataFrame(np.ones((2, 6)), index=dates,
                       columns=PRICE_COLUMNS + ["Dividends"])

    assert list(flatten_columns(raw).columns) == PRICE_COLUMNS


@pytest.mark.network
def test_load_prices_real_ticker():
    df = load_prices("AAPL")
    assert len(df) > 1000
    assert list(df.columns) == PRICE_COLUMNS
    assert df.index.is_monotonic_increasing
    assert df["Close"].notna().all()


@pytest.mark.network
def test_load_prices_fake_ticker():
    with pytest.raises(ValueError):
        load_prices("NOTAREALTICKER123")


@pytest.mark.network
def test_all_dropdown_tickers_load():
    # Catches symbols in the dropdown that have been renamed or delisted.
    failed = []
    for ticker in config.POPULAR_TICKERS:
        try:
            load_prices(ticker, period="1mo")
        except ValueError:
            failed.append(ticker)
    assert failed == []
