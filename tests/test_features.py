"""Tests for src/features.py."""

import pandas as pd
import pytest

from src.features import add_moving_averages


def make_prices(closes):
    """A tiny synthetic price table with only the Close column filled in."""
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame({"Close": closes}, index=dates, dtype=float)


def test_ma20_known_values():
    df = add_moving_averages(make_prices(range(1, 101)))  # Close = 1..100

    # Not enough history for the first 19 days.
    assert df["MA20"].iloc[:19].isna().all()
    # Day 20 averages 1..20 -> 10.5. Day 50 averages 31..50 -> 40.5.
    assert df["MA20"].iloc[19] == pytest.approx(10.5)
    assert df["MA20"].iloc[49] == pytest.approx(40.5)


def test_ma50_known_values():
    df = add_moving_averages(make_prices(range(1, 101)))

    assert df["MA50"].iloc[:49].isna().all()
    assert df["MA50"].iloc[49] == pytest.approx(25.5)  # mean of 1..50


def test_moving_averages_do_not_change_input():
    prices = make_prices(range(1, 101))
    add_moving_averages(prices)
    assert list(prices.columns) == ["Close"]
