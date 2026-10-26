"""Tests for src/backtest.py, on synthetic prices."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest


def make_prices(n=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    return pd.DataFrame({"Close": close}, index=dates)


def probs(prices, value, start=10):
    """Constant P(UP) for a 'test period' starting at row `start`."""
    return pd.Series(value, index=prices.index[start:])


def test_output_shape():
    prices = make_prices()
    result = run_backtest(prices, probs(prices, 0.7))

    assert {"strategy_value", "buy_hold_value", "invested"} <= set(result.columns)
    assert len(result) == 50                        # only the test period
    assert result.iloc[0]["strategy_value"] == 100
    assert result.iloc[0]["buy_hold_value"] == 100


def test_always_up_equals_buy_and_hold():
    prices = make_prices()
    result = run_backtest(prices, probs(prices, 1.0))
    assert result["strategy_value"].tolist() == pytest.approx(result["buy_hold_value"].tolist())


def test_always_down_stays_in_cash():
    prices = make_prices()
    result = run_backtest(prices, probs(prices, 0.0))
    assert (result["strategy_value"] == 100).all()
    assert (result["invested"] == 0).all()


def test_buy_and_hold_tracks_the_price():
    prices = make_prices()
    result = run_backtest(prices, probs(prices, 0.0))
    expected = 100 * prices["Close"].iloc[-1] / prices["Close"].iloc[10]
    assert result["buy_hold_value"].iloc[-1] == pytest.approx(expected)


def test_last_day_signal_has_no_effect():
    """Shift test: a decision on the final day would only affect the next day."""
    prices = make_prices()
    p = probs(prices, 0.0)
    flipped = p.copy()
    flipped.iloc[-1] = 1.0

    a = run_backtest(prices, p)
    b = run_backtest(prices, flipped)
    pd.testing.assert_frame_equal(a, b)


def test_signal_earns_the_next_days_return():
    """Invest only on day k: the strategy should earn exactly day k+1's return."""
    prices = make_prices()
    p = probs(prices, 0.0)
    k = 5
    p.iloc[k] = 1.0

    result = run_backtest(prices, p)
    close = prices["Close"].loc[p.index]
    next_day_return = close.iloc[k + 1] / close.iloc[k] - 1

    assert result["strategy_value"].iloc[k] == pytest.approx(100)        # not yet
    assert result["strategy_value"].iloc[-1] == pytest.approx(100 * (1 + next_day_return))
    assert result["invested"].sum() == 1
