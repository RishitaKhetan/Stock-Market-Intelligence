"""Tests for src/features.py."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    add_label,
    add_above_label,
    add_moving_averages,
    add_return_label,
    build_features,
    compute_rsi,
    make_dataset,
    make_price_dataset,
)

WARM_UP = 50  # rows needed before the longest window (MA50) is full


def make_prices(closes):
    """A tiny synthetic price table with only the Close column filled in."""
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame({"Close": closes}, index=dates, dtype=float)


def make_random_prices(n=300, seed=0):
    """A realistic-looking random walk with Close and Volume."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=dates)


# --- Moving averages ---------------------------------------------------------

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


# --- Features ----------------------------------------------------------------

def test_feature_columns_and_labels_match():
    features = build_features(make_random_prices())
    assert list(features.columns) == FEATURE_COLUMNS
    assert set(FEATURE_LABELS) == set(FEATURE_COLUMNS)


def test_no_nan_after_warm_up():
    features = build_features(make_random_prices())
    assert features.iloc[WARM_UP:].notna().all().all()


def test_no_look_ahead_leakage():
    """Changing the future must not change any feature on day t or before."""
    prices = make_random_prices()
    t = 200
    before = build_features(prices)

    changed = prices.copy()
    changed.iloc[t + 1:, changed.columns.get_loc("Close")] *= 3      # big jump
    changed.iloc[t + 1:, changed.columns.get_loc("Volume")] *= 10
    after = build_features(changed)

    pd.testing.assert_frame_equal(before.iloc[: t + 1], after.iloc[: t + 1])
    # Sanity check: the future rows really did change.
    assert not before.iloc[t + 1:].equals(after.iloc[t + 1:])


def test_rsi_extremes():
    assert compute_rsi(make_prices(range(1, 60))["Close"]).iloc[-1] == pytest.approx(100)
    assert compute_rsi(make_prices(range(60, 1, -1))["Close"]).iloc[-1] == pytest.approx(0)
    assert compute_rsi(make_prices([5.0] * 30)["Close"]).iloc[-1] == pytest.approx(50)


def test_rsi_stays_between_0_and_100():
    rsi = compute_rsi(make_random_prices()["Close"]).dropna()
    assert rsi.between(0, 100).all()


# --- Label -------------------------------------------------------------------

def test_label_rising_series():
    horizon = 5
    df = add_label(make_prices(range(1, 101)), horizon=horizon)

    assert (df["target"].iloc[:-horizon] == 1).all()
    assert df["target"].iloc[-horizon:].isna().all()


def test_label_falling_series_is_all_down():
    df = add_label(make_prices(range(100, 0, -1)), horizon=5)
    assert (df["target"].iloc[:-5] == 0).all()


# --- Dataset -----------------------------------------------------------------

def test_make_dataset_shapes_and_latest_row():
    prices = make_random_prices()
    X, y, latest_row = make_dataset(prices)

    assert list(X.columns) == FEATURE_COLUMNS
    assert X.index.equals(y.index)
    assert X.notna().all().all() and y.notna().all()
    assert set(y.unique()) <= {0, 1}
    # Warm-up rows at the start and the last 5 unlabelled rows are dropped.
    assert len(X) == len(prices) - WARM_UP + 1 - 5
    # The live prediction uses the most recent day in the data.
    assert latest_row.index[-1] == prices.index[-1]
    assert len(latest_row) == 1


# --- Return label and price dataset (stage 1) ---------------------------------

def test_return_label_known_values():
    df = add_return_label(make_prices([100, 110, 121, 100]), horizon=1)

    assert df["future_return"].iloc[0] == pytest.approx(0.10)
    assert df["future_return"].iloc[1] == pytest.approx(0.10)
    assert df["future_return"].iloc[2] == pytest.approx(100 / 121 - 1)
    assert pd.isna(df["future_return"].iloc[-1])   # no known future yet


def test_return_label_last_horizon_rows_are_nan():
    df = add_return_label(make_prices(range(1, 101)), horizon=5)
    assert df["future_return"].iloc[-5:].isna().all()
    assert df["future_return"].iloc[:-5].notna().all()


def test_make_price_dataset_shapes():
    prices = make_random_prices()
    X, y, latest_row, closes = make_price_dataset(prices)

    assert list(X.columns) == FEATURE_COLUMNS
    assert X.index.equals(y.index) and X.index.equals(closes.index)
    assert X.notna().all().all() and y.notna().all()
    assert len(X) == len(prices) - WARM_UP + 1 - 5
    assert latest_row.index[-1] == prices.index[-1]
    # The return turns each close back into the real price 5 days later.
    later = prices["Close"].shift(-5).loc[X.index]
    assert (closes * (1 + y)).values == pytest.approx(later.values)


def test_above_label_known_values():
    close = make_prices([100, 100, 110, 90, 100])["Close"]
    predicted = pd.Series([105.0, 95.0, float("nan"), 100.0, 100.0], index=close.index)
    above = add_above_label(close, predicted, horizon=1)

    assert above.iloc[0] == 0.0          # real 100 is below the predicted 105
    assert above.iloc[1] == 1.0          # real 110 is above the predicted 95
    assert pd.isna(above.iloc[2])        # no prediction for that day
    assert above.iloc[3] == 0.0          # equal is not above
    assert pd.isna(above.iloc[4])        # future still unknown
