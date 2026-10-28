"""Tests for src/model.py, all on small synthetic data (no downloads)."""

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, FEATURE_LABELS
from src.model import (
    evaluate,
    evaluate_regression,
    feature_importance,
    predict_latest,
    predict_price,
    time_split,
    train_model,
    train_regressor,
)


def make_learnable_data(n=800, seed=0):
    """Random features where the answer is simple: UP whenever ret_5 > 0."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    X = pd.DataFrame(rng.normal(0, 1, (n, len(FEATURE_COLUMNS))),
                     index=dates, columns=FEATURE_COLUMNS)
    y = (X["ret_5"] > 0).astype(int)
    return X, y


def test_time_split_keeps_order():
    X, y = make_learnable_data()
    X_train, X_test, y_train, y_test = time_split(X, y, test_days=252)

    assert X_train.index.max() < X_test.index.min()
    assert len(X_test) == 252 and len(X_train) == len(X) - 252
    assert X_train.index.equals(y_train.index) and X_test.index.equals(y_test.index)


def test_time_split_too_little_data():
    X, y = make_learnable_data(n=200)
    with pytest.raises(ValueError):
        time_split(X, y, test_days=252)


def test_model_learns_simple_rule():
    X, y = make_learnable_data()
    X_train, X_test, y_train, y_test = time_split(X, y, test_days=252)
    result = evaluate(train_model(X_train, y_train), X_test, y_test)

    assert result["model_accuracy"] > 0.8          # clearly above a coin flip
    assert result["baseline_accuracy"] == pytest.approx(y_test.mean())
    assert result["test_start"] == X_test.index[0]
    assert result["test_end"] == X_test.index[-1]


def test_predict_latest_is_a_probability():
    X, y = make_learnable_data()
    model = train_model(X, y)

    prob = predict_latest(model, X.iloc[[-1]])
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_predict_latest_when_training_had_no_up_days():
    X, _ = make_learnable_data()
    model = train_model(X, pd.Series(0, index=X.index))
    assert predict_latest(model, X.iloc[[-1]]) == 0.0


def test_feature_importance():
    X, y = make_learnable_data()
    importance = feature_importance(train_model(X, y))

    assert importance.sum() == pytest.approx(1.0)
    assert set(importance.index) == set(FEATURE_LABELS.values())
    assert importance.is_monotonic_decreasing
    # The only feature that matters should come out on top.
    assert importance.index[0] == FEATURE_LABELS["ret_5"]


# --- Stage 1: price regression -------------------------------------------------

def make_learnable_returns(n=800, seed=0):
    """Random features where the future return is simple: 2% x ret_5."""
    X, _ = make_learnable_data(n, seed)
    y = 0.02 * X["ret_5"]
    closes = pd.Series(100.0, index=X.index)
    return X, y, closes


def test_regressor_beats_naive_baseline():
    X, y, closes = make_learnable_returns()
    X_train, X_test, y_train, y_test = time_split(X, y, test_days=252)
    result = evaluate_regression(train_regressor(X_train, y_train), X_test, y_test,
                                 closes.loc[X_test.index])

    assert result["model_mae"] < 0.5 * result["naive_mae"]
    assert result["model_mape"] < result["naive_mape"]
    assert result["test_start"] == X_test.index[0]
    assert result["test_end"] == X_test.index[-1]


def test_naive_baseline_mae_is_the_average_move():
    X, y, closes = make_learnable_returns()
    X_train, X_test, y_train, y_test = time_split(X, y, test_days=252)
    result = evaluate_regression(train_regressor(X_train, y_train), X_test, y_test,
                                 closes.loc[X_test.index])
    # "No change" misses by exactly how far the price really moved.
    assert result["naive_mae"] == pytest.approx((100 * y_test).abs().mean())


def test_predict_price_applies_return_to_last_close():
    X, y, _ = make_learnable_returns()
    model = train_regressor(X, y)
    latest = X.iloc[[-1]]

    expected = 250.0 * (1 + model.predict(latest)[0])
    assert predict_price(model, latest, 250.0) == pytest.approx(expected)


def test_feature_importance_works_for_regressor():
    X, y, _ = make_learnable_returns()
    importance = feature_importance(train_regressor(X, y))
    assert importance.sum() == pytest.approx(1.0)
    assert importance.index[0] == FEATURE_LABELS["ret_5"]
