"""A simple backtest: what if we had followed the model during the test year?

Rule: at the close of day t, if the model says P(UP) > threshold, hold the
stock during day t+1; otherwise sit in cash (earning 0). The signal is shifted
by one day, so a decision never uses the return it is trying to capture.
"""

import pandas as pd


def run_backtest(prices: pd.DataFrame, test_probs: pd.Series,
                 threshold: float = 0.5) -> pd.DataFrame:
    """Grow 100 by following the model, and 100 by just holding the stock.

    prices:     price table with a Close column (any length)
    test_probs: P(UP) for each test day, from the model trained only on older data

    Returns one row per trading day in the test period with columns
    strategy_value, buy_hold_value (both start at 100) and invested
    (1 if the strategy held the stock during that day, else 0).
    """
    dates = test_probs.index
    close = prices["Close"].loc[dates[0]:dates[-1]]
    daily_return = close.pct_change().fillna(0.0)  # day 1 has no return yet

    # Signal decided at the close of each day: 1 = invest, 0 = cash.
    # (If a day has no prediction, keep the previous decision.)
    signal = (test_probs > threshold).astype(float).reindex(close.index).ffill().fillna(0.0)
    # Today's return belongs to yesterday's decision.
    invested = signal.shift(1).fillna(0.0)

    strategy_return = daily_return * invested
    return pd.DataFrame({
        "strategy_value": 100 * (1 + strategy_return).cumprod(),
        "buy_hold_value": 100 * (1 + daily_return).cumprod(),
        "invested": invested.astype(int),
    })
