"""Turning raw prices into indicators, model features and the labels.

Golden rule (no look-ahead leakage): a feature on day t may only use data
up to and including day t. Only the label looks into the future.
"""

import numpy as np
import pandas as pd

from src.config import HORIZON_DAYS, MA_LONG, MA_SHORT, RSI_WINDOW

# The model's inputs, in a fixed order.
FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "ret_10",
    "dist_ma20",
    "dist_ma50",
    "volatility_10",
    "rsi_14",
    "volume_change_5",
]

# Plain-English names shown in the app instead of the short column names.
FEATURE_LABELS = {
    "ret_1": "Return over last 1 day",
    "ret_5": "Return over last 5 days",
    "ret_10": "Return over last 10 days",
    "dist_ma20": "Distance from 20-day average",
    "dist_ma50": "Distance from 50-day average",
    "volatility_10": "Volatility over last 10 days",
    "rsi_14": "RSI (14-day momentum)",
    "volume_change_5": "Change in trading volume over 5 days",
}


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add MA20 and MA50 columns: the average Close over the last 20 / 50 days.

    A moving average on day t uses only days up to and including t, so it
    never peeks into the future. The first 19 (or 49) values are NaN because
    there aren't enough days yet to average over.
    """
    df = df.copy()
    df[f"MA{MA_SHORT}"] = df["Close"].rolling(MA_SHORT).mean()
    df[f"MA{MA_LONG}"] = df["Close"].rolling(MA_LONG).mean()
    return df


def compute_rsi(close: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    """Relative Strength Index (0 to 100), using Wilder's smoothing.

    Compares the size of recent up-moves with recent down-moves.
    Above 70 is often called "overbought", below 30 "oversold".
    """
    change = close.diff()
    gain = change.clip(lower=0)      # up-moves, 0 on down days
    loss = -change.clip(upper=0)     # down-moves as positive numbers
    # Wilder's smoothing is an exponential average with alpha = 1 / window.
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rsi = 100 - 100 / (1 + avg_gain / avg_loss)   # avg_loss == 0 gives 100
    # If the price didn't move at all, there is no momentum either way.
    rsi[(avg_gain == 0) & (avg_loss == 0)] = 50.0
    return rsi


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model's input columns (FEATURE_COLUMNS) from a price table.

    Needs Close and Volume columns. Every feature looks only backwards, so the
    first ~50 rows are NaN while the longest window (MA50) warms up.
    """
    close = df["Close"]
    daily_return = close.pct_change()

    features = pd.DataFrame(index=df.index)
    features["ret_1"] = close.pct_change(1)
    features["ret_5"] = close.pct_change(5)
    features["ret_10"] = close.pct_change(10)
    features["dist_ma20"] = close / close.rolling(MA_SHORT).mean() - 1
    features["dist_ma50"] = close / close.rolling(MA_LONG).mean() - 1
    features["volatility_10"] = daily_return.rolling(10).std()
    features["rsi_14"] = compute_rsi(close)
    features["volume_change_5"] = df["Volume"].pct_change(5)

    # A day with 0 volume would make the % change infinite; treat as missing.
    return features.replace([np.inf, -np.inf], np.nan)[FEATURE_COLUMNS]



def add_return_label(df: pd.DataFrame, horizon: int = HORIZON_DAYS) -> pd.DataFrame:
    """Add the number the price model must learn: the return over the next `horizon` days.

    future_return = Close[t + horizon] / Close[t] - 1 (e.g. 0.02 means +2%).
    The last `horizon` days don't have a known future yet, so it is NaN.
    """
    df = df.copy()
    df["future_return"] = df["Close"].shift(-horizon) / df["Close"] - 1
    return df


def add_above_label(close: pd.Series, predicted_price: pd.Series,
                    horizon: int = HORIZON_DAYS) -> pd.Series:
    """The answer for stage 2: did the real price end up ABOVE the predicted one?

    above = 1 if Close[t + horizon] > predicted_price[t], else 0.
    `close` is the full Close series; the result follows predicted_price's dates.
    NaN where the prediction or the future price is unknown.
    """
    future_close = close.shift(-horizon).reindex(predicted_price.index)
    above = (future_close > predicted_price).astype(float)
    above[future_close.isna() | predicted_price.isna()] = np.nan
    return above


def make_price_dataset(df: pd.DataFrame):
    """Split a price table into what the price model trains on and what it predicts.

    Returns:
        X: features for every day with full features and a known future return
        y: the matching future returns
        latest_row: features of the most recent day (future still unknown),
                    used for today's live prediction (a 1-row DataFrame)
        closes: the Close price on each day of X (turns returns into prices)
    """
    features = build_features(df)
    future_return = add_return_label(df)["future_return"]

    labelled = features.join(future_return).dropna()
    X = labelled[FEATURE_COLUMNS]
    y = labelled["future_return"]
    latest_row = features.iloc[[-1]]
    closes = df["Close"].loc[X.index]
    return X, y, latest_row, closes

