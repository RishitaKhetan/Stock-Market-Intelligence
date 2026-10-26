"""Turning raw prices into indicators (and, later, model features and labels)."""

import pandas as pd

from src.config import MA_LONG, MA_SHORT


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
