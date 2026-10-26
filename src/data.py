"""Downloading data: daily stock prices (and, later, news headlines)."""

import pandas as pd
import yfinance as yf

from src.config import HISTORY_PERIOD

# The only columns the rest of the project uses, in this order.
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def normalize_ticker(ticker: str) -> str:
    """Clean up what the user typed: ' tcs.ns ' -> 'TCS.NS'."""
    return ticker.strip().upper()


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Turn yfinance's two-level column headers into plain names.

    Recent yfinance versions return columns like ("Close", "AAPL") even when
    we ask for a single ticker. We keep only the level holding the price
    names ("Open", "Close", ...), then select our five columns in a fixed order.
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # Find which header level contains "Close" and keep just that one.
        for level in range(df.columns.nlevels):
            if "Close" in df.columns.get_level_values(level):
                df.columns = df.columns.get_level_values(level)
                break
    df = df[PRICE_COLUMNS]
    df.columns.name = None  # drop the leftover "Price" header label
    return df


def load_prices(ticker: str, period: str = HISTORY_PERIOD) -> pd.DataFrame:
    """Download daily prices for one ticker.

    Returns a DataFrame indexed by date (oldest first) with columns
    Open, High, Low, Close, Volume. Raises ValueError if nothing comes back.
    """
    ticker = normalize_ticker(ticker)
    if not ticker:
        raise ValueError("Please enter a ticker symbol.")

    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"No price data found for {ticker}")

    df = flatten_columns(raw)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.dropna(subset=["Close"])

    if df.empty:
        raise ValueError(f"No price data found for {ticker}")
    return df
