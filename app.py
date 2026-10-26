"""Streamlit page for the Stock Market Intelligence dashboard.

This file only draws the UI. All the real work lives in the src/ modules.
"""

import streamlit as st

from src.config import POPULAR_TICKERS
from src.data import load_prices, normalize_ticker

st.set_page_config(page_title="Stock Market Intelligence", layout="wide")

st.title("Stock Market Intelligence")
st.caption("Educational demo, not financial advice.")

ticker = st.selectbox(
    "Stock ticker",
    options=list(POPULAR_TICKERS),
    index=0,  # AAPL
    # Show "AAPL - Apple"; a typed-in symbol is shown as-is.
    format_func=lambda t: f"{t} - {POPULAR_TICKERS[t]}" if t in POPULAR_TICKERS else t,
    accept_new_options=True,
    help="Pick a stock, or type any Yahoo Finance symbol (Indian NSE stocks end in .NS).",
)
analyze = st.button("Analyze", type="primary")

if analyze:
    ticker = normalize_ticker(ticker or "")  # None if the box was cleared
    try:
        prices = load_prices(ticker)
    except ValueError as err:
        st.error(f"{err}. Check the symbol (Indian NSE stocks end in .NS, e.g. TCS.NS).")
        st.stop()

    start = prices.index[0].date()
    end = prices.index[-1].date()
    st.success(f"Loaded {len(prices):,} days of prices for {ticker} ({start} to {end}).")
    st.dataframe(prices.tail())
