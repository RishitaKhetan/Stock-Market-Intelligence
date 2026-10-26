"""Streamlit page for the Stock Market Intelligence dashboard.

This file only draws the UI. All the real work lives in the src/ modules.
"""

import plotly.graph_objects as go
import streamlit as st

from src.config import CHART_DAYS, MA_LONG, MA_SHORT, POPULAR_TICKERS
from src.data import load_prices, normalize_ticker
from src.features import add_moving_averages

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

    # --- Price chart -------------------------------------------------------
    prices = add_moving_averages(prices)
    recent = prices.tail(CHART_DAYS)

    latest_close = prices["Close"].iloc[-1]
    previous_close = prices["Close"].iloc[-2]
    change_1d = latest_close - previous_close
    change_1d_pct = change_1d / previous_close * 100
    col1, col2 = st.columns(2)
    col1.metric(f"Latest close ({end})", f"{latest_close:,.2f}")
    # Main number = price change; the coloured arrow underneath = % change.
    col2.metric("1-day change", f"{change_1d:+,.2f}", delta=f"{change_1d_pct:+.2f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent.index, y=recent["Close"], name="Close",
                             line=dict(width=2)))
    fig.add_trace(go.Scatter(x=recent.index, y=recent[f"MA{MA_SHORT}"],
                             name=f"{MA_SHORT}-day average", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=recent.index, y=recent[f"MA{MA_LONG}"],
                             name=f"{MA_LONG}-day average", line=dict(dash="dot")))
    fig.update_layout(
        title=f"{ticker}: last 6 months",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, width="stretch")

    # Kept for checking the download; collapsed so it stays out of the way.
    with st.expander("Show raw data (last 5 days)"):
        st.dataframe(prices.tail())
