"""Streamlit page for the Stock Market Intelligence dashboard.

This file only draws the UI. All the real work lives in the src/ modules.
"""

import plotly.graph_objects as go
import streamlit as st

from src.config import CHART_DAYS, HORIZON_DAYS, MA_LONG, MA_SHORT, POPULAR_TICKERS
from src.data import load_prices, normalize_ticker
from src.features import add_moving_averages, make_dataset
from src.model import (
    evaluate,
    feature_importance,
    predict_latest,
    time_split,
    train_model,
)

st.set_page_config(page_title="Stock Market Intelligence", layout="wide")


# --- Cached steps (re-run at most once an hour per ticker) --------------------

@st.cache_data(ttl=3600, show_spinner="Downloading prices...")
def get_prices(ticker: str):
    return load_prices(ticker)


@st.cache_data(ttl=3600, show_spinner="Training the model...")
def run_model(ticker: str) -> dict:
    """Train, test on the last year, then retrain on everything for today's call."""
    X, y, latest_row = make_dataset(get_prices(ticker))

    # 1) Honest score: train on older data, test on the most recent year.
    X_train, X_test, y_train, y_test = time_split(X, y)
    evaluation = evaluate(train_model(X_train, y_train), X_test, y_test)

    # 2) Live prediction: retrain on all labelled days so it knows the newest data.
    final_model = train_model(X, y)
    return {
        "prob_up": predict_latest(final_model, latest_row),
        "importance": feature_importance(final_model),
        "evaluation": evaluation,
    }


# --- Page --------------------------------------------------------------------

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
        prices = get_prices(ticker)
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

    # --- Prediction --------------------------------------------------------
    try:
        result = run_model(ticker)
    except ValueError as err:
        st.error(f"Can't build a prediction for {ticker}: {err}")
        st.stop()

    st.header("Prediction")
    left, right = st.columns([2, 3])

    with left:
        # Colour by the rounded number shown, so "UP 50%" is never painted red.
        up_pct = round(result["prob_up"] * 100)
        color = "green" if up_pct > 50 else "red" if up_pct < 50 else "gray"
        st.markdown(f"## :{color}[UP {up_pct}% / DOWN {100 - up_pct}%]")
        st.caption(f"Probability the price is higher in {HORIZON_DAYS} trading days. "
                   "Not a guarantee.")

    with right:
        importance = result["importance"]
        bars = go.Figure(go.Bar(x=importance.values, y=importance.index,
                                orientation="h"))
        bars.update_layout(
            title="Why this prediction?",
            xaxis_title="Importance (share of the model's decisions)",
            xaxis_tickformat=".0%",
            yaxis=dict(autorange="reversed"),  # most important at the top
            height=350,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(bars, width="stretch")

    # --- Accuracy ----------------------------------------------------------
    ev = result["evaluation"]
    st.header("How accurate is it?")
    col1, col2 = st.columns(2)
    gap = (ev["model_accuracy"] - ev["baseline_accuracy"]) * 100
    col1.metric("Model accuracy", f"{ev['model_accuracy']:.1%}",
                delta=f"{gap:+.1f} pts vs baseline")
    col2.metric('Baseline: "always guess UP"', f"{ev['baseline_accuracy']:.1%}")
    st.caption(
        f"Tested on {ev['test_start'].date()} to {ev['test_end'].date()}: "
        "the most recent year, which the model never saw during training. "
        "The baseline is right on every UP day, so the model must beat it to be useful."
    )
