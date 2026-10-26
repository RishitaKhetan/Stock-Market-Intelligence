"""Streamlit page for the Stock Market Intelligence dashboard.

This file only draws the UI. All the real work lives in the src/ modules.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.backtest import run_backtest
from src.config import CHART_DAYS, HORIZON_DAYS, MA_LONG, MA_SHORT, POPULAR_TICKERS
from src.data import load_news, load_prices, normalize_ticker
from src.features import add_moving_averages, make_dataset
from src.model import (
    evaluate,
    feature_importance,
    predict_latest,
    predict_proba_up,
    time_split,
    train_model,
)
from src.sentiment import load_analyzer, mood_summary, score_headlines

st.set_page_config(page_title="Stock Market Intelligence", layout="wide")

# How each headline label is shown in the news table.
LABEL_BADGES = {"positive": "🟢 positive", "negative": "🔴 negative", "neutral": "⚪ neutral"}
# Mood word under the score: green for positive, red for negative, grey for neutral.
MOOD_DELTA_COLOR = {"positive": "green", "negative": "red", "neutral": "gray"}


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
    held_out_model = train_model(X_train, y_train)
    evaluation = evaluate(held_out_model, X_test, y_test)
    # P(UP) on each test day, from the model that never saw those days (for the backtest).
    test_probs = predict_proba_up(held_out_model, X_test)

    # 2) Live prediction: retrain on all labelled days so it knows the newest data.
    final_model = train_model(X, y)
    return {
        "prob_up": predict_latest(final_model, latest_row),
        "importance": feature_importance(final_model),
        "evaluation": evaluation,
        "test_probs": test_probs,
    }


@st.cache_resource(show_spinner="Loading FinBERT (first run downloads about 400 MB)...")
def get_analyzer():
    """Load the sentiment model once and share it across all users and reruns."""
    return load_analyzer()


@st.cache_data(ttl=1800, show_spinner="Reading the news with FinBERT...")
def get_news_mood(ticker: str):
    """Fetch headlines and score them. Returns (news, scores, engine name)."""
    news = load_news(ticker)
    analyzer, engine = get_analyzer()
    scores = score_headlines([n["title"] for n in news], analyzer)
    return news, scores, engine


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

    # --- News mood ---------------------------------------------------------
    st.header("News mood")
    news, scores, engine = get_news_mood(ticker)

    if not news:
        st.info("No recent news found for this stock.")
    else:
        summary = mood_summary(scores)
        col1, col2 = st.columns([1, 3])
        col1.metric("Mood score (-1 to +1)", f"{summary['mean_score']:+.2f}",
                    delta=summary["mood"], delta_color=MOOD_DELTA_COLOR[summary["mood"]],
                    delta_arrow="off")
        counts = summary["counts"]
        col2.markdown(
            f"**{len(scores)} recent headlines:** "
            f":green[{counts['positive']} positive] · :red[{counts['negative']} negative] · "
            f":gray[{counts['neutral']} neutral]"
        )
        col2.caption(f"Each headline scored by {engine}. "
                     "Score = chance it is positive minus chance it is negative.")

        table = pd.DataFrame({
            "Mood": [LABEL_BADGES[label] for label in scores["label"]],
            "Score": scores["score"].round(2),
            "Headline": scores["title"],
            "Source": [n["publisher"] for n in news],
            "Date": [n["published"].strftime("%d %b") if pd.notna(n["published"]) else ""
                     for n in news],
            "Link": [n["link"] for n in news],
        })
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "Link": st.column_config.LinkColumn("Link", display_text="Open"),
                "Score": st.column_config.NumberColumn(format="%+.2f"),
            },
        )
        if engine != "FinBERT":
            st.caption("FinBERT could not be loaded, so the simpler VADER scorer was used.")

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

    # --- Backtest ----------------------------------------------------------
    # Indian stocks are priced in rupees, the rest in dollars.
    money = "₹" if ticker.endswith((".NS", ".BO")) else "$"
    bt = run_backtest(prices, result["test_probs"])
    final_strategy = bt["strategy_value"].iloc[-1]
    final_hold = bt["buy_hold_value"].iloc[-1]

    st.header("Backtest: would following the model have paid off?")
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{money}100 following the model", f"{money}{final_strategy:,.2f}",
                delta=f"{final_strategy - 100:+.1f}%")
    col2.metric(f"{money}100 just holding the stock", f"{money}{final_hold:,.2f}",
                delta=f"{final_hold - 100:+.1f}%")
    col3.metric("Days invested", f"{bt['invested'].sum()} of {len(bt)}")

    line = go.Figure()
    line.add_trace(go.Scatter(x=bt.index, y=bt["strategy_value"],
                              name="Following the model", line=dict(width=2)))
    line.add_trace(go.Scatter(x=bt.index, y=bt["buy_hold_value"],
                              name="Just holding the stock", line=dict(dash="dash")))
    line.update_layout(
        title=f"{money}100 following the model vs {money}100 just holding the stock",
        xaxis_title="Date",
        yaxis_title=f"Value ({money})",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(line, width="stretch")
    st.caption(
        f"Same test year as above ({bt.index[0].date()} to {bt.index[-1].date()}). "
        f"The model invests for the next day only when it says UP is more likely than DOWN, "
        "otherwise it holds cash. Ignores trading costs and taxes. "
        "Past results do not predict future results."
    )
