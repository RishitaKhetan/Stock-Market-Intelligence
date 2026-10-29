"""Streamlit page for the Stock Market Intelligence dashboard.

This file only draws the UI. All the real work lives in the src/ modules.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.backtest import run_backtest
from src.config import CHART_DAYS, HORIZON_DAYS, MA_LONG, MA_SHORT, POPULAR_TICKERS
from src.data import load_news_headlines, load_news_history, load_price_history, normalize_ticker_symbol
from src.features import add_moving_averages, make_price_dataset
from src.model import (
    evaluate_regression,
    feature_importance,
    predict_price,
    run_stage2,
    time_split,
    train_regressor,
)
from src.sentiment import load_sentiment_analyzer, summarize_news_mood, score_headline_sentiment

st.set_page_config(page_title="Stock Market Intelligence", layout="wide")

# How each headline label is shown in the news table.
LABEL_BADGES = {"positive": "🟢 positive", "negative": "🔴 negative", "neutral": "⚪ neutral"}
# Mood word under the score: green for positive, red for negative, grey for neutral.
MOOD_DELTA_COLOR = {"positive": "green", "negative": "red", "neutral": "gray"}
# One-click examples in the sidebar: two US stocks and one Indian stock.
EXAMPLE_TICKERS = ["AAPL", "NVDA", "RELIANCE.NS"]


# --- Cached steps (re-run at most once an hour per ticker) --------------------

@st.cache_data(ttl=3600, show_spinner="Downloading prices...")
def get_prices(ticker: str):
    return load_price_history(ticker)


@st.cache_data(ttl=3600, show_spinner="Training the model...")
def run_model(ticker: str) -> dict:
    """Train, test on the last year, then retrain on everything for today's call."""
    prices = get_prices(ticker)
    X_p, y_p, latest_p, closes = make_price_dataset(prices)

    # 1) Honest score: train on older data, test on the most recent year.
    Xp_train, Xp_test, yp_train, yp_test = time_split(X_p, y_p)
    held_out_regressor = train_regressor(Xp_train, yp_train)
    price_evaluation = evaluate_regression(held_out_regressor, Xp_test, yp_test,
                                           closes.loc[Xp_test.index])
    # Predicted return on each test day, from the model that never saw those days (for the backtest).
    stage1_test_returns = pd.Series(held_out_regressor.predict(Xp_test), index=Xp_test.index)

    # 2) Live prediction: retrain on all labelled days so it knows the newest data.
    final_regressor = train_regressor(X_p, y_p)
    predicted_price = predict_price(final_regressor, latest_p, prices["Close"].iloc[-1])

    # Stage 2: does the news say the real price will land above or below that?
    news_history = get_news_history(ticker)
    verdict = run_stage2(X_p, y_p, prices["Close"], news_history,
                         predicted_price / prices["Close"].iloc[-1] - 1)
    return {
        "predicted_price": predicted_price,
        "verdict": verdict,
        "news_history_size": len(news_history),
        "importance": feature_importance(final_regressor),
        "price_evaluation": price_evaluation,
        "stage1_test_returns": stage1_test_returns,
    }


@st.cache_resource(show_spinner="Loading FinBERT (first run downloads about 400 MB)...")
def get_analyzer():
    """Load the sentiment model once and share it across all users and reruns."""
    return load_sentiment_analyzer()


@st.cache_data(ttl=3600, show_spinner="Collecting 2 years of news (the first run for a stock "
                                       "takes several minutes)...")
def get_news_history(ticker: str) -> pd.DataFrame:
    """Scored dated headlines for training stage 2. Empty if anything goes wrong."""
    analyzer, _ = get_analyzer()
    try:
        return load_news_history(ticker, lambda titles: score_headline_sentiment(titles, analyzer))
    except Exception:
        return pd.DataFrame(columns=["published", "score", "label"])


@st.cache_data(ttl=1800, show_spinner="Reading the news with FinBERT...")
def get_news_mood(ticker: str):
    """Fetch headlines and score them. Returns (news, scores, engine name)."""
    news = load_news_headlines(ticker)
    analyzer, engine = get_analyzer()
    scores = score_headline_sentiment([n["title"] for n in news], analyzer)
    return news, scores, engine


# --- Page --------------------------------------------------------------------

st.title("Stock Market Intelligence")
st.caption("Educational demo, not financial advice.")

# Remember the chosen ticker and the last analysed one across reruns, so the
# results stay on screen while the user scrolls or opens an expander.
st.session_state.setdefault("ticker", "AAPL")
st.session_state.setdefault("analyzed", None)


def analyze_example(symbol: str):
    """Sidebar button: pick this ticker in the dropdown and analyse it."""
    st.session_state.ticker = symbol
    st.session_state.analyzed = symbol


with st.sidebar:
    st.header("Try an example")
    for symbol in EXAMPLE_TICKERS:
        st.button(f"{symbol} - {POPULAR_TICKERS[symbol]}", key=f"example_{symbol}",
                  on_click=analyze_example, args=(symbol,), width="stretch")
    st.caption("Or pick any stock in the dropdown, or type a Yahoo Finance symbol.")

ticker = st.selectbox(
    "Stock ticker",
    options=list(POPULAR_TICKERS),
    key="ticker",
    # Show "AAPL - Apple"; a typed-in symbol is shown as-is.
    format_func=lambda t: f"{t} - {POPULAR_TICKERS[t]}" if t in POPULAR_TICKERS else t,
    accept_new_options=True,
    help="Pick a stock, or type any Yahoo Finance symbol (Indian NSE stocks end in .NS).",
)
if st.button("Analyze", type="primary"):
    st.session_state.analyzed = normalize_ticker_symbol(ticker or "")  # None if box cleared

if st.session_state.analyzed is not None:
    ticker = st.session_state.analyzed
    try:
        prices = get_prices(ticker)
    except ValueError as err:
        st.error(f"{err}. Check the symbol (Indian NSE stocks end in .NS, e.g. TCS.NS).")
        st.stop()

    start = prices.index[0].date()
    end = prices.index[-1].date()
    st.success(f"Loaded {len(prices):,} days of prices for {ticker} ({start} to {end}).")

    try:
        result = run_model(ticker)
    except ValueError as err:
        st.error(f"Can't build a prediction for {ticker}: {err}")
        st.stop()

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
    # Weekends skipped; market holidays are not, so the date can be a day or two early.
    target_date = prices.index[-1] + pd.offsets.BDay(HORIZON_DAYS)
    fig.add_trace(go.Scatter(x=[prices.index[-1], target_date],
                             y=[latest_close, result["predicted_price"]],
                             name=f"Predicted price in {HORIZON_DAYS} days",
                             mode="lines+markers", line=dict(dash="dot")))
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
    st.header("Prediction")
    left, right = st.columns([2, 3])

    with left:
        predicted = result["predicted_price"]
        st.metric(f"Predicted price in {HORIZON_DAYS} trading days", f"{predicted:,.2f}",
                  delta=f"{(predicted / latest_close - 1) * 100:+.2f}% vs latest close")
        verdict = result["verdict"]
        if verdict is None:
            st.info("Not enough news history for an above/below verdict on this stock.")
        else:
            above_pct = round(verdict["prob_above"] * 100)
            side, color = ("ABOVE", "green") if above_pct >= 50 else ("BELOW", "red")
            st.markdown(f"### News says: :{color}[likely {side}] that price "
                        f"({max(above_pct, 100 - above_pct)}%)")
            today = verdict["news_today"]
            st.caption(f"Based on {int(today['news_count'])} headlines from the last week "
                       f"(average mood {today['news_mean']:+.2f}).")
        st.caption("Stage 1 predicts the price from price history; stage 2 reads the news "
                   "to say if the real price will land above or below it. Not a guarantee.")

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
        summary = summarize_news_mood(scores)
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
    ev = result["price_evaluation"]
    st.header("How accurate is it?")
    col1, col2 = st.columns(2)
    # Smaller miss is better, so a negative gap is shown in green.
    col1.metric("Model: average miss", f"{ev['model_mae']:,.2f}",
                delta=f"{ev['model_mae'] - ev['naive_mae']:+,.2f} vs baseline",
                delta_color="inverse")
    col1.caption(f"{ev['model_mape']:.2%} of the price")
    col2.metric('Baseline: "no change"', f"{ev['naive_mae']:,.2f}")
    col2.caption(f"{ev['naive_mape']:.2%} of the price")
    st.caption(
        f"Average gap between the predicted and the real price {HORIZON_DAYS} trading days later, "
        f"tested on {ev['test_start'].date()} to {ev['test_end'].date()}: the most recent year, "
        "which the model never saw during training. The baseline guesses today's price, "
        "so the model must miss by less to be useful."
    )

    if result["verdict"] is not None:
        with_news, price_only = result["verdict"]["with_news"], result["verdict"]["price_only"]
        st.subheader("Above / below verdict")
        col1, col2, col3 = st.columns(3)
        col1.metric("With news", f"{with_news['model_accuracy']:.1%}")
        col2.metric("Price only (no news)", f"{price_only['model_accuracy']:.1%}")
        col3.metric('Baseline: "always above"', f"{with_news['baseline_accuracy']:.1%}")
        st.caption(
            f"Share of right above/below calls on {with_news['test_start'].date()} to "
            f"{with_news['test_end'].date()}, after training on {result['verdict']['train_days']} days "
            f"({result['news_history_size']:,} headlines in total). The news only helps if "
            "\"With news\" beats both others. With this little history, expect noisy results."
        )

    # --- Backtest ----------------------------------------------------------
    # Indian stocks are priced in rupees, the rest in dollars.
    money = "₹" if ticker.endswith((".NS", ".BO")) else "$"
    # Stage 1 invests when it predicts a rise; "+ news" also needs the verdict to say ABOVE.
    stage1_signal = (result["stage1_test_returns"] > 0).astype(float)
    signals = {"Stage 1 only": stage1_signal}
    if result["verdict"] is not None:
        news_probs = result["verdict"]["test_probs"]   # stage 2's test days (the last ~6 months)
        stage1_signal = stage1_signal.reindex(news_probs.index).fillna(0.0)
        signals = {"Stage 1 only": stage1_signal,
                   "Stage 1 + news": stage1_signal * (news_probs >= 0.5)}
    runs = {name: run_backtest(prices, signal) for name, signal in signals.items()}
    bt = next(iter(runs.values()))

    st.header("Backtest: would following the model have paid off?")
    cols = st.columns(len(runs) + 1)
    for col, (name, run) in zip(cols, runs.items()):
        final = run["strategy_value"].iloc[-1]
        col.metric(f"{money}100 following {name.lower()}", f"{money}{final:,.2f}",
                   delta=f"{final - 100:+.1f}%")
        col.caption(f"Invested {run['invested'].sum()} of {len(run)} days")
    final_hold = bt["buy_hold_value"].iloc[-1]
    cols[-1].metric(f"{money}100 just holding the stock", f"{money}{final_hold:,.2f}",
                    delta=f"{final_hold - 100:+.1f}%")

    line = go.Figure()
    for name, run in runs.items():
        line.add_trace(go.Scatter(x=run.index, y=run["strategy_value"], name=name, line=dict(width=2)))
    line.add_trace(go.Scatter(x=bt.index, y=bt["buy_hold_value"],
                              name="Just holding the stock", line=dict(dash="dash")))
    line.update_layout(
        title=f"{money}100 following the models vs {money}100 just holding the stock",
        xaxis_title="Date",
        yaxis_title=f"Value ({money})",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(line, width="stretch")
    st.caption(
        f"Test days only ({bt.index[0].date()} to {bt.index[-1].date()}), which the models never "
        "trained on. Each day, a strategy holds the stock for the next day only if stage 1 predicts "
        "a rise (and, for \"+ news\", stage 2 says ABOVE); otherwise it holds cash. "
        "Ignores trading costs and taxes. Past results do not predict future results."
    )

# --- How it works (always shown at the bottom) ---------------------------------
with st.expander("How it works"):
    st.markdown(
        f"""
1. **Prices:** we download about 5 years of daily prices from Yahoo Finance.
2. **Features:** from the prices we compute 8 simple signals, such as recent returns,
   distance from the 20- and 50-day averages, volatility, RSI and volume change.
   Each one uses only data up to that day, never the future.
3. **Stage 1, price model:** a random forest (many small decision trees that vote) learns
   the return over the next {HORIZON_DAYS} trading days, turned into a predicted price.
   The "Why this prediction?" chart shows which signals it relied on most.
4. **Honest testing:** it trains on older years and is scored on the most recent year,
   which it never saw. It is compared with simply guessing "no change".
5. **News mood:** headlines are scored by FinBERT, a language model trained on financial
   text, as positive, negative or neutral.
6. **Stage 2, news verdict:** about 2 years of dated Google News headlines give a news mood
   for every past day. A second model learns from that mood whether the real price ended
   up above or below stage 1's prediction, and is scored against the same model without
   news and against "always above".
7. **Backtest:** we replay the test days, investing only when stage 1 predicts a rise
   (and, with news, when stage 2 says above), and compare with simply holding the stock.
"""
    )
