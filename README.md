# Stock Market Intelligence

A single-page Streamlit dashboard. Pick a stock (for example `AAPL`, or an Indian NSE stock such as `TCS.NS`), press **Analyze**, and the page shows:

1. **Price chart:** the last 6 months with 20-day and 50-day moving averages, plus the predicted price 5 trading days ahead.
2. **Stage 1, predicted price:** a model trained on price history predicts the price in 5 trading days.
3. **Stage 2, news verdict:** a second model reads about 2 years of FinBERT-scored headlines and says whether the real price will likely land **above or below** that prediction.
4. **Why this prediction?:** which inputs the price model relied on most (feature importance).
5. **News mood:** recent headlines scored by FinBERT, plus an overall mood score.
6. **Accuracy:** the price model's average miss vs. "no change"; the verdict vs. the same model without news and vs. "always above".
7. **Backtest:** 100 following stage 1, 100 following stage 1 + news, and 100 just holding the stock, over test days the models never saw.

Everything is computed live from free data. **Educational demo, not financial advice.**

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Then open http://localhost:8501. Use the dropdown, type any Yahoo Finance symbol, or click an example in the sidebar.

- The **first** Analyze downloads FinBERT (about 400 MB, once) and loads it into memory, which takes 30 to 60 seconds.
- The first Analyze of **each stock** also collects about 2 years of news (one Google News query per week, then FinBERT on every headline). This takes several minutes. Headlines are saved in `news_cache/<TICKER>.csv`, so later runs only fetch the newest week and take well under a minute. Results are cached for an hour.
- If you edit files in `src/` while the app is running and see an `ImportError`, stop the server (Ctrl+C) and run `streamlit run app.py` again.

## Run the tests

```bash
pytest -q                    # all tests
pytest -q -m "not network"   # offline tests only (no internet needed)
pytest -q -m network         # only the tests that download real data
```

## How it works

| Step | File | What it does |
|---|---|---|
| Data | `src/data.py` | Downloads ~5 years of daily prices (yfinance), recent headlines, and ~2 years of dated headlines (cached) |
| Features | `src/features.py` | 8 inputs per day (returns, distance from moving averages, volatility, RSI, volume change), the 5-day return label and the above/below label |
| Model | `src/model.py` | Stage 1: random forest regressor for the 5-day return, tested on the last 252 trading days. Stage 2: random forest classifier on news mood + stage 1's out-of-sample prediction, tested on the last 126 days |
| Sentiment | `src/sentiment.py` | FinBERT headline scoring (VADER fallback) and the daily 7-day news mood features |
| Backtest | `src/backtest.py` | Invest for the next day only when the signal says so, otherwise hold cash |
| Page | `app.py` | Streamlit UI only; all logic lives in `src/` |

Settings such as the 5-day horizon and test-set size are in `src/config.py`.

**News sources:** `yf.Ticker(...).news` no longer works (Yahoo retired that endpoint), so headlines come from Yahoo Finance search, keeping only articles tagged with the stock. If that finds fewer than 5, Google News RSS fills the gap (common for Indian stocks). Only headlines that name the company are kept.

## Limitations

- **Accuracy is modest.** Predicting a stock 5 days ahead from its own price history is hard. The price model usually misses by about as much as guessing "no change", and the news verdict often fails to beat its baselines. The app shows this honestly.
- **Little news history.** Stage 2 learns from only ~2 years of headlines (a few hundred training days), so its results are noisy. Headline dates are UTC days, so news released after the market closes can count on the same day.
- **No fundamentals.** The models don't use earnings, valuation or macroeconomic data.
- **The backtest is simplified.** It ignores trading costs, taxes and slippage, and covers only the test period. Past results do not predict future results.
- **News coverage varies.** Some stocks, especially smaller Indian ones, may have few or no recent headlines, and keyword matching can occasionally let an off-topic story through.
- **FinBERT reads headlines only,** not full articles, and can misjudge tone (e.g. "dilemma" in a mostly positive headline).
- **Depends on free data sources** (Yahoo Finance, Google News) that can change or rate-limit without notice.
- **Not financial advice.** No buy/sell recommendations are made.
