# Stock Market Intelligence

A single-page Streamlit dashboard. Pick a stock (for example `AAPL`, or an Indian NSE stock such as `TCS.NS`), press **Analyze**, and the page shows:

1. **Price chart:** the last 6 months with 20-day and 50-day moving averages.
2. **Prediction:** the probability that the price is higher in 5 trading days (UP % / DOWN %).
3. **Why this prediction?:** which inputs the model relied on most (feature importance).
4. **News mood:** recent headlines scored by FinBERT, plus an overall mood score.
5. **Accuracy:** the model vs. "always guess UP" on the most recent year, which the model never saw.
6. **Backtest:** 100 following the model vs. 100 just holding the stock over that same year.

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

- The **first** Analyze downloads FinBERT (about 400 MB, once) and loads it into memory, which takes 30 to 60 seconds. After that, a new stock takes under about 10 seconds, and a repeat is instant (results are cached for an hour).
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
| Data | `src/data.py` | Downloads ~5 years of daily prices (yfinance) and recent headlines |
| Features | `src/features.py` | 8 inputs per day (returns, distance from moving averages, volatility, RSI, volume change) and the UP/DOWN label |
| Model | `src/model.py` | Random forest; trains on older data, tests on the last 252 trading days, then retrains on everything for today's prediction |
| Sentiment | `src/sentiment.py` | FinBERT headline scoring, with VADER as a fallback |
| Backtest | `src/backtest.py` | Invest for the next day only when P(UP) > 50%, otherwise hold cash |
| Page | `app.py` | Streamlit UI only; all logic lives in `src/` |

Settings such as the 5-day horizon and test-set size are in `src/config.py`.

**News sources:** `yf.Ticker(...).news` no longer works (Yahoo retired that endpoint), so headlines come from Yahoo Finance search, keeping only articles tagged with the stock. If that finds fewer than 5, Google News RSS fills the gap (common for Indian stocks). Only headlines that name the company are kept.

## Limitations

- **Accuracy is modest.** Predicting a stock 5 days ahead from its own price history is hard. The model is often only a few points better than guessing UP, and sometimes worse. The app shows this honestly.
- **Price data only.** The model doesn't use fundamentals (earnings, valuation), macroeconomic data, or the news. The news mood is shown alongside the prediction but is not an input to the model.
- **The backtest is simplified.** It ignores trading costs, taxes and slippage, and covers only one year. Past results do not predict future results.
- **News coverage varies.** Some stocks, especially smaller Indian ones, may have few or no recent headlines, and keyword matching can occasionally let an off-topic story through.
- **FinBERT reads headlines only,** not full articles, and can misjudge tone (e.g. "dilemma" in a mostly positive headline).
- **Depends on free data sources** (Yahoo Finance, Google News) that can change or rate-limit without notice.
- **Not financial advice.** No buy/sell recommendations are made.
