# Stock Market Intelligence

A single-page Streamlit dashboard: type a stock ticker (e.g. `AAPL`, or an NSE stock like `TCS.NS`), press **Analyze**, and see a price chart, a 5-day UP/DOWN probability, news sentiment, model accuracy and a simple backtest.

Educational demo, not financial advice.

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

If you edit files in `src/` while the app is running and see an `ImportError`, stop the server (Ctrl+C) and run `streamlit run app.py` again.

## Run the tests

```bash
pytest -q                    # all tests
pytest -q -m "not network"   # offline tests only
```
