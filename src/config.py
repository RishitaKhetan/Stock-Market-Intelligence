"""Project-wide constants. Change values here, not inside the modules."""

HORIZON_DAYS = 5          # predict direction 5 trading days ahead
HISTORY_PERIOD = "5y"     # how much price history to download
TEST_DAYS = 252           # last ~1 year held out for testing
CHART_DAYS = 126          # ~6 months shown on the price chart
MA_SHORT = 20
MA_LONG = 50
RSI_WINDOW = 14
RANDOM_STATE = 42
FINBERT_MODEL = "ProsusAI/finbert"
MAX_HEADLINES = 15

# Suggestions for the ticker dropdown (symbol -> company name).
# Any other Yahoo Finance symbol can still be typed in by the user.
POPULAR_TICKERS = {
    # US stocks
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet (Google)",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
    "JPM": "JPMorgan Chase",
    "NFLX": "Netflix",
    # Indian stocks (NSE symbols end in .NS)
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "ITC.NS": "ITC",
    "LT.NS": "Larsen & Toubro",
}

# A headline counts as "about" a stock only if its title mentions the symbol
# (e.g. "TCS") or one of these names. Without an entry here, the first word
# of the company name is used (e.g. "Apple", "Infosys").
NEWS_KEYWORDS = {
    "GOOGL": ["Google", "Alphabet"],
    "JPM": ["JPMorgan", "JP Morgan"],
    "SBIN.NS": ["SBI", "State Bank of India"],
    "BHARTIARTL.NS": ["Airtel", "Bharti"],
    "LT.NS": ["L&T", "Larsen"],
}
