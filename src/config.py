"""Project-wide constants. Change values here, not inside the modules."""

HORIZON_DAYS = 5          # predict the price 5 trading days ahead
HISTORY_PERIOD = "5y"     # how much price history to download
TEST_DAYS = 252           # last ~1 year held out for testing
CHART_DAYS = 126          # ~6 months shown on the price chart
MA_SHORT = 20
MA_LONG = 50
RSI_WINDOW = 14
RANDOM_STATE = 42
FINBERT_MODEL = "ProsusAI/finbert"
MAX_HEADLINES = 15
NEWS_HISTORY_DAYS = 730     # ~2 years of dated news for training stage 2
SENTIMENT_WINDOW_DAYS = 7   # each day's news mood uses the last 7 calendar days
STAGE2_TEST_DAYS = 126      # last ~6 months of stage 2 held out for testing
NEWS_CACHE_DIR = "news_cache"  # scored headlines saved here, one CSV per ticker

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
