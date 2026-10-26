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
