"""Downloading data: daily stock prices and recent news headlines."""

import re
import time
from pathlib import Path
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import pandas as pd
import yfinance as yf

from src.config import (HISTORY_PERIOD, MAX_HEADLINES, NEWS_CACHE_DIR, NEWS_HISTORY_DAYS, NEWS_KEYWORDS,
                        POPULAR_TICKERS)

# The only columns the rest of the project uses, in this order.
PRICE_COLUMN_NAMES = ["Open", "High", "Low", "Close", "Volume"]

# If Yahoo finds fewer headlines than this, top up from Google News.
MIN_YAHOO_HEADLINE_COUNT = 5

# Google News also returns "live share price" pages. They aren't news, so skip
# titles like "HDFC Bank Share Price Today" or "TCS (NSEI:TCS) Stock Price".
QUOTE_PAGE_PATTERN = re.compile(r"\b(share|stock) price\b\s*(today|live|-|&|$)", re.IGNORECASE)


def normalize_ticker_symbol(ticker: str) -> str:
    """Clean up what the user typed: ' tcs.ns ' -> 'TCS.NS'."""
    return ticker.strip().upper()


def flatten_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Turn yfinance's two-level column headers into plain names.

    Recent yfinance versions return columns like ("Close", "AAPL") even when
    we ask for a single ticker. We keep only the level holding the price
    names ("Open", "Close", ...), then select our five columns in a fixed order.
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # Find which header level contains "Close" and keep just that one.
        for level in range(df.columns.nlevels):
            if "Close" in df.columns.get_level_values(level):
                df.columns = df.columns.get_level_values(level)
                break
    df = df[PRICE_COLUMN_NAMES]
    df.columns.name = None  # drop the leftover "Price" header label
    return df


def load_price_history(ticker: str, period: str = HISTORY_PERIOD) -> pd.DataFrame:
    """Download daily prices for one ticker.

    Returns a DataFrame indexed by date (oldest first) with columns
    Open, High, Low, Close, Volume. Raises ValueError if nothing comes back.
    """
    ticker = normalize_ticker_symbol(ticker)
    if not ticker:
        raise ValueError("Please enter a ticker symbol.")

    raw_download = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if raw_download is None or raw_download.empty:
        raise ValueError(f"No price data found for {ticker}")

    df = flatten_price_columns(raw_download)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.dropna(subset=["Close"])

    if df.empty:
        raise ValueError(f"No price data found for {ticker}")
    return df


# --- News --------------------------------------------------------------------
#
# The plan's source, yf.Ticker(ticker).news, stopped working: Yahoo retired the
# web address behind it (it returns 404), so it is always empty. Instead we use
#   1) yfinance's yf.Search, keeping only headlines Yahoo tags with this stock;
#   2) Google News RSS by company name, when Yahoo finds too few (common for
#      Indian stocks).
# Every headline becomes a dict: title, publisher, link, published.


def parse_yahoo_news_item(item: dict):
    """Turn one Yahoo news item into our standard dict, or None if it has no title.

    Yahoo has used two formats over the years:
      new: fields nested under item["content"] (title, pubDate,
           provider.displayName, canonicalUrl.url)
      old: top-level title, publisher, link, providerPublishTime (Unix seconds)
    """
    if isinstance(item.get("content"), dict):
        content = item["content"]
        title = content.get("title")
        publisher = (content.get("provider") or {}).get("displayName")
        link = (content.get("canonicalUrl") or {}).get("url")
        published = pd.to_datetime(content.get("pubDate"), utc=True, errors="coerce")
    else:
        title = item.get("title")
        publisher = item.get("publisher")
        link = item.get("link")
        publish_seconds = item.get("providerPublishTime")
        published = pd.to_datetime(publish_seconds, unit="s", utc=True) if publish_seconds else pd.NaT

    if not title or not str(title).strip():
        return None
    return {
        "title": str(title).strip(),
        "publisher": publisher or "",
        "link": link or "",
        "published": published,
    }


def is_tagged_with_ticker(item: dict, ticker: str) -> bool:
    """True if Yahoo tagged this headline with our stock.

    Accepts the exact symbol, or the same company on another Indian exchange
    (TCS.NS also matches TCS.BO), but not an unrelated US "TCS".
    """
    base_symbol = ticker.split(".")[0]
    for related_symbol in item.get("relatedTickers") or []:
        if related_symbol == ticker or ("." in related_symbol and related_symbol.split(".")[0] == base_symbol):
            return True
    return False


def parse_google_news_rss(xml_text: str) -> list[dict]:
    """Read a Google News RSS feed into our standard dicts."""
    headlines = []
    for item in ET.fromstring(xml_text).findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        publisher = (item.findtext("source") or "").strip()
        # Google adds " - Publisher" to the end of every title; remove it.
        if publisher and title.endswith(f" - {publisher}"):
            title = title[: -len(f" - {publisher}")].strip()
        if not title or QUOTE_PAGE_PATTERN.search(title):
            continue
        try:
            published = pd.Timestamp(parsedate_to_datetime(item.findtext("pubDate")))
        except (TypeError, ValueError):
            published = pd.NaT
        headlines.append({
            "title": title,
            "publisher": publisher,
            "link": (item.findtext("link") or "").strip(),
            "published": published,
        })
    return headlines


def fetch_yahoo_news_items(query: str) -> list[dict]:
    """Raw news items from Yahoo Finance search (network call)."""
    return yf.Search(query, news_count=25, max_results=1).news


def fetch_google_news_rss(query: str, is_indian_stock: bool) -> str:
    """Raw Google News RSS text for a search query (network call)."""
    region_params = {"hl": "en-IN", "gl": "IN", "ceid": "IN:en"} if is_indian_stock else \
                    {"hl": "en-US", "gl": "US", "ceid": "US:en"}
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode({"q": query, **region_params})
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read().decode("utf-8")


def lookup_company_name(ticker: str) -> str:
    """A company name for searching Google News (falls back to the symbol)."""
    if ticker in POPULAR_TICKERS:
        return POPULAR_TICKERS[ticker]
    try:
        ticker_info = yf.Ticker(ticker).info
        return ticker_info.get("shortName") or ticker_info.get("longName") or ticker.split(".")[0]
    except Exception:
        return ticker.split(".")[0]


def build_news_keywords(ticker: str, company_name: str) -> list[str]:
    """Words a headline must contain to count as being about this stock."""
    base_symbol = ticker.split(".")[0]
    company_names = NEWS_KEYWORDS.get(ticker) or [company_name.split()[0]]
    return [base_symbol] + company_names


def title_mentions_company(title: str, keywords: list[str]) -> bool:
    """True if the title contains any keyword as a whole word ("TCS", not "TCSx")."""
    return any(re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", title, re.IGNORECASE)
               for keyword in keywords)


def fetch_yahoo_headlines(ticker: str) -> list[dict]:
    """Headlines Yahoo tags with this stock. [] on any error."""
    try:
        raw_items = fetch_yahoo_news_items(ticker.split(".")[0])  # "TCS.NS" finds nothing; "TCS" works
    except Exception:
        return []
    parsed_items = [parse_yahoo_news_item(item) for item in raw_items if is_tagged_with_ticker(item, ticker)]
    return [p for p in parsed_items if p]


def fetch_google_headlines(company_name: str, ticker: str) -> list[dict]:
    """Last week's Google News headlines about the company. [] on any error."""
    is_indian_stock = ticker.endswith((".NS", ".BO"))
    try:
        return parse_google_news_rss(fetch_google_news_rss(f'"{company_name}" stock when:7d', is_indian_stock))
    except Exception:
        return []


def load_news_headlines(ticker: str, company_name: str | None = None) -> list[dict]:
    """Up to MAX_HEADLINES recent headlines about a stock, newest first.

    Each item is a dict with keys title, publisher, link, published.
    Returns [] if nothing can be found (never raises).
    """
    try:
        ticker = normalize_ticker_symbol(ticker)
        company_name = company_name or lookup_company_name(ticker)
        keywords = build_news_keywords(ticker, company_name)

        def keep_relevant_headlines(candidate_headlines):
            # Drop market-wide stories that only mention our stock in passing.
            return [headline for headline in candidate_headlines
                    if title_mentions_company(headline["title"], keywords)]

        headlines = keep_relevant_headlines(fetch_yahoo_headlines(ticker))
        if len(headlines) < MIN_YAHOO_HEADLINE_COUNT:
            headlines += keep_relevant_headlines(fetch_google_headlines(company_name, ticker))

        # Drop repeats (same story from both sources), then newest first.
        unique_by_title = {}
        for headline in headlines:
            unique_by_title.setdefault(headline["title"].lower(), headline)
        oldest_possible_time = pd.Timestamp.min.tz_localize("UTC")
        newest_first = sorted(unique_by_title.values(),
                              key=lambda headline: headline["published"]
                              if pd.notna(headline["published"]) else oldest_possible_time,
                              reverse=True)
        return newest_first[:MAX_HEADLINES]
    except Exception:
        return []


# --- News history (for training stage 2) ---------------------------------------
#
# Google News search accepts "after:YYYY-MM-DD before:YYYY-MM-DD", so we ask for
# one week at a time going back NEWS_HISTORY_DAYS. Headlines are scored once and
# saved to news_cache/<TICKER>.csv; later runs only fetch weeks not saved yet.

NEWS_HISTORY_COLUMNS = ["week", "title", "publisher", "link", "published",
                        "positive", "negative", "neutral", "score", "label"]
SECONDS_BETWEEN_REQUESTS = 1.0  # be polite to Google


def news_cache_path(ticker: str) -> Path:
    return Path(NEWS_CACHE_DIR) / f"{ticker.replace('/', '_')}.csv"


def fetch_google_week(company_name: str, ticker: str, week_start: pd.Timestamp) -> list[dict]:
    """Google News headlines about the company published in one week (network call)."""
    week_end = week_start + pd.Timedelta(days=7)
    query = f'"{company_name}" stock after:{week_start:%Y-%m-%d} before:{week_end:%Y-%m-%d}'
    return parse_google_news_rss(fetch_google_news_rss(query, ticker.endswith((".NS", ".BO"))))


def load_news_history(ticker: str, score_titles, days: int = NEWS_HISTORY_DAYS,
                      company_name: str | None = None) -> pd.DataFrame:
    """About `days` days of scored headlines about a stock, oldest first.

    score_titles: function(list of titles) -> DataFrame with positive, negative,
    neutral, score, label (e.g. sentiment.score_headline_sentiment with an analyzer).
    Weeks that fail to download are skipped and retried on the next run.
    """
    ticker = normalize_ticker_symbol(ticker)
    company_name = company_name or lookup_company_name(ticker)
    keywords = build_news_keywords(ticker, company_name)
    path = news_cache_path(ticker)

    cached = pd.read_csv(path, parse_dates=["week"]) if path.exists() else \
        pd.DataFrame(columns=NEWS_HISTORY_COLUMNS)
    cached["published"] = pd.to_datetime(cached["published"], utc=True, format="ISO8601")

    today = pd.Timestamp.now().normalize()
    weeks = pd.date_range(end=today, periods=days // 7, freq="W-MON")
    # The newest week is still filling up, so always fetch it again.
    # ponytail: weeks with no headlines leave no row, so they are re-fetched every run.
    cached = cached[cached["week"] != weeks[-1]]
    done = set(cached["week"])

    new_rows = []
    for week_start in weeks:
        if week_start in done:
            continue
        try:
            headlines = fetch_google_week(company_name, ticker, week_start)
        except Exception:
            continue
        for headline in headlines:
            if title_mentions_company(headline["title"], keywords):
                new_rows.append({"week": week_start, **headline})
        time.sleep(SECONDS_BETWEEN_REQUESTS)

    if new_rows:
        new = pd.DataFrame(new_rows).drop_duplicates("title")
        new["published"] = pd.to_datetime(new["published"], utc=True)
        scores = score_titles(new["title"].tolist())
        new = pd.concat([new.reset_index(drop=True),
                         scores.drop(columns="title").reset_index(drop=True)], axis=1)
        cached = pd.concat([cached, new[NEWS_HISTORY_COLUMNS]], ignore_index=True)

    history = cached.drop_duplicates("title").sort_values("published", ignore_index=True)
    path.parent.mkdir(exist_ok=True)
    history.to_csv(path, index=False)
    return history
