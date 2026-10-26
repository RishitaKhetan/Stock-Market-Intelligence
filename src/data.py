"""Downloading data: daily stock prices and recent news headlines."""

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import pandas as pd
import yfinance as yf

from src.config import HISTORY_PERIOD, MAX_HEADLINES, NEWS_KEYWORDS, POPULAR_TICKERS

# The only columns the rest of the project uses, in this order.
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# If Yahoo finds fewer headlines than this, top up from Google News.
MIN_YAHOO_HEADLINES = 5

# Google News also returns "live share price" pages. They aren't news, so skip
# titles like "HDFC Bank Share Price Today" or "TCS (NSEI:TCS) Stock Price".
QUOTE_PAGE = re.compile(r"\b(share|stock) price\b\s*(today|live|-|&|$)", re.IGNORECASE)


def normalize_ticker(ticker: str) -> str:
    """Clean up what the user typed: ' tcs.ns ' -> 'TCS.NS'."""
    return ticker.strip().upper()


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    df = df[PRICE_COLUMNS]
    df.columns.name = None  # drop the leftover "Price" header label
    return df


def load_prices(ticker: str, period: str = HISTORY_PERIOD) -> pd.DataFrame:
    """Download daily prices for one ticker.

    Returns a DataFrame indexed by date (oldest first) with columns
    Open, High, Low, Close, Volume. Raises ValueError if nothing comes back.
    """
    ticker = normalize_ticker(ticker)
    if not ticker:
        raise ValueError("Please enter a ticker symbol.")

    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"No price data found for {ticker}")

    df = flatten_columns(raw)
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


def parse_news_item(item: dict):
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
        seconds = item.get("providerPublishTime")
        published = pd.to_datetime(seconds, unit="s", utc=True) if seconds else pd.NaT

    if not title or not str(title).strip():
        return None
    return {
        "title": str(title).strip(),
        "publisher": publisher or "",
        "link": link or "",
        "published": published,
    }


def is_about_ticker(item: dict, ticker: str) -> bool:
    """True if Yahoo tagged this headline with our stock.

    Accepts the exact symbol, or the same company on another Indian exchange
    (TCS.NS also matches TCS.BO), but not an unrelated US "TCS".
    """
    base = ticker.split(".")[0]
    for related in item.get("relatedTickers") or []:
        if related == ticker or ("." in related and related.split(".")[0] == base):
            return True
    return False


def parse_google_rss(xml_text: str) -> list[dict]:
    """Read a Google News RSS feed into our standard dicts."""
    headlines = []
    for item in ET.fromstring(xml_text).findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        publisher = (item.findtext("source") or "").strip()
        # Google adds " - Publisher" to the end of every title; remove it.
        if publisher and title.endswith(f" - {publisher}"):
            title = title[: -len(f" - {publisher}")].strip()
        if not title or QUOTE_PAGE.search(title):
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


def fetch_yahoo_items(query: str) -> list[dict]:
    """Raw news items from Yahoo Finance search (network call)."""
    return yf.Search(query, news_count=25, max_results=1).news


def fetch_google_rss(query: str, india: bool) -> str:
    """Raw Google News RSS text for a search query (network call)."""
    region = {"hl": "en-IN", "gl": "IN", "ceid": "IN:en"} if india else \
             {"hl": "en-US", "gl": "US", "ceid": "US:en"}
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode({"q": query, **region})
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read().decode("utf-8")


def get_company_name(ticker: str) -> str:
    """A company name for searching Google News (falls back to the symbol)."""
    if ticker in POPULAR_TICKERS:
        return POPULAR_TICKERS[ticker]
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName") or ticker.split(".")[0]
    except Exception:
        return ticker.split(".")[0]


def news_keywords(ticker: str, company_name: str) -> list[str]:
    """Words a headline must contain to count as being about this stock."""
    base = ticker.split(".")[0]
    names = NEWS_KEYWORDS.get(ticker) or [company_name.split()[0]]
    return [base] + names


def mentions_company(title: str, keywords: list[str]) -> bool:
    """True if the title contains any keyword as a whole word ("TCS", not "TCSx")."""
    return any(re.search(rf"(?<!\w){re.escape(k)}(?!\w)", title, re.IGNORECASE)
               for k in keywords)


def yahoo_news(ticker: str) -> list[dict]:
    """Headlines Yahoo tags with this stock. [] on any error."""
    try:
        items = fetch_yahoo_items(ticker.split(".")[0])  # "TCS.NS" finds nothing; "TCS" works
    except Exception:
        return []
    parsed = [parse_news_item(item) for item in items if is_about_ticker(item, ticker)]
    return [p for p in parsed if p]


def google_news(company_name: str, ticker: str) -> list[dict]:
    """Last week's Google News headlines about the company. [] on any error."""
    india = ticker.endswith((".NS", ".BO"))
    try:
        return parse_google_rss(fetch_google_rss(f'"{company_name}" stock when:7d', india))
    except Exception:
        return []


def load_news(ticker: str, company_name: str | None = None) -> list[dict]:
    """Up to MAX_HEADLINES recent headlines about a stock, newest first.

    Each item is a dict with keys title, publisher, link, published.
    Returns [] if nothing can be found (never raises).
    """
    try:
        ticker = normalize_ticker(ticker)
        company_name = company_name or get_company_name(ticker)
        keywords = news_keywords(ticker, company_name)

        def relevant(items):
            # Drop market-wide stories that only mention our stock in passing.
            return [h for h in items if mentions_company(h["title"], keywords)]

        headlines = relevant(yahoo_news(ticker))
        if len(headlines) < MIN_YAHOO_HEADLINES:
            headlines += relevant(google_news(company_name, ticker))

        # Drop repeats (same story from both sources), then newest first.
        unique = {}
        for h in headlines:
            unique.setdefault(h["title"].lower(), h)
        oldest = pd.Timestamp.min.tz_localize("UTC")
        ordered = sorted(unique.values(),
                         key=lambda h: h["published"] if pd.notna(h["published"]) else oldest,
                         reverse=True)
        return ordered[:MAX_HEADLINES]
    except Exception:
        return []
