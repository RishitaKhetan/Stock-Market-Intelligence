"""Tests for src/data.py (and a sanity check on the config)."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import PRICE_COLUMN_NAMES, flatten_price_columns, load_price_history, normalize_ticker_symbol


def test_config_values():
    assert config.HORIZON_DAYS == 5


def test_normalize_ticker():
    assert normalize_ticker_symbol("  tcs.ns ") == "TCS.NS"


def test_flatten_columns_multiindex():
    # Mimic what new yfinance returns: ("Price", "Ticker") headers,
    # in a different order than we want.
    dates = pd.date_range("2024-01-01", periods=3)
    columns = pd.MultiIndex.from_tuples(
        [("Close", "AAPL"), ("High", "AAPL"), ("Low", "AAPL"),
         ("Open", "AAPL"), ("Volume", "AAPL")],
        names=["Price", "Ticker"],
    )
    raw = pd.DataFrame(np.arange(15).reshape(3, 5), index=dates, columns=columns)

    flat = flatten_price_columns(raw)

    assert list(flat.columns) == PRICE_COLUMN_NAMES
    # The values must stay attached to the right column.
    assert list(flat["Close"]) == [0, 5, 10]
    assert list(flat["Volume"]) == [4, 9, 14]


def test_flatten_columns_ticker_first():
    # Some versions put the ticker on the first level instead.
    dates = pd.date_range("2024-01-01", periods=2)
    columns = pd.MultiIndex.from_product([["AAPL"], PRICE_COLUMN_NAMES])
    raw = pd.DataFrame(np.ones((2, 5)), index=dates, columns=columns)

    assert list(flatten_price_columns(raw).columns) == PRICE_COLUMN_NAMES


def test_flatten_columns_already_flat():
    dates = pd.date_range("2024-01-01", periods=2)
    raw = pd.DataFrame(np.ones((2, 6)), index=dates,
                       columns=PRICE_COLUMN_NAMES + ["Dividends"])

    assert list(flatten_price_columns(raw).columns) == PRICE_COLUMN_NAMES


@pytest.mark.network
def test_load_prices_real_ticker():
    df = load_price_history("AAPL")
    assert len(df) > 1000
    assert list(df.columns) == PRICE_COLUMN_NAMES
    assert df.index.is_monotonic_increasing
    assert df["Close"].notna().all()


@pytest.mark.network
def test_load_prices_fake_ticker():
    with pytest.raises(ValueError):
        load_price_history("NOTAREALTICKER123")


@pytest.mark.network
def test_all_dropdown_tickers_load():
    # Catches symbols in the dropdown that have been renamed or delisted.
    failed = []
    for ticker in config.POPULAR_TICKERS:
        try:
            load_price_history(ticker, period="1mo")
        except ValueError:
            failed.append(ticker)
    assert failed == []


# --- News --------------------------------------------------------------------

import src.data as data_module
from src.data import is_tagged_with_ticker, load_news_headlines, parse_google_news_rss, parse_yahoo_news_item

NEW_FORMAT_ITEM = {
    "id": "abc",
    "content": {
        "title": "Apple unveils new iPad",
        "pubDate": "2026-10-20T14:30:00Z",
        "provider": {"displayName": "Reuters"},
        "canonicalUrl": {"url": "https://example.com/ipad"},
    },
}

OLD_FORMAT_ITEM = {
    "title": "Apple shares rise",
    "publisher": "CNBC",
    "link": "https://example.com/rise",
    "providerPublishTime": 1793037674,  # 2026-10-26 18:01 UTC
    "relatedTickers": ["AAPL"],
}

GOOGLE_RSS = """<?xml version="1.0"?>
<rss><channel>
  <item>
    <title>TCS share price surges 28% in October - Business Standard</title>
    <link>https://example.com/tcs-surge</link>
    <pubDate>Fri, 23 Oct 2026 07:00:00 GMT</pubDate>
    <source url="https://www.business-standard.com">Business Standard</source>
  </item>
  <item>
    <title>Tata Consultancy Services (NSEI:TCS) Stock Price - Simply Wall Street</title>
    <link>https://example.com/quote</link>
    <pubDate>Thu, 22 Oct 2026 07:00:00 GMT</pubDate>
    <source url="https://simplywall.st">Simply Wall Street</source>
  </item>
  <item>
    <title>HDFCBANK Share Price Today - HDFC Bank on NSE/BSE - scanx.trade</title>
    <link>https://example.com/quote2</link>
    <pubDate>Sun, 25 Oct 2026 07:00:00 GMT</pubDate>
    <source url="https://scanx.trade">scanx.trade</source>
  </item>
</channel></rss>"""


def test_parse_news_new_format():
    item = parse_yahoo_news_item(NEW_FORMAT_ITEM)
    assert item["title"] == "Apple unveils new iPad"
    assert item["publisher"] == "Reuters"
    assert item["link"] == "https://example.com/ipad"
    assert item["published"] == pd.Timestamp("2026-10-20 14:30", tz="UTC")


def test_parse_news_old_format():
    item = parse_yahoo_news_item(OLD_FORMAT_ITEM)
    assert item["title"] == "Apple shares rise"
    assert item["publisher"] == "CNBC"
    assert item["link"] == "https://example.com/rise"
    assert item["published"] == pd.Timestamp("2026-10-26 18:01:14", tz="UTC")


def test_parse_news_skips_items_without_title():
    assert parse_yahoo_news_item({"publisher": "CNBC"}) is None
    assert parse_yahoo_news_item({"content": {"title": "  "}}) is None


def test_is_about_ticker():
    assert is_tagged_with_ticker({"relatedTickers": ["GOOG", "AAPL"]}, "AAPL")
    assert not is_tagged_with_ticker({"relatedTickers": ["GOOG"]}, "AAPL")
    assert not is_tagged_with_ticker({}, "AAPL")
    # Same company on the other Indian exchange counts...
    assert is_tagged_with_ticker({"relatedTickers": ["TCS.BO"]}, "TCS.NS")
    # ...but a US company that happens to use the same letters does not.
    assert not is_tagged_with_ticker({"relatedTickers": ["TCS"]}, "TCS.NS")


def test_parse_google_rss_strips_publisher_and_skips_quote_pages():
    items = parse_google_news_rss(GOOGLE_RSS)
    assert [i["title"] for i in items] == ["TCS share price surges 28% in October"]
    assert items[0]["publisher"] == "Business Standard"
    assert items[0]["published"] == pd.Timestamp("2026-10-23 07:00", tz="UTC")


def fake_yahoo(n):
    """n old-format Yahoo items about AAPL, plus one unrelated item."""
    items = [dict(OLD_FORMAT_ITEM, title=f"Apple story {i}",
                  providerPublishTime=1793037674 + i) for i in range(n)]
    return items + [dict(OLD_FORMAT_ITEM, title="Unrelated", relatedTickers=["MSFT"])]


def test_load_news_uses_yahoo_when_enough(monkeypatch):
    monkeypatch.setattr(data_module, "fetch_yahoo_news_items", lambda q: fake_yahoo(20))
    called = []
    monkeypatch.setattr(data_module, "fetch_google_news_rss",
                        lambda q, india: called.append(q) or GOOGLE_RSS)

    news = load_news_headlines("AAPL", company_name="Apple")
    assert len(news) == config.MAX_HEADLINES
    assert "Unrelated" not in [n["title"] for n in news]
    assert news[0]["title"] == "Apple story 19"  # newest first
    assert called == []                          # Google not needed


def test_load_news_falls_back_to_google(monkeypatch):
    monkeypatch.setattr(data_module, "fetch_yahoo_news_items", lambda q: fake_yahoo(0))
    seen = {}
    def fake_google(query, india):
        seen.update(query=query, india=india)
        return GOOGLE_RSS
    monkeypatch.setattr(data_module, "fetch_google_news_rss", fake_google)

    news = load_news_headlines("tcs.ns", company_name="Tata Consultancy Services")
    assert [n["title"] for n in news] == ["TCS share price surges 28% in October"]
    assert "Tata Consultancy Services" in seen["query"] and seen["india"] is True


def test_load_news_returns_empty_list_on_errors(monkeypatch):
    def boom(*args):
        raise ConnectionError("no internet")
    monkeypatch.setattr(data_module, "fetch_yahoo_news_items", boom)
    monkeypatch.setattr(data_module, "fetch_google_news_rss", boom)
    assert load_news_headlines("AAPL", company_name="Apple") == []


@pytest.mark.network
@pytest.mark.parametrize("ticker", ["AAPL", "RELIANCE.NS"])
def test_load_news_real(ticker):
    news = load_news_headlines(ticker)
    assert 0 < len(news) <= config.MAX_HEADLINES
    assert set(news[0]) == {"title", "publisher", "link", "published"}


def test_mentions_company():
    from src.data import title_mentions_company, build_news_keywords

    apple = build_news_keywords("AAPL", "Apple")
    assert title_mentions_company("Apple unveils new iPad", apple)
    assert title_mentions_company("Why AAPL stock fell today", apple)
    assert not title_mentions_company("Tesla leaves the Magnificent 7 in the dust", apple)
    assert not title_mentions_company("Pineapple prices rise", apple)  # whole words only

    sbi = build_news_keywords("SBIN.NS", "State Bank of India")  # uses NEWS_KEYWORDS aliases
    assert title_mentions_company("SBI shares hit record high", sbi)
    assert not title_mentions_company("ICICI Bank share price prediction", sbi)


def test_load_news_drops_headlines_not_about_the_company(monkeypatch):
    items = fake_yahoo(10) + [dict(OLD_FORMAT_ITEM, title="Silver is down 17% in a month")]
    monkeypatch.setattr(data_module, "fetch_yahoo_news_items", lambda q: items)
    monkeypatch.setattr(data_module, "fetch_google_news_rss", lambda q, india: GOOGLE_RSS)

    titles = [n["title"] for n in load_news_headlines("AAPL", company_name="Apple")]
    assert "Silver is down 17% in a month" not in titles
    assert len(titles) == 10


# --- News history ---------------------------------------------------------------

def fake_score_titles(titles):
    return pd.DataFrame({"title": titles, "positive": 0.7, "negative": 0.1, "neutral": 0.2,
                         "score": 0.6, "label": "positive"})


def one_headline_rss(title, day):
    return f"""<rss><channel><item><title>{title} - Reuters</title><link>https://x.com/{day}</link>
    <pubDate>{pd.Timestamp(day):%a, %d %b %Y} 12:00:00 GMT</pubDate><source>Reuters</source></item>
    </channel></rss>"""


def test_load_news_history_builds_dated_queries_and_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(data_module, "NEWS_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(data_module, "SECONDS_BETWEEN_REQUESTS", 0)
    queries = []

    def fake_google(query, india):
        queries.append(query)
        after = query.split("after:")[1][:10]
        return one_headline_rss(f"Apple news {after}", after)
    monkeypatch.setattr(data_module, "fetch_google_news_rss", fake_google)

    history = data_module.load_news_history("AAPL", fake_score_titles, days=28, company_name="Apple")

    assert len(queries) == 4
    assert all("after:" in q and "before:" in q and '"Apple"' in q for q in queries)
    after, before = queries[0].split("after:")[1].split(" before:")
    assert pd.Timestamp(before) - pd.Timestamp(after) == pd.Timedelta(days=7)
    assert len(history) == 4 and (history["score"] == 0.6).all()
    assert history["published"].is_monotonic_increasing
    assert (tmp_path / "AAPL.csv").exists()

    # Second run: only the newest (still filling) week is fetched again.
    queries.clear()
    again = data_module.load_news_history("AAPL", fake_score_titles, days=28, company_name="Apple")
    assert len(queries) == 1
    assert again["title"].tolist() == history["title"].tolist()
    assert again["published"].equals(history["published"])


def test_load_news_history_skips_failed_weeks_and_off_topic(monkeypatch, tmp_path):
    monkeypatch.setattr(data_module, "NEWS_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(data_module, "SECONDS_BETWEEN_REQUESTS", 0)
    calls = []

    def flaky_google(query, india):
        calls.append(query)
        if len(calls) == 1:
            raise ConnectionError("blocked")
        return one_headline_rss("Silver is down 17%", "2026-01-05")
    monkeypatch.setattr(data_module, "fetch_google_news_rss", flaky_google)

    history = data_module.load_news_history("AAPL", fake_score_titles, days=21, company_name="Apple")
    assert len(calls) == 3 and history.empty
