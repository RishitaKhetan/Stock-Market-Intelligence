"""Tests for src/sentiment.py. A fake analyzer is used, so no model download."""

import pandas as pd
import pytest

from src.sentiment import SCORED_HEADLINE_COLUMNS, load_vader_analyzer, summarize_news_mood, score_headline_sentiment


def fake_analyzer(titles):
    """Pretends to be FinBERT: fixed probabilities chosen by a keyword."""
    results = []
    for title in titles:
        if "beats" in title:
            probs = {"positive": 0.8, "negative": 0.1, "neutral": 0.1}
        elif "plunge" in title:
            probs = {"positive": 0.05, "negative": 0.9, "neutral": 0.05}
        else:
            probs = {"positive": 0.2, "negative": 0.1, "neutral": 0.7}
        results.append([{"label": k, "score": v} for k, v in probs.items()])
    return results


def scores_df(values):
    return pd.DataFrame({"score": values,
                         "label": ["positive" if v > 0 else "negative" for v in values]})


def test_score_headlines():
    titles = ["Apple beats estimates", "Shares plunge", "Annual meeting on Tuesday"]
    df = score_headline_sentiment(titles, fake_analyzer)

    assert list(df.columns) == SCORED_HEADLINE_COLUMNS
    assert list(df["title"]) == titles
    assert df["score"].tolist() == pytest.approx([0.7, -0.85, 0.1])
    assert list(df["label"]) == ["positive", "negative", "neutral"]


def test_score_headlines_accepts_capitalised_labels():
    def shouty(titles):
        return [[{"label": "Positive", "score": 0.6}, {"label": "Negative", "score": 0.3},
                 {"label": "Neutral", "score": 0.1}] for _ in titles]

    df = score_headline_sentiment(["x"], shouty)
    assert df.loc[0, "label"] == "positive"
    assert df.loc[0, "score"] == pytest.approx(0.3)


def test_score_headlines_empty():
    df = score_headline_sentiment([], fake_analyzer)
    assert df.empty and list(df.columns) == SCORED_HEADLINE_COLUMNS


def test_mood_summary_thresholds():
    assert summarize_news_mood(scores_df([0.5, 0.2]))["mood"] == "positive"     # mean 0.35
    assert summarize_news_mood(scores_df([-0.5, -0.2]))["mood"] == "negative"   # mean -0.35
    assert summarize_news_mood(scores_df([0.15]))["mood"] == "neutral"          # exactly at edge
    assert summarize_news_mood(scores_df([-0.15]))["mood"] == "neutral"
    assert summarize_news_mood(scores_df([0.16]))["mood"] == "positive"
    assert summarize_news_mood(scores_df([-0.16]))["mood"] == "negative"


def test_mood_summary_counts_and_mean():
    df = score_headline_sentiment(["Apple beats", "Stock plunge", "Meeting", "Meeting"], fake_analyzer)
    summary = summarize_news_mood(df)

    assert summary["counts"] == {"positive": 1, "negative": 1, "neutral": 2}
    assert summary["mean_score"] == pytest.approx((0.7 - 0.85 + 0.1 + 0.1) / 4)


def test_mood_summary_no_headlines():
    summary = summarize_news_mood(score_headline_sentiment([], fake_analyzer))
    assert summary["mean_score"] == 0.0 and summary["mood"] == "neutral"


def test_vader_fallback_has_finbert_format():
    df = score_headline_sentiment(["Profits soar to a record high", "Massive losses and layoffs"],
                         load_vader_analyzer())
    assert list(df["label"]) == ["positive", "negative"]
    sums = df[["positive", "negative", "neutral"]].sum(axis=1)
    assert sums.tolist() == pytest.approx([1.0, 1.0])


@pytest.mark.network
def test_real_finbert_positive_headline():
    from src.sentiment import load_finbert_pipeline
    df = score_headline_sentiment(["Company beats earnings expectations"], load_finbert_pipeline())
    assert df.loc[0, "label"] == "positive"


def test_vader_score_equals_compound():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    titles = ["Profits soar to a record high", "Annual meeting on Tuesday"]
    df = score_headline_sentiment(titles, load_vader_analyzer())
    compounds = [SentimentIntensityAnalyzer().polarity_scores(t)["compound"] for t in titles]
    assert df["score"].tolist() == pytest.approx(compounds)
    assert df.loc[1, "label"] == "neutral"


# --- Daily sentiment features ------------------------------------------------------

from src.sentiment import SENTIMENT_FEATURE_COLUMNS, daily_sentiment_features


def dated_news(rows):
    """rows: (date, score, label)"""
    return pd.DataFrame({"published": pd.to_datetime([r[0] for r in rows], utc=True),
                         "score": [r[1] for r in rows], "label": [r[2] for r in rows]})


def test_daily_sentiment_window_and_zeros():
    days = pd.bdate_range("2026-01-05", "2026-01-23")
    news = dated_news([("2026-01-06 15:00", 0.8, "positive"),
                       ("2026-01-07 09:00", -0.4, "negative")])
    f = daily_sentiment_features(news, days, window=7)

    assert list(f.columns) == SENTIMENT_FEATURE_COLUMNS and f.index.equals(days)
    assert f.loc["2026-01-05"].tolist() == [0, 0, 0, 0]             # before any news
    assert f.loc["2026-01-06", "news_count"] == 1
    assert f.loc["2026-01-08", "news_mean"] == pytest.approx(0.2)
    assert f.loc["2026-01-08", "news_positive_share"] == 0.5
    assert f.loc["2026-01-12", "news_count"] == 2                   # Jan 6 still in the 7 days
    assert f.loc["2026-01-13", "news_count"] == 1                   # Jan 6 dropped out
    assert f.loc["2026-01-23"].tolist() == [0, 0, 0, 0]             # news too old


def test_daily_sentiment_no_look_ahead():
    days = pd.bdate_range("2026-01-05", "2026-01-16")
    news = dated_news([("2026-01-06", 0.5, "positive")])
    tomorrow = dated_news([("2026-01-06", 0.5, "positive"), ("2026-01-09", -0.9, "negative")])
    before = daily_sentiment_features(news, days)
    after = daily_sentiment_features(tomorrow, days)
    # A headline dated Jan 9 must not change any day before Jan 9.
    assert before.loc[:"2026-01-08"].equals(after.loc[:"2026-01-08"])
    assert not before.loc["2026-01-09"].equals(after.loc["2026-01-09"])


def test_daily_sentiment_with_no_news_at_all():
    days = pd.bdate_range("2026-01-05", periods=5)
    f = daily_sentiment_features(dated_news([]), days)
    assert (f == 0).all().all() and len(f) == 5
