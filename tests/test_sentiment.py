"""Tests for src/sentiment.py. A fake analyzer is used, so no model download."""

import pandas as pd
import pytest

from src.sentiment import SCORE_COLUMNS, load_vader, mood_summary, score_headlines


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
    df = score_headlines(titles, fake_analyzer)

    assert list(df.columns) == SCORE_COLUMNS
    assert list(df["title"]) == titles
    assert df["score"].tolist() == pytest.approx([0.7, -0.85, 0.1])
    assert list(df["label"]) == ["positive", "negative", "neutral"]


def test_score_headlines_accepts_capitalised_labels():
    def shouty(titles):
        return [[{"label": "Positive", "score": 0.6}, {"label": "Negative", "score": 0.3},
                 {"label": "Neutral", "score": 0.1}] for _ in titles]

    df = score_headlines(["x"], shouty)
    assert df.loc[0, "label"] == "positive"
    assert df.loc[0, "score"] == pytest.approx(0.3)


def test_score_headlines_empty():
    df = score_headlines([], fake_analyzer)
    assert df.empty and list(df.columns) == SCORE_COLUMNS


def test_mood_summary_thresholds():
    assert mood_summary(scores_df([0.5, 0.2]))["mood"] == "positive"     # mean 0.35
    assert mood_summary(scores_df([-0.5, -0.2]))["mood"] == "negative"   # mean -0.35
    assert mood_summary(scores_df([0.15]))["mood"] == "neutral"          # exactly at edge
    assert mood_summary(scores_df([-0.15]))["mood"] == "neutral"
    assert mood_summary(scores_df([0.16]))["mood"] == "positive"
    assert mood_summary(scores_df([-0.16]))["mood"] == "negative"


def test_mood_summary_counts_and_mean():
    df = score_headlines(["Apple beats", "Stock plunge", "Meeting", "Meeting"], fake_analyzer)
    summary = mood_summary(df)

    assert summary["counts"] == {"positive": 1, "negative": 1, "neutral": 2}
    assert summary["mean_score"] == pytest.approx((0.7 - 0.85 + 0.1 + 0.1) / 4)


def test_mood_summary_no_headlines():
    summary = mood_summary(score_headlines([], fake_analyzer))
    assert summary["mean_score"] == 0.0 and summary["mood"] == "neutral"


def test_vader_fallback_has_finbert_format():
    df = score_headlines(["Profits soar to a record high", "Massive losses and layoffs"],
                         load_vader())
    assert list(df["label"]) == ["positive", "negative"]
    sums = df[["positive", "negative", "neutral"]].sum(axis=1)
    assert sums.tolist() == pytest.approx([1.0, 1.0])


@pytest.mark.network
def test_real_finbert_positive_headline():
    from src.sentiment import load_finbert
    df = score_headlines(["Company beats earnings expectations"], load_finbert())
    assert df.loc[0, "label"] == "positive"


def test_vader_score_equals_compound():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    titles = ["Profits soar to a record high", "Annual meeting on Tuesday"]
    df = score_headlines(titles, load_vader())
    compounds = [SentimentIntensityAnalyzer().polarity_scores(t)["compound"] for t in titles]
    assert df["score"].tolist() == pytest.approx(compounds)
    assert df.loc[1, "label"] == "neutral"
