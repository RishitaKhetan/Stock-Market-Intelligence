"""Scoring news headlines as positive / negative / neutral.

Main engine: FinBERT, a language model trained on financial text.
Fallback: VADER, a simple word-list scorer, used if FinBERT can't load
(no internet on first run, or not enough memory).

An "analyzer" here is any function that takes a list of headlines and returns,
for each one, a list like [{"label": "positive", "score": 0.9}, ...].
That is FinBERT's output format; the VADER fallback copies it.
"""

import pandas as pd

from src.config import FINBERT_MODEL, SENTIMENT_WINDOW_DAYS

SENTIMENT_LABELS = ["positive", "negative", "neutral"]
SCORED_HEADLINE_COLUMNS = ["title", "positive", "negative", "neutral", "score", "label"]
MOOD_THRESHOLD = 0.15  # mean score above +0.15 = positive, below -0.15 = negative
VADER_NEUTRAL_BAND = 0.05  # VADER's usual cut-off for calling a text neutral


def load_finbert_pipeline():
    """FinBERT as a Hugging Face pipeline (downloads ~400 MB the first time)."""
    from transformers import pipeline  # imported here: slow, and only needed now
    return pipeline("text-classification", model=FINBERT_MODEL, top_k=None)


def load_vader_analyzer():
    """VADER wrapped to return the same format as FinBERT."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_scorer = SentimentIntensityAnalyzer()

    def analyze_titles(titles):
        results = []
        for title in titles:
            # compound runs from -1 (very negative) to +1 (very positive).
            # VADER's standard rule: within +/-0.05 is neutral.
            compound = vader_scorer.polarity_scores(title)["compound"]
            if abs(compound) < VADER_NEUTRAL_BAND:
                positive, negative, neutral = max(compound, 0.0), max(-compound, 0.0), 1.0 - abs(compound)
            else:
                positive, negative, neutral = (1 + compound) / 2, (1 - compound) / 2, 0.0
            # Either way pos - neg == c, so the score is VADER's own number.
            results.append([
                {"label": "positive", "score": positive},
                {"label": "negative", "score": negative},
                {"label": "neutral", "score": neutral},
            ])
        return results

    return analyze_titles


def load_sentiment_analyzer():
    """Try FinBERT, fall back to VADER. Returns (analyzer, engine_name)."""
    try:
        return load_finbert_pipeline(), "FinBERT"
    except Exception:
        return load_vader_analyzer(), "VADER"


def score_headline_sentiment(titles: list[str], analyzer) -> pd.DataFrame:
    """Score each headline.

    Columns: title, positive, negative, neutral (probabilities),
    score = positive - negative (from -1 to +1), and label = the most likely of the three.
    """
    if not titles:
        return pd.DataFrame(columns=SCORED_HEADLINE_COLUMNS)

    rows = []
    for title, result in zip(titles, analyzer(list(titles))):
        label_probabilities = {prediction["label"].lower(): float(prediction["score"]) for prediction in result}
        row = {"title": title, **{label: label_probabilities.get(label, 0.0) for label in SENTIMENT_LABELS}}
        row["score"] = row["positive"] - row["negative"]
        row["label"] = max(SENTIMENT_LABELS, key=lambda label: row[label])
        rows.append(row)
    return pd.DataFrame(rows, columns=SCORED_HEADLINE_COLUMNS)


def summarize_news_mood(df: pd.DataFrame) -> dict:
    """Overall mood: the average score, how many of each label, and one word."""
    mean_score = float(df["score"].mean()) if len(df) else 0.0
    if mean_score > MOOD_THRESHOLD:
        mood = "positive"
    elif mean_score < -MOOD_THRESHOLD:
        mood = "negative"
    else:
        mood = "neutral"
    counts = {label: int((df["label"] == label).sum()) for label in SENTIMENT_LABELS}
    return {"mean_score": mean_score, "counts": counts, "mood": mood}


SENTIMENT_FEATURE_COLUMNS = ["news_mean", "news_count", "news_positive_share", "news_negative_share"]


def daily_sentiment_features(scored_news: pd.DataFrame, trading_days: pd.DatetimeIndex,
                             window: int = SENTIMENT_WINDOW_DAYS) -> pd.DataFrame:
    """News mood on each trading day, from headlines of the last `window` calendar days.

    Day t uses only headlines dated t or earlier (no peeking at tomorrow's news).
    Days with no headlines get zeros. scored_news needs published, score, label.
    """
    news_day = pd.to_datetime(scored_news["published"], utc=True).dt.tz_localize(None).dt.normalize()
    daily = pd.DataFrame({
        "total": scored_news["score"].astype(float).values,
        "count": 1.0,
        "positive": (scored_news["label"] == "positive").astype(float).values,
        "negative": (scored_news["label"] == "negative").astype(float).values,
    }, index=news_day.values).groupby(level=0).sum()

    # Every calendar day, so the rolling window counts calendar days.
    all_days = pd.date_range(min(daily.index.min(), trading_days.min()) if len(daily) else trading_days.min(),
                             trading_days.max())
    rolled = daily.reindex(all_days, fill_value=0.0).rolling(window, min_periods=1).sum()
    rolled = rolled.reindex(trading_days.normalize()).fillna(0.0)

    count = rolled["count"]
    has_news = count > 0
    features = pd.DataFrame({
        "news_mean": (rolled["total"] / count).where(has_news, 0.0),
        "news_count": count,
        "news_positive_share": (rolled["positive"] / count).where(has_news, 0.0),
        "news_negative_share": (rolled["negative"] / count).where(has_news, 0.0),
    })
    features.index = trading_days
    return features[SENTIMENT_FEATURE_COLUMNS]
