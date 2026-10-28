"""Scoring news headlines as positive / negative / neutral.

Main engine: FinBERT, a language model trained on financial text.
Fallback: VADER, a simple word-list scorer, used if FinBERT can't load
(no internet on first run, or not enough memory).

An "analyzer" here is any function that takes a list of headlines and returns,
for each one, a list like [{"label": "positive", "score": 0.9}, ...].
That is FinBERT's output format; the VADER fallback copies it.
"""

import pandas as pd

from src.config import FINBERT_MODEL

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
