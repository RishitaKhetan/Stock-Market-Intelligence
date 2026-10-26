"""Training the model, measuring it honestly, and explaining its prediction."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_STATE, TEST_DAYS
from src.features import FEATURE_LABELS


def time_split(X: pd.DataFrame, y: pd.Series, test_days: int = TEST_DAYS):
    """Split by time: the last `test_days` rows are the test set.

    Never shuffle stock data. The model must be tested on days that come
    after everything it trained on, just like in real life.
    """
    if len(X) <= test_days:
        raise ValueError(
            f"Not enough price history: need more than {test_days} usable days, "
            f"found {len(X)}."
        )
    X_train, X_test = X.iloc[:-test_days], X.iloc[-test_days:]
    y_train, y_test = y.iloc[:-test_days], y.iloc[-test_days:]
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Train a random forest: many small decision trees that vote.

    Shallow trees (max_depth=5) with at least 20 days per leaf stop the model
    from memorising noise in the training data.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,  # use all CPU cores; same result, just faster
    )
    model.fit(X_train, y_train)
    return model


def predict_proba_up(model: RandomForestClassifier, X: pd.DataFrame) -> pd.Series:
    """Probability of UP (class 1) for every row of X."""
    classes = list(model.classes_)
    if 1 not in classes:  # training data had no UP days at all
        return pd.Series(0.0, index=X.index)
    return pd.Series(model.predict_proba(X)[:, classes.index(1)], index=X.index)


def evaluate(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Score the model on unseen test days and compare with a naive baseline.

    The baseline "always guess UP" is right on every UP day, so its accuracy
    is simply the share of UP days. A useful model has to beat it.
    """
    predictions = (predict_proba_up(model, X_test) > 0.5).astype(int)
    return {
        "model_accuracy": float((predictions == y_test).mean()),
        "baseline_accuracy": float(y_test.mean()),
        "test_start": X_test.index[0],
        "test_end": X_test.index[-1],
    }


def predict_latest(model: RandomForestClassifier, latest_row: pd.DataFrame) -> float:
    """Probability that the price is higher in HORIZON_DAYS trading days."""
    return float(predict_proba_up(model, latest_row).iloc[-1])


def feature_importance(model: RandomForestClassifier) -> pd.Series:
    """How much each feature mattered to the model, largest first (sums to 1)."""
    importance = pd.Series(model.feature_importances_, index=model.feature_names_in_)
    importance = importance.rename(index=FEATURE_LABELS)
    return importance.sort_values(ascending=False)
