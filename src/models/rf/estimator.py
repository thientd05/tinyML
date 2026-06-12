"""Random Forest estimator construction, training, and scoring."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def build(n_estimators: int, max_depth: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )


def train(model: RandomForestClassifier, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    model.fit(X, y)
    return model


def scores(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """P(class=1) in [0, 1]."""
    return model.predict_proba(X)[:, 1]
