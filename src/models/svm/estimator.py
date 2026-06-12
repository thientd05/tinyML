"""SVM estimator construction, stratified subsampling, training, and scoring.

linear: LinearSVC (just a weight vector — minimum size), wrapped in
CalibratedClassifierCV for predict_proba. rbf*: RBF SVC with stratified subsampling,
which controls the #support vectors and thus the on-device memory cost.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC


def build(kernel: str) -> Pipeline:
    """Always wrap in a StandardScaler so RBF kernel sees zero-mean unit-variance input."""
    if kernel == "linear":
        # CalibratedClassifierCV gives us predict_proba while keeping LinearSVC tiny.
        base = LinearSVC(C=1.0, class_weight="balanced", dual="auto",
                         max_iter=5000, random_state=42)
        clf = CalibratedClassifierCV(base, cv=3)
    else:
        clf = SVC(kernel="rbf", C=1.0, gamma="scale",
                  class_weight="balanced", probability=True,
                  cache_size=500, random_state=42)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def stratified_subsample(X: np.ndarray, y: np.ndarray, n: int, seed: int = 42):
    if n is None or n >= len(y):
        return X, y
    rng = np.random.default_rng(seed)
    idx_pos = np.flatnonzero(y == 1)
    idx_neg = np.flatnonzero(y == 0)
    pos_frac = idx_pos.size / y.size
    n_pos = max(1, int(round(n * pos_frac)))
    n_neg = n - n_pos
    pick_pos = rng.choice(idx_pos, size=min(n_pos, idx_pos.size), replace=False)
    pick_neg = rng.choice(idx_neg, size=min(n_neg, idx_neg.size), replace=False)
    idx = np.concatenate([pick_pos, pick_neg])
    rng.shuffle(idx)
    return X[idx], y[idx]


def train(pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> Pipeline:
    pipe.fit(X, y)
    return pipe


def scores(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    """P(class=1) in [0, 1] (RBF via probability=True, linear via calibration)."""
    return pipe.predict_proba(X)[:, 1]
