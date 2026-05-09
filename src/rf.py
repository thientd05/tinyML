"""Random Forest baseline for ECG anomaly detection (binary).

Run:
    python -m src.rf            # train missing sizes, then test all
    python -m src.rf --retrain  # force retrain of all sizes

Each size variant is saved as model/rf_<size>.pkl. Re-running the script will
load existing checkpoints and skip straight to test.
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src import utils

MODEL_NAME = "rf"
EXT = "pkl"

# (size_label, n_estimators, max_depth)
SIZES = [
    ("tiny",   10, 4),
    ("small",  20, 6),
    ("medium", 40, 8),
    ("large",  80, 10),
]


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


def evaluate(model: RandomForestClassifier, X: np.ndarray, y: np.ndarray) -> dict:
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    return utils.compute_metrics(y, y_pred, y_score)


def n_params(model: RandomForestClassifier) -> int:
    return int(sum(t.tree_.node_count for t in model.estimators_))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="ignore existing checkpoints")
    args = parser.parse_args()

    utils.set_seed(42)
    X_train, y_train, X_test, y_test = utils.build_dataset(mode="features")
    print(f"[rf] train shape={X_train.shape}  test shape={X_test.shape}  "
          f"pos_train={y_train.mean():.3f}  pos_test={y_test.mean():.3f}")

    print("[rf] === per-size results (test set) ===")
    for size, n_est, depth in SIZES:
        path = utils.model_path(MODEL_NAME, size, EXT)
        if path.exists() and not args.retrain:
            model = joblib.load(path)
            print(f"[rf] loaded {path.name}")
        else:
            print(f"[rf] training {size} (n_estimators={n_est}, max_depth={depth}) ...")
            model = train(build(n_est, depth), X_train, y_train)
            joblib.dump(model, path, compress=3)
            print(f"[rf] saved -> {path}")
        m = evaluate(model, X_test, y_test)
        utils.save_metrics_json(MODEL_NAME, size, m)
        utils.print_metrics_row(MODEL_NAME, size, m, n_params=n_params(model))


if __name__ == "__main__":
    main()
