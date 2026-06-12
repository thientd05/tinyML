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

from src import cost, utils

MODEL_NAME = "rf"
EXT = "pkl"

# Capacity-knob sweep. Knob = total tree node count (~ n_estimators * 2**depth),
# which drives both flash and the per-beat traversal cost on the ESP32. ~6 log-spaced
# points let us draw the quality-vs-cost curve and locate the knee, instead of 4
# gut-feel sizes. (label, n_estimators, max_depth)
SWEEP = [
    ("n10_d3",  10, 3),
    ("n10_d5",  10, 5),
    ("n20_d6",  20, 6),
    ("n30_d8",  30, 8),
    ("n50_d10", 50, 10),
    ("n80_d12", 80, 12),
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


def scores(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """P(class=1) in [0, 1]."""
    return model.predict_proba(X)[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="ignore existing checkpoints")
    args = parser.parse_args()

    utils.set_seed(42)
    X_train, y_train, X_val, y_val, X_test, y_test = utils.build_dataset(mode="features")
    print(f"[rf] train={X_train.shape} val={X_val.shape} test={X_test.shape}  "
          f"pos_train={y_train.mean():.3f} pos_val={y_val.mean():.3f} pos_test={y_test.mean():.3f}")

    print(f"[rf] === sweep (DS2; threshold tuned on DS1-val to recall>={utils.TARGET_RECALL}) ===")
    for size, n_est, depth in SWEEP:
        path = utils.model_path(MODEL_NAME, size, EXT)
        if path.exists() and not args.retrain:
            model = joblib.load(path)
            print(f"[rf] loaded {path.name}")
        else:
            print(f"[rf] training {size} (n_estimators={n_est}, max_depth={depth}) ...")
            model = train(build(n_est, depth), X_train, y_train)
            joblib.dump(model, path, compress=3)
            print(f"[rf] saved -> {path}")
        rec = utils.evaluate_with_operating_point(
            y_val, scores(model, X_val), y_test, scores(model, X_test))
        rec.update(family=MODEL_NAME, size=size, cost=cost.rf_cost(model))
        utils.save_metrics_json(MODEL_NAME, size, rec)
        _print_row(size, rec)


def _print_row(size: str, rec: dict) -> None:
    o, c = rec["test_op"], rec["cost"]
    print(f"  [{size:<16}] tau={rec['threshold']:.3f}  recall={o['recall']:.4f} "
          f"prec={o['precision']:.4f} fpr={o['fpr']:.4f} f1={o['f1']:.4f} "
          f"pr_auc={o.get('pr_auc', float('nan')):.4f}  nodes={c['n_nodes']} ops={c['macs']:.0f}")


if __name__ == "__main__":
    main()
