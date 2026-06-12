"""SVM baseline for ECG anomaly detection (binary).

tiny: LinearSVC (just a weight vector — minimum size).
small/medium/large: RBF SVC with stratified subsampling — controls #support
vectors, which determines on-device memory cost.

Run:
    python -m src.svm
    python -m src.svm --retrain
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from src import cost, utils

MODEL_NAME = "svm"
EXT = "pkl"

# Capacity-knob sweep. Knob = #support vectors (RBF), which sets both flash and the
# per-beat kernel-evaluation cost. Linear is the floor (one weight vector); RBF #SV is
# steered by the training subsample size. (label, kernel, train_subsample)
SWEEP = [
    ("linear", "linear", None),
    ("rbf1k",  "rbf",     1_000),
    ("rbf2k",  "rbf",     2_000),
    ("rbf4k",  "rbf",     4_000),
    ("rbf7k",  "rbf",     7_000),
    ("rbf10k", "rbf",    10_000),
]


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


def n_support_vectors(pipe: Pipeline) -> int | str:
    clf = pipe.named_steps["clf"]
    if isinstance(clf, SVC):
        return int(clf.support_vectors_.shape[0])
    return "linear"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    utils.set_seed(42)
    X_train, y_train, X_val, y_val, X_test, y_test = utils.build_dataset(mode="features")
    print(f"[svm] train={X_train.shape} val={X_val.shape} test={X_test.shape}  "
          f"pos_train={y_train.mean():.3f} pos_val={y_val.mean():.3f} pos_test={y_test.mean():.3f}")
    n_feat = X_train.shape[1]

    print(f"[svm] === sweep (DS2; threshold tuned on DS1-val to recall>={utils.TARGET_RECALL}) ===")
    for size, kernel, subsample in SWEEP:
        path = utils.model_path(MODEL_NAME, size, EXT)
        if path.exists() and not args.retrain:
            pipe = joblib.load(path)
            print(f"[svm] loaded {path.name}")
        else:
            Xs, ys = stratified_subsample(X_train, y_train, subsample)
            print(f"[svm] training {size} (kernel={kernel}, n_train={len(ys)}) ...")
            pipe = train(build(kernel), Xs, ys)
            joblib.dump(pipe, path, compress=3)
            print(f"[svm] saved -> {path}")
        rec = utils.evaluate_with_operating_point(
            y_val, scores(pipe, X_val), y_test, scores(pipe, X_test))
        rec.update(family=MODEL_NAME, size=size, cost=cost.svm_cost(pipe, n_feat))
        utils.save_metrics_json(MODEL_NAME, size, rec)
        o, c = rec["test_op"], rec["cost"]
        print(f"  [{size:<16}] tau={rec['threshold']:.3f}  recall={o['recall']:.4f} "
              f"prec={o['precision']:.4f} fpr={o['fpr']:.4f} f1={o['f1']:.4f} "
              f"pr_auc={o.get('pr_auc', float('nan')):.4f}  n_sv={c['n_sv']} macs={c['macs']:.0f}")


if __name__ == "__main__":
    main()
