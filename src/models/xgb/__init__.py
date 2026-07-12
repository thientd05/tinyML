"""XGBoost (gradient-boosted trees) for ECG anomaly detection (binary).

Boosting sibling of the RF baseline on the same 21 hand-crafted features.

Run:
    python -m src.models.xgb            # train missing sizes, then test all
    python -m src.models.xgb --retrain  # force retrain of all sizes

Each size variant is saved as model/xgb_<size>.pkl.
"""
from src.models.xgb.estimator import (
    base_logit,
    build,
    forward,
    scores,
    train,
    tree_arrays,
)
from src.models.xgb.sweep import SWEEP

MODEL_NAME = "xgb"
EXT = "pkl"

__all__ = [
    "SWEEP", "MODEL_NAME", "EXT",
    "build", "train", "scores", "tree_arrays", "base_logit", "forward",
]
