"""Random Forest baseline for ECG anomaly detection (binary).

Run:
    python -m src.models.rf            # train missing sizes, then test all
    python -m src.models.rf --retrain  # force retrain of all sizes

Each size variant is saved as model/rf_<size>.pkl. Re-running will load existing
checkpoints and skip straight to test.
"""
from src.models.rf.estimator import build, scores, train
from src.models.rf.sweep import SWEEP

MODEL_NAME = "rf"
EXT = "pkl"

__all__ = ["SWEEP", "MODEL_NAME", "EXT", "build", "train", "scores"]
