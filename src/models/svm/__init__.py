"""SVM baseline for ECG anomaly detection (binary).

Run:
    python -m src.models.svm
    python -m src.models.svm --retrain
"""
from src.models.svm.estimator import (
    build,
    scores,
    stratified_subsample,
    train,
)
from src.models.svm.sweep import SWEEP

MODEL_NAME = "svm"
EXT = "pkl"

__all__ = [
    "SWEEP", "MODEL_NAME", "EXT",
    "build", "stratified_subsample", "train", "scores",
]
