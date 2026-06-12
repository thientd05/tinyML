"""LSTM for ECG anomaly detection (binary), targeting ESP32.

Input is the beat downsampled 200 -> 100 (avg-pool 2).

Run:
    python -m src.models.lstm
    python -m src.models.lstm --retrain --epochs 15
"""
from src.models.lstm.architecture import ECGLSTM
from src.models.lstm.inference import n_params, scores
from src.models.lstm.sweep import SWEEP
from src.models.lstm.training import make_loader, train_one

MODEL_NAME = "lstm"
EXT = "pt"

__all__ = [
    "SWEEP", "MODEL_NAME", "EXT",
    "ECGLSTM", "make_loader", "train_one", "scores", "n_params",
]
