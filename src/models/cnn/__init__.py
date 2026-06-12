"""1D CNN for ECG anomaly detection (binary), targeting ESP32.

Run:
    python -m src.models.cnn
    python -m src.models.cnn --retrain
    python -m src.models.cnn --epochs 20 --batch-size 256

Saves model/cnn_<size>.pt (state_dict); metrics as cnn_<size>_metrics.json.
"""
from src.models.cnn.architecture import ConvBlock, ECGCNN
from src.models.cnn.inference import n_params, scores
from src.models.cnn.sweep import SWEEP
from src.models.cnn.training import make_loader, train_one

MODEL_NAME = "cnn"
EXT = "pt"

__all__ = [
    "SWEEP", "MODEL_NAME", "EXT",
    "ConvBlock", "ECGCNN", "make_loader", "train_one", "scores", "n_params",
]
