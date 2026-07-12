"""CNN-LSTM hybrid (CRNN) for ECG anomaly detection (binary), targeting ESP32.

Eats the same raw 200x1 beat as the CNN, so the training loop, loader and scoring are
reused from src.models.cnn unchanged.

Run:
    python -m src.models.crnn
    python -m src.models.crnn --retrain --epochs 20

Saves model/crnn_<size>.pt (state_dict); metrics as crnn_<size>_metrics.json.
"""
from src.models.cnn.inference import n_params, scores
from src.models.cnn.training import make_loader, train_one
from src.models.crnn.architecture import ECGCRNN
from src.models.crnn.sweep import SWEEP

MODEL_NAME = "crnn"
EXT = "pt"

__all__ = [
    "SWEEP", "MODEL_NAME", "EXT",
    "ECGCRNN", "make_loader", "train_one", "scores", "n_params",
]
