"""Shared constants for the analysis stage."""
from __future__ import annotations

from src import config

RESULTS = config.RESULTS_DIR
# Ordered classical-first, then raw-beat nets. xgb/crnn are the boosting and CNN-LSTM
# hybrid families added on top of the original four.
FAMILIES = ["rf", "xgb", "svm", "cnn", "lstm", "crnn"]
# Families fed the 21 hand-crafted features (they pay the on-device feature-extraction
# cost); everything else eats the raw beat.
FEATURE_FAMILIES = {"rf", "xgb", "svm"}
