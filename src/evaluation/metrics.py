"""Classification metrics and balanced class weights."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight


def class_weights(y: np.ndarray) -> np.ndarray:
    classes = np.array([0, 1])
    return compute_class_weight("balanced", classes=classes, y=y).astype(np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_score: np.ndarray | None = None) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),  # sensitivity
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,  # false-alarm rate
        "confusion_matrix": cm.tolist(),
    }
    if y_score is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            out["roc_auc"] = float("nan")
        try:
            # PR-AUC (average precision) is more informative than ROC-AUC under the
            # ~10% class imbalance here.
            out["pr_auc"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            out["pr_auc"] = float("nan")
    return out
