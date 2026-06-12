"""Decision-threshold selection and operating-point evaluation.

The clinical target is a sensitivity (recall) floor on the Abnormal class; these
helpers pick the highest threshold meeting it on a tuning split and report DS2
metrics at the default 0.5 threshold, the val-tuned (deployment) threshold, and a
DS2-tuned (capability ceiling) threshold.
"""
from __future__ import annotations

import numpy as np

from src.config import TARGET_RECALL
from src.evaluation.metrics import compute_metrics


def pick_threshold_for_recall(y_true: np.ndarray, y_score: np.ndarray,
                              target_recall: float = TARGET_RECALL) -> float:
    """Highest threshold whose recall >= target_recall on (y_true, y_score).

    Decision rule downstream is `predict 1 iff score >= tau`. Picking the highest
    feasible tau maximizes precision subject to the sensitivity floor. If there are
    no positives, returns 0.5.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    pos_scores = np.sort(y_score[y_true == 1])[::-1]  # descending
    P = pos_scores.size
    if P == 0:
        return 0.5
    k = min(max(int(np.ceil(target_recall * P)), 1), P)
    # tau = score of the k-th highest positive -> exactly the top-k positives pass,
    # giving recall = (#positives with score >= tau) / P >= target_recall.
    return float(pos_scores[k - 1])


def evaluate_at_threshold(y_true: np.ndarray, y_score: np.ndarray, tau: float) -> dict:
    """Metrics for the operating point `predict 1 iff score >= tau`."""
    y_pred = (np.asarray(y_score, dtype=float) >= tau).astype(np.int64)
    m = compute_metrics(y_true, y_pred, y_score)
    m["threshold"] = float(tau)
    return m


def evaluate_with_operating_point(y_val: np.ndarray, s_val: np.ndarray,
                                  y_test: np.ndarray, s_test: np.ndarray,
                                  target_recall: float = TARGET_RECALL) -> dict:
    """Tune tau on (y_val, s_val) to hit target_recall, then report DS2 metrics at the
    default 0.5 threshold and at the tuned operating point. Scores are P(class=1) in
    [0, 1] so 0.5 matches each backend's native `.predict()`.
    """
    tau = pick_threshold_for_recall(y_val, s_val, target_recall)        # deployment threshold
    tau_oracle = pick_threshold_for_recall(y_test, s_test, target_recall)  # capability threshold
    return {
        "threshold": float(tau),
        "target_recall": float(target_recall),
        # default 0.5 (== native .predict()):
        "test_05": evaluate_at_threshold(y_test, s_test, 0.5),
        # threshold tuned on DS1-val, applied to DS2 -> deployment-realistic (exposes the
        # inter-patient calibration-transfer gap: realized DS2 recall may be < target):
        "test_op": evaluate_at_threshold(y_test, s_test, tau),
        # threshold set on DS2 itself to hit target recall -> capability ceiling, i.e.
        # precision attainable AT target sensitivity (fair cross-family comparison point):
        "test_oracle": evaluate_at_threshold(y_test, s_test, tau_oracle),
        "val_op": evaluate_at_threshold(y_val, s_val, tau),
    }
