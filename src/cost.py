"""Analytical cost model: compute (MACs/ops), parameter count, and flash footprint
per model variant.

These are the *hardware-independent* cost axes used to plot each family's
quality-vs-cost curve and to sanity-check the on-device latency. They are NOT
directly comparable across families (RF does comparisons, SVM kernel evals, CNN/LSTM
multiply-accumulates) — the apples-to-apples cross-family cost axis is the *measured*
us/beat on the ESP32. Within a family, the `macs`/`ops` field is monotone in the
capacity knob and is what we sweep.

Each function returns a dict with at least: n_params, macs (compute proxy),
flash_bytes (fp32 weight footprint).
"""
from __future__ import annotations

import numpy as np
from sklearn.svm import SVC


def rf_cost(model) -> dict:
    """Random Forest: compute proxy = total comparisons ~= n_trees * mean tree depth.
    Flash stores, per node: feature(int32) + threshold(f32) + 2 children(int32) +
    leaf prob(f32) = 20 bytes."""
    trees = model.estimators_
    n_nodes = int(sum(t.tree_.node_count for t in trees))
    n_trees = len(trees)
    mean_depth = float(np.mean([t.tree_.max_depth for t in trees]))
    return {
        "n_params": n_nodes,
        "n_nodes": n_nodes,
        "n_trees": n_trees,
        "mean_depth": mean_depth,
        "macs": float(n_trees * mean_depth),      # comparisons/beat (compute proxy)
        "flash_bytes": int(n_nodes * 20),
    }


def svm_cost(pipe, n_feat: int) -> dict:
    """SVM: RBF compute proxy = n_sv * n_feat kernel-distance MACs; linear = n_feat."""
    clf = pipe.named_steps["clf"]
    if isinstance(clf, SVC):
        n_sv = int(clf.support_vectors_.shape[0])
        macs = n_sv * n_feat
        flash_bytes = n_sv * n_feat * 4 + n_sv * 4 + n_feat * 8  # SV + duals + scaler
        n_params = n_sv * n_feat
    else:  # linear (+ calibration): one weight vector
        n_sv = 0
        macs = n_feat
        flash_bytes = n_feat * 4 * 2  # coef + scaler
        n_params = n_feat
    return {"n_params": int(n_params), "n_sv": n_sv,
            "macs": float(macs), "flash_bytes": int(flash_bytes)}


def cnn_cost(channels, fc, in_len: int = 200, n_classes: int = 2, k: int = 5) -> dict:
    """1D CNN MACs: sum over conv blocks of out_ch*in_ch*k*out_len (same padding,
    then MaxPool halves the length for the next block) + the FC/GAP head."""
    macs = 0
    n_params = 0
    L = in_len
    prev = 1
    for c in channels:
        macs += c * prev * k * L
        n_params += c * prev * k + c  # conv weights + bias (BN folded into these)
        prev = c
        L //= 2
    if fc:
        flat = prev * L
        macs += flat * fc + fc * n_classes
        n_params += flat * fc + fc + fc * n_classes + n_classes
    else:  # GAP -> Linear
        macs += prev * n_classes
        n_params += prev * n_classes + n_classes
    return {"n_params": int(n_params), "macs": float(macs), "flash_bytes": int(n_params * 4)}


def lstm_cost(hidden: int, layers: int, seq_len: int = 100,
              input_size: int = 1, n_classes: int = 2) -> dict:
    """LSTM MACs: per timestep each layer does 4 gates over Wih(4H x in) and
    Whh(4H x H); summed over seq_len. Plus the linear head on the last hidden state."""
    macs = 0
    n_params = 0
    in_sz = input_size
    for _ in range(layers):
        n_params += 4 * hidden * in_sz + 4 * hidden * hidden + 8 * hidden  # +bias_ih+bias_hh
        macs += seq_len * (4 * hidden * in_sz + 4 * hidden * hidden)
        in_sz = hidden
    n_params += hidden * n_classes + n_classes
    macs += hidden * n_classes
    return {"n_params": int(n_params), "macs": float(macs), "flash_bytes": int(n_params * 4)}
