"""XGBoost estimator construction, training, scoring, and tree extraction.

Boosting counterpart to the RF baseline: same 21-D feature input, same tree-traversal
inference on the ESP32, but trees are fitted sequentially on the residuals instead of
independently on bootstrap samples. Reference: Shi et al., hierarchical weighted XGBoost
on MIT-BIH; Sensors 2021 21(16):5290, beat-level XGBoost for wearable single-lead ECG.

`tree_arrays` / `base_logit` expose the fitted booster in the exact form the plain-C
kernel walks (see tools/export_esp32.py and esp32/src/main.cpp), so the cost model,
the export and the firmware all read the same numbers.
"""
from __future__ import annotations

import json

import numpy as np
from xgboost import XGBClassifier

LEARNING_RATE = 0.1  # fixed: capacity (trees x depth) is the only swept knob


def build(n_estimators: int, max_depth: int, random_state: int = 42) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=LEARNING_RATE,
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=random_state,
    )


def train(model: XGBClassifier, X: np.ndarray, y: np.ndarray) -> XGBClassifier:
    """`scale_pos_weight` = neg/pos is XGBoost's equivalent of class_weight="balanced",
    which the RF/SVM/CNN/LSTM paths all use — keep the imbalance handling identical."""
    pos = float(np.count_nonzero(y == 1))
    neg = float(np.count_nonzero(y == 0))
    model.set_params(scale_pos_weight=(neg / pos) if pos else 1.0)
    model.fit(X, y)
    return model


def scores(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    """P(class=1) in [0, 1]."""
    return model.predict_proba(X)[:, 1]


def base_logit(model: XGBClassifier) -> float:
    """XGBoost stores `base_score` as a PROBABILITY; the raw margin starts from its logit.
    Full score = sigmoid(base_logit + sum of leaf values), leaf values already shrunk by
    the learning rate."""
    cfg = json.loads(model.get_booster().save_config())
    raw = cfg["learner"]["learner_model_param"]["base_score"]
    # xgboost >=2 serialises this as a vector string, e.g. "[5E-1]"; <2 as a bare float.
    p = float(raw.strip("[]").split(",")[0])
    p = min(max(p, 1e-7), 1 - 1e-7)
    return float(np.log(p / (1.0 - p)))


def _feature_index(split: str, names: list[str] | None) -> int:
    if names:
        return names.index(split)
    return int(split[1:])  # default dump names are "f0".."f20"


def tree_arrays(model: XGBClassifier) -> list[dict]:
    """Flatten each boosted tree into the node arrays the C kernel indexes.

    Per tree: feature/thr/left/right/leaf, node ids renumbered 0..n-1 (dump ids are
    unique per tree but not guaranteed dense after pruning). Internal nodes take the
    `left` branch when `x[feature] < thr` — XGBoost splits on STRICT less-than, unlike
    sklearn's `<=`; the firmware must match or parity breaks.
    """
    booster = model.get_booster()
    names = booster.feature_names
    out = []
    for dump in booster.get_dump(dump_format="json"):
        nodes, stack = {}, [json.loads(dump)]
        while stack:
            n = stack.pop()
            nodes[n["nodeid"]] = n
            stack.extend(n.get("children", []))
        ids = sorted(nodes)
        remap = {nid: i for i, nid in enumerate(ids)}
        t = {"feature": [], "thr": [], "left": [], "right": [], "leaf": []}
        for nid in ids:
            n = nodes[nid]
            if "leaf" in n:
                t["feature"].append(-1); t["thr"].append(0.0)
                t["left"].append(-1); t["right"].append(-1)
                t["leaf"].append(float(n["leaf"]))
            else:
                t["feature"].append(_feature_index(n["split"], names))
                t["thr"].append(float(n["split_condition"]))
                t["left"].append(remap[n["yes"]]); t["right"].append(remap[n["no"]])
                t["leaf"].append(0.0)
        t["depth"] = _max_depth(t["left"], t["right"])
        out.append(t)
    return out


def _max_depth(left: list[int], right: list[int]) -> int:
    best, stack = 0, [(0, 0)]
    while stack:
        nd, d = stack.pop()
        if left[nd] == -1:
            best = max(best, d)
        else:
            stack.append((left[nd], d + 1)); stack.append((right[nd], d + 1))
    return best


def forward(trees: list[dict], b0: float, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pure-NumPy replay of the booster (the reference the C kernel is asserted against)."""
    margin = np.full(len(X), b0, dtype=np.float64)
    for t in trees:
        for i, x in enumerate(X):
            nd = 0
            while t["left"][nd] != -1:
                nd = t["left"][nd] if x[t["feature"][nd]] < t["thr"][nd] else t["right"][nd]
            margin[i] += t["leaf"][nd]
    p1 = 1.0 / (1.0 + np.exp(-margin))
    return (p1 > 0.5).astype(np.int64), p1
