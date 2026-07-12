"""Load the per-variant metrics JSONs (and the optional ESP32 latency table) into the
flat row dicts the rest of the analysis works on.

Quality is read from the full-DS2 metrics JSONs written by the training scripts —
trustworthy because on-device parity == 1.0000.
"""
from __future__ import annotations

import json
from pathlib import Path

from src import config
from src.analysis.common import FAMILIES
from src.models.cnn.sweep import SWEEP as CNN_SWEEP
from src.models.crnn.sweep import SWEEP as CRNN_SWEEP
from src.models.lstm.sweep import SWEEP as LSTM_SWEEP
from src.models.rf.sweep import SWEEP as RF_SWEEP
from src.models.svm.sweep import SWEEP as SVM_SWEEP
from src.models.xgb.sweep import SWEEP as XGB_SWEEP

# family -> ordered list of sweep labels (defines plot order + filters out stale files)
SWEEP_LABELS = {
    "rf": [s[0] for s in RF_SWEEP], "xgb": [s[0] for s in XGB_SWEEP],
    "svm": [s[0] for s in SVM_SWEEP], "cnn": [s[0] for s in CNN_SWEEP],
    "lstm": [s[0] for s in LSTM_SWEEP], "crnn": [s[0] for s in CRNN_SWEEP],
}


def _ms(sd: dict | None, key: str, fallback: float) -> tuple[float, float]:
    """Return (mean, std) from a multi-seed block, or (fallback, 0.0) if absent."""
    return (sd[key]["mean"], sd[key]["std"]) if (sd and key in sd) else (fallback, 0.0)


def load_records() -> list[dict]:
    """One flat row per sweep variant from model/<family>_<size>_metrics.json."""
    rows = []
    for fam in FAMILIES:
        for order, label in enumerate(SWEEP_LABELS[fam]):
            p = config.MODEL_DIR / f"{fam}_{label}_metrics.json"
            if not p.exists():
                print(f"[analyze] WARN missing {p.name} (train this variant first)")
                continue
            d = json.loads(p.read_text())
            if "test_op" not in d:  # stale 4-size schema
                print(f"[analyze] WARN {p.name} has old schema, skipping")
                continue
            # capability point = precision AT target recall on DS2 (fair comparison axis);
            # deployment point = DS1-val-tuned threshold applied to DS2 (transfer reality).
            cap = d.get("test_oracle", d["test_op"])
            dep, base, cost = d["test_op"], d["test_05"], d["cost"]
            # multi-seed aggregate (tools/multiseed.py): report the MEAN across seeds so a
            # single high-variance run can't crown a non-reproducible winner. Falls back to
            # the single representative value (std=0) for any model without a seeds block.
            sd = d.get("seeds")
            p_m, p_s = _ms(sd, "precision_op", cap["precision"])
            a_m, a_s = _ms(sd, "pr_auc", cap.get("pr_auc", float("nan")))
            rd_m, rd_s = _ms(sd, "recall_deploy", dep["recall"])
            f_m, _ = _ms(sd, "fpr_op", cap["fpr"])
            ro_m, _ = _ms(sd, "recall_op", cap["recall"])
            rows.append({
                "family": fam, "size": label, "order": order,
                "macs": float(cost["macs"]), "n_params": int(cost["n_params"]),
                "flash_kb": cost["flash_bytes"] / 1024.0,
                "threshold": d["threshold"], "n_seeds": (sd["n"] if sd else 1),
                # capability @ target recall (MEAN across seeds; *_std = spread):
                "recall_op": ro_m, "precision_op": p_m, "precision_op_std": p_s,
                "fpr_op": f_m, "f1_op": cap["f1"],
                "roc_auc": cap.get("roc_auc", float("nan")),
                "pr_auc": a_m, "pr_auc_std": a_s,
                # deployment (val-tuned threshold) realized on DS2:
                "recall_deploy": rd_m, "recall_deploy_std": rd_s,
                "precision_deploy": dep["precision"],
                "recall_05": base["recall"], "precision_05": base["precision"],
            })
    return rows


def parse_latency(path: Path) -> dict[str, dict]:
    """Parse the ESP32 benchmark table: `label | ... | us/beat | heapB`."""
    out = {}
    for line in path.read_text().splitlines():
        if "|" not in line:
            continue
        cols = [c.strip() for c in line.split("|")]
        try:
            out[cols[0]] = {"us_beat": float(cols[-2]), "heap_b": float(cols[-1])}
        except (ValueError, IndexError):
            continue  # header / non-data line
    return out
