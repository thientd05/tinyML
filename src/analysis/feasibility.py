"""Enrich rows with on-device latency/RAM and apply the hard feasibility gate.

Per-beat LATENCY and working RAM come from the ESP32 benchmark log. The feature-based
families get the measured feature-extraction cost added to inference latency so
cross-family "detection time" is apples-to-apples (the raw-beat nets eat the beat
directly, feature cost = 0).
"""
from __future__ import annotations

import math

from src import config
from src.analysis.common import FEATURE_FAMILIES


def enrich(rows: list[dict], latency: dict, feature_us: float | None) -> None:
    for r in rows:
        # firmware prints full names ("{family}_{label}"); JSON rows key on bare label
        lat = latency.get(f"{r['family']}_{r['size']}") or latency.get(r["size"])
        r["infer_us"] = lat["us_beat"] if lat else float("nan")
        # feature extraction (wavelet db4 + FFT) only feeds the feature-based families
        # (RF/XGB/SVM); the raw-beat nets (CNN/LSTM/CRNN) pay 0.
        r["feature_us"] = (feature_us or 0.0) if r["family"] in FEATURE_FAMILIES else 0.0
        r["latency_ms"] = (r["infer_us"] + r["feature_us"]) / 1000.0
        r["ram_kb"] = (lat["heap_b"] / 1024.0) if lat else float("nan")


def feasibility(r: dict) -> None:
    r["fits_flash"] = r["flash_kb"] <= config.FLASH_BUDGET_KB
    r["fits_ram"] = (math.isnan(r["ram_kb"]) or r["ram_kb"] <= config.RAM_WORK_BUDGET_KB)
    if math.isnan(r["latency_ms"]):
        r["meets_latency"] = None  # device latency not measured yet
    else:
        r["meets_latency"] = r["latency_ms"] <= config.LATENCY_BUDGET_MS
    # provisionally feasible if no hard constraint is violated (unknown latency != violation)
    r["feasible"] = r["fits_flash"] and r["fits_ram"] and (r["meets_latency"] is not False)
    # did the DS1-val-tuned threshold still clear the sensitivity floor on DS2? (transfer)
    r["recall_holds"] = r["recall_deploy"] >= config.TARGET_RECALL
