"""Project-wide constants: filesystem layout, signal/beat dimensions, dataset
cleaning parameters, deployment budgets, the de Chazal patient split, and AAMI
symbol groupings.

This module holds configuration only (no logic), so it can be imported from any
layer — data pipeline, models, evaluation, analysis — without pulling in heavy
dependencies. The data/model/cache directories are created on import (idempotent).
"""
from __future__ import annotations

from pathlib import Path

# ---------------- filesystem layout ----------------
ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
MITBIH_DIR = DATASET_DIR / "mit-bih-arrhythmia-database-1.0.0"
CACHE_DIR = DATASET_DIR / "cache"
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"
for _d in (DATASET_DIR, CACHE_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------- signal / beat geometry ----------------
FS = 360
BEAT_HALF = 100  # samples on each side of R-peak -> window length 200 (~555 ms)
BEAT_LEN = 2 * BEAT_HALF
LSTM_SEQ_LEN = 100  # downsampled length for LSTM

# ---------------- dataset cleaning (v2) ----------------
# Bump CLEAN_VERSION whenever the cleaning logic changes so cached .npz are rebuilt.
# All cleaning is LABEL-FREE (uses lead identity, annotation structure, and signal
# shape only — never the Normal/Abnormal label) and applied identically to every split.
CLEAN_VERSION = 2
REALIGN_WINDOW = 15    # +/- samples to snap the window onto the true R-peak (p99 offset=13)
FLATLINE_PTP = 0.05    # mV peak-to-peak below which a beat is a dead/lead-off segment
CLIP_FRAC = 0.10       # fraction of a beat pinned to the ADC rail => saturated/clipped
CLEAN_STATS_PATH = RESULTS_DIR / "cleaning_stats.json"

# ---------------- deployment budgets (ESP32 / clinical) ----------------
# Derived, adjustable. Encode the problem criteria as hard constraints; see the
# methodology section of README. A variant is "feasible" iff it satisfies all of these
# AND a threshold exists on DS1-val reaching TARGET_RECALL. In practice ONLY latency
# binds: on the measured device timings it rejects 5 of 36 configs -- cnn_c16-32-32
# (110.3ms), cnn_c16-32-64-64 (263.6ms), lstm_h32x2 (266.0ms), crnn_c16-32_h32 (155.3ms)
# and crnn_c16-32-32_h32 (140.8ms) -- leaving 31 feasible. Flash and RAM are guardrails
# set to the real single-model hardware ceiling that no variant approaches (max flash
# rf_n80_d12 = 1005KB; max RAM cnn_c4 = 5.3KB) -- compute speed is the real selector.
LATENCY_BUDGET_MS = 100.0   # per-beat end-to-end (feature extraction + inference). THE
                            # binding constraint. Tachycardia ~200 bpm => beats ~300 ms
                            # apart; 100 ms gives ~3x headroom + "instant" detection.
TARGET_RECALL = 0.95        # clinical sensitivity floor on the Abnormal class
                            # (missing an abnormal beat is the costliest error)
FLASH_BUDGET_KB = 2048      # per-model weight ceiling for ONE deployed model: 4 MB flash,
                            # single-app (huge_app) ~3 MB partition minus ~1 MB firmware/
                            # runtime => ~2 MB free for weights. Non-binding guardrail.
RAM_WORK_BUDGET_KB = 64     # working RAM ceiling (of ~320 KB usable SRAM). Non-binding
                            # (every variant uses ~5.3 KB).

# ---------------- de Chazal DS1/DS2 patient split ----------------
# Paced beats records 102/104/107/217 excluded.
DS1_RECORDS = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
]
DS2_RECORDS = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
]
# Patient-pure validation holdout carved out of DS1, used ONLY to (a) tune the
# decision threshold to TARGET_RECALL and (b) pick the best CNN/LSTM checkpoint.
# 207/209/215 = 8227 beats (16% of DS1), 10.6% abnormal ~ DS1's global 10.1% — kept
# patient-disjoint from both the training records and DS2 to avoid leakage.
DS1_VAL_RECORDS = ["207", "209", "215"]
DS1_TRAIN_RECORDS = [r for r in DS1_RECORDS if r not in DS1_VAL_RECORDS]

# ---------------- AAMI binary grouping ----------------
# Normal beats -> 0, everything else -> 1.
NORMAL_SYMBOLS = {"N", "L", "R", "e", "j"}
# Symbols that are NOT beats (rhythm/noise/artifact annotations) -> ignore
NON_BEAT_SYMBOLS = {"+", "~", "|", "!", "[", "]", '"', "x", "(", ")"}
