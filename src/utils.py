"""Shared utilities for ECG anomaly detection on edge devices.

Pipeline (binary: Normal=0 / Abnormal=1, AAMI grouping):
    download MIT-BIH (wfdb) -> bandpass 0.5-40Hz -> z-score per record
    -> beat segmentation (+/-100 samples around R-peak) + RR intervals
    -> features (~30D) for RF/SVM | raw 200x1 for CNN | downsampled 100x1 for LSTM
    -> stratified DS1/DS2 patient split (de Chazal, paced records 102/104/107/217 excluded)
    -> npz cache to dataset/cache/
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pywt
import wfdb
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis, skew
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

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
MITBIH_DIR = DATASET_DIR / "mit-bih-arrhythmia-database-1.0.0"
CACHE_DIR = DATASET_DIR / "cache"
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"
for _d in (DATASET_DIR, CACHE_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

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
# Derived, adjustable. Encode the three problem criteria as hard constraints; see
# the methodology section of README. A variant is "feasible" iff it satisfies all of
# these AND a threshold exists on DS1-val reaching TARGET_RECALL.
LATENCY_BUDGET_MS = 100.0   # per-beat end-to-end (feature extraction + inference).
                            # Tachycardia ~200 bpm => beats ~300 ms apart; 100 ms gives
                            # ~3x headroom and "instant" detection. ("near-instant" tier: 50 ms)
TARGET_RECALL = 0.95        # clinical sensitivity floor on the Abnormal class
                            # (missing an abnormal beat is the costliest error)
FLASH_BUDGET_KB = 1024      # per-model weight footprint ceiling (of 4 MB flash)
RAM_WORK_BUDGET_KB = 64     # working RAM ceiling (of ~320 KB usable SRAM)

# de Chazal DS1/DS2 patient-level split. Paced beats records 102/104/107/217 excluded.
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

# AAMI binary grouping: normal beats -> 0, everything else -> 1
NORMAL_SYMBOLS = {"N", "L", "R", "e", "j"}
# Symbols that are NOT beats (rhythm/noise/artifact annotations) -> ignore
NON_BEAT_SYMBOLS = {"+", "~", "|", "!", "[", "]", '"', "x", "(", ")"}


# ---------------- reproducibility ----------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # local import: utils.py is also used by sklearn-only models
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------------- dataset download ----------------
def download_mitbih(force: bool = False) -> Path:
    """Idempotent download of MIT-BIH Arrhythmia DB to dataset/."""
    marker = MITBIH_DIR / "100.dat"
    if marker.exists() and not force:
        return MITBIH_DIR
    MITBIH_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[utils] downloading MIT-BIH Arrhythmia DB to {MITBIH_DIR} (~70 MB) ...")
    wfdb.io.dl_database("mitdb", str(MITBIH_DIR))
    return MITBIH_DIR


# ---------------- signal processing ----------------
def bandpass_filter(sig: np.ndarray, fs: int = FS, low: float = 0.5, high: float = 40.0,
                    order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig).astype(np.float32)


def aami_binary_label(symbol: str) -> int | None:
    if symbol in NON_BEAT_SYMBOLS:
        return None
    return 0 if symbol in NORMAL_SYMBOLS else 1


def _mlii_channel(record_id: str) -> int:
    """Index of the MLII lead for this record. MIT-BIH puts MLII on channel 0 for every
    kept record EXCEPT 114 (V5 on ch0, MLII on ch1); selecting by name feeds all records
    the same lead instead of blindly trusting channel 0. Falls back to 0 if MLII absent."""
    names = list(wfdb.rdheader(str(MITBIH_DIR / record_id)).sig_name)
    return names.index("MLII") if "MLII" in names else 0


def _load_record(record_id: str):
    path = str(MITBIH_DIR / record_id)
    rec = wfdb.rdrecord(path, channels=[_mlii_channel(record_id)])
    ann = wfdb.rdann(path, "atr")
    raw = rec.p_signal[:, 0].astype(np.float32)        # physical mV, pre-filter (artifact test)
    sig = bandpass_filter(raw, fs=int(rec.fs))
    # z-score per record (more stable across patients than min-max)
    mu, sd = sig.mean(), sig.std() + 1e-8
    sig = (sig - mu) / sd
    return sig, raw, ann


def _rail_mask(raw: np.ndarray) -> np.ndarray:
    """Boolean mask of samples pinned to the ADC rail (the record's min/max physical
    value), within 1e-3 mV — a proxy for saturation / clipping."""
    lo, hi = float(raw.min()), float(raw.max())
    return (np.abs(raw - lo) < 1e-3) | (np.abs(raw - hi) < 1e-3)


def _is_artifact(raw_win: np.ndarray, rail_win: np.ndarray) -> bool:
    """Label-free 'physically unusable beat' test. Deliberately NOT an amplitude/outlier
    filter (which would bias against high-amplitude abnormal beats) — only:
      - flatline / lead-off: peak-to-peak below FLATLINE_PTP mV
      - saturation / clipping: >= CLIP_FRAC of the window pinned to the ADC rail."""
    if float(raw_win.max() - raw_win.min()) < FLATLINE_PTP:
        return True
    if float(rail_win.mean()) >= CLIP_FRAC:
        return True
    return False


def extract_beats(record_id: str, drop_artifacts: bool = True, stats: dict | None = None):
    """Return list of (beat[200], rr_prev, rr_post, label) for one record, cleaned (v2):
      * lead selected by name (MLII) in _load_record (fixes record 114)
      * R-peak re-aligned to the local |signal| max within +/-REALIGN_WINDOW
      * RR intervals measured between consecutive BEATS (non-beat annotations skipped)
      * physically-unusable beats dropped (flatline / clipping) when drop_artifacts

    `stats` (optional) accumulates: total / kept / dropped_edge / dropped_artifact
    (+ dropped_artifact_by_label, for the no-class-bias audit in the report)."""
    sig, raw, ann = _load_record(record_id)
    samples, symbols = ann.sample, ann.symbol
    rail = _rail_mask(raw)
    # keep only real beats with their (annotated sample, label); RR is computed on THIS
    # filtered list so a non-beat marker can never sit between two RR neighbours.
    beats = [(int(samples[i]), lab)
             for i, sym in enumerate(symbols)
             if (lab := aami_binary_label(sym)) is not None]
    out = []
    for j, (s, label) in enumerate(beats):
        if stats is not None:
            stats["total"] = stats.get("total", 0) + 1
        # re-align onto the true R-peak (label-free: uses signal shape only)
        a, b = max(0, s - REALIGN_WINDOW), min(len(sig), s + REALIGN_WINDOW)
        r = a + int(np.argmax(np.abs(sig[a:b]))) if b - a > 1 else s
        lo, hi = r - BEAT_HALF, r + BEAT_HALF
        if lo < 0 or hi > len(sig):
            if stats is not None:
                stats["dropped_edge"] = stats.get("dropped_edge", 0) + 1
            continue
        if drop_artifacts and _is_artifact(raw[lo:hi], rail[lo:hi]):
            if stats is not None:
                stats["dropped_artifact"] = stats.get("dropped_artifact", 0) + 1
                stats.setdefault("dropped_artifact_by_label", [0, 0])[label] += 1
            continue
        # RR uses the original annotated samples (the R-peak references), between beats
        rr_prev = (s - beats[j - 1][0]) / FS if j > 0 else 0.0
        rr_post = (beats[j + 1][0] - s) / FS if j + 1 < len(beats) else 0.0
        out.append((sig[lo:hi].astype(np.float32), float(rr_prev), float(rr_post), label))
        if stats is not None:
            stats["kept"] = stats.get("kept", 0) + 1
    return out


# ---------------- features for RF/SVM ----------------
def extract_features(beat: np.ndarray, rr_prev: float, rr_post: float) -> np.ndarray:
    """~30 hand-crafted features per beat."""
    feats: list[float] = []
    # time-domain stats
    feats += [
        beat.mean(), beat.std(), beat.min(), beat.max(),
        float(skew(beat)), float(kurtosis(beat)),
        float(np.sqrt((beat ** 2).mean())),  # RMS
        beat.max() - beat.min(),             # peak-to-peak
        float((beat ** 2).sum()),            # energy
        float(((beat[:-1] * beat[1:]) < 0).sum()),  # zero crossings
    ]
    # wavelet energies (db4 level 4 -> 5 sub-bands)
    coeffs = pywt.wavedec(beat, "db4", level=4)
    feats += [float(np.sum(c ** 2)) for c in coeffs]
    # frequency: dominant freq + spectral entropy
    spec = np.abs(np.fft.rfft(beat)) ** 2
    spec_norm = spec / (spec.sum() + 1e-12)
    feats.append(float(np.argmax(spec)))
    feats.append(float(-np.sum(spec_norm * np.log(spec_norm + 1e-12))))
    # RR-interval features
    rr_ratio = rr_post / (rr_prev + 1e-6)
    feats += [rr_prev, rr_post, rr_ratio, rr_post - rr_prev]
    return np.asarray(feats, dtype=np.float32)


# ---------------- dataset assembly ----------------
def _build_record_arrays(record_ids: Iterable[str], mode: Literal["features", "raw", "lstm"],
                         drop_artifacts: bool = True, stats: dict | None = None):
    Xs, ys = [], []
    for rid in record_ids:
        for beat, rr_prev, rr_post, label in extract_beats(rid, drop_artifacts, stats):
            if mode == "features":
                Xs.append(extract_features(beat, rr_prev, rr_post))
            elif mode == "raw":
                Xs.append(beat)
            elif mode == "lstm":
                # avg-pool 200 -> 100 (factor 2)
                Xs.append(beat.reshape(-1, 2).mean(axis=1).astype(np.float32))
            else:
                raise ValueError(mode)
            ys.append(label)
    return np.stack(Xs), np.asarray(ys, dtype=np.int64)


def _save_clean_stats(stats_by_split: dict) -> None:
    """Persist per-split cleaning counts so the report can quote exact numbers. Identical
    across modes (drops depend on signal geometry, not feature representation)."""
    try:
        RESULTS_DIR.mkdir(exist_ok=True)
        CLEAN_STATS_PATH.write_text(json.dumps(
            {"clean_version": CLEAN_VERSION, "realign_window": REALIGN_WINDOW,
             "flatline_ptp_mv": FLATLINE_PTP, "clip_frac": CLIP_FRAC,
             "splits": stats_by_split}, indent=2))
    except OSError:
        pass


def build_dataset(mode: Literal["features", "raw", "lstm"], cache: bool = True):
    """Return (X_train, y_train, X_val, y_val, X_test, y_test), cleaned (v2).

    DS1\\DS1_VAL = train, DS1_VAL = patient-pure validation (threshold tuning +
    checkpoint selection), DS2 = test. Caches all six arrays under a clean-version key
    so changing the cleaning logic transparently rebuilds the cache.
    """
    download_mitbih()
    cache_path = CACHE_DIR / f"mitbih_{mode}_v{CLEAN_VERSION}.npz"
    if cache and cache_path.exists():
        data = np.load(cache_path)
        if "X_val" in data:
            return (data["X_train"], data["y_train"], data["X_val"], data["y_val"],
                    data["X_test"], data["y_test"])
    print(f"[utils] building dataset ({mode}, clean v{CLEAN_VERSION}) ...")
    out, stats_by_split = {}, {}
    for name, recs in (("train", DS1_TRAIN_RECORDS), ("val", DS1_VAL_RECORDS),
                       ("test", DS2_RECORDS)):
        st: dict = {}
        out[name] = _build_record_arrays(recs, mode, drop_artifacts=True, stats=st)
        stats_by_split[name] = st
        d = st.get("dropped_artifact", 0)
        by = st.get("dropped_artifact_by_label", [0, 0])
        print(f"[utils]   {name:<5} kept {st.get('kept',0)}/{st.get('total',0)} "
              f"(dropped {d} artifact [N={by[0]}, Abn={by[1]}], "
              f"{st.get('dropped_edge',0)} edge)")
    _save_clean_stats(stats_by_split)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = out["train"], out["val"], out["test"]
    if cache:
        np.savez_compressed(cache_path, X_train=X_train, y_train=y_train,
                            X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_test_unfiltered(mode: Literal["features", "raw", "lstm"]):
    """DS2 with the SAME cleaning (lead-by-name, RR-between-beats, R-peak realign) but
    WITHOUT artifact dropping — the transparency baseline for 'cleaned vs full DS2' in the
    report. Not cached (cheap, only used by analyze for the winners)."""
    return _build_record_arrays(DS2_RECORDS, mode, drop_artifacts=False)


# ---------------- evaluation helpers ----------------
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


# ---------------- operating point (decision threshold) ----------------
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


def feature_extract_benchmark(n: int = 2000, seed: int = 42) -> float:
    """Mean microseconds/beat for extract_features on this PC (db4 wavelet + FFT).

    Cross-check for the ESP32 feature-extraction microbench, whose us/beat is added
    to RF/SVM inference latency so cross-family detection time is apples-to-apples
    (CNN/LSTM consume raw beats, feature cost = 0).
    """
    import time
    set_seed(seed)
    download_mitbih()
    beats = []
    for rid in DS2_RECORDS:
        for beat, rr_prev, rr_post, _ in extract_beats(rid):
            beats.append((beat, rr_prev, rr_post))
            if len(beats) >= n:
                break
        if len(beats) >= n:
            break
    t0 = time.perf_counter()
    for beat, rr_prev, rr_post in beats:
        extract_features(beat, rr_prev, rr_post)
    return (time.perf_counter() - t0) / len(beats) * 1e6


def print_metrics_row(name: str, size: str, m: dict, n_params: int | float | None = None) -> None:
    auc = m.get("roc_auc", float("nan"))
    extra = f" params={n_params}" if n_params is not None else ""
    print(f"  [{name:<5} | {size:<6}] acc={m['accuracy']:.4f} "
          f"p={m['precision']:.4f} r={m['recall']:.4f} f1={m['f1']:.4f} "
          f"auc={auc:.4f}{extra}")


def save_metrics_json(name: str, size: str, m: dict) -> None:
    out = MODEL_DIR / f"{name}_{size}_metrics.json"
    out.write_text(json.dumps(m, indent=2))


def model_path(name: str, size: str, ext: str) -> Path:
    return MODEL_DIR / f"{name}_{size}.{ext}"
