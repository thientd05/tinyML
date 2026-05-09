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
for _d in (DATASET_DIR, CACHE_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

FS = 360
BEAT_HALF = 100  # samples on each side of R-peak -> window length 200 (~555 ms)
BEAT_LEN = 2 * BEAT_HALF
LSTM_SEQ_LEN = 100  # downsampled length for LSTM

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


def _load_record(record_id: str, channel: int = 0):
    path = str(MITBIH_DIR / record_id)
    rec = wfdb.rdrecord(path, channels=[channel])
    ann = wfdb.rdann(path, "atr")
    sig = rec.p_signal[:, 0].astype(np.float32)
    sig = bandpass_filter(sig, fs=int(rec.fs))
    # z-score per record (more stable across patients than min-max)
    mu, sd = sig.mean(), sig.std() + 1e-8
    sig = (sig - mu) / sd
    return sig, ann


def extract_beats(record_id: str, channel: int = 0):
    """Return list of (beat[200], rr_prev, rr_post, label) for one record."""
    sig, ann = _load_record(record_id, channel)
    samples = ann.sample
    symbols = ann.symbol
    out = []
    for i, (s, sym) in enumerate(zip(samples, symbols)):
        label = aami_binary_label(sym)
        if label is None:
            continue
        lo, hi = s - BEAT_HALF, s + BEAT_HALF
        if lo < 0 or hi > len(sig):
            continue
        beat = sig[lo:hi]
        rr_prev = (s - samples[i - 1]) / FS if i > 0 else 0.0
        rr_post = (samples[i + 1] - s) / FS if i + 1 < len(samples) else 0.0
        out.append((beat.astype(np.float32), float(rr_prev), float(rr_post), label))
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
def _build_record_arrays(record_ids: Iterable[str], mode: Literal["features", "raw", "lstm"]):
    Xs, ys = [], []
    for rid in record_ids:
        beats = extract_beats(rid)
        for beat, rr_prev, rr_post, label in beats:
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


def build_dataset(mode: Literal["features", "raw", "lstm"], cache: bool = True):
    """Return (X_train, y_train, X_test, y_test). DS1=train, DS2=test."""
    download_mitbih()
    cache_path = CACHE_DIR / f"mitbih_{mode}.npz"
    if cache and cache_path.exists():
        data = np.load(cache_path)
        return data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    print(f"[utils] building dataset ({mode}) ...")
    X_train, y_train = _build_record_arrays(DS1_RECORDS, mode)
    X_test, y_test = _build_record_arrays(DS2_RECORDS, mode)
    if cache:
        np.savez_compressed(cache_path, X_train=X_train, y_train=y_train,
                            X_test=X_test, y_test=y_test)
    return X_train, y_train, X_test, y_test


# ---------------- evaluation helpers ----------------
def class_weights(y: np.ndarray) -> np.ndarray:
    classes = np.array([0, 1])
    return compute_class_weight("balanced", classes=classes, y=y).astype(np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_score: np.ndarray | None = None) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_score is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            out["roc_auc"] = float("nan")
    return out


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
