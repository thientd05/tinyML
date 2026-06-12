"""Per-record signal loading and conditioning: lead selection (MLII by name),
bandpass filtering, z-scoring, and the label-free artifact tests used during beat
segmentation."""
from __future__ import annotations

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt

from src.config import CLIP_FRAC, FLATLINE_PTP, FS, MITBIH_DIR


def bandpass_filter(sig: np.ndarray, fs: int = FS, low: float = 0.5, high: float = 40.0,
                    order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig).astype(np.float32)


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
