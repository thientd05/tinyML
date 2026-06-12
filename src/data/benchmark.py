"""PC-side timing of the hand-crafted feature extraction (db4 wavelet + FFT)."""
from __future__ import annotations

import time

from src.config import DS2_RECORDS
from src.data.download import download_mitbih
from src.data.features import extract_features
from src.data.segmentation import extract_beats
from src.seeding import set_seed


def feature_extract_benchmark(n: int = 2000, seed: int = 42) -> float:
    """Mean microseconds/beat for extract_features on this PC (db4 wavelet + FFT).

    Cross-check for the ESP32 feature-extraction microbench, whose us/beat is added
    to RF/SVM inference latency so cross-family detection time is apples-to-apples
    (CNN/LSTM consume raw beats, feature cost = 0).
    """
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
