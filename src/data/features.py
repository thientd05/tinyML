"""Hand-crafted per-beat features (~21D) for the RF/SVM families."""
from __future__ import annotations

import numpy as np
import pywt
from scipy.stats import kurtosis, skew


def extract_features(beat: np.ndarray, rr_prev: float, rr_post: float) -> np.ndarray:
    """21 hand-crafted features per beat: 10 time-domain stats + 5 wavelet band
    energies (db4, level 4) + 2 spectral + 4 RR-interval."""
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
