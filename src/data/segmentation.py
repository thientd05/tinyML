"""Beat segmentation (cleaning v2, all label-free) and the AAMI binary label map."""
from __future__ import annotations

import numpy as np

from src.config import (
    BEAT_HALF,
    FS,
    NON_BEAT_SYMBOLS,
    NORMAL_SYMBOLS,
    REALIGN_WINDOW,
)
from src.data.preprocess import _is_artifact, _load_record, _rail_mask


def aami_binary_label(symbol: str) -> int | None:
    if symbol in NON_BEAT_SYMBOLS:
        return None
    return 0 if symbol in NORMAL_SYMBOLS else 1


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
