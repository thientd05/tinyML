"""Dataset assembly: turn cleaned beats into the six train/val/test arrays per
input mode, with a clean-version-keyed npz cache and per-split cleaning stats."""
from __future__ import annotations

import json
from typing import Iterable, Literal

import numpy as np

from src.config import (
    CACHE_DIR,
    CLEAN_STATS_PATH,
    CLEAN_VERSION,
    CLIP_FRAC,
    DS1_TRAIN_RECORDS,
    DS1_VAL_RECORDS,
    DS2_RECORDS,
    FLATLINE_PTP,
    REALIGN_WINDOW,
    RESULTS_DIR,
)
from src.data.download import download_mitbih
from src.data.features import extract_features
from src.data.segmentation import extract_beats


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
    print(f"[data] building dataset ({mode}, clean v{CLEAN_VERSION}) ...")
    out, stats_by_split = {}, {}
    for name, recs in (("train", DS1_TRAIN_RECORDS), ("val", DS1_VAL_RECORDS),
                       ("test", DS2_RECORDS)):
        st: dict = {}
        out[name] = _build_record_arrays(recs, mode, drop_artifacts=True, stats=st)
        stats_by_split[name] = st
        d = st.get("dropped_artifact", 0)
        by = st.get("dropped_artifact_by_label", [0, 0])
        print(f"[data]   {name:<5} kept {st.get('kept',0)}/{st.get('total',0)} "
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
