"""Idempotent MIT-BIH Arrhythmia DB download (via wfdb)."""
from __future__ import annotations

from pathlib import Path

import wfdb

from src.config import MITBIH_DIR


def download_mitbih(force: bool = False) -> Path:
    """Idempotent download of MIT-BIH Arrhythmia DB to dataset/."""
    marker = MITBIH_DIR / "100.dat"
    if marker.exists() and not force:
        return MITBIH_DIR
    MITBIH_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[data] downloading MIT-BIH Arrhythmia DB to {MITBIH_DIR} (~70 MB) ...")
    wfdb.io.dl_database("mitdb", str(MITBIH_DIR))
    return MITBIH_DIR
