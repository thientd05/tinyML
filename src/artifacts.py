"""Artifact persistence helpers: checkpoint and metrics paths."""
from __future__ import annotations

import json
from pathlib import Path

from src.config import MODEL_DIR


def save_metrics_json(name: str, size: str, m: dict) -> None:
    out = MODEL_DIR / f"{name}_{size}_metrics.json"
    out.write_text(json.dumps(m, indent=2))


def model_path(name: str, size: str, ext: str) -> Path:
    return MODEL_DIR / f"{name}_{size}.{ext}"
