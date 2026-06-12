"""Data pipeline: download -> per-record preprocessing -> beat segmentation ->
features -> train/val/test assembly (+ a PC feature-extraction benchmark).

Re-exports the public surface so callers can ``from src.data import build_dataset``
without knowing the internal module split.
"""
from src.data.assembly import (
    build_dataset,
    build_test_unfiltered,
)
from src.data.benchmark import feature_extract_benchmark
from src.data.download import download_mitbih
from src.data.features import extract_features
from src.data.preprocess import bandpass_filter
from src.data.segmentation import aami_binary_label, extract_beats

__all__ = [
    "aami_binary_label",
    "bandpass_filter",
    "build_dataset",
    "build_test_unfiltered",
    "download_mitbih",
    "extract_beats",
    "extract_features",
    "feature_extract_benchmark",
]
