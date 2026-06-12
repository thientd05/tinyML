"""SVM capacity-knob sweep.

Knob = #support vectors (RBF), which sets both flash and the per-beat
kernel-evaluation cost. Linear is the floor (one weight vector); RBF #SV is steered by
the training subsample size. (label, kernel, train_subsample)
"""
from __future__ import annotations

SWEEP = [
    ("linear", "linear", None),
    ("rbf1k",  "rbf",     1_000),
    ("rbf2k",  "rbf",     2_000),
    ("rbf4k",  "rbf",     4_000),
    ("rbf7k",  "rbf",     7_000),
    ("rbf10k", "rbf",    10_000),
]
