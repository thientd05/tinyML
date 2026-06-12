"""Random Forest capacity-knob sweep.

Knob = total tree node count (~ n_estimators * 2**depth), which drives both flash and
the per-beat traversal cost on the ESP32. ~6 log-spaced points let us draw the
quality-vs-cost curve and locate the knee, instead of 4 gut-feel sizes.
(label, n_estimators, max_depth)
"""
from __future__ import annotations

SWEEP = [
    ("n10_d3",  10, 3),
    ("n10_d5",  10, 5),
    ("n20_d6",  20, 6),
    ("n30_d8",  30, 8),
    ("n50_d10", 50, 10),
    ("n80_d12", 80, 12),
]
