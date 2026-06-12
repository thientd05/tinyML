"""1D CNN capacity-knob sweep.

Knob = MACs/beat (drives ESP32 latency); #params drives flash. ~6 width/depth points
log-spaced in MACs. (label, conv_channels, fc_hidden)
"""
from __future__ import annotations

SWEEP = [
    ("c4",           [4],             None),  # GAP -> Linear(4, 2)
    ("c8",           [8],             None),  # GAP -> Linear(8, 2)
    ("c8-16",        [8, 16],         16),
    ("c16-16",       [16, 16],        16),
    ("c16-32-32",    [16, 32, 32],    32),
    ("c16-32-64-64", [16, 32, 64, 64], 64),
]
