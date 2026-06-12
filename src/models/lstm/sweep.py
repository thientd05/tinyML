"""LSTM capacity-knob sweep.

Knob = MACs/beat ~ seq_len * 4 * H * (in + H) (gate matmuls, the LSTM bottleneck since
timesteps are sequential). ~6 points over hidden size + a 2-layer point.
(label, hidden, num_layers)  [bidirectional is always False]
"""
from __future__ import annotations

SWEEP = [
    ("h4",    4,  1),
    ("h8",    8,  1),
    ("h16",   16, 1),
    ("h24",   24, 1),
    ("h32",   32, 1),
    ("h32x2", 32, 2),
]
