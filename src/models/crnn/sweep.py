"""CNN-LSTM (CRNN) capacity-knob sweep.

Knob = MACs/beat = conv stack + LSTM over the POOLED feature map. Each conv block halves
the length, so the recurrent part runs on 200/2^n timesteps with `channels[-1]` inputs
instead of 100 timesteps of a scalar (the plain LSTM) — deeper conv fronts buy shorter,
wider sequences. ~6 points spanning ~60k..730k MACs.
(label, conv_channels, lstm_hidden)
"""
from __future__ import annotations

SWEEP = [
    ("c8_h8",         [8],          8),   # seq 100 x 8ch
    ("c8_h16",        [8],          16),
    ("c16_h16",       [16],         16),
    ("c16-32_h16",    [16, 32],     16),  # seq 50 x 32ch
    ("c16-32_h32",    [16, 32],     32),
    ("c16-32-32_h32", [16, 32, 32], 32),  # seq 25 x 32ch
]
