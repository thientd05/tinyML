"""CNN-LSTM hybrid (CRNN): a conv front-end for local QRS morphology feeding an LSTM
back-end that models the temporal ordering of those features.

This is the architecture of Oh et al. 2018 (Comput. Biol. Med. 102:278-287, "Automated
diagnosis of arrhythmia using combination of CNN and LSTM techniques"), scaled down to
the ESP32 budget. It reuses the CNN's ConvBlock verbatim, so the folded-BN export path
and the conv kernel on-device are shared — only the head differs (LSTM instead of
GAP/FC).
"""
from __future__ import annotations

import torch.nn as nn

from src.config import BEAT_LEN
from src.models.cnn.architecture import ConvBlock


class ECGCRNN(nn.Module):
    def __init__(self, channels: list[int], hidden: int, in_len: int = BEAT_LEN):
        super().__init__()
        layers, prev = [], 1
        for c in channels:
            layers.append(ConvBlock(prev, c))
            prev = c
        self.features = nn.Sequential(*layers)
        self.seq_len = in_len // (2 ** len(channels))
        self.lstm = nn.LSTM(input_size=prev, hidden_size=hidden, num_layers=1,
                            batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):                      # x: [N, 1, L]
        f = self.features(x)                   # [N, C, L']
        out, _ = self.lstm(f.transpose(1, 2))  # [N, L', H]
        return self.head(out[:, -1, :])        # classify on the last hidden state
