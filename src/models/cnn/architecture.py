"""1D CNN architecture: a stack of Conv-BN-ReLU-MaxPool blocks + a GAP or FC head."""
from __future__ import annotations

import torch.nn as nn

from src.config import BEAT_LEN


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=k, padding=k // 2),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

    def forward(self, x):  # noqa: D401
        return self.block(x)


class ECGCNN(nn.Module):
    def __init__(self, channels: list[int], fc_hidden: int | None, in_len: int = BEAT_LEN):
        super().__init__()
        layers, prev = [], 1
        for c in channels:
            layers.append(ConvBlock(prev, c))
            prev = c
        self.features = nn.Sequential(*layers)
        out_len = in_len // (2 ** len(channels))
        flat = prev * out_len
        if fc_hidden:
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat, fc_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(fc_hidden, 2),
            )
        else:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(prev, 2),
            )

    def forward(self, x):
        return self.head(self.features(x))
