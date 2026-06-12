"""LSTM architecture: a (possibly stacked) LSTM over the downsampled beat + a linear
head on the last hidden state.

LSTMs are the slowest of the four families on this task because they cannot parallelize
across timesteps, so the hidden size is kept deliberately small.
"""
from __future__ import annotations

import torch.nn as nn


class ECGLSTM(nn.Module):
    def __init__(self, hidden: int, num_layers: int, bidirectional: bool):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, 2)

    def forward(self, x):  # x: [N, L, 1]
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)
