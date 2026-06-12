"""LSTM inference: per-beat P(class=1) scores and a parameter count."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.models.lstm.training import make_loader


def scores(model: nn.Module, X, *, device, batch_size: int = 512) -> np.ndarray:
    """P(class=1) in [0, 1] for each beat."""
    model.to(device).eval()
    loader = make_loader(X, np.zeros(len(X), dtype=np.int64), batch_size, shuffle=False)
    out = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            out.append(torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
