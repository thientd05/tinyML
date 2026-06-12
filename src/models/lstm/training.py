"""LSTM training loop (fp32 + grad clipping, best-val-F1 checkpointing)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    X_t = torch.from_numpy(X).float().unsqueeze(-1)  # [N, L, 1]
    y_t = torch.from_numpy(y).long()
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


def train_one(model: nn.Module, X_tr, y_tr, X_val, y_val, *, device,
              epochs: int, batch_size: int, lr: float, weight_decay: float,
              class_weight_t: torch.Tensor) -> nn.Module:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight_t.to(device))

    train_loader = make_loader(X_tr, y_tr, batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size * 2, shuffle=False)

    best_f1, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * xb.size(0)
        sched.step()
        # validation
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                logits = model(xb)
                ps.append(logits.argmax(1).cpu().numpy())
                ys.append(yb.numpy())
        f1 = f1_score(np.concatenate(ys), np.concatenate(ps), zero_division=0)
        print(f"  epoch {epoch+1:02d}/{epochs}  loss={running/len(y_tr):.4f}  val_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model
