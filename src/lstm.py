"""LSTM for ECG anomaly detection (binary), four sizes targeting ESP32.

Input is the beat downsampled 200 -> 100 (avg-pool 2). LSTMs are the slowest of
the four models on this task because they cannot parallelize across timesteps,
so we keep the hidden size deliberately small.

Run:
    python -m src.lstm
    python -m src.lstm --retrain --epochs 15
"""
from __future__ import annotations

import argparse
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src import utils

MODEL_NAME = "lstm"
EXT = "pt"

# (size, hidden, num_layers, bidirectional)
SIZES = [
    ("tiny",   8,  1, False),
    ("small",  16, 1, False),
    ("medium", 32, 1, False),
    ("large",  32, 2, False),
]


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
        from sklearn.metrics import f1_score
        f1 = f1_score(np.concatenate(ys), np.concatenate(ps), zero_division=0)
        print(f"  epoch {epoch+1:02d}/{epochs}  loss={running/len(y_tr):.4f}  val_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate(model: nn.Module, X, y, *, device, batch_size: int = 512) -> dict:
    model.to(device).eval()
    loader = make_loader(X, y, batch_size, shuffle=False)
    ys, ps, scores = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1)[:, 1]
            ps.append(logits.argmax(1).cpu().numpy())
            scores.append(prob.cpu().numpy())
            ys.append(yb.numpy())
    return utils.compute_metrics(np.concatenate(ys), np.concatenate(ps), np.concatenate(scores))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def split_train_val(X, y, val_frac: float = 0.1, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); rng.shuffle(idx)
    cut = int(len(y) * (1 - val_frac))
    tr, va = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[va], y[va]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    utils.set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[lstm] device={device}")

    X_train_full, y_train_full, X_test, y_test = utils.build_dataset(mode="lstm")
    X_tr, y_tr, X_val, y_val = split_train_val(X_train_full, y_train_full)
    cw = torch.from_numpy(utils.class_weights(y_tr))
    print(f"[lstm] train={X_tr.shape}  val={X_val.shape}  test={X_test.shape}  "
          f"class_weights={cw.tolist()}")

    print("[lstm] === per-size results (test set) ===")
    for size, hidden, layers, bidir in SIZES:
        path = utils.model_path(MODEL_NAME, size, EXT)
        model = ECGLSTM(hidden, layers, bidir)
        if path.exists() and not args.retrain:
            model.load_state_dict(torch.load(path, map_location="cpu"))
            print(f"[lstm] loaded {path.name}")
        else:
            print(f"[lstm] training {size} (hidden={hidden}, layers={layers}, bidir={bidir}) ...")
            train_one(model, X_tr, y_tr, X_val, y_val, device=device,
                      epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, weight_decay=args.weight_decay,
                      class_weight_t=cw)
            torch.save(model.state_dict(), path)
            print(f"[lstm] saved -> {path}")
        m = evaluate(model, X_test, y_test, device=device)
        utils.save_metrics_json(MODEL_NAME, size, m)
        utils.print_metrics_row(MODEL_NAME, size, m, n_params=n_params(model))


if __name__ == "__main__":
    main()
