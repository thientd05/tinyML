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

from src import cost, utils

MODEL_NAME = "lstm"
EXT = "pt"

# Capacity-knob sweep. Knob = MACs/beat ~ seq_len * 4 * H * (in + H) (gate matmuls,
# the LSTM bottleneck since timesteps are sequential). ~6 points over hidden size +
# a 2-layer point. (label, hidden, num_layers)  [bidirectional is always False]
SWEEP = [
    ("h4",    4,  1),
    ("h8",    8,  1),
    ("h16",   16, 1),
    ("h24",   24, 1),
    ("h32",   32, 1),
    ("h32x2", 32, 2),
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

    # Patient-pure DS1-val (used for best-F1 checkpoint AND threshold tuning).
    X_tr, y_tr, X_val, y_val, X_test, y_test = utils.build_dataset(mode="lstm")
    cw = torch.from_numpy(utils.class_weights(y_tr))
    print(f"[lstm] train={X_tr.shape} val={X_val.shape} test={X_test.shape}  "
          f"class_weights={cw.tolist()}")

    print(f"[lstm] === sweep (DS2; threshold tuned on DS1-val to recall>={utils.TARGET_RECALL}) ===")
    for size, hidden, layers in SWEEP:
        path = utils.model_path(MODEL_NAME, size, EXT)
        model = ECGLSTM(hidden, layers, False)
        if path.exists() and not args.retrain:
            model.load_state_dict(torch.load(path, map_location="cpu"))
            print(f"[lstm] loaded {path.name}")
        else:
            print(f"[lstm] training {size} (hidden={hidden}, layers={layers}) ...")
            train_one(model, X_tr, y_tr, X_val, y_val, device=device,
                      epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, weight_decay=args.weight_decay,
                      class_weight_t=cw)
            torch.save(model.state_dict(), path)
            print(f"[lstm] saved -> {path}")
        rec = utils.evaluate_with_operating_point(
            y_val, scores(model, X_val, device=device),
            y_test, scores(model, X_test, device=device))
        rec.update(family=MODEL_NAME, size=size,
                   cost=cost.lstm_cost(hidden, layers, utils.LSTM_SEQ_LEN))
        utils.save_metrics_json(MODEL_NAME, size, rec)
        o, c = rec["test_op"], rec["cost"]
        print(f"  [{size:<16}] tau={rec['threshold']:.3f}  recall={o['recall']:.4f} "
              f"prec={o['precision']:.4f} fpr={o['fpr']:.4f} f1={o['f1']:.4f} "
              f"pr_auc={o.get('pr_auc', float('nan')):.4f}  params={c['n_params']} macs={c['macs']:.0f}")


if __name__ == "__main__":
    main()
