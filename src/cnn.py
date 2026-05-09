"""1D CNN for ECG anomaly detection (binary), four sizes targeting ESP32.

Run:
    python -m src.cnn
    python -m src.cnn --retrain
    python -m src.cnn --epochs 20 --batch-size 256

Saves model/cnn_<size>.pt (state_dict). Metrics dumped as cnn_<size>_metrics.json.
"""
from __future__ import annotations

import argparse
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src import utils

MODEL_NAME = "cnn"
EXT = "pt"

# (size, conv_channels, fc_hidden)
SIZES = [
    ("tiny",   [8],            None),   # GAP -> Linear(8, 2)
    ("small",  [8, 16],        16),
    ("medium", [16, 32, 32],   32),
    ("large",  [16, 32, 64, 64], 64),
]


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
    def __init__(self, channels: List[int], fc_hidden: int | None, in_len: int = utils.BEAT_LEN):
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


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    X_t = torch.from_numpy(X).float().unsqueeze(1)  # [N, 1, L]
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
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    train_loader = make_loader(X_tr, y_tr, batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size * 2, shuffle=False)

    best_f1, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
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
        y_pred = np.concatenate(ps); y_true = np.concatenate(ys)
        from sklearn.metrics import f1_score
        f1 = f1_score(y_true, y_pred, zero_division=0)
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
    print(f"[cnn] device={device}")

    X_train_full, y_train_full, X_test, y_test = utils.build_dataset(mode="raw")
    X_tr, y_tr, X_val, y_val = split_train_val(X_train_full, y_train_full)
    cw = torch.from_numpy(utils.class_weights(y_tr))
    print(f"[cnn] train={X_tr.shape}  val={X_val.shape}  test={X_test.shape}  "
          f"class_weights={cw.tolist()}")

    print("[cnn] === per-size results (test set) ===")
    for size, channels, fc in SIZES:
        path = utils.model_path(MODEL_NAME, size, EXT)
        model = ECGCNN(channels, fc)
        if path.exists() and not args.retrain:
            model.load_state_dict(torch.load(path, map_location="cpu"))
            print(f"[cnn] loaded {path.name}")
        else:
            print(f"[cnn] training {size} (channels={channels}, fc={fc}) ...")
            train_one(model, X_tr, y_tr, X_val, y_val, device=device,
                      epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, weight_decay=args.weight_decay,
                      class_weight_t=cw)
            torch.save(model.state_dict(), path)
            print(f"[cnn] saved -> {path}")
        m = evaluate(model, X_test, y_test, device=device)
        utils.save_metrics_json(MODEL_NAME, size, m)
        utils.print_metrics_row(MODEL_NAME, size, m, n_params=n_params(model))


if __name__ == "__main__":
    main()
