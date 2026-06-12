"""The three comparison plots: within-family quality-vs-cost, cross-family Pareto, and
PR operating curves for the per-family winners."""
from __future__ import annotations

import importlib
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.analysis.common import FAMILIES, RESULTS
from src.data import build_dataset
from src.io import model_path
from src.models.cnn.sweep import SWEEP as CNN_SWEEP
from src.models.lstm.sweep import SWEEP as LSTM_SWEEP

MARKERS = {"rf": "o", "svm": "s", "cnn": "^", "lstm": "D"}


def plot_within_family(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, ykey, ylab in ((axes[0], "precision_op", f"precision @ recall>={config.TARGET_RECALL}"),
                           (axes[1], "pr_auc", "PR-AUC (threshold-independent)")):
        for fam in FAMILIES:
            pts = sorted([r for r in rows if r["family"] == fam], key=lambda r: r["macs"])
            if not pts:
                continue
            # error band = +/-1 std across training seeds (multi-seed sweep)
            ax.errorbar([p["macs"] for p in pts], [p[ykey] for p in pts],
                        yerr=[p.get(f"{ykey}_std", 0.0) for p in pts],
                        marker=MARKERS[fam], label=fam, capsize=3, elinewidth=1)
        ax.set_xscale("log"); ax.set_xlabel("compute proxy (MACs/ops per beat, log)")
        ax.set_ylabel(ylab); ax.grid(True, alpha=0.3); ax.legend()
    axes[0].set_title("Within-family: quality vs cost, mean +/- std over seeds")
    fig.tight_layout(); fig.savefig(RESULTS / "within_family_quality_vs_cost.png", dpi=130)
    plt.close(fig)


def plot_cross_pareto(rows: list[dict], sel: dict) -> None:
    have_lat = any(not math.isnan(r["latency_ms"]) for r in rows)
    xkey = "latency_ms" if have_lat else "macs"
    xlab = "end-to-end latency (ms/beat, log)" if have_lat else "compute proxy (MACs, log)"
    fig, ax = plt.subplots(figsize=(8, 6))
    for fam in FAMILIES:
        pts = [r for r in rows if r["family"] == fam and not math.isnan(r[xkey])]
        if pts:
            ax.scatter([p[xkey] for p in pts], [p["precision_op"] for p in pts],
                       marker=MARKERS[fam], s=70, label=fam, alpha=0.8)
    if have_lat:
        ax.axvspan(0, config.LATENCY_BUDGET_MS, color="green", alpha=0.06)
        ax.axvline(config.LATENCY_BUDGET_MS, color="green", ls="--", lw=1,
                   label=f"latency budget {config.LATENCY_BUDGET_MS:.0f}ms")
    for fam, w in sel["within"].items():
        if not math.isnan(w[xkey]):
            ax.scatter([w[xkey]], [w["precision_op"]], s=260, facecolors="none",
                       edgecolors="black", linewidths=1.8, zorder=5)
    ax.set_xscale("log"); ax.set_xlabel(xlab)
    ax.set_ylabel(f"precision @ recall>={config.TARGET_RECALL}")
    ax.set_title("Cross-family Pareto (ringed = feasible winner per family)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS / "cross_family_pareto.png", dpi=130)
    plt.close(fig)


def plot_pr_curves(sel: dict) -> None:
    """Re-score the feasible winner of each family on DS2 and draw its PR curve with the
    recall>=target operating point marked. Lazy imports so the rest of analyze stays light."""
    from sklearn.metrics import precision_recall_curve
    import joblib
    fig, ax = plt.subplots(figsize=(8, 6))
    for fam, w in sel["within"].items():
        try:
            if fam in ("rf", "svm"):
                *_, X, y = build_dataset(mode="features")
                mod = importlib.import_module(f"src.models.{fam}")
                clf = joblib.load(model_path(fam, w["size"], "pkl"))
                s = mod.scores(clf, X)
            else:
                import torch
                mode = "raw" if fam == "cnn" else "lstm"
                *_, X, y = build_dataset(mode=mode)
                mod = importlib.import_module(f"src.models.{fam}")
                if fam == "cnn":
                    cfg = next(c for c in CNN_SWEEP if c[0] == w["size"])
                    net = mod.ECGCNN(cfg[1], cfg[2])
                else:
                    cfg = next(c for c in LSTM_SWEEP if c[0] == w["size"])
                    net = mod.ECGLSTM(cfg[1], cfg[2], False)
                net.load_state_dict(torch.load(model_path(fam, w["size"], "pt"),
                                               map_location="cpu"))
                s = mod.scores(net, X, device=torch.device("cpu"))
            prec, rec, _ = precision_recall_curve(y, s)
            ax.plot(rec, prec, label=f"{fam}/{w['size']} (AP={w['pr_auc']:.3f})")
            ax.scatter([w["recall_op"]], [w["precision_op"]], marker=MARKERS[fam], s=90, zorder=5)
        except Exception as e:  # noqa: BLE001
            print(f"[analyze] PR curve skipped for {fam}: {e}")
    ax.axvline(config.TARGET_RECALL, color="red", ls="--", lw=1,
               label=f"target recall {config.TARGET_RECALL}")
    ax.set_xlabel("recall (sensitivity)"); ax.set_ylabel("precision")
    ax.set_title("Operating curves: best feasible model per family (DS2)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS / "pr_curves.png", dpi=130)
    plt.close(fig)
