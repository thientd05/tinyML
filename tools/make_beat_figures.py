"""Render illustrative ECG beat figures straight from MIT-BIH for the report's
section 6.2 (beat morphology). Every plotted beat is a REAL beat pulled from the
dataset, annotated with its source (record + sample index + symbol), so the
report's morphology claims are backed by visible evidence.

Outputs (results/):
  beat_morphology_gallery.png  — N / VEB / SVEB / LBBB / RBBB side by side
  beat_easy_vs_hard.png        — within-patient Normal-vs-VEB (easy) and
                                 Normal-vs-SVEB (hard) overlays

Run from repo root:  PYTHONPATH=. ./env/bin/python3.10 tools/make_beat_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (  # noqa: E402
    BEAT_HALF, FS, REALIGN_WINDOW, RESULTS_DIR, _load_record, aami_binary_label,
)

# x-axis in milliseconds, R-peak at 0
T_MS = (np.arange(2 * BEAT_HALF) - BEAT_HALF) / FS * 1000.0

# AAMI original symbol -> (super-class, human label, binary)
SYMBOL_INFO = {
    "N": ("Normal", "Nhịp xoang bình thường", 0),
    "V": ("VEB", "Ngoại tâm thu THẤT (dễ)", 1),
    "A": ("SVEB", "Ngoại tâm thu TRÊN THẤT (khó)", 1),
    "L": ("Normal", "Block nhánh trái – LBBB", 0),
    "R": ("Normal", "Block nhánh phải – RBBB", 0),
}
# candidate records per symbol (first that yields a clean beat wins)
CANDIDATES = {
    "N": ["100", "101", "103", "112"],
    "V": ["233", "119", "106", "200", "203"],
    "A": ["209", "232", "222", "220", "223"],
    "L": ["109", "111", "214", "207"],
    "R": ["118", "124", "212", "231"],
}
COLOR = {"N": "#1f77b4", "V": "#d62728", "A": "#ff7f0e", "L": "#2ca02c", "R": "#9467bd"}


def _realigned_window(sig, s):
    """Window of +/-BEAT_HALF around annotation sample s, snapped to the local |signal|
    max within +/-REALIGN_WINDOW (matches extract_beats). Returns (beat, center) or None."""
    a, b = max(0, s - REALIGN_WINDOW), min(len(sig), s + REALIGN_WINDOW)
    if b <= a:
        return None
    center = a + int(np.argmax(np.abs(sig[a:b])))
    lo, hi = center - BEAT_HALF, center + BEAT_HALF
    if lo < 0 or hi > len(sig):
        return None
    return sig[lo:hi].copy(), center


def collect_beats(record_id, symbol, want=40):
    """Return list of (beat, ann_sample) for `symbol` in this record."""
    sig, _raw, ann = _load_record(record_id)
    out = []
    for s, sym in zip(ann.sample, ann.symbol):
        if sym != symbol:
            continue
        w = _realigned_window(sig, int(s))
        if w is None:
            continue
        out.append((w[0], int(s)))
        if len(out) >= want:
            break
    return out


def representative(beats):
    """Pick the beat closest to the mean shape (median correlation) -> a clean exemplar."""
    if not beats:
        return None
    X = np.stack([b for b, _ in beats])
    mean = X.mean(0)
    mean_n = (mean - mean.mean()) / (mean.std() + 1e-8)
    best, best_c = 0, -2.0
    for i, b in enumerate(X):
        bn = (b - b.mean()) / (b.std() + 1e-8)
        c = float(np.mean(bn * mean_n))
        if c > best_c:
            best_c, best = c, i
    return beats[best][0], beats[best][1], mean


def pick(symbol):
    """First candidate record that yields beats -> (record_id, beat, ann_sample, mean)."""
    for rid in CANDIDATES[symbol]:
        beats = collect_beats(rid, symbol)
        rep = representative(beats)
        if rep is not None:
            return rid, rep[0], rep[1], rep[2]
    raise RuntimeError(f"no beat found for symbol {symbol}")


def gallery():
    order = ["N", "V", "A", "L", "R"]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2), sharey=True)
    for ax, sym in zip(axes, order):
        rid, beat, s, _ = pick(sym)
        grp, desc, binlab = SYMBOL_INFO[sym]
        ax.plot(T_MS, beat, color=COLOR[sym], lw=1.6)
        ax.axvline(0, color="grey", lw=0.6, ls=":")
        ax.set_title(f"‘{sym}’ → {grp}  (nhãn {binlab})\n{desc}", fontsize=9)
        ax.set_xlabel(f"record {rid} @ mẫu {s:,}\nthời gian (ms)", fontsize=8)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("biên độ (z-score)")
    fig.suptitle("Hình dạng nhịp theo nhóm AAMI — nhịp thật từ MIT-BIH (R-peak tại 0 ms)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = RESULTS_DIR / "beat_morphology_gallery.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def _within_patient_pair(records, target):
    """Find a record (from `records`) holding BOTH N and `target`; return
    (rid, normal_mean, normal_rep, target_rep, n_sample, t_sample)."""
    for rid in records:
        nb = collect_beats(rid, "N")
        tb = collect_beats(rid, target)
        if nb and tb:
            n_rep = representative(nb)
            t_rep = representative(tb)
            return rid, n_rep[2], n_rep[0], t_rep[0], n_rep[1], t_rep[1]
    raise RuntimeError(f"no within-patient N+{target} record in {records}")


def easy_vs_hard():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # VEB: easy — distinct from this patient's own normal
    rid, n_mean, n_rep, v_rep, ns, vs = _within_patient_pair(
        ["233", "119", "106", "200"], "V")
    axL.plot(T_MS, n_rep, color="#1f77b4", lw=1.8, label=f"Normal (N) @ mẫu {ns:,}")
    axL.plot(T_MS, v_rep, color="#d62728", lw=1.8, label=f"VEB (V) @ mẫu {vs:,}")
    axL.axvline(0, color="grey", lw=0.6, ls=":")
    axL.set_title(f"DỄ — VEB méo bè, KHÁC HẲN nhịp thường\n(cùng bệnh nhân, record {rid})",
                  fontsize=10)
    axL.set_xlabel("thời gian (ms)"); axL.set_ylabel("biên độ (z-score)")
    axL.legend(fontsize=8); axL.grid(alpha=0.25)

    # SVEB: hard — near-identical to this patient's own normal
    rid2, n2_mean, n2_rep, a_rep, ns2, as2 = _within_patient_pair(
        ["209", "232", "222", "220", "223"], "A")
    axR.plot(T_MS, n2_rep, color="#1f77b4", lw=1.8, label=f"Normal (N) @ mẫu {ns2:,}")
    axR.plot(T_MS, a_rep, color="#ff7f0e", lw=1.8, label=f"SVEB (A) @ mẫu {as2:,}")
    axR.axvline(0, color="grey", lw=0.6, ls=":")
    axR.set_title(f"KHÓ — SVEB gần Y HỆT nhịp thường\n(cùng bệnh nhân, record {rid2})",
                  fontsize=10)
    axR.set_xlabel("thời gian (ms)")
    axR.legend(fontsize=8); axR.grid(alpha=0.25)

    fig.suptitle("Vì sao SVEB kéo precision xuống: hình dạng quyết định độ khó "
                 "(nhịp thật từ MIT-BIH)", fontsize=12, y=1.01)
    fig.tight_layout()
    out = RESULTS_DIR / "beat_easy_vs_hard.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    gallery()
    easy_vs_hard()
