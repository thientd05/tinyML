"""Aggregate the capacity sweep + operating points into the two comparisons the project
needs, apply the feasibility gate, pick winners, and plot.

Quality (recall/precision/PR-AUC/...) is read from the full-DS2 metrics JSONs written by
the training scripts — trustworthy because on-device parity == 1.0000. Per-beat LATENCY
and working RAM come from the ESP32 benchmark log (--latency). RF/SVM get the measured
feature-extraction cost (--feature-us, from the ESP32 microbench) added to inference
latency so cross-family "detection time" is apples-to-apples (CNN/LSTM eat raw beats,
feature cost = 0).

Run:
    python -m src.analyze                                   # quality + analytical cost only
    python -m src.analyze --latency esp32/benchmark_sweep.txt --feature-us 850
    python -m src.analyze --latency ... --feature-us 850 --pr-curves
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import utils
from src.rf import SWEEP as RF_SWEEP
from src.svm import SWEEP as SVM_SWEEP
from src.cnn import SWEEP as CNN_SWEEP
from src.lstm import SWEEP as LSTM_SWEEP

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FAMILIES = ["rf", "svm", "cnn", "lstm"]
# family -> ordered list of sweep labels (defines plot order + filters out stale files)
SWEEP_LABELS = {
    "rf": [s[0] for s in RF_SWEEP], "svm": [s[0] for s in SVM_SWEEP],
    "cnn": [s[0] for s in CNN_SWEEP], "lstm": [s[0] for s in LSTM_SWEEP],
}
MARKERS = {"rf": "o", "svm": "s", "cnn": "^", "lstm": "D"}


# ---------------- load ----------------
def load_records() -> list[dict]:
    """One flat row per sweep variant from model/<family>_<size>_metrics.json."""
    rows = []
    for fam in FAMILIES:
        for order, label in enumerate(SWEEP_LABELS[fam]):
            p = utils.MODEL_DIR / f"{fam}_{label}_metrics.json"
            if not p.exists():
                print(f"[analyze] WARN missing {p.name} (train this variant first)")
                continue
            d = json.loads(p.read_text())
            if "test_op" not in d:  # stale 4-size schema
                print(f"[analyze] WARN {p.name} has old schema, skipping")
                continue
            # capability point = precision AT target recall on DS2 (fair comparison axis);
            # deployment point = DS1-val-tuned threshold applied to DS2 (transfer reality).
            cap = d.get("test_oracle", d["test_op"])
            dep, base, cost = d["test_op"], d["test_05"], d["cost"]
            # multi-seed aggregate (tools/multiseed.py): report the MEAN across seeds so a
            # single high-variance run can't crown a non-reproducible winner. Falls back to
            # the single representative value (std=0) for any model without a seeds block.
            sd = d.get("seeds")

            def ms(key, fallback):
                return (sd[key]["mean"], sd[key]["std"]) if (sd and key in sd) else (fallback, 0.0)

            p_m, p_s = ms("precision_op", cap["precision"])
            a_m, a_s = ms("pr_auc", cap.get("pr_auc", float("nan")))
            rd_m, rd_s = ms("recall_deploy", dep["recall"])
            f_m, _ = ms("fpr_op", cap["fpr"])
            ro_m, _ = ms("recall_op", cap["recall"])
            rows.append({
                "family": fam, "size": label, "order": order,
                "macs": float(cost["macs"]), "n_params": int(cost["n_params"]),
                "flash_kb": cost["flash_bytes"] / 1024.0,
                "threshold": d["threshold"], "n_seeds": (sd["n"] if sd else 1),
                # capability @ target recall (MEAN across seeds; *_std = spread):
                "recall_op": ro_m, "precision_op": p_m, "precision_op_std": p_s,
                "fpr_op": f_m, "f1_op": cap["f1"],
                "roc_auc": cap.get("roc_auc", float("nan")),
                "pr_auc": a_m, "pr_auc_std": a_s,
                # deployment (val-tuned threshold) realized on DS2:
                "recall_deploy": rd_m, "recall_deploy_std": rd_s,
                "precision_deploy": dep["precision"],
                "recall_05": base["recall"], "precision_05": base["precision"],
            })
    return rows


def parse_latency(path: Path) -> dict[str, dict]:
    """Parse the ESP32 benchmark table: `label | ... | us/beat | heapB`."""
    out = {}
    for line in path.read_text().splitlines():
        if "|" not in line:
            continue
        cols = [c.strip() for c in line.split("|")]
        try:
            out[cols[0]] = {"us_beat": float(cols[-2]), "heap_b": float(cols[-1])}
        except (ValueError, IndexError):
            continue  # header / non-data line
    return out


# ---------------- enrich + feasibility ----------------
def enrich(rows: list[dict], latency: dict, feature_us: float | None) -> None:
    for r in rows:
        # firmware prints full names ("{family}_{label}"); JSON rows key on bare label
        lat = latency.get(f"{r['family']}_{r['size']}") or latency.get(r["size"])
        r["infer_us"] = lat["us_beat"] if lat else float("nan")
        # feature extraction (wavelet db4 + FFT) only feeds RF/SVM; raw-beat nets pay 0.
        r["feature_us"] = (feature_us or 0.0) if r["family"] in ("rf", "svm") else 0.0
        r["latency_ms"] = (r["infer_us"] + r["feature_us"]) / 1000.0
        r["ram_kb"] = (lat["heap_b"] / 1024.0) if lat else float("nan")


def feasibility(r: dict) -> None:
    r["fits_flash"] = r["flash_kb"] <= utils.FLASH_BUDGET_KB
    r["fits_ram"] = (math.isnan(r["ram_kb"]) or r["ram_kb"] <= utils.RAM_WORK_BUDGET_KB)
    if math.isnan(r["latency_ms"]):
        r["meets_latency"] = None  # device latency not measured yet
    else:
        r["meets_latency"] = r["latency_ms"] <= utils.LATENCY_BUDGET_MS
    # provisionally feasible if no hard constraint is violated (unknown latency != violation)
    r["feasible"] = r["fits_flash"] and r["fits_ram"] and (r["meets_latency"] is not False)
    # did the DS1-val-tuned threshold still clear the sensitivity floor on DS2? (transfer)
    r["recall_holds"] = r["recall_deploy"] >= utils.TARGET_RECALL


# ---------------- selection ----------------
def _key(r: dict):
    # Every variant is compared at the SAME sensitivity (precision @ target recall on DS2),
    # so selection = maximize that precision; tie-break on lower latency (or lower MACs if
    # latency unknown). Calibration transfer (recall_holds) is reported separately, not
    # used to rank, since it is fixable by recalibration and orthogonal to model capability.
    lat = r["latency_ms"] if not math.isnan(r["latency_ms"]) else r["macs"]
    return (r["precision_op"], -lat)


def select_winners(rows: list[dict]) -> dict:
    winners = {}
    for fam in FAMILIES:
        feas = [r for r in rows if r["family"] == fam and r["feasible"]]
        if feas:
            winners[fam] = max(feas, key=_key)
    cross = max(winners.values(), key=_key) if winners else None
    return {"within": winners, "cross": cross}


# ---------------- output ----------------
def write_winners(sel: dict) -> Path:
    """Persist the auto-selected winners so the report doesn't hard-code them (they can
    move after a data/cleaning change). within = {family: size}; cross = [family, size]."""
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "winners.json"
    cross = sel["cross"]
    out.write_text(json.dumps({
        "within": {fam: w["size"] for fam, w in sel["within"].items()},
        "cross": [cross["family"], cross["size"]] if cross else None,
    }, indent=2))
    return out


def write_csv(rows: list[dict]) -> Path:
    import csv
    RESULTS.mkdir(exist_ok=True)
    cols = ["family", "size", "macs", "n_params", "flash_kb", "threshold", "n_seeds",
            "recall_op", "precision_op", "precision_op_std", "fpr_op", "f1_op", "roc_auc",
            "pr_auc", "pr_auc_std", "recall_deploy", "recall_deploy_std", "precision_deploy",
            "recall_05", "precision_05",
            "infer_us", "feature_us", "latency_ms", "ram_kb",
            "fits_flash", "fits_ram", "meets_latency", "feasible", "recall_holds"]
    out = RESULTS / "summary.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["family"], r["order"])):
            w.writerow(r)
    return out


def print_tables(rows: list[dict], sel: dict) -> None:
    print(f"\n=== Capacity sweep on DS2 ===")
    print(f"  prec@{utils.TARGET_RECALL:g} = precision at target recall measured on DS2 (capability)")
    print(f"  rec_dep = DS2 recall at the DS1-val-tuned threshold (deployment; '!' = below target)")
    print(f"  budgets: latency<={utils.LATENCY_BUDGET_MS:.0f}ms  flash<={utils.FLASH_BUDGET_KB}KB  "
          f"target recall={utils.TARGET_RECALL}\n")
    hdr = (f"  {'variant':<16} {'macs':>9} {'flash':>7} {'lat_ms':>8} "
           f"{'prec@R(±std)':>14} {'fpr':>6} {'prauc':>6} {'rec_dep':>8} {'feas':>5}")
    for fam in FAMILIES:
        print(f"-- {fam} --")
        print(hdr)
        for r in [x for x in rows if x["family"] == fam]:
            lat = "     n/a" if math.isnan(r["latency_ms"]) else f"{r['latency_ms']:8.2f}"
            star = "*" if sel["within"].get(fam) is r else " "
            rk = " " if r["recall_holds"] else "!"
            prec = f"{r['precision_op']:.3f}±{r.get('precision_op_std', 0.0):.2f}"
            print(f"{star} {r['size']:<16} {r['macs']:>9.0f} {r['flash_kb']:>6.1f}K {lat} "
                  f"{prec:>14} {r['fpr_op']:>6.3f} {r['pr_auc']:>6.3f} "
                  f"{r['recall_deploy']:>7.3f}{rk} {str(r['feasible']):>5}")
        print()

    def _lat(r):
        return "n/a" if math.isnan(r["latency_ms"]) else f"{r['latency_ms']:.2f}ms"
    print(f"=== Winner per family (feasible, max precision @ recall={utils.TARGET_RECALL}) ===")
    for fam in FAMILIES:
        w = sel["within"].get(fam)
        if w:
            print(f"  {fam:<5} {w['size']:<16} prec@R={w['precision_op']:.3f} "
                  f"fpr={w['fpr_op']:.3f}  rec_deploy={w['recall_deploy']:.3f} "
                  f"lat={_lat(w)} flash={w['flash_kb']:.1f}KB")
        else:
            print(f"  {fam:<5} (no feasible variant)")
    if sel["cross"]:
        c = sel["cross"]
        print(f"\n>>> CROSS-FAMILY WINNER: {c['family']}/{c['size']} — "
              f"precision={c['precision_op']:.3f} @ recall={utils.TARGET_RECALL} on DS2, "
              f"fpr={c['fpr_op']:.3f}, latency={_lat(c)}, flash={c['flash_kb']:.1f}KB "
              f"(deployment recall at val-tuned threshold: {c['recall_deploy']:.3f})")
    if any(math.isnan(r["latency_ms"]) for r in rows):
        print("\n[note] some latency unknown -> run ESP32 benchmark and pass --latency for "
              "the full feasibility gate; winners above are provisional on the latency axis.")


# ---------------- plots ----------------
def plot_within_family(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, ykey, ylab in ((axes[0], "precision_op", f"precision @ recall>={utils.TARGET_RECALL}"),
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
        ax.axvspan(0, utils.LATENCY_BUDGET_MS, color="green", alpha=0.06)
        ax.axvline(utils.LATENCY_BUDGET_MS, color="green", ls="--", lw=1,
                   label=f"latency budget {utils.LATENCY_BUDGET_MS:.0f}ms")
    for fam, w in sel["within"].items():
        if not math.isnan(w[xkey]):
            ax.scatter([w[xkey]], [w["precision_op"]], s=260, facecolors="none",
                       edgecolors="black", linewidths=1.8, zorder=5)
    ax.set_xscale("log"); ax.set_xlabel(xlab)
    ax.set_ylabel(f"precision @ recall>={utils.TARGET_RECALL}")
    ax.set_title("Cross-family Pareto (ringed = feasible winner per family)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS / "cross_family_pareto.png", dpi=130)
    plt.close(fig)


def plot_pr_curves(sel: dict) -> None:
    """Re-score the feasible winner of each family on DS2 and draw its PR curve with the
    recall>=target operating point marked. Lazy imports so the rest of analyze stays light."""
    from sklearn.metrics import precision_recall_curve
    import joblib, numpy as np
    fig, ax = plt.subplots(figsize=(8, 6))
    for fam, w in sel["within"].items():
        try:
            if fam in ("rf", "svm"):
                *_, X, y = utils.build_dataset(mode="features")
                mod = joblib.load(utils.model_path(fam, w["size"], "pkl"))
                s = (__import__(f"src.{fam}", fromlist=["scores"]).scores(mod, X))
            else:
                import torch
                mode = "raw" if fam == "cnn" else "lstm"
                *_, X, y = utils.build_dataset(mode=mode)
                m = __import__(f"src.{fam}", fromlist=["x"])
                if fam == "cnn":
                    cfg = next(c for c in CNN_SWEEP if c[0] == w["size"])
                    net = m.ECGCNN(cfg[1], cfg[2])
                else:
                    cfg = next(c for c in LSTM_SWEEP if c[0] == w["size"])
                    net = m.ECGLSTM(cfg[1], cfg[2], False)
                net.load_state_dict(torch.load(utils.model_path(fam, w["size"], "pt"),
                                               map_location="cpu"))
                s = m.scores(net, X, device=torch.device("cpu"))
            prec, rec, _ = precision_recall_curve(y, s)
            ax.plot(rec, prec, label=f"{fam}/{w['size']} (AP={w['pr_auc']:.3f})")
            ax.scatter([w["recall_op"]], [w["precision_op"]], marker=MARKERS[fam], s=90, zorder=5)
        except Exception as e:  # noqa: BLE001
            print(f"[analyze] PR curve skipped for {fam}: {e}")
    ax.axvline(utils.TARGET_RECALL, color="red", ls="--", lw=1,
               label=f"target recall {utils.TARGET_RECALL}")
    ax.set_xlabel("recall (sensitivity)"); ax.set_ylabel("precision")
    ax.set_title("Operating curves: best feasible model per family (DS2)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS / "pr_curves.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", type=Path, help="ESP32 benchmark table (label | ... | us/beat | heapB)")
    ap.add_argument("--feature-us", type=float, default=None,
                    help="ESP32 feature-extraction us/beat, added to RF/SVM latency")
    ap.add_argument("--pr-curves", action="store_true", help="re-score winners for PR plot")
    args = ap.parse_args()

    rows = load_records()
    if not rows:
        print("[analyze] no records — run the training sweep first (python -m src.rf, ...)")
        return
    latency = parse_latency(args.latency) if args.latency and args.latency.exists() else {}
    if args.latency and not latency:
        print(f"[analyze] WARN no rows parsed from {args.latency}")
    enrich(rows, latency, args.feature_us)
    for r in rows:
        feasibility(r)
    sel = select_winners(rows)

    csv_path = write_csv(rows)
    write_winners(sel)
    print_tables(rows, sel)
    RESULTS.mkdir(exist_ok=True)
    plot_within_family(rows)
    plot_cross_pareto(rows, sel)
    if args.pr_curves:
        plot_pr_curves(sel)
    print(f"\n[analyze] wrote {csv_path} and plots to {RESULTS}/")


if __name__ == "__main__":
    main()
