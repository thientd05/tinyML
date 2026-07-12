"""Text/CSV/JSON output: summary.csv, winners.json, and the console tables."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from src import config
from src.analysis.common import FAMILIES, RESULTS


def _lat_str(r: dict) -> str:
    return "n/a" if math.isnan(r["latency_ms"]) else f"{r['latency_ms']:.2f}ms"


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
    print("\n=== Capacity sweep on DS2 ===")
    print(f"  prec@{config.TARGET_RECALL:g} = precision at target recall measured on DS2 (capability)")
    print("  rec_dep = DS2 recall at the DS1-val-tuned threshold (deployment; '!' = below target)")
    print(f"  budgets: latency<={config.LATENCY_BUDGET_MS:.0f}ms  flash<={config.FLASH_BUDGET_KB}KB  "
          f"target recall={config.TARGET_RECALL}\n")
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

    print(f"=== Winner per family (feasible, max precision @ recall={config.TARGET_RECALL}) ===")
    for fam in FAMILIES:
        w = sel["within"].get(fam)
        if w:
            print(f"  {fam:<5} {w['size']:<16} prec@R={w['precision_op']:.3f} "
                  f"fpr={w['fpr_op']:.3f}  rec_deploy={w['recall_deploy']:.3f} "
                  f"lat={_lat_str(w)} flash={w['flash_kb']:.1f}KB")
        else:
            print(f"  {fam:<5} (no feasible variant)")
    if sel["cross"]:
        c = sel["cross"]
        print(f"\n>>> CROSS-FAMILY WINNER: {c['family']}/{c['size']} — "
              f"precision={c['precision_op']:.3f} @ recall={config.TARGET_RECALL} on DS2, "
              f"fpr={c['fpr_op']:.3f}, latency={_lat_str(c)}, flash={c['flash_kb']:.1f}KB "
              f"(deployment recall at val-tuned threshold: {c['recall_deploy']:.3f})")
    if any(math.isnan(r["latency_ms"]) for r in rows):
        print("\n[note] some latency unknown -> run ESP32 benchmark and pass --latency for "
              "the full feasibility gate; winners above are provisional on the latency axis.")
