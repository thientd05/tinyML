"""Aggregate the capacity sweep + operating points into the two comparisons the project
needs, apply the feasibility gate, pick winners, and plot.

This module is just the orchestrator; the stages live in sibling modules:
    load.py         metrics JSONs (+ ESP32 latency table) -> flat rows
    feasibility.py  enrich rows with latency/RAM, apply the hard feasibility gate
    selection.py    best feasible variant per family + cross-family winner
    report.py       summary.csv, winners.json, console tables
    plots.py        within-family / cross-family Pareto / PR-curve figures

Run:
    python -m src.analysis.analyze                          # quality + analytical cost only
    python -m src.analysis.analyze --latency esp32/benchmark_sweep.txt --feature-us 850
    python -m src.analysis.analyze --latency ... --feature-us 850 --pr-curves
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.common import RESULTS
from src.analysis.feasibility import enrich, feasibility
from src.analysis.load import load_records, parse_latency
from src.analysis.plots import plot_cross_pareto, plot_pr_curves, plot_within_family
from src.analysis.selection import select_winners
from src.analysis.writers import print_tables, write_csv, write_winners


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", type=Path, help="ESP32 benchmark table (label | ... | us/beat | heapB)")
    ap.add_argument("--feature-us", type=float, default=None,
                    help="ESP32 feature-extraction us/beat, added to RF/SVM latency")
    ap.add_argument("--pr-curves", action="store_true", help="re-score winners for PR plot")
    args = ap.parse_args()

    rows = load_records()
    if not rows:
        print("[analyze] no records — run the training sweep first (python -m src.models.rf, ...)")
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
