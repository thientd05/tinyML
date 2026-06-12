"""Winner selection: best feasible variant per family, then cross-family.

Every variant is compared at the SAME sensitivity (precision @ target recall on DS2),
so selection = maximize that precision.
"""
from __future__ import annotations

import math

from src.analysis.common import FAMILIES


def _key(r: dict):
    # Tie-break on lower latency (or lower MACs if latency unknown). Calibration transfer
    # (recall_holds) is reported separately, not used to rank, since it is fixable by
    # recalibration and orthogonal to model capability.
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
