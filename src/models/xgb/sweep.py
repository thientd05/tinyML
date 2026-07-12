"""XGBoost capacity-knob sweep.

Knob = total tree node count (~ n_estimators * 2**depth), the same axis as the RF sweep,
so the bagging-vs-boosting comparison is drawn on one flash/traversal-cost scale. ~6
log-spaced points. Depths stay shallow (2-6) because boosting fits residuals with weak
learners — the deep-tree end of the axis is already covered by RF.
The learning rate is FIXED (see estimator.build) so capacity is the only knob.
(label, n_estimators, max_depth)
"""
from __future__ import annotations

SWEEP = [
    ("x50_d2",  50,  2),
    ("x100_d2", 100, 2),
    ("x100_d3", 100, 3),
    ("x200_d4", 200, 4),
    ("x300_d5", 300, 5),
    ("x400_d6", 400, 6),
]
