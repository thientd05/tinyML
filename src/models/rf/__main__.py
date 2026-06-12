"""CLI entrypoint: train/load the RF capacity sweep and dump per-variant metrics."""
from __future__ import annotations

import argparse

import joblib

from src.config import TARGET_RECALL
from src.data import build_dataset
from src.evaluation import evaluate_with_operating_point
from src.io import model_path, save_metrics_json
from src.models import cost
from src.models.rf import EXT, MODEL_NAME, SWEEP, build, scores, train
from src.seeding import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="ignore existing checkpoints")
    args = parser.parse_args()

    set_seed(42)
    X_train, y_train, X_val, y_val, X_test, y_test = build_dataset(mode="features")
    print(f"[rf] train={X_train.shape} val={X_val.shape} test={X_test.shape}  "
          f"pos_train={y_train.mean():.3f} pos_val={y_val.mean():.3f} pos_test={y_test.mean():.3f}")

    print(f"[rf] === sweep (DS2; threshold tuned on DS1-val to recall>={TARGET_RECALL}) ===")
    for size, n_est, depth in SWEEP:
        path = model_path(MODEL_NAME, size, EXT)
        if path.exists() and not args.retrain:
            model = joblib.load(path)
            print(f"[rf] loaded {path.name}")
        else:
            print(f"[rf] training {size} (n_estimators={n_est}, max_depth={depth}) ...")
            model = train(build(n_est, depth), X_train, y_train)
            joblib.dump(model, path, compress=3)
            print(f"[rf] saved -> {path}")
        rec = evaluate_with_operating_point(
            y_val, scores(model, X_val), y_test, scores(model, X_test))
        rec.update(family=MODEL_NAME, size=size, cost=cost.rf_cost(model))
        save_metrics_json(MODEL_NAME, size, rec)
        _print_row(size, rec)


def _print_row(size: str, rec: dict) -> None:
    o, c = rec["test_op"], rec["cost"]
    print(f"  [{size:<16}] tau={rec['threshold']:.3f}  recall={o['recall']:.4f} "
          f"prec={o['precision']:.4f} fpr={o['fpr']:.4f} f1={o['f1']:.4f} "
          f"pr_auc={o.get('pr_auc', float('nan')):.4f}  nodes={c['n_nodes']} ops={c['macs']:.0f}")


if __name__ == "__main__":
    main()
