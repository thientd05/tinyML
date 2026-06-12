"""CLI entrypoint: train/load the SVM capacity sweep and dump per-variant metrics."""
from __future__ import annotations

import argparse

import joblib

from src.config import TARGET_RECALL
from src.data import build_dataset
from src.evaluation import evaluate_with_operating_point
from src.io import model_path, save_metrics_json
from src.models import cost
from src.models.svm import EXT, MODEL_NAME, SWEEP, build, scores, stratified_subsample, train
from src.seeding import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    X_train, y_train, X_val, y_val, X_test, y_test = build_dataset(mode="features")
    print(f"[svm] train={X_train.shape} val={X_val.shape} test={X_test.shape}  "
          f"pos_train={y_train.mean():.3f} pos_val={y_val.mean():.3f} pos_test={y_test.mean():.3f}")
    n_feat = X_train.shape[1]

    print(f"[svm] === sweep (DS2; threshold tuned on DS1-val to recall>={TARGET_RECALL}) ===")
    for size, kernel, subsample in SWEEP:
        path = model_path(MODEL_NAME, size, EXT)
        if path.exists() and not args.retrain:
            pipe = joblib.load(path)
            print(f"[svm] loaded {path.name}")
        else:
            Xs, ys = stratified_subsample(X_train, y_train, subsample)
            print(f"[svm] training {size} (kernel={kernel}, n_train={len(ys)}) ...")
            pipe = train(build(kernel), Xs, ys)
            joblib.dump(pipe, path, compress=3)
            print(f"[svm] saved -> {path}")
        rec = evaluate_with_operating_point(
            y_val, scores(pipe, X_val), y_test, scores(pipe, X_test))
        rec.update(family=MODEL_NAME, size=size, cost=cost.svm_cost(pipe, n_feat))
        save_metrics_json(MODEL_NAME, size, rec)
        o, c = rec["test_op"], rec["cost"]
        print(f"  [{size:<16}] tau={rec['threshold']:.3f}  recall={o['recall']:.4f} "
              f"prec={o['precision']:.4f} fpr={o['fpr']:.4f} f1={o['f1']:.4f} "
              f"pr_auc={o.get('pr_auc', float('nan')):.4f}  n_sv={c['n_sv']} macs={c['macs']:.0f}")


if __name__ == "__main__":
    main()
