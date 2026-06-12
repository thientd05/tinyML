"""Multi-seed sweep so the comparison is robust to training stochasticity.

The per-beat operating point (precision @ recall 0.95 on DS2) is high-variance for the
raw-beat nets — a single run swings ~0.05 just from the seed, which previously produced a
non-reproducible "winner". This trains every (family, size) over several seeds, keeps the
SEED-MEDIAN checkpoint as the representative (so export/parity + cost stay consistent), and
writes the per-seed mean/std into each model/<family>_<size>_metrics.json under "seeds".
analyze.py then selects winners on the MEAN and the report shows mean +/- std.

Run:  PYTHONPATH=. ./env/bin/python tools/multiseed.py --seeds 5
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np

from src import cost, utils
from src import rf as rf_mod
from src import svm as svm_mod


def _agg(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std()), "values": [float(x) for x in a]}


def _metrics_of(rec):
    """The four DS2 numbers we aggregate across seeds (capability point = recall-0.95)."""
    cap, dep = rec["test_oracle"], rec["test_op"]
    return {"precision_op": cap["precision"], "pr_auc": cap.get("pr_auc", float("nan")),
            "fpr_op": cap["fpr"], "recall_op": cap["recall"], "recall_deploy": dep["recall"]}


def _finalize(family, size, per_seed_recs, per_seed_models, save_fn, ext):
    """Pick the seed-median by precision_op as representative; attach the seed aggregate."""
    precs = [_metrics_of(r)["precision_op"] for r in per_seed_recs]
    med = int(np.argsort(precs)[len(precs) // 2])
    rep = per_seed_recs[med]
    keys = ["precision_op", "pr_auc", "fpr_op", "recall_op", "recall_deploy"]
    rep["seeds"] = {"n": len(per_seed_recs), "representative_index": med,
                    **{k: _agg([_metrics_of(r)[k] for r in per_seed_recs]) for k in keys}}
    save_fn(per_seed_models[med], utils.model_path(family, size, ext))
    utils.save_metrics_json(family, size, rep)
    s = rep["seeds"]
    print(f"  [{family}/{size:<14}] prec@R {s['precision_op']['mean']:.3f}±{s['precision_op']['std']:.3f} "
          f"pr_auc {s['pr_auc']['mean']:.3f}±{s['pr_auc']['std']:.3f} "
          f"rec_dep {s['recall_deploy']['mean']:.3f}  (rep seed idx {med})")


def run_rf(seeds):
    Xtr, ytr, Xv, yv, Xte, yte = utils.build_dataset(mode="features")
    print("[multiseed] RF")
    for size, n_est, depth in rf_mod.SWEEP:
        recs, models = [], []
        for sd in seeds:
            m = rf_mod.build(n_est, depth); m.random_state = sd
            m.fit(Xtr, ytr)
            rec = utils.evaluate_with_operating_point(yv, rf_mod.scores(m, Xv), yte, rf_mod.scores(m, Xte))
            rec.update(family="rf", size=size, cost=cost.rf_cost(m))
            recs.append(rec); models.append(m)
        _finalize("rf", size, recs, models, lambda mdl, p: joblib.dump(mdl, p, compress=3), "pkl")


def run_svm(seeds):
    Xtr, ytr, Xv, yv, Xte, yte = utils.build_dataset(mode="features")
    n_feat = Xtr.shape[1]
    print("[multiseed] SVM")
    for size, kernel, subsample in svm_mod.SWEEP:
        recs, models = [], []
        for sd in seeds:
            Xs, ys = svm_mod.stratified_subsample(Xtr, ytr, subsample, seed=sd)
            pipe = svm_mod.build(kernel)
            clf = pipe.named_steps["clf"]
            if hasattr(clf, "estimator"):       # CalibratedClassifierCV(LinearSVC)
                clf.estimator.random_state = sd
            if hasattr(clf, "random_state"):    # SVC
                clf.random_state = sd
            pipe.fit(Xs, ys)
            rec = utils.evaluate_with_operating_point(yv, svm_mod.scores(pipe, Xv), yte, svm_mod.scores(pipe, Xte))
            rec.update(family="svm", size=size, cost=cost.svm_cost(pipe, n_feat))
            recs.append(rec); models.append(pipe)
        _finalize("svm", size, recs, models, lambda mdl, p: joblib.dump(mdl, p, compress=3), "pkl")


def run_torch(family, seeds):
    import torch
    mod = __import__(f"src.{family}", fromlist=["x"])
    mode = "raw" if family == "cnn" else "lstm"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, ytr, Xv, yv, Xte, yte = utils.build_dataset(mode=mode)
    cw = torch.from_numpy(utils.class_weights(ytr))
    print(f"[multiseed] {family.upper()} (device={dev})")
    for entry in mod.SWEEP:
        size = entry[0]
        recs, states = [], []
        for sd in seeds:
            utils.set_seed(sd)
            if family == "cnn":
                net = mod.ECGCNN(entry[1], entry[2])
            else:
                net = mod.ECGLSTM(entry[1], entry[2], False)
            mod.train_one(net, Xtr, ytr, Xv, yv, device=dev, epochs=15, batch_size=128,
                          lr=1e-3, weight_decay=1e-4, class_weight_t=cw)
            rec = utils.evaluate_with_operating_point(
                yv, mod.scores(net, Xv, device=dev), yte, mod.scores(net, Xte, device=dev))
            costfn = cost.cnn_cost(entry[1], entry[2], utils.BEAT_LEN) if family == "cnn" \
                else cost.lstm_cost(entry[1], entry[2], utils.LSTM_SEQ_LEN)
            rec.update(family=family, size=size, cost=costfn)
            recs.append(rec); states.append({k: v.cpu().clone() for k, v in net.state_dict().items()})
        _finalize(family, size, recs, states,
                  lambda st, p: __import__("torch").save(st, p), "pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--families", nargs="*", default=["rf", "svm", "cnn", "lstm"])
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    print(f"[multiseed] seeds={seeds} families={args.families}")
    if "rf" in args.families:
        run_rf(seeds)
    if "svm" in args.families:
        run_svm(seeds)
    if "cnn" in args.families:
        run_torch("cnn", seeds)
    if "lstm" in args.families:
        run_torch("lstm", seeds)
    print("[multiseed] done — re-run analyze to aggregate.")


if __name__ == "__main__":
    main()
