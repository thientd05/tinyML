"""CLI entrypoint: train/load the LSTM capacity sweep and dump per-variant metrics."""
from __future__ import annotations
import argparse
import torch
from src.config import LSTM_SEQ_LEN, TARGET_RECALL
from src.data import build_dataset
from src.evaluation import class_weights, evaluate_with_operating_point
from src.io import model_path, save_metrics_json
from src.models import cost
from src.models.lstm import EXT, MODEL_NAME, SWEEP, ECGLSTM, scores, train_one
from src.seeding import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true")  # bắt buộc train lại dù đã có checkpoint
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    set_seed(42)  # cố định seed để kết quả tái lập được
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[lstm] device={device}")

    # Patient-pure DS1-val (used for best-F1 checkpoint AND threshold tuning).
    # X_tr/y_tr: tập train; X_val/y_val: tập val của DS1 (chọn model tốt nhất + tune threshold);
    # X_test/y_test: tập test DS2 (đánh giá cuối cùng).
    X_tr, y_tr, X_val, y_val, X_test, y_test = build_dataset(mode="lstm")

    # Trọng số cho từng class để xử lý mất cân bằng dữ liệu khi tính loss.
    cw = torch.from_numpy(class_weights(y_tr))
    print(f"[lstm] train={X_tr.shape} val={X_val.shape} test={X_test.shape}  "
          f"class_weights={cw.tolist()}")

    print(f"[lstm] === sweep (DS2; threshold tuned on DS1-val to recall>={TARGET_RECALL}) ===")

    # Lặp qua các biến thể (variant) về dung lượng model: (tên size, hidden size, số layer).
    for size, hidden, layers in SWEEP:
        path = model_path(MODEL_NAME, size, EXT)
        model = ECGLSTM(hidden, layers, False)  # False = không dùng bidirectional

        if path.exists() and not args.retrain:
            # Có checkpoint sẵn -> load lại, không train lại.
            model.load_state_dict(torch.load(path, map_location="cpu"))
            print(f"[lstm] loaded {path.name}")
        else:
            print(f"[lstm] training {size} (hidden={hidden}, layers={layers}) ...")
            train_one(model, X_tr, y_tr, X_val, y_val, device=device,
                      epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, weight_decay=args.weight_decay,
                      class_weight_t=cw)
            torch.save(model.state_dict(), path)
            print(f"[lstm] saved -> {path}")

        # Tính điểm số (scores) trên val/test, chọn ngưỡng (threshold) từ val
        # sao cho đạt TARGET_RECALL, rồi áp ngưỡng đó lên test để lấy metric cuối.
        rec = evaluate_with_operating_point(
            y_val, scores(model, X_val, device=device),
            y_test, scores(model, X_test, device=device))

        # Gắn thêm metadata: tên family của model, size của variant, và chi phí tính toán/tham số.
        rec.update(family=MODEL_NAME, size=size,
                   cost=cost.lstm_cost(hidden, layers, LSTM_SEQ_LEN))

        # Lưu metrics của từng variant ra file JSON để so sánh/báo cáo sau.
        save_metrics_json(MODEL_NAME, size, rec)

        # In dòng tổng kết: threshold, các chỉ số trên test, và chi phí model.
        o, c = rec["test_op"], rec["cost"]
        print(f"  [{size:<16}] tau={rec['threshold']:.3f}  recall={o['recall']:.4f} "
              f"prec={o['precision']:.4f} fpr={o['fpr']:.4f} f1={o['f1']:.4f} "
              f"pr_auc={o.get('pr_auc', float('nan')):.4f}  params={c['n_params']} macs={c['macs']:.0f}")


if __name__ == "__main__":
    main()
