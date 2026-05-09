# pr_tinyml — Phát hiện bất thường trong tín hiệu ECG cho thiết bị biên (ESP32)

Pipeline so sánh **4 mô hình** ML/DL cho bài toán phân loại nhị phân
**Normal vs Abnormal** trên tín hiệu điện tâm đồ. Mục tiêu cuối cùng là chạy
trên **ESP32**; giai đoạn này chỉ huấn luyện trên PC (GPU GTX 1650).

## 1. Bài toán

- **Đầu vào**: 1 nhịp tim (cửa sổ ±100 mẫu quanh đỉnh R, ~555 ms ở 360 Hz) trên
  kênh MLII của MIT-BIH. ESP32 thực tế chỉ cần 1 kênh ECG.
- **Đầu ra**: nhãn `0 = Normal`, `1 = Abnormal`.
- **Quy ước nhãn (AAMI)**: `N, L, R, e, j → 0`; còn lại → `1`. Bốn bản ghi paced
  (102, 104, 107, 217) bị loại theo khuyến nghị chuẩn.
- **Chia dữ liệu**: theo bệnh nhân (de Chazal `DS1` train / `DS2` test) — không
  shuffle để tránh rò rỉ giữa beat của cùng một người.

## 2. Dataset

- **MIT-BIH Arrhythmia Database** (PhysioNet): 48 bản ghi, 30 phút mỗi bản,
  360 Hz, có annotation đỉnh R + nhãn beat.
- Tự động tải qua `wfdb.io.dl_database('mitdb', ...)` ở lần chạy đầu tiên,
  lưu vào `dataset/mit-bih-arrhythmia-database-1.0.0/`.
- Sau khi xử lý, ma trận đặc trưng/raw được cache vào `dataset/cache/*.npz` để
  những lần chạy sau không phải lọc + segment lại.

## 3. Bốn mô hình

| Mô hình | Framework | Đầu vào          | Sizes (tiny / small / medium / large) | Định dạng lưu  |
|---------|-----------|------------------|---------------------------------------|----------------|
| **RF**  | sklearn   | features (~30D)  | n_estimators 10 / 20 / 40 / 80; max_depth 4 / 6 / 8 / 10 | `model/rf_<size>.pkl` |
| **SVM** | sklearn   | features (~30D)  | LinearSVC; RBF-2k / RBF-5k / RBF-10k subsample          | `model/svm_<size>.pkl` |
| **CNN** | pytorch   | raw 200×1        | conv [8] / [8,16] / [16,32,32] / [16,32,64,64]          | `model/cnn_<size>.pt`  |
| **LSTM**| pytorch   | seq 100×1 (ds 2) | hidden 8 / 16 / 32 / 32×2 layers                        | `model/lstm_<size>.pt` |

**Đặc trưng tay (RF/SVM)**: time-domain stats (mean/std/min/max/skew/kurtosis/RMS/p2p/energy/zero-crossings),
năng lượng wavelet `db4` 4 mức, dominant frequency, spectral entropy, RR-interval (prev/post/ratio/diff).

Mỗi file model tự kiểm tra checkpoint: **đã train rồi → load + test; chưa có →
train rồi save**. Dùng `--retrain` để ép huấn luyện lại.

## 4. Cài đặt môi trường

```bash
# Tạo venv Python 3.10 (đã có sẵn `env/`)
python3.10 -m venv env
source env/bin/activate
pip install -U pip wheel

# Thư viện CPU-side
pip install -r requirements.txt

# PyTorch CUDA 12.6 (phù hợp GTX 1650, driver 13.0, nvcc 12.0)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Kiểm tra GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 5. Chạy huấn luyện / kiểm thử

Mỗi lệnh sẽ tự xử lý dataset (lần đầu mất ~1-2 phút), sau đó duyệt lần lượt
4 size. Size đã có checkpoint trong `model/` sẽ được load thẳng để test.

```bash
source env/bin/activate

python -m src.rf       # Random Forest
python -m src.svm      # SVM (linear + RBF với subsampling)
python -m src.cnn      # 1D CNN, GPU
python -m src.lstm     # LSTM, GPU

# Ép train lại tất cả size:
python -m src.rf --retrain

# Tinh chỉnh DL:
python -m src.cnn --epochs 25 --batch-size 256 --lr 5e-4
python -m src.lstm --epochs 20 --batch-size 256
```

Kết quả từng size in ra console (accuracy / precision / recall / F1 / ROC-AUC /
confusion matrix) và được dump vào `model/<name>_<size>_metrics.json`.

## 6. Cấu trúc thư mục

```
pr_tinyml/
├── env/                     # virtualenv (gitignored)
├── src/
│   ├── utils.py             # download, preprocess, features, metrics
│   ├── rf.py / svm.py       # sklearn
│   └── cnn.py / lstm.py     # pytorch
├── dataset/                 # tự tạo: MIT-BIH + cache .npz
├── model/                   # checkpoint + metrics json (rỗng đến khi train)
├── requirements.txt
├── README.md
└── CLAUDE.md
```

## 7. Giai đoạn tiếp theo

- Đo thực tế kích thước/độ trễ trên ESP32 → hiệu chỉnh 4 size (chỉnh trong
  `SIZES = [...]` của từng file).
- Convert sang C/C++: TFLite-Micro / ExecuTorch / micromlgen tuỳ mô hình.
