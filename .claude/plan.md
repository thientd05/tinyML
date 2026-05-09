# Plan: ECG Anomaly Detection — RF / SVM / CNN / LSTM cho ESP32

## Context

Dự án greenfield (`/home/thienta/pr_tinyml/`) xây pipeline so sánh 4 mô hình ML/DL
phục vụ "nhận diện bất thường trong tín hiệu điện tâm đồ" trên thiết bị biên ESP32.
Dataset: **MIT-BIH Arrhythmia Database** (48 bản ghi, 360 Hz, 2 kênh, đã có annotation
beat). Bài toán: **phân loại nhị phân Normal vs Abnormal** (theo grouping AAMI:
N/L/R/e/j → Normal; tất cả còn lại → Abnormal).

Giai đoạn này chỉ tạo source code + cấu trúc thư mục + tài liệu. Chưa train, chưa
convert sang C++. Mỗi mô hình có **4 size variants** đều ước lượng vừa ESP32 (sẽ
hiệu chỉnh sau khi đo thật trên phần cứng). Trong quá trình tải PyTorch CUDA, cần
chọn wheel phù hợp với GTX 1650 (compute 7.5) + driver 13.0 + nvcc 12.0.

**Đã quyết định với user:**
- Phân loại: nhị phân (Normal=0 / Abnormal=1).
- Số sizes: 4 (tiny / small / medium / large), tất cả dự kiến vừa ESP32.
- Cấu trúc: `src/utils.py` (shared) + 4 file model độc lập.

## Môi trường & cài đặt

| Hạng mục         | Lựa chọn                                                  |
|------------------|-----------------------------------------------------------|
| Python           | **3.10.19** (`/usr/bin/python3.10`) — ổn định, tương thích PyTorch 2.7, wfdb, pywt |
| Venv             | `python3.10 -m venv env` (tại root project)               |
| PyTorch wheel    | **cu126** (PyTorch 2.7.0). GTX 1650 = compute 7.5, driver CUDA 13.0 forward-compatible với toolkit 12.6. Lệnh: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` |
| sklearn ecosystem| numpy, scipy, scikit-learn, pandas, joblib                |
| ECG-specific     | wfdb (download MIT-BIH + đọc .dat/.atr), pywavelets       |
| Khác             | matplotlib, tqdm                                          |

## Cây thư mục cuối

```
/home/thienta/pr_tinyml/
├── env/                          # virtualenv (gitignored)
├── src/
│   ├── utils.py                  # download, preprocess, features, metrics, seed
│   ├── rf.py                     # Random Forest + 4 sizes (sklearn)
│   ├── svm.py                    # SVM + 4 sizes (sklearn)
│   ├── cnn.py                    # 1D CNN + 4 sizes (pytorch)
│   └── lstm.py                   # LSTM + 4 sizes (pytorch)
├── dataset/                      # auto: wfdb.dl_database → mit-bih-arrhythmia-database-1.0.0/
├── model/                        # rỗng đến khi train; lưu .pkl (sklearn) / .pt (pytorch)
├── requirements.txt
├── README.md
├── CLAUDE.md                     # < 150 dòng
└── .gitignore                    # bổ sung env/, dataset/, model/, __pycache__
```

## Thiết kế `src/utils.py` (shared)

Hàm chính:
- `set_seed(seed=42)` — random/numpy/torch/cudnn deterministic.
- `download_mitbih(dataset_dir)` — `wfdb.io.dl_database('mitdb', dataset_dir)`, idempotent.
- `DS1 / DS2 split` (chuẩn de Chazal): training/test theo bệnh nhân, loại 102/104/107/217 (paced).
- `bandpass_filter(sig, fs=360, low=0.5, high=40)` — Butterworth bậc 4, filtfilt.
- `extract_beats(record_id, channel=0)` — đọc bằng wfdb, lọc, z-score per-record, cắt cửa sổ **±100 mẫu quanh R-peak (200 samples ≈ 555 ms)**, giữ luôn `rr_prev`, `rr_post`.
- `aami_binary_label(symbol)` — N/L/R/e/j → 0, còn lại → 1.
- `extract_features(beat, rr_prev, rr_post)` — ~30 đặc trưng cho RF/SVM:
  - Time-domain: mean, std, max, min, skew, kurtosis, rms, peak-to-peak, energy, zero-crossings.
  - Wavelet (pywt, db4, level 4): năng lượng từng level (5 giá trị).
  - Frequency: dominant freq, spectral entropy (rfft).
  - RR: `rr_prev`, `rr_post`, `rr_ratio`, `rr_diff`.
- `build_dataset(mode='features'|'raw', cache=True)` — gọi tất cả bản ghi DS1/DS2, trả về `(X_train, y_train, X_test, y_test)`. Cache numpy `.npz` trong `dataset/cache/` để tránh xử lý lại.
- `compute_metrics(y_true, y_pred, y_score=None)` — accuracy, precision, recall, F1, confusion-matrix, ROC-AUC; trả dict.
- `class_weights(y)` — balanced weights cho sklearn / pytorch loss.

## 4 size variants — ước lượng ban đầu (đều dự kiến vừa ESP32 fp32)

### `src/rf.py` — RandomForestClassifier, lưu joblib `.pkl`

| Size   | n_estimators | max_depth | param/size ước lượng |
|--------|--------------|-----------|----------------------|
| tiny   | 10           | 4         | ~5 KB                |
| small  | 20           | 6         | ~25 KB               |
| medium | 40           | 8         | ~80 KB               |
| large  | 80           | 10        | ~250 KB              |

Input: features (~30D). `class_weight='balanced'`.

### `src/svm.py` — sklearn SVM, lưu joblib `.pkl`

| Size   | Kernel       | Cấu hình                                    |
|--------|--------------|---------------------------------------------|
| tiny   | linear       | `LinearSVC(C=1.0)` — chỉ vector w           |
| small  | RBF          | `SVC(rbf, C=1.0)` train trên 2000 sample    |
| medium | RBF          | `SVC(rbf, C=1.0)` train trên 5000 sample    |
| large  | RBF          | `SVC(rbf, C=1.0)` train trên 10000 sample   |

Input: features. Subsample stratified để kiểm soát số support vectors. `class_weight='balanced'`.

### `src/cnn.py` — 1D CNN PyTorch, lưu `.pt` (state_dict)

Block chung: `Conv1d → BatchNorm1d → ReLU → MaxPool1d(2)`.

| Size   | Conv channels       | Params ước lượng |
|--------|---------------------|------------------|
| tiny   | [8] + GAP + FC      | ~0.5 K           |
| small  | [8, 16] + FC        | ~3 K             |
| medium | [16, 32, 32] + FC   | ~15 K            |
| large  | [16, 32, 64, 64]+FC | ~50 K            |

Input: raw beat 200×1, output: logit 2-class. Train: AdamW, CrossEntropyLoss với class weights, early stopping theo val F1, mixed-precision (fp16) trên GTX 1650.

### `src/lstm.py` — LSTM PyTorch, lưu `.pt`

| Size   | Hidden | Layers | Bidir | Params ước lượng |
|--------|--------|--------|-------|------------------|
| tiny   | 8      | 1      | no    | ~0.3 K           |
| small  | 16     | 1      | no    | ~1.2 K           |
| medium | 32     | 1      | no    | ~4.5 K           |
| large  | 32     | 2      | no    | ~9 K             |

Input: chuỗi 100×1 (downsample 200→100 bằng decimate / avg-pool 2). Train tương tự CNN.

## Khuôn chung mỗi file model (`main()`)

```
1. set_seed → load (hoặc build) dataset từ utils.
2. for size in ['tiny','small','medium','large']:
     path = model/<name>_<size>.<ext>
     if path tồn tại:  load → evaluate trên test → in metrics
     else:             train (CV/holdout) → save → evaluate → in metrics
3. In bảng tổng hợp.
```

CLI: `python -m src.rf` (hoặc `python src/rf.py`). Hỗ trợ `--retrain` (tùy chọn).

## `requirements.txt` (PyTorch cài tay riêng)

```
numpy>=1.24,<2.2
scipy>=1.10
scikit-learn>=1.3
pandas>=2.0
joblib>=1.3
wfdb>=4.1
pywavelets>=1.5
matplotlib>=3.7
tqdm>=4.65
```

## `README.md`

Giới thiệu bài toán, dataset (link PhysioNet), pipeline, mô tả 4 mô hình + ý tưởng size,
hướng dẫn setup (venv → cài requirements → cài PyTorch CUDA), cách chạy từng file.

## `CLAUDE.md` (<150 dòng)

Sections: project overview, environment (Python 3.10 / PyTorch 2.7 cu126 / GTX 1650),
key commands (activate env, install, train từng model), data pipeline tóm tắt, model
roster (bảng size), gotchas (DS1/DS2 split, class imbalance, paced records bị loại,
đường dẫn cache dataset, model file naming convention), how to add new dataset trong
tương lai.

## Critical files cần tạo

- `/home/thienta/pr_tinyml/requirements.txt`
- `/home/thienta/pr_tinyml/src/utils.py`
- `/home/thienta/pr_tinyml/src/rf.py`
- `/home/thienta/pr_tinyml/src/svm.py`
- `/home/thienta/pr_tinyml/src/cnn.py`
- `/home/thienta/pr_tinyml/src/lstm.py`
- `/home/thienta/pr_tinyml/README.md`
- `/home/thienta/pr_tinyml/CLAUDE.md` (đang rỗng — sẽ ghi đè)
- `/home/thienta/pr_tinyml/.gitignore` (mở rộng)
- Thư mục: `model/`, `dataset/`, `src/`

## Thứ tự thực thi (sau khi exit plan)

1. `python3.10 -m venv env` → activate → `pip install -U pip wheel`.
2. Viết `requirements.txt`, `pip install -r requirements.txt`.
3. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` rồi `python -c "import torch; print(torch.cuda.is_available())"` để verify.
4. Tạo `src/`, `model/`, `dataset/`.
5. Viết `src/utils.py` (kèm `__init__.py` nếu cần import gói).
6. Viết `src/rf.py`, `src/svm.py`, `src/cnn.py`, `src/lstm.py`.
7. Viết `README.md`, `CLAUDE.md`, mở rộng `.gitignore`.
8. Verify imports: `python -c "from src import utils, rf, svm, cnn, lstm; print('ok')"` (chỉ import, không train).

## Verification

- `torch.cuda.is_available() == True` và `torch.cuda.get_device_name(0) == "NVIDIA GeForce GTX 1650"`.
- `python -c "import wfdb, pywt, sklearn, scipy, numpy; print('all ok')"`.
- Smoke run import từng module trong `src/`.
- Mỗi file model chạy được dạng `python src/<name>.py --help` (hoặc smoke `--dry-run` nếu hữu ích) — chưa train thật, chỉ kiểm tra logic load dataset cache (lần đầu sẽ tải MIT-BIH ~70MB về `dataset/`).
- User sẽ tự chạy huấn luyện theo hướng dẫn trong README (không phải nhiệm vụ giai đoạn này).

## Lưu ý quan trọng (giai đoạn sau)

- Sau khi đo trên ESP32, có thể cần co lại / nới ra 4 sizes — chỗ chỉnh nằm trong các hàm `get_configs()` của mỗi file model.
- Chưa làm: chuyển sang C/C++ (giai đoạn tiếp theo, dùng micromlgen / TFLite-Micro / ExecuTorch).
- Cache `.npz` của dataset không commit (đã thêm vào `.gitignore`).

Sources:
- [PyTorch Get Started](https://pytorch.org/get-started/locally/)
- [PyTorch torch · PyPI](https://pypi.org/project/torch/)
