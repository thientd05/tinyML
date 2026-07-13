# pr_tinyml — Phát hiện bất thường ECG trên ESP32

Pipeline TinyML so sánh **6 họ mô hình** (Random Forest, XGBoost, SVM, 1D CNN, LSTM, CNN-LSTM/CRNN)
cho bài toán phân loại nhịp tim nhị phân **Normal vs Abnormal** (MIT-BIH), rồi **convert sang C
thuần và chạy thật trên ESP32-WROOM-32**.

- **Giai đoạn 1** — huấn luyện + so sánh trên PC (GPU GTX 1650): [`src/`](src/)
- **Giai đoạn 2** — convert sang C thuần + benchmark trên thiết bị: [`tools/`](tools/) + [`esp32/`](esp32/)

📄 **Báo cáo đồ án** (chi tiết phương pháp, kết quả, phân tích):
[`docs/BaoCaoDoAn.docx`](docs/BaoCaoDoAn.docx) (bản cô đọng) ·
[`docs/BaoCaoDuAn.docx`](docs/BaoCaoDuAn.docx) (bản đầy đủ).
File README này là tài liệu **đi kèm mã nguồn**: cách cài đặt, phiên bản công cụ, mô tả module.

## Tóm tắt kết quả

Mỗi họ được quét 6 điểm dung lượng (**36 mô hình**), train lại trên **5 seed**, đánh giá tại
**điểm vận hành lâm sàng** (recall ≥ 0.95, ngưỡng dò trên tập validation patient-pure) và lọc qua
**feasibility gate** (latency ≤ 100 ms + flash ≤ 2 MB + RAM ≤ 64 KB, đo thật trên board). Winner =
max precision **trung bình** @ recall 0.95 trong vùng hợp lệ (31/36 cấu hình).

| Họ | Biến thể tốt nhất | prec@0.95 (±std) | PR-AUC | recall triển khai | latency | flash |
|----|----|:----:|:----:|:----:|----:|----:|
| RF   | `rf_n10_d5`         | 0.127±0.01 | 0.74 | 0.78 ❗ | 1.0 ms | 12 KB |
| XGB  | `xgb_x50_d2`        | 0.130±0.00 | 0.72 | 0.74 ❗ | 1.0 ms | 6.8 KB |
| **SVM** 🏆 | **`svm_linear`** | **0.193±0.00** | 0.73 | 0.92 ❗ | **1.0 ms** | **0.2 KB** |
| CNN  | `cnn_c8-16`         | 0.172±0.05 | 0.63 | 0.965 ✓ | 21.1 ms | 53 KB |
| LSTM | `lstm_h8`           | 0.152±0.01 | 0.48 | 0.987 ✓ | 14.2 ms | 1.4 KB |
| CRNN | `crnn_c16-32_h16`   | 0.136±0.01 | 0.58 | 0.945 ❗ | 54.8 ms | 23 KB |

**Winner: `svm_linear`** — đơn giản nhất, rẻ nhất, tất định (std = 0.00) và tái lập được.
Bảng đầy đủ 36 dòng: [`results/summary.csv`](results/summary.csv); biểu đồ: `results/*.png`.

Ba phát hiện chính (phân tích đầy đủ trong báo cáo): (1) tại recall 0.95 trên **bệnh nhân lạ**,
precision của **cả 6 họ** chỉ ~0.12–0.19 và chồng lấn trong dải nhiễu seed — bài toán khó thật,
không như F1@0.5 tô hồng; (2) **mô hình mạnh hơn không thắng** — boosting không hơn bagging, lai
conv+recurrent không hơn conv thuần → nút thắt nằm ở **đặc trưng + cấu trúc nhãn**; (3) precision
của CNN/LSTM/CRNN **dao động ±0.05 chỉ do seed** → phải train 5 seed và xếp hạng theo trung bình.

> 📟 **Benchmark thiết bị: 36/36** mô hình đã chạy thật trên board (hai nhóm build), **tất cả đạt
> parity 1.0000** — log: [`esp32/benchmark_sweep.txt`](esp32/benchmark_sweep.txt), ảnh:
> [`docs/img/`](docs/img/). Phép đo thật đã **loại 5 cấu hình** vì vượt ngân sách 100 ms — trong đó
> có `crnn_c16-32-32_h32`, cấu hình *precision cao nhất* của họ CRNN (0.154 nhưng 140.8 ms) → winner
> CRNN là `c16-32_h16`. Đây là lý do phải **đo** thay vì ước lượng. Winner chung **không đổi**.

## 1. Cách cài đặt

Yêu cầu: **Python 3.10+**, pip, **PlatformIO CLI**, GPU NVIDIA (tùy chọn — cho CNN/LSTM/CRNN).

```bash
git clone https://github.com/thientd05/tinyML.git && cd tinyML
python3.10 -m venv env && source env/bin/activate
pip install -r requirements.txt
# PyTorch cài RIÊNG (wheel CUDA 12.6, forward-compatible với driver CUDA 13.0):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Dataset MIT-BIH (~70 MB) **tự tải** qua `wfdb` ở lần chạy đầu và cache vào
`dataset/cache/mitbih_<mode>_v2.npz` — các lần sau khởi động trong vài giây.

### 1.1. Phiên bản các công cụ dùng để phát triển

Đây là phiên bản **thực tế đã cài** trong môi trường tạo ra mọi số liệu của dự án
(`requirements.txt` chỉ khai báo phiên bản *tối thiểu*).

| Công cụ / thư viện | Phiên bản | Vai trò |
|---|---|---|
| Python | **3.10.19** | ngôn ngữ chính (venv tại `./env`) |
| PyTorch | **2.11.0+cu126** | CNN, LSTM, CRNN (wheel CUDA 12.6) |
| scikit-learn | **1.7.2** | Random Forest, SVM, scaler, calibration |
| XGBoost | **3.2.0** | họ boosting |
| NumPy | **2.1.3** | tính toán mảng; dựng lại forward khi export (cổng parity) |
| SciPy | **1.15.3** | bandpass Butterworth, thống kê |
| PyWavelets | **1.8.0** | năng lượng wavelet `db4` |
| wfdb | **4.3.1** | tải + đọc MIT-BIH (định dạng WFDB) |
| pandas | **2.3.3** | tổng hợp bảng kết quả |
| matplotlib | **3.10.9** | biểu đồ quality-vs-cost, Pareto, PR |
| joblib | **1.5.3** | lưu/nạp checkpoint sklearn (`.pkl`) |
| tqdm | **4.67.3** | thanh tiến trình |
| python-docx | **1.2.0** | sinh `.docx` template (`tools/make_report.py`) |
| pyserial | **3.5** | đọc benchmark qua Serial (`tools/read_serial.py`) |
| Ruff | **0.15.21** | lint (cấu hình trong `pyproject.toml`) |
| PlatformIO Core | **6.1.19** | build + nạp firmware ESP32 |
| platform `espressif32` | **7.0.1** | toolchain + framework Arduino |

**Phần cứng phát triển**: GPU NVIDIA GTX 1650 (4 GB, compute 7.5) · board ESP32-WROOM-32
(ESP32-D0WD-V3, Xtensa LX6 @240 MHz, 320 KB RAM, 4 MB flash, không PSRAM).

## 2. Cách chạy

Mọi lệnh chạy từ **gốc repo** (để `from src ...` resolve được).

```bash
source env/bin/activate

# chạy 1 họ end-to-end (tự bỏ qua điểm sweep đã train; --retrain để ép train lại)
python -m src.models.rf     python -m src.models.cnn    # 1D CNN  (GPU)
python -m src.models.xgb    python -m src.models.lstm   # LSTM    (GPU)
python -m src.models.svm    python -m src.models.crnn   # CRNN    (GPU)

# ĐƯỜNG CHẠY ĐƯỢC BÁO CÁO: train mỗi điểm trên 5 seed, lưu seed-median + ghi mean/std
PYTHONPATH=. python tools/multiseed.py --seeds 5
PYTHONPATH=. python tools/multiseed.py --seeds 5 --families xgb crnn   # chỉ vài họ

# tổng hợp -> results/summary.csv (+ winners.json, cleaning_stats.json) + biểu đồ
python -m src.analysis.analyze --latency esp32/benchmark_sweep.txt --feature-us 925.5 --pr-curves

./env/bin/ruff check .    # lint
```

⚠️ Luôn so sánh qua `tools/multiseed.py` (trung bình 5 seed) — một lần chạy đơn lẻ xê dịch ±0.05 ở
operating point này và sẽ tôn nhầm winner.

### 2.1. Convert và chạy trên ESP32

Trọng số của cả 36 mô hình (~4.2 MB) vượt phân vùng app 3 MB, nên firmware chia **hai nhóm build**
(`group_a` = rf+svm+lstm, `group_b` = xgb+cnn+crnn). Nạp lần lượt cả hai và **nối** cả hai bảng
serial vào `esp32/benchmark_sweep.txt` — `analyze.py` đọc hợp của chúng.

```bash
# 1) Sinh weights + 20 nhịp DS2 nhúng -> esp32/include/*.h (CỔNG PARITY: lệch 1 nhãn là dừng)
PYTHONPATH=. ./env/bin/python tools/export_esp32.py

# 2) Build + flash TỪNG NHÓM (GIỮ NÚT BOOT khi esptool báo "Connecting....")
./env/bin/pio run -d esp32 -e group_a -t upload     # rf + svm + lstm
PYTHONPATH=. ./env/bin/python tools/read_serial.py /dev/ttyUSB0 90 "loop idle"
./env/bin/pio run -d esp32 -e group_b -t upload     # xgb + cnn + crnn
PYTHONPATH=. ./env/bin/python tools/read_serial.py /dev/ttyUSB0 90 "loop idle"

# 3) Nối cả hai bảng vào esp32/benchmark_sweep.txt rồi chạy lại analyze --latency
```

**Lưu ý phần cứng (board này)**: mạch auto-reset vào bootloader không ăn → phải **giữ nút BOOT** lúc
upload. `pio device monitor` cần TTY nên không chạy headless được — dùng
[`tools/read_serial.py`](tools/read_serial.py) thay thế. `upload_speed` để **115200**: ở 460800 (mặc
định của `esp32dev`) cổng USB-serial rớt giữa lúc ghi ảnh 2.37 MB của `group_a` và esptool chết ở 8%.

## 3. Mô tả các module

Hai quy ước giữ cho cây thư mục không rối: **`src/` là thứ được *import*, `tools/` là thứ được
*chạy*** (`src/` không bao giờ import ngược từ `tools/`; mọi `tools/*.py` cần `PYTHONPATH=.`), và
**`docs/` là thứ người viết tay, `results/` là thứ máy sinh** (đừng sửa tay `results/` — lần chạy
`analyze.py` kế tiếp sẽ ghi đè).

```
pr_tinyml/
├── src/                      # THƯ VIỆN (được import)
│   ├── config.py             # hằng số: paths, FS/beat, cleaning, budgets, DS1/DS2 split, AAMI
│   ├── seeding.py            # set_seed
│   ├── artifacts.py          # model_path / save_metrics_json
│   ├── data/                 # download · preprocess · segmentation · features · assembly · benchmark
│   ├── evaluation/           # metrics · operating_point
│   ├── models/               # cost.py + rf/ xgb/ svm/ cnn/ lstm/ crnn/
│   ├── training/             # multiseed.py — ĐƯỜNG CHẠY ĐƯỢC BÁO CÁO (5 seed → mean/std)
│   └── analysis/             # analyze · feasibility · selection · writers · plots · load
├── tools/                    # ĐIỂM CHẠY (vỏ CLI mỏng gọi vào src/)
├── esp32/                    # PlatformIO project (Arduino framework)
├── docs/                     # BÁO CÁO (viết tay) + img/ (ảnh demo chạy thật trên board)
├── results/                  # máy sinh: summary.csv · winners.json · *.png   └── archive/
├── dataset/  model/  env/    # tự tạo khi chạy — gitignored
└── requirements.txt  pyproject.toml  README.md  CLAUDE.md
```

| Module | Vai trò |
|---|---|
| [`src/config.py`](src/config.py) | Hằng số toàn cục: paths, FS/kích thước nhịp, tham số cleaning, **ngân sách phần cứng** (`LATENCY_BUDGET_MS=100`, `FLASH_BUDGET_KB=2048`, `RAM_WORK_BUDGET_KB=64`), `TARGET_RECALL=0.95`, danh sách bản ghi DS1/DS2, ký hiệu AAMI |
| [`src/seeding.py`](src/seeding.py) | `set_seed()` — cố định seed cho numpy / torch / sklearn |
| [`src/artifacts.py`](src/artifacts.py) | `model_path()` / `save_metrics_json()` — quy ước đường dẫn checkpoint + file metrics |
| [`src/data/`](src/data/) | `download` (tải MIT-BIH qua wfdb) · `preprocess` (chọn lead MLII **theo tên** — sửa record 114, bandpass 0.5–40 Hz, z-score) · `segmentation` (cắt nhịp, RR giữa nhịp thật, căn đỉnh R, lọc nhịp hỏng) · `features` (21 đặc trưng tay) · `assembly` (`build_dataset` + cache) · `benchmark` |
| [`src/evaluation/`](src/evaluation/) | `metrics` · `operating_point` — dò ngưỡng đạt recall mục tiêu, tính 3 kịch bản `test_05` / `test_op` / `test_oracle` |
| [`src/models/cost.py`](src/models/cost.py) | Mô hình chi phí giải tích: MACs, #params, flash-bytes cho từng họ |
| [`src/models/{rf,xgb,svm}/`](src/models/) | `sweep` · `estimator` · `__main__` — ba họ ăn **21 đặc trưng tay** |
| [`src/models/{cnn,lstm,crnn}/`](src/models/) | `sweep` · `architecture` · `training` · `inference` · `__main__` — ba họ ăn **nhịp thô** |
| [`src/training/multiseed.py`](src/training/multiseed.py) | `run_multiseed()` — **đường chạy được báo cáo**: 5 seed, lưu checkpoint seed-median, ghi mean/std |
| [`src/analysis/`](src/analysis/) | `analyze` (tổng hợp → winner) · `feasibility` (gate phần cứng) · `selection` (chọn winner) · `writers` (summary.csv / winners.json) · `plots` · `load` |
| [`tools/multiseed.py`](tools/multiseed.py) | Vỏ CLI của `src/training/multiseed.py` |
| [`tools/export_esp32.py`](tools/export_esp32.py) | Convert checkpoint → C header; **cổng parity** chặn export nếu lệch dù một nhãn |
| [`tools/read_serial.py`](tools/read_serial.py) | Đọc bảng benchmark từ ESP32 qua Serial (không cần TTY) |
| [`tools/make_beat_figures.py`](tools/make_beat_figures.py) | Sinh hình minh họa hình thái nhịp cho báo cáo |
| [`tools/make_report.py`](tools/make_report.py) | Sinh `results/BaoCao_TinyML_ECG.docx` — **chỉ là template máy sinh**, không phải báo cáo |
| [`esp32/src/main.cpp`](esp32/src/main.cpp) | Firmware benchmark + 6 kernel C generic (`rf/xgb/svm/cnn/lstm/crnn_infer`) |
| [`esp32/include/`](esp32/include/) | Header sinh tự động: `model_*.h`, `test_data.h`, `kernels.h`, `feature_bench.h` (đo db4+FFT) |
| [`esp32/platformio.ini`](esp32/platformio.ini) | Board `esp32dev` + `huge_app`; hai nhóm build `group_a` / `group_b` |

### 3.1. Dữ liệu (`src/data/`)

Đầu vào là **1 nhịp** (±100 mẫu quanh đỉnh R, ~555 ms @360 Hz) trên chuyển đạo **MLII**; đầu ra
`0 = Normal` / `1 = Abnormal` theo AAMI (`N,L,R,e,j → 0`, còn lại → 1). Loại 4 bản ghi paced
(102/104/107/217); mất cân bằng ~11% Abnormal. **Chia theo bệnh nhân** (de Chazal `DS1` train /
`DS2` test; `DS1-val` = records 207/209/215 patient-pure để dò ngưỡng) — không shuffle chung, tránh
rò rỉ beat cùng người. Dữ liệu được **làm sạch label-free** trước khi train (chọn lead theo tên, RR
giữa nhịp thật, căn đỉnh R, lọc nhịp vật lý hỏng) — thống kê:
[`results/cleaning_stats.json`](results/cleaning_stats.json).

`build_dataset(mode)` trả 6 mảng `(X_train, y_train, X_val, y_val, X_test, y_test)`:
`mode="features"` → 21 đặc trưng tay (RF/XGB/SVM) · `mode="raw"` → nhịp thô 200×1 (CNN/CRNN) ·
`mode="lstm"` → 100×1 (downsample ×2).

### 3.2. Roster mô hình (`src/models/`)

| Package | Backend | Đầu vào | Capacity knob → SWEEP (`sweep.py`) |
|---|---|---|---|
| `rf/`   | sklearn | 21 features | #node cây: `n10_d3 … n80_d12` |
| `xgb/`  | xgboost | 21 features | #node cây: `x50_d2 … x400_d6` (lr cố định 0.1) |
| `svm/`  | sklearn | 21 features | #support vector: `linear`, `rbf1k … rbf10k` |
| `cnn/`  | pytorch | raw 200×1   | MACs: `c4`, `c8`, `c8-16`, `c16-16`, `c16-32-32`, `c16-32-64-64` |
| `lstm/` | pytorch | seq 100×1   | MACs: `h4 … h32`, `h32x2` (2 lớp) |
| `crnn/` | pytorch | raw 200×1   | MACs: `c8_h8 … c16-32-32_h32` |

Mất cân bằng lớp xử lý cùng một cách ở mọi họ: `class_weight="balanced"` (RF/SVM),
`scale_pos_weight=neg/pos` (XGB), weighted CE (nets). Mỗi điểm sweep dump `model/<name>_<size>_metrics.json`
gồm **cost** + ba operating point (`test_05` / `test_op` / `test_oracle`) + block `seeds` (mean/std).

## 4. Việc tương lai

- **Debounce** ở tầng ứng dụng + **cá nhân hóa ngưỡng** — nâng precision mà không đổi mô hình.
- **Lượng tử hóa int8** (hiện fp32) + **ESP-DSP / CMSIS-NN** để giảm flash và độ trễ.
- **Tích hợp** trích đặc trưng db4+FFT vào đường chạy RF/XGB/SVM on-device (hiện mới *đo* chi phí).
- Vì trần nằm ở **đặc trưng**, hướng triển vọng nhất là **đặc trưng tốt hơn / nhiều nhịp ngữ cảnh
  hơn**, chứ không phải mô hình lớn hơn.
