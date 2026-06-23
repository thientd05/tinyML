# pr_tinyml — Phát hiện bất thường ECG trên ESP32

Pipeline so sánh **4 họ mô hình** (Random Forest, SVM, 1D CNN, LSTM) cho bài toán
phân loại nhịp tim nhị phân **Normal vs Abnormal** (MIT-BIH), rồi **convert và chạy
thật trên ESP32-WROOM-32**.

- **Giai đoạn 1** — huấn luyện + so sánh trên PC (GPU GTX 1650): `src/`.
- **Giai đoạn 2** — convert sang C thuần và benchmark trên thiết bị: `tools/` + `esp32/`.

> **TL;DR**: dữ liệu được **làm sạch không rò rỉ nhãn** (sửa lead record 114, RR giữa các
> nhịp thật, căn đỉnh R, lọc nhịp nhiễu) trước khi train. Mỗi họ **quét capacity knob**
> (~6 điểm), **train lại trên 5 seed** (báo cáo trung bình ± std, xếp hạng theo trung bình),
> đánh giá tại **điểm vận hành lâm sàng** (recall ≥ 0.95) và lọc qua **feasibility gate**
> (latency ≤ 100ms + flash + RAM đo thật). Cả **24 mô hình** parity **1.0000**.
> **Winner: SVM `linear`** — precision **0.193±0.00 @ recall 0.95**, latency **1.0ms**,
> flash **0.2KB** — đơn giản nhất, rẻ nhất, và *tái lập được*. Xem phương pháp + bảng dưới.

> ⚠️ **Phát hiện quan trọng**:
> 1. Tại recall 0.95 trên **bệnh nhân lạ** (DS2), precision trung bình các họ chỉ
>    **~0.13–0.19** và **chồng lấn trong dải nhiễu seed (~±0.05)** — không họ nào áp đảo;
>    bài toán khó thật sự, KHÔNG như F1@0.5 (rf "0.71") tô hồng.
> 2. precision@recall0.95 của CNN/LSTM **dao động ±0.05 chỉ do seed khởi tạo** → bản cũ báo
>    "CNN c8-16 = 0.242" là một lần **bốc seed may mắn, KHÔNG tái lập** (mean ~0.17). Vì vậy
>    nay train **5 seed** và xếp hạng theo **trung bình**.
> 3. ngưỡng dò trên DS1 chỉ giữ recall ~0.78–0.92 trên DS2 với RF/SVM (khe hở hiệu chỉnh
>    giữa bệnh nhân), trong khi CNN/LSTM chuyển ngưỡng tốt (recall ~0.96–0.99).

---

## 🔬 Phương pháp lựa chọn (thay cho 4 size cảm tính)

Phát biểu việc triển khai như **tối ưu có ràng buộc**, mã hóa đúng 3 tiêu chí bài toán:

1. **Quét capacity knob** — mỗi họ có 1 tham số dung lượng đơn điệu chi phối chi phí
   ESP32 (RF: tổng số node cây; SVM: #support vector; CNN/LSTM: MACs/nhịp). Quét ~6 điểm
   trên lưới log để **vẽ đường chất lượng–chi phí** và tìm điểm bão hòa (knee), thay vì
   đoán 4 size. Cấu hình trong `SWEEP=[...]` ở mỗi `src/models/<name>/sweep.py`.
2. **Điểm vận hành lâm sàng** — "bỏ sót bất thường là nguy hiểm nhất" → đặt **sàn độ nhạy
   recall ≥ 0.95** và **tinh chỉnh ngưỡng** (thay ngưỡng 0.5 mặc định). Ngưỡng dò trên
   tập **DS1-val patient-pure** (records 207/209/215), báo cáo trên DS2.
3. **Feasibility gate** — biến thể hợp lệ phải: `latency ≤ 100ms/nhịp` (đo thật ESP32, đã
   cộng chi phí trích đặc trưng cho RF/SVM) ∧ `flash ≤ 2MB` ∧ `RAM ≤ 64KB`. Thực tế **chỉ
   latency bite**: nó loại đúng `cnn_c16-32-64-64` (238ms) và `lstm_h32x2` (262ms). Triển khai
   thật chỉ nạp 1 model nên flash (max 1005KB) và RAM (~5.3KB) là lan can không bao giờ chạm —
   tốc độ tính toán mới là ràng buộc thật.
4. **Train lại trên nhiều seed** — precision @ recall 0.95 của mạng raw-beat dao động ~±0.05
   chỉ do seed khởi tạo, nên mỗi điểm được train **5 seed** (`tools/multiseed.py`); báo cáo
   **trung bình ± std** và lưu checkpoint seed-median làm đại diện (export/parity dùng nó).
5. **Chọn**: trong vùng hợp lệ, **max precision *trung bình* @ recall 0.95** (tối thiểu báo
   động giả, và bền vững với nhiễu huấn luyện).

**Hai trục đo tách bạch**: *năng lực* (precision tại recall 0.95 đo trên DS2) vs *chuyển
ngưỡng* (recall thực tế khi dùng ngưỡng dò trên DS1). Quality lấy từ **toàn DS2 trên PC**
(tin được vì parity = 1.0000); thiết bị chỉ đo **latency/RAM**.

## 🏆 Kết quả (winner mỗi họ, tại recall 0.95 trên DS2)

**Phần cứng**: ESP32-D0WD-V3 (Xtensa LX6 @240MHz, 320KB RAM, 4MB flash, không PSRAM).
`prec@R` = precision tại recall 0.95 (năng lực); `lat` = độ trễ end-to-end đo thật
(RF/SVM đã + 0.94ms trích đặc trưng db4+FFT); `rec_dep` = recall khi dùng ngưỡng DS1.

| Họ | Biến thể tốt nhất | prec@0.95 (±std) | fpr | PR-AUC | rec_dep | latency | flash | hợp lệ |
|----|------|:----:|:----:|:----:|:----:|----:|----:|:----:|
| RF   | `rf_n10_d5`  | 0.127±0.01 | 0.83 | 0.74 | 0.78 ❗ | 1.0 ms | 12 KB | ✓ |
| **SVM**  | **`svm_linear`** | **0.193±0.00** | 0.49 | 0.73 | 0.92 ❗ | **1.0 ms** | **0.2 KB** | ✓ |
| CNN  | `cnn_c8-16` | 0.172±0.05 | 0.63 | 0.63 | **0.965 ✓** | 19.8 ms | 53 KB | ✓ |
| LSTM | `lstm_h8`    | 0.152±0.01 | 0.66 | 0.48 | 0.987 ✓ | 31.1 ms | 1.4 KB | ✓ |

> 🏆 **Cross-family winner: `svm_linear`** — precision 0.193±0.00 @ recall 0.95 (cao nhất
> theo *trung bình 5 seed*, lại tất định), fpr 0.49, latency 1.0ms, flash 0.2KB. CNN `c8-16`
> (0.172±0.05) nằm trong dải nhiễu seed của nó nhưng mean thấp hơn và đắt hơn ~20×. Lưu ý:
> winner thắng theo *năng lực* (precision @ recall 0.95) nhưng có **khe hở hiệu chỉnh ngưỡng**
> (rec_dep 0.92 ❗); CNN/LSTM chuyển ngưỡng tốt hơn. Cả 24 mô hình parity **1.0000** trên
> ESP32. Bảng đầy đủ + biểu đồ: `results/summary.csv`,
> `results/{within_family_quality_vs_cost,cross_family_pareto,pr_curves}.png`;
> trước/sau làm sạch: `results/summary_v1_before_clean.csv`. Log thiết bị:
> [`esp32/benchmark_sweep.txt`](esp32/benchmark_sweep.txt).

### Nhận xét (điều cách F1@0.5 cũ che mất)

- **To hơn KHÔNG tốt hơn.** Knee nằm ở model nhỏ; feasibility gate loại đúng các cấu hình
  phình to: `cnn_c16-32-64-64` (239ms), `lstm_h32x2` (262ms) vượt latency.
- **Single-run không đáng tin ở operating point này.** precision@recall0.95 của CNN/LSTM
  dao động **~±0.05 chỉ do seed** → train 5 seed, xếp hạng theo trung bình. "CNN c8-16 =
  0.242" của bản cũ không tái lập (mean ~0.17).
- **Mô hình đơn giản nhất thắng theo trung bình:** `svm_linear` (0.193±0.00, tất định, 0.2KB)
  ≥ các họ phức tạp hơn — vốn chồng lấn trong dải nhiễu seed.
- **Khe hở hiệu chỉnh ngưỡng**: `rec_dep` ❗ ≈0.78–0.92 cho RF/SVM vs ✓ ≈0.96–0.99 cho
  CNN/LSTM. Winner thắng theo *năng lực*; chuyển ngưỡng là điểm yếu của họ tuyến tính/cây.
- **Precision tuyệt đối thấp (mean 0.13–0.19)** ở recall 0.95 trên bệnh nhân lạ — bài toán
  khó thật, không tô hồng.
- **Nút thắt là tính toán, không phải bộ nhớ**: working RAM ~5.5KB/320KB.

---

## 1. Bài toán & dữ liệu

- **Đầu vào**: 1 nhịp (±100 mẫu quanh đỉnh R, ~555 ms @360 Hz) chuyển đạo **MLII** của
  MIT-BIH (chọn theo *tên lead* để mọi bản ghi cùng MLII — sửa record 114 vốn để V5 ở kênh 0).
- **Đầu ra**: `0 = Normal`, `1 = Abnormal`. Nhãn AAMI: `N,L,R,e,j → 0`, còn lại → `1`.
  Loại 4 bản ghi paced (102/104/107/217). Mất cân bằng ~11% Abnormal.
- **Chia theo bệnh nhân** (de Chazal `DS1` train / `DS2` test) — không shuffle chung
  để tránh rò rỉ beat cùng người.
- **Làm sạch (label-free, không rò rỉ)** trước khi train, áp y hệt mọi tập: (1) chọn lead
  MLII theo tên; (2) RR đo giữa các **nhịp thật** liên tiếp (sửa ~4.2% nhịp bị nhiễm bởi
  annotation không-phải-nhịp); (3) căn cửa sổ vào **đỉnh R thật** (±15 mẫu); (4) lọc nhịp
  **vật lý hỏng** (flatline/clipping) theo tiêu chí tín hiệu — KHÔNG theo nhãn, KHÔNG theo
  biên độ. Thực tế chỉ loại vài nhịp (DS2 mất 0 → không cherry-pick test). Chi tiết + bảng
  trước/sau: `tools/inspect_dataset.py`, `results/cleaning_stats.json`, mục 2 của báo cáo.
- **Dataset**: MIT-BIH Arrhythmia DB (PhysioNet), tự tải qua `wfdb` lần chạy đầu, cache
  vào `dataset/cache/mitbih_<mode>_v2.npz` (key `v2` = phiên bản làm sạch).

## 2. Bốn họ mô hình (mỗi họ ~6 điểm quét theo capacity knob)

| Mô hình | Framework | Đầu vào | Capacity knob → SWEEP (`src/models/<name>/`) |
|---------|-----------|---------|------------------------------------------|
| **RF**  | sklearn   | 21 features | #node cây: `n10_d3 … n80_d12` (n_estimators×max_depth) |
| **SVM** | sklearn   | 21 features | #support vector: `linear`, `rbf1k … rbf10k` |
| **CNN** | pytorch   | raw 200×1 | MACs: `c4`, `c8`, `c8-16`, `c16-16`, `c16-32-32`, `c16-32-64-64` |
| **LSTM**| pytorch   | seq 100×1 | MACs: `h4 … h32`, `h32x2` (2 lớp) |

Knob, #params, flash-bytes tính trong [`src/models/cost.py`](src/models/cost.py). Mỗi điểm dump
metrics @0.5 / @ngưỡng-DS1 / @recall-0.95 + cost ra `model/<name>_<size>_metrics.json`.

**Đặc trưng tay (RF/SVM, 21 chiều)**: thống kê time-domain (mean/std/min/max/skew/
kurtosis/RMS/p2p/energy/zero-crossings), năng lượng wavelet `db4` 4 mức, dominant
frequency + spectral entropy, RR-interval (prev/post/ratio/diff). Chi phí trích đặc
trưng được **đo thật trên ESP32** (db4+FFT, ~0.94ms/nhịp) và cộng vào latency RF/SVM.

## 3. Cài đặt & huấn luyện (PC)

```bash
python3.10 -m venv env && source env/bin/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

PYTHONPATH=. python tools/inspect_dataset.py   # (tùy chọn) khảo sát lead/RR/nhiễu trước khi làm sạch

python -m src.models.rf   # Random Forest      python -m src.models.cnn   # 1D CNN (GPU)
python -m src.models.svm  # SVM (linear + RBF)  python -m src.models.lstm  # LSTM (GPU)
#  --retrain để ép train lại tất cả điểm sweep (single-seed, nhanh)

# So sánh bền vững: train mỗi điểm trên 5 seed, lưu seed-median + ghi mean/std
PYTHONPATH=. python tools/multiseed.py --seeds 5

python -m src.analysis.analyze     # tổng hợp quality + cost giải tích, chọn winner, vẽ
python -m src.analysis.analyze \
  --latency esp32/benchmark_sweep.txt --feature-us 937.4 --pr-curves   # + latency thật

python tools/make_report.py        # → results/BaoCao_TinyML_ECG.docx (báo cáo đầy đủ)
```

Mỗi họ model tự load checkpoint nếu có, không thì train rồi lưu vào `model/`. Chạy
`tools/multiseed.py` để có số liệu **mean ± std trên 5 seed** (winner xếp theo trung bình).
`src/analysis/analyze.py` gom tất cả → `results/summary.csv` (+ `winners.json`, `cleaning_stats.json`)
+ 3 biểu đồ (có error band) + in winner. Không có `--latency` thì trục độ trễ để trống.

## 4. Convert & chạy trên ESP32

Pipeline gồm **một script export** (sinh C từ checkpoint, có cổng verify parity) và
**một firmware benchmark** (PlatformIO/Arduino) chạy cả 24 mô hình bằng kernel C tự
viết — không dùng TFLite-Micro nên khớp 100% với PC và không phụ thuộc toolchain convert.
Vì parity = 1.0000, thiết bị chỉ cần đo **latency + RAM** (N nhúng nhỏ = 20 nhịp); chất
lượng lấy từ toàn DS2 trên PC.

```bash
# 1) Sinh weights + 20 nhịp DS2 nhúng + bảng model -> esp32/include/*.h
#    (assert: numpy reproduction == sklearn/torch trên từng mẫu, sai là dừng)
PYTHONPATH=. ./env/bin/python tools/export_esp32.py

# 2) Build + flash (giữ nút BOOT khi esptool báo "Connecting....")
./env/bin/pio run -d esp32 -t upload

# 3) Đọc kết quả Serial (headless; tự reset board + dừng ở marker kết thúc)
PYTHONPATH=. ./env/bin/python tools/read_serial.py /dev/ttyUSB0 60 "loop idle"
```

**Lưu ý phần cứng (board này)**: mạch auto-reset vào bootloader không ăn → phải **giữ
nút BOOT** lúc upload. `pio device monitor` cần TTY nên không chạy headless được — dùng
`tools/read_serial.py` thay thế.

### Cách hoạt động

- `tools/export_esp32.py` — duyệt từng checkpoint, **dựng lại forward bằng NumPy** và
  assert khớp `sklearn/torch.predict`, rồi xuất bảng config + trọng số ra
  `esp32/include/model_*.h`. RF→if/else; SVM→scaler + (linear+calibration | RBF SV);
  CNN→conv (BN gập sẵn) + GAP/FC head; LSTM→4 cổng i,f,g,o, hỗ trợ nhiều lớp.
- `esp32/src/main.cpp` — kernel C generic cho 4 họ + đo độ trễ/RAM, in bảng. Danh sách
  model dựng từ **bảng sinh tự động** (`MODEL_NAMES/FAM/SUB` trong `test_data.h`) nên tự
  co giãn theo sweep. `esp32/include/feature_bench.h` đo chi phí db4+FFT cho RF/SVM.
- Mở rộng/đổi điểm chỉ cần sửa `SWEEP = [...]` trong `src/models/<name>/sweep.py`, train,
  rồi export lại — kernel C + bảng model đã generic.

## 5. Cấu trúc thư mục

```
pr_tinyml/
├── src/                    # thư viện PC, tách theo mối quan tâm
│   ├── config.py           # hằng số: paths, FS/beat, cleaning, budgets, DS1/DS2 split, AAMI
│   ├── seeding.py  io.py   # set_seed | model_path/save_metrics_json
│   ├── data/               # download · preprocess · segmentation · features · assembly · benchmark
│   ├── evaluation/         # metrics · operating_point
│   ├── models/             # cost.py + rf/ svm/ cnn/ lstm/ (sweep·architecture·training·inference·__main__)
│   └── analysis/analyze.py # gom + feasibility gate + chọn winner + biểu đồ
├── tools/
│   ├── export_esp32.py     # checkpoint -> esp32/include/*.h (+ parity gate + bảng model)
│   └── read_serial.py      # đọc Serial ESP32 không cần TTY
├── esp32/                  # PlatformIO project (Arduino framework)
│   ├── platformio.ini      # board=esp32dev, partition huge_app
│   ├── src/main.cpp        # harness + kernel C generic (dựng model từ bảng sinh tự động)
│   ├── include/            # *.h sinh tự động (kernels.h layout; feature_bench.h db4+FFT)
│   └── benchmark_sweep.txt # log latency on-device (24 model + feat_extract)
├── results/                # summary.csv + 3 biểu đồ (analyze sinh ra)
├── dataset/ model/ env/    # tự tạo / gitignored
├── requirements.txt  README.md  CLAUDE.md
```

## 6. Việc tương lai

- Tối ưu kernel CNN/LSTM bằng **ESP-DSP / CMSIS-NN** để giảm độ trễ (mở rộng vùng hợp lệ).
- Lượng tử hóa int8 (hiện đang fp32) để giảm flash + tăng tốc.
- **Tích hợp** trích đặc trưng db4+FFT vào đường chạy RF/SVM on-device (hiện mới đo chi
  phí qua `feature_bench.h`) để chạy end-to-end từ tín hiệu thô.
- Thu hẹp khe hở hiệu chỉnh ngưỡng giữa bệnh nhân (calibration transfer) cho RF/SVM.
