# pr_tinyml — Phát hiện bất thường ECG trên ESP32

Pipeline so sánh **6 họ mô hình** (Random Forest, XGBoost, SVM, 1D CNN, LSTM, CNN-LSTM/CRNN)
cho bài toán phân loại nhịp tim nhị phân **Normal vs Abnormal** (MIT-BIH), rồi **convert và
chạy thật trên ESP32-WROOM-32** bằng kernel C tự viết.

- **Giai đoạn 1** — huấn luyện + so sánh trên PC (GPU GTX 1650): [`src/`](src/).
- **Giai đoạn 2** — convert sang C thuần và benchmark trên thiết bị: [`tools/`](tools/) + [`esp32/`](esp32/).

> **TL;DR**: dữ liệu được **làm sạch không rò rỉ nhãn** (sửa lead record 114, RR giữa các
> nhịp thật, căn đỉnh R, lọc nhịp nhiễu) trước khi train. Mỗi họ **quét capacity knob**
> (6 điểm → **36 mô hình**), **train lại trên 5 seed** (báo cáo trung bình ± std, xếp hạng
> theo trung bình), đánh giá tại **điểm vận hành lâm sàng** (recall ≥ 0.95) và lọc qua
> **feasibility gate** (latency ≤ 100ms + flash ≤ 2MB + RAM ≤ 64KB, đo thật trên board).
> **Winner: SVM `linear`** — precision **0.193±0.00 @ recall 0.95**, latency **1.0ms**,
> flash **0.2KB** — đơn giản nhất, rẻ nhất, và *tái lập được*.

> ⚠️ **Ba phát hiện chính**:
> 1. Tại recall 0.95 trên **bệnh nhân lạ** (DS2), precision trung bình của **cả 6 họ** chỉ
>    **~0.12–0.19** và **chồng lấn trong dải nhiễu seed (~±0.05)** — không họ nào áp đảo.
>    Bài toán khó thật sự, KHÔNG như F1@0.5 (rf "0.71") tô hồng.
> 2. **Mô hình mạnh hơn không thắng.** *Boosting không hơn bagging* (XGBoost chạm đúng trần
>    của RF: prec@R 0.110–0.130 vs RF 0.116–0.127) và *lai conv+recurrent không hơn conv
>    thuần* (CRNN tốt nhất 0.154±0.03 < CNN `c8-16` 0.172±0.05, dù tốn ~9× MACs). Nút thắt
>    nằm ở **đặc trưng + cấu trúc nhãn**, không phải ở bộ phân loại.
> 3. precision@recall0.95 của CNN/LSTM/CRNN **dao động ±0.05 chỉ do seed khởi tạo** → bản cũ
>    báo "CNN c8-16 = 0.242" là một lần **bốc seed may mắn, KHÔNG tái lập** (mean ~0.17).
>    Vì vậy nay train **5 seed** và xếp hạng theo **trung bình**.

---

## 🔬 Phương pháp lựa chọn (thay cho việc chọn size theo cảm tính)

Phát biểu việc triển khai như **tối ưu có ràng buộc**, mã hóa đúng 3 tiêu chí bài toán:

1. **Quét capacity knob** — mỗi họ có 1 tham số dung lượng đơn điệu chi phối chi phí ESP32
   (RF/XGB: tổng số node cây; SVM: #support vector; CNN/LSTM/CRNN: MACs/nhịp). Quét 6 điểm
   trên lưới log để **vẽ đường chất lượng–chi phí** và tìm điểm bão hòa (knee). Cấu hình
   trong `SWEEP=[...]` ở mỗi `src/models/<name>/sweep.py`.
2. **Điểm vận hành lâm sàng** — "bỏ sót bất thường là nguy hiểm nhất" → đặt **sàn độ nhạy
   recall ≥ 0.95** và **tinh chỉnh ngưỡng** (thay ngưỡng 0.5 mặc định). Ngưỡng dò trên tập
   **DS1-val patient-pure** (records 207/209/215), báo cáo trên DS2.
3. **Feasibility gate** — biến thể hợp lệ phải: `latency ≤ 100ms/nhịp` (đo thật trên ESP32,
   đã cộng chi phí trích đặc trưng cho RF/XGB/SVM) ∧ `flash ≤ 2MB` ∧ `RAM ≤ 64KB`. Thực tế
   **chỉ latency bite**: nó loại đúng `cnn_c16-32-64-64` (239ms) và `lstm_h32x2` (262ms).
   Triển khai thật chỉ nạp 1 model nên flash (max 1005KB) và RAM (~5.3KB) là lan can không
   bao giờ chạm — **tốc độ tính toán mới là ràng buộc thật**.
4. **Train lại trên nhiều seed** — precision @ recall 0.95 của mạng raw-beat dao động ~±0.05
   chỉ do seed khởi tạo, nên mỗi điểm được train **5 seed** ([`tools/multiseed.py`](tools/multiseed.py));
   báo cáo **trung bình ± std** và lưu checkpoint seed-median làm đại diện (export/parity dùng nó).
5. **Chọn**: trong vùng hợp lệ, **max precision *trung bình* @ recall 0.95** (tối thiểu báo
   động giả, và bền vững với nhiễu huấn luyện).

**Hai trục đo tách bạch**: *năng lực* (precision tại recall 0.95 đo trên DS2) vs *chuyển
ngưỡng* (recall thực tế khi dùng ngưỡng dò trên DS1). Quality lấy từ **toàn DS2 trên PC**
(tin được vì parity = 1.0000); thiết bị chỉ đo **latency/RAM**.

## 🏆 Kết quả (winner mỗi họ, tại recall 0.95 trên DS2)

**Phần cứng**: ESP32-D0WD-V3 (Xtensa LX6 @240MHz, 320KB RAM, 4MB flash, không PSRAM).
`prec@R` = precision tại recall 0.95 (*năng lực*); `lat` = độ trễ end-to-end đo thật
(RF/XGB/SVM đã + 0.94ms trích đặc trưng db4+FFT); `rec_dep` = recall khi dùng ngưỡng DS1
(*chuyển ngưỡng*).

| Họ | Biến thể tốt nhất | prec@0.95 (±std) | fpr | PR-AUC | rec_dep | latency | flash | hợp lệ |
|----|------|:----:|:----:|:----:|:----:|----:|----:|:----:|
| RF   | `rf_n10_d5`         | 0.127±0.01 | 0.83 | 0.74 | 0.78 ❗ | 1.0 ms | 12 KB | ✓ |
| XGB  | `xgb_x50_d2`        | 0.130±0.00 | 0.80 | 0.72 | 0.74 ❗ | *chưa đo* | 6.8 KB | ✓ |
| **SVM** | **`svm_linear`** | **0.193±0.00** | **0.49** | 0.73 | 0.92 ❗ | **1.0 ms** | **0.2 KB** | ✓ |
| CNN  | `cnn_c8-16`         | 0.172±0.05 | 0.63 | 0.63 | **0.965 ✓** | 19.8 ms | 53 KB | ✓ |
| LSTM | `lstm_h8`           | 0.152±0.01 | 0.66 | 0.48 | 0.987 ✓ | 31.1 ms | 1.4 KB | ✓ |
| CRNN | `crnn_c16-32-32_h32`| 0.154±0.03 | 0.68 | 0.55 | 0.885 ❗ | *chưa đo* | 64 KB | ✓ |

> 🏆 **Cross-family winner: `svm_linear`** — precision 0.193±0.00 @ recall 0.95 (cao nhất theo
> *trung bình 5 seed*, lại tất định), fpr 0.49, latency 1.0ms, flash 0.2KB. CNN `c8-16`
> (0.172±0.05) nằm trong dải nhiễu seed của nó nhưng mean thấp hơn và đắt hơn ~20×. Lưu ý:
> winner thắng theo *năng lực* nhưng có **khe hở hiệu chỉnh ngưỡng** (rec_dep 0.92 ❗); CNN/LSTM
> chuyển ngưỡng tốt hơn. Bảng đầy đủ 36 dòng + biểu đồ: [`results/summary.csv`](results/summary.csv),
> `results/{within_family_quality_vs_cost,cross_family_pareto,pr_curves}.png`; trước/sau làm
> sạch: [`results/archive/summary_v1_before_clean.csv`](results/archive/summary_v1_before_clean.csv).

> 📟 **Trạng thái benchmark thiết bị**: **24/36** mô hình (RF, SVM, CNN, LSTM) đã chạy thật
> trên board với **parity 1.0000** — log: [`esp32/benchmark_sweep.txt`](esp32/benchmark_sweep.txt).
> **XGBoost và CRNN đã có kernel C + đã export, nhưng chưa chạy lần đo trên board** (`group_b`),
> nên cột latency của chúng còn trống và feasibility gate chưa ràng buộc được. Điều này **không
> đổi winner** (prec@R cao nhất của XGB là 0.130 và của CRNN là 0.154, đều dưới 0.193 của
> `svm_linear`). Xem [§4](#4-convert--chạy-trên-esp32) để chạy nốt.

### Nhận xét (điều mà F1@0.5 che mất)

- **To hơn KHÔNG tốt hơn.** Knee nằm ở model nhỏ; feasibility gate loại đúng các cấu hình
  phình to: `cnn_c16-32-64-64` (239ms), `lstm_h32x2` (262ms) vượt latency.
- **Đổi bộ phân loại không cứu được bài toán.** XGBoost (boosting) chạm đúng trần của RF
  (bagging) trong *cùng* không gian 21 đặc trưng; CRNN (conv+recurrent) không hơn CNN thuần.
  Hai kết quả âm này **chính là phát hiện**: trần nằm ở **đặc trưng + cấu trúc nhãn**.
- **Single-run không đáng tin ở operating point này.** precision@recall0.95 của CNN/LSTM/CRNN
  dao động **~±0.05 chỉ do seed** → train 5 seed, xếp hạng theo trung bình. (XGB thì **tất
  định**, std = 0.000.)
- **Mô hình đơn giản nhất thắng theo trung bình:** `svm_linear` (0.193±0.00, tất định, 0.2KB)
  ≥ mọi họ phức tạp hơn — vốn chồng lấn trong dải nhiễu seed.
- **PR-AUC và precision@recall-0.95 mâu thuẫn nhau.** PR-AUC cao nhất toàn dự án là
  `rf_n80_d12` (**0.753**) nhưng prec@R của nó chỉ **0.116**; winner `svm_linear` có PR-AUC
  *thấp hơn* (0.730) mà prec@R cao nhất (0.193). **Không** đổi metric lựa chọn sang PR-AUC/F1.
- **Khe hở hiệu chỉnh ngưỡng**: `rec_dep` ❗ ≈0.74–0.92 cho RF/XGB/SVM vs ✓ ≈0.96–0.99 cho
  CNN/LSTM. Winner thắng theo *năng lực*; chuyển ngưỡng là điểm yếu của họ tuyến tính/cây.
- **Nút thắt là tính toán, không phải bộ nhớ**: working RAM ~5.5KB / 320KB.

---

## 1. Bài toán & dữ liệu

- **Đầu vào**: 1 nhịp (±100 mẫu quanh đỉnh R, ~555 ms @360 Hz) chuyển đạo **MLII** của
  MIT-BIH (chọn theo *tên lead* để mọi bản ghi cùng MLII — sửa record 114 vốn để V5 ở kênh 0).
- **Đầu ra**: `0 = Normal`, `1 = Abnormal`. Nhãn AAMI: `N,L,R,e,j → 0`, còn lại → `1`.
  Loại 4 bản ghi paced (102/104/107/217). Mất cân bằng ~11% Abnormal.
- **Chia theo bệnh nhân** (de Chazal `DS1` train / `DS2` test) — không shuffle chung để
  tránh rò rỉ beat cùng người (nếu shuffle, metrics bị thổi lên ~5–10 pp).
- **Làm sạch (label-free, không rò rỉ)** trước khi train, áp y hệt mọi tập: (1) chọn lead
  MLII theo tên; (2) RR đo giữa các **nhịp thật** liên tiếp (sửa ~4.2% nhịp bị nhiễm bởi
  annotation không-phải-nhịp); (3) căn cửa sổ vào **đỉnh R thật** (±15 mẫu); (4) lọc nhịp
  **vật lý hỏng** (flatline/clipping) theo tiêu chí tín hiệu — KHÔNG theo nhãn, KHÔNG theo
  biên độ. Thực tế chỉ loại vài nhịp (DS2 mất 0 → không cherry-pick test). Chi tiết + bảng
  trước/sau: [`results/cleaning_stats.json`](results/cleaning_stats.json), mục 2 của báo cáo.
- **Dataset**: MIT-BIH Arrhythmia DB (PhysioNet), tự tải qua `wfdb` lần chạy đầu, cache vào
  `dataset/cache/mitbih_<mode>_v2.npz` (key `v2` = phiên bản làm sạch).

## 2. Sáu họ mô hình (mỗi họ 6 điểm quét theo capacity knob = 36 mô hình)

Roster là một lưới **2×2 giữa *biểu diễn đầu vào* × *lớp mô hình***: 21 đặc trưng tay nuôi
RF / XGBoost / SVM; nhịp thô nuôi CNN / LSTM / CRNN.

| Mô hình | Framework | Đầu vào | Capacity knob → SWEEP (`src/models/<name>/sweep.py`) |
|---------|-----------|---------|------------------------------------------|
| **RF**   | sklearn  | 21 features | #node cây: `n10_d3 … n80_d12` (n_estimators × max_depth) |
| **XGB**  | xgboost  | 21 features | #node cây: `x50_d2 … x400_d6` (lr cố định 0.1) |
| **SVM**  | sklearn  | 21 features | #support vector: `linear`, `rbf1k … rbf10k` |
| **CNN**  | pytorch  | raw 200×1   | MACs: `c4`, `c8`, `c8-16`, `c16-16`, `c16-32-32`, `c16-32-64-64` |
| **LSTM** | pytorch  | seq 100×1   | MACs: `h4 … h32`, `h32x2` (2 lớp) |
| **CRNN** | pytorch  | raw 200×1   | MACs: `c8_h8 … c16-32-32_h32` (conv channels × LSTM hidden) |

XGBoost là đối trọng *boosting* của RF (*bagging*); CRNN là kiến trúc lai conv+recurrent của
Oh et al. 2018 (*Comput. Biol. Med.* 102:278-287). Cả hai được thêm bằng **đúng cùng một
phương pháp luận** (cùng sweep/seed/operating-point/gate) — nên hai kết quả âm ở trên là so
sánh công bằng.

Mất cân bằng lớp xử lý cùng một cách ở mọi họ: `class_weight="balanced"` (RF/SVM),
`scale_pos_weight=neg/pos` (XGB), weighted CE (nets). Knob, #params, flash-bytes tính trong
[`src/models/cost.py`](src/models/cost.py). Mỗi điểm dump metrics @0.5 / @ngưỡng-DS1 /
@recall-0.95 + cost ra `model/<name>_<size>_metrics.json`.

**Đặc trưng tay (RF/XGB/SVM, 21 chiều)**: thống kê time-domain (mean/std/min/max/skew/
kurtosis/RMS/p2p/energy/zero-crossings), năng lượng wavelet `db4` 4 mức, dominant frequency +
spectral entropy, RR-interval (prev/post/ratio/diff). Chi phí trích đặc trưng được **đo thật
trên ESP32** (db4+FFT, ~0.94ms/nhịp) và cộng vào latency của RF/XGB/SVM.

## 3. Cài đặt & huấn luyện (PC)

```bash
python3.10 -m venv env && source env/bin/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # xem requirements.txt

# chạy 1 họ end-to-end (tự tải dataset, tự bỏ qua điểm sweep đã train)
python -m src.models.rf     python -m src.models.cnn    # 1D CNN  (GPU)
python -m src.models.xgb    python -m src.models.lstm   # LSTM    (GPU)
python -m src.models.svm    python -m src.models.crnn   # CRNN    (GPU)
#  --retrain để ép train lại tất cả điểm sweep (single-seed, nhanh)

# ĐƯỜNG CHẠY ĐƯỢC BÁO CÁO: train mỗi điểm trên 5 seed, lưu seed-median + ghi mean/std
PYTHONPATH=. python tools/multiseed.py --seeds 5
PYTHONPATH=. python tools/multiseed.py --seeds 5 --families xgb crnn   # chỉ 1 vài họ

# tổng hợp + feasibility gate + chọn winner + vẽ biểu đồ
python -m src.analysis.analyze
python -m src.analysis.analyze \
  --latency esp32/benchmark_sweep.txt --feature-us 937.4 --pr-curves   # + latency đo thật
```

Mỗi họ tự load checkpoint nếu có, không thì train rồi lưu vào `model/`. **Luôn so sánh qua
`tools/multiseed.py`** (mean ± std trên 5 seed), đừng tin một lần chạy đơn lẻ — ở operating
point này nó xê dịch ±0.05 và sẽ tôn nhầm winner. `src/analysis/analyze.py` gom tất cả →
`results/summary.csv` (+ `winners.json`, `cleaning_stats.json`) + 3 biểu đồ (có error band) +
in winner. Không truyền `--latency` thì trục độ trễ để trống.

## 4. Convert & chạy trên ESP32

Pipeline gồm **một script export** (sinh C từ checkpoint, có cổng verify parity) và **một
firmware benchmark** (PlatformIO/Arduino) chạy các mô hình bằng kernel C tự viết — không dùng
TFLite-Micro nên khớp 100% với PC và không phụ thuộc toolchain convert. Vì parity = 1.0000,
thiết bị chỉ cần đo **latency + RAM** (N nhúng nhỏ = 20 nhịp); chất lượng lấy từ toàn DS2 trên PC.

> **Firmware phải chia LÀM HAI nhóm build.** Trọng số của cả 36 mô hình (~4.2 MB) vượt phân
> vùng `huge_app` 3 MB, nên [`esp32/platformio.ini`](esp32/platformio.ini) định nghĩa
> `group_a` (rf + svm + lstm, 75.5% flash) và `group_b` (xgb + cnn + crnn, 63.6%), chọn bằng
> `-DINC_<FAMILY>`. **Flash lần lượt cả hai nhóm và *nối* (append) cả hai bảng serial vào
> `esp32/benchmark_sweep.txt`** — `analyze.py` đọc hợp của chúng. Mỗi model giữ **chỉ số toàn
> cục** vào `Y_PC` nên parity vẫn đúng theo từng nhóm.

```bash
# 1) Sinh weights + 20 nhịp DS2 nhúng + bảng model -> esp32/include/*.h
#    (assert: numpy reproduction == sklearn/torch trên từng mẫu, sai là dừng)
PYTHONPATH=. ./env/bin/python tools/export_esp32.py

# 2) Build + flash TỪNG NHÓM (giữ nút BOOT khi esptool báo "Connecting....")
./env/bin/pio run -d esp32 -e group_a -t upload     # rf + svm + lstm
PYTHONPATH=. ./env/bin/python tools/read_serial.py /dev/ttyUSB0 90 "loop idle"

./env/bin/pio run -d esp32 -e group_b -t upload     # xgb + cnn + crnn
PYTHONPATH=. ./env/bin/python tools/read_serial.py /dev/ttyUSB0 90 "loop idle"

# 3) Nối cả hai bảng vào esp32/benchmark_sweep.txt rồi chạy lại analyze --latency
```

**Lưu ý phần cứng (board này)**: mạch auto-reset vào bootloader không ăn → phải **giữ nút
BOOT** lúc upload. `pio device monitor` cần TTY nên không chạy headless được — dùng
[`tools/read_serial.py`](tools/read_serial.py) thay thế.

### Cách hoạt động

- [`tools/export_esp32.py`](tools/export_esp32.py) — duyệt từng checkpoint, **dựng lại forward
  bằng NumPy** và assert khớp `sklearn/xgboost/torch.predict`, rồi xuất bảng config + trọng số
  ra `esp32/include/model_*.h`. RF→if/else; XGB→cây riêng (xem cảnh báo dưới); SVM→scaler +
  (linear+calibration | RBF SV); CNN→conv (BN gập sẵn) + GAP/FC head; LSTM→4 cổng i,f,g,o;
  CRNN→conv front-end + LSTM back-end.
- [`esp32/src/main.cpp`](esp32/src/main.cpp) — kernel C generic cho 6 họ + đo độ trễ/RAM, in
  bảng. Danh sách model dựng từ **bảng sinh tự động** (`MODEL_NAMES/FAM/SUB` trong
  `test_data.h`) nên tự co giãn theo sweep. `esp32/include/feature_bench.h` đo chi phí db4+FFT.
- Mở rộng/đổi điểm chỉ cần sửa `SWEEP = [...]` trong `src/models/<name>/sweep.py`, train, rồi
  export lại — kernel C + bảng model đã generic.

> ⚠️ **Duyệt cây của XGBoost KHÁC của RF.** XGB tách theo `<` (sklearn dùng `<=`) và lá của nó
> là **logit CỘNG DỒN** lên `logit(base_score)` (RF thì *trung bình* xác suất lá). Vì thế có
> `XGBCfg`/`xgb_infer` riêng — **đừng "hợp nhất"** hai kernel, parity sẽ vỡ trong im lặng.

## 5. Cấu trúc thư mục

```
pr_tinyml/
├── src/                      # THƯ VIỆN (thứ được import) — tách theo mối quan tâm
│   ├── config.py             # hằng số: paths, FS/beat, cleaning, budgets, DS1/DS2 split, AAMI
│   ├── seeding.py            # set_seed
│   ├── artifacts.py          # model_path / save_metrics_json
│   ├── data/                 # download · preprocess · segmentation · features · assembly · benchmark
│   ├── evaluation/           # metrics · operating_point
│   ├── models/               # cost.py + rf/ xgb/ svm/ cnn/ lstm/ crnn/
│   │                         #   (sweep · architecture|estimator · training · inference · __main__)
│   ├── training/             # multiseed.py — ĐƯỜNG CHẠY ĐƯỢC BÁO CÁO (5 seed → mean/std)
│   └── analysis/             # analyze.py (chọn winner) · writers.py (summary.csv/winners.json)
│                             #   · plots.py · load.py · selection.py · feasibility.py
├── tools/                    # ENTRY POINT (thứ được chạy) — vỏ CLI mỏng gọi vào src/
│   ├── multiseed.py          # vỏ CLI của src/training/multiseed.py
│   ├── export_esp32.py       # checkpoint → esp32/include/*.h (+ cổng parity + bảng model)
│   ├── read_serial.py        # đọc Serial ESP32 không cần TTY (không import src — tiện ích thuần)
│   ├── make_beat_figures.py  # hình minh họa hình thái nhịp cho báo cáo
│   └── make_report.py        # sinh results/BaoCao_TinyML_ECG.docx — CHỈ LÀ TEMPLATE
├── esp32/                    # PlatformIO project (Arduino framework)
│   ├── platformio.ini        # board=esp32dev, huge_app; env group_a / group_b
│   ├── src/main.cpp          # harness + kernel C generic cho 6 họ
│   ├── include/              # *.h sinh tự động (kernels.h; feature_bench.h db4+FFT)
│   └── benchmark_sweep.txt   # log latency đo trên board (nối cả 2 nhóm build vào đây)
├── docs/
│   └── BaoCaoDuAn.docx       # 📄 BÁO CÁO CHÍNH (viết tay — deliverable của đồ án)
├── results/                  # mọi thứ do MÁY sinh ra: summary.csv, winners.json, *.png
│   └── archive/              # ảnh chụp lịch sử (summary trước khi làm sạch dữ liệu)
├── dataset/  model/  env/    # tự tạo khi chạy — gitignored
├── requirements.txt  pyproject.toml  README.md
```

Hai quy ước giữ cho cây thư mục không rối:

1. **`src/` là thứ được *import*, `tools/` là thứ được *chạy*.** `src/` **không bao giờ** import
   ngược từ `tools/` — phụ thuộc chỉ đi một chiều. Script trong `tools/` chỉ là vỏ CLI mỏng, không
   phải chỗ chứa logic. Mọi `tools/*.py` cần `PYTHONPATH=.` (package `src` chưa pip-install).
2. **`docs/` là thứ người viết tay, `results/` là thứ máy sinh ra.** Đừng sửa tay file trong
   `results/` — lần chạy `analyze.py` kế tiếp sẽ ghi đè.

## 6. Việc tương lai

- **Chạy nốt benchmark `group_b`** (xgb + crnn) trên board để lấp cột latency còn trống.
- Tối ưu kernel CNN/LSTM/CRNN bằng **ESP-DSP / CMSIS-NN** để giảm độ trễ (mở rộng vùng hợp lệ).
- **Lượng tử hóa int8** (hiện đang fp32) để giảm flash + tăng tốc.
- **Tích hợp** trích đặc trưng db4+FFT vào đường chạy RF/XGB/SVM on-device (hiện mới *đo* chi
  phí qua `feature_bench.h`) để chạy end-to-end từ tín hiệu thô.
- Thu hẹp khe hở hiệu chỉnh ngưỡng giữa bệnh nhân (calibration transfer) cho RF/XGB/SVM.
- Vì trần nằm ở **đặc trưng**, hướng có triển vọng nhất là **đặc trưng tốt hơn / nhiều nhịp
  ngữ cảnh hơn**, chứ không phải mô hình lớn hơn.
