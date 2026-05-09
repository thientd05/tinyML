# CLAUDE.md — pr_tinyml

Project context for Claude Code. Update this file when the architecture or
workflow changes; keep it under ~150 lines.

## Project overview

ECG anomaly detection pipeline targeting **ESP32**. Compares four model
families (Random Forest, SVM, 1D CNN, LSTM) on the MIT-BIH Arrhythmia DB,
binary task (Normal vs Abnormal). Stage 1 (this repo): training + comparison
on PC. Stage 2 (later): convert + deploy to ESP32. **No on-device code yet.**

## Environment

- **Python**: 3.10.19 (`/usr/bin/python3.10`). Venv at `./env/`.
- **GPU**: NVIDIA GTX 1650 (4 GB, compute 7.5). Driver CUDA 13.0, system nvcc 12.0.
- **PyTorch**: 2.7.x with CUDA wheel `cu126` (forward-compat with our driver).
  Installed *separately* from `requirements.txt`.
- **Other key libs**: scikit-learn, scipy, wfdb (MIT-BIH I/O + downloader),
  PyWavelets, joblib, matplotlib, tqdm.

## Common commands

```bash
source env/bin/activate

# (re)install
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# verify CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# run a model end-to-end (auto-downloads dataset, auto-skips trained sizes)
python -m src.rf
python -m src.svm
python -m src.cnn         # GPU
python -m src.lstm        # GPU
python -m src.cnn --retrain --epochs 25 --batch-size 256
```

All commands assume CWD is the repo root so that `from src import utils` works.

## Data pipeline (in `src/utils.py`)

1. `download_mitbih()` — idempotent `wfdb.io.dl_database('mitdb', dataset/...)`.
2. Per record: bandpass 0.5–40 Hz Butterworth (filtfilt) → z-score per record.
3. `extract_beats(rid)` — slice ±100 samples around each annotated R-peak (200
   samples ≈ 555 ms at 360 Hz). RR intervals (prev/post) returned alongside.
4. `aami_binary_label`: `N/L/R/e/j → 0`, anything else → `1`. Non-beat
   annotations are skipped.
5. `build_dataset(mode)` — assembles full DS1/DS2 split, returns numpy arrays,
   caches to `dataset/cache/mitbih_<mode>.npz`.
   - `mode="features"` → ~30 hand-crafted features (RF, SVM)
   - `mode="raw"` → 200×1 raw beat (CNN)
   - `mode="lstm"` → 100×1 raw beat downsampled by 2 (LSTM)

**Patient-level split** (de Chazal). Records 102/104/107/217 (paced beats) are
excluded from both DS1 and DS2.

## Model roster — 4 sizes per family (all sized to fit ESP32 in fp32, TBD)

| File         | Backend  | Input          | Sizes (config in file)                                     |
|--------------|----------|----------------|------------------------------------------------------------|
| `src/rf.py`  | sklearn  | features       | RF n_est=10/20/40/80, max_depth=4/6/8/10                   |
| `src/svm.py` | sklearn  | features       | LinearSVC; RBF SVC trained on 2k/5k/10k stratified samples |
| `src/cnn.py` | pytorch  | raw 200×1      | conv [8] → [8,16] → [16,32,32] → [16,32,64,64]             |
| `src/lstm.py`| pytorch  | raw 100×1 (ds2)| hidden=8/16/32/32; layers=1/1/1/2                          |

Checkpoints: `model/<name>_<size>.<pkl|pt>`. Per-size metrics dumped at
`model/<name>_<size>_metrics.json` (accuracy, precision, recall, F1, AUC, CM).

**Convention**: each model file's `main()` iterates 4 sizes, *loads* if the
checkpoint exists, *trains* if it doesn't, then runs evaluation on `DS2`. Add
`--retrain` to force retraining all sizes.

## Gotchas

- **First run is slow**: downloads ~70 MB MIT-BIH + builds the per-mode cache.
  Subsequent runs hit `dataset/cache/*.npz` and start in seconds.
- **Class imbalance**: ~10–15% abnormal beats. All training paths use balanced
  class weights. Don't switch metrics to plain accuracy as a primary signal —
  watch F1 and recall on the abnormal class.
- **Patient-level split is critical**. Do not shuffle DS1 + DS2 together — that
  leaks beats from the same patient and inflates metrics by ~5–10 pp.
- **GTX 1650 has 4 GB VRAM**. CNN uses mixed precision; the LSTM is small
  enough to run fp32. If you OOM, drop `--batch-size`.
- **Don't commit weights or cache**: `model/`, `dataset/`, `env/` are
  gitignored. Metrics JSONs are small and could be committed if you want to
  track regressions, but currently are also gitignored as part of `model/`.
- **Sizes are educated guesses for ESP32**. After Stage 2 measurements, edit
  the `SIZES = [...]` list at the top of each `src/<name>.py`.

## Adding a new dataset later

`src/utils.py` is single-dataset for now. To extend:

1. Add a new `download_<name>()` + `extract_beats_<name>()`.
2. Branch in `build_dataset(...)` on a `dataset` argument.
3. Keep the output schema identical so the four model files don't have to change.

## What's intentionally NOT in this repo

- On-device code (C/C++ for ESP32). Stage 2.
- Quantization / pruning. Stage 2 too — current sizes are fp32.
- Multi-class AAMI (5-class). User chose binary explicitly; if revisited,
  generalize `aami_binary_label` and `compute_metrics` (multi-class F1).
