# AI4BGC v0.1 – Artificial Intelligence for Biogeochemical Cycles

AI4BGC is a deep learning framework for modeling terrestrial biogeochemical cycles with modern neural architectures and a flexible data pipeline.

---

## 🚀 Quick Start

### 1) Clone and setup environment
```bash
git clone <your-repo-url>
cd AI4BGC
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Train the quick-start default model
```bash
python train_model.py
```
- Small example dataset, fast to run, great for validation and demos.

### 3) Train the full CNP model
```bash
python train_cnp_model.py \
  --output-dir cnp_results \
  --epochs 50 \
  --batch-size 128 \
  --learning-rate 1e-4 \
  --use-trendy1 --use-trendy05 \
  --variable-list CNP_IO_default.txt
```
- Uses production datasets (not included here) and full variable lists.
- See `python train_cnp_model.py --help` for all options.

---

## 🧠 Training defaults and controls

- Determinism: relaxed by default for performance. Enable strict determinism when needed:
  - CLI: `--strict-determinism`
- Mixed precision: enabled by default (AMP + grad scaler) when supported.
- Outputs per run (under `--output-dir` with timestamped subfolders):
  - `cnp_model.pt`, `cnp_training_losses.csv`, `cnp_predictions/` (including `predictions_scalar.csv`, `pft_1d_predictions/`, `soil_2d_predictions/`).

---

## 📊 Comparing runs

Use `scripts/compare_cnp_runs.py` to compare two runs (losses + predictions):
```bash
python scripts/compare_cnp_runs.py \
  cnp_results/run_YYYYmmdd_HHMMSS \
  cnp_results/run_YYYYmmdd_HHMMSS \
  --out-dir compare_out \
  --soil-layers 10 \
  --pft-count 16 \
  --plot-scalar  # use --no-plot-scalar to disable
```
- Saves into `--out-dir`:
  - `loss.png`
  - `summary_stats.csv` (RMSE, Corr, means for all compared columns)
  - Scatter plots:
    - Soil 2D: first N layers per variable (default 10)
    - PFT 1D: PFT1..PFTN per variable (default 16)
    - Scalar: plotted only if `--plot-scalar` (on by default)

---

## 🔁 Updating restart files with AI predictions

The `scripts/update_restart_with_ai.py` utilities can run inference or reuse an existing run to write predictions back into restart NetCDFs.

Examples:
- Reuse an existing run’s predictions:
```bash
python scripts/update_restart_with_ai.py \
  --run-dir cnp_results/run_YYYYmmdd_HHMMSS \
  --input-nc path/to/input_restart.nc \
  --output-nc path/to/output_restart_with_ai.nc \
  --variable-list CNP_IO_default.txt
```
- Run inference first, then update restart:
```bash
python scripts/update_restart_with_aiprediction.py \
  --model-path cnp_results/run_YYYYmmdd_HHMMSS/cnp_predictions/model.pth \
  --data-paths /path/to/pkls1 /path/to/pkls2 \
  --file-pattern '1_training_data_batch_*.pkl' \
  --device cuda \
  --out-dir cnp_infer \
  --input-nc path/to/input_restart.nc \
  --output-nc path/to/output_restart_with_aiprediction.nc \
  --variable-list CNP_IO_default.txt
```

---

## 🧪 NetCDF diff utility

Compare two NetCDF files variable-by-variable with per-layer metrics:
```bash
python scripts/ncdiff2.py file1.nc file2.nc --save-diff-list diffs.csv
```
- Reports dtype/shape differences, NaN/mask counts, NRMSE, and per-layer summaries (treating the last axis as layers).

---

## 📁 Repository structure (abridged)
```
AI4BGC/
├── config/                    # Training/data/model configs
├── training/                  # Training loop and helpers
├── models/                    # Model architectures
├── data/                      # Data loading utilities
├── scripts/                   # Analysis and utility scripts
│   ├── compare_cnp_runs.py
│   ├── compare_predictions.py
│   ├── cnp_result_validationplot.py
│   ├── update_restart_with_aiprediction.py
│   └── ncdiff2.py
├── train_model.py             # Quick-start training
├── train_cnp_model.py         # Full CNP training
├── requirements.txt
└── README.md
```

---

## 📝 Release notes (v0.1)
- Relaxed determinism by default; opt-in strict determinism via `--strict-determinism`.
- `compare_cnp_runs.py`: consolidated outputs to `--out-dir`, plots for soil (N layers) and PFT (N PFTs), summary CSV, scalar plotting toggle.
- Improved prediction I/O alignment and naming normalization for PFT columns.
- Cleaned CLI ergonomics with sensible defaults.

---

## 🤝 Contributing
Issues and PRs are welcome. Please include reproduction steps, logs, and environment details. 
