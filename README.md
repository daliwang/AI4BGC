# LandSim v0.1 – Land Simulator for Multimodal Dynamics

LandSim (formerly AI4BGC) is a deep learning framework for modeling terrestrial ecosystem dynamics with modern neural architectures and a flexible data pipeline.

---

## 🚀 Quick Start

### 1) Clone and setup environment
```bash
git clone <your-repo-url>
cd LandSim
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

## 🔗 CNP pipeline workflow (from runbook)

Follow this streamlined workflow using a user-defined `CNP_IO` list (see `docs/CNP_pipeline_runbook.md` for details):

1) Create your `CNP_IO` list (e.g., `CNP_IO_demo1.txt`).

2) Train the AI model:
```bash
python train_cnp_model.py --variable-list CNP_IO_demo.txt --epochs 100 2>&1 &
```

3) Navigate to the run directory:
```bash
cd cnp_results/run_YYYYMMDD_HHMMSS
```

4) Validate predictions vs ground truth (test split):
```bash
python ../../scripts/cnp_result_validationplot.py > cnp_results_validation.log 2>&1 &
```

5) Run inference on the entire dataset:
```bash
python ../../scripts/run_inference_all.py > run_inference_all.log 2>&1 &
```

6) Export AI predictions to NetCDF for plotting/comparison:
```bash
python ../../scripts/ai_predictions_to_netcdf.py  > ai_prediction_to_netcdf.log 2>&1 &
```

7) Generate comparison plots:
```bash
python ../../scripts/ai_model_comparison_plot.py  > ai_model_comparison.log 2>&1 &
```

8) Create a new ELM restart file using AI predictions:
```bash
python ../../scripts/ai_predictions_to_restart.py --variable-list ../../CNP_IO_demo.txt
```

9) Compare restart files (optionally inspect with `restart_variable_plot.py`):
```bash
python ../../scripts/ai_restart_comparison.py --variable-list ../../CNP_IO_demo.txt --layers 0,5,9 --pfts 0,1,2,3,4,5
# Optional: ../../scripts/restart_variable_plot.py
```

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

Preferred workflow uses `scripts/ai_predictions_to_restart.py` after generating predictions (see workflow above). For bespoke flows, the legacy helper is available:

```bash
python scripts/update_restart_with_aiprediction.py \
  --model-path cnp_results/run_YYYYmmdd_HHMMSS/cnp_predictions/model.pth \
  --data-paths /path/to/pkls1 /path/to/pkls2 \
  --file-pattern '1_training_data_batch_*_.pkl' \
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
python scripts/ncdiff3.py file1.nc file2.nc --save-diff-list diffs.csv
```
- Reports dtype/shape differences, NaN/mask counts, NRMSE, and per-layer summaries (treating the last axis as layers).

---

## 📁 Repository structure (abridged)
```
LandSim/
├── config/                    # Training/data/model configs
├── training/                  # Training loop and helpers
├── models/                    # Model architectures
├── data/                      # Data loading utilities
├── scripts/                   # Analysis and utility scripts
│   ├── compare_cnp_runs.py
│   ├── compare_predictions.py
│   ├── cnp_result_validationplot.py
│   ├── ai_predictions_to_netcdf.py
│   ├── ai_model_comparison_plot.py
│   ├── run_inference_all.py
│   ├── ai_predictions_to_restart.py
│   ├── ai_restart_comparison.py
│   └── ncdiff3.py
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
