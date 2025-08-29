## Model Scaling Guide (PFT1D and Soil2D)

This guide explains which configuration knobs to adjust as your inputs grow (e.g., more PFT1D variables, more Soil2D variables, and Soil2D columns from 1 → 18).

### Key files and fields
- Model configuration: `config/training_config.py` → class `ModelConfig`
- Data configuration: `config/training_config.py` → class `DataConfig`
- CNP combined config builder: `config/training_config.py` → `get_cnp_combined_config`
- Loader (Soil2D slice to first-column/top-10): `data/data_loader_individual.py`
- CNP model (dual 1D/2D Soil encoder): `models/cnp_combined_model.py`

---

### 1) Output shapes MUST match targets
Set these in `ModelConfig` so that the model’s output head sizes match your preprocessed tensors.

- Soil2D output head
  - `matrix_output_size`: number of selected Soil2D variables → typically `len(y_list_columns_2d)`
  - `matrix_rows`: number of columns (grid columns)
  - `matrix_cols`: number of layers (depth layers)

- PFT1D output head
  - `vector_output_size`: number of selected PFT1D variables → `len(y_list_columns_1d)`
  - `vector_length`: number of PFTs (default 16)

- Scalars
  - `scalar_output_size`: number of selected scalars → `len(y_list_scalar_columns)`

Note: In `get_cnp_combined_config`, the guide currently aligns Soil2D to 1×10 (first column × top 10 layers) by setting:
`matrix_output_size = len(output_2d)`, `matrix_rows = 1`, `matrix_cols = 10`.

---

### 2) Data shape config (inputs)
Configure these to reflect how your dataset is stored on disk before slicing/padding:

- `DataConfig.max_2d_rows`: total Soil2D columns available in data (e.g., 2 or 18)
- `DataConfig.max_2d_cols`: total layers available in data (e.g., 15 or 10)

If you move from the special 1×10 mode to full multi-column mode, disable or toggle the slicing in `data/data_loader_individual.py` that extracts “first column, top 10 layers”.

---

### 3) Increase model capacity as inputs grow
Modify these in `ModelConfig`:

- Encoders
  - Time series (LSTM): `lstm_hidden_size`
  - Static/Scalars: `static_fc_size`, `scalar_fc_size`
  - PFT1D: `pft_1d_fc_size`
  - PFT parameters: set `use_cnn_for_pft_param = True` and adjust `pft_param_cnn_channels`
  - Soil2D CNN (dual-path 1D/2D): widen/deepen `conv_channels` (e.g., from `[16, 32, 64]` to `[32, 64, 128, 256]` when going 18×10)

- Fusion/Transformer
  - `token_dim`: width of fused features (increases capacity of fusion)
  - `transformer_layers`, `transformer_heads`: depth/width of fusion
  - `num_tokens` is computed in-model from concatenated feature size; you usually don’t need to set it manually

---

### 4) Training knobs (for larger models)
- `training_config.batch_size`: reduce if you hit OOM
- Mixed precision: `use_mixed_precision=True`, `use_amp=True`
- `learning_rate`: may need slight reduction for deeper/wider models

---

### 5) Quick recipes

#### A) From 1×10 to 18×10 Soil2D
1. Data config: set `DataConfig.max_2d_rows = 18`, `DataConfig.max_2d_cols = 10`
2. Disable the slice-to-1×10 in `data/data_loader_individual.py`, or add a toggle to switch modes
3. Model config: set `matrix_rows = 18`, `matrix_cols = 10`; keep `matrix_output_size = len(y_list_columns_2d)`
4. Increase Soil2D `conv_channels` (e.g., `[32, 64, 128, 256]`) and consider bumping `token_dim` / transformer sizes

#### B) Add more Soil2D variables
1. Update `x_list_columns_2d` and `y_list_columns_2d`
2. Ensure `matrix_output_size = len(y_list_columns_2d)`

#### C) Add more PFT1D variables
1. Update `x_list_columns_1d` and `y_list_columns_1d`
2. Set `vector_output_size = len(y_list_columns_1d)`; keep `vector_length = 16`
3. Consider increasing `pft_1d_fc_size` and fusion dims (`token_dim`, transformer)

---

### 6) Optional: make it switchable via flags
To avoid code edits when changing modes, add CLI flags (examples):
- `--soil2d-mode first_col_top10|all`
- `--soil2d-cols 1|18`, `--soil2d-layers 10|15`
- `--soil2d-conv "32,64,128,256"`, `--token-dim 256`, `--transformer-layers 6`, etc.

These can be wired in `train_cnp_model.py` to update `DataConfig` and `ModelConfig` before building the model.

---

### 7) Sanity checks
- Ensure output head sizes match target tensors exactly:
  - Soil2D: `matrix_output_size * matrix_rows * matrix_cols == y_soil_2d.view(N, -1).shape[1]`
  - PFT1D: `vector_output_size * vector_length == y_pft_1d.view(N, -1).shape[1]`
  - Scalars: `scalar_output_size == y_scalar.shape[1]`

If you see shape mismatches in loss (e.g., 540 vs 30), reconcile `ModelConfig` vs preprocessed target shapes and adjust the slice mode and/or rows/cols accordingly.


