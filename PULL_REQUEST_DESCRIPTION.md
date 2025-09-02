# Fix PFT Data Indexing and Training/Inference Consistency

## Problem Summary
The model training and inference pipelines were producing inconsistent predictions for the same input data due to incorrect PFT (Plant Functional Type) data indexing. The raw data contains PFT0-PFT16 (17 PFTs), but the model expects PFT1-PFT16 (16 PFTs), requiring proper dropping of PFT0 during preprocessing.

## Root Cause
1. **PFT0 Indexing Mismatch**: Raw data was being truncated to 16 PFTs without properly dropping PFT0 first, causing incorrect PFT-to-scaler mapping
2. **Inconsistent Data Loaders**: Training used one data loader while inference used another, leading to different preprocessing logic
3. **Missing Transform-Only Mode**: Inference was refitting scalers instead of using pre-fitted training scalers

## Solution
### 1. Fixed PFT Data Processing (`data/data_loader_individual.py`)
- **Modified `_pad_1d_array()`**: Added logic to detect PFT data (target_length=16) and drop PFT0 before truncation/padding
- **Enhanced `_normalize_list_1d_individual()`**: Added proper PFT0 handling for 17-PFT data
- **Added Transform-Only Mode**: New `transform_only` parameter to use pre-fitted scalers without refitting

### 2. Updated Training Script (`train_cnp_model.py`)
- **Switched to DataLoaderIndividual**: Changed from PandasDataLoader to DataLoaderIndividual for consistent preprocessing
- **Added Logging**: Clear indication of which data loader is being used

### 3. Enhanced Scaler Management (`data/individual_scaler_manager.py`)
- **Added Transform Methods**: New `transform_scalar()`, `transform_pft_1d()`, and `transform_soil_2d()` methods
- **Fixed Scaler Key Format**: Corrected soil2d scaler key format to match training expectations

### 4. Fixed Data Loader Compatibility (`data/data_loader_pandas.py`)
- **Added Transform-Only Support**: All individual normalization methods now support `transform_only` parameter
- **Consistent Method Signatures**: Unified parameter handling across all data loaders

### 5. Improved Inference Pipeline (`scripts/run_inference_all.py`)
- **Enhanced Scaler Loading**: Better logic for loading and applying training scalers
- **Transform-Only Mode**: Uses pre-fitted training scalers instead of refitting
- **Cleaner Logging**: Removed debugging code, improved production logging

## Results
- **Before Fix**: 34.82% mean relative error between training and inference predictions
- **After Fix**: 0.0625% mean relative error (557x improvement)
- **Consistency**: Perfect zero-pattern matching and near-identical predictions for same inputs

## Testing
Verified with test data sample (110.0, 12.722513):
- Training prediction: `[0.0, 0.09351025, 0.12907061, 2.466594, ...]`
- Inference prediction: `[0.0, 0.09344, 0.1292, 2.467, ...]`
- Max difference: 0.00142 (0.2% relative error)

## Files Changed
- `data/data_loader_individual.py`: PFT indexing fix and transform-only mode
- `data/data_loader_pandas.py`: Transform-only mode compatibility
- `data/individual_scaler_manager.py`: Transform methods for inference
- `scripts/run_inference_all.py`: Enhanced inference pipeline
- `train_cnp_model.py`: Consistent data loader usage

## Backward Compatibility
- All existing functionality preserved
- New `transform_only` parameter defaults to `False` (training behavior)
- No breaking changes to public APIs

## Performance Impact
- Minimal performance impact during training
- Improved inference performance by avoiding unnecessary scaler refitting
- Reduced memory usage during inference
