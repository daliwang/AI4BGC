# CNP Model Normalization Fix Guide

## Overview

This document describes the comprehensive fix for the normalization issue in CNP model training results. The problem was that all predictions and ground truth data were being saved in normalized form (0-1 range) without preserving the original physical units, making the results difficult to interpret and use.

## Problem Description

### What Was Wrong

1. **All predictions were normalized**: GPP, NPP, AR, HR values were in 0-1 range instead of physical units (gC/m²/day)
2. **Ground truth was also normalized**: Making comparison meaningless
3. **No inverse transformation**: No way to restore original scale
4. **Misleading metrics**: R² values of 0.98+ were comparing normalized values, not actual carbon fluxes
5. **Loss of physical meaning**: Could not interpret results in real-world units

### Impact

- **Loss of interpretability**: Cannot tell if model predicts realistic carbon fluxes
- **Misleading performance**: Excellent R² values were artifacts of normalization
- **Limited usability**: Results cannot be used for ecosystem modeling without denormalization
- **Reproducibility issues**: Cannot compare with other models or observations

## Solution Components

### 1. Enhanced Data Loader (`data/data_loader.py`)

**New Features:**
- `save_scalers()`: Saves all scalers to disk with metadata
- `load_scalers()`: Loads saved scalers from disk
- `inverse_transform_predictions()`: Applies inverse transformation to predictions
- `get_original_data_ranges()`: Captures original data ranges before normalization

**What It Does:**
- Persists scaler objects during training
- Saves comprehensive metadata about each scaler
- Provides inverse transformation capabilities
- Documents original data ranges for reference

### 2. Enhanced Trainer (`training/trainer.py`)

**New Features:**
- Automatically saves scalers during training
- Saves both normalized and denormalized predictions
- Captures original data ranges
- Creates comprehensive scaler metadata

**What It Does:**
- Saves scalers alongside model weights
- Applies inverse transformation automatically
- Provides both versions of predictions for comparison
- Documents the complete transformation pipeline

### 3. Scaler Management Utility (`utils/scaler_manager.py`)

**New Features:**
- `ScalerManager` class for loading and using saved scalers
- Inverse transformation methods for all data types
- Validation and compatibility checking
- Comprehensive error handling

**What It Does:**
- Loads saved scalers from disk
- Applies inverse transformations to new predictions
- Validates data compatibility
- Provides detailed logging and error reporting

### 4. Fix Existing Results Script (`scripts/fix_existing_results.py`)

**New Features:**
- Processes existing training results
- Applies inverse transformations to saved predictions
- Creates fallback scalers when originals are missing
- Generates comprehensive reports

**What It Does:**
- Fixes results from previous training runs
- Creates denormalized versions of existing predictions
- Handles cases where original scalers are missing
- Provides detailed migration reports

## Migration Strategy

### Phase 1: Immediate Fix for New Training

1. **Update existing branches**: Apply the enhanced data loader and trainer
2. **Retrain models**: New training runs will automatically save scalers
3. **Verify functionality**: Ensure both normalized and denormalized outputs are saved

### Phase 2: Fix Existing Results

1. **Identify affected results**: Find all training runs without scalers
2. **Apply fix script**: Use `fix_existing_results.py` to process existing results
3. **Validate outputs**: Verify denormalized predictions make physical sense
4. **Update documentation**: Document the new denormalized results

### Phase 3: Update All Branches

1. **Main branch**: Apply all fixes
2. **Release branches**: Propagate fixes to release0.1, cnp_model_exp, etc.
3. **Feature branches**: Update any active development branches
4. **Documentation**: Update all relevant documentation and examples

## Usage Examples

### For New Training Runs

The enhanced trainer will automatically:
- Save scalers to `results/scalers/` directory
- Save both normalized and denormalized predictions
- Create comprehensive metadata files

### For Existing Results

```bash
# Fix existing results with fallback scalers
python scripts/fix_existing_results.py \
    --results-dir /path/to/existing/results \
    --output-dir /path/to/fixed/results \
    --create-fallbacks

# Create denormalization report
python utils/scaler_manager.py \
    --predictions-dir /path/to/results \
    --create-report
```

### For New Predictions

```python
from utils.scaler_manager import ScalerManager

# Load saved scalers
scaler_manager = ScalerManager("/path/to/results/scalers")

# Apply inverse transformation to new predictions
denormalized = scaler_manager.inverse_transform_scalar(normalized_predictions)
```

## File Structure After Fix

```
results/
├── model.pth                    # Model weights
├── cnp_training_losses.csv     # Training history
├── test_metrics.csv            # Performance metrics
├── scalers/                    # NEW: Saved scalers
│   ├── y_scalar_scaler.pkl
│   ├── y_pft_1d_scaler.pkl
│   ├── y_soil_2d_scaler.pkl
│   └── scaler_metadata.json
├── original_data_ranges.json   # NEW: Original data ranges
├── predictions_scalar_normalized.csv      # Normalized predictions
├── predictions_scalar_denormalized.csv   # NEW: Denormalized predictions
├── ground_truth_scalar_normalized.csv    # Normalized ground truth
├── ground_truth_scalar_denormalized.csv # NEW: Denormalized ground truth
└── [other prediction files...]
```

## Validation and Testing

### What to Check

1. **Scaler files exist**: Verify `scalers/` directory is created
2. **Both versions saved**: Check for both normalized and denormalized files
3. **Physical units**: Verify denormalized values are in reasonable ranges
4. **Metadata complete**: Ensure all scaler information is documented

### Expected Ranges

After denormalization, expect:
- **GPP**: 0-20 gC/m²/day (typical range)
- **NPP**: 0-15 gC/m²/day (typical range)
- **AR**: 0-10 gC/m²/day (typical range)
- **HR**: 0-12 gC/m²/day (typical range)

### Quality Checks

1. **No negative values**: Carbon fluxes should be non-negative
2. **Reasonable ranges**: Values should match ecosystem expectations
3. **Consistent units**: All variables should use consistent units
4. **Metadata accuracy**: Scaler information should be complete

## Rollback Plan

If issues arise:

1. **Keep original files**: Normalized predictions are still available
2. **Disable new features**: Can temporarily disable scaler saving
3. **Gradual rollout**: Apply fixes to subset of branches first
4. **Monitoring**: Watch for any unexpected behavior

## Future Improvements

### Planned Enhancements

1. **Unit standardization**: Ensure consistent units across all variables
2. **Validation scripts**: Automated checks for physical plausibility
3. **Integration tests**: Comprehensive testing of the transformation pipeline
4. **Performance optimization**: Optimize inverse transformation for large datasets

### Long-term Goals

1. **Standardized output**: Consistent format across all model variants
2. **Unit documentation**: Clear documentation of all physical units
3. **Validation framework**: Automated validation of model outputs
4. **Interoperability**: Easy integration with other ecosystem models

## Support and Troubleshooting

### Common Issues

1. **Missing scalers**: Use `--create-fallbacks` option
2. **Shape mismatches**: Check data dimensions match scaler expectations
3. **Memory issues**: Process large datasets in chunks
4. **Compatibility**: Ensure sklearn version compatibility

### Getting Help

1. **Check logs**: Detailed logging is provided
2. **Review metadata**: Scaler metadata contains diagnostic information
3. **Test with small data**: Verify functionality with subset of data
4. **Report issues**: Document any problems for future fixes

## Conclusion

This normalization fix addresses a critical issue that was undermining the interpretability and usability of CNP model results. By implementing comprehensive scaler persistence and inverse transformation capabilities, we ensure that:

1. **New training runs** automatically save all necessary information
2. **Existing results** can be fixed and restored to physical units
3. **Future predictions** can be easily denormalized
4. **All branches** benefit from the improved functionality

The fix is designed to be backward-compatible and provides multiple fallback options to handle edge cases. With proper implementation across all branches, the CNP model will provide results that are both technically accurate and physically meaningful.
