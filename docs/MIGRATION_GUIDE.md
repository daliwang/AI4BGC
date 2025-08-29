# CNP Model Migration Guide

## Overview

This guide explains how to migrate from the old CNP model structure to the new organized structure while maintaining backward compatibility.

## 🚨 Important: No Breaking Changes!

**All existing validation scripts will continue to work without modification.** The new structure creates a backward compatibility layer that duplicates all data in the old format.

## 📁 Directory Structure Comparison

### Old Structure (Legacy)
```
cnp_results/run_YYYYMMDD_HHMMSS/
├── cnp_predictions/
│   ├── predictions_scalar.csv
│   ├── ground_truth_scalar.csv
│   ├── pft_1d_predictions/
│   ├── soil_2d_predictions/
│   └── test_metrics.csv
├── cnp_training_losses.csv
├── cnp_config.json
└── cnp_training_YYYYMMDD_HHMMSS.log
```

### New Structure (Organized)
```
cnp_results/run_YYYYMMDD_HHMMSS/
├── predictions/
│   ├── normalized/          # 0-1 range predictions
│   │   ├── predictions_scalar.csv
│   │   ├── ground_truth_scalar.csv
│   │   ├── pft_1d_predictions/
│   │   └── soil_2d_predictions/
│   ├── denormalized/        # Physical units (gC/m²/day)
│   │   ├── predictions_scalar.csv
│   │   ├── ground_truth_scalar.csv
│   │   ├── pft_1d_predictions/
│   │   └── soil_2d_predictions/
│   └── scalers/             # Saved scalers
│       ├── y_scalar_scaler.pkl
│       ├── y_pft_1d_scaler.pkl
│       ├── y_soil_2d_scaler.pkl
│       └── scaler_metadata.json
├── training/
│   ├── losses.csv
│   └── cnp_training_YYYYMMDD_HHMMSS.log
├── validation/
│   └── cnp_metrics.json
├── metadata/
│   ├── cnp_config.json
│   └── resolved_config.json
├── cnp_predictions/         # BACKWARD COMPATIBILITY
│   ├── predictions_scalar.csv
│   ├── ground_truth_scalar.csv
│   ├── pft_1d_predictions/
│   ├── soil_2d_predictions/
│   └── test_metrics.csv
├── cnp_training_losses.csv  # BACKWARD COMPATIBILITY
└── README.md
```

## 🔄 Migration Paths

### Path 1: Immediate (No Changes Required)
- **Status**: ✅ Ready to use
- **Action**: None required
- **Benefit**: All existing scripts work unchanged
- **Trade-off**: Data is duplicated (larger disk usage)

### Path 2: Gradual (Update Scripts)
- **Status**: 🚧 Requires script updates
- **Action**: Update validation scripts to use new structure
- **Benefit**: Better organization, access to denormalized data
- **Trade-off**: Requires development effort

### Path 3: Full Migration (Leverage New Features)
- **Status**: 🚧 Requires script updates
- **Action**: Update scripts to use scalers and denormalized data
- **Benefit**: Physical units, inverse transformations, better analysis
- **Trade-off**: Requires significant development effort

## 📋 Script Compatibility Status

### ✅ Fully Compatible (No Changes Needed)
- `cnp_result_validationplot.py` - Works with old structure
- `ai_predictions_to_netcdf.py` - Works with old structure
- `ai_model_comparison_plot.py` - Works with old structure
- `run_inference_all.py` - Updated for new structure

### 🔄 Partially Compatible (Minor Updates Recommended)
- Custom validation scripts using hardcoded paths
- Scripts expecting specific file locations

## 🛠️ How to Update Scripts for New Structure

### 1. Detect Structure Automatically
```python
def detect_structure(results_dir):
    """Detect whether results use new or old structure."""
    if (Path(results_dir) / "predictions" / "normalized").exists():
        return "new"
    elif (Path(results_dir) / "cnp_predictions").exists():
        return "legacy"
    else:
        return "unknown"

def get_predictions_path(results_dir, structure_type):
    """Get predictions path based on structure type."""
    if structure_type == "new":
        return Path(results_dir) / "predictions"
    else:
        return Path(results_dir) / "cnp_predictions"
```

### 2. Use New Structure Paths
```python
# Instead of hardcoded paths:
# old_path = os.path.join(results_dir, 'cnp_predictions', 'predictions_scalar.csv')

# Use flexible paths:
structure = detect_structure(results_dir)
predictions_dir = get_predictions_path(results_dir, structure)

if structure == "new":
    # Use organized structure
    normalized_file = predictions_dir / "normalized" / "predictions_scalar.csv"
    denormalized_file = predictions_dir / "denormalized" / "predictions_scalar.csv"
    scalers_dir = predictions_dir / "scalers"
else:
    # Use legacy structure
    predictions_file = predictions_dir / "predictions_scalar.csv"
```

### 3. Access Denormalized Data
```python
def load_predictions(results_dir, structure_type, data_type="normalized"):
    """Load predictions with automatic structure detection."""
    if structure_type == "new":
        if data_type == "denormalized":
            # Load physical units data
            file_path = Path(results_dir) / "predictions" / "denormalized" / "predictions_scalar.csv"
        else:
            # Load normalized data
            file_path = Path(results_dir) / "predictions" / "normalized" / "predictions_scalar.csv"
    else:
        # Load from legacy structure
        file_path = Path(results_dir) / "cnp_predictions" / "predictions_scalar.csv"
    
    return pd.read_csv(file_path)
```

## 🎯 Benefits of New Structure

### 1. **Scaler Persistence**
- All normalization parameters are saved
- Can apply inverse transformations to new data
- Reproducible results across different runs

### 2. **Physical Units**
- Denormalized results in meaningful units (gC/m²/day)
- Better scientific interpretation
- Easier comparison with literature values

### 3. **Better Organization**
- Clear separation of normalized vs. denormalized data
- Logical grouping of related files
- Easier to find specific data types

### 4. **Future-Proof**
- Extensible structure for new data types
- Better metadata and documentation
- Easier to add new analysis tools

## 📊 Performance Impact

### Disk Usage
- **Old structure**: Base usage
- **New structure**: ~2x usage (data duplicated for compatibility)
- **Migration complete**: ~1.5x usage (remove compatibility layer)

### Script Performance
- **Old scripts**: No change in performance
- **New scripts**: Potentially faster (better organized data)
- **Hybrid approach**: Best of both worlds

## 🚀 Next Steps

### For Users (Immediate)
1. **No action required** - all existing scripts work
2. **Explore new structure** - check `/predictions/` directory
3. **Use denormalized data** - for scientific analysis

### For Developers (Gradual)
1. **Update validation scripts** to detect structure automatically
2. **Add support** for denormalized data
3. **Leverage scalers** for inverse transformations

### For System Administrators (Future)
1. **Monitor disk usage** - new structure uses more space
2. **Plan migration** - when to remove compatibility layer
3. **Update documentation** - reflect new capabilities

## ❓ Frequently Asked Questions

### Q: Will my existing scripts break?
**A**: No! The backward compatibility layer ensures all existing scripts continue to work unchanged.

### Q: How much more disk space will I need?
**A**: Approximately 2x the current usage due to data duplication. This can be reduced by migrating scripts to use the new structure.

### Q: Can I access the denormalized data?
**A**: Yes! The new structure provides both normalized and denormalized versions. Use `/predictions/denormalized/` for physical units.

### Q: How do I migrate my custom scripts?
**A**: Use the structure detection functions provided above. Start with automatic detection, then gradually add support for new features.

### Q: When should I remove the compatibility layer?
**A**: Only after all your scripts and workflows have been updated to use the new structure. This ensures a smooth transition.

## 📞 Support

If you encounter any issues during migration:

1. **Check the README.md** in your results directory
2. **Verify structure detection** using the provided functions
3. **Fall back to legacy paths** if needed
4. **Report issues** to the development team

## 🎉 Conclusion

The new CNP model structure provides significant improvements while maintaining full backward compatibility. You can:

- **Start immediately** with no changes required
- **Gradually migrate** to leverage new features
- **Plan for the future** with better organized data

The choice is yours - migrate at your own pace while enjoying the benefits of the new structure!
