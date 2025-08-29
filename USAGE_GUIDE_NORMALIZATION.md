# CNP Model Normalization Usage Guide

This guide explains how to use the three different normalization approaches available in the `DataLoaderIndividual` class.

## Overview

The `DataLoaderIndividual` class provides three normalization approaches:

1. **Group Normalization** (default) - Uses one scaler per data type
2. **Individual Normalization** - Uses individual scalers for each variable
3. **Hybrid Normalization** - Mix of both approaches

## Quick Start

### 1. Group Normalization (Default)

```python
from data.data_loader_individual import DataLoaderIndividual

# Create data loader
data_loader = DataLoaderIndividual(data_config, preprocessing_config)

# Load and preprocess data
data_loader.load_data()
data_loader.preprocess_data()

# Use group normalization (default)
normalized_data = data_loader.normalize_data()
```

**When to use:**
- Quick testing and development
- Memory-constrained environments
- When you want consistent behavior with the original system

### 2. Individual Normalization

```python
# Use individual normalization for optimal performance
normalized_data = data_loader.normalize_data_individual()
```

**When to use:**
- Production training runs
- When you need optimal normalization for each variable
- When dealing with variables with vastly different ranges
- When you want perfect denormalization accuracy

### 3. Hybrid Normalization

```python
# Use individual normalization only for specific data types
use_individual_for = ['scalar', 'y_scalar']  # Only scalar variables
normalized_data = data_loader.normalize_data_hybrid(use_individual_for)

# Or for more data types
use_individual_for = ['scalar', 'y_scalar', 'pft_1d', 'y_pft_1d']
normalized_data = data_loader.normalize_data_hybrid(use_individual_for)
```

**When to use:**
- When you want to optimize specific variable types
- When you want to balance performance and memory usage
- When you want to gradually migrate from group to individual normalization

## Available Data Types for Individual Normalization

The following data types can be individually normalized:

- `'scalar'` - Input scalar variables (GPP, NPP, AR, HR)
- `'y_scalar'` - Output scalar variables (Y_GPP, Y_NPP, Y_AR, Y_HR)
- `'pft_1d'` - Input PFT1D variables (tlai, deadstemc)
- `'y_pft_1d'` - Output PFT1D variables (Y_tlai, Y_deadstemc)
- `'soil_2d'` - Input Soil2D variables (cwdc_vr, sminn_vr)
- `'y_soil_2d'` - Output Soil2D variables (Y_cwdc_vr, Y_sminn_vr)

## Complete Example

```python
import torch
from config.training_config import DataConfig, PreprocessingConfig
from data.data_loader_individual import DataLoaderIndividual

# 1. Create configurations
data_config = DataConfig(
    data_paths=["/path/to/your/data"],
    file_pattern="*.pkl",
    time_series_columns=["FLDS", "PSRF", "FSDS", "QBOT", "PRECTmms", "TBOT"],
    time_series_length=240,
    static_columns=["Latitude", "Longitude", "AREA"],
    x_list_scalar_columns=["GPP", "NPP", "AR", "HR"],
    y_list_scalar_columns=["Y_GPP", "Y_NPP", "Y_AR", "Y_HR"],
    x_list_columns_1d=["tlai", "deadstemc"],
    y_list_columns_1d=["Y_tlai", "Y_deadstemc"],
    x_list_columns_2d=["cwdc_vr", "sminn_vr"],
    y_list_columns_2d=["Y_cwdc_vr", "Y_sminn_vr"],
    pft_param_columns=["pft_deadwdcn", "pft_frootcn"],
    max_1d_length=17,
    max_2d_rows=360,
    max_2d_cols=10,
    random_state=42
)

preprocessing_config = PreprocessingConfig(
    time_series_normalization="minmax",
    static_normalization="minmax",
    list_1d_normalization="minmax",
    list_2d_normalization="minmax",
    data_type=torch.float32
)

# 2. Create data loader
data_loader = DataLoaderIndividual(data_config, preprocessing_config)

# 3. Load and preprocess data
data_loader.load_data()
data_loader.preprocess_data()

# 4. Choose normalization approach

# Option A: Group normalization (default)
print("Using group normalization...")
normalized_data = data_loader.normalize_data()

# Option B: Individual normalization
print("Using individual normalization...")
normalized_data = data_loader.normalize_data_individual()

# Option C: Hybrid normalization
print("Using hybrid normalization...")
use_individual_for = ['scalar', 'y_scalar']  # Only scalar variables
normalized_data = data_loader.normalize_data_hybrid(use_individual_for)

# 5. Access normalized data
scalar_data = normalized_data['scalar_data']
y_scalar_data = normalized_data['y_scalar']
pft_1d_data = normalized_data['variables_1d_pft']
soil_2d_data = normalized_data['variables_2d_soil']
scalers = normalized_data['scalers']

# 6. Save scalers for later use
data_loader.save_scalers("/path/to/save/scalers")

# 7. Load scalers in a new session
new_data_loader = DataLoaderIndividual(data_config, preprocessing_config)
new_data_loader.load_scalers("/path/to/save/scalers")
```

## Migration Strategy

### Phase 1: Start with Group Normalization
```python
# Use the default approach
normalized_data = data_loader.normalize_data()
```

### Phase 2: Test Individual Normalization
```python
# Test with individual normalization
normalized_data = data_loader.normalize_data_individual()
```

### Phase 3: Use Hybrid Approach
```python
# Start with individual normalization for scalar variables only
use_individual_for = ['scalar', 'y_scalar']
normalized_data = data_loader.normalize_data_hybrid(use_individual_for)

# Gradually add more data types
use_individual_for = ['scalar', 'y_scalar', 'pft_1d', 'y_pft_1d']
normalized_data = data_loader.normalize_data_hybrid(use_individual_for)
```

### Phase 4: Full Individual Normalization
```python
# Use individual normalization for all supported data types
normalized_data = data_loader.normalize_data_individual()
```

## Benefits of Each Approach

### Group Normalization
- ✅ **Fast**: Single scaler per data type
- ✅ **Memory efficient**: Minimal memory overhead
- ✅ **Simple**: Easy to understand and debug
- ❌ **Range compression**: Variables with different ranges may lose precision
- ❌ **Less accurate**: May not provide optimal normalization for each variable

### Individual Normalization
- ✅ **Optimal**: Each variable gets its own normalization
- ✅ **Accurate**: Perfect denormalization possible
- ✅ **Flexible**: Can handle variables with vastly different ranges
- ❌ **Memory overhead**: More scalers stored
- ❌ **Slower**: More complex normalization process

### Hybrid Normalization
- ✅ **Balanced**: Best of both worlds
- ✅ **Flexible**: Choose what to optimize
- ✅ **Gradual migration**: Easy to transition from group to individual
- ❌ **Complexity**: Need to understand which approach to use when

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Make sure you're in the project root directory
   from data.data_loader_individual import DataLoaderIndividual
   ```

2. **Configuration Errors**
   ```python
   # Ensure all required columns are specified in data_config
   assert len(data_config.x_list_scalar_columns) > 0
   ```

3. **Memory Issues**
   ```python
   # Use group normalization for memory-constrained environments
   normalized_data = data_loader.normalize_data()
   ```

4. **Shape Mismatches**
   ```python
   # Check data shapes after preprocessing
   print(f"DataFrame shape: {data_loader.df.shape}")
   ```

### Performance Tips

1. **Start small**: Test with a small dataset first
2. **Monitor memory**: Individual normalization uses more memory
3. **Choose wisely**: Use individual normalization only where it provides benefits
4. **Save scalers**: Always save scalers for reproducibility

## Next Steps

1. **Test the approaches**: Run the test script to see all approaches in action
2. **Choose your approach**: Decide which normalization method fits your needs
3. **Integrate into training**: Update your training script to use the chosen approach
4. **Monitor results**: Compare training performance between approaches
5. **Optimize**: Use hybrid approach to fine-tune normalization for specific variables

## Support

If you encounter issues:

1. Check the test scripts for working examples
2. Verify your configuration matches the expected format
3. Start with group normalization and gradually move to individual
4. Use the hybrid approach for selective optimization

The system is designed to be backward compatible, so you can always fall back to group normalization if needed.
