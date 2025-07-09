# CNP Model Development Journal

## Overview
This journal documents the development and evolution of the CNP (Carbon-Nitrogen-Phosphorus) model architecture based on the CNP_IO_list1.txt structure. It tracks key decisions, changes, and rationale throughout the development process.

---

## Entry 1: Initial Model Architecture Design
**Date**: [Current Date]  
**Version**: 1.0  
**Status**: Initial Design

### Objective
Create a specialized AI model for CNP cycle prediction based on the CNP_IO_list1.txt structure with the following architecture:
- LSTM for time series processing
- Multiple FC layers for different variable groups
- CNN for 2D soil/litter data
- Transformer encoder for feature fusion
- Multi-task perceptrons for separate predictions

### Architecture Components
1. **LSTM** for 6 time-series variables (20 years)
2. **FC** for surface properties (geographic, soil texture, P forms, PFT coverage)
3. **FC** for 44 PFT characteristics parameters
4. **FC** for water variables (optional)
5. **FC** for scalar variables
6. **FC** for 1D variables
7. **CNN** for 2D variables
8. **Transformer Encoder** for feature fusion
9. **Multi-task Perceptrons** for separate predictions

### Initial Variable Grouping
- **Surface Properties**: 19 variables (Geographic: 7, Soil Phosphorus Forms: 4, PFT Coverage: 20)
- **2D Variables**: 87 variables (including soil texture)
- **PFT Parameters**: 44 variables (separate group)
- **Water Variables**: 6 variables (optional)

### Files Created
- `config/training_config.py` - Added CNP configuration functions
- `models/cnp_combined_model.py` - New CNP model implementation
- `train_cnp_model.py` - Dedicated CNP training script
- `model_training_refactored.py` - Updated to support CNP configurations
- `CNP_MODEL_README.md` - Comprehensive documentation

---

## Entry 2: Variable Reorganization - Soil Texture Migration
**Date**: [Current Date]  
**Version**: 1.1  
**Status**: Variable Reorganization

### Change Request
Move soil texture variables (CLAY_1_ through CLAY_10_ and SAND_1_ through SAND_10_) from the 2D variables group to the surface properties group.

### Rationale
Soil texture is a fundamental surface property that affects soil processes and should be grouped with other surface characteristics rather than with layered soil data.

### Changes Made

#### 1. Surface Properties Group
**Before**: 19 variables
- Geographic: 7 variables
- Soil Phosphorus Forms: 4 variables  
- PFT Coverage: 20 variables

**After**: 39 variables
- Geographic: 7 variables
- Soil Phosphorus Forms: 4 variables
- PFT Coverage: 20 variables
- **Soil Texture: 20 variables** (CLAY_1_ through CLAY_10_, SAND_1_ through SAND_10_)

#### 2. 2D Variables Group
**Before**: 87 variables
- Litter Variables: 16 variables
- Soil Properties: 88 variables (including soil texture)

**After**: 67 variables
- Litter Variables: 16 variables
- Soil Properties: 68 variables (soil texture removed)

#### 3. Model Architecture Updates
- **Surface Properties FC**: Increased from 64 → 128 units
- **CNN**: Now processes 67 variables instead of 87

### Files Modified
1. `config/training_config.py`
   - Updated `surface_properties` list to include soil texture variables
   - Removed soil texture from `variables_2d` list
   - Updated variable counts in comments
   - Increased `static_fc_size` from 64 to 128

2. `CNP_MODEL_README.md`
   - Updated architecture documentation
   - Updated variable counts and descriptions
   - Updated input variable organization

### Impact Analysis
- **Positive**: Better logical grouping of variables
- **Positive**: Improved model interpretability
- **Neutral**: No change in total variable count
- **Consideration**: Slightly larger surface properties processing layer

---

## Entry 3: Configuration Management
**Date**: [Current Date]  
**Version**: 1.1  
**Status**: Configuration Finalized

### Configuration Options
The model supports two main configurations:

1. **With Water Variables** (`get_cnp_model_config(include_water=True)`)
   - Includes H2OSOI variables in both input and output
   - Total input variables: 159
   - Total output variables: 62

2. **Without Water Variables** (`get_cnp_model_config(include_water=False)`)
   - Excludes H2OSOI variables
   - Total input variables: 153
   - Total output variables: 56

### Usage Commands
```bash
# With water variables
python train_cnp_model.py --include-water
python model_training_refactored.py cnp

# Without water variables  
python train_cnp_model.py --no-water
python model_training_refactored.py cnp_no_water
```

---

## Entry 4: Final Architecture Summary
**Date**: [Current Date]  
**Version**: 1.1  
**Status**: Finalized

### Complete Variable Distribution

#### Input Variables (159 with water, 153 without water)
1. **Time Series**: 6 variables (FLDS, PSRF, FSDS, QBOT, PRECTmms, TBOT)
2. **Surface Properties**: 39 variables
   - Geographic: 7 variables
   - Soil Phosphorus Forms: 4 variables
   - PFT Coverage: 20 variables
   - Soil Texture: 20 variables (CLAY_1_ through CLAY_10_, SAND_1_ through SAND_10_)
3. **PFT Parameters**: 44 variables (plant functional type characteristics)
4. **Water Variables**: 6 variables (optional, H2OSOI layers)
5. **Scalar Variables**: 5 variables (GPP, NPP, AR, HR, LAI)
6. **1D Variables**: 16 variables (CNP pools)
7. **2D Variables**: 67 variables (soil and litter properties)

#### Output Variables (62 with water, 56 without water)
1. **Water Variables**: 6 variables (optional, Y_H2OSOI layers)
2. **Scalar Variables**: 5 variables (Y_GPP, Y_NPP, Y_AR, Y_HR, Y_LAI)
3. **1D Variables**: 16 variables (Y_CNP pools)
4. **2D Variables**: 28 variables (Y_soil and Y_litter properties)

### Model Architecture
- **LSTM**: 2-layer, 128 hidden units for time series
- **Surface Properties FC**: 2-layer, 128 → 64 units
- **PFT Parameters FC**: 2-layer, 128 → 64 units
- **Water Variables FC**: 2-layer, 32 → 16 units (optional)
- **Scalar Variables FC**: 2-layer, 32 → 16 units
- **1D Variables FC**: 2-layer, 64 → 32 units
- **CNN**: 4-layer with [32, 64, 128, 256] channels for 2D data
- **Transformer**: 4-layer encoder with 8 heads, 128 token dimension
- **Output Heads**: Separate perceptrons for each output type

### Training Configuration
- **Optimizer**: AdamW with weight decay 0.01
- **Learning Rate**: 0.001 with cosine annealing scheduler
- **Batch Size**: 32
- **Epochs**: 100 with early stopping (patience=15)
- **Loss Weights**: Equal weighting for all output types
- **Mixed Precision**: Enabled for GPU optimization

---

## Entry 5: Key Design Decisions
**Date**: [Current Date]  
**Version**: 1.1  
**Status**: Documented

### 1. Variable Grouping Strategy
- **Surface Properties**: Geographic, soil phosphorus forms, PFT coverage, soil texture
- **PFT Parameters**: Separate group for plant functional type characteristics
- **2D Variables**: Only layered soil and litter data (soil texture moved to surface properties)

### 2. Water Variables Flexibility
- Optional inclusion/exclusion of water variables
- Allows for different modeling scenarios
- Maintains consistent architecture with/without water

### 3. Multi-task Learning Approach
- Separate output heads for different variable types
- Allows for different loss weights and optimization strategies
- Enables focused training on specific output types

### 4. Transformer-based Feature Fusion
- Combines features from all input processing components
- Enables cross-modal learning between different variable types
- Provides attention mechanism for feature importance

### 5. CNN for 2D Data
- Handles layered soil and litter data effectively
- Preserves spatial relationships in soil layers
- Reduces dimensionality while maintaining important features

---

## Entry 6: Future Considerations
**Date**: [Current Date]  
**Version**: 1.1  
**Status**: Planning

### Potential Improvements
1. **Dynamic Variable Selection**: Allow runtime selection of variable groups
2. **Adaptive Architecture**: Automatically adjust model size based on available data
3. **Ensemble Methods**: Combine multiple CNP models for improved predictions
4. **Interpretability**: Add attention visualization and feature importance analysis
5. **Transfer Learning**: Pre-train on larger datasets and fine-tune for specific regions

### Monitoring and Validation
1. **Performance Metrics**: Track separate metrics for each output type
2. **Model Interpretability**: Analyze attention patterns and feature importance
3. **Physical Consistency**: Validate predictions against known biogeochemical relationships
4. **Uncertainty Quantification**: Add uncertainty estimates to predictions

### Documentation and Reproducibility
1. **Experiment Tracking**: Log all training runs and hyperparameters
2. **Model Versioning**: Version control for model configurations
3. **Data Versioning**: Track dataset versions and preprocessing steps
4. **Reproducible Training**: Ensure deterministic training with fixed seeds

---

## Entry 7: Lessons Learned
**Date**: [Current Date]  
**Version**: 1.1  
**Status**: Reflection

### Technical Insights
1. **Variable Grouping**: Logical grouping improves model interpretability and performance
2. **Modular Architecture**: Separate processing components enable flexible experimentation
3. **Configuration Management**: Centralized configuration simplifies model management
4. **Documentation**: Comprehensive documentation is essential for complex models

### Process Insights
1. **Iterative Development**: Architecture evolved through multiple iterations
2. **User Feedback**: User requirements drove important architectural decisions
3. **Flexibility**: Built-in options (like water variable inclusion) increase model utility
4. **Maintainability**: Clean code organization and documentation support long-term maintenance

### Scientific Insights
1. **Physical Consistency**: Variable grouping should reflect underlying physical processes
2. **Multi-scale Modeling**: Different variable types require different processing approaches
3. **Uncertainty**: Complex models need robust uncertainty quantification
4. **Validation**: Multiple validation approaches are needed for complex biogeochemical models

---

*This journal will be updated as the CNP model continues to evolve and improve.* 