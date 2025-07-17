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

#### Input Variables (119 with water, 113 without water)
1. **Time Series**: 6 variables (FLDS, PSRF, FSDS, QBOT, PRECTmms, TBOT)
2. **Surface Properties**: 27 variables
   - Geographic: 7 variables
   - Soil Phosphorus Forms: 4 variables
   - PFT Coverage: 20 variables
   - Soil Texture: 20 variables (CLAY_1_ through CLAY_10_, SAND_1_ through SAND_10_)
3. **PFT Parameters**: 44 variables (plant functional type characteristics)
4. **Water Variables**: 6 variables (optional, H2OCAN, H2OSFC, H2OSNO, TH2OSFC, H2OSOI_LIQ, H2OSOI_ICE)
5. **Scalar Variables**: 5 variables (GPP, NPP, AR, HR, LAI)
6. **1D Variables**: 13 variables (CNP pools)
7. **2D Variables**: 28 variables (soil and litter properties, including soil variables)

#### Output Variables (45 with water, 39 without water)
1. **Water Variables**: 6 variables (optional, Y_H2OCAN, Y_H2OSFC, Y_H2OSNO, Y_TH2OSFC, Y_H2OSOI_LIQ, Y_H2OSOI_ICE)
2. **Scalar Variables**: 5 variables (Y_GPP, Y_NPP, Y_AR, Y_HR, Y_LAI)
3. **1D Variables**: 13 variables (Y_CNP pools)
4. **2D Variables**: 28 variables (Y_soil and Y_litter properties)
5. **Temperature Variables**: 7 variables (excluded for first experiments)

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

## Entry 5: Dataset Structure Discovery and Variable Reorganization
**Date**: 2025-07-09  
**Version**: 1.2  
**Status**: Major Restructuring

### Issues Discovered
1. **File Pattern Mismatch**: Configuration was looking for `1_training_data_batch_*.pkl` but files are named `dataset_part_*.pkl`
2. **Variable Name Mismatches**: Many variables had incorrect names compared to actual dataset
3. **Dataset Structure Misunderstanding**: Thought variables were 1D but they're actually 2D (list variables)

### Dataset Structure Analysis
**Actual Dataset Structure:**
- **Scalar Y_ variables (4)**: `Y_GPP`, `Y_NPP`, `Y_AR`, `Y_HR`
- **List Y_ variables (59)**: All other Y_ variables are lists (2D data)
- **No 1D Y_ variables**: All variables thought to be 1D are actually 2D

### Variable Name Corrections
- Geographic: `lat`, `lon`, `area` → `Latitude`, `Longitude`, `AREA`
- PFT: `PCT_NAT_PFT0` → `PCT_NAT_PFT_0` (with underscores)
- Soil texture: `CLAY_1_` → `PCT_CLAY_0` (different naming convention)
- 2D variables: `SOILC_1C_vr` → `soil1c_vr` (lowercase)

### Variable Reorganization (NEW LOGICAL GROUPING)
**Instead of combining all 2D variables, organized into logical groups:**

1. **2D_PFT_variables** (15 variables): Plant functional type related variables
   - `deadcrootc`, `deadcrootn`, `deadcrootp`, `deadstemc`, `deadstemn`, `deadstemp`
   - `frootc`, `frootc_storage`, `leafc`, `leafc_storage`, `totcolp`, `totlitc`, `totvegc`
   - `cwdp`, `tlai`

2. **2D_Soil_variables** (28 variables): Soil-related layered data
   - Soil carbon, nitrogen, phosphorus variables
   - Litter variables
   - Coarse woody debris
   - Soil mineral nitrogen

3. **2D_Other_variables** (16 variables): Additional layered data - **EXCLUDED FOR FIRST EXPERIMENTS**
   - Water, temperature, and other variables
   - These will be added back in future experiments

### Model Configuration Updates
- `scalar_output_size`: 4 (Y_GPP, Y_NPP, Y_AR, Y_HR)
- `vector_output_size`: 15 (2D PFT variables)
- `matrix_output_size`: 28 (2D soil variables only, 16 other variables excluded for first experiments)

### Benefits of New Organization
- **Better Interpretability**: Clear separation between PFT and soil variables
- **Logical Grouping**: Variables grouped by their physical/biological meaning
- **Easier Debugging**: Can isolate issues to specific variable types
- **Flexible Architecture**: Can process different variable types separately

---

## Entry 6: Key Design Decisions
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

## Entry 7: Dataset Variable Alignment Fix
**Date**: [Current Date]  
**Version**: 1.3  
**Status**: Critical Bug Fix

### Issue Identified
The CNP model configuration was using incorrect variable names that didn't match the actual dataset structure. This caused validation errors during training.

### Root Cause Analysis
1. **Variable Name Mismatch**: Configuration used theoretical variable names instead of actual dataset variables
2. **Count Discrepancies**: Variable counts didn't match between input and output
3. **Dataset Structure**: The actual dataset has different variable names than expected

### Actual Dataset Variables (Corrected)

#### Input 2D Variables (16 variables):
- **Coarse woody debris**: cwdc_vr, cwdn_vr, secondp_vr, cwdp_vr
- **Litter variables**: litr1c_vr, litr2c_vr, litr3c_vr, litr1n_vr, litr2n_vr, litr3n_vr, litr1p_vr, litr2p_vr, litr3p_vr
- **Soil mineral nitrogen**: sminn_vr, smin_no3_vr, smin_nh4_vr

#### Output 2D Variables (28 variables):
- **Soil variables**: Y_soil3c_vr, Y_soil4c_vr, Y_soil1c_vr, Y_soil1n_vr, Y_soil1p_vr, Y_soil2c_vr, Y_soil2n_vr, Y_soil2p_vr, Y_soil3n_vr, Y_soil3p_vr, Y_soil4n_vr, Y_soil4p_vr
- **Coarse woody debris**: Y_cwdc_vr, Y_cwdn_vr, Y_secondp_vr, Y_cwdp_vr
- **Litter variables**: Y_litr1c_vr, Y_litr2c_vr, Y_litr3c_vr, Y_litr1n_vr, Y_litr2n_vr, Y_litr3n_vr, Y_litr1p_vr, Y_litr2p_vr, Y_litr3p_vr
- **Soil mineral nitrogen**: Y_sminn_vr, Y_smin_no3_vr, Y_smin_nh4_vr

#### Water Variables (6 variables):
- **Input**: H2OCAN, H2OSFC, H2OSNO, TH2OSFC, H2OSOI_LIQ, H2OSOI_ICE
- **Output**: Y_H2OCAN, Y_H2OSFC, Y_H2OSNO, Y_TH2OSFC, Y_H2OSOI_LIQ, Y_H2OSOI_ICE

#### 1D Variables (13 variables):
- **Input**: deadcrootc, deadcrootn, deadcrootp, deadstemc, deadstemn, deadstemp, frootc, frootc_storage, leafc, leafc_storage, totcolp, totlitc, totvegc
- **Output**: Y_deadcrootc, Y_deadcrootn, Y_deadcrootp, Y_deadstemc, Y_deadstemn, Y_deadstemp, Y_frootc, Y_frootc_storage, Y_leafc, Y_leafc_storage, Y_totcolp, Y_totlitc, Y_totvegc

### Fixes Applied
1. **Updated Input 2D Variables**: Changed from 67 theoretical variables to 16 actual dataset variables
2. **Updated Output 2D Variables**: Changed from 28 theoretical variables to 28 actual dataset variables
3. **Updated Water Variables**: Changed from H2OSOI_1_ through H2OSOI_6_ to actual dataset water variables
4. **Updated 1D Variables**: Removed cwdp from input (moved to 2D) and adjusted counts
5. **Updated Documentation**: Corrected all variable counts and names in README and development journal

### Updated Variable Counts
- **Input Variables**: 107 with water, 101 without water (down from 159/153)
- **Output Variables**: 52 with water, 46 without water (down from 56/50)
- **2D Input**: 16 variables (down from 67)
- **2D Output**: 28 variables (unchanged)

### Files Modified
1. `config/training_config.py`
   - Updated all variable lists to match actual dataset
   - Corrected variable counts in model configuration
   - Updated comments and documentation

2. `CNP_MODEL_README.md`
   - Updated variable names and counts
   - Corrected architecture documentation
   - Updated data structure section

3. `CNP_MODEL_DEVELOPMENT_JOURNAL.md`
   - Added this entry documenting the fix
   - Updated variable counts in previous entries

### Impact
- **Positive**: Model now uses correct dataset variables
- **Positive**: Eliminates validation errors during training
- **Positive**: Improved model accuracy by using actual data structure
- **Consideration**: Reduced input variable count may affect model capacity

---

## Entry 8: Soil Variables Reorganization
**Date**: [Current Date]  
**Version**: 1.4  
**Status**: Architecture Correction

### Issue Identified
Soil variables (SOILC_1C_vr, SOILC_2C_vr, etc.) were incorrectly placed in the Surface Properties group instead of the 2D Variables group where they belong.

### Root Cause Analysis
1. **Incorrect Grouping**: Soil variables were placed in Surface Properties FC layer instead of 2D CNN layer
2. **Logical Inconsistency**: Soil variables are 2D layered data and should be processed by CNN
3. **Architecture Mismatch**: Surface Properties FC is designed for static geographic and parameter data, not 2D soil data

### Correction Applied
1. **Moved Soil Variables**: Transferred 12 soil variables from Surface Properties to 2D Variables
   - SOILC_1C_vr, SOILC_2C_vr, SOILC_3C_vr, SOILC_4C_vr
   - SOILN_1N_vr, SOILN_2N_vr, SOILN_3N_vr, SOILN_4N_vr
   - SOILP_1P_vr, SOILP_2P_vr, SOILP_3P_vr, SOILP_4P_vr

2. **Updated Variable Counts**:
   - **Surface Properties**: 39 → 27 variables (removed 12 soil variables)
   - **2D Variables**: 16 → 28 variables (added 12 soil variables)
   - **Total Input**: 107 → 119 variables with water, 101 → 113 without water

3. **Architecture Alignment**:
   - **Surface Properties FC**: Now processes only static geographic and parameter data
   - **2D CNN**: Now processes all 2D layered data including soil variables
   - **Perfect Input-Output Match**: 28 input 2D variables → 28 output 2D variables

### Benefits of Correction
1. **Logical Consistency**: Soil variables are now processed by the appropriate CNN layer
2. **Better Feature Learning**: CNN can capture spatial relationships in soil layers
3. **Perfect Alignment**: Input and output 2D variables now have matching counts (28 each)
4. **Improved Architecture**: Each processing component now handles appropriate data types

### Files Modified
1. `config/training_config.py`
   - Moved soil variables from surface_properties to variables_2d
   - Updated variable counts and comments
   - Updated model configuration parameters

2. `CNP_MODEL_README.md`
   - Updated architecture documentation
   - Corrected variable counts and descriptions
   - Updated data structure section

3. `CNP_MODEL_DEVELOPMENT_JOURNAL.md`
   - Added this entry documenting the correction
   - Updated variable counts in previous entries

### Impact
- **Positive**: Improved logical consistency in variable grouping
- **Positive**: Better alignment between input and output 2D variables
- **Positive**: More appropriate processing of soil data by CNN
- **Positive**: Enhanced model architecture and feature learning

---

## Entry 9: Temperature Variables Exclusion for First Experiments
**Date**: [Current Date]  
**Version**: 1.5  
**Status**: Experimental Simplification

### Rationale for Temperature Exclusion
For the first experiments, temperature variables are excluded to:
1. **Simplify the model**: Focus on core CNP cycle variables
2. **Reduce complexity**: Fewer output variables to predict
3. **Faster training**: Smaller model and faster convergence
4. **Clearer analysis**: Easier to interpret results without temperature dependencies

### Temperature Variables Excluded
**Output Temperature Variables (7 variables)**:
- Y_T_GRND_R, Y_T_GRND_U, Y_T_LAKE, Y_T_SOISNO
- Y_T_GRND_1_, Y_T_GRND_2_, Y_T_GRND_3_

### Updated Variable Counts
- **Input Variables**: 119 with water, 113 without water (unchanged)
- **Output Variables**: 45 with water, 39 without water (reduced from 52/46)
- **Total Variables**: 164 with water, 158 without water (reduced from 171/165)

### Model Architecture Impact
- **Output Heads**: Reduced from 4 to 3 heads (water, scalar, 1D, 2D)
- **Training Complexity**: Simplified with fewer output variables
- **Loss Functions**: Reduced number of loss components
- **Validation**: Focus on CNP cycle variables only

### Files Modified
1. `config/training_config.py`
   - Commented out temperature output variables
   - Added note about temperature exclusion

2. `CNP_MODEL_README.md`
   - Updated output variable counts
   - Added note about temperature exclusion

3. `CNP_IO_list1.txt`
   - Updated output variable counts
   - Marked temperature variables as excluded
   - Updated final counts

4. `CNP_MODEL_DEVELOPMENT_JOURNAL.md`
   - Added this entry documenting the exclusion
   - Updated variable counts in previous entries

### Future Considerations
- **Phase 2**: Add temperature variables back for comprehensive modeling
- **Validation**: Compare performance with and without temperature variables
- **Interpretation**: Analyze temperature-CNP interactions in future experiments

---

## Entry 10: Lessons Learned
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