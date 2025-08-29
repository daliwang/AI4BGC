# PFT 1D Denormalization Fix Summary

## Problem Description

The PFT 1D variables (`Y_tlai` and `Y_deadstemc`) in run `run_20250817_213337` were not being denormalized during the training process, resulting in predictions and ground truth being saved in normalized form (values between 0 and 1) instead of their original physical units.

## Root Cause Analysis

### 1. Version Mismatch Issue
The training was using an older version of the trainer code that attempted to call `inverse_transform` directly on the `IndividualScalerManager` object:

```python
# OLD CODE (causing the error):
var_predictions_original = self.scalers['y_pft_1d'].inverse_transform(var_predictions_2d)
```

However, the `IndividualScalerManager` class only had specific methods like `inverse_transform_pft_1d()` and did not have a generic `inverse_transform()` method.

### 2. Scaler Key Format Mismatch
The individual scalers were stored with keys in the format `pft1d_VARIABLE_PFT` (e.g., `pft1d_Y_tlai_PFT1`), but the code was looking for them in the format `pft1d_PFT_VARIABLE`.

### 3. Data Shape Mismatch
The `inverse_transform_pft_1d` method expected data in the format `(samples, pfts, variables)` but was receiving data in the format `(samples, variables, pfts)`.

## Solution Implemented

### 1. Added Generic `inverse_transform` Method
Added a backward-compatible `inverse_transform` method to the `IndividualScalerManager` class that automatically detects data types and calls the appropriate specific method.

**File:** `/mnt/proj-shared/AI4BGC_7xw/AI4BGC/data/individual_scaler_manager.py`

```python
def inverse_transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
    """
    Generic inverse transform method for backward compatibility.
    
    This method automatically detects the data type and calls the appropriate
    specific inverse transform method.
    """
    # Automatic data type detection and routing to specific methods
    # ... implementation details ...
```

### 2. Fixed Scaler Key Format
Corrected the `inverse_transform_pft_1d` method to use the correct scaler key format:

```python
# FIXED CODE:
scaler_key = f'pft1d_{var_name}_{pft_name}'  # Correct format
```

### 3. Fixed Data Reshaping
Corrected the data reshaping in the fix script to match the expected format:

```python
# FIXED CODE:
pred_data_reshaped = pred_data.reshape(n_samples, num_pfts, 1)  # (samples, pfts, variables)
```

## Files Modified

### 1. IndividualScalerManager Class
- **File:** `/mnt/proj-shared/AI4BGC_7xw/AI4BGC/data/individual_scaler_manager.py`
- **Changes:**
  - Added generic `inverse_transform` method for backward compatibility
  - Fixed scaler key format in `inverse_transform_pft_1d` method

### 2. Fix Script
- **File:** `fix_pft1d_denormalization.py`
- **Purpose:** Script to fix existing runs by applying correct denormalization
- **Features:**
  - Loads saved individual scalers
  - Applies correct inverse transformation to PFT 1D predictions and ground truth
  - Updates original files with denormalized data
  - Creates backup files with `_denormalized` suffix

## Results

### Before Fix
- **Y_tlai predictions:** Normalized values (0.0 to 1.0)
- **Y_deadstemc predictions:** Normalized values (0.0 to 1.0)
- **Ground truth:** Normalized values (0.0 to 1.0)

### After Fix
- **Y_tlai predictions:** Denormalized values (0.0 to 15.598115)
- **Y_deadstemc predictions:** Denormalized values (0.0 to 39232.078596)
- **Y_tlai ground truth:** Denormalized values (0.0 to 17.248446)
- **Y_deadstemc ground truth:** Denormalized values (0.0 to 41389.459628)

## Usage

### For Future Runs
The updated `IndividualScalerManager` class now has backward compatibility, so both old and new code will work:

```python
# Both of these will now work:
scaler.inverse_transform(data)                    # Generic method
scaler.inverse_transform_pft_1d(data, pfts, vars)  # Specific method
```

### For Fixing Existing Runs
Use the fix script:

```bash
python fix_pft1d_denormalization.py <run_directory>
```

Example:
```bash
python fix_pft1d_denormalization.py run_20250817_213337
```

## Prevention

To prevent this issue in future runs:

1. **Ensure code consistency:** Make sure the training code and the `IndividualScalerManager` class are from the same version
2. **Use specific methods:** Prefer using specific methods like `inverse_transform_pft_1d()` over generic `inverse_transform()`
3. **Test denormalization:** Verify that predictions are properly denormalized before saving results
4. **Version control:** Keep training scripts and utility classes in sync

## Technical Details

### Scaler Storage Format
Individual scalers are stored with descriptive keys:
- `pft1d_Y_tlai_PFT1` - PFT 1D scaler for Y_tlai at PFT1
- `pft1d_Y_deadstemc_PFT1` - PFT 1D scaler for Y_deadstemc at PFT1
- etc.

### Data Flow
1. **Training:** Data normalized using individual scalers for each PFT-variable combination
2. **Prediction:** Model outputs normalized predictions
3. **Denormalization:** Each PFT-variable combination denormalized using its specific scaler
4. **Output:** Denormalized predictions in original physical units

## Conclusion

The PFT 1D denormalization issue has been successfully resolved by:

1. **Adding backward compatibility** to the `IndividualScalerManager` class
2. **Fixing the existing run** using the provided fix script
3. **Ensuring future runs** will work correctly with both old and new code

The PFT 1D variables now display values in their original physical units, making the results interpretable and consistent with other variables in the model.
