#!/usr/bin/env python3
"""
Test if the individual scalers from the new run can actually denormalize data.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('..')

run_dir = Path('../cnp_results/run_20250818_125330')
scalers_dir = run_dir / 'cnp_predictions' / 'scalers'

def test_individual_denorm():
    """Test if individual scalers can denormalize data."""
    
    # Load individual scalers
    try:
        with open(scalers_dir / 'individual_y_pft_1d_scaler.pkl', 'rb') as f:
            pft_scaler = pickle.load(f)
        print("✅ Loaded individual_y_pft_1d_scaler.pkl")
    except Exception as e:
        print(f"❌ Failed to load PFT scaler: {e}")
        return
    
    try:
        with open(scalers_dir / 'individual_y_soil_2d_scaler.pkl', 'rb') as f:
            soil_scaler = pickle.load(f)
        print("✅ Loaded individual_y_soil_2d_scaler.pkl")
    except Exception as e:
        print(f"❌ Failed to load soil scaler: {e}")
        return
    
    # Load some predictions to test
    try:
        pft_pred = pd.read_csv(run_dir / 'cnp_predictions' / 'pft_1d_predictions' / 'predictions_Y_tlai.csv')
        print(f"✅ Loaded PFT predictions: shape={pft_pred.shape}")
        
        # Take a few non-zero rows
        non_zero_rows = pft_pred[(pft_pred != 0).any(axis=1)]
        if len(non_zero_rows) > 0:
            test_data = non_zero_rows.iloc[0:2].values
            print(f"✅ Found {len(non_zero_rows)} non-zero rows, testing with first 2")
            print(f"   Test data shape: {test_data.shape}")
            print(f"   Sample values: {test_data[0, :5]}")  # First 5 values of first row
        else:
            print("❌ No non-zero rows found in PFT predictions")
            return
    except Exception as e:
        print(f"❌ Failed to load PFT predictions: {e}")
        return
    
    # Test PFT 1D denormalization
    print("\n--- Testing PFT 1D Denormalization ---")
    try:
        # The data should be in shape (samples, features) where features are the PFTs
        # We need to reshape to (samples, pfts, variables) for the individual scaler
        test_data_reshaped = test_data.reshape(test_data.shape[0], test_data.shape[1], 1)
        print(f"   Reshaped data shape: {test_data_reshaped.shape}")
        
        # Try to denormalize using the individual scaler
        # The method signature is: inverse_transform_pft_1d(data, pft_names, variable_names)
        pft_names = [f'PFT{i}' for i in range(16)]
        variable_names = ['Y_tlai']
        denorm_data = pft_scaler.inverse_transform_pft_1d(test_data_reshaped, pft_names, variable_names)
        print(f"✅ Denormalization successful!")
        print(f"   Original shape: {test_data.shape}")
        print(f"   Denormalized shape: {denorm_data.shape}")
        print(f"   Original values: {test_data[0, :5]}")
        print(f"   Denormalized values: {denorm_data[0, :5]}")
        
        # Reshape denorm_data back to original shape for comparison
        denorm_data_flat = denorm_data.reshape(denorm_data.shape[0], -1)
        print(f"   Denormalized data flattened shape: {denorm_data_flat.shape}")
        
        # Check if values changed significantly
        if np.allclose(test_data, denorm_data_flat, atol=1e-6):
            print("⚠️  WARNING: Denormalized values are very close to original - possible identity transformation")
        else:
            print("✅ Denormalization appears to be working - values changed significantly")
            
    except Exception as e:
        print(f"❌ PFT 1D denormalization failed: {e}")
        print(f"   Error type: {type(e).__name__}")
    
    # Test Soil 2D denormalization
    print("\n--- Testing Soil 2D Denormalization ---")
    try:
        soil_pred = pd.read_csv(run_dir / 'cnp_predictions' / 'soil_2d_predictions' / 'predictions_Y_cwdc_vr.csv')
        print(f"✅ Loaded soil predictions: shape={soil_pred.shape}")
        
        # Take a few non-zero rows
        non_zero_rows = soil_pred[(soil_pred != 0).any(axis=1)]
        if len(non_zero_rows) > 0:
            test_data = non_zero_rows.iloc[0:2].values
            print(f"✅ Found {len(non_zero_rows)} non-zero rows, testing with first 2")
            print(f"   Original shape: {test_data.shape}")
            print(f"   Sample values: {test_data[0, :5]}")  # First 5 values of first row
        else:
            print("❌ No non-zero rows found in soil predictions")
            return
    except Exception as e:
        print(f"❌ Failed to load soil predictions: {e}")
        return
    
    try:
        # The data should be in shape (samples, features) where features are (columns, layers)
        # We need to reshape to (samples, columns, layers) for the individual scaler
        # The data has 180 features, which should be 18 columns × 10 layers
        test_data_reshaped = test_data.reshape(test_data.shape[0], 18, 10)  # 18 columns, 10 layers
        print(f"   Reshaped data shape: {test_data_reshaped.shape}")
        
        # Try to denormalize using the individual scaler
        # The method signature is: inverse_transform_soil_2d(data, variable_names, num_layers)
        # But the data needs to be in shape (samples, variables, columns, layers)
        # So we need to add a variable dimension
        test_data_final = test_data_reshaped.reshape(test_data.shape[0], 1, 18, 10)  # (samples, variables, columns, layers)
        print(f"   Final reshaped data shape: {test_data_final.shape}")
        
        variable_names = ['Y_cwdc_vr']
        denorm_data = soil_scaler.inverse_transform_soil_2d(test_data_final, variable_names, 10)
        print(f"✅ Denormalization successful!")
        print(f"   Original shape: {test_data.shape}")
        print(f"   Denormalized shape: {denorm_data.shape}")
        print(f"   Original values: {test_data[0, :5]}")
        print(f"   Denormalized values: {denorm_data[0, :5]}")
        
        # Reshape denorm_data back to original shape for comparison
        denorm_data_flat = denorm_data.reshape(denorm_data.shape[0], -1)
        print(f"   Denormalized data flattened shape: {denorm_data_flat.shape}")
        
        # Check if values changed significantly
        if np.allclose(test_data, denorm_data_flat, atol=1e-6):
            print("⚠️  WARNING: Denormalized values are very close to original - possible identity transformation")
        else:
            print("✅ Denormalization appears to be working - values changed significantly")
            
    except Exception as e:
        print(f"❌ Soil 2D denormalization failed: {e}")
        print(f"   Error type: {type(e).__name__}")

if __name__ == "__main__":
    test_individual_denorm()
