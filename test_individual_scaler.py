#!/usr/bin/env python3
"""
Test script for IndividualScalerManager

This script tests the basic functionality of the IndividualScalerManager
to ensure it works correctly before integration.
"""

import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add the data directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data'))

from individual_scaler_manager import IndividualScalerManager


def test_scalar_normalization():
    """Test individual scalar variable normalization."""
    print("Testing scalar normalization...")
    
    # Create sample data with different ranges
    np.random.seed(42)
    n_samples = 1000
    
    # GPP: 0-20 gC/m²/day
    gpp = np.random.uniform(0, 20, n_samples)
    # NPP: 0-15 gC/m²/day  
    npp = np.random.uniform(0, 15, n_samples)
    # AR: 0-10 gC/m²/day
    ar = np.random.uniform(0, 10, n_samples)
    # HR: 0-12 gC/m²/day
    hr = np.random.uniform(0, 12, n_samples)
    
    # Combine into array
    data = np.column_stack([gpp, npp, ar, hr])
    variable_names = ['GPP', 'NPP', 'AR', 'HR']
    
    print(f"Original data shape: {data.shape}")
    print(f"Original ranges:")
    for i, name in enumerate(variable_names):
        print(f"  {name}: [{data[:, i].min():.4f}, {data[:, i].max():.4f}]")
    
    # Create scaler manager
    scaler_manager = IndividualScalerManager(normalization_type='minmax')
    
    # Normalize data
    normalized_data = scaler_manager.fit_transform_scalar(data, variable_names)
    
    print(f"\nNormalized data shape: {normalized_data.shape}")
    print(f"Normalized ranges:")
    for i, name in enumerate(variable_names):
        print(f"  {name}: [{normalized_data[:, i].min():.4f}, {normalized_data[:, i].max():.4f}]")
    
    # Inverse transform
    denormalized_data = scaler_manager.inverse_transform_scalar(normalized_data, variable_names)
    
    print(f"\nDenormalized data shape: {denormalized_data.shape}")
    print(f"Denormalized ranges:")
    for i, name in enumerate(variable_names):
        print(f"  {name}: [{denormalized_data[:, i].min():.4f}, {denormalized_data[:, i].max():.4f}]")
    
    # Check accuracy
    mse = np.mean((data - denormalized_data) ** 2)
    print(f"\nReconstruction MSE: {mse:.2e}")
    
    if mse < 1e-10:
        print("✅ Scalar normalization test PASSED")
        return True
    else:
        print("❌ Scalar normalization test FAILED")
        return False


def test_pft_1d_normalization():
    """Test PFT1D variable normalization."""
    print("\nTesting PFT1D normalization...")
    
    # Create sample PFT1D data
    np.random.seed(42)
    n_samples = 1000
    n_pfts = 17
    n_variables = 2
    
    # tlai: 0-10 m²/m²
    tlai = np.random.uniform(0, 10, (n_samples, n_pfts, 1))
    # deadstemc: 0-1000 gC/m²
    deadstemc = np.random.uniform(0, 1000, (n_samples, n_pfts, 1))
    
    # Combine into array
    data = np.concatenate([tlai, deadstemc], axis=2)
    pft_names = [f'PFT{i}' for i in range(n_pfts)]
    variable_names = ['tlai', 'deadstemc']
    
    print(f"Original PFT1D data shape: {data.shape}")
    print(f"Original ranges:")
    for i, name in enumerate(variable_names):
        print(f"  {name}: [{data[:, :, i].min():.4f}, {data[:, :, i].max():.4f}]")
    
    # Create scaler manager
    scaler_manager = IndividualScalerManager(normalization_type='minmax')
    
    # Normalize data
    normalized_data = scaler_manager.fit_transform_pft_1d(data, pft_names, variable_names)
    
    print(f"Normalized PFT1D data shape: {normalized_data.shape}")
    print(f"Normalized ranges:")
    for i, name in enumerate(variable_names):
        print(f"  {name}: [{normalized_data[:, :, i].min():.4f}, {normalized_data[:, :, i].max():.4f}]")
    
    # Inverse transform
    denormalized_data = scaler_manager.inverse_transform_pft_1d(normalized_data, pft_names, variable_names)
    
    print(f"Denormalized PFT1D data shape: {denormalized_data.shape}")
    print(f"Denormalized ranges:")
    for i, name in enumerate(variable_names):
        print(f"  {name}: [{denormalized_data[:, :, i].min():.4f}, {denormalized_data[:, :, i].max():.4f}]")
    
    # Check accuracy
    mse = np.mean((data - denormalized_data) ** 2)
    print(f"Reconstruction MSE: {mse:.2e}")
    
    if mse < 1e-10:
        print("✅ PFT1D normalization test PASSED")
        return True
    else:
        print("❌ PFT1D normalization test FAILED")
        return False


def test_soil_2d_normalization():
    """Test Soil2D variable normalization."""
    print("\nTesting Soil2D normalization...")
    
    # Create sample Soil2D data
    np.random.seed(42)
    n_samples = 1000
    n_variables = 2
    n_columns = 360  # Grid columns
    n_layers = 10    # Soil layers
    
    # cwdc_vr: 0-1000 gC/m²
    cwdc_vr = np.random.uniform(0, 1000, (n_samples, 1, n_columns, n_layers))
    # sminn_vr: 0-50 gN/m²
    sminn_vr = np.random.uniform(0, 50, (n_samples, 1, n_columns, n_layers))
    
    # Combine into array
    data = np.concatenate([cwdc_vr, sminn_vr], axis=1)
    variable_names = ['cwdc_vr', 'sminn_vr']
    
    print(f"Original Soil2D data shape: {data.shape}")
    print(f"Original ranges:")
    for i, name in enumerate(variable_names):
        print(f"  {name}: [{data[:, i, :, :].min():.4f}, {data[:, i, :, :].max():.4f}]")
    
    # Create scaler manager
    scaler_manager = IndividualScalerManager(normalization_type='minmax')
    
    # Normalize data
    normalized_data = scaler_manager.fit_transform_soil_2d(data, variable_names, n_layers)
    
    print(f"Normalized Soil2D data shape: {normalized_data.shape}")
    print(f"Normalized ranges:")
    for i, name in enumerate(variable_names):
        print(f"  {name}: [{normalized_data[:, i, :, :].min():.4f}, {normalized_data[:, i, :, :].max():.4f}]")
    
    # Inverse transform
    denormalized_data = scaler_manager.inverse_transform_soil_2d(normalized_data, variable_names, n_layers)
    
    print(f"Denormalized Soil2D data shape: {denormalized_data.shape}")
    print(f"Denormalized ranges:")
    for i, name in enumerate(variable_names):
        print(f"  {name}: [{denormalized_data[:, i, :, :].min():.4f}, {denormalized_data[:, i, :, :].max():.4f}]")
    
    # Check accuracy
    mse = np.mean((data - denormalized_data) ** 2)
    print(f"Reconstruction MSE: {mse:.2e}")
    
    if mse < 1e-10:
        print("✅ Soil2D normalization test PASSED")
        return True
    else:
        print("❌ Soil2D normalization test FAILED")
        return False


def test_scaler_persistence():
    """Test saving and loading scalers."""
    print("\nTesting scaler persistence...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    data = np.random.uniform(0, 100, (n_samples, 3))
    variable_names = ['var1', 'var2', 'var3']
    
    # Create scaler manager and fit
    scaler_manager = IndividualScalerManager()
    normalized_data = scaler_manager.fit_transform_scalar(data, variable_names)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save scalers
        scaler_manager.save_scalers(temp_path)
        
        # Check if files were created
        scaler_files = list(temp_path.glob("*.pkl"))
        metadata_file = temp_path / "scaler_metadata.json"
        
        print(f"Saved {len(scaler_files)} scaler files")
        print(f"Metadata file exists: {metadata_file.exists()}")
        
        # Create new scaler manager and load
        new_scaler_manager = IndividualScalerManager()
        new_scaler_manager.load_scalers(temp_path)
        
        # Test inverse transform with loaded scalers
        denormalized_data = new_scaler_manager.inverse_transform_scalar(normalized_data, variable_names)
        
        # Check accuracy
        mse = np.mean((data - denormalized_data) ** 2)
        print(f"Reconstruction MSE after save/load: {mse:.2e}")
        
        if mse < 1e-10:
            print("✅ Scaler persistence test PASSED")
            return True
        else:
            print("❌ Scaler persistence test FAILED")
            return False


def main():
    """Run all tests."""
    print("🧪 Testing IndividualScalerManager")
    print("=" * 50)
    
    tests = [
        test_scalar_normalization,
        test_pft_1d_normalization,
        test_soil_2d_normalization,
        test_scaler_persistence
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests PASSED")
    
    if passed == total:
        print("🎉 All tests passed! IndividualScalerManager is ready for integration.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
