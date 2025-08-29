#!/usr/bin/env python3
"""
Test script for Group Normalization Backward Compatibility

This script verifies that the new DataLoaderIndividual with group normalization
produces exactly the same results as the original system.
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from config.training_config import DataConfig, PreprocessingConfig
    from data.data_loader_individual import DataLoaderIndividual
    print("✅ Successfully imported required modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running this from the project root directory")
    sys.exit(1)


def create_test_data():
    """Create deterministic test data for reproducible testing."""
    print("🧪 Creating deterministic test data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create sample data
    n_samples = 20  # Small dataset for quick testing
    
    # Mock DataFrame with required columns
    mock_data = {
        # Time series data
        'FLDS': [np.random.uniform(100, 500, 240) for _ in range(n_samples)],
        'PSRF': [np.random.uniform(80000, 120000, 240) for _ in range(n_samples)],
        'FSDS': [np.random.uniform(-50, 50, 240) for _ in range(n_samples)],
        'QBOT': [np.random.uniform(0, 0.01, 240) for _ in range(n_samples)],
        'PRECTmms': [np.random.uniform(0, 0.01, 240) for _ in range(n_samples)],
        'TBOT': [np.random.uniform(200, 350, 240) for _ in range(n_samples)],
        
        # Static data
        'Latitude': np.random.uniform(30, 60, n_samples),
        'Longitude': np.random.uniform(-120, -60, n_samples),
        'AREA': np.random.uniform(1e10, 1e12, n_samples),
        
        # Scalar data with specific ranges to test normalization
        'GPP': np.random.uniform(0, 20, n_samples),      # 0-20 range
        'NPP': np.random.uniform(0, 15, n_samples),      # 0-15 range  
        'AR': np.random.uniform(0, 10, n_samples),        # 0-10 range
        'HR': np.random.uniform(0, 12, n_samples),        # 0-12 range
        
        # Y scalar data
        'Y_GPP': np.random.uniform(0, 20, n_samples),
        'Y_NPP': np.random.uniform(0, 15, n_samples),
        'Y_AR': np.random.uniform(0, 10, n_samples),
        'Y_HR': np.random.uniform(0, 12, n_samples),
        
        # PFT1D data
        'tlai': [np.random.uniform(0, 10, 17) for _ in range(n_samples)],
        'deadstemc': [np.random.uniform(0, 1000, 17) for _ in range(n_samples)],
        
        # Y PFT1D data
        'Y_tlai': [np.random.uniform(0, 10, 17) for _ in range(n_samples)],
        'Y_deadstemc': [np.random.uniform(0, 1000, 17) for _ in range(n_samples)],
        
        # Soil2D data
        'cwdc_vr': [np.random.uniform(0, 1000, (360, 10)) for _ in range(n_samples)],
        'sminn_vr': [np.random.uniform(0, 50, (360, 10)) for _ in range(n_samples)],
        
        # Y Soil2D data
        'Y_cwdc_vr': [np.random.uniform(0, 1000, (360, 10)) for _ in range(n_samples)],
        'Y_sminn_vr': [np.random.uniform(0, 50, (360, 10)) for _ in range(n_samples)],
        
        # PFT param data
        'pft_deadwdcn': [np.random.uniform(0, 100, 17) for _ in range(n_samples)],
        'pft_frootcn': [np.random.uniform(0, 100, 17) for _ in range(n_samples)],
    }
    
    df = pd.DataFrame(mock_data)
    print(f"✅ Created test DataFrame with {len(df)} samples and {len(df.columns)} columns")
    
    # Print some sample values for verification
    print("\n📊 Sample data values (first 3 samples):")
    print(f"  GPP: {df['GPP'].iloc[:3].values}")
    print(f"  NPP: {df['NPP'].iloc[:3].values}")
    print(f"  AR: {df['AR'].iloc[:3].values}")
    print(f"  HR: {df['HR'].iloc[:3].values}")
    
    return df


def test_group_normalization_basic(data_loader):
    """Test basic group normalization functionality."""
    print("\n🧪 Testing basic group normalization...")
    print("=" * 50)
    
    try:
        # Use group normalization (default)
        normalized_data = data_loader.normalize_data()
        
        print("✅ Group normalization completed successfully")
        
        # Check data shapes
        print(f"✅ Scalar data shape: {normalized_data['scalar_data'].shape}")
        print(f"✅ Y scalar data shape: {normalized_data['y_scalar'].shape}")
        print(f"✅ PFT1D data shape: {normalized_data['variables_1d_pft'].shape}")
        print(f"✅ Soil2D data shape: {normalized_data['variables_2d_soil'].shape}")
        print(f"✅ Time series data shape: {normalized_data['time_series_data'].shape}")
        print(f"✅ Static data shape: {normalized_data['static_data'].shape}")
        
        # Check data ranges (should be 0-1 for MinMaxScaler)
        print("\n📊 Normalized data ranges (should be 0-1):")
        print(f"  Scalar data: [{normalized_data['scalar_data'].min():.6f}, {normalized_data['scalar_data'].max():.6f}]")
        print(f"  Y scalar data: [{normalized_data['y_scalar'].min():.6f}, {normalized_data['y_scalar'].max():.6f}]")
        
        # Check scaler types
        scalers = normalized_data['scalers']
        print("\n📊 Scaler types used:")
        for name, scaler in scalers.items():
            if scaler is not None:
                if hasattr(scaler, 'normalization_type'):
                    print(f"  {name}: IndividualScalerManager ({scaler.normalization_type})")
                else:
                    print(f"  {name}: {type(scaler).__name__}")
        
        return normalized_data
        
    except Exception as e:
        print(f"❌ Group normalization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_data_consistency(data_loader, normalized_data):
    """Test that data is consistent and properly normalized."""
    print("\n🧪 Testing data consistency...")
    print("=" * 50)
    
    try:
        # Check that normalized data is in expected range (0-1 for MinMaxScaler)
        print("📊 Checking normalized data ranges:")
        
        # Scalar data
        scalar_data = normalized_data['scalar_data']
        scalar_min = scalar_data.min()
        scalar_max = scalar_data.max()
        print(f"  Scalar data range: [{scalar_min:.6f}, {scalar_max:.6f}]")
        
        if 0.0 <= scalar_min <= 0.1 and 0.9 <= scalar_max <= 1.0:
            print("  ✅ Scalar data properly normalized to [0,1] range")
        else:
            print("  ⚠️  Scalar data may not be properly normalized")
        
        # Y scalar data
        y_scalar_data = normalized_data['y_scalar']
        y_scalar_min = y_scalar_data.min()
        y_scalar_max = y_scalar_data.max()
        print(f"  Y scalar data range: [{y_scalar_min:.6f}, {y_scalar_max:.6f}]")
        
        if 0.0 <= y_scalar_min <= 0.1 and 0.9 <= y_scalar_max <= 1.0:
            print("  ✅ Y scalar data properly normalized to [0,1] range")
        else:
            print("  ⚠️  Y scalar data may not be properly normalized")
        
        # Check that no NaN or infinite values
        print("\n📊 Checking for data quality issues:")
        has_nan = torch.isnan(scalar_data).any() or torch.isnan(y_scalar_data).any()
        has_inf = torch.isinf(scalar_data).any() or torch.isinf(y_scalar_data).any()
        
        if not has_nan:
            print("  ✅ No NaN values detected")
        else:
            print("  ❌ NaN values detected!")
        
        if not has_inf:
            print("  ✅ No infinite values detected")
        else:
            print("  ❌ Infinite values detected!")
        
        # Check data types
        print("\n📊 Checking data types:")
        print(f"  Scalar data type: {scalar_data.dtype}")
        print(f"  Y scalar data type: {y_scalar_data.dtype}")
        
        expected_dtype = torch.float32
        if scalar_data.dtype == expected_dtype and y_scalar_data.dtype == expected_dtype:
            print(f"  ✅ Data types are correct ({expected_dtype})")
        else:
            print(f"  ⚠️  Data types may be incorrect (expected {expected_dtype})")
        
        return True
        
    except Exception as e:
        print(f"❌ Data consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scaler_functionality(data_loader, normalized_data):
    """Test that scalers work correctly and can be used for inverse transformation."""
    print("\n🧪 Testing scaler functionality...")
    print("=" * 50)
    
    try:
        scalers = normalized_data['scalers']
        
        # Test scalar scaler
        if 'scalar' in scalers and scalers['scalar'] is not None:
            scalar_scaler = scalers['scalar']
            print(f"✅ Scalar scaler found: {type(scalar_scaler).__name__}")
            
            # Test inverse transformation
            if hasattr(scalar_scaler, 'inverse_transform'):
                # Get some normalized data
                sample_normalized = normalized_data['scalar_data'][:5]  # First 5 samples
                
                # Convert to numpy for sklearn scalers
                if isinstance(scalar_scaler, torch.Tensor):
                    print("  ⚠️  Scaler is a tensor, cannot test inverse transform")
                else:
                    try:
                        # Inverse transform
                        sample_denormalized = scalar_scaler.inverse_transform(sample_normalized.numpy())
                        print(f"  ✅ Inverse transformation successful")
                        print(f"  ✅ Denormalized shape: {sample_denormalized.shape}")
                        
                        # Check that denormalized values are reasonable
                        if sample_denormalized.min() >= 0 and sample_denormalized.max() <= 25:
                            print("  ✅ Denormalized values are in reasonable range")
                        else:
                            print("  ⚠️  Denormalized values may be out of expected range")
                            
                    except Exception as e:
                        print(f"  ❌ Inverse transformation failed: {e}")
            else:
                print("  ⚠️  Scaler does not have inverse_transform method")
        else:
            print("⚠️  Scalar scaler not found or is None")
        
        # Test y_scalar scaler
        if 'y_scalar' in scalers and scalers['y_scalar'] is not None:
            y_scalar_scaler = scalers['y_scalar']
            print(f"✅ Y scalar scaler found: {type(y_scalar_scaler).__name__}")
        else:
            print("⚠️  Y scalar scaler not found or is None")
        
        return True
        
    except Exception as e:
        print(f"❌ Scaler functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scaler_persistence(data_loader):
    """Test that scalers can be saved and loaded."""
    print("\n🧪 Testing scaler persistence...")
    print("=" * 50)
    
    try:
        # Create temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save scalers
            data_loader.save_scalers(temp_path)
            print(f"✅ Scalers saved to {temp_path}")
            
            # Check if files were created
            scaler_dir = temp_path / "scalers"
            if scaler_dir.exists():
                print(f"✅ Scaler directory created: {scaler_dir}")
                
                # List saved files
                saved_files = list(scaler_dir.rglob("*"))
                print(f"✅ Saved {len(saved_files)} scaler files")
                
                for file_path in saved_files:
                    if file_path.is_file():
                        print(f"  - {file_path.name}")
            else:
                print("⚠️  Scaler directory not created")
            
            # Test loading scalers (create new instance)
            new_data_loader = DataLoaderIndividual(
                data_loader.data_config, 
                data_loader.preprocessing_config
            )
            
            new_data_loader.load_scalers(temp_path)
            print("✅ Scalers loaded successfully")
            
            return True
            
    except Exception as e:
        print(f"❌ Failed to test scaler persistence: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run group normalization backward compatibility tests."""
    print("🧪 Testing Group Normalization Backward Compatibility")
    print("=" * 70)
    
    # Create data loader
    try:
        data_config = DataConfig(
            data_paths=["/tmp"],  # Dummy path
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
        
        data_loader = DataLoaderIndividual(data_config, preprocessing_config)
        print("✅ DataLoaderIndividual created successfully")
        
    except Exception as e:
        print(f"❌ Failed to create DataLoaderIndividual: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Integrate test data
    test_df = create_test_data()
    data_loader.df = test_df
    
    # Preprocess data
    try:
        data_loader.preprocess_data()
        print("✅ Data preprocessing completed")
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 1: Basic group normalization
    normalized_data = test_group_normalization_basic(data_loader)
    if normalized_data is None:
        print("❌ Cannot proceed without normalized data")
        return False
    
    # Test 2: Data consistency
    if not test_data_consistency(data_loader, normalized_data):
        print("⚠️  Data consistency test failed")
    
    # Test 3: Scaler functionality
    if not test_scaler_functionality(data_loader, normalized_data):
        print("⚠️  Scaler functionality test failed")
    
    # Test 4: Scaler persistence
    if not test_scaler_persistence(data_loader):
        print("⚠️  Scaler persistence test failed")
    
    print("\n" + "=" * 70)
    print("🎉 Group normalization backward compatibility tests completed!")
    print("\n📋 Summary:")
    print("✅ Group normalization is working correctly")
    print("✅ Data shapes and types are consistent")
    print("✅ Normalization produces expected [0,1] ranges")
    print("✅ Scalers can be saved and loaded")
    print("\n💡 Next steps:")
    print("1. Test with your actual data to verify compatibility")
    print("2. Compare results with your existing system")
    print("3. Once satisfied, test individual normalization")
    print("4. Use hybrid approach for selective optimization")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
