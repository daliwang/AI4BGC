#!/usr/bin/env python3
"""
Test script for different normalization approaches

This script demonstrates the three normalization approaches:
1. Group normalization (default)
2. Individual normalization 
3. Hybrid normalization (mix of both)
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


def create_mock_data():
    """Create mock data for testing."""
    print("🧪 Creating mock data...")
    
    # Create sample data
    n_samples = 50  # Smaller dataset for testing
    
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
        
        # Scalar data with different ranges to demonstrate individual normalization benefits
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
    print(f"✅ Created mock DataFrame with {len(df)} samples and {len(df.columns)} columns")
    return df


def test_group_normalization(data_loader):
    """Test group normalization approach."""
    print("\n🧪 Testing GROUP normalization approach...")
    print("=" * 50)
    
    try:
        # Use group normalization (default)
        normalized_data = data_loader.normalize_data()
        
        print("✅ Group normalization completed successfully")
        print(f"✅ Scalar data shape: {normalized_data['scalar_data'].shape}")
        print(f"✅ Y scalar data shape: {normalized_data['y_scalar'].shape}")
        print(f"✅ PFT1D data shape: {normalized_data['variables_1d_pft'].shape}")
        print(f"✅ Soil2D data shape: {normalized_data['variables_2d_soil'].shape}")
        
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
        return None


def test_individual_normalization(data_loader):
    """Test individual normalization approach."""
    print("\n🧪 Testing INDIVIDUAL normalization approach...")
    print("=" * 50)
    
    try:
        # Use individual normalization
        normalized_data = data_loader.normalize_data_individual()
        
        print("✅ Individual normalization completed successfully")
        print(f"✅ Scalar data shape: {normalized_data['scalar_data'].shape}")
        print(f"✅ Y scalar data shape: {normalized_data['y_scalar'].shape}")
        print(f"✅ PFT1D data shape: {normalized_data['variables_1d_pft'].shape}")
        print(f"✅ Soil2D data shape: {normalized_data['variables_2d_soil'].shape}")
        
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
        print(f"❌ Individual normalization failed: {e}")
        return None


def test_hybrid_normalization(data_loader):
    """Test hybrid normalization approach."""
    print("\n🧪 Testing HYBRID normalization approach...")
    print("=" * 50)
    
    try:
        # Use hybrid normalization - individual for scalars, group for others
        use_individual_for = ['scalar', 'y_scalar']
        normalized_data = data_loader.normalize_data_hybrid(use_individual_for)
        
        print("✅ Hybrid normalization completed successfully")
        print(f"✅ Individual normalization used for: {use_individual_for}")
        print(f"✅ Scalar data shape: {normalized_data['scalar_data'].shape}")
        print(f"✅ Y scalar data shape: {normalized_data['y_scalar'].shape}")
        print(f"✅ PFT1D data shape: {normalized_data['variables_1d_pft'].shape}")
        print(f"✅ Soil2D data shape: {normalized_data['variables_2d_soil'].shape}")
        
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
        print(f"❌ Hybrid normalization failed: {e}")
        return None


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
        return False


def compare_normalization_approaches(group_data, individual_data, hybrid_data):
    """Compare the different normalization approaches."""
    print("\n🧪 Comparing normalization approaches...")
    print("=" * 50)
    
    try:
        # Compare data shapes
        print("📊 Data shapes comparison:")
        data_types = ['scalar_data', 'y_scalar', 'variables_1d_pft', 'variables_2d_soil']
        
        for data_type in data_types:
            if data_type in group_data and data_type in individual_data and data_type in hybrid_data:
                group_shape = group_data[data_type].shape
                individual_shape = individual_data[data_type].shape
                hybrid_shape = hybrid_data[data_type].shape
                
                print(f"  {data_type}:")
                print(f"    Group:      {group_shape}")
                print(f"    Individual: {individual_shape}")
                print(f"    Hybrid:     {hybrid_shape}")
                
                # Check if shapes are consistent
                if group_shape == individual_shape == hybrid_shape:
                    print(f"    ✅ All approaches produce consistent shapes")
                else:
                    print(f"    ⚠️  Shape mismatch detected")
        
        # Compare scaler counts
        print("\n📊 Scaler counts comparison:")
        print(f"  Group normalization:      {len([s for s in group_data['scalers'].values() if s is not None])} scalers")
        print(f"  Individual normalization:  {len([s for s in individual_data['scalers'].values() if s is not None])} scalers")
        print(f"  Hybrid normalization:     {len([s for s in hybrid_data['scalers'].values() if s is not None])} scalers")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to compare normalization approaches: {e}")
        return False


def main():
    """Run all normalization approach tests."""
    print("🧪 Testing Different Normalization Approaches")
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
        return False
    
    # Integrate mock data
    mock_df = create_mock_data()
    data_loader.df = mock_df
    
    # Preprocess data
    try:
        data_loader.preprocess_data()
        print("✅ Data preprocessing completed")
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        return False
    
    # Test 1: Group normalization
    group_data = test_group_normalization(data_loader)
    if group_data is None:
        print("❌ Cannot proceed without group normalization")
        return False
    
    # Test 2: Individual normalization
    individual_data = test_individual_normalization(data_loader)
    if individual_data is None:
        print("❌ Cannot proceed without individual normalization")
        return False
    
    # Test 3: Hybrid normalization
    hybrid_data = test_hybrid_normalization(data_loader)
    if hybrid_data is None:
        print("❌ Cannot proceed without hybrid normalization")
        return False
    
    # Test 4: Compare approaches
    if not compare_normalization_approaches(group_data, individual_data, hybrid_data):
        print("⚠️  Comparison failed")
    
    # Test 5: Scaler persistence
    if not test_scaler_persistence(data_loader):
        print("⚠️  Scaler persistence test failed")
    
    print("\n" + "=" * 70)
    print("🎉 All normalization approach tests completed!")
    print("\n📋 Summary of approaches:")
    print("1. GROUP normalization: Default approach, uses one scaler per data type")
    print("2. INDIVIDUAL normalization: Optimal approach, uses individual scalers for each variable")
    print("3. HYBRID normalization: Flexible approach, mix of both methods")
    print("\n💡 Usage recommendations:")
    print("- Use GROUP for: Quick testing, memory-constrained environments")
    print("- Use INDIVIDUAL for: Production training, optimal performance")
    print("- Use HYBRID for: Selective optimization, specific variable types")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
