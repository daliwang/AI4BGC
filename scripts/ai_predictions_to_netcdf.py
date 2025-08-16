#!/usr/bin/env python3
"""
Convert AI predictions to NetCDF format compatible with restart_variable_plot.py

This script takes AI predictions from the cnp_predictions directory and converts them
to a NetCDF file that can be used with the restart_variable_plot.py script for
creating comparison maps.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional

# Project imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.training_config import parse_cnp_io_list


def parse_variable_list_file(variable_list_path: str) -> Dict[str, List[str]]:
    """Parse the CNP IO list file to extract scalar, PFT1D, and soil2D variables."""
    print(f"Parsing variable list file: {variable_list_path}")
    
    variables = {
        'scalar': [],
        'pft1d': [],
        'soil2d': []
    }
    
    try:
        with open(variable_list_path, 'r') as f:
            lines = f.readlines()
        
        current_section = None
        for line in lines:
            line = line.strip()
            
            # Detect sections - handle variations in section names
            if 'SCALAR VARIABLES' in line or 'SCALAR VARIABLES (1D' in line:
                current_section = 'scalar'
                print(f"  Found scalar section: {line}")
                continue
            elif '1D PFT VARIABLES' in line or '1D PFT VARIABLES (' in line:
                current_section = 'pft1d'
                print(f"  Found PFT1D section: {line}")
                continue
            elif '2D VARIABLES' in line or '2D VARIABLES (' in line:
                current_section = 'soil2d'
                print(f"  Found 2D section: {line}")
                continue
            elif 'OUTPUT VARIABLES' in line:
                current_section = None
                continue
            
            # Parse variables in current section
            if current_section and line.startswith('•'):
                # Extract variable names from the line
                # Handle multiple variables per line separated by commas
                var_part = line.replace('•', '').strip()
                var_names = [v.strip() for v in var_part.split(',')]
                
                for var_name in var_names:
                    if var_name:  # Skip empty strings
                        # Remove Y_ prefix if present
                        if var_name.startswith('Y_'):
                            var_name = var_name[2:]
                        
                        if current_section == 'scalar':
                            variables['scalar'].append(var_name)
                        elif current_section == 'pft1d':
                            variables['pft1d'].append(var_name)
                        elif current_section == 'soil2d':
                            variables['soil2d'].append(var_name)
        
        print(f"  Scalar variables: {len(variables['scalar'])}")
        print(f"  PFT1D variables: {len(variables['pft1d'])}")
        print(f"  Soil2D variables: {len(variables['soil2d'])}")
        
        # Print the actual variables found for debugging
        print(f"  Scalar: {variables['scalar']}")
        print(f"  PFT1D: {variables['pft1d']}")
        print(f"  Soil2D: {variables['soil2d']}")
        
        return variables
        
    except Exception as e:
        print(f"Warning: Could not parse variable list file: {e}")
        # Return default variables if parsing fails
        return {
            'scalar': ['GPP', 'NPP', 'AR', 'HR'],
            'pft1d': ['deadcrootc', 'deadcrootn', 'deadcrootp', 'deadstemc', 'deadstemn', 'deadstemp', 'frootc', 'frootc_storage', 'leafc', 'leafc_storage', 'totcolp', 'totlitc', 'totvegc', 'tlai'],
            'soil2d': ['cwdc_vr', 'cwdn_vr', 'cwdp_vr', 'litr1c_vr', 'litr2c_vr', 'litr3c_vr', 'litr1n_vr', 'litr2n_vr', 'litr3n_vr', 'litr1p_vr', 'litr2p_vr', 'litr3p_vr', 'sminn_vr', 'smin_no3_vr', 'smin_nh4_vr', 'soil1c_vr', 'soil1n_vr', 'soil1p_vr', 'soil2c_vr', 'soil2n_vr', 'soil2p_vr', 'soil3c_vr', 'soil3n_vr', 'soil3p_vr', 'soil4c_vr', 'soil4n_vr', 'soil4p_vr']
        }


def load_ai_predictions(predictions_dir: Path) -> Dict[str, Any]:
    """Load AI model predictions from the predictions directory."""
    print(f"Loading AI predictions from: {predictions_dir}")
    
    preds = {}
    
    # Load scalar predictions
    scalar_path = predictions_dir / 'predictions_scalar.csv'
    if scalar_path.exists():
        preds['scalar'] = pd.read_csv(scalar_path)
        print(f"  Loaded scalar predictions: {preds['scalar'].shape}")
    
    # Load 1D PFT predictions
    pft_dir = predictions_dir / 'pft_1d_predictions'
    if pft_dir.exists():
        preds['pft_1d'] = {}
        for p in sorted(pft_dir.glob('predictions_*.csv')):
            # Extract variable name from filename (e.g., predictions_Y_tlai.csv -> tlai)
            var_name = p.stem.replace('predictions_Y_', '')
            preds['pft_1d'][var_name] = pd.read_csv(p)
            print(f"  Loaded PFT predictions for {var_name}: {preds['pft_1d'][var_name].shape}")
            print(f"    File: {p.name}")
            print(f"    Columns: {list(preds['pft_1d'][var_name].columns)[:5]}...")
    
    # Load 2D soil predictions
    soil_dir = predictions_dir / 'soil_2d_predictions'
    if soil_dir.exists():
        preds['soil_2d'] = {}
        for p in sorted(soil_dir.glob('predictions_*.csv')):
            # Extract variable name from filename (e.g., predictions_Y_cwdc_vr.csv -> cwdc_vr)
            var_name = p.stem.replace('predictions_Y_', '')
            preds['soil_2d'][var_name] = pd.read_csv(p)
            print(f"  Loaded soil predictions for {var_name}: {preds['soil_2d'][var_name].shape}")
            print(f"    File: {p.name}")
            print(f"    Columns: {list(preds['soil_2d'][var_name].columns)[:5]}...")
    
    # Load static inverse mapping for coordinates
    static_inv = predictions_dir / 'test_static_inverse.csv'
    if static_inv.exists():
        preds['test_static_inverse'] = pd.read_csv(static_inv)
        print(f"  Loaded static inverse mapping: {preds['test_static_inverse'].shape}")
    
    return preds


def create_netcdf_structure(ai_preds: Dict[str, Any], variable_list: Dict[str, List[str]], 
                           output_path: Path) -> xr.Dataset:
    """Create NetCDF structure compatible with restart_variable_plot.py"""
    print("Creating NetCDF structure...")
    
    # Get coordinates from static inverse mapping
    if 'test_static_inverse' not in ai_preds:
        raise ValueError("test_static_inverse.csv not found - needed for coordinates")
    
    static_df = ai_preds['test_static_inverse']
    n_samples = len(static_df)
    
    # Create coordinate variables
    coords = {
        'gridcell': np.arange(n_samples),
        'pft': np.arange(1, 17),  # PFT1-16 (excluding PFT0)
        'column': np.arange(1),    # Only first column
        'levgrnd': np.arange(10),  # Only first 10 layers
    }
    
    # Create the dataset
    ds = xr.Dataset(coords=coords)
    
    # Add coordinate variables that restart_variable_plot.py expects
    # Grid coordinates: 1D arrays for gridcell dimension
    ds['grid1d_lon'] = xr.DataArray(static_df['Longitude'].values, dims=['gridcell'])
    ds['grid1d_lat'] = xr.DataArray(static_df['Latitude'].values, dims=['gridcell'])
    
    # PFT coordinates: For PFT variables, we need to create proper mapping
    # Each PFT gets assigned to the first gridcell (index 1, 1-based)
    pft_lon = np.full(16, static_df['Longitude'].iloc[0], dtype=float)
    pft_lat = np.full(16, static_df['Latitude'].iloc[0], dtype=float)
    ds['pfts1d_lon'] = xr.DataArray(pft_lon, dims=['pft'])
    ds['pfts1d_lat'] = xr.DataArray(pft_lat, dims=['pft'])
    
    # Column coordinates: For column variables, we need to create proper mapping
    # Each column gets assigned to the first gridcell (index 1, 1-based)
    col_lon = np.full(1, static_df['Longitude'].iloc[0], dtype=float)
    col_lat = np.full(1, static_df['Latitude'].iloc[0], dtype=float)
    ds['cols1d_lon'] = xr.DataArray(col_lon, dims=['column'])
    ds['cols1d_lat'] = xr.DataArray(col_lat, dims=['column'])
    
    # Add gridcell indices (1-based as expected by restart_variable_plot.py)
    # All PFTs and columns are assigned to gridcell 1
    ds['pfts1d_gridcell_index'] = xr.DataArray(np.ones(16, dtype=int), dims=['pft'])
    ds['cols1d_gridcell_index'] = xr.DataArray(np.ones(1, dtype=int), dims=['column'])
    
    print(f"  Created base structure with {n_samples} gridcells")
    print(f"  PFT coordinates: {pft_lon.shape} (all assigned to first gridcell)")
    print(f"  Column coordinates: {col_lon.shape} (all assigned to first gridcell)")
    return ds


def add_scalar_variables(ds: xr.Dataset, ai_preds: Dict[str, Any], 
                        variable_list: Dict[str, List[str]]) -> None:
    """Add scalar variables to the dataset."""
    if 'scalar' not in ai_preds:
        return
    
    print("Adding scalar variables...")
    scalar_df = ai_preds['scalar']
    
    for var_name in variable_list['scalar']:
        # Look for column with Y_ prefix
        col_name = f'Y_{var_name}'
        if col_name in scalar_df.columns:
            values = scalar_df[col_name].values
            ds[var_name] = xr.DataArray(values, dims=['gridcell'])
            print(f"  Added {var_name}: {values.shape}")


def add_pft_variables(ds: xr.Dataset, ai_preds: Dict[str, Any], 
                     variable_list: Dict[str, List[str]]) -> None:
    """Add PFT variables to the dataset."""
    if 'pft_1d' not in ai_preds:
        print("  Warning: No PFT predictions found in ai_preds")
        print(f"    Available keys: {list(ai_preds.keys())}")
        return
    
    print("Adding PFT variables...")
    print(f"  PFT variables to add: {variable_list['pft1d']}")
    print(f"  Available PFT predictions: {list(ai_preds['pft_1d'].keys())}")
    
    for var_name in variable_list['pft1d']:
        print(f"    Processing PFT variable: {var_name}")
        if var_name in ai_preds['pft_1d']:
            pft_df = ai_preds['pft_1d'][var_name]
            print(f"      Found PFT dataframe: {pft_df.shape}")
            print(f"      Columns: {list(pft_df.columns)[:5]}...")  # Show first 5 columns
            
            # PFT variables should have columns like Y_leafc_pft1, Y_leafc_pft2, etc.
            pft_cols = [c for c in pft_df.columns if c.startswith(f'Y_{var_name}_pft')]
            print(f"      PFT columns starting with Y_{var_name}_pft: {len(pft_cols)}")
            
            if pft_cols:
                # Sort PFT columns numerically
                def pft_num(col):
                    try:
                        return int(col.split('_pft')[-1])
                    except:
                        return 999
                
                pft_cols = sorted(pft_cols, key=pft_num)
                
                # Create array with shape (pft, gridcell)
                # This matches what restart_variable_plot.py expects for PFT variables
                pft_data = np.zeros((16, len(pft_df)), dtype=float)
                
                for i, col in enumerate(pft_cols):
                    if i < 16:  # PFT1-16
                        pft_data[i, :] = pft_df[col].values
                
                ds[var_name] = xr.DataArray(pft_data, dims=['pft', 'gridcell'])
                print(f"      Added {var_name}: {pft_data.shape}")
                print(f"      PFT columns found: {len(pft_cols)}")
                print(f"      PFT range: {pft_cols[0]} to {pft_cols[-1]}")
            else:
                print(f"      Warning: No PFT columns found for {var_name}")
                print(f"      Available columns: {list(pft_df.columns)[:10]}...")
        else:
            print(f"      Warning: {var_name} not found in PFT predictions")
            print(f"      Available PFT variables: {list(ai_preds['pft_1d'].keys())}")


def add_soil_variables(ds: xr.Dataset, ai_preds: Dict[str, Any], 
                      variable_list: Dict[str, List[str]]) -> None:
    """Add soil 2D variables to the dataset."""
    if 'soil_2d' not in ai_preds:
        print("  Warning: No soil predictions found in ai_preds")
        print(f"    Available keys: {list(ai_preds.keys())}")
        return
    
    print("Adding soil 2D variables...")
    print(f"  Soil variables to add: {variable_list['soil2d']}")
    print(f"  Available soil predictions: {list(ai_preds['soil_2d'].keys())}")
    
    for var_name in variable_list['soil2d']:
        print(f"    Processing soil variable: {var_name}")
        if var_name in ai_preds['soil_2d']:
            soil_df = ai_preds['soil_2d'][var_name]
            print(f"      Found soil dataframe: {soil_df.shape}")
            print(f"      Total columns: {len(soil_df.columns)}")
            print(f"      Expected: 18 columns × 10 layers = 180 columns")
            print(f"      First few columns: {list(soil_df.columns)[:5]}...")
            print(f"      Last few columns: {list(soil_df.columns)[-5:]}...")
            
            # Soil variables have columns like Y_cwdc_vr_col1_layer1, Y_cwdc_vr_col1_layer2, etc.
            # We want the first column (col1) and its 10 layers
            first_col_layer_cols = [c for c in soil_df.columns if c.startswith(f'Y_{var_name}_col1_layer')]
            print(f"      First column layer columns starting with Y_{var_name}_col1_layer: {len(first_col_layer_cols)}")
            
            if first_col_layer_cols:
                # Sort layer columns numerically
                def layer_num(col):
                    try:
                        return int(col.split('_layer')[-1])
                    except:
                        return 999
                
                first_col_layer_cols = sorted(first_col_layer_cols, key=layer_num)
                
                # Create array with shape (column, levgrnd, gridcell)
                # This matches what restart_variable_plot.py expects for column/levgrnd variables
                soil_data = np.zeros((1, 10, len(soil_df)), dtype=float)
                
                for i, col in enumerate(first_col_layer_cols):
                    if i < 10:  # First 10 layers
                        soil_data[0, i, :] = soil_df[col].values
                
                ds[var_name] = xr.DataArray(soil_data, dims=['column', 'levgrnd', 'gridcell'])
                print(f"      Added {var_name}: {soil_data.shape}")
                print(f"      First column layer columns found: {len(first_col_layer_cols)}")
                print(f"      Layer range: {first_col_layer_cols[0]} to {first_col_layer_cols[-1]}")
            else:
                print(f"      Warning: No first column layer columns found for {var_name}")
                print(f"      Available columns: {list(soil_df.columns)[:10]}...")
                print(f"      Looking for pattern: Y_{var_name}_col1_layer*")
        else:
            print(f"      Warning: {var_name} not found in soil predictions")
            print(f"      Available soil variables: {list(ai_preds['soil_2d'].keys())}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert AI predictions to NetCDF format compatible with restart_variable_plot.py'
    )
    parser.add_argument('--ai-predictions', default='cnp_inference_entire_dataset/cnp_predictions',
                       help='Path to AI predictions directory (cnp_predictions)')
    parser.add_argument('--variable-list', required=True,
                       help='Path to CNP_IO_list file')
    parser.add_argument('--output', default='comparison_results/ai_predictions_for_plotting.nc',
                       help='Output NetCDF file path')
    parser.add_argument('--examples', action='store_true', 
                       help='Show example usage and exit')
    
    args = parser.parse_args()
    
    if args.examples:
        print("""
Examples:
  # Convert AI predictions to NetCDF
  python ai_predictions_to_netcdf.py \\
    --ai-predictions cnp_results/run_20250814_193455/cnp_predictions \\
    --variable-list CNP_IO_list1.txt \\
    --output ai_predictions.nc

  # Use with restart_variable_plot.py
  python restart_variable_plot.py  # Edit FILE_NEW to point to ai_predictions.nc
        """)
        return
    
    # Validate inputs
    ai_predictions_dir = Path(args.ai_predictions)
    if not ai_predictions_dir.exists():
        parser.error(f'AI predictions directory not found: {ai_predictions_dir}')
    
    variable_list_path = args.variable_list
    if not Path(variable_list_path).exists():
        parser.error(f'Variable list file not found: {variable_list_path}')
    
    output_path = Path(args.output)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("AI Predictions to NetCDF Converter")
    print("=" * 60)
    print(f"AI predictions: {ai_predictions_dir}")
    print(f"Variable list: {variable_list_path}")
    print(f"Output: {output_path}")
    print("=" * 60)
    
    # Parse variable list
    variable_list = parse_variable_list_file(variable_list_path)
    
    # Load AI predictions
    ai_preds = load_ai_predictions(ai_predictions_dir)
    
    # Create NetCDF structure
    ds = create_netcdf_structure(ai_preds, variable_list, output_path)
    
    # Add variables
    add_scalar_variables(ds, ai_preds, variable_list)
    add_pft_variables(ds, ai_preds, variable_list)
    add_soil_variables(ds, ai_preds, variable_list)
    
    # Save to NetCDF
    print(f"\nSaving NetCDF file to: {output_path}")
    ds.to_netcdf(output_path)
    
    print("\nNetCDF file created successfully!")
    print(f"You can now use this file with restart_variable_plot.py")
    print(f"Edit the FILE_NEW variable in restart_variable_plot.py to point to: {output_path}")
    
    # Print summary
    print(f"\nDataset summary:")
    print(f"  Gridcells: {ds.sizes['gridcell']}")
    print(f"  PFTs: {ds.sizes['pft']} (PFT1-16)")
    print(f"  Columns: {ds.sizes['column']} (first column only)")
    print(f"  Soil layers: {ds.sizes['levgrnd']} (first 10 layers only)")
    print(f"  Variables: {len(ds.data_vars)}")
    
    # Print variable details
    print(f"\nVariable details:")
    for var_name, var_data in ds.data_vars.items():
        if var_name not in ['grid1d_lon', 'grid1d_lat', 'pfts1d_lon', 'pfts1d_lat', 
                           'cols1d_lon', 'cols1d_lat', 'pfts1d_gridcell_index', 'cols1d_gridcell_index']:
            print(f"  {var_name}: {var_data.dims} {var_data.shape}")
    
    # Print coordinate details
    print(f"\nCoordinate details:")
    for coord_name, coord_data in ds.coords.items():
        print(f"  {coord_name}: {coord_data.dims} {coord_data.shape}")


if __name__ == '__main__':
    main()
