#!/usr/bin/env python3
"""
AI Predictions to Restart File Updater

This script directly updates model restart files with AI predictions for PFT1D and soil2D variables.
It uses netCDF4 for direct file manipulation to avoid xarray encoding issues.

Usage:
    python ai_predictions_to_restart.py --variable-list CNP_IO_demo1.txt

The script will:
1. Load AI predictions and model restart file
2. Create spatial mapping between AI and model gridcells
3. Update only CNP_IO variables (PFT1D and soil2D)
4. Save updated restart file with original attributes preserved
"""

import argparse
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil
import netCDF4 as nc
import sys

# Project imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.training_config import parse_cnp_io_list


def _safe_get(ds, name):
    """Safely get a variable from dataset, with error handling."""
    if name not in ds:
        raise KeyError(f"Missing required variable/coordinate: {name}")
    return ds[name]

def _to_zero_based_index(idx_raw, n_grid):
    """Convert one-based indices to zero-based indices."""
    idx = np.asarray(idx_raw, dtype=np.int64).copy()
    if idx.size == 0:
        return np.full_like(idx, -1)
    is_one_based = (np.any(idx == n_grid) or (np.nanmin(idx) == 1))
    if is_one_based:
        idx = idx - 1
    idx[(idx < 0) | (idx >= n_grid)] = -1
    return idx

def _build_gridcell_groups(one_d_to_grid, n_grid):
    """Build mapping from gridcell to indices."""
    groups = [[] for _ in range(n_grid)]
    for idx, g in enumerate(one_d_to_grid):
        if 0 <= g < n_grid:
            groups[g].append(idx)
    return groups


def load_datasets(ai_predictions_path: Path, restart_file_path: Path) -> tuple[xr.Dataset, xr.Dataset]:
    """Load AI predictions and model restart datasets."""
    print("Loading datasets...")
    
    # Load AI predictions
    print(f"  Loading AI predictions: {ai_predictions_path}")
    ds_ai = xr.open_dataset(ai_predictions_path)
    print(f"    AI dataset shape: {dict(ds_ai.sizes)}")
    
    # Load model restart file
    print(f"  Loading model restart file: {restart_file_path}")
    ds_model = xr.open_dataset(restart_file_path)
    print(f"    Model dataset shape: {dict(ds_model.sizes)}")
    
    return ds_ai, ds_model


def create_spatial_mapping(ds_ai: xr.Dataset, ds_model: xr.Dataset) -> tuple[np.ndarray, Dict[str, Any]]:
    """Create spatial mapping between AI and model gridcells."""
    print("Creating spatial mapping...")
    
    # Get coordinates
    ai_lon = ds_ai['grid1d_lon'].values
    ai_lat = ds_ai['grid1d_lat'].values
    model_lon = ds_model['grid1d_lon'].values
    model_lat = ds_model['grid1d_lat'].values
    
    print(f"  AI coordinates: {len(ai_lon)} gridcells")
    print(f"  Model coordinates: {len(model_lon)} gridcells")
    
    # Create spatial mapping using nearest neighbor (exact same as working script)
    from scipy.spatial.distance import cdist
    ai_coords = np.column_stack([ai_lon, ai_lat])
    model_coords = np.column_stack([model_lon, model_lat])
    distances = cdist(ai_coords, model_coords)
    ai_to_model_mapping = np.argmin(distances, axis=1)
    
    print(f"  Spatial mapping created: {len(ai_to_model_mapping)} AI -> {len(set(ai_to_model_mapping))} Model")
    print(f"  Mapping range: AI gridcell 0->model gridcell {ai_to_model_mapping[0]}")
    print(f"  Mapping range: AI gridcell {len(ai_lon)-1}->model gridcell {ai_to_model_mapping[-1]}")
    
    # Get grid information from the MODEL file as the master coordinate system
    n_grid = ds_model.sizes["gridcell"]
    print(f"  Using MODEL gridcell count: {n_grid}")
    
    # Build mappings from the model file for extracting model data (exact same as working script)
    col2grid = _to_zero_based_index(_safe_get(ds_model, "cols1d_gridcell_index").values, n_grid)
    pft2grid = _to_zero_based_index(_safe_get(ds_model, "pfts1d_gridcell_index").values, n_grid)
    
    grid_to_cols = _build_gridcell_groups(col2grid, n_grid)
    grid_to_pfts = _build_gridcell_groups(pft2grid, n_grid)
    
    print(f"  Model mappings: total columns: {col2grid.size} | total pfts: {pft2grid.size}")
    print(f"  Example: gridcell 0 -> columns {grid_to_cols[0][:5]}, pfts {grid_to_pfts[0][:5]}")
    
    # Create variable mapping for PFT and column indices
    variable_mapping = {
        'grid_to_cols': grid_to_cols,
        'grid_to_pfts': grid_to_pfts,
        'n_grid': n_grid
    }
    
    return ai_to_model_mapping, variable_mapping





def create_updated_restart_file(restart_file_path: Path, output_path: Path, 
                               ai_predictions_path: Path, cnp_io_variables: List[str],
                               ai_to_model_mapping: np.ndarray, variable_mapping: Dict[str, Any]) -> None:
    """Directly update the restart file using netCDF4 without xarray encoding issues."""
    print(f"Saving updated restart file to: {output_path}")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy the original restart file
    shutil.copy2(restart_file_path, output_path)
    print(f"Copied original restart file to: {output_path}")
    
    # Open the output file for direct modification
    with nc.Dataset(output_path, 'r+') as ds_out:
        # Load AI predictions
        with nc.Dataset(ai_predictions_path, 'r') as ds_ai:
            # Update PFT variables
            for var_name in ds_ai.variables:
                if (var_name in ds_out.variables and 
                    'pft' in ds_ai.variables[var_name].dimensions and
                    var_name in cnp_io_variables):
                    print(f"  Updating PFT variable: {var_name}")
                    ai_data = ds_ai.variables[var_name][:]  # (pft, gridcell)
                    model_var = ds_out.variables[var_name]
                    print(f"    AI data shape: {ai_data.shape}")
                    print(f"    Model variable shape: {model_var.shape}")
                    
                    # Get the grid-to-pfts mapping
                    if 'grid_to_pfts' in variable_mapping:
                        grid_to_pfts = variable_mapping['grid_to_pfts']
                        
                        # For each model gridcell, update PFT data
                        for g in range(variable_mapping['n_grid']):
                            if g < len(grid_to_pfts) and len(grid_to_pfts[g]) > 0:
                                # Get PFTs in this gridcell
                                gridcell_pfts = grid_to_pfts[g]
                                
                                # Find corresponding AI gridcell using spatial mapping
                                ai_gridcell_idx = np.where(ai_to_model_mapping == g)[0]
                                if len(ai_gridcell_idx) > 0:
                                    ai_gridcell_idx = ai_gridcell_idx[0]
                                    
                                    # Update each PFT instance in this gridcell
                                    # Only update PFT1-PFT16 (skip PFT0), and only in the first column
                                    # Get the first 16 PFTs in this gridcell (exact same as working script)
                                    gridcell_pfts = gridcell_pfts[:16]  # First 16 PFTs
                                    for pft_idx, model_pft_idx in enumerate(gridcell_pfts):
                                        # Skip PFT0 (index 0), start from PFT1 (index 1)
                                        if 1 <= pft_idx <= 16 and model_pft_idx < len(model_var):
                                            # AI PFT0 -> Model PFT1, AI PFT1 -> Model PFT2, etc.
                                            # Adjust index: AI PFT k corresponds to Model PFT (k+1) in the first 16
                                            adjusted_k = pft_idx - 1  # AI PFT0 -> Model PFT1, AI PFT1 -> Model PFT2
                                            if adjusted_k < ai_data.shape[0]:
                                                model_var[model_pft_idx] = ai_data[adjusted_k, ai_gridcell_idx]
            
            # Update soil variables
            for var_name in ds_ai.variables:
                if (var_name in ds_out.variables and 
                    'column' in ds_ai.variables[var_name].dimensions and 
                    'levgrnd' in ds_ai.variables[var_name].dimensions and
                    var_name in cnp_io_variables):
                    print(f"  Updating soil variable: {var_name}")
                    ai_data = ds_ai.variables[var_name][:]  # (column, levgrnd, gridcell)
                    model_var = ds_out.variables[var_name]
                    print(f"    AI data shape: {ai_data.shape}")
                    print(f"    Model variable shape: {model_var.shape}")
                    
                    # Get the grid-to-cols mapping
                    if 'grid_to_cols' in variable_mapping:
                        grid_to_cols = variable_mapping['grid_to_cols']
                        
                        # For each model gridcell, update column data
                        for g in range(variable_mapping['n_grid']):
                            if g < len(grid_to_cols) and len(grid_to_cols[g]) > 0:
                                # Get columns in this gridcell
                                gridcell_cols = grid_to_cols[g]
                                
                                # Find corresponding AI gridcell using spatial mapping
                                ai_gridcell_idx = np.where(ai_to_model_mapping == g)[0]
                                if len(ai_gridcell_idx) > 0:
                                    ai_gridcell_idx = ai_gridcell_idx[0]
                                    
                                    # Update first column in this gridcell (use AI column 0)
                                    if len(gridcell_cols) > 0:
                                        model_col_idx = gridcell_cols[0]  # First column of this gridcell
                                        if model_col_idx < model_var.shape[0]:
                                            # Update only first 10 layers for this column (even if model has 15 layers)
                                            layers_to_update = min(10, ai_data.shape[1])
                                            for layer_idx in range(layers_to_update):
                                                # Handle AI data indexing - shape is (column, levgrnd, gridcell)
                                                if ai_data.ndim == 3:
                                                    model_var[model_col_idx, layer_idx] = ai_data[0, layer_idx, ai_gridcell_idx]
                                                else:
                                                    model_var[model_col_idx, layer_idx] = ai_data[0, layer_idx]
    
    print(f"Updated restart file saved successfully!")
    print(f"File size: {output_path.stat().st_size / (1024*1024):.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='Insert AI predictions (PFT1D and soil2D variables only) into model restart file to create updated restart file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update restart file with AI predictions
  python ai_predictions_to_restart.py \\
    --ai-predictions ai_predictions_for_plotting.nc \\
    --restart-file model_restart.nc \\
    --output updated_restart.nc

  # Use with specific variable list
  python ai_predictions_to_restart.py \\
    --ai-predictions ai_predictions_for_plotting.nc \\
    --restart-file model_restart.nc \\
    --output updated_restart.nc \\
    --variable-list CNP_IO_demo1.txt

  # Preview changes without saving
  python ai_predictions_to_restart.py \\
    --ai-predictions ai_predictions_for_plotting.nc \\
    --restart-file model_restart.nc \\
    --output updated_restart.nc \\
    --preview-only
        """
    )
    
    parser.add_argument('--ai-predictions', default='./comparison_results/ai_predictions_for_plotting.nc',
                       help='Path to AI predictions NetCDF file (ai_predictions_for_plotting.nc)')
    parser.add_argument('--restart-file', default='/mnt/proj-shared/AI4BGC_7xw/AI4BGC/ELM_data/original_20250408_trendytest_ICB1850CNPRDCTCBC.elm.r.0021-01-01-00000.nc',
                       help='Path to model restart file to update')
    parser.add_argument('--output', default=None,
                       help='Output path for updated restart file [default: auto-generated based on variable list]')
    parser.add_argument('--variable-list', type=str,
                       help='Path to CNP_IO_list file to specify which variables to update')
    parser.add_argument('--preview-only', action='store_true',
                       help='Preview changes without saving updated restart file')
    parser.add_argument('--backup', action='store_true',
                       help='Create backup of original restart file before updating')
    
    args = parser.parse_args()
    
    # Validate input files
    ai_predictions_path = Path(args.ai_predictions)
    restart_file_path = Path(args.restart_file)
    
    if not ai_predictions_path.exists():
        parser.error(f'AI predictions file not found: {ai_predictions_path}')
    
    if not restart_file_path.exists():
        parser.error(f'Restart file not found: {restart_file_path}')
    
    # Generate output path based on variable list if not specified
    if args.output is None:
        if args.variable_list:
            # Extract variable list name and create descriptive output filename
            var_list_name = Path(args.variable_list).stem
            restart_name = Path(args.restart_file).stem
            output_path = Path(f"updated_restart_{var_list_name}_{restart_name}.nc")
        else:
            # Fallback to default name
            output_path = Path("updated_restart.nc")
    else:
        output_path = Path(args.output)
    
    print("=" * 60)
    print("AI Predictions to Restart File Updater")
    print("=" * 60)
    print(f"AI predictions: {ai_predictions_path}")
    print(f"Restart file: {restart_file_path}")
    print(f"Output: {output_path}")
    print(f"Preview only: {args.preview_only}")
    print(f"Create backup: {args.backup}")
    print("=" * 60)
    
    # Load datasets
    ds_ai, ds_model = load_datasets(ai_predictions_path, restart_file_path)
    
    # Create spatial mapping
    ai_to_model_mapping, variable_mapping = create_spatial_mapping(ds_ai, ds_model)
    
    # Parse CNP_IO list if provided
    cnp_io_variables = []
    if args.variable_list:
        var_list_path = Path(args.variable_list)
        if var_list_path.exists():
            try:
                parsed_vars = parse_cnp_io_list(var_list_path)
                if 'pft_1d_variables' in parsed_vars:
                    cnp_io_variables.extend(parsed_vars['pft_1d_variables'])
                if 'variables_2d_soil' in parsed_vars:
                    cnp_io_variables.extend(parsed_vars['variables_2d_soil'])
                
                print(f"  Variables to update: {cnp_io_variables}")
            except Exception as e:
                print(f"  Warning: Could not parse variable list: {e}")
                parsed_vars = None
                cnp_io_variables = []
        else:
            print(f"  Warning: Variable list file not found: {var_list_path}")
            parsed_vars = None
            cnp_io_variables = []
    else:
        parsed_vars = None
        cnp_io_variables = []
    
    # Note: We only update variables in the CNP_IO list
    # All other variables (including timemgr_rst_nstep_rad_prev) remain completely unchanged
    print("Note: Only variables in CNP_IO list will be updated")
    print("All other variables and attributes remain unchanged")
    
    # Print summary of changes
    print("\n" + "=" * 60)
    print("UPDATE SUMMARY")
    print("=" * 60)
    
    # Count variables that will be updated (only PFT1D and soil2D from CNP_IO list)
    updated_vars = []
    for var_name in ds_ai.data_vars:
        if var_name in ds_model.data_vars and var_name in cnp_io_variables:
            # Only count PFT1D and soil2D variables that are in the CNP_IO list
            if ('pft' in ds_ai[var_name].dims) or ('column' in ds_ai[var_name].dims and 'levgrnd' in ds_ai[var_name].dims):
                updated_vars.append(var_name)
    
    print(f"Variables to update (PFT1D and soil2D from CNP_IO list): {len(updated_vars)}")
    for var_name in updated_vars:
        ai_shape = ds_ai[var_name].shape
        model_shape = ds_model[var_name].shape
        var_type = "PFT1D" if 'pft' in ds_ai[var_name].dims else "Soil2D"
        print(f"  {var_name} ({var_type}): AI {ai_shape} -> Model {model_shape}")
    
    # Show which variables were skipped
    skipped_vars = []
    for var_name in ds_ai.data_vars:
        if (var_name in ds_model.data_vars and 
            ('pft' in ds_ai[var_name].dims or ('column' in ds_ai[var_name].dims and 'levgrnd' in ds_ai[var_name].dims)) and
            var_name not in cnp_io_variables):
            skipped_vars.append(var_name)
    
    if skipped_vars:
        print(f"\nVariables skipped (not in CNP_IO list): {len(skipped_vars)}")
        for var_name in skipped_vars:
            print(f"  {var_name}")
    
    # Print coordinate information
    print(f"\nCoordinate mapping:")
    print(f"  AI gridcells: {ds_ai.sizes.get('gridcell', 'N/A')}")
    print(f"  Model gridcells: {ds_model.sizes.get('gridcell', 'N/A')}")
    print(f"  Spatial mapping: {len(ai_to_model_mapping)} AI -> {len(set(ai_to_model_mapping))} Model")
    
    if not args.preview_only:
        # Create backup if requested
        if args.backup:
            backup_path = restart_file_path.with_suffix('.backup.nc')
            print(f"\nCreating backup: {backup_path}")
            shutil.copy2(restart_file_path, backup_path)
            print(f"Backup created: {backup_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Save updated restart file using direct NetCDF manipulation
        create_updated_restart_file(restart_file_path, output_path, ai_predictions_path, 
                                   cnp_io_variables, ai_to_model_mapping, variable_mapping)
        
        print(f"\nRestart file updated successfully!")
        print(f"Original: {restart_file_path}")
        print(f"Updated: {output_path}")
        if args.backup:
            print(f"Backup: {restart_file_path.with_suffix('.backup.nc')}")
        
        print(f"\nUpdate Summary:")
        print(f"  PFT1D variables: Updated PFT1-PFT16 (skipped PFT0) in first column of each gridcell")
        print(f"  Soil2D variables: Updated first column and first 10 layers in each gridcell")
        print(f"  Model layers 11-15: Preserved (not modified)")
        print(f"  Other columns: Preserved (not modified)")
        print(f"  Spatial mapping: Used geographic coordinates to map AI gridcells to model gridcells")
        print(f"  Coordinate system: Model coordinates used as master reference for alignment")
        print(f"  Important: Only CNP_IO variables were modified - all other variables and attributes unchanged")
        
        print(f"\nYou can now use the updated restart file for model simulations!")
    else:
        print(f"\nPreview mode - no files were modified")
        print(f"To apply changes, run without --preview-only flag")
    
    # Clean up
    ds_ai.close()
    ds_model.close()


if __name__ == '__main__':
    main()
