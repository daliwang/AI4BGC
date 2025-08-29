#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
import textwrap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import seaborn as sns
from scipy import stats
from scipy.spatial import cKDTree
import warnings

import torch
from netCDF4 import Dataset

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
            
            # Detect sections
            if 'SCALAR VARIABLES' in line:
                current_section = 'scalar'
                continue
            elif '1D PFT VARIABLES' in line:
                current_section = 'pft1d'
                continue
            elif '2D VARIABLES' in line:
                current_section = 'soil2d'
                continue
            elif 'OUTPUT VARIABLES' in line:
                current_section = None
                continue
            
            # Parse variables in current section
            if current_section and line.startswith('•'):
                # Extract variable name (remove Y_ prefix and bullet point)
                var_name = line.replace('•', '').strip()
                if var_name.startswith('Y_'):
                    var_name = var_name[2:]  # Remove Y_ prefix
                
                if current_section == 'scalar':
                    variables['scalar'].append(var_name)
                elif current_section == 'pft1d':
                    variables['pft1d'].append(var_name)
                elif current_section == 'soil2d':
                    variables['soil2d'].append(var_name)
        
        print(f"  Scalar variables: {len(variables['scalar'])}")
        print(f"  PFT1D variables: {len(variables['pft1d'])}")
        print(f"  Soil2D variables: {len(variables['soil2d'])}")
        
        return variables
        
    except Exception as e:
        print(f"Warning: Could not parse variable list file: {e}")
        # Return default variables if parsing fails
        return {
            'scalar': ['GPP', 'NPP', 'AR', 'HR'],
            'pft1d': ['deadcrootc', 'deadcrootn', 'deadcrootp', 'deadstemc', 'deadstemn', 'deadstemp', 'frootc', 'frootc_storage', 'leafc', 'leafc_storage', 'totcolp', 'totlitc', 'totvegc', 'tlai'],
            'soil2d': ['cwdc_vr', 'cwdn_vr', 'cwdp_vr', 'litr1c_vr', 'litr2c_vr', 'litr3c_vr', 'litr1n_vr', 'litr2n_vr', 'litr3n_vr', 'litr1p_vr', 'litr2p_vr', 'litr3p_vr', 'sminn_vr', 'smin_no3_vr', 'smin_nh4_vr', 'soil1c_vr', 'soil1n_vr', 'soil1p_vr', 'soil2c_vr', 'soil2n_vr', 'soil2p_vr', 'soil3c_vr', 'soil3n_vr', 'soil3p_vr', 'soil4c_vr', 'soil4n_vr', 'soil4p_vr']
        }


def load_numerical_model_data(nc_file: str, variable_list: Dict[str, List[str]]) -> Dict[str, Any]:
    """Load numerical model data from NetCDF file based on variable list."""
    print(f"Loading numerical model data from: {nc_file}")
    
    data = {}
    with Dataset(nc_file, 'r') as ds:
        # Load grid information
        if 'grid1d_lon' in ds.variables and 'grid1d_lat' in ds.variables:
            data['grid_lon'] = ds.variables['grid1d_lon'][:]
            data['grid_lat'] = ds.variables['grid1d_lat'][:]
            data['grid_ncells'] = len(data['grid_lon'])
        
        # Load PFT information
        if 'pfts1d_lon' in ds.variables and 'pfts1d_lat' in ds.variables:
            data['pft_lon'] = ds.variables['pfts1d_lon'][:]
            data['pft_lat'] = ds.variables['pfts1d_lat'][:]
            data['pft_gridcell_index'] = ds.variables['pfts1d_gridcell_index'][:]
            data['pft_ncells'] = len(data['pft_lon'])
        
        # Load column information
        if 'cols1d_lon' in ds.variables and 'cols1d_lat' in ds.variables:
            data['col_lon'] = ds.variables['cols1d_lat'][:]
            data['col_lat'] = ds.variables['cols1d_lat'][:]
            data['col_gridcell_index'] = ds.variables['cols1d_gridcell_index'][:]
            data['col_ncells'] = len(data['col_lon'])
        
        # Load variables from the parsed variable list
        all_vars = variable_list['scalar'] + variable_list['pft1d'] + variable_list['soil2d']
        
        for var_name in all_vars:
            if var_name in ds.variables:
                var_data = ds.variables[var_name][:]
                # Store the full variable data for proper comparison
                data[var_name] = var_data
                print(f"  Loaded {var_name}: {var_data.shape}")
    
    print(f"Numerical model data loaded: {len(data)} variables")
    return data


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
            var_name = p.stem.replace('predictions_', '')
            preds['pft_1d'][var_name] = pd.read_csv(p)
            print(f"  Loaded PFT predictions for {var_name}: {preds['pft_1d'][var_name].shape}")
    
    # Load 2D soil predictions
    soil_dir = predictions_dir / 'soil_2d_predictions'
    if soil_dir.exists():
        preds['soil_2d'] = {}
        for p in sorted(soil_dir.glob('predictions_*.csv')):
            var_name = p.stem.replace('predictions_', '')
            preds['soil_2d'][var_name] = pd.read_csv(p)
            print(f"  Loaded soil predictions for {var_name}: {preds['soil_2d'][var_name].shape}")
    
    # Load static inverse mapping
    static_inv = predictions_dir / 'test_static_inverse.csv'
    if static_inv.exists():
        preds['test_static_inverse'] = pd.read_csv(static_inv)
        print(f"  Loaded static inverse mapping: {preds['test_static_inverse'].shape}")
    
    return preds


def create_spatial_mapping(nc_data: Dict[str, Any], ai_preds: Dict[str, Any]) -> Dict[str, Any]:
    """Create spatial mapping between numerical model grid and AI predictions."""
    print("Creating spatial mapping...")
    
    mapping = {}
    
    # Grid-level mapping (for scalar variables)
    if 'grid_lon' in nc_data and 'test_static_inverse' in ai_preds:
        nc_lon = nc_data['grid_lon']
        nc_lat = nc_data['grid_lat']
        ai_lon = ai_preds['test_static_inverse']['Longitude'].values
        ai_lat = ai_preds['test_static_inverse']['Latitude'].values
        
        # Create KDTree for nearest neighbor mapping
        tree = cKDTree(np.column_stack([nc_lon, nc_lat]))
        dists, idxs = tree.query(np.column_stack([ai_lon, ai_lat]), k=1)
        
        mapping['grid'] = {
            'ai_to_nc': idxs,
            'distances': dists,
            'nc_lon': nc_lon,
            'nc_lat': nc_lat,
            'ai_lon': ai_lon,
            'ai_lat': ai_lat
        }
        print(f"  Grid mapping: {len(idxs)} AI points mapped to {len(nc_lon)} NC points")
    
    # PFT-level mapping
    if 'pft_gridcell_index' in nc_data:
        pft_indices = nc_data['pft_gridcell_index']
        # Group PFTs by gridcell
        from collections import defaultdict
        grid_to_pfts = defaultdict(list)
        for pft_i, grid_i in enumerate(pft_indices):
            grid_to_pfts[int(grid_i - 1)].append(pft_i)
        
        mapping['pft'] = {
            'grid_to_pfts': dict(grid_to_pfts),
            'pft_gridcell_index': pft_indices
        }
        print(f"  PFT mapping: {len(grid_to_pfts)} gridcells with PFTs")
    
    # Column-level mapping
    if 'col_gridcell_index' in nc_data:
        col_indices = nc_data['col_gridcell_index']
        from collections import defaultdict
        grid_to_cols = defaultdict(list)
        for col_i, grid_i in enumerate(col_indices):
            grid_to_cols[int(grid_i - 1)].append(col_i)
        
        mapping['column'] = {
            'grid_to_cols': dict(grid_to_cols),
            'col_gridcell_index': col_indices
        }
        print(f"  Column mapping: {len(grid_to_cols)} gridcells with columns")
        print(f"  Note: AI predictions use only the first column (index 0) of each gridcell")
    
    return mapping


def compare_variables(nc_data: Dict[str, Any], ai_preds: Dict[str, Any], 
                     mapping: Dict[str, Any], variable_list: Dict[str, List[str]]) -> Dict[str, Any]:
    """Compare AI predictions with numerical model data based on parsed variable list."""
    print("Comparing variables...")
    
    comparisons = {}
    
    # Compare scalar variables
    if 'scalar' in ai_preds and 'grid' in mapping:
        scalar_comps = {}
        for col in ai_preds['scalar'].columns:
            var_name = col.replace('Y_', '')
            if var_name in nc_data and var_name in variable_list['scalar']:
                ai_vals = ai_preds['scalar'][col].values
                nc_vals = nc_data[var_name][mapping['grid']['ai_to_nc']]
                
                # Calculate statistics
                stats_dict = calculate_comparison_stats(ai_vals, nc_vals)
                scalar_comps[var_name] = {
                    'ai_values': ai_vals,
                    'nc_values': nc_vals,
                    'statistics': stats_dict
                }
                print(f"  Scalar {var_name}: RMSE={stats_dict['rmse']:.4f}, R²={stats_dict['r2']:.4f}")
        
        comparisons['scalar'] = scalar_comps
    
    # Compare PFT variables
    if 'pft_1d' in ai_preds and 'pft' in mapping:
        pft_comps = {}
        print(f"  Comparing {len(ai_preds['pft_1d'])} PFT variables...")
        for var_name, df in ai_preds['pft_1d'].items():
            nc_var_name = var_name.replace('Y_', '')
            if nc_var_name in nc_data and nc_var_name in variable_list['pft1d']:
                print(f"    Comparing {var_name} (AI shape: {df.shape}, NC shape: {nc_data[nc_var_name].shape})")
                pft_comps[var_name] = compare_pft_variables(
                    df, nc_data[nc_var_name], mapping['pft'], var_name
                )
                print(f"      Created {len(pft_comps[var_name])} PFT comparisons")
            else:
                print(f"    Skipping {var_name}: not found in NC data or variable list")
        
        comparisons['pft_1d'] = pft_comps
    
    # Compare soil variables
    if 'soil_2d' in ai_preds and 'column' in mapping:
        soil_comps = {}
        print(f"  Comparing {len(ai_preds['soil_2d'])} 2D soil variables...")
        for var_name, df in ai_preds['soil_2d'].items():
            nc_var_name = var_name.replace('Y_', '')
            if nc_var_name in nc_data and nc_var_name in variable_list['soil2d']:
                print(f"    Comparing {var_name} (AI shape: {df.shape}, NC shape: {nc_data[nc_var_name].shape})")
                soil_comps[var_name] = compare_soil_variables(
                    df, nc_data[nc_var_name], mapping['column'], var_name
                )
                print(f"      Created {len(soil_comps[var_name])} layer comparisons")
            else:
                print(f"    Skipping {var_name}: not found in NC data or variable list")
        
        comparisons['soil_2d'] = soil_comps
    
    return comparisons


def calculate_comparison_stats(ai_vals: np.ndarray, nc_vals: np.ndarray) -> Dict[str, float]:
    """Calculate comparison statistics between AI and numerical model values."""
    # Remove NaN values
    mask = ~(np.isnan(ai_vals) | np.isnan(nc_vals))
    if np.sum(mask) == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'correlation': np.nan}
    
    ai_clean = ai_vals[mask]
    nc_clean = nc_vals[mask]
    
    if len(ai_clean) == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'correlation': np.nan}
    
    # Calculate statistics
    rmse = np.sqrt(np.mean((ai_clean - nc_clean) ** 2))
    mae = np.mean(np.abs(ai_clean - nc_clean))
    
    # Correlation and R²
    if len(ai_clean) > 1:
        correlation = np.corrcoef(ai_clean, nc_clean)[0, 1]
        # R² calculation
        ss_res = np.sum((ai_clean - nc_clean) ** 2)
        ss_tot = np.sum((nc_clean - np.mean(nc_clean)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    else:
        correlation = np.nan
        r2 = np.nan
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'correlation': correlation,
        'n_samples': len(ai_clean)
    }


def compare_pft_variables(df: pd.DataFrame, nc_data: np.ndarray, 
                         pft_mapping: Dict, var_name: str) -> Dict[str, Any]:
    """Compare PFT variables between AI predictions and numerical model."""
    # Get PFT columns (e.g., Y_leafc_pft1, Y_leafc_pft2, etc.)
    pft_cols = [c for c in df.columns if c.startswith(var_name + '_pft')]
    if not pft_cols:
        return {}
    
    # Sort PFT columns numerically
    def pft_num(col):
        try:
            return int(col.split('_pft')[-1])
        except:
            return 999
    
    pft_cols = sorted(pft_cols, key=pft_num)
    
    comparisons = {}
    for i, col in enumerate(pft_cols):
        if i < len(pft_cols):
            ai_vals = df[col].values
            # AI predictions contain PFT1-16 (excluding PFT0) for the first column of each gridcell
            # Numerical model has PFT0-16, so we need to map AI PFT1-16 to NC PFT1-16 (skip PFT0)
            if len(ai_vals) <= len(nc_data):
                # For PFT variables, take the first column (index 0) and skip PFT0
                # AI PFT1 corresponds to NC PFT1 (index 1), AI PFT2 to NC PFT2 (index 2), etc.
                pft_idx = i + 1  # AI PFT indices are 1-16, map to NC PFT indices 1-16
                if pft_idx <= nc_data.shape[0]:  # Check if PFT index exists in NC data
                    nc_vals = nc_data[pft_idx-1, :len(ai_vals)]  # PFT index, all samples
                    stats = calculate_comparison_stats(ai_vals, nc_vals)
                    comparisons[col] = {
                        'ai_values': ai_vals,
                        'nc_values': nc_vals,
                        'statistics': stats
                    }
    
    return comparisons


def compare_soil_variables(df: pd.DataFrame, nc_data: np.ndarray, 
                          col_mapping: Dict, var_name: str) -> Dict[str, Any]:
    """Compare soil variables between AI predictions and numerical model."""
    # AI predictions contain 10 layers for the first column of each gridcell
    # Numerical model has 15 layers, so we compare AI layers 1-10 with NC layers 1-10
    
    # Get all columns for this variable
    all_cols = [c for c in df.columns if c.startswith(var_name)]
    if not all_cols:
        return {}
    
    # AI predictions have 10 layers × 1 column = 10 columns
    # Each column represents a layer (layer1, layer2, ..., layer10)
    layer_cols = []
    for i in range(1, 11):  # Layers 1-10
        layer_col = f"{var_name}_layer{i}"
        if layer_col in df.columns:
            layer_cols.append(layer_col)
    
    if not layer_cols:
        return {}
    
    comparisons = {}
    
    # Compare each layer (1-10) between AI predictions and numerical model
    for i, layer_col in enumerate(layer_cols):
        ai_vals = df[layer_col].values
        
        # AI layer i+1 corresponds to NC layer i+1 (both start from layer 1)
        # NC data has shape (layers, samples) or (samples, layers)
        if nc_data.ndim == 2:
            if nc_data.shape[0] == 15:  # (layers, samples)
                nc_vals = nc_data[i, :len(ai_vals)]  # Layer i, all samples
            else:  # (samples, layers)
                nc_vals = nc_data[:len(ai_vals), i]  # All samples, layer i
        else:
            # 1D array, assume it's flattened
            nc_vals = nc_data[:len(ai_vals)]
        
        stats = calculate_comparison_stats(ai_vals, nc_vals)
        comparisons[layer_col] = {
            'ai_values': ai_vals,
            'nc_values': nc_vals,
            'statistics': stats
        }
    
    return comparisons


def print_data_structure_summary(nc_data: Dict[str, Any], ai_preds: Dict[str, Any]) -> None:
    """Print a summary of the data structure differences between AI predictions and numerical model."""
    print("\n" + "="*60)
    print("DATA STRUCTURE SUMMARY")
    print("="*60)
    
    print("Numerical Model (NetCDF):")
    for var_name, var_data in nc_data.items():
        if isinstance(var_data, np.ndarray):
            print(f"  {var_name}: {var_data.shape}")
    
    print("\nAI Predictions:")
    if 'scalar' in ai_preds:
        print(f"  Scalar: {ai_preds['scalar'].shape}")
    
    if 'pft_1d' in ai_preds:
        print(f"  PFT 1D: {len(ai_preds['pft_1d'])} variables")
        for var_name, df in ai_preds['pft_1d'].items():
            print(f"    {var_name}: {df.shape} (PFT1-16, first column only)")
    
    if 'soil_2d' in ai_preds:
        print(f"  Soil 2D: {len(ai_preds['soil_2d'])} variables")
        for var_name, df in ai_preds['soil_2d'].items():
            print(f"    {var_name}: {df.shape} (10 layers, first column only)")
    
    print("\nKey Differences:")
    print("  - AI PFT predictions: PFT1-16 only (excludes PFT0), first column of each gridcell")
    print("  - AI Soil predictions: 10 layers only (vs 15 in model), first column of each gridcell")
    print("  - Numerical model: Full PFT0-16, full 15 layers, all columns")
    print("="*60)


def create_comparison_plots(comparisons: Dict[str, Any], mapping: Dict[str, Any], 
                           output_dir: Path) -> None:
    """Create comparison plots and maps."""
    print(f"Creating comparison plots in: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    plots_created = 0
    
    # 1. Scalar variable comparison plots
    if 'scalar' in comparisons and comparisons['scalar']:
        try:
            create_scalar_comparison_plots(comparisons['scalar'], output_dir)
            plots_created += 1
            print("  ✓ Scalar comparison plots created")
        except Exception as e:
            print(f"  ✗ Error creating scalar plots: {e}")
    
    # 2. Spatial distribution maps
    if 'grid' in mapping:
        try:
            create_spatial_maps(comparisons, mapping, output_dir)
            plots_created += 1
            print("  ✓ Spatial distribution map created")
        except Exception as e:
            print(f"  ✗ Error creating spatial map: {e}")
    
    # 3. Statistical summary plots
    try:
        create_statistical_summary(comparisons, output_dir)
        plots_created += 1
        print("  ✓ Statistical summary plots created")
    except Exception as e:
        print(f"  ✗ Error creating statistical plots: {e}")
    
    if plots_created > 0:
        print(f"Comparison plots created successfully! ({plots_created} plot types)")
    else:
        print("Warning: No comparison plots were created due to errors or missing data.")


def create_scalar_comparison_plots(scalar_comps: Dict[str, Any], output_dir: Path) -> None:
    """Create scatter plots for scalar variable comparisons."""
    n_vars = len(scalar_comps)
    if n_vars == 0:
        return
    
    # Calculate grid size
    cols = min(3, n_vars)
    rows = (n_vars + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_vars == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.flatten()
    
    for i, (var_name, comp_data) in enumerate(scalar_comps.items()):
        if i >= len(axes):
            break
        
        ax = axes[i]
        ai_vals = comp_data['ai_values']
        nc_vals = comp_data['nc_values']
        stats = comp_data['statistics']
        
        # Scatter plot
        ax.scatter(nc_vals, ai_vals, alpha=0.6, s=20)
        
        # Perfect correlation line
        min_val = min(np.min(ai_vals), np.min(nc_vals))
        max_val = max(np.max(ai_vals), np.max(nc_vals))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        # Labels and title
        ax.set_xlabel('Numerical Model')
        ax.set_ylabel('AI Prediction')
        ax.set_title(f'{var_name}\nRMSE: {stats["rmse"]:.4f}, R²: {stats["r2"]:.4f}')
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scalar_comparisons.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_spatial_maps(comparisons: Dict[str, Any], mapping: Dict[str, Any], 
                       output_dir: Path) -> None:
    """Create spatial distribution maps for comparisons."""
    if 'grid' not in mapping:
        return
    
    grid_mapping = mapping['grid']
    nc_lon = grid_mapping['nc_lon']
    nc_lat = grid_mapping['nc_lat']
    
    # Create a simple map showing the grid structure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot numerical model grid
    scatter = ax.scatter(nc_lon, nc_lat, c='blue', s=1, alpha=0.6, label='Numerical Model Grid')
    
    # Plot AI prediction points
    ai_lon = grid_mapping['ai_lon']
    ai_lat = grid_mapping['ai_lat']
    ax.scatter(ai_lon, ai_lat, c='red', s=10, alpha=0.8, label='AI Prediction Points')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Spatial Distribution: Numerical Model vs AI Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spatial_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_statistical_summary(comparisons: Dict[str, Any], output_dir: Path) -> None:
    """Create statistical summary plots."""
    # Collect all statistics
    all_stats = []
    
    if 'scalar' in comparisons:
        for var_name, comp_data in comparisons['scalar'].items():
            stats = comp_data['statistics']
            all_stats.append({
                'variable': var_name,
                'type': 'scalar',
                'rmse': stats['rmse'],
                'mae': stats['mae'],
                'r2': stats['r2'],
                'correlation': stats['correlation']
            })
    
    if 'pft_1d' in comparisons:
        for var_name, var_comps in comparisons['pft_1d'].items():
            for pft_name, comp_data in var_comps.items():
                stats = comp_data['statistics']
                all_stats.append({
                    'variable': f"{var_name}_{pft_name}",
                    'type': 'pft_1d',
                    'rmse': stats['rmse'],
                    'mae': stats['mae'],
                    'r2': stats['r2'],
                    'correlation': stats['correlation']
                })
    
    if 'soil_2d' in comparisons:
        for var_name, var_comps in comparisons['soil_2d'].items():
            for soil_name, comp_data in var_comps.items():
                stats = comp_data['statistics']
                all_stats.append({
                    'variable': f"{var_name}_{soil_name}",
                    'type': 'soil_2d',
                    'rmse': stats['rmse'],
                    'mae': stats['mae'],
                    'r2': stats['r2'],
                    'correlation': stats['correlation']
                })
    
    if not all_stats:
        return
    
    # Create DataFrame for plotting
    stats_df = pd.DataFrame(all_stats)
    
    # RMSE comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # RMSE by variable type
    sns.boxplot(data=stats_df, x='type', y='rmse', ax=axes[0, 0])
    axes[0, 0].set_title('RMSE by Variable Type')
    axes[0, 0].set_ylabel('RMSE')
    
    # R² by variable type
    sns.boxplot(data=stats_df, x='type', y='r2', ax=axes[0, 1])
    axes[0, 1].set_title('R² by Variable Type')
    axes[0, 1].set_ylabel('R²')
    
    # RMSE distribution
    axes[1, 0].hist(stats_df['rmse'].dropna(), bins=20, alpha=0.7)
    axes[1, 0].set_title('RMSE Distribution')
    axes[1, 0].set_xlabel('RMSE')
    axes[1, 0].set_ylabel('Frequency')
    
    # R² distribution
    axes[1, 1].hist(stats_df['r2'].dropna(), bins=20, alpha=0.7)
    axes[1, 1].set_title('R² Distribution')
    axes[1, 1].set_xlabel('R²')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'statistical_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save statistics to CSV
    stats_df.to_csv(output_dir / 'comparison_statistics.csv', index=False)
    print(f"Statistics saved to: {output_dir / 'comparison_statistics.csv'}")


def get_user_input_variable_list() -> str:
    """Ask user to input the path to the variable list file."""
    print("\n" + "="*60)
    print("Variable List File Input")
    print("="*60)
    print("Please provide the path to your CNP IO list file (e.g., CNP_IO_test1.txt)")
    print("This file should contain sections for SCALAR VARIABLES, 1D PFT VARIABLES, and 2D VARIABLES")
    
    while True:
        user_input = input("\nEnter the path to your variable list file: ").strip()
        
        if not user_input:
            print("Please enter a valid file path.")
            continue
        
        # Check if file exists
        if Path(user_input).exists():
            return user_input
        else:
            print(f"File not found: {user_input}")
            print("Please check the path and try again.")


def main():
    examples = textwrap.dedent(
        """
        Examples:
          # Compare AI predictions with numerical model data (using default paths)
          python scripts/inference_map.py \
            --variable-list CNP_IO_list1.txt \
            --output-dir comparison_results

          # Use custom AI predictions path
          python scripts/inference_map.py \
            --variable-list CNP_IO_list1.txt \
            --ai-predictions cnp_results/run_20250807_215454/cnp_predictions \
            --output-dir comparison_results

          # Use custom numerical model path
          python scripts/inference_map.py \
            --variable-list CNP_IO_list1.txt \
            --numerical-model /path/to/your/model.nc \
            --output-dir comparison_results

          # Interactive mode (will prompt for variable list file, uses default paths)
          python scripts/inference_map.py
        """
    )

    p = argparse.ArgumentParser(
        description='Compare AI model inference results with numerical model data from NetCDF files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples
    )
    p.add_argument('--variable-list', help='Path to CNP_IO_list file (optional if using interactive mode)')
    p.add_argument('--ai-predictions', default='./cnp_inference_entire_dataset/cnp_predictions', help='Path to AI predictions directory (cnp_predictions) [default: ./cnp_inference_entire_dataset/cnp_predictions]')
    p.add_argument('--numerical-model', default='/mnt/proj-shared/AI4BGC_7xw/AI4BGC/ELM_data/original_780_spinup_from_modelsimulation.nc', help='Path to numerical model NetCDF file [default: /mnt/proj-shared/AI4BGC_7xw/AI4BGC/ELM_data/original_780_spinup_from_modelsimulation.nc]')
    p.add_argument('--output-dir', default='comparison_results', help='Output directory for comparison results')
    p.add_argument('--examples', action='store_true', help='Show example usage and exit')
    args = p.parse_args()

    if args.examples:
        print(p.format_help())
        sys.exit(0)

    # Validate inputs
    if not Path(args.ai_predictions).exists():
        p.error(f'AI predictions directory not found: {args.ai_predictions}')
    
    if not Path(args.numerical_model).exists():
        p.error(f'Numerical model file not found: {args.numerical_model}')
    
    # Handle variable list - either from command line or interactive input
    if args.variable_list:
        if not Path(args.variable_list).exists():
            p.error(f'Variable list file not found: {args.variable_list}')
        variable_list_path = args.variable_list
    else:
        print("No variable list file provided. Switching to interactive mode...")
        variable_list_path = get_user_input_variable_list()
        print(f"Using variable list file: {variable_list_path}")

    # Load data
    print("=" * 60)
    print("AI vs Numerical Model Comparison")
    print("=" * 60)
    print(f"Numerical model: {args.numerical_model}")
    print(f"AI predictions: {args.ai_predictions}")
    print(f"Variable list: {variable_list_path}")
    print("=" * 60)
    
    # Parse variable list file first
    variable_list = parse_variable_list_file(variable_list_path)
    
    # Display summary of variables to be compared
    print("\n" + "="*60)
    print("Variables to be compared:")
    print("="*60)
    print(f"Scalar variables ({len(variable_list['scalar'])}): {', '.join(variable_list['scalar'][:5])}{'...' if len(variable_list['scalar']) > 5 else ''}")
    print(f"PFT1D variables ({len(variable_list['pft1d'])}): {', '.join(variable_list['pft1d'][:5])}{'...' if len(variable_list['pft1d']) > 5 else ''}")
    print(f"Soil2D variables ({len(variable_list['soil2d'])}): {', '.join(variable_list['soil2d'][:5])}{'...' if len(variable_list['soil2d']) > 5 else ''}")
    print("="*60)
    
    # Load numerical model data based on parsed variable list
    nc_data = load_numerical_model_data(args.numerical_model, variable_list)
    ai_preds = load_ai_predictions(Path(args.ai_predictions))
    
    # Print data structure summary
    print_data_structure_summary(nc_data, ai_preds)
    
    # Create spatial mapping
    mapping = create_spatial_mapping(nc_data, ai_preds)
    
    # Compare variables based on parsed variable list
    comparisons = compare_variables(nc_data, ai_preds, mapping, variable_list)
    
    # Check if we have any comparisons
    total_comparisons = sum(len(comp) if isinstance(comp, dict) else 0 for comp in comparisons.values())
    if total_comparisons == 0:
        print("\nWarning: No variables could be compared between AI predictions and numerical model data.")
        print("This might be due to:")
        print("  - Missing variables in the numerical model NetCDF file")
        print("  - Mismatched variable names between AI predictions and numerical model")
        print("  - Missing spatial mapping data")
        print("\nPlease check your data and variable list file.")
        return
    
    print(f"\nSuccessfully compared {total_comparisons} variables!")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    
    # Create comparison plots
    create_comparison_plots(comparisons, mapping, output_dir)
    
    # Save comparison data
    comparison_file = output_dir / 'comparison_data.json'
    try:
        # Convert numpy arrays to lists for JSON serialization
        serializable_comparisons = {}
        for comp_type, comp_data in comparisons.items():
            serializable_comparisons[comp_type] = {}
            for var_name, var_data in comp_data.items():
                serializable_comparisons[comp_type][var_name] = {}
                for key, value in var_data.items():
                    if key == 'statistics':
                        serializable_comparisons[comp_type][var_name][key] = value
                    else:
                        # Convert numpy arrays to lists
                        if isinstance(value, np.ndarray):
                            serializable_comparisons[comp_type][var_name][key] = value.tolist()
                        else:
                            serializable_comparisons[comp_type][var_name][key] = value
        
        with open(comparison_file, 'w') as f:
            json.dump(serializable_comparisons, f, indent=2, default=str)
        print(f"Comparison data saved to: {comparison_file}")
    except Exception as e:
        print(f"Warning: Could not save comparison data: {e}")
    
    print(f"\nComparison complete! Results saved to: {output_dir}")
    print(f"Check the following files:")
    print(f"  - {output_dir / 'scalar_comparisons.png'}")
    print(f"  - {output_dir / 'spatial_distribution.png'}")
    print(f"  - {output_dir / 'statistical_summary.png'}")
    print(f"  - {output_dir / 'comparison_statistics.csv'}")
    print(f"  - {output_dir / 'comparison_data.json'}")


if __name__ == '__main__':
    main()
