#!/usr/bin/env python3
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, ListedColormap, BoundaryNorm
import argparse
from pathlib import Path
import sys
from typing import List

# Project imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.training_config import parse_cnp_io_list

# Default paths
DEFAULT_AI_PREDICTIONS = './comparison_results/ai_predictions_for_plotting.nc'
DEFAULT_MODEL = '/mnt/proj-shared/AI4BGC_7xw/AI4BGC/ELM_data/original_780_spinup_from_modelsimulation.nc'
DEFAULT_OUTPUT_DIR = "./ai_model_comparison_plots"

# Default variables to plot (will be overridden by variable list file)
VARIABLES = ['cwdc_vr', 'tlai']  # Fallback variables if no variable list provided

# Layers for column-type variables (AI has 10 layers, model has 15)
LEVGRND_LAYERS = [0, 4, 9]  # Layers 0, 4, 9 (corresponding to AI layers 1, 5, 10)

# PFTs to plot (AI has PFT1-16, model has PFT0-16)
PFT_PICK_LIST = [1, 2, 3, 4, 5]  # PFT1, PFT2, PFT3, PFT4, PFT5

def _safe_get(ds, name):
    """Safely get a variable from dataset, with error handling."""
    if name not in ds:
        raise KeyError(f"Missing required variable/coordinate: {name}")
    return ds[name]

def _to_nan_fillvalue(arr, fill_threshold=1e35):
    """Convert fill values to NaN."""
    a = np.asarray(arr, dtype=float)
    a[np.abs(a) >= fill_threshold] = np.nan
    return a

def _gridcell_lonlat(ds):
    """Extract gridcell longitude and latitude."""
    lon = _safe_get(ds, "grid1d_lon").values
    lat = _safe_get(ds, "grid1d_lat").values
    return lon, lat

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

def _plot_map(ax, lon, lat, data, title, vmin=None, vmax=None, cmap="viridis", norm=None):
    """Plot a map with the given data."""
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.OCEAN, color="lightblue", alpha=0.5)
    ax.add_feature(cfeature.LAND, color="lightgray", alpha=0.3)
    im = ax.scatter(lon, lat, c=data, s=5, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, transform=ccrs.PlateCarree())
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_global()
    gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    plt.colorbar(im, ax=ax, shrink=0.7, pad=0.05)

def _percent_diff_categories(model_vals: np.ndarray, ai_vals: np.ndarray) -> np.ndarray:
    """Create percent difference categories for visualization."""
    model = np.asarray(model_vals, dtype=float)
    ai = np.asarray(ai_vals, dtype=float)
    cat = np.full(model.shape, np.nan, dtype=float)
    finite = np.isfinite(model) & np.isfinite(ai)
    if not np.any(finite):
        return cat
    
    # Threshold: zero-out percentage where |model| <= 2% of mean(|model|)
    mean_abs_model = np.nanmean(np.abs(model[finite])) if np.any(finite) else 0.0
    threshold = 0.02 * mean_abs_model
    denom = np.abs(model[finite])
    
    # Compute percent (AI - model)/model in %
    pct = (ai[finite] - model[finite]) / np.where(denom > 0, denom, 1.0) * 100.0
    
    # Apply threshold rule
    pct[denom <= threshold] = 0.0
    sign = np.sign(pct)
    mag = np.abs(pct)
    bins = np.zeros_like(mag, dtype=int)
    bins[(mag >= 0) & (mag < 5)] = 1
    bins[(mag >= 5) & (mag < 15)] = 2
    bins[(mag >= 15) & (mag < 30)] = 3
    bins[mag >= 30] = 4
    bins[(mag == 0)] = 0
    categories = sign.astype(int) * bins
    cat[finite] = categories.astype(float)
    return cat

def _plot_tripanel(var, label_suffix, lon, lat, data_ai, data_model, out_dir,
                   label_ai="AI Predictions", label_model="Model Results"):
    """Create a tri-panel comparison plot."""
    diff = data_ai - data_model
    vmin_orig = np.nanmin([np.nanmin(data_ai), np.nanmin(data_model)])
    vmax_orig = np.nanmax([np.nanmax(data_ai), np.nanmax(data_model)])
    diff_abs = np.nanmax(np.abs(diff))

    # Compute stats and metrics
    finite_ai = np.isfinite(data_ai)
    finite_model = np.isfinite(data_model)
    mask = finite_ai & finite_model
    n = int(mask.sum())

    def _nan_min(a):
        return float(np.nanmin(a)) if np.any(np.isfinite(a)) else float('nan')

    def _nan_max(a):
        return float(np.nanmax(a)) if np.any(np.isfinite(a)) else float('nan')

    sum_ai = float(np.nansum(data_ai))
    min_ai = _nan_min(data_ai)
    max_ai = _nan_max(data_ai)

    sum_model = float(np.nansum(data_model))
    min_model = _nan_min(data_model)
    max_model = _nan_max(data_model)

    if n > 0:
        y = data_ai[mask]
        yhat = data_model[mask]
        mse = float(np.mean((y - yhat) ** 2))
        rmse = float(np.sqrt(mse))
        mean_y = float(np.mean(y))
        nrmse = (rmse / mean_y) if mean_y != 0 else float('inf')
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
    else:
        rmse = float('nan')
        nrmse = float('nan')
        r2 = float('nan')

    print(f"Stats for {var}{label_suffix}:")
    print(f"  {label_ai}: sum={sum_ai:.6g} min={min_ai:.6g} max={max_ai:.6g}")
    print(f"  {label_model}: sum={sum_model:.6g} min={min_model:.6g} max={max_model:.6g}")
    print(f"  Metrics (AI vs Model): n={n} rmse={rmse:.6g} nrmse={nrmse:.6g} r2={r2:.6g}")

    fig = plt.figure(figsize=(12, 20))
    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.3)

    # Panel 1: AI Predictions
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    _plot_map(ax1, lon, lat, data_ai, f"{var} - {label_ai}", vmin=vmin_orig, vmax=vmax_orig, cmap="viridis")

    # Panel 2: Model Results
    ax2 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    _plot_map(ax2, lon, lat, data_model, f"{var} - {label_model}", vmin=vmin_orig, vmax=vmax_orig, cmap="viridis")

    # Panel 3: Difference (AI - Model)
    ax3 = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())
    if not np.isfinite(diff_abs) or diff_abs <= 0:
        msg = "No Difference" if diff_abs == 0 else "All NaN"
        ax3.text(0.5, 0.5, msg, ha="center", va="center", transform=ax3.transAxes,
                 fontsize=14, fontweight="bold", color="gray")
        ax3.set_title(f"{var} - Difference (AI - Model)", fontsize=14, fontweight="bold")
        ax3.set_global()
        gl = ax3.gridlines(draw_labels=True, alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
    else:
        norm = TwoSlopeNorm(vmin=-diff_abs, vcenter=0, vmax=diff_abs)
        _plot_map(ax3, lon, lat, diff, f"{var} - Difference (AI - Model)", cmap="RdBu_r", norm=norm)

    # Panel 4: Percent Difference Categories
    ax4 = fig.add_subplot(gs[3, 0], projection=ccrs.PlateCarree())
    # Treat model as reference, AI as comparison
    cat = _percent_diff_categories(data_model, data_ai)
    colors = [
        "#08519c",  # -4: 30%+
        "#3182bd",  # -3: 15–30%
        "#6baed6",  # -2: 5–15%
        "#c6dbef",  # -1: 0–5%
        "#bdbdbd",  #  0: 0
        "#fcbba1",  # +1: 0–5%
        "#fc9272",  # +2: 5–15%
        "#fb6a4a",  # +3: 15–30%
        "#cb181d",  # +4: 30%+
    ]
    cmap = ListedColormap(colors)
    boundaries = [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm_cat = BoundaryNorm(boundaries, cmap.N)

    ax4.add_feature(cfeature.COASTLINE)
    ax4.add_feature(cfeature.BORDERS)
    ax4.add_feature(cfeature.OCEAN, color="lightblue", alpha=0.5)
    ax4.add_feature(cfeature.LAND, color="lightgray", alpha=0.3)
    im4 = ax4.scatter(lon, lat, c=cat, s=5, cmap=cmap, norm=norm_cat, transform=ccrs.PlateCarree())
    ax4.set_title(f"{var} - Percent Diff bins ((AI-Model)/Model)", fontsize=14, fontweight="bold")
    ax4.set_global()
    gl4 = ax4.gridlines(draw_labels=True, alpha=0.5, linestyle="--")
    gl4.top_labels = False
    gl4.right_labels = False
    cbar = plt.colorbar(im4, ax=ax4, shrink=0.7, pad=0.05, ticks=[-4, -3, -2, -1, 0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels([
        "-30%+", "-15–30%", "-5–15%", "-0–5%", "0", "0–5%", "5–15%", "15–30%", "30%+"
    ])

    plt.suptitle(f"{var} {label_suffix}", fontsize=16, fontweight="bold", y=0.96)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{var}{label_suffix}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def parse_variable_list_file(variable_list_path: str) -> List[str]:
    """Parse the CNP IO list file to extract all variables."""
    print(f"Parsing variable list file: {variable_list_path}")
    
    try:
        parsed = parse_cnp_io_list(variable_list_path)
        
        # Extract all variables from all sections
        all_variables = []
        
        # Add scalar variables
        if 'scalar_variables' in parsed:
            all_variables.extend(parsed['scalar_variables'])
            print(f"  Found {len(parsed['scalar_variables'])} scalar variables")
        
        # Add PFT 1D variables
        if 'pft_1d_variables' in parsed:
            all_variables.extend(parsed['pft_1d_variables'])
            print(f"  Found {len(parsed['pft_1d_variables'])} PFT 1D variables")
        
        # Add soil 2D variables
        if 'variables_2d_soil' in parsed:
            all_variables.extend(parsed['variables_2d_soil'])
            print(f"  Found {len(parsed['variables_2d_soil'])} soil 2D variables")
        
        print(f"  Total variables to plot: {len(all_variables)}")
        print(f"  Variables: {all_variables}")
        
        return all_variables
        
    except Exception as e:
        print(f"Warning: Could not parse variable list file: {e}")
        print("  Falling back to default variables")
        return VARIABLES


def discover_common_variables(ds_ai, ds_model):
    """Discover variables that exist in both datasets."""
    ai_vars = set(ds_ai.data_vars.keys())
    model_vars = set(ds_model.data_vars.keys())
    common_vars = ai_vars.intersection(model_vars)
    
    # Filter out coordinate variables and metadata
    exclude_patterns = ['lon', 'lat', 'index', 'period', 'time', 'bnds']
    filtered_vars = []
    
    for var in common_vars:
        if not any(pattern in var.lower() for pattern in exclude_patterns):
            filtered_vars.append(var)
    
    return sorted(filtered_vars)

def main():
    parser = argparse.ArgumentParser(
        description='Compare AI predictions with model results using tri-panel plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  python ai_model_comparison_plot.py

  # Use variable list file to plot all variables
  python ai_model_comparison_plot.py \\
    --variable-list CNP_IO_demo1.txt

  # Specify custom paths
  python ai_model_comparison_plot.py \\
    --ai-predictions ./my_ai_predictions.nc \\
    --model ./my_model.nc \\
    --output-dir ./my_comparison_plots

  # Plot specific variables only
  python ai_model_comparison_plot.py --variables cwdc_vr tlai
        """
    )
    
    parser.add_argument('--ai-predictions', default=DEFAULT_AI_PREDICTIONS,
                       help=f'Path to AI predictions NetCDF file [default: {DEFAULT_AI_PREDICTIONS}]')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                       help=f'Path to model results NetCDF file [default: {DEFAULT_MODEL}]')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                       help=f'Output directory for plots [default: {DEFAULT_OUTPUT_DIR}]')
    parser.add_argument('--variable-list', type=str,
                       help='Path to CNP_IO_list file to extract all variables for plotting')
    parser.add_argument('--variables', nargs='*', default=VARIABLES,
                       help=f'Variables to plot [default: {VARIABLES}]')
    parser.add_argument('--layers', nargs='*', type=int, default=LEVGRND_LAYERS,
                       help=f'Layers to plot for column variables [default: {LEVGRND_LAYERS}]')
    parser.add_argument('--pfts', nargs='*', type=int, default=PFT_PICK_LIST,
                       help=f'PFTs to plot [default: {PFT_PICK_LIST}]')
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.ai_predictions).exists():
        raise FileNotFoundError(f"AI predictions file not found: {args.ai_predictions}")
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    print("="*60)
    print("AI vs Model Comparison")
    print("="*60)
    print(f"AI predictions: {args.ai_predictions}")
    print(f"Model results: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Variables to plot: {args.variables}")
    print(f"Layers to plot: {args.layers}")
    print(f"PFTs to plot: {args.pfts}")
    print("="*60)
    
    # Parse variable list file if provided
    if args.variable_list:
        if not Path(args.variable_list).exists():
            raise FileNotFoundError(f"Variable list file not found: {args.variable_list}")
        
        # Parse the variable list file to get all variables
        all_variables = parse_variable_list_file(args.variable_list)
        
        # Filter to only include variables that exist in both datasets
        ds_ai = xr.open_dataset(args.ai_predictions)
        ds_model = xr.open_dataset(args.model)
        
        ai_vars = set(ds_ai.data_vars.keys())
        model_vars = set(ds_model.data_vars.keys())
        available_vars = [var for var in all_variables if var in ai_vars and var in model_vars]
        
        if available_vars:
            args.variables = available_vars
            print(f"Using {len(available_vars)} variables from variable list: {available_vars}")
        else:
            print("Warning: No variables from variable list found in both datasets!")
            return
    else:
        # Open datasets
        ds_ai = xr.open_dataset(args.ai_predictions)
        ds_model = xr.open_dataset(args.model)
        
        # Discover common variables if none specified
        if not args.variables or args.variables == VARIABLES:
            discovered_vars = discover_common_variables(ds_ai, ds_model)
            if discovered_vars:
                args.variables = discovered_vars
                print(f"Discovered {len(discovered_vars)} common variables: {discovered_vars}")
            else:
                print("Warning: No common variables found between AI predictions and model!")
                return
    
    # Get grid information from the model file since that's where the variable data comes from
    # This ensures the column/PFT indices match the variable data dimensions
    grid_lon, grid_lat = _gridcell_lonlat(ds_model)
    n_grid = ds_model.sizes["gridcell"]
    
    # Build mappings from the model file
    col2grid = _to_zero_based_index(_safe_get(ds_model, "cols1d_gridcell_index").values, n_grid)
    pft2grid = _to_zero_based_index(_safe_get(ds_model, "pfts1d_gridcell_index").values, n_grid)
    
    grid_to_cols = _build_gridcell_groups(col2grid, n_grid)
    grid_to_pfts = _build_gridcell_groups(pft2grid, n_grid)
    
    print(f"Total gridcells: {n_grid} | total columns: {col2grid.size} | total pfts: {pft2grid.size}")
    print(f"Example: gridcell 0 -> columns {grid_to_cols[0][:5]}, pfts {grid_to_pfts[0][:5]}")
    
    # Create output directory
    if args.variable_list:
        # Use a more descriptive output directory name when using variable list
        var_list_name = Path(args.variable_list).stem
        output_dir = Path(args.output_dir) / f"comparison_{var_list_name}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    else:
        output_dir = Path(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # Update the output directory for the plotting function
    args.output_dir = str(output_dir)
    
    print(f"\nStart plotting: {len(args.variables)} variables")
    for var in args.variables:
        if (var not in ds_ai.data_vars) or (var not in ds_model.data_vars):
            print(f"Skip {var} (not found in both files)")
            continue

        da_ai = ds_ai[var]
        da_model = ds_model[var]
        dims = da_ai.dims
        print(f"\nVariable {var}, dims: {dims}")
        print(f"  AI shape: {da_ai.shape}")
        print(f"  Model shape: {da_model.shape}")

        if ("column" in dims) and ("levgrnd" in dims):
            # Handle column-type variables (e.g., soil variables)
            # Handle different dimension orders
            if len(dims) == 3:
                # If we have (column, levgrnd, gridcell) or similar, transpose to (column, levgrnd)
                if "gridcell" in dims:
                    da_ai_cl = da_ai.transpose("column", "levgrnd", ...)
                    da_model_cl = da_model.transpose("column", "levgrnd", ...)
                else:
                    da_ai_cl = da_ai.transpose("column", "levgrnd")
                    da_model_cl = da_model.transpose("column", "levgrnd")
            else:
                da_ai_cl = da_ai.transpose("column", "levgrnd")
                da_model_cl = da_model.transpose("column", "levgrnd")
            
            vals_ai = _to_nan_fillvalue(da_ai_cl.values)
            vals_model = _to_nan_fillvalue(da_model_cl.values)
            
            # Debug: Print data structure information
            print(f"  Column variable: AI shape {vals_ai.shape}, Model shape {vals_model.shape}")
            print(f"  Grid mapping: {len(grid_to_cols)} gridcells with columns")
            print(f"  Sample gridcell 0 has columns: {grid_to_cols[0][:5] if grid_to_cols[0] else 'none'}")

            for lev in args.layers:
                if lev < 0 or lev >= da_ai_cl.sizes["levgrnd"]:
                    print(f"  Layer {lev} out of range, skipped")
                    continue

                ai_grid = np.full(n_grid, np.nan, dtype=float)
                model_grid = np.full(n_grid, np.nan, dtype=float)

                for g in range(n_grid):
                    cols = grid_to_cols[g]
                    if len(cols) == 0:
                        continue
                    
                    # For AI data: since it only has 1 column, use column 0 for all gridcells
                    # The AI data represents the entire grid with a single column
                    ai_col_idx = 0
                    
                    # Handle AI data indexing - shape is (column, levgrnd, gridcell)
                    if vals_ai.ndim == 3:
                        ai_grid[g] = vals_ai[ai_col_idx, lev, g]
                    else:
                        ai_grid[g] = vals_ai[ai_col_idx, lev]
                    
                    # For model: use the first column of this gridcell
                    # The model has 253,641 columns, each corresponding to a specific location
                    if g < len(grid_to_cols) and len(grid_to_cols[g]) > 0:
                        model_col_idx = grid_to_cols[g][0]  # First column of this gridcell
                        if model_col_idx < vals_model.shape[0]:
                            # Handle model data indexing - shape is (column, levgrnd)
                            if vals_model.ndim == 2:
                                if lev < vals_model.shape[1]:
                                    model_grid[g] = vals_model[model_col_idx, lev]
                            else:
                                model_grid[g] = vals_model[model_col_idx]

                _plot_tripanel(var, f"_lev{lev}", grid_lon, grid_lat, ai_grid, model_grid, args.output_dir,
                               label_ai="AI Predictions", label_model="Model Results")

        elif ("pft" in dims):
            # Handle PFT-type variables
            # Handle different dimension orders
            if len(dims) == 2 and "gridcell" in dims:
                # If we have (pft, gridcell), transpose to (pft, ...)
                da_ai_p = da_ai.transpose("pft", ...)
                da_model_p = da_model.transpose("pft", ...)
            else:
                da_ai_p = da_ai.transpose("pft")
                da_model_p = da_model.transpose("pft")
            
            vals_ai = _to_nan_fillvalue(da_ai_p.values)
            vals_model = _to_nan_fillvalue(da_model_p.values)

            for k in args.pfts:
                if k >= vals_ai.shape[0]:
                    print(f"  PFT {k} out of range for AI data, skipped")
                    continue
                    
                ai_grid = np.full(n_grid, np.nan, dtype=float)
                model_grid = np.full(n_grid, np.nan, dtype=float)

                for g in range(n_grid):
                    pfts = grid_to_pfts[g]
                    if len(pfts) == 0:
                        continue
                    if k < len(pfts):
                        p_idx = pfts[k]
                        
                        # Handle AI data indexing - shape might be (pft, gridcell) or (pft,)
                        if vals_ai.ndim == 2:
                            ai_grid[g] = vals_ai[k, g]
                        else:
                            ai_grid[g] = vals_ai[k]
                        
                        # For model, use the same PFT index (k) as in restart_variable_plot.py
                        # The model data structure is (pft,) where PFT indices correspond to the same physical location
                        if k < vals_model.shape[0]:
                            # Handle model data indexing
                            if vals_model.ndim == 2:
                                model_grid[g] = vals_model[k, g]
                            else:
                                model_grid[g] = vals_model[k]

                _plot_tripanel(var, f"_pft{k+1}", grid_lon, grid_lat, ai_grid, model_grid, args.output_dir,
                               label_ai="AI Predictions", label_model="Model Results")

        elif "gridcell" in dims:
            # Handle gridcell-level variables (e.g., GPP, NPP)
            # Handle different dimension orders
            if len(dims) == 1:
                da_ai_gc = da_ai
                da_model_gc = da_model
            else:
                # If we have multiple dimensions including gridcell, transpose to put gridcell last
                da_ai_gc = da_ai.transpose(..., "gridcell")
                da_model_gc = da_model.transpose(..., "gridcell")
            
            vals_ai = _to_nan_fillvalue(da_ai_gc.values)
            vals_model = _to_nan_fillvalue(da_model_gc.values)
            
            # Ensure we have the same number of gridcells
            n_ai = len(vals_ai)
            n_model = len(vals_model)
            
            if n_ai != n_model:
                print(f"  Warning: Gridcell count mismatch - AI: {n_ai}, Model: {n_model}")
                n_compare = min(n_ai, n_model)
                vals_ai = vals_ai[:n_compare]
                vals_model = vals_model[:n_compare]
                grid_lon = grid_lon[:n_compare]
                grid_lat = grid_lat[:n_compare]
            
            _plot_tripanel(var, "", grid_lon, grid_lat, vals_ai, vals_model, args.output_dir,
                           label_ai="AI Predictions", label_model="Model Results")

        else:
            print(f"  Skip {var} (unsupported dimensions: {dims})")

    ds_ai.close()
    ds_model.close()
    print(f"\nAll plots done! Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()
