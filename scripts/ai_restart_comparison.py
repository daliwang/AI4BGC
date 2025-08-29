import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, ListedColormap, BoundaryNorm
import argparse
import glob
from pathlib import Path
import sys

# Project imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.training_config import parse_cnp_io_list

# Default file paths
DATA_DIR = '/mnt/proj-shared/AI4BGC_7xw/AI4BGC/ELM_data/'
DEFAULT_FILE_OLD = DATA_DIR + 'original_780_spinup_from_modelsimulation.nc'

# Default layers and PFTs to plot
LEVGRND_LAYERS = [0, 5, 9]  # layers for (column, levgrnd)-type variables (0,9)
PFT_PICK_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # PFT0-PFT15 (representing PFT1-PFT16)

LABEL_NEW = "AI Generated"
LABEL_OLD = "Original Model"
OUTPUT_DIR = "./ai_restart_comparison_plots"

def _safe_get(ds, name):
    if name not in ds:
        raise KeyError(f"Missing required variable/coordinate: {name}")
    return ds[name]

def _to_nan_fillvalue(arr, fill_threshold=1e35):
    a = np.asarray(arr, dtype=float)
    a[np.abs(a) >= fill_threshold] = np.nan
    return a

def _gridcell_lonlat(ds):
    lon = _safe_get(ds, "grid1d_lon").values
    lat = _safe_get(ds, "grid1d_lat").values
    return lon, lat

def _to_zero_based_index(idx_raw, n_grid):
    idx = np.asarray(idx_raw, dtype=np.int64).copy()
    if idx.size == 0:
        return np.full_like(idx, -1)
    is_one_based = (np.any(idx == n_grid) or (np.nanmin(idx) == 1))
    if is_one_based:
        idx = idx - 1
    idx[(idx < 0) | (idx >= n_grid)] = -1
    return idx

def _build_gridcell_groups(one_d_to_grid, n_grid):
    groups = [[] for _ in range(n_grid)]
    for idx, g in enumerate(one_d_to_grid):
        if 0 <= g < n_grid:
            groups[g].append(idx)
    return groups

def _plot_map(ax, lon, lat, data, title, vmin=None, vmax=None, cmap="viridis", norm=None):
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
    bins[(mag >= 0) & (mag < 10)] = 1
    bins[(mag >= 10) & (mag < 30)] = 2
    bins[mag >= 30] = 3
    bins[(mag == 0)] = 0
    categories = sign.astype(int) * bins
    cat[finite] = categories.astype(float)
    return cat

def _plot_tripanel(var, label_suffix, lon, lat, data_new, data_old, out_dir,
                   label_new="AI Enhanced", label_old="Original Model"):
    diff = data_new - data_old
    vmin_orig = np.nanmin([np.nanmin(data_new), np.nanmin(data_old)])
    vmax_orig = np.nanmax([np.nanmax(data_new), np.nanmax(data_old)])
    diff_abs = np.nanmax(np.abs(diff))

    # Compute stats and metrics (old vs new) with NaN safety
    finite_new = np.isfinite(data_new)
    finite_old = np.isfinite(data_old)
    mask = finite_new & finite_old
    n = int(mask.sum())

    def _nan_min(a):
        return float(np.nanmin(a)) if np.any(np.isfinite(a)) else float('nan')

    def _nan_max(a):
        return float(np.nanmax(a)) if np.any(np.isfinite(a)) else float('nan')

    sum_new = float(np.nansum(data_new))
    min_new = _nan_min(data_new)
    max_new = _nan_max(data_new)

    sum_old = float(np.nansum(data_old))
    min_old = _nan_min(data_old)
    max_old = _nan_max(data_old)

    if n > 0:
        y = data_new[mask]
        yhat = data_old[mask]
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
    print(f"  {label_new}: sum={sum_new:.6g} min={min_new:.6g} max={max_new:.6g}")
    print(f"  {label_old}: sum={sum_old:.6g} min={min_old:.6g} max={max_old:.6g}")
    print(f"  Metrics (AI vs Model): n={n} rmse={rmse:.6g} nrmse={nrmse:.6g} r2={r2:.6g}")

    fig = plt.figure(figsize=(12, 20))
    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    _plot_map(ax1, lon, lat, data_new, f"{var} - {label_new}", vmin=vmin_orig, vmax=vmax_orig, cmap="viridis")

    ax2 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    _plot_map(ax2, lon, lat, data_old, f"{var} - {label_old}", vmin=vmin_orig, vmax=vmax_orig, cmap="viridis")

    ax3 = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())
    if not np.isfinite(diff_abs) or diff_abs <= 0:
        msg = "No Difference" if diff_abs == 0 else "All NaN"
        ax3.text(0.5, 0.5, msg, ha="center", va="center", transform=ax3.transAxes,
                 fontsize=14, fontweight="bold", color="gray")
        ax3.set_title(f"{var} - Analysis", fontsize=14, fontweight="bold")
        ax3.set_global()
        gl = ax3.gridlines(draw_labels=True, alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
    else:
        norm = TwoSlopeNorm(vmin=-diff_abs, vcenter=0, vmax=diff_abs)
        _plot_map(ax3, lon, lat, diff, f"{var} - Diff ({label_new} - {label_old})", cmap="RdBu_r", norm=norm)

    # Percent-difference categorical map (fourth panel)
    ax4 = fig.add_subplot(gs[3, 0], projection=ccrs.PlateCarree())
    # Treat data_old as model, data_new as AI
    cat = _percent_diff_categories(data_old, data_new)
    colors = [
        "#08519c",  # -3: 30%+
        "#6baed6",  # -2: 10–30%
        "#c6dbef",  # -1: 0–10%
        "#bdbdbd",  #  0: 0
        "#fcbba1",  # +1: 0–10%
        "#fb6a4a",  # +2: 10–30%
        "#cb181d",  # +3: 30%+
    ]
    cmap = ListedColormap(colors)
    boundaries = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    norm_cat = BoundaryNorm(boundaries, cmap.N)

    ax4.add_feature(cfeature.COASTLINE)
    ax4.add_feature(cfeature.BORDERS)
    ax4.add_feature(cfeature.OCEAN, color="lightblue", alpha=0.5)
    ax4.add_feature(cfeature.LAND, color="lightgray", alpha=0.3)
    im4 = ax4.scatter(lon, lat, c=cat, s=5, cmap=cmap, norm=norm_cat, transform=ccrs.PlateCarree())
    ax4.set_title(f"{var} - Percent Diff bins (({label_new}-{label_old})/{label_old})", fontsize=14, fontweight="bold")
    ax4.set_global()
    gl4 = ax4.gridlines(draw_labels=True, alpha=0.5, linestyle="--")
    gl4.top_labels = False
    gl4.right_labels = False
    cbar = plt.colorbar(im4, ax=ax4, shrink=0.7, pad=0.05, ticks=[-3, -2, -1, 0, 1, 2, 3])
    cbar.ax.set_yticklabels([
        "-30%+", "-10–30%", "-0–10%", "0", "0–10%", "10–30%", "30%+"
    ])

    plt.suptitle(f"{var} {label_suffix}", fontsize=16, fontweight="bold", y=0.96)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{var}{label_suffix}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare AI-enhanced restart file with original model file')
    parser.add_argument('--variable-list', type=str, required=True,
                       help='Path to CNP_IO list file')
    parser.add_argument('--ai-restart', type=str, default=None,
                       help='Path to AI-enhanced restart file (auto-detected if not specified)')
    parser.add_argument('--original-restart', type=str, default=None,
                       help='Path to original model restart file (default: 2025 model)')
    parser.add_argument('--layers', type=str, default='0,5,9',
                       help='Comma-separated list of soil layers to plot (default: 0,5,9)')
    parser.add_argument('--pfts', type=str, default='1,2,4,5,6,7,8,9',
                       help='Comma-separated list of PFTs to plot (default: all PFT0-PFT15)')
    
    return parser.parse_args()

def find_ai_restart_file():
    """Find AI-enhanced restart file in current directory."""
    current_dir = Path('.')
    pattern = '*updated_restart_CNP_IO*'
    matching_files = list(current_dir.glob(pattern))
    
    if matching_files:
        # Return the most recent file
        return str(sorted(matching_files, key=lambda x: x.stat().st_mtime)[-1])
    else:
        return None

def main():
    args = parse_arguments()
    
    # Parse layers and PFTs
    global LEVGRND_LAYERS, PFT_PICK_LIST
    LEVGRND_LAYERS = [int(x.strip()) for x in args.layers.split(',')]
    PFT_PICK_LIST = [int(x.strip()) for x in args.pfts.split(',')]
    
    # Set file paths
    if args.ai_restart:
        FILE_NEW = args.ai_restart
    else:
        FILE_NEW = find_ai_restart_file()
        if not FILE_NEW:
            print("Error: No AI-enhanced restart file found. Please specify with --ai-restart")
            return
    
    if args.original_restart:
        FILE_OLD = args.original_restart
    else:
        FILE_OLD = DEFAULT_FILE_OLD
    
    print(f"Using AI-enhanced restart: {FILE_NEW}")
    print(f"Using original restart: {FILE_OLD}")
    print(f"Layers to plot: {LEVGRND_LAYERS}")
    print(f"PFTs to plot: {PFT_PICK_LIST}")
    print("-" * 80)
    
    # Parse CNP_IO list to get variables
    print("Parsing CNP_IO list...")
    cnp_io_vars = parse_cnp_io_list(Path(args.variable_list))
    
    # Extract variable names from the parsed structure
    pft_1d_variables = cnp_io_vars.get('pft_1d_variables', [])
    variables_2d_soil = cnp_io_vars.get('variables_2d_soil', [])
    
    # Combine all variables to plot
    VARIABLES = pft_1d_variables + variables_2d_soil
    
    if not VARIABLES:
        print("Error: No variables found in CNP_IO list")
        return
    
    print(f"Variables to plot: {VARIABLES}")
    print(f"  PFT1D: {pft_1d_variables}")
    print(f"  Soil2D: {variables_2d_soil}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ds_new = xr.open_dataset(FILE_NEW)
    ds_old = xr.open_dataset(FILE_OLD)

    grid_lon, grid_lat = _gridcell_lonlat(ds_new)
    n_grid = ds_new.sizes["gridcell"]

    col2grid = _to_zero_based_index(_safe_get(ds_new, "cols1d_gridcell_index").values, n_grid)
    pft2grid = _to_zero_based_index(_safe_get(ds_new, "pfts1d_gridcell_index").values, n_grid)

    grid_to_cols = _build_gridcell_groups(col2grid, n_grid)
    grid_to_pfts = _build_gridcell_groups(pft2grid, n_grid)

    print(f"Total gridcells: {n_grid} | total columns: {col2grid.size} | total pfts: {pft2grid.size}")
    print(f"Example: gridcell 0 -> columns {grid_to_cols[0][:5]}, pfts {grid_to_pfts[0][:5]}")

    print(f"\nStart plotting: {len(VARIABLES)} variables")
    for var in VARIABLES:
        if (var not in ds_new.data_vars) or (var not in ds_old.data_vars):
            print(f"Skip {var} (not found in both files)")
            continue

        da_new = ds_new[var]
        da_old = ds_old[var]
        dims = da_new.dims
        print(f"\nVariable {var}, dims: {dims}")

        if ("column" in dims) and ("levgrnd" in dims):
            # Soil2D variable
            da_new_cl = da_new.transpose("column", "levgrnd")
            da_old_cl = da_old.transpose("column", "levgrnd")
            vals_new = _to_nan_fillvalue(da_new_cl.values)
            vals_old = _to_nan_fillvalue(da_old_cl.values)

            for lev in LEVGRND_LAYERS:
                if lev < 0 or lev >= da_new_cl.sizes["levgrnd"]:
                    print(f"  Layer {lev} out of range, skipped")
                    continue

                new_grid = np.full(n_grid, np.nan, dtype=float)
                old_grid = np.full(n_grid, np.nan, dtype=float)

                for g in range(n_grid):
                    cols = grid_to_cols[g]
                    if len(cols) == 0:
                        continue
                    c0 = cols[0]  # First column only
                    new_grid[g] = vals_new[c0, lev]
                    old_grid[g] = vals_old[c0, lev]

                _plot_tripanel(var, f"_lev{lev}", grid_lon, grid_lat, new_grid, old_grid, OUTPUT_DIR,
                               label_new=LABEL_NEW, label_old=LABEL_OLD)

        elif ("pft" in dims) and (len(dims) == 1):
            # PFT1D variable
            da_new_p = da_new.transpose("pft")
            da_old_p = da_old.transpose("pft")
            vals_new = _to_nan_fillvalue(da_new_p.values)
            vals_old = _to_nan_fillvalue(da_old_p.values)

            for k in PFT_PICK_LIST:
                if k < 0 or k >= da_new_p.sizes["pft"]:
                    print(f"  PFT {k} out of range, skipped")
                    continue
                    
                new_grid = np.full(n_grid, np.nan, dtype=float)
                old_grid = np.full(n_grid, np.nan, dtype=float)

                for g in range(n_grid):
                    pfts = grid_to_pfts[g]
                    if len(pfts) == 0:
                        continue
                    if k < len(pfts):
                        p_idx = pfts[k]
                        new_grid[g] = vals_new[p_idx]
                        old_grid[g] = vals_old[p_idx]

                _plot_tripanel(var, f"_pft{k}", grid_lon, grid_lat, new_grid, old_grid, OUTPUT_DIR,
                               label_new=LABEL_NEW, label_old=LABEL_OLD)

        else:
            print(f"  Skip {var} (only supports (column, levgrnd) and (pft,))")

    ds_new.close()
    ds_old.close()
    
    # Print comparison summary
    print("\n" + "="*80)
    print("AI RESTART COMPARISON SUMMARY:")
    print("="*80)
    print(f"Original restart: {os.path.abspath(FILE_OLD)}")
    print(f"AI-enhanced restart: {os.path.abspath(FILE_NEW)}")
    print(f"Labels: {LABEL_OLD} vs {LABEL_NEW}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Variables plotted: {VARIABLES}")
    print(f"Layers plotted: {LEVGRND_LAYERS}")
    print(f"PFTs plotted: {PFT_PICK_LIST}")
    print("="*80)
    print("\nAll plots done! Output dir:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
    
    # Print usage examples
    print("\n" + "="*60)
    print("USAGE EXAMPLES:")
    print("="*60)
    print("1. Basic usage with auto-detection:")
    print("   python ai_restart_comparison.py --variable-list ../../CNP_IO_demo1.txt")
    print()
    print("2. Specify custom AI restart file:")
    print("   python ai_restart_comparison.py --variable-list ../../CNP_IO_demo1.txt --ai-restart /path/to/ai_restart.nc")
    print()
    print("3. Specify custom original restart file:")
    print("   python ai_restart_comparison.py --variable-list ../../CNP_IO_demo1.txt --original-restart /path/to/original.nc")
    print()
    print("4. Custom layers and PFTs:")
    print("   python ai_restart_comparison.py --variable-list ../../CNP_IO_demo1.txt --layers 0,2,4,6,8 --pfts 0,1,2,3")
    print()
    print("5. Full custom configuration:")
    print("   python ai_restart_comparison.py --variable-list ../../CNP_IO_demo1.txt --ai-restart ai_file.nc --original-restart orig_file.nc --layers 0,5,9 --pfts 0,1,2,3,4,5")
    print("="*60)
