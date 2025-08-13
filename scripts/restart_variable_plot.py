import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, ListedColormap, BoundaryNorm

VARIABLES = ['labilep_vr', 'primp_vr', 'deadstemc', 'tlai'] # variables 
LEVGRND_LAYERS = [0, 5, 9] # layers for (column, levgrnd)-type variables (0,9)
PFT_PICK_LIST = [2,3,4, 5, 6] # which PFT(s) to plot per gridcell   (1,16)

#FILE_NEW = '/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data_700/output/20250117_trendytest_ICB1850CNPRDCTCBC.elm.r.0781-01-01-00000.nc'
#FILE_OLD = '/home/UNT/dg0997/all_gdw/0_oak_weather/26_automatical_dataset_train_restart/auto_dataset_training_restart/backup/results/enhanced_restart.nc'

FILE_OLD = '/mnt/proj-shared/AI4BGC_7xw/AI4BGC/ELM_data/original_780_spinup_from_modelsimulation.nc'
#FILE_OLD = '/mnt/proj-shared/AI4BGC_7xw/AI4BGC/ELM_data/original_20250408_trendytest_ICB1850CNPRDCTCBC.elm.r.0021-01-01-00000.nc'
#FILE_NEW = '/mnt/proj-shared/AI4BGC_7xw/AI4BGC/cnp_results/run_20250812_172831/restart_file_CNP_IO_updated13_20250812_172831_updated.nc'
FILE_NEW = '/mnt/proj-shared/AI4BGC_7xw/AI4BGC/ELM_data/updated13_20250408_trendytest_ICB1850CNPRDCTCBC.elm.r.0021-01-01-00000.nc'

LABEL_NEW = "AI results"
LABEL_OLD = "780 year"

OUTPUT_DIR = "./restart_gridcell_plots"

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
    # Compute percent (model - AI)/model in %
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

def _plot_tripanel(var, label_suffix, lon, lat, data_new, data_old, out_dir,
                   label_new="780 year", label_old="AI results"):
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
    print(f"  Metrics (old vs new): n={n} rmse={rmse:.6g} nrmse={nrmse:.6g} r2={r2:.6g}")

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
    ax4.set_title(f"{var} - Percent Diff bins (({label_new}-{label_old})/{label_old})", fontsize=14, fontweight="bold")
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


def main():
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
                    c0 = cols[0]
                    new_grid[g] = vals_new[c0, lev]
                    old_grid[g] = vals_old[c0, lev]

                _plot_tripanel(var, f"_lev{lev}", grid_lon, grid_lat, new_grid, old_grid, OUTPUT_DIR,
                               label_new=LABEL_NEW, label_old=LABEL_OLD)

        elif ("pft" in dims) and (len(dims) == 1):
            da_new_p = da_new.transpose("pft")
            da_old_p = da_old.transpose("pft")
            vals_new = _to_nan_fillvalue(da_new_p.values)
            vals_old = _to_nan_fillvalue(da_old_p.values)

            for k in PFT_PICK_LIST:
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
    print("\nAll plots done! Output dir:", OUTPUT_DIR)

if __name__ == "__main__":
    main()