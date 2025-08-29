from netCDF4 import Dataset
import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy.spatial import cKDTree

orig_nc = "../../ELM_data/original_20250408_trendytest_ICB1850CNPRDCTCBC.elm.r.0021-01-01-00000.nc"
upd_nc  = "restart_file_run_20250807_231637_updated.nc"

# Where predictions (including test_static_inverse.csv) are saved (adjust if needed)
pred_dir = "cnp_infer/cnp_predictions"

rtol, atol = 1e-6, 1e-8

# Load IO list used
io_json_path = Path("CNP_IO_list_used.json")
if not io_json_path.exists():
    raise FileNotFoundError("CNP_IO_list_used.json not found in current directory")
with open(io_json_path) as f:
    io_cfg = json.load(f)

pft1d_vars = io_cfg.get('pft_1d_variables', [])
soil2d_vars = io_cfg.get('variables_2d_soil', [])

# Helper: build gridcell->columns map and gridcell->pfts map
with Dataset(orig_nc) as f0:
    cols_idx_all = f0.variables['cols1d_gridcell_index'][:] if 'cols1d_gridcell_index' in f0.variables else None
    pfts_idx_all = f0.variables['pfts1d_gridcell_index'][:] if 'pfts1d_gridcell_index' in f0.variables else None
    glon = f0.variables['grid1d_lon'][:] if 'grid1d_lon' in f0.variables else None
    glat = f0.variables['grid1d_lat'][:] if 'grid1d_lat' in f0.variables else None

# Build grid->cols map
grid_to_cols = {}
if cols_idx_all is not None:
    for col_idx, g1 in enumerate(cols_idx_all):
        g = int(g1) - 1
        grid_to_cols.setdefault(g, []).append(col_idx)
    for g in grid_to_cols:
        grid_to_cols[g] = sorted(grid_to_cols[g])

# Build grid->pfts map
grid_to_pfts = {}
if pfts_idx_all is not None:
    for pft_i, g1 in enumerate(pfts_idx_all):
        g = int(g1) - 1
        grid_to_pfts.setdefault(g, []).append(pft_i)
    for g in grid_to_pfts:
        grid_to_pfts[g] = sorted(grid_to_pfts[g])

# Optional: build row->grid mapping for test set if static inverse available
row_to_grid = None
static_inv_csv = Path(pred_dir) / 'test_static_inverse.csv'
if glon is not None and glat is not None and static_inv_csv.exists():
    try:
        df_static = pd.read_csv(static_inv_csv)
        if {'Longitude', 'Latitude'}.issubset(df_static.columns):
            tree = cKDTree(np.column_stack([glon, glat]))
            q = np.column_stack([df_static['Longitude'].values, df_static['Latitude'].values])
            _, row_to_grid = tree.query(q, k=1)
            row_to_grid = row_to_grid.astype(int)
    except Exception:
        row_to_grid = None

soil_rows = []
pft_rows = []

with Dataset(orig_nc) as f0, Dataset(upd_nc) as f1:
    # Check Soil2D variables
    for var in soil2d_vars:
        if var not in f0.variables or var not in f1.variables:
            continue
        a = f0.variables[var][:]
        b = f1.variables[var][:]
        only_first_changed = 0
        other_changes = 0
        unchanged_all = 0
        # Per gridcell
        for g, cols in grid_to_cols.items():
            if not cols:
                continue
            first_col = cols[0]
            if a.ndim == 2:
                first_changed = not np.allclose(a[first_col, :], b[first_col, :], rtol=rtol, atol=atol)
                others_equal = all(np.allclose(a[c, :], b[c, :], rtol=rtol, atol=atol) for c in cols[1:])
            else:
                first_changed = not np.allclose(a[first_col], b[first_col], rtol=rtol, atol=atol)
                others_equal = all(np.allclose(a[c], b[c], rtol=rtol, atol=atol) for c in cols[1:])
            if first_changed and others_equal:
                only_first_changed += 1
            elif first_changed or not others_equal:
                other_changes += 1
            else:
                unchanged_all += 1
        # Test-only breakdown
        test_total = None
        test_first_changed = None
        test_unchanged = None
        test_unchanged_allzero_firstcol = None
        if row_to_grid is not None:
            test_grids = np.unique(row_to_grid)
            test_total = int(test_grids.size)
            test_first_changed = 0
            test_unchanged = 0
            test_unchanged_allzero_firstcol = 0
            for g in test_grids:
                cols = grid_to_cols.get(int(g), [])
                if not cols:
                    continue
                first_col = cols[0]
                if a.ndim == 2:
                    first_changed = not np.allclose(a[first_col, :], b[first_col, :], rtol=rtol, atol=atol)
                    first_is_zero_upd = np.allclose(b[first_col, :], 0.0, rtol=rtol, atol=atol)
                else:
                    first_changed = not np.allclose(a[first_col], b[first_col], rtol=rtol, atol=atol)
                    first_is_zero_upd = np.allclose(b[first_col], 0.0, rtol=rtol, atol=atol)
                if first_changed:
                    test_first_changed += 1
                else:
                    test_unchanged += 1
                    if first_is_zero_upd:
                        test_unchanged_allzero_firstcol += 1
        soil_rows.append({
            'variable': var,
            'only_first_changed': only_first_changed,
            'other_changes': other_changes,
            'unchanged': unchanged_all,
            'test_total': test_total,
            'test_first_changed': test_first_changed,
            'test_unchanged': test_unchanged,
            'test_unchanged_allzero_firstcol': test_unchanged_allzero_firstcol,
        })
    # Check PFT1D variables
    for var in pft1d_vars:
        if var not in f0.variables or var not in f1.variables:
            continue
        a = f0.variables[var][:]
        b = f1.variables[var][:]
        ok_only_pft1_16 = 0
        violations = 0
        perfectly_unchanged = 0
        for g, pfts in grid_to_pfts.items():
            if not pfts:
                continue
            p0 = pfts[0]
            p0_unchanged = np.allclose(a[p0], b[p0], rtol=rtol, atol=atol)
            end_upd = min(16, len(pfts) - 1)
            changed_flags = []
            for k in range(1, len(pfts)):
                idx = pfts[k]
                changed = not np.allclose(a[idx], b[idx], rtol=rtol, atol=atol)
                changed_flags.append(changed)
            first_block = changed_flags[:end_upd]
            rest_block = changed_flags[end_upd:]
            if p0_unchanged and all(not x for x in rest_block):
                if any(first_block):
                    ok_only_pft1_16 += 1
                else:
                    perfectly_unchanged += 1
            else:
                violations += 1
        pft_rows.append({
            'variable': var,
            'ok_only_pft1_16_changed': ok_only_pft1_16,
            'violations': violations,
            'unchanged_all': perfectly_unchanged,
        })

# Save results in current directory
pd.DataFrame(soil_rows).to_csv('soil2d_check_results.csv', index=False)
pd.DataFrame(pft_rows).to_csv('pft1d_check_results.csv', index=False)

# Also print quick summaries
print("Soil2D checks saved to soil2d_check_results.csv")
print("PFT1D checks saved to pft1d_check_results.csv")