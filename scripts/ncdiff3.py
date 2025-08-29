import numpy as np
import netCDF4 as nc
import sys
import csv


def nrmse(y_true, y_pred):
    diff = y_true - y_pred
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mean_true = np.mean(y_true)
    if mean_true == 0:
        return np.inf
    return rmse / mean_true


def to_array_and_mask(a):
    """Return (ndarray_with_nan, mask_array_or_none).
    - If masked array: cast to float, fill masked with NaN, return original mask.
    - If plain array: cast ints to float for NaN ops; return None mask.
    """
    if np.ma.isMaskedArray(a):
        # Cast to float before filling with NaN to avoid int fill errors
        a_float = a.astype(np.float64)
        return a_float.filled(np.nan), np.ma.getmaskarray(a)
    arr = np.asarray(a)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float64)
    return arr, None


def compare_per_layer(data1, data2, layer_name="layer"):
    """Compare arrays per layer along the last axis and print metrics per layer.
    Prints: SUM(file1), SUM(file2), NRMSE, max(diff), min(diff), NaN and masked counts for each layer index.
    """
    a1, m1 = to_array_and_mask(data1)
    a2, m2 = to_array_and_mask(data2)
    if a1.shape != a2.shape:
        print(f"  Shapes differ: {a1.shape} vs {a2.shape}")
        return
    # Flatten all leading dims except the last (treated as layers)
    L = a1.shape[-1]
    d1 = a1.reshape(-1, L)
    d2 = a2.reshape(-1, L)
    if m1 is None:
        m1 = np.zeros_like(a1, dtype=bool)
    if m2 is None:
        m2 = np.zeros_like(a2, dtype=bool)
    m1f = m1.reshape(-1, L)
    m2f = m2.reshape(-1, L)
    for l in range(L):
        y1 = d1[:, l]
        y2 = d2[:, l]
        n_nan1 = int(np.isnan(y1).sum())
        n_nan2 = int(np.isnan(y2).sum())
        n_mask1 = int(m1f[:, l].sum())
        n_mask2 = int(m2f[:, l].sum())
        # Mask NaNs consistently
        mask = ~np.isnan(y1) & ~np.isnan(y2)
        if np.sum(mask) == 0:
            print(
                f"  {layer_name}[{l}]: no valid values to compare (NaNs: file1={n_nan1}, file2={n_nan2}; masked: file1={n_mask1}, file2={n_mask2})"
            )
            continue
        y1m = y1[mask]
        y2m = y2[mask]
        diff = y1m - y2m
        s1 = float(np.sum(y1m))
        s2 = float(np.sum(y2m))
        metric_nrmse = nrmse(y1m, y2m)
        dmax = float(np.max(diff))
        dmin = float(np.min(diff))
        print(
            f"  {layer_name}[{l}]: SUM1={s1:.6g}, SUM2={s2:.6g}, NRMSE={metric_nrmse:.6g}, "
            f"max_diff={dmax:.6g}, min_diff={dmin:.6g}, NaNs(file1={n_nan1}, file2={n_nan2}), "
            f"masked(file1={n_mask1}, file2={n_mask2})"
        )


def compare_variables(name, var1, var2):
    dims1 = getattr(var1, 'dimensions', ())
    dims2 = getattr(var2, 'dimensions', ())
    print(f"Variable: {name}")
    print(f"  dtype: {var1.dtype} vs {var2.dtype}")
    print(f"  dims: {dims1} vs {dims2}")
    print(f"  shape: {var1.shape} vs {var2.shape}")

    summary = {
        'variable': name,
        'dtype1': str(var1.dtype),
        'dtype2': str(var2.dtype),
        'shape1': str(var1.shape),
        'shape2': str(var2.shape),
        'differs': False,
        'overall_nrmse': None,
        'valid_points_compared': 0,
        'nan1': 0,
        'nan2': 0,
        'masked1': 0,
        'masked2': 0,
    }

    if var1.dtype != var2.dtype:
        print(f"  -> Different data types")
    if var1.shape != var2.shape:
        print(f"  -> Different shapes; skipping data comparison")
        summary['differs'] = True
        return summary

    # Only compare numeric types and skip short to avoid compression artifacts
    if not (np.issubdtype(var1.dtype, np.number) and var1.dtype != 'short'):
        return summary

    try:
        raw1 = var1[:]
        raw2 = var2[:]
    except Exception as e:
        print(f"  -> Failed to read data: {e}")
        summary['differs'] = True
        return summary

    a1, m1 = to_array_and_mask(raw1)
    a2, m2 = to_array_and_mask(raw2)

    # NaN and masked counts overall
    n_nan1 = int(np.isnan(a1).sum())
    n_nan2 = int(np.isnan(a2).sum())
    n_mask1 = int(m1.sum()) if m1 is not None else 0
    n_mask2 = int(m2.sum()) if m2 is not None else 0
    summary['nan1'] = n_nan1
    summary['nan2'] = n_nan2
    summary['masked1'] = n_mask1
    summary['masked2'] = n_mask2
    if n_nan1 or n_nan2 or n_mask1 or n_mask2:
        print(f"  NaN counts: file1={n_nan1}, file2={n_nan2}; masked: file1={n_mask1}, file2={n_mask2}")

    # If identical within tolerance, report and return
    if np.allclose(a1, a2, equal_nan=True):
        print("  -> Data equal within tolerance")
        summary['differs'] = False
        return summary

    # Overall comparison across all valid positions
    valid = ~np.isnan(a1) & ~np.isnan(a2)
    summary['valid_points_compared'] = int(valid.sum())
    if summary['valid_points_compared'] > 0:
        summary['overall_nrmse'] = float(nrmse(a1[valid], a2[valid]))
    summary['differs'] = True

    # For 0D/1D: print simple summary
    if a1.ndim <= 1:
        mask = valid
        if np.any(mask):
            diff = a1[mask] - a2[mask]
            print(
                f"  Diff summary: SUM1={np.sum(a1[mask]):.6g}, SUM2={np.sum(a2[mask]):.6g}, "
                f"NRMSE={nrmse(a1[mask], a2[mask]):.6g}, max_diff={np.max(diff):.6g}, min_diff={np.min(diff):.6g}, "
                f"NaNs(file1={n_nan1}, file2={n_nan2}), masked(file1={n_mask1}, file2={n_mask2})"
            )
        else:
            print("  No valid values to compare (all NaNs)")
        return summary

    # For 2D or higher: treat last axis as layers
    layer_name = dims1[-1] if len(dims1) > 0 else "layer"
    compare_per_layer(a1, a2, layer_name=layer_name)
    return summary


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: python scripts/ncdiff2.py file1.nc file2.nc [--save-diff-list diff.csv]")
        sys.exit(1)

    # Simple arg parsing to allow optional --save-diff-list
    save_path = None
    files = []
    i = 0
    while i < len(args):
        if args[i] == '--save-diff-list' and i + 1 < len(args):
            save_path = args[i + 1]
            i += 2
        else:
            files.append(args[i])
            i += 1
    if len(files) < 2:
        print("Usage: python scripts/ncdiff2.py file1.nc file2.nc [--save-diff-list diff.csv]")
        sys.exit(1)

    file_name1, file_name2 = files[0], files[1]

    file1 = nc.Dataset(file_name1)
    file2 = nc.Dataset(file_name2)

    variables1 = file1.variables
    variables2 = file2.variables

    diff_records = []

    for var in variables1:
        if var in variables2:
            summary = compare_variables(var, variables1[var], variables2[var])
            if summary and summary.get('differs'):
                diff_records.append(summary)
        else:
            print(f'Variable {var} is not in the second file')

    for var in variables2:
        if var not in variables1:
            print(f'Variable {var} is not in the first file')

    file1.close()
    file2.close()

    if save_path is not None:
        # Write CSV of differing variables
        fieldnames = ['variable', 'dtype1', 'dtype2', 'shape1', 'shape2', 'differs', 'overall_nrmse', 'valid_points_compared', 'nan1', 'nan2', 'masked1', 'masked2']
        with open(save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in diff_records:
                writer.writerow(rec)
        print(f"Saved list of differing variables to: {save_path}")


if __name__ == '__main__':
    main()