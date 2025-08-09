#!/usr/bin/env python3
import argparse
import os
import sys
import json
from glob import glob
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import textwrap


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_examples_epilog() -> str:
    return textwrap.dedent(
        """
        Examples:
          # Compare scalar predictions between two runs
          python scripts/compare_predictions.py --kind scalar \
            --run-a cnp_results/run_20250807_215454 \
            --run-b cnp_results/run_20250807_215858 \
            --out comparison/scalar

          # Compare scalar predictions vs. ground truth for a single run
          python scripts/compare_predictions.py --kind scalar \
            --run-a cnp_results/run_20250807_215454 \
            --against-gt \
            --out comparison/scalar_vs_gt

          # Compare 1D PFT predictions between two runs
          python scripts/compare_predictions.py --kind pft1d \
            --run-a cnp_results/run_20250807_215454 \
            --run-b cnp_results/run_20250807_215858 \
            --out comparison/pft1d

          # Compare 1D PFT predictions vs. ground truth
          python scripts/compare_predictions.py --kind pft1d \
            --run-a cnp_results/run_20250807_215454 \
            --against-gt \
            --out comparison/pft1d_vs_gt

          # Compare 2D soil predictions between two runs
          python scripts/compare_predictions.py --kind soil2d \
            --run-a cnp_results/run_20250807_215454 \
            --run-b cnp_results/run_20250807_215858 \
            --out comparison/soil2d

          # Compare 2D soil predictions vs. ground truth
          python scripts/compare_predictions.py --kind soil2d \
            --run-a cnp_results/run_20250807_215454 \
            --against-gt \
            --out comparison/soil2d_vs_gt

        Outputs:
          - scalar_runs_diff.csv, scalar_vs_gt.csv
          - pft1d_runs_diff.csv, pft1d_vs_gt.csv
          - soil2d_runs_diff.csv, soil2d_vs_gt.csv
        """
    )


# ===== Scalar loaders =====

def load_scalar(run_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred = os.path.join(run_dir, 'cnp_predictions', 'predictions_scalar.csv')
    gt = os.path.join(run_dir, 'cnp_predictions', 'ground_truth_scalar.csv')
    pred_df = pd.read_csv(pred) if os.path.exists(pred) else None
    gt_df = pd.read_csv(gt) if os.path.exists(gt) else None
    return pred_df, gt_df


# ===== 1D PFT loaders =====

def list_pft1d_vars(run_dir: str) -> List[str]:
    pred_dir = os.path.join(run_dir, 'cnp_predictions', 'pft_1d_predictions')
    files = glob(os.path.join(pred_dir, 'predictions_*.csv'))
    vars_ = [os.path.basename(f)[len('predictions_'):-len('.csv')] for f in files]
    return sorted(vars_)


def load_pft1d_var(run_dir: str, var_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred_path = os.path.join(run_dir, 'cnp_predictions', 'pft_1d_predictions', f'predictions_{var_name}.csv')
    gt_path = os.path.join(run_dir, 'cnp_predictions', 'pft_1d_ground_truth', f'ground_truth_{var_name}.csv')
    pred_df = pd.read_csv(pred_path) if os.path.exists(pred_path) else None
    gt_df = pd.read_csv(gt_path) if os.path.exists(gt_path) else None
    return pred_df, gt_df


# ===== 2D Soil loaders =====

def list_soil2d_vars(run_dir: str) -> List[str]:
    pred_dir = os.path.join(run_dir, 'cnp_predictions', 'soil_2d_predictions')
    files = glob(os.path.join(pred_dir, 'predictions_*.csv'))
    vars_ = [os.path.basename(f)[len('predictions_'):-len('.csv')] for f in files]
    return sorted(vars_)


def load_soil2d_var(run_dir: str, var_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred_path = os.path.join(run_dir, 'cnp_predictions', 'soil_2d_predictions', f'predictions_{var_name}.csv')
    gt_path = os.path.join(run_dir, 'cnp_predictions', 'soil_2d_ground_truth', f'ground_truth_{var_name}.csv')
    pred_df = pd.read_csv(pred_path) if os.path.exists(pred_path) else None
    gt_df = pd.read_csv(gt_path) if os.path.exists(gt_path) else None
    return pred_df, gt_df


# ===== Metrics helpers =====

def per_series_diff(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    diff = a - b
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_abs = float(np.max(np.abs(diff)))
    try:
        corr = float(np.corrcoef(a, b)[0, 1])
    except Exception:
        corr = float('nan')
    return {
        'mae': mae,
        'rmse': rmse,
        'max_abs_diff': max_abs,
        'corr_between_runs': corr,
    }


def per_series_vs_gt(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    mask = ~np.isnan(pred) & ~np.isnan(gt)
    if np.sum(mask) == 0:
        return {'mse': float('nan'), 'rmse': float('nan'), 'nrmse': float('nan'), 'r2': float('nan')}
    y = gt[mask]
    yhat = pred[mask]
    mse = float(np.mean((y - yhat) ** 2))
    rmse = float(np.sqrt(mse))
    mean_y = float(np.mean(y))
    nrmse = (rmse / mean_y) if mean_y != 0 else float('inf')
    # r2
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
    return {'mse': mse, 'rmse': rmse, 'nrmse': nrmse, 'r2': r2}


# ===== Compare functions =====

def compare_scalar_runs(run_a: str, run_b: str, out_dir: str):
    a_pred, _ = load_scalar(run_a)
    b_pred, _ = load_scalar(run_b)
    if a_pred is None or b_pred is None:
        raise FileNotFoundError('predictions_scalar.csv not found in one of the runs')
    common_cols = [c for c in a_pred.columns if c in b_pred.columns]
    a = a_pred[common_cols].to_numpy()
    b = b_pred[common_cols].to_numpy()
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    rows = []
    for i, col in enumerate(common_cols):
        m = per_series_diff(a[:, i], b[:, i])
        rows.append({'variable': col, 'rows_compared': n, **m})
    ensure_dir(out_dir)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'scalar_runs_diff.csv'), index=False)


def compare_scalar_vs_gt(run_dir: str, out_dir: str):
    pred, gt = load_scalar(run_dir)
    if pred is None or gt is None:
        raise FileNotFoundError('scalar prediction or ground truth missing')
    common_cols = [c for c in pred.columns if c in gt.columns]
    p = pred[common_cols].to_numpy()
    g = gt[common_cols].to_numpy()
    n = min(len(p), len(g))
    p = p[:n]
    g = g[:n]
    rows = []
    for i, col in enumerate(common_cols):
        m = per_series_vs_gt(p[:, i], g[:, i])
        rows.append({'variable': col, 'rows_compared': n, **m})
    ensure_dir(out_dir)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'scalar_vs_gt.csv'), index=False)


def compare_pft1d_runs(run_a: str, run_b: str, out_dir: str):
    vars_a = set(list_pft1d_vars(run_a))
    vars_b = set(list_pft1d_vars(run_b))
    common_vars = sorted(vars_a & vars_b)
    if not common_vars:
        raise ValueError('No common 1D PFT variables found between runs')
    rows = []
    for var in common_vars:
        a_pred, _ = load_pft1d_var(run_a, var)
        b_pred, _ = load_pft1d_var(run_b, var)
        if a_pred is None or b_pred is None:
            continue
        # Align by columns (PFTs)
        common_cols = [c for c in a_pred.columns if c in b_pred.columns]
        A = a_pred[common_cols].to_numpy()
        B = b_pred[common_cols].to_numpy()
        n = min(len(A), len(B))
        A = A[:n]
        B = B[:n]
        for j, col in enumerate(common_cols):
            m = per_series_diff(A[:, j], B[:, j])
            rows.append({'variable': var, 'pft': col, 'rows_compared': n, **m})
    ensure_dir(out_dir)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'pft1d_runs_diff.csv'), index=False)


def compare_pft1d_vs_gt(run_dir: str, out_dir: str):
    vars_ = list_pft1d_vars(run_dir)
    if not vars_:
        raise FileNotFoundError('No 1D PFT prediction files found')
    rows = []
    for var in vars_:
        pred_df, gt_df = load_pft1d_var(run_dir, var)
        if pred_df is None or gt_df is None:
            continue
        common_cols = [c for c in pred_df.columns if c in gt_df.columns]
        P = pred_df[common_cols].to_numpy()
        G = gt_df[common_cols].to_numpy()
        n = min(len(P), len(G))
        P = P[:n]
        G = G[:n]
        for j, col in enumerate(common_cols):
            m = per_series_vs_gt(P[:, j], G[:, j])
            rows.append({'variable': var, 'pft': col, 'rows_compared': n, **m})
    ensure_dir(out_dir)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'pft1d_vs_gt.csv'), index=False)


def compare_soil2d_runs(run_a: str, run_b: str, out_dir: str):
    vars_a = set(list_soil2d_vars(run_a))
    vars_b = set(list_soil2d_vars(run_b))
    common_vars = sorted(vars_a & vars_b)
    if not common_vars:
        raise ValueError('No common 2D soil variables found between runs')
    rows = []
    for var in common_vars:
        a_pred, _ = load_soil2d_var(run_a, var)
        b_pred, _ = load_soil2d_var(run_b, var)
        if a_pred is None or b_pred is None:
            continue
        common_cols = [c for c in a_pred.columns if c in b_pred.columns]
        A = a_pred[common_cols].to_numpy()
        B = b_pred[common_cols].to_numpy()
        n = min(len(A), len(B))
        A = A[:n]
        B = B[:n]
        for j, col in enumerate(common_cols):
            m = per_series_diff(A[:, j], B[:, j])
            rows.append({'variable': var, 'col_layer': col, 'rows_compared': n, **m})
    ensure_dir(out_dir)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'soil2d_runs_diff.csv'), index=False)


def compare_soil2d_vs_gt(run_dir: str, out_dir: str):
    vars_ = list_soil2d_vars(run_dir)
    if not vars_:
        raise FileNotFoundError('No 2D soil prediction files found')
    rows = []
    for var in vars_:
        pred_df, gt_df = load_soil2d_var(run_dir, var)
        if pred_df is None or gt_df is None:
            continue
        common_cols = [c for c in pred_df.columns if c in gt_df.columns]
        P = pred_df[common_cols].to_numpy()
        G = gt_df[common_cols].to_numpy()
        n = min(len(P), len(G))
        P = P[:n]
        G = G[:n]
        for j, col in enumerate(common_cols):
            m = per_series_vs_gt(P[:, j], G[:, j])
            rows.append({'variable': var, 'col_layer': col, 'rows_compared': n, **m})
    ensure_dir(out_dir)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'soil2d_vs_gt.csv'), index=False)


def main():
    parser = argparse.ArgumentParser(
        description='Compare predictions between runs or vs. ground truth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=build_examples_epilog()
    )
    parser.add_argument('--kind', required=True, choices=['scalar', 'pft1d', 'soil2d'])
    parser.add_argument('--run-a', required=True, help='Path to first run directory (cnp_results/run_...)')
    parser.add_argument('--run-b', help='Path to second run directory (for between-run comparison)')
    parser.add_argument('--against-gt', action='store_true', help='Compare run-a predictions against ground truth')
    parser.add_argument('--out', default='comparison', help='Output directory for reports')
    parser.add_argument('--examples', action='store_true', help='Show example usage and exit')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.examples:
        print(parser.format_help())
        sys.exit(0)

    if args.against_gt and args.run_b:
        print('[WARN] --against-gt ignores --run-b; comparing run-a vs GT only', file=sys.stderr)

    if args.kind == 'scalar':
        if args.against_gt:
            compare_scalar_vs_gt(args.run_a, os.path.join(args.out, 'scalar_vs_gt'))
        elif args.run_b:
            compare_scalar_runs(args.run_a, args.run_b, os.path.join(args.out, 'scalar_runs'))
        else:
            raise ValueError('Provide --run-b or --against-gt for scalar')
    elif args.kind == 'pft1d':
        if args.against_gt:
            compare_pft1d_vs_gt(args.run_a, os.path.join(args.out, 'pft1d_vs_gt'))
        elif args.run_b:
            compare_pft1d_runs(args.run_a, args.run_b, os.path.join(args.out, 'pft1d_runs'))
        else:
            raise ValueError('Provide --run-b or --against-gt for pft1d')
    elif args.kind == 'soil2d':
        if args.against_gt:
            compare_soil2d_vs_gt(args.run_a, os.path.join(args.out, 'soil2d_vs_gt'))
        elif args.run_b:
            compare_soil2d_runs(args.run_a, args.run_b, os.path.join(args.out, 'soil2d_runs'))
        else:
            raise ValueError('Provide --run-b or --against-gt for soil2d')

    print(json.dumps({'status': 'ok', 'out': args.out}, indent=2))


if __name__ == '__main__':
    main() 