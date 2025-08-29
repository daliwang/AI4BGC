import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from typing import List, Optional


def load_losses(case_dir):
    path = os.path.join(case_dir, "cnp_training_losses.csv")
    return pd.read_csv(path)


def _try_paths(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def load_predictions(case_dir, pred_type):
    base = os.path.join(case_dir, "cnp_predictions")
    base_test = os.path.join(case_dir, "cnp_predictions_test")

    if pred_type == 'scalar':
        path = _try_paths(
            os.path.join(base, "predictions_scalar.csv"),
            os.path.join(base_test, "predictions_scalar.csv"),
        )
        if path:
            return pd.read_csv(path)
        return None

    if pred_type == '1d':
        dir_candidates = [
            os.path.join(base, "pft_1d_predictions"),
            os.path.join(base_test, "pft_1d_predictions"),
        ]
        for d in dir_candidates:
            if os.path.isdir(d):
                result = {}
                for fname in sorted(os.listdir(d)):
                    if fname.startswith("predictions_") and fname.endswith(".csv"):
                        var_name = fname[len("predictions_"):-4]
                        result[var_name] = pd.read_csv(os.path.join(d, fname))
                return result if result else None
        return None

    if pred_type == '2d':
        dir_candidates = [
            os.path.join(base, "soil_2d_predictions"),
            os.path.join(base_test, "soil_2d_predictions"),
        ]
        for d in dir_candidates:
            if os.path.isdir(d):
                result = {}
                for fname in sorted(os.listdir(d)):
                    if fname.startswith("predictions_") and fname.endswith(".csv"):
                        var_name = fname[len("predictions_"):-4]
                        result[var_name] = pd.read_csv(os.path.join(d, fname))
                return result if result else None
        return None

    # Fallback to legacy flat filenames if pred_type is directly provided
    path = os.path.join(base, f"predictions_{pred_type}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def compare_losses(loss1, loss2, label1, label2, output_dir):
    print(f"\n--- Training Loss Comparison ({label1} vs {label2}) ---")
    print(f"Final loss {label1}: {loss1.iloc[-1].to_dict()}")
    print(f"Final loss {label2}: {loss2.iloc[-1].to_dict()}")
    print(f"Mean loss {label1}: {loss1.mean().to_dict()}")
    print(f"Mean loss {label2}: {loss2.mean().to_dict()}")
    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(loss1['Epoch'], loss1['Train Loss'], label=f'{label1} train')
    plt.plot(loss1['Epoch'], loss1['Validation Loss'], label=f'{label1} val')
    plt.plot(loss2['Epoch'], loss2['Train Loss'], label=f'{label2} train')
    plt.plot(loss2['Epoch'], loss2['Validation Loss'], label=f'{label2} val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Comparison')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()


# Helpers

def _normalize_pft_name(col: str) -> Optional[str]:
    if not isinstance(col, str):
        return None
    m = re.match(r"^pft[_-]?(\d+)$", col.strip(), re.IGNORECASE)
    if m:
        return f"pft{int(m.group(1))}"
    # If column is purely a digit string like '1', treat as PFT index
    if col.strip().isdigit():
        return f"pft{int(col.strip())}"
    return None


def _ensure_pft_columns(df: pd.DataFrame) -> pd.DataFrame:
    # If columns look like PFTs (e.g., 'pft_1', 'pft1', or '1'), normalize to 'pft1'..'pftN'
    normalized = []
    matched_any = False
    for c in df.columns:
        norm = _normalize_pft_name(c)
        if norm is not None:
            normalized.append(norm)
            matched_any = True
        else:
            normalized.append(c)
    if matched_any:
        out = df.copy()
        out.columns = normalized
        return out
    # Otherwise, if they are unnamed/non-string indices, map sequentially
    if not all(isinstance(c, str) for c in df.columns):
        out = df.copy()
        out.columns = [f'pft{i+1}' for i in range(len(df.columns))]
        return out
    return df


def _sorted_pft_cols(cols: List[str]) -> List[str]:
    def pft_key(c: str) -> int:
        try:
            norm = _normalize_pft_name(c)
            if norm and norm.startswith('pft'):
                return int(norm.replace('pft', ''))
            return 10**6
        except Exception:
            return 10**6
    return [c for c in sorted(cols, key=pft_key) if _normalize_pft_name(c) is not None]


# Core comparison

def _compare_prediction_frames(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    label1: str,
    label2: str,
    pred_type: str,
    stats_rows: list,
    output_dir: str,
    suffix: str = "",
    plot_columns: Optional[List[str]] = None,
    plot_prefix: str = "",
):
    # Align by index length
    min_len = min(len(df1), len(df2))
    df1 = df1.iloc[:min_len]
    df2 = df2.iloc[:min_len]

    # Normalize PFT-1D column names to pft1..pftN if not already in that format
    if pred_type == '1d':
        df1 = _ensure_pft_columns(df1)
        df2 = _ensure_pft_columns(df2)

    shared_cols = [c for c in df1.columns if c in df2.columns]
    if not shared_cols:
        print(f"No shared columns to compare for {pred_type}{(' ' + suffix) if suffix else ''}.")
        return

    # Determine which columns to plot
    cols_to_plot = set(plot_columns) if plot_columns else set()

    if pred_type == '1d':
        # Debug: show which PFT columns will be plotted
        pft_cols = _sorted_pft_cols(shared_cols)
        if pft_cols:
            print(f"PFT 1D shared columns for {suffix or 'unknown'}: {pft_cols}")

    for col in shared_cols:
        arr1 = df1[col].values
        arr2 = df2[col].values
        mse = mean_squared_error(arr1, arr2)
        rmse = np.sqrt(mse)
        corr = np.corrcoef(arr1, arr2)[0, 1]
        print(f"{pred_type}{(' ' + suffix) if suffix else ''} {col}: RMSE={rmse:.4f}, Corr={corr:.4f}, Mean1={arr1.mean():.4f}, Mean2={arr2.mean():.4f}")
        stats_rows.append({
            'pred_type': pred_type,
            'variable': suffix or '',
            'column': col,
            'rmse': rmse,
            'corr': corr,
            'mean1': arr1.mean(),
            'mean2': arr2.mean(),
        })

        if col in cols_to_plot:
            plt.figure(figsize=(5,5))
            plt.scatter(arr1, arr2, alpha=0.3)
            plt.xlabel(f"{label1} {col}")
            plt.ylabel(f"{label2} {col}")
            title_suffix = f" {suffix}" if suffix else ""
            plt.title(f"{pred_type}{title_suffix} {col}: {label1} vs {label2}")
            plt.plot([arr1.min(), arr1.max()], [arr1.min(), arr1.max()], 'r--')
            os.makedirs(output_dir, exist_ok=True)
            fname = f"{plot_prefix}{suffix + '_' if suffix else ''}{col}.png"
            plt.savefig(os.path.join(output_dir, fname))
            plt.close()


def compare_predictions(
    pred1,
    pred2,
    label1,
    label2,
    pred_type,
    stats_rows,
    output_dir,
    pft_count: int,
    soil_layers: int,
    plot_scalar: bool,
):
    print(f"\n--- {pred_type.upper()} Prediction Comparison ({label1} vs {label2}) ---")
    if pred1 is None or pred2 is None:
        print(f"Prediction file for {pred_type} missing in one or both cases.")
        return

    # Handle dict-of-DataFrames (per-variable predictions)
    if isinstance(pred1, dict) and isinstance(pred2, dict):
        keys = sorted(set(pred1.keys()) & set(pred2.keys()))
        if not keys:
            print(f"No overlapping variables found for {pred_type} between runs.")
            return
        for key in keys:
            df1 = pred1.get(key)
            df2 = pred2.get(key)
            if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
                # Normalize PFT columns before selecting plot columns
                if pred_type == '1d':
                    df1 = _ensure_pft_columns(df1)
                    df2 = _ensure_pft_columns(df2)
                # Select columns to plot
                shared_cols = [c for c in df1.columns if c in df2.columns]
                if pred_type == '1d':
                    pft_cols = _sorted_pft_cols(shared_cols)
                    plot_cols = [
                        c for c in (pft_cols if pft_cols else shared_cols)
                        if (c in pft_cols) or (not pft_cols)
                    ][:pft_count]
                    plot_prefix = "1d_"
                elif pred_type == '2d':
                    plot_cols = shared_cols[:soil_layers]
                    plot_prefix = "2d_"
                else:  # scalar or others
                    plot_cols = shared_cols if plot_scalar else []
                    plot_prefix = f"{pred_type}_"

                _compare_prediction_frames(
                    df1, df2, label1, label2, pred_type,
                    stats_rows=stats_rows,
                    output_dir=output_dir,
                    suffix=key,
                    plot_columns=plot_cols,
                    plot_prefix=plot_prefix,
                )
            else:
                print(f"Skipping variable {key}: unexpected data format.")
        return

    # Handle single DataFrame case
    if isinstance(pred1, pd.DataFrame) and isinstance(pred2, pd.DataFrame):
        # Normalize PFT columns before selecting plot columns
        if pred_type == '1d':
            pred1 = _ensure_pft_columns(pred1)
            pred2 = _ensure_pft_columns(pred2)
        # Decide plot columns similarly
        shared_cols = [c for c in pred1.columns if c in pred2.columns]
        if pred_type == '1d':
            pft_cols = _sorted_pft_cols(shared_cols)
            plot_cols = [
                c for c in (pft_cols if pft_cols else shared_cols)
            ][:pft_count]
            plot_prefix = "1d_"
        elif pred_type == '2d':
            plot_cols = shared_cols[:soil_layers]
            plot_prefix = "2d_"
        else:
            plot_cols = shared_cols if plot_scalar else []
            plot_prefix = f"{pred_type}_"

        _compare_prediction_frames(
            pred1, pred2, label1, label2, pred_type,
            stats_rows=stats_rows,
            output_dir=output_dir,
            suffix="",
            plot_columns=plot_cols,
            plot_prefix=plot_prefix,
        )
        return

    print(f"Unsupported prediction data type for {pred_type}.")


def main(case1, case2, out_dir=None, soil_layers: int = 10, pft_count: int = 16, plot_scalar: bool = False):
    label1 = os.path.basename(case1)
    label2 = os.path.basename(case2)

    # Prepare output directory
    output_dir = out_dir or f"comparison_{label1}_vs_{label2}"
    os.makedirs(output_dir, exist_ok=True)

    # Collect statistics across all comparisons
    stats_rows = []

    # Losses
    loss1 = load_losses(case1)
    loss2 = load_losses(case2)
    compare_losses(loss1, loss2, label1, label2, output_dir)

    # Predictions
    for pred_type in ['2d', '1d', 'scalar']:
        pred1 = load_predictions(case1, pred_type)
        pred2 = load_predictions(case2, pred_type)
        compare_predictions(
            pred1, pred2, label1, label2, pred_type,
            stats_rows, output_dir,
            pft_count=pft_count,
            soil_layers=soil_layers,
            plot_scalar=plot_scalar,
        )

    # Persist statistics
    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        stats_df.to_csv(os.path.join(output_dir, "summary_stats.csv"), index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare two CNP model runs.")
    parser.add_argument("case1", type=str, help="Path to first case directory (e.g. run_20250804_103043)")
    parser.add_argument("case2", type=str, help="Path to second case directory (e.g. run_20250803_115521)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory to store comparison plots and statistics")
    parser.add_argument("--soil-layers", type=int, default=10, help="Number of soil 2D layers to plot (default: 10)")
    parser.add_argument("--pft-count", type=int, default=16, help="Number of PFTs to plot for 1D (default: 16)")
    scalar_group = parser.add_mutually_exclusive_group()
    scalar_group.add_argument("--plot-scalar", dest="plot_scalar", action='store_true', help="Enable plotting for scalar variables (default)")
    scalar_group.add_argument("--no-plot-scalar", dest="plot_scalar", action='store_false', help="Disable plotting for scalar variables")
    parser.set_defaults(plot_scalar=True)
    args = parser.parse_args()
    main(
        args.case1,
        args.case2,
        out_dir=args.out_dir,
        soil_layers=args.soil_layers,
        pft_count=args.pft_count,
        plot_scalar=args.plot_scalar,
    )