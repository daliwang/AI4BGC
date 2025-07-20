import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

def plot_gt_vs_pred(gt, pred, title, save_path):
    plt.figure(figsize=(6,6))
    plt.scatter(gt, pred, alpha=0.5)
    plt.plot([gt.min(), gt.max()], [gt.min(), gt.max()], 'r--')
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_pair(gt_path, pred_path, label, out_dir, per_column=False):
    gt = pd.read_csv(gt_path)
    pred = pd.read_csv(pred_path)
    if per_column:
        # Per-column comparison for scalar
        for col in gt.columns:
            if col in pred.columns:
                print(f"Analyzing variable: {col}")
                gt_col = gt[col].values.flatten()
                pred_col = pred[col].values.flatten()
                rmse = np.sqrt(mean_squared_error(gt_col, pred_col))
                mae = mean_absolute_error(gt_col, pred_col)
                r2 = r2_score(gt_col, pred_col)
                print(f"{label} - {col}: RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
                plot_gt_vs_pred(gt_col, pred_col, f"{label} {col} GT vs Pred", os.path.join(out_dir, f"{label}_{col}_gt_vs_pred.png"))
            else:
                print(f"Column {col} missing in predictions for {label}")
        return None
    else:
        # Flatten if needed
        gt_flat = gt.values.flatten()
        pred_flat = pred.values.flatten()
        # Metrics
        rmse = np.sqrt(mean_squared_error(gt_flat, pred_flat))
        mae = mean_absolute_error(gt_flat, pred_flat)
        r2 = r2_score(gt_flat, pred_flat)
        print(f"{label} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        # Plot
        plot_gt_vs_pred(gt_flat, pred_flat, f"{label} GT vs Pred", os.path.join(out_dir, f"{label}_gt_vs_pred.png"))
        return {'rmse': rmse, 'mae': mae, 'r2': r2}

def analyze_1d(gt_path, pred_path, label, out_dir):
    gt = pd.read_csv(gt_path)
    pred = pd.read_csv(pred_path)
    # Read variable names from cnp_config.json
    config_path = os.path.join(out_dir, 'cnp_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        variable_names = config.get('data_info', {}).get('variables_1d_pft', None)
        if variable_names is None:
            print('[ERROR] Could not find variables_1d_pft in cnp_config.json!')
            return
    else:
        print(f'[ERROR] cnp_config.json not found in {out_dir}!')
        return
    num_vars = len(variable_names)
    num_pfts = 16  # pft0 is dropped in training
    num_samples = gt.shape[0]
    expected_cols = num_vars * num_pfts
    print(f"[DEBUG] 1D: num_samples={num_samples}, num_vars={num_vars}, num_pfts={num_pfts}, expected_cols={expected_cols}, actual_cols={gt.shape[1]}")
    if gt.shape[1] != expected_cols:
        print("[ERROR] Unexpected number of columns in 1D data!")
        print("Column names:", list(gt.columns))
        return
    gt_reshaped = gt.values.reshape(num_samples, num_vars, num_pfts)
    pred_reshaped = pred.values.reshape(num_samples, num_vars, num_pfts)
    # For each variable and each PFT column, compare
    for i, var in enumerate(variable_names):
        print(f"Analyzing variable: {var}")
        for j in range(num_pfts):
            gt_col = gt_reshaped[:, i, j]
            pred_col = pred_reshaped[:, i, j]
            rmse = np.sqrt(mean_squared_error(gt_col, pred_col))
            mae = mean_absolute_error(gt_col, pred_col)
            r2 = r2_score(gt_col, pred_col)
            print(f"{label} - {var} (PFT {j+1}): RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            # Plot only for the first column (PFT 1) as the real PFT0 has been dropped in training
            plot_gt_vs_pred(gt_col, pred_col, f"{label} {var} PFT{j+1} GT vs Pred", os.path.join(out_dir, f"{label}_{var}_PFT{j+1}_gt_vs_pred.png"))

def plot_train_val_accuracy(loss_csv, out_dir):
    df = pd.read_csv(loss_csv)
    plt.figure()
    if 'Train Loss' in df.columns and 'Validation Loss' in df.columns:
        plt.plot(df['Train Loss'], label='Train Loss')
        plt.plot(df['Validation Loss'], label='Validation Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.title('Train/Validation Loss')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'train_val_loss.png'))
        plt.close()
    else:
        print("train_loss or val_loss columns not found in loss CSV.")

if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='Postprocess CNP model results')
    parser.add_argument('results_dir', help='Results directory (run_xxxxx)')
    parser.add_argument('--no-plot-loss', action='store_false', dest='plot_loss', help='Do not plot train/val loss curve')
    parser.set_defaults(plot_loss=True)
    args = parser.parse_args()
    def main_with_flag(results_dir, plot_loss):
        pairs = [
            ('ground_truth_1d.csv', 'predictions_1d.csv', '1D'),
            #('ground_truth_2d.csv', 'predictions_2d.csv', '2D'),  # 2D analysis disabled
            ('ground_truth_scalar.csv', 'predictions_scalar.csv', 'Scalar')
        ]
        metrics = {}
        for gt_file, pred_file, label in pairs:
            gt_path = os.path.join(results_dir, 'cnp_predictions', gt_file)
            pred_path = os.path.join(results_dir, 'cnp_predictions', pred_file)
            if os.path.exists(gt_path) and os.path.exists(pred_path):
                if label == 'Scalar':
                    analyze_pair(gt_path, pred_path, label, results_dir, per_column=True)
                elif label == '1D':
                    analyze_1d(gt_path, pred_path, label, results_dir)
                else:
                    metrics[label] = analyze_pair(gt_path, pred_path, label, results_dir)
            else:
                print(f"Missing files for {label}: {gt_path}, {pred_path}")
        # Plot train/val accuracy if available and requested
        if plot_loss:
            loss_csv = os.path.join(results_dir, 'cnp_training_losses.csv')
            if os.path.exists(loss_csv):
                plot_train_val_accuracy(loss_csv, results_dir)
            else:
                print("Loss CSV not found for train/val accuracy plot.")
        # Print test metrics if available
        test_metrics_path = os.path.join(results_dir, 'test_metrics.csv')
        if os.path.exists(test_metrics_path):
            print("\nTest Metrics:")
            print(pd.read_csv(test_metrics_path))
        else:
            print("test_metrics.csv not found.")
    if len(sys.argv) < 2:
        print("Usage: python postprocess_result.py <results_dir> [--no-plot-loss]")
    else:
        main_with_flag(args.results_dir, args.plot_loss)