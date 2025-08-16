import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from glob import glob

def plot_gt_vs_pred(gt, pred, title, save_path):
    plt.figure(figsize=(6,6))
    plt.scatter(gt, pred, alpha=0.5)
    plt.plot([gt.min(), gt.max()], [gt.min(), gt.max()], 'r--')
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.title(title)
    plt.tight_layout()
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
                
                # Calculate statistics
                gt_stats = {'min': np.nanmin(gt_col), 'max': np.nanmax(gt_col), 'sum': np.nansum(gt_col)}
                pred_stats = {'min': np.nanmin(pred_col), 'max': np.nanmax(pred_col), 'sum': np.nansum(pred_col)}
                
                # Print statistics
                print(f"  Ground Truth - min: {gt_stats['min']:.6g}, max: {gt_stats['max']:.6g}, sum: {gt_stats['sum']:.6g}")
                print(f"  Predictions - min: {pred_stats['min']:.6g}, max: {pred_stats['max']:.6g}, sum: {pred_stats['sum']:.6g}")
                
                rmse = np.sqrt(mean_squared_error(gt_col, pred_col))
                mae = mean_absolute_error(gt_col, pred_col)
                r2 = r2_score(gt_col, pred_col)
                print(f"  {label} - {col}: RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
                plot_gt_vs_pred(gt_col, pred_col, f"{label} {col} GT vs Pred", os.path.join(out_dir, f"{label}_{col}_gt_vs_pred.png"))
            else:
                print(f"Column {col} missing in predictions for {label}")
        return None
    else:
        # Flatten if needed
        gt_flat = gt.values.flatten()
        pred_flat = pred.values.flatten()
        
        # Calculate statistics
        gt_stats = {'min': np.nanmin(gt_flat), 'max': np.nanmax(gt_flat), 'sum': np.nansum(gt_flat)}
        pred_stats = {'min': np.nanmin(pred_flat), 'max': np.nanmax(pred_flat), 'sum': np.nansum(pred_flat)}
        
        # Print statistics
        print(f"Ground Truth - min: {gt_stats['min']:.6g}, max: {gt_stats['max']:.6g}, sum: {gt_stats['sum']:.6g}")
        print(f"Predictions - min: {pred_stats['min']:.6g}, max: {pred_stats['max']:.6g}, sum: {pred_stats['sum']:.6g}")
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(gt_flat, pred_flat))
        mae = mean_absolute_error(gt_flat, pred_flat)
        r2 = r2_score(gt_flat, pred_flat)
        print(f"{label} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        # Plot
        plot_gt_vs_pred(gt_flat, pred_flat, f"{label} GT vs Pred", os.path.join(out_dir, f"{label}_gt_vs_pred.png"))
        return {'rmse': rmse, 'mae': mae, 'r2': r2}

def analyze_1d_new_structure(results_dir, label, out_dir):
    """Analyze 1D data using the new directory structure with individual variable files"""
    gt_dir = os.path.join(results_dir, 'cnp_predictions', 'pft_1d_ground_truth')
    pred_dir = os.path.join(results_dir, 'cnp_predictions', 'pft_1d_predictions')
    
    if not os.path.exists(gt_dir) or not os.path.exists(pred_dir):
        print(f"Missing 1D directories: {gt_dir} or {pred_dir}")
        return
    
    # Get all ground truth files
    gt_files = glob(os.path.join(gt_dir, 'ground_truth_Y_*.csv'))
    if not gt_files:
        print(f"No ground truth files found in {gt_dir}")
        return
    
    print(f"Found {len(gt_files)} 1D variables to analyze")
    
    for gt_file in gt_files:
        # Extract variable name from filename
        var_name = os.path.basename(gt_file).replace('ground_truth_Y_', '').replace('.csv', '')
        
        # Find corresponding prediction file
        pred_file = os.path.join(pred_dir, f'predictions_Y_{var_name}.csv')
        
        if not os.path.exists(pred_file):
            print(f"Missing prediction file for {var_name}: {pred_file}")
            continue
            
        print(f"Analyzing variable: {var_name}")
        
        # Read data
        gt_data = pd.read_csv(gt_file)
        pred_data = pd.read_csv(pred_file)
        
        # Ensure same shape
        if gt_data.shape != pred_data.shape:
            print(f"Shape mismatch for {var_name}: GT {gt_data.shape} vs Pred {pred_data.shape}")
            continue
        
        # Analyze each PFT column (assuming columns are PFTs)
        num_pfts = gt_data.shape[1]
        print(f"  {var_name}: {num_pfts} PFT columns")
        
        for pft_idx in range(num_pfts):
            gt_col = gt_data.iloc[:, pft_idx].values
            pred_col = pred_data.iloc[:, pft_idx].values
            
            # Skip if all values are NaN
            if np.all(np.isnan(gt_col)) or np.all(np.isnan(pred_col)):
                continue
                
            # Remove NaN pairs
            valid_mask = ~(np.isnan(gt_col) | np.isnan(pred_col))
            if np.sum(valid_mask) < 10:  # Skip if too few valid points
                continue
                
            gt_valid = gt_col[valid_mask]
            pred_valid = pred_col[valid_mask]
            
            # Calculate statistics
            gt_stats = {'min': np.nanmin(gt_valid), 'max': np.nanmax(gt_valid), 'sum': np.nansum(gt_valid)}
            pred_stats = {'min': np.nanmin(pred_valid), 'max': np.nanmax(pred_valid), 'sum': np.nansum(pred_valid)}
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(gt_valid, pred_valid))
            mae = mean_absolute_error(gt_valid, pred_valid)
            r2 = r2_score(gt_valid, pred_valid)
            
            print(f"    PFT {pft_idx+1}: RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            print(f"      GT - min: {gt_stats['min']:.6g}, max: {gt_stats['max']:.6g}, sum: {gt_stats['sum']:.6g}")
            print(f"      Pred - min: {pred_stats['min']:.6g}, max: {pred_stats['max']:.6g}, sum: {pred_stats['sum']:.6g}")
            
            # Plot only for first few PFTs to avoid too many plots
            if pft_idx < 8:  # Limit to first 8 PFTs
                plot_gt_vs_pred(
                    gt_valid, pred_valid, 
                    f"{label} {var_name} PFT{pft_idx+1} GT vs Pred", 
                    os.path.join(out_dir, f"{label}_{var_name}_PFT{pft_idx+1}_gt_vs_pred.png")
                )

def analyze_1d(gt_path, pred_path, label, out_dir, results_dir):
    """Legacy function for old single-file 1D format - kept for compatibility"""
    gt = pd.read_csv(gt_path)
    pred = pd.read_csv(pred_path)
    # Read variable names from cnp_config.json
    config_path = os.path.join(results_dir, 'cnp_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        variable_names = config.get('data_info', {}).get('variables_1d_pft', None)
        if variable_names is None:
            print('[ERROR] Could not find variables_1d_pft in cnp_config.json!')
            return
    else:
        print(f'[ERROR] cnp_config.json not found in {results_dir}!')
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
            
            # Calculate statistics
            gt_stats = {'min': np.nanmin(gt_col), 'max': np.nanmax(gt_col), 'sum': np.nansum(gt_col)}
            pred_stats = {'min': np.nanmin(pred_col), 'max': np.nanmax(pred_col), 'sum': np.nansum(pred_col)}
            
            rmse = np.sqrt(mean_squared_error(gt_col, pred_col))
            mae = mean_absolute_error(gt_col, pred_col)
            r2 = r2_score(gt_col, pred_col)
            print(f"{label} - {var} (PFT {j+1}): RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            print(f"  GT - min: {gt_stats['min']:.6g}, max: {gt_stats['max']:.6g}, sum: {gt_stats['sum']:.6g}")
            print(f"  Pred - min: {pred_stats['min']:.6g}, max: {pred_stats['max']:.6g}, sum: {pred_stats['sum']:.6g}")
            
            # Plot only for the first column (PFT 1) as the real PFT0 has been dropped in training
            plot_gt_vs_pred(gt_col, pred_col, f"{label} {var} PFT{j+1} GT vs Pred", os.path.join(out_dir, f"{label}_{var}_PFT{j+1}_gt_vs_pred.png"))

def analyze_2d_new_structure(results_dir, label, out_dir):
    """Analyze 2D data using the new directory structure with individual variable files"""
    gt_dir = os.path.join(results_dir, 'cnp_predictions', 'soil_2d_ground_truth')
    pred_dir = os.path.join(results_dir, 'cnp_predictions', 'soil_2d_predictions')
    
    if not os.path.exists(gt_dir) or not os.path.exists(pred_dir):
        print(f"Missing 2D directories: {gt_dir} or {pred_dir}")
        return
    
    # Get all ground truth files
    gt_files = glob(os.path.join(gt_dir, 'ground_truth_Y_*.csv'))
    if not gt_files:
        print(f"No 2D ground truth files found in {gt_dir}")
        return
    
    print(f"Found {len(gt_files)} 2D variables to analyze")
    
    for gt_file in gt_files:
        # Extract variable name from filename
        var_name = os.path.basename(gt_file).replace('ground_truth_Y_', '').replace('.csv', '')
        
        # Find corresponding prediction file
        pred_file = os.path.join(pred_dir, f'predictions_Y_{var_name}.csv')
        
        if not os.path.exists(pred_file):
            print(f"Missing prediction file for {var_name}: {pred_file}")
            continue
            
        print(f"Analyzing 2D variable: {var_name}")
        
        # Read data
        gt_data = pd.read_csv(gt_file)
        pred_data = pd.read_csv(pred_file)
        
        # Ensure same shape
        if gt_data.shape != pred_data.shape:
            print(f"Shape mismatch for {var_name}: GT {gt_data.shape} vs Pred {pred_data.shape}")
            continue
        
        # 2D data has 18 columns × 10 layers = 180 total columns (only first 10 layers predicted)
        total_columns = gt_data.shape[1]
        expected_columns = 18 * 10  # 180
        
        if total_columns != expected_columns:
            print(f"  Warning: Expected {expected_columns} columns for 2D data, but found {total_columns}")
            if total_columns % 10 != 0:
                print(f"  Error: Number of columns ({total_columns}) is not divisible by 10")
                continue
            num_columns = total_columns // 10
            print(f"  Assuming {num_columns} columns with 10 layers each")
        else:
            num_columns = 18
            print(f"  {var_name}: {num_columns} columns, each with 10 layers ({total_columns} total columns)")
        
        # Analyze all 10 layers of the first column
        first_column_idx = 0
        layers_to_analyze = 10
        
        for layer_idx in range(layers_to_analyze):
            # Calculate the correct column index: first_column * 10 + layer
            col_idx = first_column_idx * 10 + layer_idx
            if col_idx >= total_columns:
                print(f"    Warning: Column index {col_idx} out of range for {total_columns} columns")
                continue
                
            gt_col = gt_data.iloc[:, col_idx].values
            pred_col = pred_data.iloc[:, col_idx].values
            
            # Skip if all values are NaN
            if np.all(np.isnan(gt_col)) or np.all(np.isnan(pred_col)):
                continue
                
            # Remove NaN pairs
            valid_mask = ~(np.isnan(gt_col) | np.isnan(pred_col))
            if np.sum(valid_mask) < 10:  # Skip if too few valid points
                continue
                
            gt_valid = gt_col[valid_mask]
            pred_valid = pred_col[valid_mask]
            
            # Calculate statistics
            gt_stats = {'min': np.nanmin(gt_valid), 'max': np.nanmax(gt_valid), 'sum': np.nansum(gt_valid)}
            pred_stats = {'min': np.nanmin(pred_valid), 'max': np.nanmax(pred_valid), 'sum': np.nansum(pred_valid)}
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(gt_valid, pred_valid))
            mae = mean_absolute_error(gt_valid, pred_valid)
            r2 = r2_score(gt_valid, pred_valid)
            
            print(f"    Layer {layer_idx+1}: RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            print(f"      GT - min: {gt_stats['min']:.6g}, max: {gt_stats['max']:.6g}, sum: {gt_stats['sum']:.6g}")
            print(f"      Pred - min: {pred_stats['min']:.6g}, max: {pred_stats['max']:.6g}, sum: {pred_stats['sum']:.6g}")
            
            # Plot for each layer
            plot_gt_vs_pred(
                gt_valid, pred_valid, 
                f"{label} {var_name} Layer{layer_idx+1} GT vs Pred", 
                os.path.join(out_dir, f"{label}_{var_name}_Layer{layer_idx+1}_gt_vs_pred.png")
            )

def analyze_2d(gt_path, pred_path, label, out_dir, results_dir):
    """Legacy function for old single-file 2D format - kept for compatibility"""
    gt = pd.read_csv(gt_path)
    pred = pd.read_csv(pred_path)
    # Read variable names from cnp_config.json
    config_path = os.path.join(results_dir, 'cnp_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        variable_names = config.get('data_info', {}).get('variables_2d_soil', None)
        if variable_names is None:
            print('[ERROR] Could not find variables_2d_soil in cnp_config.json!')
            return
    else:
        print(f'[ERROR] cnp_config.json not found in {results_dir}!')
        return
    num_vars = len(variable_names)
    num_columns = 18  # 2D soil data has 18 columns
    num_layers_per_column = 10  # Each column has 10 layers (only first 10 predicted)
    num_samples = gt.shape[0]
    expected_cols = num_vars * num_columns * num_layers_per_column
    print(f"[DEBUG] 2D: num_samples={num_samples}, num_vars={num_vars}, num_columns={num_columns}, layers_per_column={num_layers_per_column}, expected_cols={expected_cols}, actual_cols={gt.shape[1]}")
    if gt.shape[1] != expected_cols:
        print("[ERROR] Unexpected number of columns in 2D data!")
        print("Column names:", list(gt.columns))
        return
    gt_reshaped = gt.values.reshape(num_samples, num_vars, num_columns, num_layers_per_column)
    pred_reshaped = pred.values.reshape(num_samples, num_vars, num_columns, num_layers_per_column)
    # For each variable, analyze all 10 layers of the first column
    for i, var in enumerate(variable_names):
        print(f"Analyzing 2D variable: {var}")
        for j in range(10):  # Changed from 15 to 10 to analyze all predicted layers
            gt_col = gt_reshaped[:, i, 0, j]  # First column (index 0), all 10 layers
            pred_col = pred_reshaped[:, i, 0, j]  # First column (index 0), all 10 layers
            
            # Calculate statistics
            gt_stats = {'min': np.nanmin(gt_col), 'max': np.nanmax(gt_col), 'sum': np.nansum(gt_col)}
            pred_stats = {'min': np.nanmin(pred_col), 'max': np.nanmax(pred_col), 'sum': np.nansum(pred_col)}
            
            rmse = np.sqrt(mean_squared_error(gt_col, pred_col))
            mae = mean_absolute_error(gt_col, pred_col)
            r2 = r2_score(gt_col, pred_col)
            print(f"{label} - {var} (Layer {j+1}): RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            print(f"  GT - min: {gt_stats['min']:.6g}, max: {gt_stats['max']:.6g}, sum: {gt_stats['sum']:.6g}")
            print(f"  Pred - min: {pred_stats['min']:.6g}, max: {pred_stats['max']:.6g}, sum: {pred_stats['sum']:.6g}")
            
            plot_gt_vs_pred(gt_col, pred_col, f"{label} {var} Layer{j+1} GT vs Pred", os.path.join(out_dir, f"{label}_{var}_Layer{j+1}_gt_vs_pred.png"))

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
        # Ensure the directory exists
        os.makedirs(out_dir, exist_ok=True)
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
        # Create plots subdirectory
        plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Check for new directory structure first
        pft_gt_dir = os.path.join(results_dir, 'cnp_predictions', 'pft_1d_ground_truth')
        pft_pred_dir = os.path.join(results_dir, 'cnp_predictions', 'pft_1d_predictions')
        
        # Handle 1D data with new structure
        if os.path.exists(pft_gt_dir) and os.path.exists(pft_pred_dir):
            print("Using new 1D directory structure")
            analyze_1d_new_structure(results_dir, '1D', plots_dir)
        else:
            # Fall back to old single-file format
            print("Using legacy 1D single-file format")
            gt_path = os.path.join(results_dir, 'cnp_predictions', 'ground_truth_1d.csv')
            pred_path = os.path.join(results_dir, 'cnp_predictions', 'predictions_1d.csv')
            if os.path.exists(gt_path) and os.path.exists(pred_path):
                analyze_1d(gt_path, pred_path, '1D', plots_dir, results_dir)
            else:
                print("No 1D data found in either format")
        
        # Handle scalar data
        scalar_gt = os.path.join(results_dir, 'cnp_predictions', 'ground_truth_scalar.csv')
        scalar_pred = os.path.join(results_dir, 'cnp_predictions', 'predictions_scalar.csv')
        if os.path.exists(scalar_gt) and os.path.exists(scalar_pred):
            print("Analyzing scalar data...")
            analyze_pair(scalar_gt, scalar_pred, 'Scalar', plots_dir, per_column=True)
        else:
            print("Scalar data files not found")
        
        # Handle 2D data if available
        soil_gt_dir = os.path.join(results_dir, 'cnp_predictions', 'soil_2d_ground_truth')
        soil_pred_dir = os.path.join(results_dir, 'cnp_predictions', 'soil_2d_predictions')
        if os.path.exists(soil_gt_dir) and os.path.exists(soil_pred_dir):
            print("Using new 2D directory structure")
            analyze_2d_new_structure(results_dir, '2D', plots_dir)
        else:
            # Fall back to old single-file format
            print("Using legacy 2D single-file format")
            soil_gt = os.path.join(results_dir, 'cnp_predictions', 'ground_truth_2d.csv')
            soil_pred = os.path.join(results_dir, 'cnp_predictions', 'predictions_2d.csv')
            if os.path.exists(soil_gt) and os.path.exists(soil_pred):
                analyze_2d(soil_gt, soil_pred, '2D', plots_dir, results_dir)
            else:
                print("No 2D data found in either format")
        
        # Plot train/val accuracy if available and requested
        if plot_loss:
            loss_csv = os.path.join(results_dir, 'cnp_training_losses.csv')
            if os.path.exists(loss_csv):
                plot_train_val_accuracy(loss_csv, plots_dir)
            else:
                print("Loss CSV not found for train/val accuracy plot.")
        
        # Print test metrics if available
        test_metrics_path = os.path.join(results_dir, 'cnp_predictions','test_metrics.csv')
        if os.path.exists(test_metrics_path):
            print("\nTest Metrics:")
            print(pd.read_csv(test_metrics_path))
        else:
            print("test_metrics.csv not found.")
    
    if len(sys.argv) < 2:
        print("Usage: python cnp_result_validationplot.py <results_dir> [--no-plot-loss]")
    else:
        main_with_flag(args.results_dir, args.plot_loss)