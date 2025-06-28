#!/usr/bin/env python3
"""
Analyze scalar RMSE with and without Y_COL_FIRE_CLOSS
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_scalar_predictions(cpu_file, gpu_file):
    """Analyze scalar predictions to understand the contribution of Y_COL_FIRE_CLOSS."""
    
    # Load predictions
    cpu_pred = pd.read_csv(cpu_file)
    gpu_pred = pd.read_csv(gpu_file)
    
    print("=== SCALAR PREDICTIONS ANALYSIS ===")
    print(f"CPU predictions shape: {cpu_pred.shape}")
    print(f"GPU predictions shape: {gpu_pred.shape}")
    
    # Check column order differences
    print(f"\nCPU columns: {list(cpu_pred.columns)}")
    print(f"GPU columns: {list(gpu_pred.columns)}")
    
    # Reorder GPU columns to match CPU if needed
    if list(cpu_pred.columns) != list(gpu_pred.columns):
        gpu_pred = gpu_pred[list(cpu_pred.columns)]
        print("Reordered GPU columns to match CPU")
    
    # Analyze each column
    print("\n=== COLUMN-BY-COLUMN ANALYSIS ===")
    
    for col in cpu_pred.columns:
        cpu_vals = cpu_pred[col].values
        gpu_vals = gpu_pred[col].values
        
        cpu_mean = np.mean(cpu_vals)
        cpu_std = np.std(cpu_vals)
        gpu_mean = np.mean(gpu_vals)
        gpu_std = np.std(gpu_vals)
        
        # Calculate difference
        diff = abs(cpu_mean - gpu_mean)
        rel_diff = (diff / max(abs(cpu_mean), abs(gpu_mean))) * 100 if max(abs(cpu_mean), abs(gpu_mean)) > 0 else 0
        
        print(f"\n{col}:")
        print(f"  CPU: mean={cpu_mean:.6f}, std={cpu_std:.6f}")
        print(f"  GPU: mean={gpu_mean:.6f}, std={gpu_std:.6f}")
        print(f"  Diff: {diff:.6f} ({rel_diff:.2f}%)")
        
        # Check if this column might be causing the large RMSE difference
        if rel_diff > 50:
            print(f"  ⚠️  LARGE DIFFERENCE - This column may be contributing to RMSE discrepancy")
    
    # Calculate estimated RMSE without Y_COL_FIRE_CLOSS
    print("\n=== ESTIMATED RMSE WITHOUT Y_COL_FIRE_CLOSS ===")
    
    # Get the columns excluding Y_COL_FIRE_CLOSS
    non_fire_cols = [col for col in cpu_pred.columns if col != 'Y_COL_FIRE_CLOSS']
    
    print(f"Columns excluding Y_COL_FIRE_CLOSS: {non_fire_cols}")
    
    # Calculate variance for each column (proxy for contribution to RMSE)
    cpu_variances = {}
    gpu_variances = {}
    
    for col in non_fire_cols:
        cpu_variances[col] = np.var(cpu_pred[col].values)
        gpu_variances[col] = np.var(gpu_pred[col].values)
    
    print("\nVariance analysis (lower = more stable predictions):")
    for col in non_fire_cols:
        print(f"  {col}: CPU_var={cpu_variances[col]:.6f}, GPU_var={gpu_variances[col]:.6f}")
    
    # Estimate relative contribution
    total_cpu_var = sum(cpu_variances.values())
    total_gpu_var = sum(gpu_variances.values())
    
    print(f"\nTotal variance (non-fire columns):")
    print(f"  CPU: {total_cpu_var:.6f}")
    print(f"  GPU: {total_gpu_var:.6f}")
    
    # Estimate RMSE ratio
    if total_cpu_var > 0 and total_gpu_var > 0:
        estimated_rmse_ratio = np.sqrt(total_gpu_var / total_cpu_var)
        print(f"  Estimated RMSE ratio (GPU/CPU): {estimated_rmse_ratio:.3f}")
        
        # Compare with actual ratio from comparison
        actual_cpu_rmse = 0.0736
        actual_gpu_rmse = 0.2099
        actual_ratio = actual_gpu_rmse / actual_cpu_rmse
        print(f"  Actual RMSE ratio (GPU/CPU): {actual_ratio:.3f}")
        
        if abs(estimated_rmse_ratio - actual_ratio) < 0.5:
            print("  ✅ Estimated ratio is close to actual ratio")
        else:
            print("  ⚠️  Estimated ratio differs significantly from actual ratio")
    
    # Check Y_COL_FIRE_CLOSS specifically
    if 'Y_COL_FIRE_CLOSS' in cpu_pred.columns:
        fire_col = 'Y_COL_FIRE_CLOSS'
        cpu_fire_vals = cpu_pred[fire_col].values
        gpu_fire_vals = gpu_pred[fire_col].values
        
        cpu_fire_mean = np.mean(cpu_fire_vals)
        gpu_fire_mean = np.mean(gpu_fire_vals)
        cpu_fire_var = np.var(cpu_fire_vals)
        gpu_fire_var = np.var(gpu_fire_vals)
        
        fire_diff = abs(cpu_fire_mean - gpu_fire_mean)
        fire_rel_diff = (fire_diff / max(abs(cpu_fire_mean), abs(gpu_fire_mean))) * 100 if max(abs(cpu_fire_mean), abs(gpu_fire_mean)) > 0 else 0
        
        print(f"\n=== Y_COL_FIRE_CLOSS ANALYSIS ===")
        print(f"CPU: mean={cpu_fire_mean:.6f}, var={cpu_fire_var:.6f}")
        print(f"GPU: mean={gpu_fire_mean:.6f}, var={gpu_fire_var:.6f}")
        print(f"Diff: {fire_diff:.6f} ({fire_rel_diff:.2f}%)")
        
        if fire_rel_diff > 50:
            print("  ⚠️  Y_COL_FIRE_CLOSS shows large differences between CPU and GPU")
            print("  This column is likely contributing significantly to the RMSE discrepancy")

def main():
    """Main function."""
    # Paths to the prediction files
    cpu_file = "cpu_gpu_comparison/cpu_test_20250628_091137/predictions/scalar_predictions.csv"
    gpu_file = "cpu_gpu_comparison/cpu_test_20250628_091137/cpu_gpu_comparison/gpu_test_20250628_091212/predictions/scalar_predictions.csv"
    
    if not Path(cpu_file).exists():
        print(f"CPU file not found: {cpu_file}")
        return
    
    if not Path(gpu_file).exists():
        print(f"GPU file not found: {gpu_file}")
        return
    
    analyze_scalar_predictions(cpu_file, gpu_file)

if __name__ == "__main__":
    main() 