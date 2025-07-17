#!/usr/bin/env python3
"""
Fair CPU vs GPU comparison script.

This script runs the model on both CPU and GPU with identical settings
for a fair comparison of performance and results.
"""

import subprocess
import sys
import time
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_training(config_name, output_dir):
    """Run training with specified configuration."""
    logger.info(f"Running training with config: {config_name}")
    
    cmd = [
        sys.executable, "train_model.py",
        "--config", config_name,
        "--output_dir", output_dir,
        "--log_level", "INFO"
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    duration = end_time - start_time
    
    if result.returncode == 0:
        logger.info(f"Training completed successfully in {duration:.2f} seconds")
        return True, duration, result.stdout
    else:
        logger.error(f"Training failed: {result.stderr}")
        return False, duration, result.stderr


def load_metrics(output_dir):
    """Load metrics from the output directory."""
    metrics_file = Path(output_dir) / "metrics.csv"
    if metrics_file.exists():
        return pd.read_csv(metrics_file)
    else:
        logger.warning(f"Metrics file not found: {metrics_file}")
        return None


def compare_results(cpu_dir, gpu_dir):
    """Compare results between CPU and GPU runs."""
    logger.info("Comparing results...")
    
    # Load metrics
    cpu_metrics = load_metrics(cpu_dir)
    gpu_metrics = load_metrics(gpu_dir)
    
    if cpu_metrics is None or gpu_metrics is None:
        logger.error("Could not load metrics for comparison")
        return
    
    # Compare scalar predictions
    cpu_scalar_file = Path(cpu_dir) / "predictions" / "predictions_scalar.csv"
    gpu_scalar_file = Path(gpu_dir) / "predictions" / "predictions_scalar.csv"
    
    if cpu_scalar_file.exists() and gpu_scalar_file.exists():
        cpu_scalar = pd.read_csv(cpu_scalar_file)
        gpu_scalar = pd.read_csv(gpu_scalar_file)
        
        # Calculate differences
        scalar_diff = np.abs(cpu_scalar.values - gpu_scalar.values)
        scalar_diff_pct = (scalar_diff / (np.abs(cpu_scalar.values) + 1e-8)) * 100
        
        logger.info("Scalar prediction differences:")
        for i, col in enumerate(cpu_scalar.columns):
            mean_diff = np.mean(scalar_diff[:, i])
            max_diff = np.max(scalar_diff[:, i])
            mean_diff_pct = np.mean(scalar_diff_pct[:, i])
            logger.info(f"  {col}: mean_diff={mean_diff:.6f}, max_diff={max_diff:.6f}, mean_diff_pct={mean_diff_pct:.2f}%")
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("FAIR COMPARISON SUMMARY")
    logger.info("="*50)
    
    if cpu_metrics is not None and gpu_metrics is not None:
        for metric in cpu_metrics.columns:
            if metric in gpu_metrics.columns:
                cpu_val = cpu_metrics[metric].iloc[0]
                gpu_val = gpu_metrics[metric].iloc[0]
                diff = abs(cpu_val - gpu_val)
                diff_pct = (diff / (abs(cpu_val) + 1e-8)) * 100
                logger.info(f"{metric}: CPU={cpu_val:.6f}, GPU={gpu_val:.6f}, diff={diff:.6f} ({diff_pct:.2f}%)")


def main():
    """Main comparison function."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Output directories
    cpu_dir = f"results_fair_cpu_{timestamp}"
    gpu_dir = f"results_fair_gpu_{timestamp}"
    
    logger.info("Starting fair CPU vs GPU comparison")
    logger.info("="*50)
    
    # Run CPU training
    logger.info("Step 1: Running CPU training...")
    cpu_success, cpu_duration, cpu_output = run_training("fair_cpu", cpu_dir)
    
    if not cpu_success:
        logger.error("CPU training failed. Aborting comparison.")
        return
    
    # Run GPU training
    logger.info("Step 2: Running GPU training...")
    gpu_success, gpu_duration, gpu_output = run_training("fair_gpu", gpu_dir)
    
    if not gpu_success:
        logger.error("GPU training failed. Aborting comparison.")
        return
    
    # Compare results
    logger.info("Step 3: Comparing results...")
    compare_results(cpu_dir, gpu_dir)
    
    # Performance comparison
    speedup = cpu_duration / gpu_duration
    logger.info("\n" + "="*50)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("="*50)
    logger.info(f"CPU duration: {cpu_duration:.2f} seconds")
    logger.info(f"GPU duration: {gpu_duration:.2f} seconds")
    logger.info(f"GPU speedup: {speedup:.2f}x")
    
    # Save comparison summary
    summary_file = f"fair_comparison_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("FAIR CPU vs GPU COMPARISON SUMMARY\n")
        f.write("="*50 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"CPU duration: {cpu_duration:.2f} seconds\n")
        f.write(f"GPU duration: {gpu_duration:.2f} seconds\n")
        f.write(f"GPU speedup: {speedup:.2f}x\n")
        f.write(f"CPU output dir: {cpu_dir}\n")
        f.write(f"GPU output dir: {gpu_dir}\n")
        f.write("\nCPU Output:\n")
        f.write(cpu_output)
        f.write("\nGPU Output:\n")
        f.write(gpu_output)
    
    logger.info(f"Comparison summary saved to: {summary_file}")
    logger.info("Fair comparison completed!")


if __name__ == "__main__":
    main() 