#!/usr/bin/env python3
"""
CPU vs GPU Comparison Script
Runs the same model on both CPU and GPU with 3 files and compares results.
"""

import os
import sys
import time
import logging
import argparse
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def setup_logging(log_level: str = 'INFO') -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('cpu_gpu_comparison.log'),
            logging.StreamHandler()
        ]
    )


def run_device_test(device: str, output_dir: str) -> Dict[str, Any]:
    """
    Run test on specified device (CPU or GPU).
    
    Args:
        device: 'cpu' or 'gpu'
        output_dir: Directory to save results
        
    Returns:
        Dictionary with test results
    """
    logger = logging.getLogger(__name__)
    
    config_name = f"full_model_test_{device}"
    experiment_dir = Path(output_dir) / f"{device}_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting {device.upper()} test...")
    logger.info(f"Output directory: {experiment_dir}")
    
    # Change to experiment directory
    os.chdir(experiment_dir)
    
    # Run the training script
    script_dir = Path(__file__).parent
    cmd = [
        sys.executable, 
        str(script_dir / "model_training_refactored.py"), 
        config_name,
        "--log-level", "INFO"
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Save output
        with open("stdout.log", "w") as f:
            f.write(result.stdout)
        
        with open("stderr.log", "w") as f:
            f.write(result.stderr)
        
        # Extract metrics from output
        metrics = extract_metrics_from_log(result.stdout)
        
        test_info = {
            "device": device,
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "duration_seconds": duration,
            "return_code": result.returncode,
            "output_dir": str(experiment_dir),
            "metrics": metrics
        }
        
        # Save test info
        with open("test_info.txt", "w") as f:
            for key, value in test_info.items():
                if key != 'metrics':
                    f.write(f"{key}: {value}\n")
            f.write(f"\nMetrics:\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        
        if result.returncode == 0:
            logger.info(f"{device.upper()} test completed successfully in {duration:.2f}s")
        else:
            logger.error(f"{device.upper()} test failed with return code {result.returncode}")
            
        return test_info
        
    except subprocess.TimeoutExpired:
        logger.error(f"{device.upper()} test timed out after 30 minutes")
        return {
            "device": device,
            "status": "timeout",
            "output_dir": str(experiment_dir)
        }
    except Exception as e:
        logger.error(f"{device.upper()} test failed with error: {e}")
        return {
            "device": device,
            "status": "error",
            "error": str(e),
            "output_dir": str(experiment_dir)
        }


def extract_metrics_from_log(log_output: str) -> Dict[str, float]:
    """Extract final metrics from training log output."""
    metrics = {}
    
    # Look for final metrics line
    lines = log_output.split('\n')
    for line in lines:
        if 'Final metrics:' in line:
            # Parse the metrics dictionary
            try:
                # Extract the metrics part
                metrics_str = line.split('Final metrics: ')[1]
                # Convert string representation to actual dict
                metrics_str = metrics_str.replace('np.float64(', '').replace(')', '')
                metrics_str = metrics_str.replace("'", '"')
                
                # Simple parsing for the metrics
                if 'scalar_rmse' in metrics_str:
                    scalar_rmse = float(metrics_str.split('scalar_rmse')[1].split(',')[0].split(':')[1].strip())
                    metrics['scalar_rmse'] = scalar_rmse
                
                if 'vector_rmse' in metrics_str:
                    vector_rmse = float(metrics_str.split('vector_rmse')[1].split(',')[0].split(':')[1].strip())
                    metrics['vector_rmse'] = vector_rmse
                
                if 'matrix_rmse' in metrics_str:
                    matrix_rmse = float(metrics_str.split('matrix_rmse')[1].split(',')[0].split(':')[1].strip())
                    metrics['matrix_rmse'] = matrix_rmse
                
            except Exception as e:
                print(f"Error parsing metrics: {e}")
                break
    
    return metrics


def compare_results(cpu_results: Dict[str, Any], gpu_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare CPU and GPU results.
    
    Args:
        cpu_results: CPU test results
        gpu_results: GPU test results
        
    Returns:
        Comparison results
    """
    logger = logging.getLogger(__name__)
    
    comparison = {
        "cpu_duration": cpu_results.get("duration_seconds", 0),
        "gpu_duration": gpu_results.get("duration_seconds", 0),
        "speedup": 0,
        "cpu_metrics": cpu_results.get("metrics", {}),
        "gpu_metrics": gpu_results.get("metrics", {}),
        "metric_differences": {},
        "status": "unknown"
    }
    
    # Calculate speedup
    if comparison["cpu_duration"] > 0 and comparison["gpu_duration"] > 0:
        comparison["speedup"] = comparison["cpu_duration"] / comparison["gpu_duration"]
    
    # Compare metrics
    cpu_metrics = comparison["cpu_metrics"]
    gpu_metrics = comparison["gpu_metrics"]
    
    for metric in ['scalar_rmse', 'vector_rmse', 'matrix_rmse']:
        if metric in cpu_metrics and metric in gpu_metrics:
            cpu_val = cpu_metrics[metric]
            gpu_val = gpu_metrics[metric]
            diff = abs(cpu_val - gpu_val)
            rel_diff = diff / max(cpu_val, gpu_val) * 100 if max(cpu_val, gpu_val) > 0 else 0
            
            comparison["metric_differences"][metric] = {
                "cpu": cpu_val,
                "gpu": gpu_val,
                "absolute_diff": diff,
                "relative_diff_percent": rel_diff
            }
    
    # Determine status
    if cpu_results.get("return_code") == 0 and gpu_results.get("return_code") == 0:
        # Check if metrics are similar (within 5% relative difference)
        max_rel_diff = max([
            comparison["metric_differences"].get(metric, {}).get("relative_diff_percent", 0)
            for metric in ['scalar_rmse', 'vector_rmse', 'matrix_rmse']
        ])
        
        if max_rel_diff < 5.0:
            comparison["status"] = "✅ PASSED - Results are similar"
        else:
            comparison["status"] = "⚠️  WARNING - Results differ significantly"
    else:
        comparison["status"] = "❌ FAILED - One or both tests failed"
    
    return comparison


def print_comparison_report(comparison: Dict[str, Any]) -> None:
    """Print a formatted comparison report."""
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*60)
    print("CPU vs GPU COMPARISON REPORT")
    print("="*60)
    
    # Performance comparison
    print(f"\n📊 PERFORMANCE:")
    print(f"   CPU Duration:  {comparison['cpu_duration']:.2f} seconds")
    print(f"   GPU Duration:  {comparison['gpu_duration']:.2f} seconds")
    print(f"   Speedup:       {comparison['speedup']:.2f}x")
    
    # Metrics comparison
    print(f"\n📈 METRICS COMPARISON:")
    for metric, diff_info in comparison["metric_differences"].items():
        print(f"   {metric.upper()}:")
        print(f"     CPU:  {diff_info['cpu']:.6f}")
        print(f"     GPU:  {diff_info['gpu']:.6f}")
        print(f"     Diff: {diff_info['absolute_diff']:.6f} ({diff_info['relative_diff_percent']:.2f}%)")
    
    # Overall status
    print(f"\n🎯 STATUS: {comparison['status']}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if "PASSED" in comparison['status']:
        print("   ✅ CPU and GPU results are consistent")
        print("   ✅ GPU implementation is working correctly")
        print("   ✅ Safe to proceed with full dataset training")
    elif "WARNING" in comparison['status']:
        print("   ⚠️  Results differ significantly - investigate further")
        print("   ⚠️  Check for numerical precision issues")
        print("   ⚠️  Consider adjusting model parameters")
    else:
        print("   ❌ Tests failed - fix issues before proceeding")
    
    print("="*60)


def main():
    """Main function for CPU vs GPU comparison."""
    parser = argparse.ArgumentParser(description='CPU vs GPU Comparison for Climate Model')
    parser.add_argument(
        '--output-dir',
        default='./cpu_gpu_comparison',
        help='Output directory for comparison results'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--skip-cpu',
        action='store_true',
        help='Skip CPU test (run only GPU)'
    )
    parser.add_argument(
        '--skip-gpu',
        action='store_true',
        help='Skip GPU test (run only CPU)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting CPU vs GPU comparison...")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cpu_results = None
    gpu_results = None
    
    # Run CPU test
    if not args.skip_cpu:
        logger.info("Running CPU test...")
        cpu_results = run_device_test("cpu", str(output_dir))
        logger.info("CPU test completed")
    else:
        logger.info("Skipping CPU test")
    
    # Run GPU test
    if not args.skip_gpu:
        logger.info("Running GPU test...")
        gpu_results = run_device_test("gpu", str(output_dir))
        logger.info("GPU test completed")
    else:
        logger.info("Skipping GPU test")
    
    # Compare results if both tests completed
    if cpu_results and gpu_results and cpu_results.get("return_code") == 0 and gpu_results.get("return_code") == 0:
        logger.info("Comparing results...")
        comparison = compare_results(cpu_results, gpu_results)
        
        # Save comparison results
        comparison_file = Path(args.output_dir) / "comparison_results.txt"
        comparison_file.parent.mkdir(parents=True, exist_ok=True)
        with open(comparison_file, "w") as f:
            f.write("CPU vs GPU Comparison Results\n")
            f.write("="*40 + "\n\n")
            f.write(f"CPU Duration: {comparison['cpu_duration']:.2f}s\n")
            f.write(f"GPU Duration: {comparison['gpu_duration']:.2f}s\n")
            f.write(f"Speedup: {comparison['speedup']:.2f}x\n\n")
            
            f.write("Metrics Comparison:\n")
            for metric, diff_info in comparison["metric_differences"].items():
                f.write(f"  {metric}: CPU={diff_info['cpu']:.6f}, GPU={diff_info['gpu']:.6f}, Diff={diff_info['relative_diff_percent']:.2f}%\n")
            
            f.write(f"\nStatus: {comparison['status']}\n")
        
        # Print report
        print_comparison_report(comparison)
        
        logger.info(f"Comparison results saved to {comparison_file}")
        
    else:
        logger.warning("Cannot compare results - one or both tests failed")
        if cpu_results:
            logger.info(f"CPU test status: {cpu_results.get('return_code', 'unknown')}")
        if gpu_results:
            logger.info(f"GPU test status: {gpu_results.get('return_code', 'unknown')}")


if __name__ == "__main__":
    main() 