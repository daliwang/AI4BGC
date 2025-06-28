#!/usr/bin/env python3
"""
Batch training script for running multiple experiments with different configurations.
"""

import os
import sys
import time
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.training_config import TrainingConfigManager


def setup_logging(log_level: str = 'INFO') -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('batch_training.log'),
            logging.StreamHandler()
        ]
    )


def run_single_experiment(config_name: str, output_dir: str, log_level: str = 'INFO') -> Dict[str, Any]:
    """
    Run a single experiment with the specified configuration.
    
    Args:
        config_name: Name of the configuration to use
        output_dir: Directory to save results
        log_level: Logging level
        
    Returns:
        Dictionary with experiment results
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    experiment_dir = Path(output_dir) / f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting experiment: {config_name}")
    logger.info(f"Output directory: {experiment_dir}")
    
    # Change to experiment directory
    os.chdir(experiment_dir)
    
    # Run the training script
    cmd = [
        sys.executable, 
        "../model_training_refactored.py", 
        config_name,
        "--log-level", log_level
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Save output
        with open("stdout.log", "w") as f:
            f.write(result.stdout)
        
        with open("stderr.log", "w") as f:
            f.write(result.stderr)
        
        # Save experiment info
        experiment_info = {
            "config_name": config_name,
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "duration_seconds": duration,
            "return_code": result.returncode,
            "output_dir": str(experiment_dir)
        }
        
        with open("experiment_info.txt", "w") as f:
            for key, value in experiment_info.items():
                f.write(f"{key}: {value}\n")
        
        if result.returncode == 0:
            logger.info(f"Experiment {config_name} completed successfully in {duration:.2f}s")
        else:
            logger.error(f"Experiment {config_name} failed with return code {result.returncode}")
            
        return experiment_info
        
    except subprocess.TimeoutExpired:
        logger.error(f"Experiment {config_name} timed out after 1 hour")
        return {
            "config_name": config_name,
            "status": "timeout",
            "output_dir": str(experiment_dir)
        }
    except Exception as e:
        logger.error(f"Experiment {config_name} failed with error: {e}")
        return {
            "config_name": config_name,
            "status": "error",
            "error": str(e),
            "output_dir": str(experiment_dir)
        }


def run_batch_experiments(configs: List[str], output_base_dir: str, log_level: str = 'INFO') -> List[Dict[str, Any]]:
    """
    Run multiple experiments in sequence.
    
    Args:
        configs: List of configuration names to run
        output_base_dir: Base directory for all experiment outputs
        log_level: Logging level
        
    Returns:
        List of experiment results
    """
    logger = logging.getLogger(__name__)
    
    # Create base output directory
    batch_dir = Path(output_base_dir) / f"batch_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting batch run with {len(configs)} experiments")
    logger.info(f"Batch directory: {batch_dir}")
    
    results = []
    
    for i, config_name in enumerate(configs, 1):
        logger.info(f"Running experiment {i}/{len(configs)}: {config_name}")
        
        # Create experiment-specific output directory
        experiment_output_dir = batch_dir / config_name
        
        result = run_single_experiment(config_name, str(experiment_output_dir), log_level)
        results.append(result)
        
        # Add delay between experiments to avoid resource conflicts
        if i < len(configs):
            logger.info("Waiting 30 seconds before next experiment...")
            time.sleep(30)
    
    # Save batch summary
    summary_file = batch_dir / "batch_summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"Batch run completed at {datetime.now().isoformat()}\n")
        f.write(f"Total experiments: {len(configs)}\n")
        f.write(f"Successful: {sum(1 for r in results if r.get('return_code') == 0)}\n")
        f.write(f"Failed: {sum(1 for r in results if r.get('return_code') != 0)}\n\n")
        
        for result in results:
            f.write(f"Config: {result['config_name']}\n")
            f.write(f"Status: {'Success' if result.get('return_code') == 0 else 'Failed'}\n")
            f.write(f"Duration: {result.get('duration_seconds', 'N/A'):.2f}s\n")
            f.write(f"Output: {result.get('output_dir', 'N/A')}\n\n")
    
    logger.info(f"Batch run completed. Summary saved to {summary_file}")
    return results


def create_custom_batch_configs() -> List[Dict[str, Any]]:
    """
    Create custom batch configurations for different experiments.
    
    Returns:
        List of configuration dictionaries
    """
    configs = [
        {
            "name": "minimal_test",
            "description": "Minimal configuration for quick testing",
            "config_name": "minimal"
        },
        {
            "name": "full_model_test", 
            "description": "Full model with 3 files for testing",
            "config_name": "full_model_test"
        },
        {
            "name": "production_full",
            "description": "Production training with full dataset",
            "config_name": "full_dataset"
        }
    ]
    
    return configs


def main():
    """Main function for batch training."""
    parser = argparse.ArgumentParser(description='Batch Training for Climate Model')
    parser.add_argument(
        '--configs',
        nargs='+',
        default=['minimal', 'full_model_test', 'full_dataset'],
        help='List of configurations to run'
    )
    parser.add_argument(
        '--output-dir',
        default='./batch_results',
        help='Base output directory for all experiments'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='List available configurations and exit'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    if args.list_configs:
        logger.info("Available configurations:")
        configs = create_custom_batch_configs()
        for config in configs:
            logger.info(f"  {config['name']}: {config['description']}")
        return
    
    logger.info(f"Starting batch training with configurations: {args.configs}")
    
    # Run batch experiments
    results = run_batch_experiments(args.configs, args.output_dir, args.log_level)
    
    # Print summary
    successful = sum(1 for r in results if r.get('return_code') == 0)
    total = len(results)
    
    logger.info(f"Batch training completed: {successful}/{total} experiments successful")
    
    for result in results:
        status = "✅ Success" if result.get('return_code') == 0 else "❌ Failed"
        logger.info(f"  {result['config_name']}: {status}")


if __name__ == "__main__":
    main() 