#!/usr/bin/env python3
"""
CNP Model Training Script

This script trains the CNP (Carbon-Nitrogen-Phosphorus) model based on the 
CNP_IO_list1.txt structure with the following architecture:

- LSTM for 6 time-series variables (20 years)
- FC for surface properties (geographic, soil texture, P forms, PFT coverage)
- FC for 44 PFT characteristics parameters
- FC for water variables (optional)
- FC for scalar variables
- FC for 1D variables
- CNN for 2D variables
- Transformer encoder for feature fusion
- Multi-task perceptrons for separate predictions

Usage:
    python train_cnp_model.py [--include-water] [--no-water]
"""

import sys
import os
import logging
import argparse
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from config.training_config import get_cnp_model_config, get_cnp_model_config_no_water
from data.data_loader import DataLoader
from models.cnp_combined_model import CNPCombinedModel
from training.trainer import ModelTrainer


def setup_logging(level: str = 'INFO') -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('cnp_training.log')
        ]
    )


def main():
    """Main training function for CNP model."""
    parser = argparse.ArgumentParser(description='CNP Model Training')
    parser.add_argument(
        '--include-water', 
        action='store_true',
        help='Include water variables in both input and output'
    )
    parser.add_argument(
        '--no-water', 
        action='store_true',
        help='Exclude water variables from both input and output'
    )
    parser.add_argument(
        '--log-level', 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--output-dir',
        default='cnp_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Determine water inclusion
        if args.include_water and args.no_water:
            raise ValueError("Cannot specify both --include-water and --no-water")
        
        include_water = args.include_water if args.include_water else not args.no_water
        logger.info(f"Water variables included: {include_water}")
        
        # Get configuration
        if include_water:
            config = get_cnp_model_config(include_water=True)
            logger.info("Using CNP configuration with water variables")
        else:
            config = get_cnp_model_config(include_water=False)
            logger.info("Using CNP configuration without water variables")
        
        # Override training parameters if specified
        config.update_training_config(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            model_save_path=str(output_dir / "cnp_model.pt"),
            losses_save_path=str(output_dir / "cnp_training_losses.csv"),
            predictions_dir=str(output_dir / "cnp_predictions")
        )
        
        # Initialize data loader
        logger.info("Initializing data loader...")
        data_loader = DataLoader(
            config.data_config,
            config.preprocessing_config
        )
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        df = data_loader.load_data()
        df = data_loader.preprocess_data()
        
        # Get data information
        data_info = data_loader.get_data_info()
        logger.info(f"Data info: {data_info}")
        
        # Log variable counts
        logger.info(f"Time series variables: {len(data_info['time_series_columns'])}")
        logger.info(f"Surface properties: {len(data_info['static_columns'])}")
        logger.info(f"1D input variables: {len(data_info['x_list_columns_1d'])}")
        logger.info(f"2D input variables: {len(data_info['x_list_columns_2d'])}")
        logger.info(f"1D output variables: {len(data_info['y_list_columns_1d'])}")
        logger.info(f"2D output variables: {len(data_info['y_list_columns_2d'])}")
        
        # Normalize data
        logger.info("Normalizing data...")
        normalized_data = data_loader.normalize_data()
        
        # Split data
        logger.info("Splitting data...")
        split_data = data_loader.split_data(normalized_data)
        
        # Create CNP model
        logger.info("Creating CNP model...")
        
        # Calculate actual data dimensions
        actual_1d_size = None
        actual_2d_channels = None
        
        # Check if we have 1D data and get its actual size
        if split_data['train']['list_1d']:
            actual_1d_size = len(data_info['x_list_columns_1d']) * config.model_config.vector_length
            logger.info(f"Detected actual 1D input size: {actual_1d_size}")
        
        # Check if we have 2D data and get its actual channels
        if split_data['train']['list_2d']:
            actual_2d_channels = len(data_info['x_list_columns_2d'])
            logger.info(f"Detected actual 2D channels: {actual_2d_channels}")
        
        model = CNPCombinedModel(
            config.model_config,
            data_info,
            include_water=include_water,
            actual_1d_size=actual_1d_size,
            actual_2d_channels=actual_2d_channels
        )
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = ModelTrainer(
            config.training_config,
            model,
            split_data['train'],
            split_data['test'],
            normalized_data['scalers'],
            data_info
        )
        
        # Run training pipeline
        logger.info("Starting CNP model training pipeline...")
        results = trainer.run_training_pipeline()
        
        logger.info("CNP model training completed successfully!")
        logger.info(f"Final metrics: {results['metrics']}")
        
        # Save results to output directory
        import json
        with open(output_dir / "cnp_metrics.json", "w") as f:
            json.dump(results['metrics'], f, indent=2)
        
        # Save model configuration
        with open(output_dir / "cnp_config.json", "w") as f:
            config_dict = {
                'include_water': include_water,
                'data_info': data_info,
                'model_config': config.model_config.__dict__,
                'training_config': config.training_config.__dict__
            }
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"CNP model results saved to: {output_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"CNP model training failed: {e}")
        raise


if __name__ == "__main__":
    main() 