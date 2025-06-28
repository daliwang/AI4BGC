#!/usr/bin/env python3
"""
Refactored Model Training Script

This script provides a flexible and modular approach to training climate models
with different input and output configurations. It uses a configuration-driven
approach to make it easy to modify training parameters without changing the core code.

Usage:
    python model_training_refactored.py [config_name]

Examples:
    python model_training_refactored.py default
    python model_training_refactored.py minimal
    python model_training_refactored.py extended
"""

import sys
import os
import logging
import argparse
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from config.training_config import (
    get_default_config, get_minimal_config, get_extended_config,
    TrainingConfigManager
)
from data.data_loader import DataLoader
from models.combined_model import CombinedModel, FlexibleCombinedModel
from training.trainer import ModelTrainer


def setup_logging(level: str = 'INFO') -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )


def get_config(config_name: str) -> TrainingConfigManager:
    """
    Get configuration based on name.
    
    Args:
        config_name: Name of the configuration to use
        
    Returns:
        TrainingConfigManager with the specified configuration
    """
    config_map = {
        'default': get_default_config,
        'minimal': get_minimal_config,
        'extended': get_extended_config
    }
    
    if config_name not in config_map:
        raise ValueError(f"Unknown config name: {config_name}. Available: {list(config_map.keys())}")
    
    return config_map[config_name]()


def create_custom_config() -> TrainingConfigManager:
    """
    Create a custom configuration for specific use cases.
    
    This function demonstrates how to create custom configurations
    for different training scenarios.
    """
    config = TrainingConfigManager()
    
    # Example: Train with only soil-related features
    config.update_data_config(
        x_list_columns_2d=['soil3c_vr', 'soil4c_vr'],
        y_list_columns_2d=['Y_soil3c_vr', 'Y_soil4c_vr'],
        x_list_columns_1d=['deadcrootc'],
        y_list_columns_1d=['Y_deadcrootc']
    )
    
    # Example: Use different time series features
    config.update_data_config(
        time_series_columns=['FLDS', 'PSRF', 'FSDS', 'TBOT']  # Removed QBOT and PRECTmms
    )
    
    # Example: Use different model architecture
    config.update_model_config(
        lstm_hidden_size=128,
        fc_hidden_size=64,
        transformer_layers=3
    )
    
    # Example: Use different training parameters
    config.update_training_config(
        num_epochs=100,
        batch_size=32,
        learning_rate=0.0005,
        scalar_loss_weight=1.0,
        vector_loss_weight=0.5,
        matrix_loss_weight=0.5
    )
    
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Flexible Climate Model Training')
    parser.add_argument(
        'config_name', 
        nargs='?', 
        default='default',
        choices=['default', 'minimal', 'extended', 'custom'],
        help='Configuration to use for training'
    )
    parser.add_argument(
        '--log-level', 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--custom-inputs',
        nargs='+',
        help='Custom input features (overrides config)'
    )
    parser.add_argument(
        '--custom-outputs',
        nargs='+',
        help='Custom output features (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting training with configuration: {args.config_name}")
        
        # Get configuration
        if args.config_name == 'custom':
            config = create_custom_config()
        else:
            config = get_config(args.config_name)
        
        # Override with command line arguments if provided
        if args.custom_inputs:
            logger.info(f"Overriding inputs with: {args.custom_inputs}")
            # This would need to be implemented based on your specific needs
            # For now, we'll just log it
        
        if args.custom_outputs:
            logger.info(f"Overriding outputs with: {args.custom_outputs}")
            # This would need to be implemented based on your specific needs
            # For now, we'll just log it
        
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
        
        # Normalize data
        logger.info("Normalizing data...")
        normalized_data = data_loader.normalize_data()
        
        # Split data
        logger.info("Splitting data...")
        split_data = data_loader.split_data(normalized_data)
        
        # Create model
        logger.info("Creating model...")
        model = CombinedModel(
            config.model_config,
            data_info
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
        logger.info("Starting training pipeline...")
        results = trainer.run_training_pipeline()
        
        logger.info("Training completed successfully!")
        logger.info(f"Final metrics: {results['metrics']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def run_experiment(experiment_name: str, config_overrides: dict) -> dict:
    """
    Run a specific experiment with custom configuration overrides.
    
    Args:
        experiment_name: Name of the experiment
        config_overrides: Dictionary of configuration overrides
        
    Returns:
        Training results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running experiment: {experiment_name}")
    
    # Start with default config
    config = get_default_config()
    
    # Apply overrides
    for key, value in config_overrides.items():
        if hasattr(config, key):
            getattr(config, key).update(**value)
        else:
            logger.warning(f"Unknown config key: {key}")
    
    # Run training
    return main()


if __name__ == "__main__":
    # Example usage scenarios
    
    # 1. Basic training with default configuration
    # python model_training_refactored.py default
    
    # 2. Quick test with minimal configuration
    # python model_training_refactored.py minimal
    
    # 3. Extended training with more features
    # python model_training_refactored.py extended
    
    # 4. Custom training with specific features
    # python model_training_refactored.py custom
    
    # 5. Example experiment
    # experiment_config = {
    #     'data_config': {
    #         'x_list_columns_2d': ['soil3c_vr'],
    #         'y_list_columns_2d': ['Y_soil3c_vr'],
    #         'x_list_columns_1d': ['deadcrootc'],
    #         'y_list_columns_1d': ['Y_deadcrootc']
    #     },
    #     'training_config': {
    #         'num_epochs': 10,
    #         'batch_size': 8
    #     }
    # }
    # results = run_experiment("soil_only_experiment", experiment_config)
    
    main() 