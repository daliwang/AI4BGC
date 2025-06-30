#!/usr/bin/env python3
"""
Training script for Combined Dataset (both datasets).
"""

import os
import sys
import logging
import torch
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.training_config import get_combined_dataset_config
from data.data_loader import DataLoader
from models.combined_model import CombinedModel
from training.trainer import ModelTrainer
from utils.gpu_monitor import GPUMonitor

def setup_logging(output_dir: str):
    """Setup logging configuration."""
    log_file = os.path.join(output_dir, "training.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main training function for Combined Dataset."""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/combined_dataset_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info("Starting training with configuration: combined_dataset")
    
    try:
        # Get configuration
        config = get_combined_dataset_config()
        logger.info("Initializing data loader...")
        
        # Initialize data loader
        data_loader = DataLoader(config.data_config, config.preprocessing_config)
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        data_loader.load_data()
        data_loader.preprocess_data()
        
        # Get data info for model creation
        data_info = data_loader.get_data_info()
        
        # Normalize data
        logger.info("Normalizing data...")
        normalized_data = data_loader.normalize_data()
        
        # Split data
        logger.info("Splitting data...")
        split_data = data_loader.split_data(normalized_data)
        train_data = split_data['train']
        test_data = split_data['test']
        
        # Create model
        logger.info("Creating model...")
        model = CombinedModel(config.model_config, data_info)
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = ModelTrainer(
            training_config=config.training_config,
            model=model,
            train_data=train_data,
            test_data=test_data,
            scalers=data_loader.scalers,
            data_info=data_info
        )
        
        # Start training
        logger.info("Starting training pipeline...")
        results = trainer.run_training_pipeline()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 