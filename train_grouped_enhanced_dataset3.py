#!/usr/bin/env python3
"""
Training script for Dataset 3 using the Grouped Enhanced Model architecture.
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

from config.training_config import get_grouped_enhanced_dataset3_config
from data.data_loader import DataLoader
from models.grouped_enhanced_model import GroupedEnhancedModel
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
    """Main training function for Dataset 3 with Grouped Enhanced Model."""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/grouped_enhanced_dataset3_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info("Starting training with Grouped Enhanced Model architecture for Dataset 3")
    
    try:
        # Get configuration
        config = get_grouped_enhanced_dataset3_config()
        
        # Update configuration paths to save in output directory
        config.training_config.model_save_path = os.path.join(output_dir, "model.pt")
        config.training_config.losses_save_path = os.path.join(output_dir, "training_validation_losses.csv")
        config.training_config.predictions_dir = os.path.join(output_dir, "predictions")
        
        logger.info("Updated output paths:")
        logger.info(f"  Model: {config.training_config.model_save_path}")
        logger.info(f"  Losses: {config.training_config.losses_save_path}")
        logger.info(f"  Predictions: {config.training_config.predictions_dir}")
        
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
        
        # Create grouped enhanced model
        logger.info("Creating Grouped Enhanced Model...")
        model = GroupedEnhancedModel(config.model_config, data_info)
        
        # Log model architecture details
        logger.info(f"Grouped Enhanced Model created with {model._count_parameters()} parameters")
        logger.info(f"Model architecture: {type(model).__name__}")
        
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
        logger.info(f"All results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 