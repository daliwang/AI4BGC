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
    python train_cnp_model.py [--with-water]
"""

import sys
import os
import json
import torch
import torch.nn as nn
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from config.training_config import get_cnp_model_config
from data.data_loader import DataLoader
from models.cnp_combined_model import CNPCombinedModel
from training.trainer import ModelTrainer


def setup_logging(log_file: str, level: str = 'INFO') -> None:
    """Set up logging configuration with a specific log file."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )


def main():
    """Main training function for CNP model."""
    parser = argparse.ArgumentParser(description='CNP Model Training')
    # turn off water for now
    # parser.add_argument(
    #     '--with-water',
    #     action='store_true',
    #     help='Include water variables in both input and output (default: no water)'
    # )
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
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.0001,
        help='Learning rate'
    )

    parser.add_argument(
        '--use-trendy1',
        action='store_true',
        help='Include Trendy_1_data_CNP dataset'
    )
    parser.add_argument(
        '--use-trendy05',
        action='store_true',
        help='Include Trend_05_data_CNP dataset'
    )
    parser.add_argument(
        '--variable-list',
        type=str,
        default=None,
        help='Path to variable list file (e.g., CNP_IO_list_general.txt) for dynamic configuration'
    )
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging with timestamped log file in output directory
    log_file = output_dir / f"cnp_training_{timestamp}.log"
    setup_logging(str(log_file), args.log_level if args.log_level else 'WARNING')
    logger = logging.getLogger(__name__)
    logger.info(f"Output directory: {output_dir}")
    
    if not args.use_trendy1 and not args.use_trendy05:
        args.use_trendy1 = True
        args.use_trendy05 = False

    try:
        # turn off water for now
        include_water = False
        # include_water = args.with_water
        logger.info(f"Water variables included: {include_water}")
        
        # Get configuration
        if args.variable_list is not None:
            from config.training_config import get_cnp_combined_config
            config = get_cnp_combined_config(
                use_trendy1=args.use_trendy1,
                use_trendy05=args.use_trendy05,
                max_files=None,  # You can add a CLI arg for this if needed
                include_water=include_water,
                variable_list_path=args.variable_list
            )
            logger.info(f"Using CNP configuration from variable list file: {args.variable_list}")
        else:
            from config.training_config import get_cnp_model_config
            config = get_cnp_model_config(
                include_water=include_water,
                use_trendy1=args.use_trendy1,
                use_trendy05=args.use_trendy05
            )
            logger.info(f"Using default CNP configuration{' with water' if include_water else ' without water'}")
        # Set train/validation split to 70/30
        config.update_data_config(train_split=0.7)
        # Ensure GPU and all files
        config.update_training_config(device='cuda')
        config.update_data_config(max_files=None)
        # Turn off GPU monitoring and debug logging
        config.update_training_config(log_gpu_memory=False, log_gpu_utilization=False)
        # Override training parameters if specified
        effective_lr = args.learning_rate if args.learning_rate is not None else config.training_config.learning_rate
        config.update_training_config(
            num_epochs=args.epochs,  
            batch_size=args.batch_size,
            learning_rate=effective_lr,
            model_save_path=str(output_dir / "cnp_model.pt"),
            losses_save_path=str(output_dir / "cnp_training_losses.csv"),
            predictions_dir=str(output_dir / "cnp_predictions"),
            use_early_stopping=False
        )
        logger.info(f"Effective learning rate for this run: {effective_lr}")
        
        # --- FORCE 2D COLUMN ALIGNMENT FOR SAFETY ---
        #aligned_2d_vars = [
        #    'cwdc_vr', 'cwdn_vr', 'secondp_vr', 'cwdp_vr',
        #    'litr1c_vr', 'litr2c_vr', 'litr3c_vr',
        #    'litr1n_vr', 'litr2n_vr', 'litr3n_vr',
        #    'litr1p_vr', 'litr2p_vr', 'litr3p_vr',
        #    'sminn_vr', 'smin_no3_vr', 'smin_nh4_vr',
        #    'soil1c_vr', 'soil2c_vr', 'soil3c_vr', 'soil4c_vr',
        #    'soil1n_vr', 'soil2n_vr', 'soil3n_vr', 'soil4n_vr',
        #    'soil1p_vr', 'soil2p_vr', 'soil3p_vr', 'soil4p_vr'
        #]
        #config.data_config.x_list_columns_2d = aligned_2d_vars
        #config.data_config.y_list_columns_2d = ['Y_' + v for v in aligned_2d_vars]
        # print('x_list_columns_2d (forced):', config.data_config.x_list_columns_2d)
        # print('y_list_columns_2d (forced):', config.data_config.y_list_columns_2d)
        assert config.data_config.y_list_columns_2d == ['Y_' + v for v in config.data_config.x_list_columns_2d], \
            f"2D columns not aligned!\nX: {config.data_config.x_list_columns_2d}\nY: {config.data_config.y_list_columns_2d}"

        # Initialize data loader
        logger.info("Initializing data loader...")
        data_loader = DataLoader(
            config.data_config,
            config.preprocessing_config
        )
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        data_loader.load_data()
        data_loader.preprocess_data()
        data_info = data_loader.get_data_info()
        # One-time config dump: resolved variables and training params
        resolved = {
            'time_series_columns': data_info.get('time_series_columns', []),
            'static_columns': data_info.get('static_columns', []),
            'pft_param_columns': data_info.get('pft_param_columns', []),
            'x_list_scalar_columns': data_info.get('x_list_scalar_columns', []),
            'y_list_scalar_columns': data_info.get('y_list_scalar_columns', []),
            'variables_1d_pft': data_info.get('variables_1d_pft', []),
            'y_list_columns_1d': data_info.get('y_list_columns_1d', []),
            'x_list_columns_2d': data_info.get('x_list_columns_2d', []),
            'y_list_columns_2d': data_info.get('y_list_columns_2d', []),
            'training': {
                'num_epochs': config.training_config.num_epochs,
                'batch_size': config.training_config.batch_size,
                'learning_rate': config.training_config.learning_rate,
                'optimizer_type': config.training_config.optimizer_type,
                'use_scheduler': config.training_config.use_scheduler,
                'scheduler_type': config.training_config.scheduler_type,
                'device': config.training_config.device,
                'use_mixed_precision': config.training_config.use_mixed_precision,
                'use_amp': config.training_config.use_amp,
                'use_grad_scaler': config.training_config.use_grad_scaler,
                'random_seed': getattr(config.training_config, 'random_seed', None),
                'deterministic': getattr(config.training_config, 'deterministic', None),
                'train_split': getattr(config.data_config, 'train_split', None),
                'shuffle_seed': getattr(config.training_config, 'shuffle_seed', None),
                'max_files': getattr(config.data_config, 'max_files', None)
            }
        }
        logger.info(f"Resolved variable lists and params: {json.dumps(resolved, indent=2)}")
        with open(output_dir / 'resolved_config.json', 'w') as f:
            json.dump(resolved, f, indent=2)
        
        # Normalize data
        logger.info("Normalizing data...")
        normalized_data = data_loader.normalize_data()
        
        # Split data
        logger.info("Splitting data into train/test sets...")
        split_data = data_loader.split_data(normalized_data)
        train_data = split_data['train']
        test_data = split_data['test']
        
        # print(f"Training samples: {train_data['time_series'].shape[0]}")
        # print(f"Validation/Evaluation samples: {test_data['time_series'].shape[0]}")
        
        # Debug print for pft_param
        # if 'pft_param' in train_data:
        #     print('[DEBUG] pft_param shape (train):', train_data['pft_param'].shape)
        # else:
        #     print('[DEBUG] pft_param not found in train split!')
        # if 'pft_param' in test_data:
        #     print('[DEBUG] pft_param shape (test):', test_data['pft_param'].shape)
        # else:
        #     print('[DEBUG] pft_param not found in test split!')

        # Debug: Print train_data keys and check for y_scalar
        # print('DEBUG: train_data keys:', train_data.keys())
        # print('DEBUG: y_scalar in train_data:', 'y_scalar' in train_data)

        # Create CNP model
        logger.info("Creating CNP model...")

        model = CNPCombinedModel(
            config.model_config,
            data_info,
            include_water=include_water,
            use_learnable_loss_weights=config.training_config.use_learnable_loss_weights
        )
        
        # Ensure proper initialization for all layers
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)  # Lower gain for stability
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = ModelTrainer(
            config.training_config,
            model,
            train_data,
            test_data,
            normalized_data['scalers'],
            data_info
        )
        
        # Run training pipeline
        logger.info("Starting CNP model training pipeline...")
        results = trainer.run_training_pipeline()
        
        logger.info("CNP model training completed successfully!")
        logger.info(f"Final metrics: {results['metrics']}")
        
        # Save results to output directory

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