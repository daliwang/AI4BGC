#!/usr/bin/env python3
"""
Multi-GPU Training Script for AI4BGC using DistributedDataParallel (DDP)
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import logging
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.training_config_gpus import get_multi_gpu_config
from data.data_loader import DataLoader
from models.combined_model import CombinedModel
from training.trainer import ModelTrainer
from utils.logging_utils import setup_logging

def setup_distributed(rank, world_size, backend='nccl'):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    # Set the device for this process
    torch.cuda.set_device(rank)
    
    print(f"Process {rank}/{world_size} initialized on GPU {rank}")

def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()

def train_worker(rank, world_size, config_name, args):
    """Training worker function for each GPU process."""
    logger = None
    try:
        # Setup distributed training
        setup_distributed(rank, world_size)
        
        # Setup logging
        log_level = getattr(logging, args.log_level.upper())
        setup_logging(level=log_level)
        logger = logging.getLogger(f"rank_{rank}")
        
        logger.info(f"Starting multi-GPU training on rank {rank}/{world_size}")
        logger.info(f"Configuration: {config_name}")
        
        # Get configuration
        data_config, model_config, training_config = get_multi_gpu_config()
        
        # Override batch size for multi-GPU (effective batch size = batch_size * world_size)
        effective_batch_size = training_config.batch_size
        per_gpu_batch_size = effective_batch_size // world_size
        training_config.batch_size = per_gpu_batch_size
        
        logger.info(f"Effective batch size: {effective_batch_size}")
        logger.info(f"Per-GPU batch size: {per_gpu_batch_size}")
        
        # Initialize data loader
        logger.info("Initializing data loader...")
        data_loader = DataLoader(data_config)
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        train_data, test_data, scalers, data_info = data_loader.load_and_preprocess_data()
        
        # Create model
        logger.info("Creating model...")
        model = CombinedModel(
            lstm_hidden_size=model_config.lstm_hidden_size,
            fc_hidden_size=model_config.fc_hidden_size,
            static_fc_size=model_config.static_fc_size,
            conv_channels=model_config.conv_channels,
            conv_kernel_size=model_config.conv_kernel_size,
            conv_padding=model_config.conv_padding,
            num_tokens=model_config.num_tokens,
            token_dim=model_config.token_dim,
            transformer_layers=model_config.transformer_layers,
            transformer_heads=model_config.transformer_heads,
            scalar_output_size=model_config.scalar_output_size,
            vector_output_size=model_config.vector_output_size,
            vector_length=model_config.vector_length,
            matrix_output_size=model_config.matrix_output_size,
            matrix_rows=model_config.matrix_rows,
            matrix_cols=model_config.matrix_cols,
            data_info=data_info
        )
        
        # Move model to GPU
        device = torch.device(f'cuda:{rank}')
        model = model.to(device)
        
        # Wrap model with DDP
        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=training_config.find_unused_parameters,
            gradient_as_bucket_view=training_config.gradient_as_bucket_view,
            broadcast_buffers=training_config.broadcast_buffers,
            bucket_cap_mb=training_config.bucket_cap_mb,
            static_graph=training_config.static_graph
        )
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = ModelTrainer(
            training_config=training_config,
            model=model,
            train_data=train_data,
            test_data=test_data,
            scalers=scalers,
            data_info=data_info
        )
        
        # Run training
        logger.info("Starting training pipeline...")
        results = trainer.run_training_pipeline()
        
        # Save results (only on rank 0)
        if rank == 0:
            output_dir = Path(args.output_dir) if args.output_dir else Path("multi_gpu_results")
            output_dir.mkdir(exist_ok=True)
            
            # Save model
            if training_config.save_model:
                model_path = output_dir / "model.pth"
                torch.save(model.module.state_dict(), model_path)
                logger.info(f"Model saved to {model_path}")
            
            # Save results
            results_path = output_dir / "training_results.json"
            import json
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {results_path}")
        
        logger.info(f"Training completed on rank {rank}")
        
    except Exception as e:
        if logger:
            logger.error(f"Error on rank {rank}: {e}")
        else:
            print(f"Error on rank {rank}: {e}")
        raise
    finally:
        cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Training for AI4BGC")
    parser.add_argument("--config", type=str, default="multi_gpu", 
                       help="Configuration name")
    parser.add_argument("--output_dir", type=str, default="multi_gpu_results",
                       help="Output directory for results")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--world_size", type=int, default=None,
                       help="Number of GPUs to use (default: all available)")
    parser.add_argument("--backend", type=str, default="nccl",
                       choices=["nccl", "gloo"],
                       help="Distributed backend")
    
    args = parser.parse_args()
    
    # Determine number of GPUs to use
    if args.world_size is None:
        args.world_size = torch.cuda.device_count()
    
    if args.world_size <= 0:
        print("No GPUs available. Please check CUDA installation.")
        return
    
    print(f"Starting multi-GPU training with {args.world_size} GPUs")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Launch training processes
    mp.spawn(
        train_worker,
        args=(args.world_size, args.config, args),
        nprocs=args.world_size,
        join=True
    )

if __name__ == "__main__":
    main() 