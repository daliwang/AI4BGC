#!/usr/bin/env python3
"""
Test script to verify DDP setup with single GPU (for testing purposes)
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time

def setup_distributed(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    # Set the device for this process
    torch.cuda.set_device(rank)
    
    print(f"Process {rank}/{world_size} initialized on GPU {rank}")

def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()

def test_worker(rank, world_size):
    """Test worker function for each GPU process."""
    try:
        # Setup distributed training
        setup_distributed(rank, world_size)
        
        # Create a simple model
        model = torch.nn.Linear(100, 10).cuda(rank)
        
        # Wrap with DDP
        model = DDP(model, device_ids=[rank])
        
        # Create some test data
        x = torch.randn(32, 100).cuda(rank)
        y = torch.randn(32, 10).cuda(rank)
        
        # Test forward pass
        start_time = time.time()
        for i in range(10):
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
            loss.backward()
        
        end_time = time.time()
        
        # Print results
        if rank == 0:
            print(f"DDP test completed successfully!")
            print(f"World size: {world_size}")
            print(f"Time for 10 iterations: {end_time - start_time:.4f} seconds")
            print(f"Average time per iteration: {(end_time - start_time) / 10:.4f} seconds")
            print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        print(f"Rank {rank}: Test completed successfully")
        
    except Exception as e:
        print(f"Error on rank {rank}: {e}")
        raise
    finally:
        cleanup_distributed()

def main():
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    if num_gpus < 1:
        print("No GPUs available. Please check CUDA installation.")
        return
    
    # For testing, we'll use 1 GPU even if more are available
    world_size = 1
    print(f"Testing DDP with {world_size} GPU(s)")
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Launch test processes
    mp.spawn(
        test_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
    
    print("DDP test completed!")

if __name__ == "__main__":
    main() 