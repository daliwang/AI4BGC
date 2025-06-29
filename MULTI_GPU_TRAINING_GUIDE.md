# Multi-GPU Training Guide for AI4BGC

This guide explains how to test and use multi-GPU training to speed up your AI4BGC model training.

## Overview

Multi-GPU training uses PyTorch's DistributedDataParallel (DDP) to distribute training across multiple GPUs, which can significantly speed up training time. The setup includes:

- **DistributedDataParallel (DDP)**: Wraps your model for distributed training
- **Multi-GPU Configuration**: Optimized settings for multi-GPU training
- **SLURM Integration**: Ready-to-use SLURM scripts for cluster deployment

## Files Created

1. **`model_training_multi_gpu.py`**: Main multi-GPU training script
2. **`run_multi_gpu_slurm.sh`**: SLURM script for multi-GPU training
3. **`test_multi_gpu.py`**: Test script for multi-GPU setup
4. **`test_single_gpu_ddp.py`**: Test script for single GPU DDP setup
5. **`config/training_config_gpus.py`**: Multi-GPU configuration

## Testing Multi-GPU Setup

### 1. Test Single GPU DDP Setup

First, test that DDP works with a single GPU:

```bash
python test_single_gpu_ddp.py
```

Expected output:
```
Available GPUs: 1
Testing DDP with 1 GPU(s)
Process 0/1 initialized on GPU 0
DDP test completed successfully!
World size: 1
Time for 10 iterations: 0.2042 seconds
Average time per iteration: 0.0204 seconds
Model parameters: 1010
Rank 0: Test completed successfully
DDP test completed!
```

### 2. Test Multi-GPU Setup (when multiple GPUs available)

```bash
python test_multi_gpu.py
```

This will test with all available GPUs.

## Running Multi-GPU Training

### Option 1: Direct Execution (Interactive)

```bash
# Activate virtual environment
source /global/cfs/cdirs/m4814/wangd/AI4BGC/ai4bgc_env/bin/activate

# Run multi-GPU training
python model_training_multi_gpu.py \
    --config multi_gpu \
    --output_dir multi_gpu_results \
    --log_level INFO \
    --world_size 4 \
    --backend nccl
```

### Option 2: SLURM Submission

```bash
# Submit the SLURM job
sbatch run_multi_gpu_slurm.sh
```

## Configuration Options

### Multi-GPU Configuration (`training_config_gpus.py`)

Key settings for multi-GPU training:

```python
# Multi-GPU/Distributed settings
use_multi_gpu: bool = True
distributed_backend: str = 'nccl'
find_unused_parameters: bool = False
gradient_as_bucket_view: bool = True
broadcast_buffers: bool = True
bucket_cap_mb: int = 25
static_graph: bool = True
distributed_sampler: bool = True
fp16_allreduce: bool = True
gradient_accumulation_steps: int = 1
max_grad_norm: float = 1.0
sync_batch_norm: bool = True
```

### Command Line Arguments

- `--config`: Configuration name (default: "multi_gpu")
- `--output_dir`: Output directory for results
- `--log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `--world_size`: Number of GPUs to use (default: all available)
- `--backend`: Distributed backend (nccl, gloo)

## SLURM Configuration

The `run_multi_gpu_slurm.sh` script requests:

```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=2:00:00
```

### Modifying GPU Count

To use different numbers of GPUs, modify:

1. **SLURM script**: Change `--gpus-per-node=4` to your desired number
2. **Training script**: The script automatically detects `SLURM_GPUS_PER_NODE`

## Performance Optimization

### 1. Batch Size Scaling

The effective batch size scales with the number of GPUs:
- **Effective batch size** = `batch_size` × `world_size`
- **Per-GPU batch size** = `batch_size` ÷ `world_size`

Example: With batch_size=128 and 4 GPUs:
- Effective batch size: 128 × 4 = 512
- Per-GPU batch size: 128 ÷ 4 = 32

### 2. Learning Rate Scaling

For optimal performance, scale the learning rate with batch size:

```python
# In training_config_gpus.py
learning_rate: float = 0.0005 * world_size  # Scale with number of GPUs
```

### 3. Mixed Precision Training

Multi-GPU training automatically uses mixed precision for better performance:

```python
use_amp: bool = True
fp16_allreduce: bool = True
```

## Monitoring and Debugging

### 1. GPU Utilization

Monitor GPU usage during training:

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or use the built-in monitoring in the training script
```

### 2. Logging

The training script provides detailed logging:
- GPU memory usage
- Training/validation loss
- Batch processing times
- Distributed training statistics

### 3. Common Issues

#### NCCL Communication Issues
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
```

#### Memory Issues
- Reduce batch size per GPU
- Enable gradient accumulation
- Use gradient checkpointing

## Expected Speedup

With proper configuration, you can expect:

- **2 GPUs**: ~1.8x speedup
- **4 GPUs**: ~3.5x speedup
- **8 GPUs**: ~6.5x speedup

Actual speedup depends on:
- Model size and complexity
- Data loading efficiency
- GPU communication bandwidth
- Batch size scaling

## Testing Different Configurations

### 1. Test with Minimal Data

```bash
python model_training_multi_gpu.py \
    --config minimal \
    --world_size 2 \
    --output_dir test_results
```

### 2. Test with Full Dataset

```bash
python model_training_multi_gpu.py \
    --config full_dataset \
    --world_size 4 \
    --output_dir full_results
```

## Troubleshooting

### 1. CUDA Out of Memory
- Reduce batch size per GPU
- Enable gradient accumulation
- Use gradient checkpointing

### 2. NCCL Communication Errors
- Check GPU connectivity
- Disable InfiniBand if needed
- Use gloo backend as fallback

### 3. Process Group Initialization Errors
- Check port availability
- Ensure proper environment setup
- Verify GPU visibility

## Next Steps

1. **Test with your data**: Run multi-GPU training on your specific dataset
2. **Optimize hyperparameters**: Tune batch size, learning rate, and other parameters
3. **Scale to more GPUs**: Extend to multiple nodes if needed
4. **Monitor performance**: Track training speed and resource utilization

## Support

For issues or questions:
1. Check the logs in the output directory
2. Verify GPU availability and configuration
3. Test with the provided test scripts
4. Review PyTorch DDP documentation 