#!/bin/bash
#SBATCH --job-name=ai4bgc_interactive
#SBATCH --account=m4814
#SBATCH --constraint=gpu
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=2:00:00
#SBATCH --output=logs/interactive_%j.out
#SBATCH --error=logs/interactive_%j.err
#SBATCH --exclusive  # Request a dedicated node for this session

# Load modules
#module load python/3.9-anaconda-2021.11
#module load cuda/11.7

# Get the number of GPUs allocated by SLURM
NUM_GPUS=${SLURM_GPUS_PER_NODE:-1}

# Set CUDA_VISIBLE_DEVICES based on allocated GPUs
# Create a comma-separated list from 0 to NUM_GPUS-1
GPU_LIST=$(seq -s, 0 $((NUM_GPUS-1)))
export CUDA_VISIBLE_DEVICES=$GPU_LIST
export OMP_NUM_THREADS=8

# Create logs directory
mkdir -p logs

# Print job information
echo "=== Interactive Session Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs allocated: $NUM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo ""

# Activate virtual environment
echo "=== Activating Virtual Environment ==="
source /global/cfs/cdirs/m4814/wangd/AI4BGC/ai4bgc_env/bin/activate
echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA device count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Check GPU
echo "=== GPU Information ==="
nvidia-smi
echo ""

# Start interactive shell
echo "=== Starting Interactive Shell ==="
echo "You are now in an interactive session on a compute node!"
echo "Current directory: $(pwd)"
echo "Available commands:"
echo "  - python model_training_refactored.py [config_name]"
echo "  - python train_model.py [config_name]"
echo "  - nvidia-smi (to check GPU)"
echo "  - exit (to end session)"
echo ""

# Start bash shell
exec bash 