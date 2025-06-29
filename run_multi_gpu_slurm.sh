#!/bin/bash
#SBATCH --job-name=ai4bgc_multi_gpu
#SBATCH --account=m4814
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=2:00:00
#SBATCH --output=logs/multi_gpu_%j.out
#SBATCH --error=logs/multi_gpu_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wangd@nersc.gov

# Load modules
module load python/3.9-anaconda-2021.11
module load cuda/11.7

# Get the number of GPUs allocated by SLURM
NUM_GPUS=${SLURM_GPUS_PER_NODE:-4}

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Create logs directory
mkdir -p logs

# Print job information
echo "=== Multi-GPU Training Job Started ==="
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

# Check GPU information
echo "=== GPU Information ==="
nvidia-smi
echo ""

# Set up output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/multi_gpu_${NUM_GPUS}gpu_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

# Run multi-GPU training
echo "=== Starting Multi-GPU Training ==="
echo "Number of GPUs: $NUM_GPUS"
echo "Output directory: $OUTPUT_DIR"
echo "Start time: $(date)"
echo ""

python model_training_multi_gpu.py \
    --config multi_gpu \
    --output_dir $OUTPUT_DIR \
    --log_level INFO \
    --world_size $NUM_GPUS \
    --backend nccl

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=== Multi-GPU Training Completed Successfully ==="
    echo "End time: $(date)"
    echo "Results saved to: $OUTPUT_DIR"
    
    # Print summary statistics
    echo ""
    echo "=== Results Summary ==="
    if [ -f "$OUTPUT_DIR/training_results.json" ]; then
        echo "Training results:"
        cat "$OUTPUT_DIR/training_results.json" | python -m json.tool
    fi
    
    # Print GPU utilization summary
    echo ""
    echo "=== GPU Utilization Summary ==="
    echo "Final GPU status:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
    
else
    echo ""
    echo "=== Multi-GPU Training Failed ==="
    echo "End time: $(date)"
    echo "Check logs for details"
    exit 1
fi

echo ""
echo "=== Job Completed ==="
echo "Total runtime: $(date)" 