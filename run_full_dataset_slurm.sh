#!/bin/bash
#SBATCH --job-name=ai4bgc_full_dataset
#SBATCH --account=m4814
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/full_dataset_%j.out
#SBATCH --error=logs/full_dataset_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wangd@nersc.gov

# Set environment variables
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Create logs directory
mkdir -p logs

# Print job information
echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Activate virtual environment
echo "=== Activating Virtual Environment ==="
source /global/cfs/cdirs/m4814/wangd/AI4BGC/ai4bgc_env/bin/activate
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA device count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Set up output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/full_dataset_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

# Run the full dataset training
echo "=== Starting Full Dataset Training ==="
echo "Output directory: $OUTPUT_DIR"
echo "Start time: $(date)"
echo ""

python model_training_refactored.py \
    full_dataset \
    --output-dir $OUTPUT_DIR \
    --log-level INFO

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=== Training Completed Successfully ==="
    echo "End time: $(date)"
    echo "Results saved to: $OUTPUT_DIR"
    
    # Print summary statistics
    echo ""
    echo "=== Results Summary ==="
    if [ -f "$OUTPUT_DIR/metrics.json" ]; then
        echo "Final metrics:"
        cat "$OUTPUT_DIR/metrics.json" | python -m json.tool
    fi
    
    if [ -f "$OUTPUT_DIR/training_log.txt" ]; then
        echo ""
        echo "Training log summary:"
        tail -20 "$OUTPUT_DIR/training_log.txt"
    fi
    
else
    echo ""
    echo "=== Training Failed ==="
    echo "End time: $(date)"
    echo "Check logs for details"
    exit 1
fi

echo ""
echo "=== Job Completed ==="
echo "Total runtime: $(date)" 