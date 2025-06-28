#!/bin/bash

# Full Dataset Training Script
# This script runs the production training with the complete dataset

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/full_dataset_results"
LOG_FILE="${OUTPUT_DIR}/full_dataset_training.log"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=== Full Dataset Training Started at $(date) ==="
echo "Output directory: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Activate virtual environment (if needed)
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Virtual environment already activated: $VIRTUAL_ENV"
else
    echo "Activating virtual environment..."
    source ~/.bashrc
    conda activate ai4bgc_env
fi

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

# Run the full dataset training
echo "Starting full dataset training..."
cd "${SCRIPT_DIR}"

python model_training_refactored.py full_dataset \
    --log-level INFO \
    2>&1 | tee "${LOG_FILE}"

# Check exit status
if [ $? -eq 0 ]; then
    echo "=== Full Dataset Training Completed Successfully at $(date) ==="
    echo "Results saved in: ${OUTPUT_DIR}"
    
    # Copy final results to timestamped directory
    FINAL_DIR="${OUTPUT_DIR}/run_${TIMESTAMP}"
    mkdir -p "${FINAL_DIR}"
    
    if [ -d "predictions" ]; then
        cp -r predictions/* "${FINAL_DIR}/"
        echo "Predictions copied to: ${FINAL_DIR}"
    fi
    
    if [ -f "training_validation_losses.csv" ]; then
        cp training_validation_losses.csv "${FINAL_DIR}/"
        echo "Training losses copied to: ${FINAL_DIR}"
    fi
    
else
    echo "=== Full Dataset Training Failed at $(date) ==="
    echo "Check log file for details: ${LOG_FILE}"
    exit 1
fi 