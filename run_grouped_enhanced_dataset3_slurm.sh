#!/bin/bash
#SBATCH --job-name=grouped_enhanced_dataset3
#SBATCH --output=logs/grouped_enhanced_dataset3_%j.out
#SBATCH --error=logs/grouped_enhanced_dataset3_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Load modules
module load python/3.11
module load cuda/11.8

# Activate virtual environment
source /global/cfs/cdirs/m4814/wangd/AI4BGC/ai4bgc_env/bin/activate

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the training script
cd /global/cfs/cdirs/m4814/wangd/AI4BGC/AImodel
python train_grouped_enhanced_dataset3.py

echo "Grouped Enhanced Dataset 3 training completed!" 