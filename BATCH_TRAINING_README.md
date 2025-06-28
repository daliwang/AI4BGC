# Batch Training System for AI4BGC

This document explains how to use the batch training system for running multiple experiments with different configurations.

## Overview

The batch training system allows you to:
1. **Test with 3 files first** - Quick validation of the full model
2. **Run production training** - Full dataset with optimized parameters
3. **Batch multiple experiments** - Run different configurations sequentially

## Available Configurations

### 1. Minimal Test (`minimal`)
- **Purpose**: Quick validation with minimal model size
- **Data**: 3 files, limited features
- **Model**: Small architecture for fast testing
- **Training**: 5 epochs, small batch size
- **Use case**: Development and debugging

### 2. Full Model Test (`full_model_test`)
- **Purpose**: Test full model architecture with limited data
- **Data**: 3 files, all features
- **Model**: Full architecture (256 LSTM, 512 FC, 8 transformer layers)
- **Training**: 10 epochs, batch size 64
- **Use case**: Validate full model before production

### 3. Production Full Dataset (`full_dataset`)
- **Purpose**: Production training with complete dataset
- **Data**: All available files (84 total)
- **Model**: Large architecture (512 LSTM, 1024 FC, 16 transformer layers)
- **Training**: 50 epochs, batch size 128
- **Use case**: Final model training for deployment

## Quick Start

### 1. Test with 3 Files (Recommended First Step)

```bash
# Run full model test with 3 files
python model_training_refactored.py full_model_test
```

This will:
- Load 3 data files (~3000 samples)
- Train for 10 epochs (~2-3 minutes)
- Save results to `predictions/` directory

### 2. Run Production Training

```bash
# Option 1: Direct execution
python model_training_refactored.py full_dataset

# Option 2: Using shell script (recommended)
./run_full_dataset.sh
```

The shell script provides:
- Automatic environment activation
- GPU availability check
- Comprehensive logging
- Result organization

### 3. Batch Multiple Experiments

```bash
# List available configurations
python batch_training.py --list-configs

# Run specific configurations
python batch_training.py --configs minimal full_model_test

# Run all configurations
python batch_training.py --configs minimal full_model_test full_dataset
```

## Expected Performance

### Full Model Test (3 files)
- **Duration**: ~2-3 minutes
- **GPU Memory**: ~0.1 GB
- **Samples**: ~168 after filtering
- **Final Loss**: ~0.02-0.05

### Production Training (84 files)
- **Duration**: ~30-60 minutes
- **GPU Memory**: ~2-5 GB
- **Samples**: ~5000+ after filtering
- **Final Loss**: ~0.01-0.03

## Output Structure

### Individual Run Outputs
```
predictions/
├── model.pth                    # Trained model weights
├── test_metrics.csv            # Evaluation metrics
├── scalar_predictions.csv      # Scalar output predictions
├── vector_predictions.csv      # 1D list predictions
├── matrix_predictions.csv      # 2D list predictions
└── training_validation_losses.csv  # Training history
```

### Batch Run Outputs
```
batch_results/
└── batch_run_20250628_090000/
    ├── minimal/
    │   ├── experiment_info.txt
    │   ├── stdout.log
    │   ├── stderr.log
    │   └── predictions/
    ├── full_model_test/
    │   └── ...
    ├── full_dataset/
    │   └── ...
    └── batch_summary.txt
```

## Configuration Details

### Data Configuration
- **Time Series**: 6 columns (FLDS, PSRF, FSDS, QBOT, PRECTmms, TBOT)
- **Static Features**: 78 columns (soil properties, climate, etc.)
- **1D Lists**: 3 columns (deadcrootc, deadstemc, tlai)
- **2D Lists**: 3 columns (soil3c_vr, soil4c_vr, cwdc_vr)
- **Targets**: 5 scalar outputs (Y_HR, Y_NPP, Y_COL_FIRE_CLOSS, Y_GPP, Y_AR)

### Model Architecture
- **LSTM**: Processes time series data
- **Static FC**: Processes static features
- **Transformer**: Processes 1D and 2D list data
- **Multi-Output**: Scalar, vector, and matrix predictions

## Monitoring and Debugging

### GPU Monitoring
The system automatically monitors GPU usage:
- Memory consumption
- Utilization percentage
- Temperature (if available)

### Logging
- **Console**: Real-time progress
- **File**: Detailed logs for debugging
- **Batch**: Summary of all experiments

### Common Issues
1. **Out of Memory**: Reduce batch size in config
2. **Slow Training**: Check GPU utilization
3. **Data Loading**: Verify file paths and permissions

## Best Practices

### 1. Always Test First
```bash
# Start with minimal test
python model_training_refactored.py minimal

# Then test full model with 3 files
python model_training_refactored.py full_model_test
```

### 2. Monitor Resources
```bash
# Check GPU before starting
nvidia-smi

# Monitor during training
watch -n 1 nvidia-smi
```

### 3. Use Batch Processing for Multiple Experiments
```bash
# Run systematic comparison
python batch_training.py --configs minimal full_model_test full_dataset
```

### 4. Organize Results
```bash
# Use timestamped directories
./run_full_dataset.sh  # Automatically organizes results
```

## Troubleshooting

### GPU Issues
- **CUDA out of memory**: Reduce batch size
- **GPU not detected**: Check CUDA installation
- **Low utilization**: Check data loading pipeline

### Data Issues
- **File not found**: Verify data paths in config
- **Memory errors**: Reduce max_files parameter
- **Shape mismatches**: Check data preprocessing

### Training Issues
- **Loss not decreasing**: Check learning rate
- **Overfitting**: Increase weight decay
- **Slow convergence**: Adjust model architecture

## Next Steps

After successful testing with 3 files:

1. **Run production training** with full dataset
2. **Analyze results** in predictions directory
3. **Compare metrics** across different configurations
4. **Optimize hyperparameters** based on performance
5. **Deploy model** for inference

For questions or issues, check the logs in the output directories or contact the development team. 