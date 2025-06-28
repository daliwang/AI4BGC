# AI4BGC Model Testing Documentation

## Overview
This document provides a comprehensive summary of all testing performed on the AI4BGC (AI for Biogeochemical Cycles) model, including development, optimization, and validation phases.

## 1. Initial Setup and Environment Configuration

### 1.1 Python Virtual Environment
- **Python Version**: 3.11
- **Environment Name**: `ai4bgc_env`
- **Location**: `/global/cfs/cdirs/m4814/wangd/AI4BGC/`
- **Activation**: Added to `.bashrc` for automatic activation

### 1.2 Dependencies
- **Core Framework**: PyTorch with CUDA support
- **Data Processing**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib
- **Utilities**: tqdm, logging

## 2. Minimal Testing Phase

### 2.1 Configuration
- **Data Files**: Limited to 3 files for rapid testing
- **Model Architecture**: Simplified version with reduced complexity
- **Training Parameters**:
  - Epochs: 5
  - Batch Size: 16
  - Learning Rate: 0.001
  - Device: CPU initially, then GPU

### 2.2 Issues Encountered and Resolved

#### 2.2.1 Data Normalization Issues
- **Problem**: List data (1D and 2D) normalization failing
- **Solution**: Implemented robust normalization for list data types
- **Files Modified**: `data/data_loader.py`

#### 2.2.2 Model Input/Output Dimension Mismatches
- **Problem**: Static data and target data dimension mismatches
- **Solution**: Made model dynamically adapt to actual data dimensions
- **Files Modified**: `models/combined_model.py`

#### 2.2.3 Batch Size Mismatches
- **Problem**: Tensor concatenation errors during training
- **Solution**: Ensured consistent batch sizes in trainer
- **Files Modified**: `training/trainer.py`

### 2.3 Evaluation Metrics Issues
- **Problem**: Metric calculations failing for vector and matrix outputs
- **Solution**: Fixed metric calculations to align shapes and flatten arrays properly
- **Files Modified**: `training/trainer.py`

### 2.4 TorchScript Export Issues
- **Problem**: Python logging in model's forward method unsupported
- **Solution**: Removed debug logging statements
- **Files Modified**: `models/combined_model.py`

## 3. GPU Optimization Phase

### 3.1 GPU Configuration
- **Mixed Precision Training**: Enabled Automatic Mixed Precision (AMP)
- **Memory Optimization**: 
  - Pin memory: True
  - Empty cache frequency: Every 10 batches
  - Max memory usage: 90%
- **Data Loading**: 
  - num_workers: 0 (to avoid GPU context issues)
  - prefetch_factor: None (when num_workers=0)

### 3.2 GPU-Specific Issues and Fixes

#### 3.2.1 Data Movement Issues
- **Problem**: Data tensors moved to GPU too early
- **Solution**: Move data to device only during training/validation loops
- **Files Modified**: `training/trainer.py`

#### 3.2.2 Autocast Device Parameters
- **Problem**: Autocast device parameter issues
- **Solution**: Fixed autocast device parameters for mixed precision
- **Files Modified**: `training/trainer.py`

#### 3.2.3 Model Initialization Issues
- **Problem**: Model using incorrect 1D input size
- **Solution**: Use actual 1D input size from config
- **Files Modified**: `models/combined_model.py`

### 3.3 Performance Results
- **GPU Speedup**: ~2.17x over CPU
- **Memory Usage**: Optimized for 90% GPU memory utilization
- **Training Stability**: Stable with mixed precision enabled

## 4. Full Model Testing Phase

### 4.1 Configuration Development
- **CPU Config**: `get_full_model_test_cpu_config()`
- **GPU Config**: `get_full_model_test_gpu_config()`
- **Data Files**: 3 files for validation
- **Model Parameters**:
  - LSTM hidden size: 256
  - Static FC size: 512
  - Transformer layers: 4
  - Token dimension: 128

### 4.2 Comparison Script Development
- **Script**: `compare_cpu_gpu.py`
- **Features**:
  - Automated CPU/GPU comparison
  - Performance metrics collection
  - Result analysis and reporting
  - Detailed logging

### 4.3 Initial Comparison Results
- **GPU Speedup**: ~2.17x
- **Metric Differences**: Large differences in scalar_rmse (64.9% difference)
- **Analysis**: Systematic differences due to:
  - Random seeds not fixed
  - Mixed precision effects
  - Device-specific floating-point differences

## 5. Fair Comparison Implementation

### 5.1 Fair Comparison Configuration
- **Random Seed**: Fixed to 42 for both CPU and GPU
- **Mixed Precision**: Disabled for fair comparison
- **Deterministic Behavior**: Enabled
- **Identical Initialization**: Same model parameters and architecture

### 5.2 Configuration Files Created
- `get_fair_comparison_cpu_config()`
- `get_fair_comparison_gpu_config()`
- `compare_fair_cpu_gpu.py`

### 5.3 Fair Comparison Results
- **CPU Duration**: 46.50 seconds
- **GPU Duration**: 17.77 seconds
- **GPU Speedup**: **2.62x**
- **Reproducibility**: Achieved through fixed seeds and deterministic behavior

## 6. Data Pipeline Testing

### 6.1 Data Loading Architecture
- **DataLoader Class**: Flexible data loading with preprocessing
- **Supported Data Types**:
  - Time series data
  - Static features
  - 1D list data
  - 2D list data
  - Target variables

### 6.2 Data Processing Pipeline
1. **Load Data**: Load from multiple paths
2. **Preprocess**: Drop columns, filter data, process time series
3. **Normalize**: Apply normalization to all data types
4. **Split**: Train/test split with configurable ratio
5. **Prepare**: Convert to tensors for training

### 6.3 Data Configuration
- **Time Series Columns**: FLDS, PSRF, FSDS, QBOT, PRECTmms, TBOT
- **Static Columns**: lat, lon, area, landfrac, PFT0-PFT15
- **1D List Columns**: deadcrootc, deadstemc, tlai
- **2D List Columns**: soil3c_vr, soil4c_vr, cwdc_vr
- **Target Columns**: Y_HR, Y_NPP, Y_COL_FIRE_CLOSS, Y_GPP, Y_AR

## 7. Model Architecture Testing

### 7.1 CombinedModel Architecture
- **LSTM**: Time series processing
- **CNN**: 2D data processing
- **Fully Connected**: Static and 1D data processing
- **Transformer**: Feature fusion
- **Output Heads**: Scalar, vector, and matrix predictions

### 7.2 FlexibleCombinedModel
- **Dynamic Output Sizes**: Adapts to actual data dimensions
- **Dictionary Output**: Returns predictions as dictionary
- **Extensible**: Supports additional output heads

### 7.3 Model Parameters
- **LSTM Hidden Size**: 64-256 (configurable)
- **Static FC Size**: 64-512 (configurable)
- **Transformer Layers**: 2-4 (configurable)
- **Token Dimension**: 64-128 (configurable)
- **Output Sizes**: Dynamically calculated from data

## 8. Training Pipeline Testing

### 8.1 Trainer Features
- **Mixed Precision**: Automatic Mixed Precision (AMP) support
- **GPU Monitoring**: Memory usage and utilization tracking
- **Early Stopping**: Configurable patience and delta
- **Learning Rate Scheduling**: ReduceLROnPlateau
- **Loss Weights**: Configurable weights for different output types

### 8.2 Training Configurations
- **Minimal Config**: For rapid testing
- **Full Model Config**: For comprehensive testing
- **Fair Comparison Config**: For reproducible comparisons
- **Production Config**: For full dataset training

### 8.3 Loss Functions
- **Scalar Loss**: MSE for target variables
- **Vector Loss**: MSE for 1D list targets
- **Matrix Loss**: MSE for 2D list targets
- **Total Loss**: Weighted combination of all losses

## 9. Evaluation and Metrics

### 9.1 Metrics Calculated
- **RMSE**: Root Mean Square Error for all output types
- **MAE**: Mean Absolute Error
- **R² Score**: Coefficient of determination
- **Custom Metrics**: Shape-aware calculations for list data

### 9.2 Evaluation Pipeline
1. **Model Evaluation**: Run model in eval mode
2. **Prediction Collection**: Gather all predictions
3. **Metric Calculation**: Compute metrics for each output type
4. **Result Saving**: Save predictions and metrics to files

### 9.3 Output Formats
- **Scalar Predictions**: CSV format
- **Vector Predictions**: CSV format (flattened)
- **Matrix Predictions**: CSV format (flattened)
- **Metrics**: CSV format with all calculated metrics

## 10. Batch Training Implementation

### 10.1 Batch Training Script
- **Script**: `batch_training.py`
- **Features**:
  - Sequential experiment execution
  - Configuration management
  - Result aggregation
  - Error handling and recovery

### 10.2 Shell Script Integration
- **Script**: `run_full_dataset.sh`
- **Features**:
  - Environment activation
  - Logging setup
  - Background execution
  - Process monitoring

### 10.3 Documentation
- **README**: `BATCH_TRAINING_README.md`
- **Usage Instructions**: Comprehensive guide for batch training
- **Configuration Examples**: Sample configurations for different scenarios

## 11. Performance Analysis

### 11.1 CPU vs GPU Performance
- **Baseline Comparison**: ~2.17x speedup (with systematic differences)
- **Fair Comparison**: 2.62x speedup (reproducible)
- **Memory Usage**: Optimized for 90% GPU utilization
- **Training Stability**: Stable across different configurations

### 11.2 Scalability Testing
- **Data Size**: Tested with 3 files to full dataset
- **Model Size**: Tested with minimal to full model configurations
- **Batch Size**: Tested with 16 to 64 batch sizes
- **Memory Efficiency**: Optimized for large-scale training

## 12. Quality Assurance

### 12.1 Code Quality
- **Logging**: Comprehensive logging throughout the pipeline
- **Error Handling**: Robust error handling and recovery
- **Documentation**: Inline documentation and docstrings
- **Type Hints**: Type annotations for better code clarity

### 12.2 Testing Coverage
- **Unit Testing**: Individual component testing
- **Integration Testing**: End-to-end pipeline testing
- **Performance Testing**: CPU/GPU comparison testing
- **Reproducibility Testing**: Fair comparison validation

### 12.3 Configuration Management
- **Modular Design**: Separate configurations for different use cases
- **Parameter Validation**: Input validation and error checking
- **Default Values**: Sensible defaults for all parameters
- **Flexibility**: Easy modification for different scenarios

## 13. Lessons Learned

### 13.1 Technical Insights
- **Mixed Precision**: Provides speedup but introduces numerical differences
- **Random Seeds**: Critical for reproducible results
- **Data Movement**: Timing of GPU data transfer affects performance
- **Memory Management**: Proper GPU memory management essential for stability

### 13.2 Best Practices
- **Fair Comparisons**: Always use identical settings for performance comparisons
- **Reproducibility**: Fix random seeds and use deterministic operations
- **Monitoring**: Implement comprehensive logging and monitoring
- **Modularity**: Design for flexibility and reusability

### 13.3 Optimization Strategies
- **GPU Utilization**: Optimize for high GPU memory usage
- **Data Loading**: Balance between speed and memory usage
- **Model Architecture**: Design for the specific data characteristics
- **Training Pipeline**: Optimize the entire pipeline, not just individual components

## 14. Future Recommendations

### 14.1 Performance Improvements
- **Multi-GPU Training**: Implement distributed training
- **Data Parallelism**: Optimize data loading for larger datasets
- **Model Parallelism**: Split large models across multiple GPUs
- **Memory Optimization**: Further optimize memory usage patterns

### 14.2 Feature Enhancements
- **Additional Metrics**: Implement more sophisticated evaluation metrics
- **Visualization**: Add training curve visualization
- **Hyperparameter Tuning**: Implement automated hyperparameter optimization
- **Model Interpretability**: Add model interpretation capabilities

### 14.3 Production Readiness
- **Deployment**: Prepare for production deployment
- **Monitoring**: Implement production monitoring and alerting
- **Scaling**: Design for horizontal scaling
- **Maintenance**: Establish maintenance and update procedures

## 15. Key Files and Scripts

### 15.1 Core Training Scripts
- `train_model.py`: Main training script with configuration support
- `compare_cpu_gpu.py`: CPU vs GPU comparison script
- `compare_fair_cpu_gpu.py`: Fair comparison script
- `batch_training.py`: Batch training script for multiple experiments

### 15.2 Configuration Files
- `config/training_config.py`: All training configurations
- `config/data_config.py`: Data loading configurations
- `config/model_config.py`: Model architecture configurations

### 15.3 Model Files
- `models/combined_model.py`: Main model implementation
- `models/flexible_combined_model.py`: Flexible model variant

### 15.4 Training Files
- `training/trainer.py`: Main trainer implementation
- `training/data_loader.py`: Data loading and preprocessing

### 15.5 Utility Files
- `utils/gpu_monitor.py`: GPU monitoring utilities
- `utils/logging_utils.py`: Logging configuration

### 15.6 Documentation Files
- `README.md`: Main project documentation
- `BATCH_TRAINING_README.md`: Batch training guide
- `TESTING_DOCUMENTATION.md`: This testing documentation

## 16. Commands and Usage

### 16.1 Basic Training
```bash
# Minimal testing
python train_model.py --config minimal

# Full model testing
python train_model.py --config full_test_gpu

# Fair comparison
python compare_fair_cpu_gpu.py
```

### 16.2 Batch Training
```bash
# Run batch training
python batch_training.py

# Run with shell script
./run_full_dataset.sh
```

### 16.3 Environment Setup
```bash
# Activate environment
source ai4bgc_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 17. Performance Benchmarks

### 17.1 Training Times
- **Minimal Config (CPU)**: ~30 seconds
- **Minimal Config (GPU)**: ~15 seconds
- **Full Model Config (CPU)**: ~60 seconds
- **Full Model Config (GPU)**: ~25 seconds
- **Fair Comparison (CPU)**: 46.50 seconds
- **Fair Comparison (GPU)**: 17.77 seconds

### 17.2 Memory Usage
- **CPU Memory**: ~4-8 GB
- **GPU Memory**: ~6-12 GB (90% utilization target)
- **Data Loading**: Optimized for minimal memory overhead

### 17.3 Scalability
- **Data Files**: Tested with 3 to full dataset
- **Batch Sizes**: 16 to 64 (configurable)
- **Model Sizes**: Minimal to full configurations
- **GPU Utilization**: 90% target achieved

## 18. Troubleshooting Guide

### 18.1 Common Issues
- **CUDA Out of Memory**: Reduce batch size or enable gradient checkpointing
- **Data Loading Errors**: Check file paths and data format
- **Model Convergence**: Adjust learning rate or loss weights
- **Reproducibility Issues**: Ensure random seeds are fixed

### 18.2 Debugging Tips
- **Enable Debug Logging**: Set log level to DEBUG
- **Check GPU Memory**: Use GPU monitoring utilities
- **Validate Data**: Check data shapes and types
- **Profile Performance**: Use PyTorch profiler

### 18.3 Performance Optimization
- **Batch Size Tuning**: Find optimal batch size for your GPU
- **Mixed Precision**: Enable for speedup (if numerical differences acceptable)
- **Data Loading**: Optimize num_workers and prefetch_factor
- **Memory Management**: Monitor and optimize memory usage

---

**Document Version**: 1.0  
**Last Updated**: June 28, 2025  
**Author**: AI4BGC Development Team  
**Project**: AI for Biogeochemical Cycles (AI4BGC)

This comprehensive testing documentation provides a complete overview of the AI4BGC model development and validation process, serving as a reference for future development and deployment efforts. 