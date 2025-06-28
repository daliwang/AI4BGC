# Refactored Climate Model Training Framework

This repository contains a refactored and modular climate model training framework that provides flexibility for training models with different input and output configurations.

## 🚀 Key Improvements

### 1. **Modular Architecture**
- **Configuration Management**: Centralized configuration system for easy parameter modification
- **Data Loading**: Flexible data loader that handles different input/output combinations
- **Model Architecture**: Modular model design that adapts to different data types
- **Training Pipeline**: Comprehensive training framework with validation and evaluation

### 2. **Flexibility**
- **Dynamic Input/Output**: Easy to modify input and output features without code changes
- **Multiple Configurations**: Pre-defined configurations for different use cases
- **Custom Configurations**: Easy to create custom training scenarios
- **Command-line Interface**: Simple CLI for different training modes

### 3. **Better Code Organization**
- **Separation of Concerns**: Clear separation between data, model, and training logic
- **Type Hints**: Comprehensive type annotations for better code understanding
- **Error Handling**: Robust error handling and validation
- **Logging**: Comprehensive logging throughout the pipeline

## 📁 Project Structure

```
kiloCraft/
├── config/
│   └── training_config.py          # Configuration management
├── data/
│   └── data_loader.py              # Data loading and preprocessing
├── models/
│   └── combined_model.py           # Flexible model architecture
├── training/
│   └── trainer.py                  # Training pipeline
├── model_training_refactored.py    # Main training script
└── README_REFACTORED.md           # This file
```

## 🛠️ Installation and Setup

### Prerequisites
```bash
pip install torch pandas numpy scikit-learn matplotlib pyproj netCDF4
```

### Basic Usage

1. **Default Training** (reproduces original behavior):
```bash
python model_training_refactored.py default
```

2. **Minimal Training** (quick testing):
```bash
python model_training_refactored.py minimal
```

3. **Extended Training** (more features):
```bash
python model_training_refactored.py extended
```

4. **Custom Training**:
```bash
python model_training_refactored.py custom
```

## 🔧 Configuration System

### Pre-defined Configurations

#### 1. Default Configuration
- **Input Features**: All original features
- **Output Features**: All original targets
- **Training**: 50 epochs, batch size 16
- **Model**: Standard architecture

#### 2. Minimal Configuration
- **Input Features**: Reduced set for quick testing
- **Output Features**: Core targets only
- **Training**: 5 epochs, batch size 8
- **Model**: Simplified architecture

#### 3. Extended Configuration
- **Input Features**: Extended feature set
- **Output Features**: All available targets
- **Training**: 100 epochs, batch size 32
- **Model**: Enhanced architecture

### Custom Configurations

You can easily create custom configurations by modifying the `create_custom_config()` function:

```python
def create_custom_config():
    config = TrainingConfigManager()
    
    # Modify input features
    config.update_data_config(
        x_list_columns_2d=['soil3c_vr', 'soil4c_vr'],
        y_list_columns_2d=['Y_soil3c_vr', 'Y_soil4c_vr'],
        x_list_columns_1d=['deadcrootc'],
        y_list_columns_1d=['Y_deadcrootc']
    )
    
    # Modify training parameters
    config.update_training_config(
        num_epochs=100,
        batch_size=32,
        learning_rate=0.0005
    )
    
    return config
```

## 📊 Data Management

### Supported Data Types

1. **Time Series Data**: Meteorological variables (FLDS, PSRF, FSDS, etc.)
2. **Static Data**: Non-temporal features
3. **1D List Data**: Vector features (deadcrootc, deadstemc, tlai, etc.)
4. **2D List Data**: Matrix features (soil3c_vr, soil4c_vr, cwdc_vr, etc.)

### Data Preprocessing

The framework automatically handles:
- **Data Loading**: From multiple pickle files
- **Data Cleaning**: Removing specified columns and NaN values
- **Data Normalization**: MinMax, Standard, or Robust scaling
- **Data Splitting**: Train/test split with configurable ratio
- **Data Validation**: Checking for missing columns and data integrity

### Adding New Features

To add new input or output features:

1. **Update Configuration**:
```python
config.update_data_config(
    x_list_columns_2d=['new_2d_feature'],
    y_list_columns_2d=['Y_new_2d_feature'],
    x_list_columns_1d=['new_1d_feature'],
    y_list_columns_1d=['Y_new_1d_feature']
)
```

2. **Ensure Data Availability**: Make sure the features exist in your data files

3. **Run Training**: The framework will automatically adapt to the new features

## 🧠 Model Architecture

### Flexible Design

The model automatically adapts to:
- **Variable Input Sizes**: Based on the number of features
- **Variable Output Sizes**: Based on the number of targets
- **Different Data Types**: Time series, static, 1D, and 2D data

### Architecture Components

1. **LSTM Module**: Processes time series data
2. **CNN Module**: Processes 2D data (soil profiles, etc.)
3. **Fully Connected Layers**: Process static and 1D data
4. **Transformer Fusion**: Combines all features
5. **Output Heads**: Separate heads for scalar, vector, and matrix outputs

### Model Customization

You can customize the model architecture:

```python
config.update_model_config(
    lstm_hidden_size=128,      # LSTM hidden size
    fc_hidden_size=64,         # Fully connected hidden size
    transformer_layers=3,      # Number of transformer layers
    conv_channels=[32, 64, 128]  # CNN channel configuration
)
```

## 🎯 Training Pipeline

### Features

- **Automatic Device Detection**: GPU/CPU selection
- **Learning Rate Scheduling**: Step, cosine, or plateau scheduling
- **Early Stopping**: Configurable patience and minimum delta
- **Loss Weighting**: Adjustable weights for different output types
- **Memory Management**: Automatic memory optimization
- **Progress Monitoring**: Comprehensive logging and metrics

### Training Configuration

```python
config.update_training_config(
    num_epochs=100,              # Number of training epochs
    batch_size=32,               # Batch size
    learning_rate=0.001,         # Learning rate
    scalar_loss_weight=1.0,      # Weight for scalar outputs
    vector_loss_weight=0.5,      # Weight for vector outputs
    matrix_loss_weight=0.5,      # Weight for matrix outputs
    use_early_stopping=True,     # Enable early stopping
    patience=10,                 # Early stopping patience
    use_scheduler=True,          # Enable learning rate scheduling
    scheduler_type='cosine'      # Scheduler type
)
```

## 📈 Results and Evaluation

### Automatic Output

The framework automatically saves:
- **Training Curves**: Loss plots
- **Predictions**: All prediction types (scalar, vector, matrix)
- **Ground Truth**: Actual values for comparison
- **Metrics**: RMSE, MSE for each output type
- **Model**: Trained model in TorchScript format

### Evaluation Metrics

- **Scalar Metrics**: RMSE and MSE for scalar outputs
- **Vector Metrics**: RMSE and MSE for 1D outputs
- **Matrix Metrics**: RMSE and MSE for 2D outputs

## 🔬 Experiment Management

### Running Experiments

```python
# Define experiment configuration
experiment_config = {
    'data_config': {
        'x_list_columns_2d': ['soil3c_vr'],
        'y_list_columns_2d': ['Y_soil3c_vr'],
        'x_list_columns_1d': ['deadcrootc'],
        'y_list_columns_1d': ['Y_deadcrootc']
    },
    'training_config': {
        'num_epochs': 10,
        'batch_size': 8
    }
}

# Run experiment
results = run_experiment("soil_only_experiment", experiment_config)
```

### Comparing Configurations

You can easily compare different configurations:

```python
# Configuration 1: Soil only
config1 = get_default_config()
config1.update_data_config(
    x_list_columns_2d=['soil3c_vr'],
    y_list_columns_2d=['Y_soil3c_vr']
)

# Configuration 2: All features
config2 = get_default_config()

# Run both and compare results
```

## 🐛 Troubleshooting

### Common Issues

1. **Missing Data Files**:
   - Check data paths in configuration
   - Ensure pickle files exist and are readable

2. **Memory Issues**:
   - Reduce batch size
   - Use minimal configuration for testing
   - Enable memory optimization in configuration

3. **CUDA Out of Memory**:
   - Reduce batch size
   - Use CPU training: `device='cpu'`
   - Reduce model complexity

4. **Missing Features**:
   - Check feature names in data files
   - Verify configuration matches available data
   - Use `get_data_info()` to see available features

### Debug Mode

Enable debug logging for detailed information:

```bash
python model_training_refactored.py default --log-level DEBUG
```

## 📝 Examples

### Example 1: Quick Testing
```bash
# Run minimal training for quick testing
python model_training_refactored.py minimal
```

### Example 2: Custom Soil Model
```python
# Create custom configuration for soil-focused model
config = TrainingConfigManager()
config.update_data_config(
    x_list_columns_2d=['soil3c_vr', 'soil4c_vr'],
    y_list_columns_2d=['Y_soil3c_vr', 'Y_soil4c_vr'],
    x_list_columns_1d=['deadcrootc'],
    y_list_columns_1d=['Y_deadcrootc']
)
config.update_training_config(num_epochs=50, batch_size=16)
```

### Example 3: Extended Model
```bash
# Run extended training with all features
python model_training_refactored.py extended
```

## 🤝 Contributing

To contribute to this framework:

1. **Follow the modular structure**
2. **Add type hints** to new functions
3. **Update documentation** for new features
4. **Add tests** for new functionality
5. **Use the configuration system** for new parameters

## 📄 License

This project is part of the kiloCraft framework for climate modeling.

## 🙏 Acknowledgments

This refactored framework builds upon the original climate model training code, making it more flexible and maintainable for future research and development. 