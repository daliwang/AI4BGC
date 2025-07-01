# Grouped Enhanced Model Architecture for Dataset 3

This document describes the enhanced grouped model architecture specifically designed for Dataset 3, which groups input data into logical categories and applies separate embedding layers for each group, then concatenates them for transformer-based fusion.

## Overview

The Grouped Enhanced Model is designed to handle the rich feature set of Dataset 3 by organizing input data into logical groups and applying specialized processing for each group. This architecture provides better feature learning and more focused processing compared to the standard combined model.

## Architecture Components

### 1. Data Groups

The model processes data through **6 main input groups**:

1. **Time Series Group (Forcing Variables)**: Atmospheric forcing data
   - Variables: `['FLDS', 'PSRF', 'FSDS', 'QBOT', 'PRECTmms', 'TBOT']`
   - Processing: LSTM encoder with projection

2. **Static Surface Group**: Geographic, PFT, and soil properties
   - Variables: Geographic coordinates, PFT distributions, soil properties, auxiliary variables
   - Processing: Group embedding layer (MLP)

3. **Water Group**: Water-related variables
   - Variables: `['H2OCAN', 'H2OSFC', 'H2OSNO', 'H2OSOI_LIQ', 'H2OSOI_ICE', 'H2OSOI_10CM']`
   - Processing: Group embedding layer (MLP)

4. **Temperature Group**: Temperature-related variables
   - Variables: `['T_VEG', 'T10_VALUE', 'TH2OSFC', 'T_GRND', 'T_GRND_R', 'T_GRND_U', 'T_SOISNO', 'T_LAKE', 'TS_TOPO']`
   - Processing: Group embedding layer (MLP)

5. **1D CNP Group**: Carbon/Nitrogen/Phosphorus pools (1D)
   - Variables: `['deadcrootc', 'deadstemc', 'tlai', 'leafc', 'frootc', 'totlitc']`
   - Processing: Group embedding layer (MLP)

6. **2D CNP Group**: Soil and litter CNP pools (2D)
   - Variables: `['soil1c_vr', 'soil2c_vr', 'soil3c_vr', 'soil4c_vr', 'litr1c_vr', 'litr2c_vr', 'litr3c_vr']`
   - Processing: Enhanced CNN with residual connections

### 2. Group Embedding Layers

Each group (except time series and 2D CNP) has its own `GroupEmbeddingLayer` that includes:

- **Multi-layer Perceptron**: Input → 256 → 128 → 128
- **Layer Normalization**: For stable training
- **Dropout**: For regularization

**Why MLPs instead of Attention?**
- **Efficiency**: Much faster and less memory-intensive for small feature groups
- **Sufficiency**: MLPs can capture all necessary feature interactions for 6-9 variables
- **Simplicity**: Easier to train and debug
- **Proven effectiveness**: MLPs work well for tabular data

### 3. Enhanced LSTM Encoder

The time series group uses an enhanced LSTM encoder:
- **3-layer Bidirectional LSTM**: Hidden size 256, dropout 0.1
- **Direct Projection**: LSTM output → 128 dimensions
- **No Transformer Head**: Simplified architecture for efficiency

**Why Larger LSTM instead of LSTM + Transformer Head?**
- **Simplicity**: Single, well-understood architecture
- **Efficiency**: No redundant processing
- **Proven effectiveness**: LSTMs work excellently for time series
- **Easier training**: Fewer parameters to tune

### 4. Enhanced CNN Encoder

The 2D CNP group uses an enhanced CNN with:
- **Residual Blocks**: For better gradient flow
- **Batch Normalization**: For stable training
- **Progressive Channel Increase**: 64 → 128 → 256 channels
- **MaxPooling**: Between layers for dimension reduction



### 5. Main Transformer Fusion

All group features are concatenated and processed through:
- **6-layer Transformer Encoder**: 16 attention heads, 512 feedforward dimension
- **Fixed Token Dimension**: 128 dimensions per group
- **Global Feature Fusion**: Comprehensive feature interaction

### 6. Output Heads

The model produces three types of outputs:
- **Scalar Output**: 5 scalar predictions
- **Vector Output**: 6 vector predictions (16 length each)
- **Matrix Output**: 7 matrix predictions (18×10 each)

## Key Advantages

1. **Specialized Processing**: Each group gets dedicated neural network architectures optimized for its data type
2. **Separate Water/Temperature Groups**: Allows the model to learn specific patterns in water and temperature variables
3. **Enhanced CNP Processing**: Comprehensive carbon/nitrogen/phosphorus cycle modeling
4. **Modular Design**: Easy to modify or extend individual groups without affecting others
5. **Better Feature Learning**: Grouped processing allows for more focused feature extraction
6. **Simplified Architecture**: Efficient processing without unnecessary complexity

## Model Configuration

The model uses the following key configuration parameters:

```python
# Model Architecture
lstm_hidden_size=512
fc_hidden_size=256
static_fc_size=256
conv_channels=[64, 128, 256]
transformer_layers=6
transformer_heads=16
token_dim=128

# Output Dimensions
scalar_output_size=5
vector_output_size=6
matrix_output_size=7
vector_length=16
matrix_rows=18
matrix_cols=10
```

## Training Configuration

```python
# Training Parameters
batch_size=32
learning_rate=0.001
num_epochs=100
early_stopping_patience=15
```

## Usage

To train the Grouped Enhanced Model on Dataset 3:

```bash
# Using SLURM
sbatch run_grouped_enhanced_dataset3_slurm.sh

# Or directly
python train_grouped_enhanced_dataset3.py
```

## File Structure

- `models/grouped_enhanced_model.py`: Main model implementation
- `config/training_config.py`: Configuration function `get_grouped_enhanced_dataset3_config()`
- `train_grouped_enhanced_dataset3.py`: Training script
- `run_grouped_enhanced_dataset3_slurm.sh`: SLURM job script

## Comparison with Standard Model

| Feature | Standard Combined Model | Grouped Enhanced Model |
|---------|------------------------|------------------------|
| Data Organization | Single processing pipeline | Grouped processing |
| Embedding Layers | Shared embeddings | Group-specific embeddings |
| Attention | Single transformer | Multiple transformer heads |
| Feature Learning | Global | Group-specific + global |
| Architecture | Fixed | Modular and extensible |
| Dataset Support | All datasets | Dataset 3 optimized |

The Grouped Enhanced Model provides a more sophisticated architecture specifically designed to handle the complex feature relationships in Dataset 3, while maintaining the proven transformer-based fusion approach. 