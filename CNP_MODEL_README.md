# CNP Model Architecture

This document describes the CNP (Carbon-Nitrogen-Phosphorus) model architecture based on the `CNP_IO_list1.txt` structure.

## Architecture Overview

The CNP model implements a specialized neural network architecture designed for predicting CNP cycle variables:

### Input Processing Components

1. **LSTM for Time Series** (6 variables, 20 years)
   - Input: FLDS, PSRF, FSDS, QBOT, PRECTmms, TBOT
   - Architecture: 2-layer LSTM with 128 hidden units
   - Output: 128-dimensional features

2. **FC for Surface Properties** (27 variables)
   - Input: Geographic (7), Soil Phosphorus Forms (4), PFT Coverage (20), Soil Texture (20)
   - Architecture: 2-layer FC with 128 → 64 units
   - Output: 64-dimensional features

3. **FC for PFT Parameters** (44 variables)
   - Input: 44 PFT characteristics parameters
   - Architecture: 2-layer FC with 128 → 64 units
   - Output: 64-dimensional features

4. **FC for Water Variables** (6 variables, optional)
   - Input: H2OCAN, H2OSFC, H2OSNO, TH2OSFC, H2OSOI_LIQ, H2OSOI_ICE
   - Architecture: 2-layer FC with 32 → 16 units
   - Output: 16-dimensional features

5. **FC for Scalar Variables** (5 variables)
   - Input: GPP, NPP, AR, HR, LAI
   - Architecture: 2-layer FC with 32 → 16 units
   - Output: 16-dimensional features

6. **FC for 1D Variables** (13 variables)
   - Input: CNP pool variables (cwdp, deadcrootc, etc.)
   - Architecture: 2-layer FC with 64 → 32 units
   - Output: 32-dimensional features

7. **CNN for 2D Variables** (28 variables)
   - Input: Soil and litter properties (including soil variables)
   - Architecture: 4-layer CNN with [32, 64, 128, 256] channels
   - Output: 64-dimensional features

### Feature Fusion

- **Transformer Encoder**: 4-layer transformer with 8 heads
- **Token Dimension**: 128
- **Global Average Pooling**: After transformer processing

### Output Heads

1. **Water Head** (6 variables, optional)
   - Output: Y_H2OCAN, Y_H2OSFC, Y_H2OSNO, Y_TH2OSFC, Y_H2OSOI_LIQ, Y_H2OSOI_ICE

2. **Scalar Head** (5 variables)
   - Output: Y_GPP, Y_NPP, Y_AR, Y_HR, Y_LAI

3. **1D Head** (13 variables)
   - Output: Y_cwdp, Y_deadcrootc, etc.

4. **2D Head** (28 variables)
   - Output: Y_LITR1C_1C_vr, Y_SOILC_1C_vr, etc.

## Usage

### Training with Water Variables

```bash
# Using the dedicated CNP training script
python train_cnp_model.py --include-water

# Using the main training script
python model_training_refactored.py cnp
```

### Training without Water Variables

```bash
# Using the dedicated CNP training script
python train_cnp_model.py --no-water

# Using the main training script
python model_training_refactored.py cnp_no_water
```

### Custom Training Parameters

```bash
python train_cnp_model.py --include-water \
    --epochs 150 \
    --batch-size 64 \
    --learning-rate 0.0005 \
    --output-dir my_cnp_results
```

## Configuration

The model uses two main configuration functions:

- `get_cnp_model_config(include_water=True)`: Configuration with water variables
- `get_cnp_model_config(include_water=False)`: Configuration without water variables

### Key Configuration Parameters

```python
# Model Architecture
lstm_hidden_size=128
static_fc_size=64
fc_hidden_size=128
conv_channels=[32, 64, 128, 256]
transformer_layers=4
transformer_heads=8

# Training
num_epochs=100
batch_size=32
learning_rate=0.001
optimizer_type='adamw'
weight_decay=0.01
```

## Data Structure

### Input Variables (119 with water, 113 without water)

1. **Time Series** (6): FLDS, PSRF, FSDS, QBOT, PRECTmms, TBOT
2. **Surface Properties** (27): Geographic, soil phosphorus forms, PFT coverage, soil texture
3. **PFT Parameters** (44): Plant functional type characteristics
4. **Water Variables** (6, optional): H2OCAN, H2OSFC, H2OSNO, TH2OSFC, H2OSOI_LIQ, H2OSOI_ICE
5. **Scalar Variables** (5): GPP, NPP, AR, HR, LAI
6. **1D Variables** (13): CNP pools
7. **2D Variables** (28): Soil and litter properties (including soil variables)

### Output Variables (45 with water, 39 without water)

1. **Water Variables** (6, optional): Y_H2OCAN, Y_H2OSFC, Y_H2OSNO, Y_TH2OSFC, Y_H2OSOI_LIQ, Y_H2OSOI_ICE
2. **Scalar Variables** (5): Y_GPP, Y_NPP, Y_AR, Y_HR, Y_LAI
3. **1D Variables** (13): Y_CNP pools
4. **2D Variables** (28): Y_soil and Y_litter properties
5. **Temperature Variables** (7): Excluded for first experiments

## Model Files

- `models/cnp_combined_model.py`: Main CNP model implementation
- `config/training_config.py`: CNP configuration functions
- `train_cnp_model.py`: Dedicated CNP training script
- `CNP_IO_list1.txt`: Variable structure specification

## Performance Considerations

1. **Memory Usage**: The model uses ~87 2D variables, which can be memory-intensive
2. **Training Time**: Large model size may require longer training time
3. **GPU Optimization**: Uses mixed precision training and gradient scaling
4. **Early Stopping**: Configured with patience=15 to prevent overfitting

## Monitoring

The training script provides comprehensive logging:

- Model architecture details
- Variable counts for each group
- Training progress and metrics
- GPU memory usage (if available)
- Final model performance

## Output Files

Training produces the following files:

- `cnp_model.pt`: Trained model weights
- `cnp_training_losses.csv`: Training and validation losses
- `cnp_metrics.json`: Final performance metrics
- `cnp_config.json`: Model configuration
- `cnp_predictions/`: Prediction outputs
- `cnp_training.log`: Training log file 