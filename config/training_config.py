"""
Configuration module for model training.

This module centralizes all training parameters and configurations,
making it easy to modify training settings without changing the core code.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Data paths
    data_paths: List[str] = field(default_factory=lambda: [
        "/global/cfs/cdirs/m4814/daweigao/0_trendy_case/dataset",
        "/global/cfs/cdirs/m4814/daweigao/1_0.5_degree/dataset"
    ])
    
    # File patterns
    file_pattern: str = "1_training_data_batch_*.pkl"
    
    # Columns to drop
    columns_to_drop: List[str] = field(default_factory=lambda: [
        'Y_OCCLUDED_P', 'Y_SECONDARY_P', 'Y_LABILE_P', 'Y_APATITE_P'
    ])
    
    # Time series columns (can be modified for different inputs)
    time_series_columns: List[str] = field(default_factory=lambda: [
        'FLDS', 'PSRF', 'FSDS', 'QBOT', 'PRECTmms', 'TBOT'
    ])
    
    # 2D input features (can be modified for different inputs)
    x_list_columns_2d: List[str] = field(default_factory=lambda: [
        'soil3c_vr', 'soil4c_vr', 'cwdc_vr'
    ])
    
    # 2D target features (can be modified for different outputs)
    y_list_columns_2d: List[str] = field(default_factory=lambda: [
        'Y_soil3c_vr', 'Y_soil4c_vr', 'Y_cwdc_vr'
    ])
    
    # 1D input features (can be modified for different inputs)
    x_list_columns_1d: List[str] = field(default_factory=lambda: [
        'deadcrootc', 'deadstemc', 'tlai'
    ])
    
    # 1D target features (can be modified for different outputs)
    y_list_columns_1d: List[str] = field(default_factory=lambda: [
        'Y_deadcrootc', 'Y_deadstemc', 'Y_tlai'
    ])
    
    # Data preprocessing settings
    time_series_length: int = 240
    max_time_series_length: int = 1476
    max_1d_length: int = 16
    max_2d_rows: int = 18
    max_2d_cols: int = 10
    
    # Data splitting
    train_split: float = 0.8
    random_state: int = 42
    
    # Filtering
    filter_column: Optional[str] = 'H2OSOI_10CM'  # Column to filter NaN values


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    # LSTM parameters
    lstm_hidden_size: int = 64
    
    # Fully connected layers
    fc_hidden_size: int = 32
    static_fc_size: int = 64
    
    # CNN parameters for 2D data
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    conv_kernel_size: int = 3
    conv_padding: int = 1
    
    # Transformer parameters
    num_tokens: int = 4
    token_dim: int = 64
    transformer_layers: int = 2
    transformer_heads: int = 4
    
    # Output dimensions
    scalar_output_size: int = 5
    vector_output_size: int = 3
    vector_length: int = 16
    matrix_output_size: int = 3
    matrix_rows: int = 18
    matrix_cols: int = 10


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Basic training parameters
    num_epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.001
    
    # Loss weights (can be adjusted for different output priorities)
    scalar_loss_weight: float = 1.0
    vector_loss_weight: float = 1.0
    matrix_loss_weight: float = 1.0
    
    # Optimizer
    optimizer_type: str = 'adam'  # 'adam', 'sgd', 'adamw'
    weight_decay: float = 0.0
    
    # Learning rate scheduler
    use_scheduler: bool = False
    scheduler_type: str = 'step'  # 'step', 'cosine', 'plateau'
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    
    # Early stopping
    use_early_stopping: bool = False
    patience: int = 10
    min_delta: float = 0.001
    
    # Device
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    
    # Logging and saving
    save_model: bool = True
    model_save_path: str = "LSTM_model.pt"
    save_losses: bool = True
    losses_save_path: str = "training_validation_losses.csv"
    save_predictions: bool = True
    predictions_dir: str = "predictions"
    
    # Validation
    validation_frequency: int = 1  # Validate every N epochs
    
    def get_device(self) -> torch.device:
        """Get the appropriate device for training."""
        if self.device == 'auto':
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(self.device)
    
    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Get the optimizer based on configuration."""
        if self.optimizer_type.lower() == 'adam':
            return optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type.lower() == 'sgd':
            return optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type.lower() == 'adamw':
            return optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
    
    def get_scheduler(self, optimizer: optim.Optimizer) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Get the learning rate scheduler based on configuration."""
        if not self.use_scheduler:
            return None
        
        if self.scheduler_type.lower() == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma
            )
        elif self.scheduler_type.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        elif self.scheduler_type.lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=self.scheduler_gamma, patience=self.scheduler_step_size
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    
    # Normalization methods
    time_series_normalization: str = 'minmax'  # 'minmax', 'standard', 'robust'
    static_normalization: str = 'minmax'
    target_normalization: str = 'minmax'
    list_1d_normalization: str = 'minmax'
    list_2d_normalization: str = 'minmax'
    
    # Normalization ranges
    minmax_range: tuple = (0, 1)
    
    # Data type
    data_type: torch.dtype = torch.float32
    
    # Memory management
    memory_save_threshold: int = 50  # Save to disk every N variables


class TrainingConfigManager:
    """Manager class for training configurations."""
    
    def __init__(self):
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        self.training_config = TrainingConfig()
        self.preprocessing_config = PreprocessingConfig()
    
    def update_data_config(self, **kwargs):
        """Update data configuration."""
        for key, value in kwargs.items():
            if hasattr(self.data_config, key):
                setattr(self.data_config, key, value)
            else:
                raise ValueError(f"Unknown data config parameter: {key}")
    
    def update_model_config(self, **kwargs):
        """Update model configuration."""
        for key, value in kwargs.items():
            if hasattr(self.model_config, key):
                setattr(self.model_config, key, value)
            else:
                raise ValueError(f"Unknown model config parameter: {key}")
    
    def update_training_config(self, **kwargs):
        """Update training configuration."""
        for key, value in kwargs.items():
            if hasattr(self.training_config, key):
                setattr(self.training_config, key, value)
            else:
                raise ValueError(f"Unknown training config parameter: {key}")
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configurations as a dictionary."""
        return {
            'data_config': self.data_config,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'preprocessing_config': self.preprocessing_config
        }


# Predefined configurations for different scenarios
def get_default_config() -> TrainingConfigManager:
    """Get default configuration."""
    return TrainingConfigManager()


def get_minimal_config() -> TrainingConfigManager:
    """Get minimal configuration for quick testing."""
    config = TrainingConfigManager()
    config.update_training_config(num_epochs=5, batch_size=8)
    config.update_data_config(
        x_list_columns_2d=['soil3c_vr'],
        y_list_columns_2d=['Y_soil3c_vr'],
        x_list_columns_1d=['deadcrootc'],
        y_list_columns_1d=['Y_deadcrootc']
    )
    return config


def get_extended_config() -> TrainingConfigManager:
    """Get extended configuration with more features."""
    config = TrainingConfigManager()
    config.update_data_config(
        x_list_columns_2d=['soil3c_vr', 'soil4c_vr', 'cwdc_vr', 'litr1c_vr', 'litr2c_vr'],
        y_list_columns_2d=['Y_soil3c_vr', 'Y_soil4c_vr', 'Y_cwdc_vr', 'Y_litr1c_vr', 'Y_litr2c_vr'],
        x_list_columns_1d=['deadcrootc', 'deadstemc', 'tlai', 'tveg'],
        y_list_columns_1d=['Y_deadcrootc', 'Y_deadstemc', 'Y_tlai', 'Y_tveg']
    )
    config.update_model_config(
        lstm_hidden_size=128,
        fc_hidden_size=64,
        transformer_layers=3
    )
    config.update_training_config(
        num_epochs=100,
        batch_size=32,
        learning_rate=0.0005
    )
    return config 