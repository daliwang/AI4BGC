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
    
    # Static columns (can be modified for different inputs)
    static_columns: List[str] = field(default_factory=lambda: [
        'lat', 'lon', 'area', 'landfrac', 'PFT0', 'PFT1', 'PFT2', 'PFT3', 'PFT4', 'PFT5',
        'PFT6', 'PFT7', 'PFT8', 'PFT9', 'PFT10', 'PFT11', 'PFT12', 'PFT13', 'PFT14', 'PFT15'
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
    
    # File loading limits (for testing)
    max_files: Optional[int] = None  # Maximum number of files to load (None = all files)


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
    
    # GPU Optimization
    use_mixed_precision: bool = True  # Use mixed precision training for faster training
    use_amp: bool = True  # Use Automatic Mixed Precision
    use_grad_scaler: bool = True  # Use gradient scaler for mixed precision
    pin_memory: bool = True  # Pin memory for faster data transfer to GPU
    num_workers: int = 0  # Number of data loading workers (0 to avoid GPU context issues)
    prefetch_factor: int = 2  # Number of batches to prefetch
    persistent_workers: bool = False  # Keep workers alive between epochs
    
    # GPU Memory Optimization
    empty_cache_freq: int = 10  # Empty GPU cache every N batches
    max_memory_usage: float = 0.9  # Maximum GPU memory usage (0.9 = 90%)
    memory_efficient_attention: bool = True  # Use memory efficient attention if available
    
    # GPU Monitoring
    log_gpu_memory: bool = True  # Log GPU memory usage
    log_gpu_utilization: bool = True  # Log GPU utilization
    gpu_monitor_interval: int = 100  # Log GPU stats every N batches
    
    # Logging and saving
    save_model: bool = True
    model_save_path: str = "LSTM_model.pt"
    save_losses: bool = True
    losses_save_path: str = "training_validation_losses.csv"
    save_predictions: bool = True
    predictions_dir: str = "predictions"
    
    # Validation
    validation_frequency: int = 1  # Validate every N epochs
    
    # Fair comparison settings
    random_seed: int = 42  # Fixed random seed
    deterministic: bool = True  # Ensure deterministic behavior
    
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
    """Get minimal configuration for fast testing."""
    config = TrainingConfigManager()
    
    # Update data config for minimal testing
    config.update_data_config(
        max_files=3,  # Limit to 3 files for fast testing
        train_split=0.8
    )
    
    # Update model config for minimal testing
    config.update_model_config(
        lstm_hidden_size=64,
        static_fc_size=128,
        num_tokens=8,
        token_dim=64,
        transformer_heads=4,
        transformer_layers=2,
        scalar_output_size=5,  # Number of target columns
        vector_output_size=len(config.data_config.x_list_columns_1d),
        matrix_output_size=len(config.data_config.x_list_columns_2d)
    )
    
    # Update training config for minimal testing
    config.update_training_config(
        num_epochs=2,
        batch_size=32,  # Increased for A100 GPU
        learning_rate=0.001,
        weight_decay=1e-5,
        device='cuda',  # Use GPU for this run
        log_gpu_memory=True,
        prefetch_factor=None
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


def get_full_model_test_cpu_config() -> TrainingConfigManager:
    """Get full model configuration for CPU testing with 3 files."""
    config = TrainingConfigManager()
    
    # Update data config for full model testing with 3 files
    config.update_data_config(
        max_files=3,  # Use 3 files for testing
        train_split=0.8
    )
    
    # Update model config for full model (not minimal)
    config.update_model_config(
        lstm_hidden_size=256,
        static_fc_size=512,
        num_tokens=16,
        token_dim=128,
        transformer_heads=8,
        transformer_layers=4,
        scalar_output_size=5,
        vector_output_size=len(config.data_config.x_list_columns_1d),
        matrix_output_size=len(config.data_config.x_list_columns_2d)
    )
    
    # Update training config for CPU
    config.update_training_config(
        num_epochs=10,
        batch_size=32,  # Smaller batch size for CPU
        learning_rate=0.001,
        weight_decay=1e-5,
        device='cpu',
        log_gpu_memory=False,
        prefetch_factor=None
    )
    
    return config


def get_full_model_test_gpu_config() -> TrainingConfigManager:
    """Get full model configuration for GPU testing with 3 files."""
    config = TrainingConfigManager()
    
    # Update data config for full model testing with 3 files
    config.update_data_config(
        max_files=3,  # Use 3 files for testing
        train_split=0.8
    )
    
    # Update model config for full model (not minimal)
    config.update_model_config(
        lstm_hidden_size=256,
        static_fc_size=512,
        num_tokens=16,
        token_dim=128,
        transformer_heads=8,
        transformer_layers=4,
        scalar_output_size=5,
        vector_output_size=len(config.data_config.x_list_columns_1d),
        matrix_output_size=len(config.data_config.x_list_columns_2d)
    )
    
    # Update training config for GPU
    config.update_training_config(
        num_epochs=10,
        batch_size=64,  # Larger batch size for GPU
        learning_rate=0.001,
        weight_decay=1e-5,
        device='cuda',
        log_gpu_memory=True,
        prefetch_factor=None
    )
    
    return config


def get_full_dataset_config() -> TrainingConfigManager:
    """Get full dataset configuration for production training."""
    config = TrainingConfigManager()
    
    # Update data config for full dataset
    config.update_data_config(
        max_files=None,  # Use all available files
        train_split=0.8
    )
    
    # Update model config for production (larger model)
    config.update_model_config(
        lstm_hidden_size=512,
        static_fc_size=1024,
        num_tokens=32,
        token_dim=256,
        transformer_heads=16,
        transformer_layers=8,
        scalar_output_size=5,
        vector_output_size=len(config.data_config.x_list_columns_1d),
        matrix_output_size=len(config.data_config.x_list_columns_2d)
    )
    
    # Update training config for production (no epoch constraint)
    config.update_training_config(
        num_epochs=500,  # Set to 20 epochs for testing
        batch_size=128,
        learning_rate=0.0005,
        weight_decay=1e-4,
        device='cuda',
        log_gpu_memory=True,
        prefetch_factor=None,
        # Enable early stopping to prevent overfitting
        use_early_stopping=False,
        patience=20,
        min_delta=0.001
    )
    
    return config


def get_fair_comparison_cpu_config() -> TrainingConfigManager:
    """Get CPU configuration for fair comparison with fixed random seeds and no mixed precision."""
    config = TrainingConfigManager()
    
    # Update data config for fair comparison with 3 files
    config.update_data_config(
        max_files=3,  # Use 3 files for testing
        train_split=0.8
    )
    
    # Update model config for fair comparison
    config.update_model_config(
        lstm_hidden_size=256,
        static_fc_size=512,
        num_tokens=16,
        token_dim=128,
        transformer_heads=8,
        transformer_layers=4,
        scalar_output_size=5,
        vector_output_size=len(config.data_config.x_list_columns_1d),
        matrix_output_size=len(config.data_config.x_list_columns_2d)
    )
    
    # Update training config for fair CPU comparison
    config.update_training_config(
        num_epochs=10,
        batch_size=32,  # Smaller batch size for CPU
        learning_rate=0.001,
        weight_decay=1e-5,
        device='cpu',
        log_gpu_memory=False,
        prefetch_factor=None,
        # Fair comparison settings
        random_seed=42,  # Fixed random seed
        use_amp=False,   # No mixed precision
        deterministic=True  # Ensure deterministic behavior
    )
    
    return config


def get_fair_comparison_gpu_config() -> TrainingConfigManager:
    """Get GPU configuration for fair comparison with fixed random seeds and no mixed precision."""
    config = TrainingConfigManager()
    
    # Update data config for fair comparison with 3 files
    config.update_data_config(
        max_files=3,  # Use 3 files for testing
        train_split=0.8
    )
    
    # Update model config for fair comparison
    config.update_model_config(
        lstm_hidden_size=256,
        static_fc_size=512,
        num_tokens=16,
        token_dim=128,
        transformer_heads=8,
        transformer_layers=4,
        scalar_output_size=5,
        vector_output_size=len(config.data_config.x_list_columns_1d),
        matrix_output_size=len(config.data_config.x_list_columns_2d)
    )
    
    # Update training config for fair GPU comparison
    config.update_training_config(
        num_epochs=10,
        batch_size=32,  # Same batch size as CPU for fair comparison
        learning_rate=0.001,
        weight_decay=1e-5,
        device='cuda',
        log_gpu_memory=True,
        prefetch_factor=None,
        # Fair comparison settings
        random_seed=42,  # Same random seed as CPU
        use_amp=False,   # No mixed precision for fair comparison
        deterministic=True  # Ensure deterministic behavior
    )
    
    return config


def get_dataset1_config() -> TrainingConfigManager:
    """Get configuration for training on Dataset 1 only (0_trendy_case)."""
    config = TrainingConfigManager()
    
    # Update data config for Dataset 1 only
    config.update_data_config(
        data_paths=["/global/cfs/cdirs/m4814/daweigao/0_trendy_case/dataset"],
        max_files=None,  # Use all available files
        train_split=0.8
    )
    
    # Update model config for Dataset 1
    config.update_model_config(
        lstm_hidden_size=512,
        static_fc_size=1024,
        num_tokens=32,
        token_dim=256,
        transformer_heads=16,
        transformer_layers=8,
        scalar_output_size=5,
        vector_output_size=len(config.data_config.x_list_columns_1d),
        matrix_output_size=len(config.data_config.x_list_columns_2d)
    )
    
    # Update training config for Dataset 1 with comprehensive saving settings
    config.update_training_config(
        num_epochs=100,  # Set to 150 epochs as requested
        batch_size=128,
        learning_rate=0.0005,
        scalar_loss_weight=1.0,
        vector_loss_weight=1.0,
        matrix_loss_weight=1.0,
        optimizer_type='adam',
        weight_decay=1e-4,
        use_scheduler=False,
        use_early_stopping=False,  # Disable early stopping for safety
        patience=15,  # Not used when early stopping is disabled
        min_delta=0.001,
        device='cuda',
        use_mixed_precision=True,
        use_amp=True,
        use_grad_scaler=True,
        pin_memory=True,
        num_workers=0,
        prefetch_factor=None,  # Fixed: must be None when num_workers=0
        persistent_workers=False,
        empty_cache_freq=10,
        max_memory_usage=0.9,
        memory_efficient_attention=True,
        log_gpu_memory=True,
        log_gpu_utilization=True,
        gpu_monitor_interval=100,
        save_model=True,
        model_save_path="model.pt",  # Will be saved in case directory
        save_losses=True,
        losses_save_path="training_validation_losses.csv",  # Will be saved in case directory
        save_predictions=True,
        predictions_dir="predictions",  # Will be saved in case directory
        validation_frequency=1,
        random_seed=42,
        deterministic=True
    )
    
    return config


def get_dataset2_config() -> TrainingConfigManager:
    """Get configuration for training on Dataset 2 only (1_0.5_degree)."""
    config = TrainingConfigManager()
    
    # Update data config for Dataset 2 only - without filtering
    config.update_data_config(
        data_paths=["/global/cfs/cdirs/m4814/daweigao/1_0.5_degree/dataset"],
        max_files=None,  # Use all available files
        train_split=0.8,
        filter_column=None  # Remove filtering to use all samples
    )
    
    # Update model config for Dataset 2
    config.update_model_config(
        lstm_hidden_size=512,
        static_fc_size=1024,
        num_tokens=32,
        token_dim=256,
        transformer_heads=16,
        transformer_layers=8,
        scalar_output_size=5,
        vector_output_size=len(config.data_config.x_list_columns_1d),
        matrix_output_size=len(config.data_config.x_list_columns_2d)
    )
    
    # Update training config for Dataset 2 with reduced batch size and more epochs
    config.update_training_config(
        num_epochs=150,  # Increased from 50 to 150
        batch_size=32,   # Reduced batch size to handle larger dataset
        learning_rate=0.001,
        scalar_loss_weight=1.0,
        vector_loss_weight=1.0,
        matrix_loss_weight=1.0,
        optimizer_type='adam',
        weight_decay=0.0,
        use_scheduler=False,
        use_early_stopping=True,
        patience=15,  # Increased patience for longer training
        min_delta=0.001,
        device='auto',
        use_mixed_precision=True,
        use_amp=True,
        use_grad_scaler=True,
        pin_memory=True,
        num_workers=0,
        prefetch_factor=None,  # Fixed: must be None when num_workers=0
        persistent_workers=False,
        empty_cache_freq=10,
        max_memory_usage=0.9,
        memory_efficient_attention=True,
        log_gpu_memory=True,
        log_gpu_utilization=True,
        gpu_monitor_interval=100,
        save_model=True,
        model_save_path="dataset2_model.pt",
        save_losses=True,
        losses_save_path="dataset2_training_validation_losses.csv",
        save_predictions=True,
        predictions_dir="dataset2_predictions",
        validation_frequency=1,
        random_seed=42,
        deterministic=True
    )
    
    return config


def get_combined_dataset_config() -> TrainingConfigManager:
    """Get configuration for training on both datasets combined."""
    config = TrainingConfigManager()
    
    # Update data config for combined dataset (both paths)
    config.update_data_config(
        data_paths=[
            "/global/cfs/cdirs/m4814/daweigao/0_trendy_case/dataset",
            "/global/cfs/cdirs/m4814/daweigao/1_0.5_degree/dataset"
        ],
        max_files=None,  # Use all available files
        train_split=0.8
    )
    
    # Update model config for combined dataset
    config.update_model_config(
        lstm_hidden_size=512,
        static_fc_size=1024,
        num_tokens=32,
        token_dim=256,
        transformer_heads=16,
        transformer_layers=8,
        scalar_output_size=5,
        vector_output_size=len(config.data_config.x_list_columns_1d),
        matrix_output_size=len(config.data_config.x_list_columns_2d)
    )
    
    # Update training config for combined dataset with comprehensive saving settings
    config.update_training_config(
        num_epochs=150,  # Set to 150 epochs for consistency
        batch_size=128,
        learning_rate=0.0005,
        scalar_loss_weight=1.0,
        vector_loss_weight=1.0,
        matrix_loss_weight=1.0,
        optimizer_type='adam',
        weight_decay=1e-4,
        use_scheduler=False,
        use_early_stopping=True,
        patience=15,  # Early stopping patience
        min_delta=0.001,
        device='cuda',
        use_mixed_precision=True,
        use_amp=True,
        use_grad_scaler=True,
        pin_memory=True,
        num_workers=0,
        prefetch_factor=None,  # Fixed: must be None when num_workers=0
        persistent_workers=False,
        empty_cache_freq=10,
        max_memory_usage=0.9,
        memory_efficient_attention=True,
        log_gpu_memory=True,
        log_gpu_utilization=True,
        gpu_monitor_interval=100,
        save_model=True,
        model_save_path="combined_model.pt",
        save_losses=True,
        losses_save_path="combined_training_validation_losses.csv",
        save_predictions=True,
        predictions_dir="combined_predictions",
        validation_frequency=1,
        random_seed=42,
        deterministic=True
    )
    
    return config 