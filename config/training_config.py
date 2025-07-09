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
    
    # Fixed static columns to ensure consistency across datasets
    static_columns: List[str] = field(default_factory=lambda: [
        'lat', 'lon', 'area', 'landfrac', 'PFT0', 'PFT1', 'PFT2', 'PFT3', 'PFT4', 'PFT5',
        'PFT6', 'PFT7', 'PFT8', 'PFT9', 'PFT10', 'PFT11', 'PFT12', 'PFT13', 'PFT14', 'PFT15',
        'PCT_NAT_PFT_0', 'PCT_NAT_PFT_1', 'PCT_NAT_PFT_2', 'PCT_NAT_PFT_3', 'PCT_NAT_PFT_4',
        'PCT_NAT_PFT_5', 'PCT_NAT_PFT_6', 'PCT_NAT_PFT_7', 'PCT_NAT_PFT_8', 'PCT_NAT_PFT_9',
        'PCT_NAT_PFT_10', 'PCT_NAT_PFT_11', 'PCT_NAT_PFT_12', 'PCT_NAT_PFT_13', 'PCT_NAT_PFT_14',
        'PCT_NAT_PFT_15', 'PCT_NAT_PFT_16', 'PCT_NATVEG', 'LANDFRAC_PFT', 'PCT_CLAY_0',
        'PCT_CLAY_1', 'PCT_CLAY_2', 'PCT_CLAY_3', 'PCT_CLAY_4', 'PCT_CLAY_5', 'PCT_CLAY_6',
        'PCT_CLAY_7', 'PCT_CLAY_8', 'PCT_CLAY_9', 'PCT_SAND_0', 'PCT_SAND_1', 'PCT_SAND_2',
        'PCT_SAND_3', 'PCT_SAND_4', 'PCT_SAND_5', 'PCT_SAND_6', 'PCT_SAND_7', 'PCT_SAND_8',
        'PCT_SAND_9', 'SCALARAVG_vr_0', 'SCALARAVG_vr_1', 'SCALARAVG_vr_2', 'SCALARAVG_vr_3',
        'SCALARAVG_vr_4', 'SCALARAVG_vr_5', 'SCALARAVG_vr_6', 'SCALARAVG_vr_7', 'SCALARAVG_vr_8',
        'SCALARAVG_vr_9', 'SCALARAVG_vr_10', 'SCALARAVG_vr_11', 'SCALARAVG_vr_12', 'SCALARAVG_vr_13',
        'SCALARAVG_vr_14', 'SOIL_ORDER', 'SOIL_COLOR', 'SNOWDP', 'peatf', 'abm', 'NPP', 'GPP',
        'HR', 'AR', 'COL_FIRE_CLOSS', 'OCCLUDED_P', 'SECONDARY_P', 'LABILE_P', 'APATITE_P',
        'H2OSOI_10CM', 'Y_NPP', 'Y_GPP', 'Y_HR', 'Y_AR', 'Y_COL_FIRE_CLOSS'
    ])
    
    # Reorganized input groups for enhanced model
    static_surface_columns: List[str] = field(default_factory=list)
    water_group_columns: List[str] = field(default_factory=list)
    temperature_group_columns: List[str] = field(default_factory=list)
    
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
    
    # New parameter for filtering NaN in time series
    filter_time_series_nan: bool = False


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


def get_dataset3_config() -> TrainingConfigManager:
    """Get configuration for training on Dataset 3 only (3_trendy_case_add_water_variables)."""
    config = TrainingConfigManager()
    
    # Update data config for Dataset 3 only
    config.update_data_config(
        data_paths=["/global/cfs/cdirs/m4814/daweigao/3_trendy_case_add_water_variables/dataset"],
        max_files=None,  # Use all available files
        train_split=0.8,
        filter_column=None  # Remove filtering to use all samples
    )
    
    # Update model config for Dataset 3
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
    
    # Update training config for Dataset 3
    config.update_training_config(
        num_epochs=150,
        batch_size=32,   # Reduced batch size to handle larger dataset
        learning_rate=0.001,
        scalar_loss_weight=1.0,
        vector_loss_weight=1.0,
        matrix_loss_weight=1.0,
        optimizer_type='adam',
        weight_decay=0.0,
        use_scheduler=False,
        use_early_stopping=True,
        patience=15,
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
        model_save_path="dataset3_model.pt",
        save_losses=True,
        losses_save_path="dataset3_training_validation_losses.csv",
        save_predictions=True,
        predictions_dir="dataset3_predictions",
        validation_frequency=1,
        random_seed=42,
        deterministic=True
    )
    
    return config


def get_reorganized_dataset3_config() -> TrainingConfigManager:
    """Get reorganized configuration for training on Dataset 3 with separated input groups."""
    config = TrainingConfigManager()
    
    # Reorganized data config for Dataset 3 with separated input groups
    config.update_data_config(
        data_paths=["/global/cfs/cdirs/m4814/daweigao/3_trendy_case_add_water_variables/dataset"],
        max_files=None,  # Use all available files
        train_split=0.8,
        filter_column=None,  # Remove filtering to use all samples
        # Time series columns (unchanged)
        time_series_columns=['FLDS', 'PSRF', 'FSDS', 'QBOT', 'PRECTmms', 'TBOT'],
        # Static surface columns (geographic, PFT, soil, auxiliary)
        static_surface_columns=[
            # Geographic
            'lat', 'lon', 'area', 'landfrac',
            # PFTs
            'PFT0', 'PFT1', 'PFT2', 'PFT3', 'PFT4', 'PFT5', 'PFT6', 'PFT7', 'PFT8', 'PFT9',
            'PFT10', 'PFT11', 'PFT12', 'PFT13', 'PFT14', 'PFT15',
            'PCT_NAT_PFT_0', 'PCT_NAT_PFT_1', 'PCT_NAT_PFT_2', 'PCT_NAT_PFT_3', 'PCT_NAT_PFT_4',
            'PCT_NAT_PFT_5', 'PCT_NAT_PFT_6', 'PCT_NAT_PFT_7', 'PCT_NAT_PFT_8', 'PCT_NAT_PFT_9',
            'PCT_NAT_PFT_10', 'PCT_NAT_PFT_11', 'PCT_NAT_PFT_12', 'PCT_NAT_PFT_13', 'PCT_NAT_PFT_14',
            'PCT_NAT_PFT_15', 'PCT_NAT_PFT_16', 'PCT_NATVEG', 'LANDFRAC_PFT',
            # Soil properties
            'PCT_CLAY_0', 'PCT_CLAY_1', 'PCT_CLAY_2', 'PCT_CLAY_3', 'PCT_CLAY_4', 'PCT_CLAY_5',
            'PCT_CLAY_6', 'PCT_CLAY_7', 'PCT_CLAY_8', 'PCT_CLAY_9',
            'PCT_SAND_0', 'PCT_SAND_1', 'PCT_SAND_2', 'PCT_SAND_3', 'PCT_SAND_4', 'PCT_SAND_5',
            'PCT_SAND_6', 'PCT_SAND_7', 'PCT_SAND_8', 'PCT_SAND_9',
            'SCALARAVG_vr_0', 'SCALARAVG_vr_1', 'SCALARAVG_vr_2', 'SCALARAVG_vr_3', 'SCALARAVG_vr_4',
            'SCALARAVG_vr_5', 'SCALARAVG_vr_6', 'SCALARAVG_vr_7', 'SCALARAVG_vr_8', 'SCALARAVG_vr_9',
            'SCALARAVG_vr_10', 'SCALARAVG_vr_11', 'SCALARAVG_vr_12', 'SCALARAVG_vr_13', 'SCALARAVG_vr_14',
            'SOIL_ORDER', 'SOIL_COLOR',
            # Auxiliary variables
            'SNOWDP', 'peatf', 'abm', 'GPP', 'HR', 'AR', 'NPP', 'COL_FIRE_CLOSS',
            'OCCLUDED_P', 'SECONDARY_P', 'LABILE_P', 'APATITE_P',
            'sminn_vr', 'smin_no3_vr', 'smin_nh4_vr', 'LAKE_SOILC', 'taf'
        ],
        # Water group columns (separate processing)
        water_group_columns=[
            'H2OCAN', 'H2OSFC', 'H2OSNO', 'H2OSOI_LIQ', 'H2OSOI_ICE', 'H2OSOI_10CM'
        ],
        # Temperature group columns (separate processing)
        temperature_group_columns=[
            'T_VEG', 'T10_VALUE', 'TH2OSFC', 'T_GRND', 'T_GRND_R', 'T_GRND_U', 'T_SOISNO', 'T_LAKE', 'TS_TOPO'
        ],
        # Enhanced 1D CNP input/output features
        x_list_columns_1d=['deadcrootc', 'deadstemc', 'tlai', 'leafc', 'frootc', 'totlitc'],
        y_list_columns_1d=['Y_deadcrootc', 'Y_deadstemc', 'Y_tlai', 'Y_leafc', 'Y_frootc', 'Y_totlitc'],
        # Enhanced 2D CNP input/output features
        x_list_columns_2d=['soil1c_vr', 'soil2c_vr', 'soil3c_vr', 'soil4c_vr', 'litr1c_vr', 'litr2c_vr', 'litr3c_vr'],
        y_list_columns_2d=['Y_soil1c_vr', 'Y_soil2c_vr', 'Y_soil3c_vr', 'Y_soil4c_vr', 'Y_litr1c_vr', 'Y_litr2c_vr', 'Y_litr3c_vr']
    )
    
    # Enhanced model config for reorganized Dataset 3
    config.update_model_config(
        lstm_hidden_size=512,
        static_fc_size=1024,
        fc_hidden_size=128,  # Increased for more complex features
        num_tokens=32,
        token_dim=128,  # Fixed token dimension for stability
        transformer_heads=16,
        transformer_layers=6,  # More transformer layers
        scalar_output_size=5,
        vector_output_size=6,  # Increased for more 1D outputs
        matrix_output_size=7,  # Increased for more 2D outputs
        vector_length=16,
        matrix_rows=18,
        matrix_cols=10
    )
    
    # Enhanced training config for reorganized Dataset 3
    config.update_training_config(
        num_epochs=150,
        batch_size=32,   # Reduced batch size for larger model
        learning_rate=0.0005,  # Reduced learning rate for stability
        scalar_loss_weight=1.0,
        vector_loss_weight=1.0,
        matrix_loss_weight=1.0,
        optimizer_type='adam',
        weight_decay=1e-4,  # Added weight decay for regularization
        use_scheduler=True,  # Enable learning rate scheduling
        scheduler_type='cosine',
        use_early_stopping=True,
        patience=20,  # Increased patience for complex model
        min_delta=0.001,
        device='auto',
        use_mixed_precision=True,
        use_amp=True,
        use_grad_scaler=True,
        pin_memory=True,
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=False,
        empty_cache_freq=10,
        max_memory_usage=0.9,
        memory_efficient_attention=True,
        log_gpu_memory=True,
        log_gpu_utilization=True,
        gpu_monitor_interval=100,
        save_model=True,
        model_save_path="reorganized_dataset3_model.pt",
        save_losses=True,
        losses_save_path="reorganized_dataset3_training_validation_losses.csv",
        save_predictions=True,
        predictions_dir="reorganized_dataset3_predictions",
        validation_frequency=1,
        random_seed=42,
        deterministic=True
    )
    
    return config


def get_grouped_enhanced_dataset3_config() -> TrainingConfigManager:
    """Get grouped enhanced configuration for training on Dataset 3 with grouped architecture."""
    config = TrainingConfigManager()
    
    # Data config for Dataset 3 with grouped input structure
    config.update_data_config(
        data_paths=["/global/cfs/cdirs/m4814/daweigao/3_trendy_case_add_water_variables/dataset"],
        max_files=None,
        train_split=0.8,
        filter_column=None,
        # Time series columns (forcing group)
        time_series_columns=['FLDS', 'PSRF', 'FSDS', 'QBOT', 'PRECTmms', 'TBOT'],
        # Static surface columns (geographic, PFT, soil properties)
        static_surface_columns=[
            # Geographic
            'lat', 'lon', 'area', 'landfrac',
            # PFTs
            'PFT0', 'PFT1', 'PFT2', 'PFT3', 'PFT4', 'PFT5', 'PFT6', 'PFT7', 'PFT8', 'PFT9',
            'PFT10', 'PFT11', 'PFT12', 'PFT13', 'PFT14', 'PFT15',
            'PCT_NAT_PFT_0', 'PCT_NAT_PFT_1', 'PCT_NAT_PFT_2', 'PCT_NAT_PFT_3', 'PCT_NAT_PFT_4',
            'PCT_NAT_PFT_5', 'PCT_NAT_PFT_6', 'PCT_NAT_PFT_7', 'PCT_NAT_PFT_8', 'PCT_NAT_PFT_9',
            'PCT_NAT_PFT_10', 'PCT_NAT_PFT_11', 'PCT_NAT_PFT_12', 'PCT_NAT_PFT_13', 'PCT_NAT_PFT_14',
            'PCT_NAT_PFT_15', 'PCT_NAT_PFT_16', 'PCT_NATVEG', 'LANDFRAC_PFT',
            # Soil properties
            'PCT_CLAY_0', 'PCT_CLAY_1', 'PCT_CLAY_2', 'PCT_CLAY_3', 'PCT_CLAY_4', 'PCT_CLAY_5',
            'PCT_CLAY_6', 'PCT_CLAY_7', 'PCT_CLAY_8', 'PCT_CLAY_9',
            'PCT_SAND_0', 'PCT_SAND_1', 'PCT_SAND_2', 'PCT_SAND_3', 'PCT_SAND_4', 'PCT_SAND_5',
            'PCT_SAND_6', 'PCT_SAND_7', 'PCT_SAND_8', 'PCT_SAND_9',
            'SCALARAVG_vr_0', 'SCALARAVG_vr_1', 'SCALARAVG_vr_2', 'SCALARAVG_vr_3', 'SCALARAVG_vr_4',
            'SCALARAVG_vr_5', 'SCALARAVG_vr_6', 'SCALARAVG_vr_7', 'SCALARAVG_vr_8', 'SCALARAVG_vr_9',
            'SCALARAVG_vr_10', 'SCALARAVG_vr_11', 'SCALARAVG_vr_12', 'SCALARAVG_vr_13', 'SCALARAVG_vr_14',
            'SOIL_ORDER', 'SOIL_COLOR', 'SNOWDP', 'peatf', 'abm', 'NPP', 'GPP', 'HR', 'AR', 
            'COL_FIRE_CLOSS', 'OCCLUDED_P', 'SECONDARY_P', 'LABILE_P', 'APATITE_P',
            'H2OSOI_10CM', 'Y_NPP', 'Y_GPP', 'Y_HR', 'Y_AR', 'Y_COL_FIRE_CLOSS'
        ],
        # Water group columns
        water_group_columns=['H2OCAN', 'H2OSFC', 'H2OSNO', 'H2OSOI_LIQ', 'H2OSOI_ICE', 'H2OSOI_10CM'],
        # Temperature group columns
        temperature_group_columns=['T_VEG', 'T10_VALUE', 'TH2OSFC', 'T_GRND', 'T_GRND_R', 'T_GRND_U', 'T_SOISNO', 'T_LAKE', 'TS_TOPO'],
        # 1D CNP columns
        x_list_columns_1d=['deadcrootc', 'deadstemc', 'tlai', 'leafc', 'frootc', 'totlitc'],
        y_list_columns_1d=['Y_deadcrootc', 'Y_deadstemc', 'Y_tlai', 'Y_leafc', 'Y_frootc', 'Y_totlitc'],
        # 2D CNP columns
        x_list_columns_2d=['soil1c_vr', 'soil2c_vr', 'soil3c_vr', 'soil4c_vr', 'litr1c_vr', 'litr2c_vr', 'litr3c_vr'],
        y_list_columns_2d=['Y_soil1c_vr', 'Y_soil2c_vr', 'Y_soil3c_vr', 'Y_soil4c_vr', 'Y_litr1c_vr', 'Y_litr2c_vr', 'Y_litr3c_vr']
    )
    
    # Enhanced model config for grouped architecture
    config.update_model_config(
        lstm_hidden_size=512,
        fc_hidden_size=256,
        static_fc_size=256,
        conv_channels=[64, 128, 256],
        conv_kernel_size=3,
        conv_padding=1,
        num_tokens=8,
        token_dim=256,
        transformer_layers=6,
        transformer_heads=16,
        scalar_output_size=5,
        vector_output_size=6,
        vector_length=16,
        matrix_output_size=7,
        matrix_rows=18,
        matrix_cols=10
    )
    
    # Enhanced training config
    config.update_training_config(
        batch_size=32,
        learning_rate=0.001,
        num_epochs=100,
        early_stopping_patience=15,
        model_save_path="results/grouped_enhanced_dataset3_model.pt",
        losses_save_path="results/grouped_enhanced_dataset3_losses.csv",
        predictions_dir="results/grouped_enhanced_dataset3_predictions"
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
    """Get configuration for training on all three datasets combined."""
    config = TrainingConfigManager()
    
    # Update data config for combined dataset (all three paths) - no file limit
    config.update_data_config(
        data_paths=[
            "/global/cfs/cdirs/m4814/daweigao/0_trendy_case/dataset",
            "/global/cfs/cdirs/m4814/daweigao/1_0.5_degree/dataset",
            "/global/cfs/cdirs/m4814/daweigao/3_trendy_case_add_water_variables/dataset"
        ],
        max_files=None,  # Use all available files
        train_split=0.8,
        filter_column=None,
        filter_time_series_nan=True
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
    
    # Update training config for combined dataset with 120 epochs
    config.update_training_config(
        num_epochs=120,  # Set to 120 epochs for full training
        batch_size=128,
        learning_rate=0.0005,
        scalar_loss_weight=1.0,
        vector_loss_weight=1.0,
        matrix_loss_weight=1.0,
        optimizer_type='adam',
        weight_decay=1e-4,
        use_scheduler=False,
        use_early_stopping=True,  # Enable early stopping for 120-epoch training
        patience=20,  # Increased patience for longer training
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


def get_cnp_model_config(include_water: bool = True) -> TrainingConfigManager:
    """
    Get CNP model configuration based on CNP_IO_list1.txt structure.
    
    Args:
        include_water: Whether to include water variables in both input and output
        
    Returns:
        TrainingConfigManager with CNP model configuration
    """
    config = TrainingConfigManager()
    
    # Data paths for Dataset 3
    config.update_data_config(
        data_paths=["/global/cfs/cdirs/m4814/wangd/AI4BGC/TrainingData/Trend_1_data_CNP"],
        max_files=None,  # Use all available files
        train_split=0.8,
        filter_column=None,  # Remove filtering to use all samples
        time_series_length=240,  # 20 years * 12 months
        max_time_series_length=240,
        max_1d_length=16,
        max_2d_rows=10,  # Soil layers
        max_2d_cols=10,  # Soil layers
    )
    
    # Time series variables (6 variables)
    config.update_data_config(
        time_series_columns=['FLDS', 'PSRF', 'FSDS', 'QBOT', 'PRECTmms', 'TBOT']
    )
    
    # Surface Properties (39 variables) - Geographic, Soil Phosphorus Forms, PFT Coverage, Soil Texture
    surface_properties = [
        # Geographic (7 variables)
        'lat', 'lon', 'area', 'landfrac', 'Latitude', 'Longitude', 'AREA',
        # Soil Phosphorus Forms (4 variables)
        'OCCLUDED_P', 'SECONDARY_P', 'LABILE_P', 'APATITE_P',
        # PFT Coverage (20 variables)
        'PCT_NAT_PFT0', 'PCT_NAT_PFT1', 'PCT_NAT_PFT2', 'PCT_NAT_PFT3', 'PCT_NAT_PFT4', 
        'PCT_NAT_PFT5', 'PCT_NAT_PFT6', 'PCT_NAT_PFT7', 'PCT_NAT_PFT8', 'PCT_NAT_PFT9', 
        'PCT_NAT_PFT10', 'PCT_NAT_PFT11', 'PCT_NAT_PFT12', 'PCT_NAT_PFT13', 'PCT_NAT_PFT14', 
        'PCT_NAT_PFT15', 'PCT_NAT_PFT16', 'PCT_NAT_PFT17', 'LANDFRAC_PFT', 'PCT_NATVEG', 'SNOWDP',
        # Soil Texture (20 variables)
        'CLAY_1_', 'CLAY_2_', 'CLAY_3_', 'CLAY_4_', 'CLAY_5_', 'CLAY_6_', 'CLAY_7_', 'CLAY_8_', 'CLAY_9_', 'CLAY_10_',
        'SAND_1_', 'SAND_2_', 'SAND_3_', 'SAND_4_', 'SAND_5_', 'SAND_6_', 'SAND_7_', 'SAND_8_', 'SAND_9_', 'SAND_10_'
    ]
    
    # PFT Parameters (44 variables)
    pft_parameters = [
        'pft_deadwdcn', 'pft_frootcn', 'pft_leafcn', 'pft_lflitcn', 'pft_livewdcn',
        'pft_c3psn', 'pft_croot_stem', 'pft_crop', 'pft_dleaf', 'pft_dsladlai', 'pft_evergreen', 
        'pft_fcur', 'pft_flivewd', 'pft_flnr', 'pft_fr_fcel', 'pft_fr_flab', 'pft_fr_flig', 
        'pft_froot_leaf', 'pft_grperc', 'pft_grpnow', 'pft_leaf_long', 'pft_lf_fcel', 
        'pft_lf_flab', 'pft_lf_flig', 'pft_rholnir', 'pft_rholvis', 'pft_rhosnir', 'pft_rhosvis', 
        'pft_roota_par', 'pft_rootb_par', 'pft_rootprof_beta', 'pft_season_decid', 'pft_slatop', 
        'pft_smpsc', 'pft_smpso', 'pft_stem_leaf', 'pft_stress_decid', 'pft_taulnir', 
        'pft_taulvis', 'pft_tausnir', 'pft_tausvis', 'pft_woody', 'pft_xl', 'pft_z0mr'
    ]
    
    # Water variables (5 variables - H2OSOI_10CM excluded)
    water_variables = ['H2OSOI_1_', 'H2OSOI_2_', 'H2OSOI_3_', 'H2OSOI_4_', 'H2OSOI_5_', 'H2OSOI_6_']
    
    # Scalar variables (5 variables)
    scalar_variables = ['GPP', 'NPP', 'AR', 'HR', 'LAI']
    
    # 1D variables (16 variables - excluded variables removed)
    variables_1d = [
        'cwdp', 'deadcrootc', 'deadcrootn', 'deadcrootp', 'deadstemc', 'deadstemn', 'deadstemp',
        'frootc', 'frootc_storage', 'leafc', 'leafc_storage', 'totcolp', 'totlitc', 'totvegc'
    ]
    
    # 2D variables (67 variables - soil properties moved from surface properties, soil texture moved to surface properties)
    variables_2d = [
        # Litter Variables (16)
        'LITR1C_1C_vr', 'LITR1C_2C_vr', 'LITR1C_3C_vr', 'LITR1C_4C_vr',
        'LITR1N_1N_vr', 'LITR1N_2N_vr', 'LITR1N_3N_vr', 'LITR1N_4N_vr',
        'LITR1P_1P_vr', 'LITR1P_2P_vr', 'LITR1P_3P_vr', 'LITR1P_4P_vr',
        'LITR2C_1C_vr', 'LITR2C_2C_vr', 'LITR2C_3C_vr', 'LITR2C_4C_vr',
        
        # Soil Properties (68 variables)
        'SOILC_1C', 'SOILC_2C', 'SOILC_3C', 'SOILC_4C',
        'SOILN_1N', 'SOILN_2N', 'SOILN_3N', 'SOILN_4N',
        'SOILP_1P', 'SOILP_2P', 'SOILP_3P', 'SOILP_4P',
        'SOILC_1C_vr', 'SOILC_2C_vr', 'SOILC_3C_vr', 'SOILC_4C_vr',
        'SOILN_1N_vr', 'SOILN_2N_vr', 'SOILN_3N_vr', 'SOILN_4N_vr',
        'SOILP_1P_vr', 'SOILP_2P_vr', 'SOILP_3P_vr', 'SOILP_4P_vr',
        
        # Soil Carbon layers (40 variables)
        'SOILC_1C_1_', 'SOILC_1C_2_', 'SOILC_1C_3_', 'SOILC_1C_4_', 'SOILC_1C_5_', 'SOILC_1C_6_', 'SOILC_1C_7_', 'SOILC_1C_8_', 'SOILC_1C_9_', 'SOILC_1C_10_',
        'SOILC_2C_1_', 'SOILC_2C_2_', 'SOILC_2C_3_', 'SOILC_2C_4_', 'SOILC_2C_5_', 'SOILC_2C_6_', 'SOILC_2C_7_', 'SOILC_2C_8_', 'SOILC_2C_9_', 'SOILC_2C_10_',
        'SOILC_3C_1_', 'SOILC_3C_2_', 'SOILC_3C_3_', 'SOILC_3C_4_', 'SOILC_3C_5_', 'SOILC_3C_6_', 'SOILC_3C_7_', 'SOILC_3C_8_', 'SOILC_3C_9_', 'SOILC_3C_10_',
        'SOILC_4C_1_', 'SOILC_4C_2_', 'SOILC_4C_3_', 'SOILC_4C_4_', 'SOILC_4C_5_', 'SOILC_4C_6_', 'SOILC_4C_7_', 'SOILC_4C_8_', 'SOILC_4C_9_', 'SOILC_4C_10_',
        
        # Soil Nitrogen layers (40 variables)
        'SOILN_1N_1_', 'SOILN_1N_2_', 'SOILN_1N_3_', 'SOILN_1N_4_', 'SOILN_1N_5_', 'SOILN_1N_6_', 'SOILN_1N_7_', 'SOILN_1N_8_', 'SOILN_1N_9_', 'SOILN_1N_10_',
        'SOILN_2N_1_', 'SOILN_2N_2_', 'SOILN_2N_3_', 'SOILN_2N_4_', 'SOILN_2N_5_', 'SOILN_2N_6_', 'SOILN_2N_7_', 'SOILN_2N_8_', 'SOILN_2N_9_', 'SOILN_2N_10_',
        'SOILN_3N_1_', 'SOILN_3N_2_', 'SOILN_3N_3_', 'SOILN_3N_4_', 'SOILN_3N_5_', 'SOILN_3N_6_', 'SOILN_3N_7_', 'SOILN_3N_8_', 'SOILN_3N_9_', 'SOILN_3N_10_',
        'SOILN_4N_1_', 'SOILN_4N_2_', 'SOILN_4N_3_', 'SOILN_4N_4_', 'SOILN_4N_5_', 'SOILN_4N_6_', 'SOILN_4N_7_', 'SOILN_4N_8_', 'SOILN_4N_9_', 'SOILN_4N_10_',
        
        # Soil Phosphorus layers (40 variables)
        'SOILP_1P_1_', 'SOILP_1P_2_', 'SOILP_1P_3_', 'SOILP_1P_4_', 'SOILP_1P_5_', 'SOILP_1P_6_', 'SOILP_1P_7_', 'SOILP_1P_8_', 'SOILP_1P_9_', 'SOILP_1P_10_',
        'SOILP_2P_1_', 'SOILP_2P_2_', 'SOILP_2P_3_', 'SOILP_2P_4_', 'SOILP_2P_5_', 'SOILP_2P_6_', 'SOILP_2P_7_', 'SOILP_2P_8_', 'SOILP_2P_9_', 'SOILP_2P_10_',
        'SOILP_3P_1_', 'SOILP_3P_2_', 'SOILP_3P_3_', 'SOILP_3P_4_', 'SOILP_3P_5_', 'SOILP_3P_6_', 'SOILP_3P_7_', 'SOILP_3P_8_', 'SOILP_3P_9_', 'SOILP_3P_10_',
        'SOILP_4P_1_', 'SOILP_4P_2_', 'SOILP_4P_3_', 'SOILP_4P_4_', 'SOILP_4P_5_', 'SOILP_4P_6_', 'SOILP_4P_7_', 'SOILP_4P_8_', 'SOILP_4P_9_', 'SOILP_4P_10_'
    ]
    
    # Output variables
    output_water = ['Y_H2OSOI_1_', 'Y_H2OSOI_2_', 'Y_H2OSOI_3_', 'Y_H2OSOI_4_', 'Y_H2OSOI_5_', 'Y_H2OSOI_6_']
    output_temperature = ['Y_T_GRND_R', 'Y_T_GRND_U', 'Y_T_LAKE', 'Y_T_SOISNO', 'Y_T_GRND_1_', 'Y_T_GRND_2_', 'Y_T_GRND_3_']
    output_scalar = ['Y_GPP', 'Y_NPP', 'Y_AR', 'Y_HR', 'Y_LAI']
    output_1d = [
        'Y_TS_TOPO', 'Y_cwdp', 'Y_deadcrootc', 'Y_deadcrootn', 'Y_deadcrootp',
        'Y_deadstemc', 'Y_deadstemn', 'Y_deadstemp', 'Y_frootc', 'Y_frootc_storage',
        'Y_leafc', 'Y_leafc_storage', 'Y_totcolp', 'Y_totlitc', 'Y_totvegc'
    ]
    output_2d = [
        # Litter Variables (16)
        'Y_LITR1C_1C_vr', 'Y_LITR1C_2C_vr', 'Y_LITR1C_3C_vr', 'Y_LITR1C_4C_vr',
        'Y_LITR1N_1N_vr', 'Y_LITR1N_2N_vr', 'Y_LITR1N_3N_vr', 'Y_LITR1N_4N_vr',
        'Y_LITR1P_1P_vr', 'Y_LITR1P_2P_vr', 'Y_LITR1P_3P_vr', 'Y_LITR1P_4P_vr',
        'Y_LITR2C_1C_vr', 'Y_LITR2C_2C_vr', 'Y_LITR2C_3C_vr', 'Y_LITR2C_4C_vr',
        # Soil Properties (12)
        'Y_SOILC_1C_vr', 'Y_SOILC_2C_vr', 'Y_SOILC_3C_vr', 'Y_SOILC_4C_vr',
        'Y_SOILN_1N_vr', 'Y_SOILN_2N_vr', 'Y_SOILN_3N_vr', 'Y_SOILN_4N_vr',
        'Y_SOILP_1P_vr', 'Y_SOILP_2P_vr', 'Y_SOILP_3P_vr', 'Y_SOILP_4P_vr'
    ]
    
    # Configure input variables based on water inclusion
    if include_water:
        # Include water variables
        config.update_data_config(
            static_columns=surface_properties,
            x_list_columns_1d=water_variables + scalar_variables + variables_1d,
            x_list_columns_2d=variables_2d,
            y_list_columns_1d=output_water + output_scalar + output_1d,
            y_list_columns_2d=output_2d
        )
    else:
        # Exclude water variables
        config.update_data_config(
            static_columns=surface_properties,
            x_list_columns_1d=scalar_variables + variables_1d,
            x_list_columns_2d=variables_2d,
            y_list_columns_1d=output_scalar + output_1d,
            y_list_columns_2d=output_2d
        )
    
    # Model configuration for CNP architecture
    config.update_model_config(
        # LSTM for time series (6 variables, 20 years)
        lstm_hidden_size=128,
        
        # FC for surface properties (39 variables - including soil texture)
        static_fc_size=128,
        
        # FC for PFT parameters (44 variables) - separate from surface properties
        fc_hidden_size=128,
        
        # CNN for 2D variables (67 variables - soil texture moved to surface properties)
        conv_channels=[32, 64, 128, 256],
        conv_kernel_size=3,
        conv_padding=1,
        
        # Transformer parameters
        num_tokens=8,
        token_dim=128,
        transformer_layers=4,
        transformer_heads=8,
        
        # Output dimensions
        scalar_output_size=5,  # Y_GPP, Y_NPP, Y_AR, Y_HR, Y_LAI
        vector_output_size=16 if include_water else 15,  # 1D variables (water + others)
        vector_length=16,
        matrix_output_size=28,  # 2D variables
        matrix_rows=10,
        matrix_cols=10
    )
    
    # Training configuration
    config.update_training_config(
        num_epochs=100,
        batch_size=32,
        learning_rate=0.001,
        
        # Loss weights for different output types
        scalar_loss_weight=1.0,
        vector_loss_weight=1.0,
        matrix_loss_weight=1.0,
        
        # Optimizer
        optimizer_type='adamw',
        weight_decay=0.01,
        
        # Learning rate scheduler
        use_scheduler=True,
        scheduler_type='cosine',
        
        # Early stopping
        use_early_stopping=True,
        patience=15,
        min_delta=0.001,
        
        # Device and optimization
        device='auto',
        use_mixed_precision=True,
        use_amp=True,
        use_grad_scaler=True,
        
        # Logging
        save_model=True,
        model_save_path="cnp_model.pt",
        save_losses=True,
        losses_save_path="cnp_training_losses.csv",
        save_predictions=True,
        predictions_dir="cnp_predictions"
    )
    
    return config


def get_cnp_model_config_no_water() -> TrainingConfigManager:
    """
    Get CNP model configuration without water variables.
    
    Returns:
        TrainingConfigManager with CNP model configuration (no water)
    """
    return get_cnp_model_config(include_water=False) 