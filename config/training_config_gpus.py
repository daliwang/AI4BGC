"""
Multi-GPU/Distributed Training Configuration for AI4BGC
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch
import torch.nn as nn
import torch.optim as optim

@dataclass
class DataConfigGPUs:
    data_paths: List[str] = field(default_factory=lambda: [
        "/global/cfs/cdirs/m4814/daweigao/0_trendy_case/dataset",
        "/global/cfs/cdirs/m4814/daweigao/1_0.5_degree/dataset"
    ])
    file_pattern: str = "1_training_data_batch_*.pkl"
    columns_to_drop: List[str] = field(default_factory=lambda: [
        'Y_OCCLUDED_P', 'Y_SECONDARY_P', 'Y_LABILE_P', 'Y_APATITE_P'
    ])
    time_series_columns: List[str] = field(default_factory=lambda: [
        'FLDS', 'PSRF', 'FSDS', 'QBOT', 'PRECTmms', 'TBOT'
    ])
    static_columns: List[str] = field(default_factory=lambda: [
        'lat', 'lon', 'area', 'landfrac', 'PFT0', 'PFT1', 'PFT2', 'PFT3', 'PFT4', 'PFT5',
        'PFT6', 'PFT7', 'PFT8', 'PFT9', 'PFT10', 'PFT11', 'PFT12', 'PFT13', 'PFT14', 'PFT15'
    ])
    x_list_columns_2d: List[str] = field(default_factory=lambda: [
        'soil3c_vr', 'soil4c_vr', 'cwdc_vr'
    ])
    y_list_columns_2d: List[str] = field(default_factory=lambda: [
        'Y_soil3c_vr', 'Y_soil4c_vr', 'Y_cwdc_vr'
    ])
    x_list_columns_1d: List[str] = field(default_factory=lambda: [
        'deadcrootc', 'deadstemc', 'tlai'
    ])
    y_list_columns_1d: List[str] = field(default_factory=lambda: [
        'Y_deadcrootc', 'Y_deadstemc', 'Y_tlai'
    ])
    time_series_length: int = 240
    max_time_series_length: int = 1476
    max_1d_length: int = 16
    max_2d_rows: int = 18
    max_2d_cols: int = 10
    train_split: float = 0.8
    random_state: int = 42
    filter_column: Optional[str] = 'H2OSOI_10CM'
    max_files: Optional[int] = None

@dataclass
class ModelConfigGPUs:
    lstm_hidden_size: int = 512
    fc_hidden_size: int = 256
    static_fc_size: int = 256
    conv_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    conv_kernel_size: int = 3
    conv_padding: int = 1
    num_tokens: int = 8
    token_dim: int = 256
    transformer_layers: int = 8
    transformer_heads: int = 16
    scalar_output_size: int = 5
    vector_output_size: int = 3
    vector_length: int = 16
    matrix_output_size: int = 3
    matrix_rows: int = 18
    matrix_cols: int = 10

@dataclass
class TrainingConfigGPUs:
    num_epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 0.0005
    scalar_loss_weight: float = 1.0
    vector_loss_weight: float = 1.0
    matrix_loss_weight: float = 1.0
    optimizer_type: str = 'adamw'
    weight_decay: float = 1e-4
    use_scheduler: bool = False
    scheduler_type: str = 'step'
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    use_early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001
    device: str = 'cuda'
    use_mixed_precision: bool = True
    use_amp: bool = True
    use_grad_scaler: bool = True
    pin_memory: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    empty_cache_freq: int = 10
    max_memory_usage: float = 0.9
    memory_efficient_attention: bool = True
    log_gpu_memory: bool = False
    log_gpu_utilization: bool = False
    gpu_monitor_interval: int = 100
    save_model: bool = True
    model_save_path: str = "LSTM_model.pt"
    save_losses: bool = True
    losses_save_path: str = "training_validation_losses.csv"
    save_predictions: bool = True
    predictions_dir: str = "predictions"
    validation_frequency: int = 1
    random_seed: int = 42
    deterministic: bool = True
    # Multi-GPU/Distributed
    use_multi_gpu: bool = True
    distributed_backend: str = 'nccl'
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25
    static_graph: bool = True
    distributed_sampler: bool = True
    shuffle_seed: int = 42
    allreduce_bucket_size: int = 10
    fp16_allreduce: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    sync_batch_norm: bool = True
    sync_batch_norm_momentum: float = 0.1

    def get_device(self) -> torch.device:
        if self.device == 'auto':
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(self.device)
    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        if self.optimizer_type.lower() == 'adam':
            return optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type.lower() == 'sgd':
            return optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type.lower() == 'adamw':
            return optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
    def get_scheduler(self, optimizer: optim.Optimizer):
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

def get_multi_gpu_config():
    """Return a config tuple for multi-GPU training."""
    return DataConfigGPUs(), ModelConfigGPUs(), TrainingConfigGPUs() 