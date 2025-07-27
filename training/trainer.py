"""
Flexible trainer module for model training.

This module provides a flexible training framework that can handle
different model configurations, data types, and training strategies.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from sklearn.metrics import mean_squared_error
from pathlib import Path
import time
from tqdm import tqdm
import warnings
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# from config.training_config import TrainingConfig  # Uncomment if TrainingConfig is defined
from models.combined_model import CombinedModel, FlexibleCombinedModel
from models.cnp_combined_model import CNPCombinedModel

# Import GPU monitoring
from utils.gpu_monitor import GPUMonitor, log_memory_usage

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Flexible trainer for climate model training.
    
    This class handles the complete training pipeline including
    data preparation, model training, validation, and saving results.
    """
    
    def __init__(self, training_config: Any, model: nn.Module, 
                 train_data: Dict[str, Any], test_data: Dict[str, Any],
                 scalers: Dict[str, Any], data_info: Dict[str, Any]):
        """
        Initialize the trainer.
        
        Args:
            training_config: Training configuration
            model: Model to train
            train_data: Training data
            test_data: Test data
            scalers: Data scalers for inverse transformation
            data_info: Information about the data structure
        """
        self.config = training_config
        self.device = self.config.get_device()
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.scalers = scalers
        self.data_info = data_info

        # --- ENFORCE COLUMN ORDER CONSISTENCY ---
        # These lists should be used everywhere for model input/output order
        self.scalar_columns = self.data_info.get('x_list_scalar_columns', [])
        self.y_scalar_columns = self.data_info.get('y_list_scalar_columns', [])
        self.variables_1d_pft_columns = self.data_info.get('variables_1d_pft', [])  # Canonical 1D PFT variable list
        self.y_pft_1d_columns = self.data_info.get('y_list_columns_1d', [])
        self.variables_2d_soil_columns = self.data_info.get('x_list_columns_2d', [])
        self.y_soil_2d_columns = self.data_info.get('y_list_columns_2d', [])
        self.pft_param_columns = self.data_info.get('pft_param_columns', [])
        
        # Set random seeds for reproducibility
        if hasattr(self.config, 'random_seed'):
            torch.manual_seed(self.config.random_seed)
            torch.cuda.manual_seed(self.config.random_seed)
            torch.cuda.manual_seed_all(self.config.random_seed)
            np.random.seed(self.config.random_seed)
            logger.info(f"Random seed set to {self.config.random_seed}")
        
        # Set deterministic behavior if requested
        if hasattr(self.config, 'deterministic') and self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("Deterministic behavior enabled")
        
        # Ensure all arrays in train/test splits have the same number of samples
        final_tensor_keys = [
            'time_series', 'static', 'pft_param', 'scalar',
            'variables_1d_pft', 'variables_2d_soil',
            'y_scalar', 'y_pft_1d', 'y_soil_2d',
            'water', 'y_water'
        ]

        # --- DEBUG: Force small batch size and minimal DataLoader workers for OOM debugging ---
        # print the shape of the tensors in the train and test data
        # print(f"[DEBUG] train_data['time_series'].shape: {self.train_data['time_series'].shape}")
        # print(f"[DEBUG] train_data['static'].shape: {self.train_data['static'].shape}")
        # print(f"[DEBUG] train_data['pft_param'].shape: {self.train_data['pft_param'].shape}")
        # print(f"[DEBUG] train_data['scalar'].shape: {self.train_data['scalar'].shape}")
        # print(f"[DEBUG] train_data['variables_1d_pft'].shape: {self.train_data['variables_1d_pft'].shape}")
        # print(f"[DEBUG] train_data['variables_2d_soil'].shape: {self.train_data['variables_2d_soil'].shape}")

        # print(f"[DEBUG] test_data['time_series'].shape: {self.test_data['time_series'].shape}")
        # print(f"[DEBUG] test_data['static'].shape: {self.test_data['static'].shape}")
        # print(f"[DEBUG] test_data['pft_param'].shape: {self.test_data['pft_param'].shape}")
        # print(f"[DEBUG] test_data['scalar'].shape: {self.test_data['scalar'].shape}")
        # print(f"[DEBUG] test_data['variables_1d_pft'].shape: {self.test_data['variables_1d_pft'].shape}")
        # print(f"[DEBUG] test_data['variables_2d_soil'].shape: {self.test_data['variables_2d_soil'].shape}")

        # --- DEBUG: Force small batch size and minimal DataLoader workers for OOM debugging ---

        for split_name in ['train', 'test']:
            split = getattr(self, f'{split_name}_data')
            keys_present = [k for k in final_tensor_keys if k in split and isinstance(split[k], torch.Tensor)]
            if not keys_present:
                continue
            min_length = min(split[k].shape[0] for k in keys_present)
            for k in keys_present:
                if split[k].shape[0] != min_length:
                    split[k] = split[k][:min_length]

        # Move all final tensors to device (for both train and test splits)
        for split_name in ['train', 'test']:
            split = getattr(self, f'{split_name}_data')
            for key in final_tensor_keys:
                if key in split:
                    split[key] = split[key].to(self.device)

        # Setup device for model
        self.model.to(self.device)
        
        # Initialize GPU monitoring
        self.gpu_monitor = GPUMonitor(self.device)
        
        # Setup mixed precision training
        self.use_amp = self.config.use_amp and self.device.type == "cuda"
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            logger.info("Automatic Mixed Precision (AMP) enabled")
        else:
            self.scaler = None
            logger.info("Mixed precision disabled for fair comparison")
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Setup loss function
        self.criterion = nn.MSELoss()
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Ensure all tensors have the same batch size for TensorDataset
        self._ensure_consistent_batch_sizes()
        
        # Log initial GPU stats
        if self.config.log_gpu_memory:
            self.gpu_monitor.log_gpu_stats("Initial ")
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
        # Use learnable loss weights if specified in config
        self.use_learnable_loss_weights = getattr(self.config, 'use_learnable_loss_weights', False)

    def _ensure_consistent_batch_sizes(self):
        """Ensure all final tensors in train_data and test_data have the same batch size."""
        final_tensor_keys = [
            'time_series', 'static', 'pft_param', 'scalar',
            'variables_1d_pft', 'variables_2d_soil',
            'y_scalar', 'y_pft_1d', 'y_soil_2d',
            'water', 'y_water'
        ]
        for split_name in ['train', 'test']:
            split = getattr(self, f'{split_name}_data')
            keys_present = [k for k in final_tensor_keys if k in split and isinstance(split[k], torch.Tensor)]
            if not keys_present:
                continue
            min_length = min(split[k].shape[0] for k in keys_present)
            for k in keys_present:
                if split[k].shape[0] != min_length:
                    split[k] = split[k][:min_length]
    
    def _concat_list_columns(self, list_dict, col_names):
        """Concatenate 1D list columns into a tensor."""
        tensors = [list_dict[col] for col in col_names if col in list_dict]
        if tensors:
            return torch.cat(tensors, dim=1)
        else:
            # Return empty tensor with correct batch size from other data sources
            # Use the batch size from time_series data if available
            if hasattr(self, 'train_data') and 'time_series' in self.train_data:
                batch_size = self.train_data['time_series'].shape[0]
            elif hasattr(self, 'test_data') and 'time_series' in self.test_data:
                batch_size = self.test_data['time_series'].shape[0]
            else:
                batch_size = 0
            return torch.empty((batch_size, 0), device=self.device)

    def _concat_list_columns_2d(self, list_dict, col_names):
        """Concatenate 2D list columns into a tensor."""
        tensors = [list_dict[col].unsqueeze(1) for col in col_names if col in list_dict]
        if tensors:
            return torch.cat(tensors, dim=1)
        else:
            # Return empty tensor with correct batch size from other data sources
            # Use the batch size from time_series data if available
            if hasattr(self, 'train_data') and 'time_series' in self.train_data:
                batch_size = self.train_data['time_series'].shape[0]
            elif hasattr(self, 'test_data') and 'time_series' in self.test_data:
                batch_size = self.test_data['time_series'].shape[0]
            else:
                batch_size = 0
            return torch.empty((batch_size, 0, 0, 0), device=self.device)
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Debug: Check tensor sizes before creating TensorDataset
        tensors_to_check = [
            self.train_data['time_series'],
            self.train_data['static'],
            self.train_data['pft_param'],
            self.train_data['scalar'],
            self.train_data['variables_1d_pft'],
            self.train_data['variables_2d_soil'],
            self.train_data['y_scalar'],
            self.train_data['y_pft_1d'],
            self.train_data['y_soil_2d']
        ]
        
        tensor_names = ['time_series', 'static', 'pft_param', 'scalar', 'variables_1d_pft', 'variables_2d_soil', 'y_scalar', 'y_pft_1d', 'y_soil_2d']
        batch_sizes = [t.shape[0] for t in tensors_to_check]
        
        logger.info(f"Training tensor batch sizes: {dict(zip(tensor_names, batch_sizes))}")
        
        if len(set(batch_sizes)) > 1:
            logger.error(f"Tensor batch sizes are inconsistent: {dict(zip(tensor_names, batch_sizes))}")
            raise ValueError(f"Tensor batch sizes must be consistent. Found: {dict(zip(tensor_names, batch_sizes))}")
        
        # Add water to tensors_to_check and tensor_names if present
        if 'water' in self.train_data:
            tensors_to_check.append(self.train_data['water'])
            tensor_names.append('water')
        if 'y_water' in self.train_data:
            tensors_to_check.append(self.train_data['y_water'])
            tensor_names.append('y_water')
        
        # Create data loader with GPU optimizations
        if 'water' in self.train_data and 'y_water' in self.train_data:
            train_dataset = TensorDataset(
                self.train_data['time_series'],
                self.train_data['static'],
                self.train_data['pft_param'],
                self.train_data['scalar'],
                self.train_data['variables_1d_pft'],
                self.train_data['variables_2d_soil'],
                self.train_data['y_pft_1d'],
                self.train_data['y_scalar'],
                self.train_data['y_soil_2d'],
                self.train_data['water'],
                self.train_data['y_water']
            )
        else:
            train_dataset = TensorDataset(
                self.train_data['time_series'],
                self.train_data['static'],
                self.train_data['pft_param'],
                self.train_data['scalar'],
                self.train_data['variables_1d_pft'],
                self.train_data['variables_2d_soil'],
                self.train_data['y_scalar'],
                self.train_data['y_pft_1d'],
                self.train_data['y_soil_2d']
            )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=self.config.pin_memory and self.device.type == 'cpu',  # Only pin memory for CPU tensors
            num_workers=self.config.num_workers,
            prefetch_factor=(2 if self.config.num_workers > 0 else None),
            persistent_workers=False
        )
        progress_bar = tqdm(train_loader, desc="Training")
        
        def get_loss_value(loss):
            return loss.item() if hasattr(loss, 'item') else loss

        for batch_idx, batch in enumerate(progress_bar):
            if 'water' in self.train_data and 'y_water' in self.train_data:
                (time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil, y_scalar, y_pft_1d, y_soil_2d, water, y_water) = batch
            else:
                (time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil, y_scalar, y_pft_1d, y_soil_2d) = batch
            # --- DEBUG: Print tensor shapes and device before model call ---
            # print(f"[DEBUG] Batch {batch_idx} tensor shapes and device:")
            # print(f"  time_series: {time_series.shape}, device: {time_series.device}")
            # print(f"  static: {static.shape}, device: {static.device}")
            # print(f"  pft_param: {pft_param.shape}, device: {pft_param.device}")
            # print(f"  scalar: {scalar.shape}, device: {scalar.device}")
            # print(f"  variables_1d_pft: {variables_1d_pft.shape}, device: {variables_1d_pft.device}")
            # print(f"  variables_2d_soil: {variables_2d_soil.shape}, device: {variables_2d_soil.device}")
            # print(f"  y_scalar: {y_scalar.shape}, device: {y_scalar.device}")
            # print(f"  y_pft_1d: {y_pft_1d.shape}, device: {y_pft_1d.device}")
            # print(f"  y_soil_2d: {y_soil_2d.shape}, device: {y_soil_2d.device}")
            # if 'water' in self.train_data and 'y_water' in self.train_data:
            #     print(f"  water: {water.shape}, device: {water.device}")
            #     print(f"  y_water: {y_water.shape}, device: {y_water.device}")
            # Move data to device and ensure contiguous
            time_series = time_series.to(self.device, non_blocking=True).contiguous()
            static = static.to(self.device, non_blocking=True).contiguous()
            variables_1d_pft = variables_1d_pft.to(self.device, non_blocking=True).contiguous()
            variables_2d_soil = variables_2d_soil.to(self.device, non_blocking=True).contiguous()
            pft_param = pft_param.to(self.device, non_blocking=True).contiguous()
            scalar = scalar.to(self.device, non_blocking=True).contiguous()
            y_scalar = y_scalar.to(self.device, non_blocking=True).contiguous()   
            y_pft_1d = y_pft_1d.to(self.device, non_blocking=True).contiguous()                     
            y_soil_2d = y_soil_2d.to(self.device, non_blocking=True).contiguous()
            if 'water' in self.train_data and 'y_water' in self.train_data:
                water = water.to(self.device, non_blocking=True).contiguous()
                y_water = y_water.to(self.device, non_blocking=True).contiguous()

            # print(f"[DEBUG] variables_1d_pft shape before model: {variables_1d_pft.shape}")
            # if variables_1d_pft.dim() == 2 and variables_1d_pft.shape[1] == 224:
            #     variables_1d_pft = variables_1d_pft.view(-1, 14, 16)
            #     print(f"[DEBUG] variables_1d_pft reshaped to: {variables_1d_pft.shape}")

            self.optimizer.zero_grad()
            
            # Forward pass (with or without mixed precision)
            if self.use_amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    if 'water' in self.train_data and 'y_water' in self.train_data:
                        outputs = self.model(time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil, water)
                    else:
                        outputs = self.model(time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil)
            else:
                if 'water' in self.train_data and 'y_water' in self.train_data:
                    outputs = self.model(time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil, water)
                else:
                    outputs = self.model(time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil)

            # Compute loss
            loss = self._compute_loss(outputs['scalar'], y_scalar)
            #loss += self._compute_loss(outputs['pft_1d'], y_pft_1d)
            loss += self._compute_loss(outputs['pft_1d'], y_pft_1d.view(y_pft_1d.size(0), -1))
            #loss += self._compute_loss(outputs['soil_2d'], y_soil_2d)
            loss += self._compute_loss(outputs['soil_2d'].view(y_soil_2d.size(0), -1), y_soil_2d.view(y_soil_2d.size(0), -1))
            if 'water' in self.train_data and 'y_water' in self.train_data and 'water' in outputs:
                loss += self._compute_loss(outputs['water'], y_water)

            # Backward and optimizer step
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            loss_value = get_loss_value(loss)
            total_loss += loss_value
            num_batches += 1
            progress_bar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'avg_loss': f'{(total_loss/num_batches):.4f}'
            })
            
            # GPU memory management
            if batch_idx % self.config.empty_cache_freq == 0:
                self.gpu_monitor.empty_cache()
            
            # GPU monitoring
            if (batch_idx % self.config.gpu_monitor_interval == 0 and 
                self.config.log_gpu_memory):
                self.gpu_monitor.log_gpu_stats(f"Batch {batch_idx} ")
            
            # Check memory threshold
            if self.gpu_monitor.check_memory_threshold(self.config.max_memory_usage):
                logger.warning(f"GPU memory usage exceeds {self.config.max_memory_usage*100}%")
                self.gpu_monitor.empty_cache()
            # --- DEBUG: Call torch.cuda.empty_cache() after each batch ---
            torch.cuda.empty_cache()
        
        return total_loss / num_batches
    
    def validate_epoch(self) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Debug: Check tensor sizes before creating TensorDataset
        tensors_to_check = [
            self.test_data['time_series'],
            self.test_data['static'],
            self.test_data['variables_1d_pft'],
            self.test_data['variables_2d_soil'],
            self.test_data['y_soil_2d'],
            self.test_data['pft_param'],
            self.test_data['scalar'],
            self.test_data['y_pft_1d'],
            self.test_data['y_scalar']
        ]
        
        tensor_names = ['time_series', 'static', 'variables_1d_pft', 'variables_2d_soil', 'y_soil_2d', 'pft_param', 'scalar', 'y_pft_1d', 'y_scalar']
        batch_sizes = [t.shape[0] for t in tensors_to_check]
        
        logger.info(f"Validation tensor batch sizes: {dict(zip(tensor_names, batch_sizes))}")
        
        # Check if we have any validation data
        if all(size == 0 for size in batch_sizes):
            logger.warning("No validation data available. Skipping validation.")
            return float('inf')  # Return infinity to indicate no validation
        
        if len(set(batch_sizes)) > 1:
            logger.error(f"Validation tensor batch sizes are inconsistent: {dict(zip(tensor_names, batch_sizes))}")
            raise ValueError(f"Validation tensor batch sizes must be consistent. Found: {dict(zip(tensor_names, batch_sizes))}")
        
        # Add water to tensors_to_check and tensor_names if present
        if 'water' in self.test_data:
            tensors_to_check.append(self.test_data['water'])
            tensor_names.append('water')
        if 'y_water' in self.test_data:
            tensors_to_check.append(self.test_data['y_water'])
            tensor_names.append('y_water')
        
        # Create data loader with GPU optimizations
        if 'water' in self.test_data and 'y_water' in self.test_data:
            val_dataset = TensorDataset(
                self.test_data['time_series'],
                self.test_data['static'],
                self.test_data['pft_param'],
                self.test_data['scalar'],
                self.test_data['variables_1d_pft'],
                self.test_data['variables_2d_soil'],
                self.test_data['y_scalar'],
                self.test_data['y_pft_1d'],
                self.test_data['y_soil_2d'],
                self.test_data['water'],
                self.test_data['y_water']
            )
        else:
            val_dataset = TensorDataset(
                self.test_data['time_series'],
                self.test_data['static'],
                self.test_data['pft_param'],
                self.test_data['scalar'],
                self.test_data['variables_1d_pft'],
                self.test_data['variables_2d_soil'],
                self.test_data['y_scalar'],
                self.test_data['y_pft_1d'],
                self.test_data['y_soil_2d']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=self.config.pin_memory and self.device.type == 'cpu',  # Only pin memory for CPU tensors
            num_workers=self.config.num_workers,
            prefetch_factor=(self.config.prefetch_factor if self.config.num_workers > 0 else None),
            persistent_workers=self.config.persistent_workers
        )
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            def get_loss_value(loss):
                return loss.item() if hasattr(loss, 'item') else loss
            for batch_idx, batch in enumerate(progress_bar):
                if 'water' in self.test_data and 'y_water' in self.test_data:
                    (time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil, y_scalar, y_pft_1d, y_soil_2d, water, y_water) = batch
                else:
                    (time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil, y_scalar, y_pft_1d, y_soil_2d) = batch
                # Move data to device and ensure contiguous
                time_series = time_series.to(self.device, non_blocking=True).contiguous()
                static = static.to(self.device, non_blocking=True).contiguous()
                variables_1d_pft = variables_1d_pft.to(self.device, non_blocking=True).contiguous()
                variables_2d_soil = variables_2d_soil.to(self.device, non_blocking=True).contiguous()
                pft_param = pft_param.to(self.device, non_blocking=True).contiguous()
                scalar = scalar.to(self.device, non_blocking=True).contiguous()
                y_scalar = y_scalar.to(self.device, non_blocking=True).contiguous()
                y_pft_1d = y_pft_1d.to(self.device, non_blocking=True).contiguous()
                y_soil_2d = y_soil_2d.to(self.device, non_blocking=True).contiguous()
                if 'water' in self.test_data and 'y_water' in self.test_data:
                    water = water.to(self.device, non_blocking=True).contiguous()
                    y_water = y_water.to(self.device, non_blocking=True).contiguous()

                # print(f"[DEBUG] variables_1d_pft shape before model (val): {variables_1d_pft.shape}")
                # if variables_1d_pft.dim() == 2 and variables_1d_pft.shape[1] == 224:
                #     variables_1d_pft = variables_1d_pft.view(-1, 14, 16)
                #     print(f"[DEBUG] variables_1d_pft reshaped to: {variables_1d_pft.shape}")

                # Forward pass (with or without mixed precision)
                if self.use_amp and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        if 'water' in self.test_data and 'y_water' in self.test_data:
                            outputs = self.model(time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil, water)
                        else:
                            outputs = self.model(time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil)
                else:
                    if 'water' in self.test_data and 'y_water' in self.test_data:
                        outputs = self.model(time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil, water)
                    else:
                        outputs = self.model(time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil)

                # Compute loss
                loss = self._compute_loss(outputs['scalar'], y_scalar)
                #loss += self._compute_loss(outputs['pft_1d'], y_pft_1d)
                loss += self._compute_loss(outputs['pft_1d'].view(y_pft_1d.size(0), -1), y_pft_1d.view(y_pft_1d.size(0), -1))
                #loss += self._compute_loss(outputs['soil_2d'], y_soil_2d)
                loss += self._compute_loss(outputs['soil_2d'].view(y_soil_2d.size(0), -1), y_soil_2d.view(y_soil_2d.size(0), -1))
                if 'water' in self.test_data and 'y_water' in self.test_data and 'water' in outputs:
                    loss += self._compute_loss(outputs['water'], y_water)
                
                loss_value = get_loss_value(loss)
                total_loss += loss_value
                num_batches += 1
                progress_bar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'avg_loss': f'{(total_loss/num_batches):.4f}'
                })
        
        # Handle the case where no batches were processed
        if num_batches == 0:
            logger.warning("No validation batches processed. Returning infinity.")
            return float('inf')
        
        return total_loss / num_batches
    
    def _prepare_batch_data(self, data: Dict[str, Any], start_idx: int, end_idx: int) -> Dict[str, Any]:
        """Prepare batch data for training/validation using only final model-ready tensors, in config order."""
        batch_data = {}
        batch_size = end_idx - start_idx
        device = data['time_series'].device
        
        # Required inputs (order enforced by config)
        batch_data['time_series'] = data['time_series'][start_idx:end_idx]
        batch_data['static'] = data['static'][start_idx:end_idx]
        batch_data['pft_param'] = data['pft_param'][start_idx:end_idx]
        batch_data['scalar'] = data['scalar'][start_idx:end_idx]
        batch_data['variables_1d_pft'] = data['variables_1d_pft'][start_idx:end_idx]
        batch_data['variables_2d_soil'] = data['variables_2d_soil'][start_idx:end_idx]
        batch_data['y_scalar'] = data['y_scalar'][start_idx:end_idx]
        batch_data['y_pft_1d'] = data['y_pft_1d'][start_idx:end_idx]
        batch_data['y_soil_2d'] = data['y_soil_2d'][start_idx:end_idx]

        # Optional water variables
        if 'water' in data:
            batch_data['water'] = data['water'][start_idx:end_idx]
        if 'y_water' in data:
            batch_data['y_water'] = data['y_water'][start_idx:end_idx]

        # Assert batch sizes for all present keys
        for k, v in batch_data.items():
            assert v.shape[0] == batch_size, f"{k} batch size mismatch: {v.shape[0]} vs {batch_size}"

        # Assert feature order for key tensors (optional, for debugging)
        # Example: assert batch_data['scalar'].shape[1] == len(self.scalar_columns)
        # Example: assert batch_data['variables_1d_pft'].shape[1] == len(self.variables_1d_pft_columns)
        
        return batch_data
    
    def _compute_loss(self, scalar_pred, target, **kwargs):
        """Compute only scalar loss for quick test or full loss with learnable weights if enabled."""
        # If using CNPCombinedModel and learnable loss weights, use log_sigma weighting
        if isinstance(self.model, CNPCombinedModel) and self.use_learnable_loss_weights:
            # Assume outputs and targets are dicts with keys: scalar, matrix, water, pft_1d
            outputs = kwargs.get('outputs', None)
            targets = kwargs.get('targets', None)
            if outputs is None or targets is None:
                # Fallback to scalar only
                return self.criterion(scalar_pred, target)
            loss = 0.0
            # Scalar
            if 'scalar' in outputs and 'scalar' in targets and self.model.log_sigma_scalar is not None:
                loss += (torch.exp(-2 * self.model.log_sigma_scalar) * self.criterion(outputs['scalar'], targets['scalar']) + self.model.log_sigma_scalar)
            # Matrix
            if 'matrix' in outputs and 'matrix' in targets and self.model.log_sigma_matrix is not None:
                loss += (torch.exp(-2 * self.model.log_sigma_matrix) * self.criterion(outputs['matrix'], targets['matrix']) + self.model.log_sigma_matrix)
            # Water
            if hasattr(self.model, 'log_sigma_water') and self.model.log_sigma_water is not None and 'water' in outputs and 'water' in targets:
                loss += (torch.exp(-2 * self.model.log_sigma_water) * self.criterion(outputs['water'], targets['water']) + self.model.log_sigma_water)
            # PFT 1D
            if hasattr(self.model, 'log_sigma_pft_1d') and self.model.log_sigma_pft_1d is not None and 'pft_1d' in outputs and 'pft_1d' in targets:
                loss += (torch.exp(-2 * self.model.log_sigma_pft_1d) * self.criterion(outputs['pft_1d'], targets['pft_1d']) + self.model.log_sigma_pft_1d)
            return loss
        else:
            return self.criterion(scalar_pred, target)
    
    def train(self) -> Dict[str, List[float]]:
        """
        Complete training loop.
        
        Returns:
            Dictionary containing training and validation losses
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs...")
        logger.info("=" * 60)
        
        # Print training summary
        print(f"\n{'='*20} TRAINING SUMMARY {'='*20}")
        print(f"📋 Configuration:")
        print(f"   • Total epochs: {self.config.num_epochs}")
        print(f"   • Batch size: {self.config.batch_size}")
        print(f"   • Learning rate: {self.config.learning_rate}")
        print(f"   • Device: {self.device}")
        print(f"   • Mixed precision: {'Enabled' if self.use_amp else 'Disabled'}")
        
        # Calculate approximate data info
        train_samples = self.train_data['time_series'].shape[0]
        test_samples = self.test_data['time_series'].shape[0]
        total_batches_per_epoch = (train_samples + self.config.batch_size - 1) // self.config.batch_size
        
        print(f"📊 Data Info:")
        print(f"   • Training samples: {train_samples:,}")
        print(f"   • Test samples: {test_samples:,}")
        print(f"   • Batches per epoch: ~{total_batches_per_epoch}")
        print(f"   • Total training batches: ~{total_batches_per_epoch * self.config.num_epochs:,}")
        
        print(f"🎯 Training Goals:")
        print(f"   • Target: Minimize combined loss (scalar + matrix)")
        print(f"   • Early stopping: {'Enabled' if self.config.use_early_stopping else 'Disabled'}")
        if self.config.use_early_stopping:
            print(f"   • Patience: {self.config.patience} epochs")
        print(f"   • Validation frequency: Every {self.config.validation_frequency} epoch(s)")
        
        print(f"{'='*60}")
        
        for epoch in range(self.config.num_epochs):
            # Check for NaNs/Infs in training data at the start of the epoch
            def count_nans_infs(name, tensor):
                n_nan = torch.isnan(tensor).sum().item() if torch.is_tensor(tensor) else 0
                n_inf = torch.isinf(tensor).sum().item() if torch.is_tensor(tensor) else 0
                print(f"Epoch {epoch+1}: {name} - NaNs: {n_nan}, Infs: {n_inf}")
            count_nans_infs('time_series', self.train_data['time_series'])
            count_nans_infs('static', self.train_data['static'])
            if 'list_1d' in self.train_data:
                for k, v in self.train_data['list_1d'].items():
                    count_nans_infs(f'list_1d[{k}]', v)
            if 'list_2d' in self.train_data:
                for k, v in self.train_data['list_2d'].items():
                    count_nans_infs(f'list_2d[{k}]', v)
            if 'list_scalar' in self.train_data:
                count_nans_infs('list_scalar', self.train_data['list_scalar'])
            if 'pft_param' in self.train_data:
                count_nans_infs('pft_param', self.train_data['pft_param'])
            if 'y_pft_1d' in self.train_data:
                count_nans_infs('y_pft_1d', self.train_data['y_pft_1d'])
            if 'y_scalar' in self.train_data:
                count_nans_infs('y_scalar', self.train_data['y_scalar'])
            # Print epoch header
            # print(f"\n{'='*20} EPOCH {epoch+1}/{self.config.num_epochs} {'='*20}")
            logger.info(f"Starting Epoch {epoch+1}/{self.config.num_epochs}")
            
            # Training
            # print(f"Training...")
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation
            if epoch % self.config.validation_frequency == 0:
                # print(f"Validating...")
                val_loss = self.validate_epoch()
                self.val_losses.append(val_loss)
                
                # Enhanced progress logging
                if val_loss == float('inf'):
                    progress_msg = f"Epoch [{epoch+1}/{self.config.num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: N/A (no validation data)"
                    print(f"✓ {progress_msg}")
                    logger.info(progress_msg)
                else:
                    progress_msg = f"Epoch [{epoch+1}/{self.config.num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                    print(f"✓ {progress_msg}")
                    logger.info(progress_msg)
                    
                    # Show improvement indicator
                    if val_loss < self.best_val_loss:
                        improvement = self.best_val_loss - val_loss
                        print(f"🎉 New best validation loss! Improved by {improvement:.4f}")
                        logger.info(f"New best validation loss! Improved by {improvement:.4f}")
                
                # Log loss weights
                loss_weights = self.model.get_loss_weights()
                log_msg = f"Loss weights - "
                if 'scalar' in loss_weights:
                    log_msg += f"Scalar: {loss_weights['scalar']:.4f}, "
                log_msg += f"Matrix: {loss_weights['matrix']:.4f}"
                logger.info(log_msg)
                
                # Show progress percentage
                progress_pct = ((epoch + 1) / self.config.num_epochs) * 100
                print(f"📊 Progress: {progress_pct:.1f}% complete ({epoch+1}/{self.config.num_epochs} epochs)")
                
                # Early stopping (only if we have validation data)
                if self.config.use_early_stopping and val_loss != float('inf'):
                    if self._check_early_stopping(val_loss):
                        logger.info("Early stopping triggered")
                        break
                
                # Learning rate scheduling (only if we have validation data)
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        if val_loss != float('inf'):
                            old_lr = self.optimizer.param_groups[0]['lr']
                            self.scheduler.step(val_loss)
                            new_lr = self.optimizer.param_groups[0]['lr']
                            if new_lr != old_lr:
                                print(f"📉 Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
                                logger.info(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
                    else:
                        self.scheduler.step()
            else:
                # For epochs without validation, still show training progress
                progress_msg = f"Epoch [{epoch+1}/{self.config.num_epochs}] - Train Loss: {train_loss:.4f}"
                print(f"✓ {progress_msg}")
                logger.info(progress_msg)
                
                # Show progress percentage
                progress_pct = ((epoch + 1) / self.config.num_epochs) * 100
                print(f"📊 Progress: {progress_pct:.1f}% complete ({epoch+1}/{self.config.num_epochs} epochs)")
        
        print(f"\n{'='*20} TRAINING COMPLETED {'='*20}")
        logger.info("Training completed")
        
        # Print final summary
        print(f" Training completed successfully!")
        print(f"📈 Final Results:")
        print(f"   • Total epochs completed: {len(self.train_losses)}")
        print(f"   • Final training loss: {self.train_losses[-1]:.4f}")
        
        if self.val_losses:
            print(f"   • Final validation loss: {self.val_losses[-1]:.4f}")
            print(f"   • Best validation loss: {min(self.val_losses):.4f}")
            
            # Show improvement
            if len(self.train_losses) > 1:
                train_improvement = self.train_losses[0] - self.train_losses[-1]
                print(f"   • Training loss improvement: {train_improvement:.4f}")
            
            if len(self.val_losses) > 1:
                val_improvement = self.val_losses[0] - min(self.val_losses)
                print(f"   • Validation loss improvement: {val_improvement:.4f}")
        
        print(f"💾 Results saved to:")
        print(f"   • Training log: training.log")
        print(f"   • Loss curves: training_validation_losses.csv")
        print(f"   • Model predictions: predictions/")
        
        return {'train_losses': self.train_losses, 'val_losses': self.val_losses}
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping should be triggered."""
        if val_loss < self.best_val_loss - self.config.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.patience
    
    def evaluate(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Evaluate the model and return predictions and metrics."""
        self.model.eval()
        
        # Check if we have any test data
        test_batch_sizes = [
            self.test_data['time_series'].shape[0],
            self.test_data['static'].shape[0],
            self.test_data['pft_param'].shape[0],
            self.test_data['scalar'].shape[0],
            self.test_data['variables_1d_pft'].shape[0],
            self.test_data['variables_2d_soil'].shape[0],
            self.test_data['y_scalar'].shape[0],
            self.test_data['y_pft_1d'].shape[0],
            self.test_data['y_soil_2d'].shape[0]
        ]
        if all(size == 0 for size in test_batch_sizes):
            logger.warning("No test data available. Skipping evaluation.")
            empty_predictions = {
                'scalar': torch.empty(0, 0),
                'pft_1d': torch.empty(0, 0),
                'soil_2d': torch.empty(0, 0, 0, 0)
            }
            default_metrics = {
                'scalar_rmse': 0.0,
                'scalar_mse': 0.0,
                'pft_1d_rmse': 0.0,
                'pft_1d_mse': 0.0,
                'soil_2d_rmse': 0.0,
                'soil_2d_mse': 0.0
            }
            return empty_predictions, default_metrics
        
        # Create evaluation data loader
        eval_dataset = TensorDataset(
            self.test_data['time_series'],
            self.test_data['static'],
            self.test_data['pft_param'],
            self.test_data['scalar'],
            self.test_data['variables_1d_pft'],
            self.test_data['variables_2d_soil'],
            self.test_data['y_scalar'],
            self.test_data['y_pft_1d'],
            self.test_data['y_soil_2d']
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=self.config.pin_memory and self.device.type == 'cpu',  # Only pin memory for CPU tensors
            num_workers=self.config.num_workers
        )
        all_predictions = {
            'scalar': [],
            'pft_1d': [],
            'soil_2d': []
        }
        all_targets = {
            'y_scalar': [],
            'y_pft_1d': [],
            'y_soil_2d': []
        }
        with torch.no_grad():
            for time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil, y_scalar, y_pft_1d, y_soil_2d in eval_loader:
                # Move to device
                time_series = time_series.to(self.device, non_blocking=True)
                static = static.to(self.device, non_blocking=True)
                pft_param = pft_param.to(self.device, non_blocking=True)
                scalar = scalar.to(self.device, non_blocking=True)
                variables_1d_pft = variables_1d_pft.to(self.device, non_blocking=True)
                variables_2d_soil = variables_2d_soil.to(self.device, non_blocking=True)
                y_scalar = y_scalar.to(self.device, non_blocking=True)
                y_pft_1d = y_pft_1d.to(self.device, non_blocking=True)
                y_soil_2d = y_soil_2d.to(self.device, non_blocking=True)
                # Forward pass
                if self.use_amp and self.scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil)
                else:
                    outputs = self.model(time_series, static, pft_param, scalar, variables_1d_pft, variables_2d_soil)
                all_predictions['scalar'].append(outputs['scalar'].cpu())
                all_predictions['pft_1d'].append(outputs['pft_1d'].cpu())
                all_predictions['soil_2d'].append(outputs['soil_2d'].cpu())
                all_targets['y_scalar'].append(y_scalar.cpu())
                all_targets['y_pft_1d'].append(y_pft_1d.cpu())
                all_targets['y_soil_2d'].append(y_soil_2d.cpu())
        # Concatenate all batches
        predictions = {k: torch.cat(v, dim=0) for k, v in all_predictions.items()}
        targets = {k: torch.cat(v, dim=0) for k, v in all_targets.items()}
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, targets)
        return predictions, metrics
    
    def _calculate_metrics(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {}
        # Scalar
        pred_scalar = predictions['scalar'].cpu().numpy().flatten()
        target_scalar = targets['y_scalar'].cpu().numpy().flatten()
        mask = ~np.isnan(pred_scalar) & ~np.isnan(target_scalar)
        mse_scalar = mean_squared_error(target_scalar[mask], pred_scalar[mask])
        metrics['scalar_rmse'] = np.sqrt(mse_scalar)
        metrics['scalar_mse'] = mse_scalar
        # PFT 1D
        pred_pft_1d = predictions['pft_1d'].cpu().numpy().flatten()
        target_pft_1d = targets['y_pft_1d'].cpu().numpy().flatten()
        mask = ~np.isnan(pred_pft_1d) & ~np.isnan(target_pft_1d)
        mse_pft_1d = mean_squared_error(target_pft_1d[mask], pred_pft_1d[mask])
        metrics['pft_1d_rmse'] = np.sqrt(mse_pft_1d)
        metrics['pft_1d_mse'] = mse_pft_1d
        # Soil 2D
        pred_soil_2d = predictions['soil_2d'].cpu().numpy().flatten()
        target_soil_2d = targets['y_soil_2d'].cpu().numpy().flatten()
        mask = ~np.isnan(pred_soil_2d) & ~np.isnan(target_soil_2d)
        mse_soil_2d = mean_squared_error(target_soil_2d[mask], pred_soil_2d[mask])
        metrics['soil_2d_rmse'] = np.sqrt(mse_soil_2d)
        metrics['soil_2d_mse'] = mse_soil_2d
        return metrics
    
    def save_results(self, predictions: Dict[str, np.ndarray], metrics: Dict[str, float]):
        """Save training results and predictions."""
        # Save losses (independent of predictions)
        if self.config.save_losses:
            losses_df = pd.DataFrame({
                'Epoch': list(range(1, len(self.train_losses) + 1)),
                'Train Loss': self.train_losses,
                'Validation Loss': self.val_losses
            })
            losses_df.to_csv(self.config.losses_save_path, index=False)
            logger.info(f"Losses saved to {self.config.losses_save_path}")
        
        # Save predictions (only if enabled)
        if self.config.save_predictions:
            # Create predictions directory
            predictions_dir = Path(self.config.predictions_dir)
            predictions_dir.mkdir(exist_ok=True)
        
        # Save predictions
        self._save_predictions(predictions, predictions_dir)
        
        # Save model
        model_path = predictions_dir / "model.pth"
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save scripted model (commented out due to TorchScript limitations with custom objects)
        # scripted_model = torch.jit.script(self.model)
        # scripted_model_path = predictions_dir / "model_scripted.pt"
        # scripted_model.save(str(scripted_model_path))
        # logger.info(f"Scripted model saved to {scripted_model_path}")
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(predictions_dir / "test_metrics.csv", index=False)
        logger.info(f"Metrics saved to {predictions_dir / 'test_metrics.csv'}")
    
    def _save_predictions(self, predictions: Dict[str, np.ndarray], predictions_dir: Path):
        """Save predictions with inverse transformation."""

        for key, tensor in predictions.items():
            if hasattr(tensor, 'shape'):
                logger.info(f"  {key}: {tensor.shape}")
            else:
                logger.info(f"  {key}: {type(tensor)}")

        # Save scalar predictions (renamed to predictions_scalar.csv)
        predictions_scalar_np = predictions['scalar'].cpu().numpy()
        scalar_cols = self.data_info['y_list_scalar_columns'][:predictions_scalar_np.shape[1]]
        predictions_df = pd.DataFrame(predictions_scalar_np, columns=scalar_cols)
        predictions_df.to_csv(os.path.join(predictions_dir, 'predictions_scalar.csv'), index=False)
        # Save ground truth scalar if available
        if 'y_scalar' in self.test_data:
            ground_truth_scalar_np = self.test_data['y_scalar'].cpu().numpy()
            ground_truth_scalar_df = pd.DataFrame(ground_truth_scalar_np, columns=scalar_cols)
            ground_truth_scalar_df.to_csv(os.path.join(predictions_dir, 'ground_truth_scalar.csv'), index=False)

        # Save pft_1d predictions if available
        if 'pft_1d' in predictions and predictions['pft_1d'].numel() > 0:
            predictions_pft_1d_np = predictions['pft_1d'].cpu().numpy()
            n_samples = predictions_pft_1d_np.shape[0]
            # Flatten if 3D
            if predictions_pft_1d_np.ndim == 3:
                predictions_pft_1d_np = predictions_pft_1d_np.reshape(n_samples, -1)
            if 'y_list_pft_1d_columns' in self.data_info and len(self.data_info['y_list_pft_1d_columns']) == predictions_pft_1d_np.shape[1]:
                pft_1d_cols = self.data_info['y_list_pft_1d_columns']
            else:
                pft_1d_cols = [f"pft_1d_{i}" for i in range(predictions_pft_1d_np.shape[1])]
            predictions_1d_df = pd.DataFrame(predictions_pft_1d_np, columns=pft_1d_cols)
            predictions_1d_df.to_csv(os.path.join(predictions_dir, 'predictions_1d.csv'), index=False)
            if 'y_pft_1d' in self.test_data:
                ground_truth_pft_1d_np = self.test_data['y_pft_1d'].cpu().numpy()
                if ground_truth_pft_1d_np.ndim == 3:
                    ground_truth_pft_1d_np = ground_truth_pft_1d_np.reshape(n_samples, -1)
                ground_truth_1d_df = pd.DataFrame(ground_truth_pft_1d_np, columns=pft_1d_cols)
                ground_truth_1d_df.to_csv(os.path.join(predictions_dir, 'ground_truth_1d.csv'), index=False)
            logger.info("pft_1d predictions and ground truth saved successfully")

        # Save soil_2d predictions if available
        if 'soil_2d' in predictions and predictions['soil_2d'].numel() > 0:
            predictions_soil_2d_np = predictions['soil_2d'].cpu().numpy()
            n_samples = predictions_soil_2d_np.shape[0]
            # Flatten if 3D or higher
            if predictions_soil_2d_np.ndim > 2:
                predictions_soil_2d_np = predictions_soil_2d_np.reshape(n_samples, -1)
            if 'y_list_soil_2d_columns' in self.data_info and len(self.data_info['y_list_soil_2d_columns']) == predictions_soil_2d_np.shape[1]:
                soil_2d_cols = self.data_info['y_list_soil_2d_columns']
            else:
                soil_2d_cols = [f"soil_2d_{i}" for i in range(predictions_soil_2d_np.shape[1])]
            predictions_2d_df = pd.DataFrame(predictions_soil_2d_np, columns=soil_2d_cols)
            predictions_2d_df.to_csv(os.path.join(predictions_dir, 'predictions_2d.csv'), index=False)
            if 'y_soil_2d' in self.test_data:
                ground_truth_soil_2d_np = self.test_data['y_soil_2d'].cpu().numpy()
                if ground_truth_soil_2d_np.ndim > 2:
                    ground_truth_soil_2d_np = ground_truth_soil_2d_np.reshape(n_samples, -1)
                ground_truth_2d_df = pd.DataFrame(ground_truth_soil_2d_np, columns=soil_2d_cols)
                ground_truth_2d_df.to_csv(os.path.join(predictions_dir, 'ground_truth_2d.csv'), index=False)
            logger.info("soil_2d predictions and ground truth saved successfully")

        logger.info("All predictions saved successfully")
    
    def plot_training_curves(self, save_path: str = "training_curves.png"):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, label='Training Loss', color='blue', linewidth=2)
        
        if self.val_losses:
            val_epochs = range(1, len(self.val_losses) + 1, self.config.validation_frequency)
            plt.plot(val_epochs, self.val_losses, label='Validation Loss', color='red', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Training and Validation Loss', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to {save_path}")
    
    def run_training_pipeline(self) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        try:
            logger.info("Starting training pipeline...")
            
            # Log initial GPU stats
            if self.config.log_gpu_memory:
                self.gpu_monitor.log_gpu_stats("Training start - ")
            
            # Initialize loss lists in instance variables
            self.train_losses = []
            self.val_losses = []
            
            # Initialize early stopping variables
            self.best_val_loss = float('inf')
            self.patience_counter = 0
            
            logger.info(f"Starting training for {self.config.num_epochs} epochs...")
            
            for epoch in range(self.config.num_epochs):
                epoch_start_time = time.time()
                
                # Train
                train_loss = self.train_epoch()
                self.train_losses.append(train_loss)
                
                # Validate
                val_loss = self.validate_epoch()
                self.val_losses.append(val_loss)
                
                # Update learning rate
                self.scheduler.step(val_loss)
                
                epoch_time = time.time() - epoch_start_time
                
                # Log epoch results
                logger.info(f"Epoch [{epoch+1}/{self.config.num_epochs}] - "
                           f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                           f"Time: {epoch_time:.2f}s")
                
                # Log GPU stats periodically
                if (epoch % 5 == 0 and self.config.log_gpu_memory):
                    self.gpu_monitor.log_gpu_stats(f"Epoch {epoch+1} - ")
                
                # Check for early stopping (only if enabled in config)
                if self.config.use_early_stopping:
                    if self._check_early_stopping(val_loss):
                        logger.info("Early stopping triggered")
                        break
            
            logger.info("Training completed")
            
            # Final GPU stats
            if self.config.log_gpu_memory:
                self.gpu_monitor.log_gpu_stats("Training end - ")
            
            # Evaluate and save results
            logger.info("Evaluating model on test data...")
            predictions, metrics = self.evaluate()
            
            # Save results
            self.save_results(predictions, metrics)
            
            return {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'predictions': predictions,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise 