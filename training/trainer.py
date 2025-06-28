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
import matplotlib.pyplot as plt
from pathlib import Path
import time
from tqdm import tqdm
import warnings
from torch.utils.data import TensorDataset, DataLoader

from config.training_config import TrainingConfig
from models.combined_model import CombinedModel, FlexibleCombinedModel

# Import GPU monitoring
from utils.gpu_monitor import GPUMonitor, log_memory_usage

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Flexible trainer for climate model training.
    
    This class handles the complete training pipeline including
    data preparation, model training, validation, and saving results.
    """
    
    def __init__(self, training_config: TrainingConfig, model: nn.Module, 
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
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.scalers = scalers
        self.data_info = data_info
        
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
        for split_name in ['train', 'test']:
            split = getattr(self, f'{split_name}_data')
            # Find min length
            lengths = [v.shape[0] for k, v in split.items() if isinstance(v, torch.Tensor)]
            for k, v in split.items():
                if isinstance(v, torch.Tensor) and v.shape[0] != min(lengths):
                    split[k] = v[:min(lengths)]
            # For dicts (list_1d, list_2d)
            for k in ['list_1d', 'list_2d']:
                if k in split and isinstance(split[k], dict):
                    dict_lengths = [vv.shape[0] for vv in split[k].values()]
                    min_dict_len = min(dict_lengths) if dict_lengths else min(lengths)
                    for kk, vv in split[k].items():
                        if vv.shape[0] != min_dict_len:
                            split[k][kk] = vv[:min_dict_len]
        
        # Setup device
        self.device = self.config.get_device()
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
        
        # Prepare y_list_1d and y_list_2d as tensors for train and test
        self.train_data['y_list_1d'] = self._concat_list_columns(self.train_data['list_1d'], self.data_info['y_list_columns_1d'])
        self.test_data['y_list_1d'] = self._concat_list_columns(self.test_data['list_1d'], self.data_info['y_list_columns_1d'])
        self.train_data['y_list_2d'] = self._concat_list_columns_2d(self.train_data['list_2d'], self.data_info['y_list_columns_2d'])
        self.test_data['y_list_2d'] = self._concat_list_columns_2d(self.test_data['list_2d'], self.data_info['y_list_columns_2d'])
        
        # Prepare input list_1d and list_2d as tensors for train and test
        self.train_data['list_1d_tensor'] = self._concat_list_columns(self.train_data['list_1d'], self.data_info['x_list_columns_1d'])
        self.test_data['list_1d_tensor'] = self._concat_list_columns(self.test_data['list_1d'], self.data_info['x_list_columns_1d'])
        self.train_data['list_2d_tensor'] = self._concat_list_columns_2d(self.train_data['list_2d'], self.data_info['x_list_columns_2d'])
        self.test_data['list_2d_tensor'] = self._concat_list_columns_2d(self.test_data['list_2d'], self.data_info['x_list_columns_2d'])
        
        # Log initial GPU stats
        if self.config.log_gpu_memory:
            self.gpu_monitor.log_gpu_stats("Initial ")
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def _move_data_to_device(self):
        """Move all data to the specified device."""
        # Move time series data
        self.train_data['time_series'] = self.train_data['time_series'].to(self.device)
        self.test_data['time_series'] = self.test_data['time_series'].to(self.device)
        
        # Move static data
        self.train_data['static'] = self.train_data['static'].to(self.device)
        self.test_data['static'] = self.test_data['static'].to(self.device)
        
        # Move target data
        self.train_data['target'] = self.train_data['target'].to(self.device)
        self.test_data['target'] = self.test_data['target'].to(self.device)
        
        # Move list data
        for key in ['list_1d', 'list_2d']:
            self.train_data[key] = {k: v.to(self.device) for k, v in self.train_data[key].items()}
            self.test_data[key] = {k: v.to(self.device) for k, v in self.test_data[key].items()}
    
    def _concat_list_columns(self, list_dict, col_names):
        """Concatenate 1D list columns into a tensor."""
        tensors = [list_dict[col] for col in col_names if col in list_dict]
        if tensors:
            return torch.cat(tensors, dim=1)
        else:
            # Return empty tensor with correct batch size
            batch_size = next(iter(list_dict.values())).shape[0] if list_dict else 0
            return torch.empty((batch_size, 0), device=self.device)

    def _concat_list_columns_2d(self, list_dict, col_names):
        """Concatenate 2D list columns into a tensor."""
        tensors = [list_dict[col].unsqueeze(1) for col in col_names if col in list_dict]
        if tensors:
            return torch.cat(tensors, dim=1)
        else:
            batch_size = next(iter(list_dict.values())).shape[0] if list_dict else 0
            return torch.empty((batch_size, 0, 0, 0), device=self.device)
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Create data loader with GPU optimizations
        train_dataset = TensorDataset(
            self.train_data['time_series'],
            self.train_data['static'],
            self.train_data['target'],
            self.train_data['list_1d_tensor'],
            self.train_data['list_2d_tensor'],
            self.train_data['y_list_1d'],
            self.train_data['y_list_2d']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=self.config.persistent_workers
        )
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (time_series, static, target, list_1d, list_2d, y_list_1d, y_list_2d) in enumerate(progress_bar):
            # Move data to device
            time_series = time_series.to(self.device, non_blocking=True)
            static = static.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            list_1d = list_1d.to(self.device, non_blocking=True)
            list_2d = list_2d.to(self.device, non_blocking=True)
            y_list_1d = y_list_1d.to(self.device, non_blocking=True)
            y_list_2d = y_list_2d.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(
                        time_series, static, list_1d, list_2d
                    )
                    # Extract outputs from dictionary
                    scalar_pred = outputs['scalar']
                    vector_pred = outputs['vector']
                    matrix_pred = outputs['matrix']
                    loss = self._compute_loss(
                        scalar_pred, vector_pred, matrix_pred,
                        target, y_list_1d, y_list_2d
                    )
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    time_series, static, list_1d, list_2d
                )
                # Extract outputs from dictionary
                scalar_pred = outputs['scalar']
                vector_pred = outputs['vector']
                matrix_pred = outputs['matrix']
                loss = self._compute_loss(
                    scalar_pred, vector_pred, matrix_pred,
                    target, y_list_1d, y_list_2d
                )
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
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
        
        return total_loss / num_batches
    
    def validate_epoch(self) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Create data loader with GPU optimizations
        val_dataset = TensorDataset(
            self.test_data['time_series'],
            self.test_data['static'],
            self.test_data['target'],
            self.test_data['list_1d_tensor'],
            self.test_data['list_2d_tensor'],
            self.test_data['y_list_1d'],
            self.test_data['y_list_2d']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=self.config.persistent_workers
        )
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            
            for batch_idx, (time_series, static, target, list_1d, list_2d, y_list_1d, y_list_2d) in enumerate(progress_bar):
                # Move data to device
                time_series = time_series.to(self.device, non_blocking=True)
                static = static.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                list_1d = list_1d.to(self.device, non_blocking=True)
                list_2d = list_2d.to(self.device, non_blocking=True)
                y_list_1d = y_list_1d.to(self.device, non_blocking=True)
                y_list_2d = y_list_2d.to(self.device, non_blocking=True)
                
                # Forward pass with mixed precision
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(
                            time_series, static, list_1d, list_2d
                        )
                        # Extract outputs from dictionary
                        scalar_pred = outputs['scalar']
                        vector_pred = outputs['vector']
                        matrix_pred = outputs['matrix']
                        loss = self._compute_loss(
                            scalar_pred, vector_pred, matrix_pred,
                            target, y_list_1d, y_list_2d
                        )
                else:
                    outputs = self.model(
                        time_series, static, list_1d, list_2d
                    )
                    # Extract outputs from dictionary
                    scalar_pred = outputs['scalar']
                    vector_pred = outputs['vector']
                    matrix_pred = outputs['matrix']
                    loss = self._compute_loss(
                        scalar_pred, vector_pred, matrix_pred,
                        target, y_list_1d, y_list_2d
                    )
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/num_batches:.4f}'
                })
        
        return total_loss / num_batches
    
    def _prepare_batch_data(self, data: Dict[str, Any], start_idx: int, end_idx: int) -> Dict[str, Any]:
        """Prepare batch data for training/validation."""
        batch_data = {}
        batch_size = end_idx - start_idx
        device = data['time_series'].device
        
        # Time series and static data
        batch_data['time_series'] = data['time_series'][start_idx:end_idx]
        batch_data['static'] = data['static'][start_idx:end_idx]
        batch_data['target'] = data['target'][start_idx:end_idx]
        assert batch_data['time_series'].shape[0] == batch_size, f"time_series batch size mismatch: {batch_data['time_series'].shape[0]} vs {batch_size}"
        assert batch_data['static'].shape[0] == batch_size, f"static batch size mismatch: {batch_data['static'].shape[0]} vs {batch_size}"
        assert batch_data['target'].shape[0] == batch_size, f"target batch size mismatch: {batch_data['target'].shape[0]} vs {batch_size}"
        
        # List data - separate input and target
        # Input 1D data (only x_list_columns_1d)
        input_1d_tensors = []
        for col in self.data_info['x_list_columns_1d']:
            if col in data['list_1d']:
                t = data['list_1d'][col][start_idx:end_idx]
                input_1d_tensors.append(t)
            else:
                input_1d_tensors.append(torch.zeros(batch_size, self.data_info['vector_length'], device=device))
        if input_1d_tensors:
            batch_data['list_1d'] = torch.cat(input_1d_tensors, dim=1)
        else:
            batch_data['list_1d'] = torch.empty(batch_size, 0, device=device)
        assert batch_data['list_1d'].shape[0] == batch_size, f"list_1d batch size mismatch: {batch_data['list_1d'].shape[0]} vs {batch_size}"
        
        # Input 2D data (only x_list_columns_2d)
        input_2d_tensors = []
        for col in self.data_info['x_list_columns_2d']:
            if col in data['list_2d']:
                t = data['list_2d'][col][start_idx:end_idx]
                if t.shape[0] != batch_size:
                    t = torch.zeros(batch_size, self.data_info['matrix_rows'], self.data_info['matrix_cols'], device=device)
                input_2d_tensors.append(t.unsqueeze(1))
            else:
                input_2d_tensors.append(torch.zeros(batch_size, 1, self.data_info['matrix_rows'], self.data_info['matrix_cols'], device=device))
        if input_2d_tensors:
            batch_data['list_2d'] = torch.cat(input_2d_tensors, dim=1)
        else:
            batch_data['list_2d'] = torch.empty(batch_size, 0, 0, 0, device=device)
        assert batch_data['list_2d'].shape[0] == batch_size, f"list_2d batch size mismatch: {batch_data['list_2d'].shape[0]} vs {batch_size}"
        
        # Target 1D data (only y_list_columns_1d)
        target_1d_tensors = []
        for col in self.data_info['y_list_columns_1d']:
            if col in data['list_1d']:
                t = data['list_1d'][col][start_idx:end_idx]
                target_1d_tensors.append(t)
            else:
                target_1d_tensors.append(torch.zeros(batch_size, self.data_info['vector_length'], device=device))
        if target_1d_tensors:
            batch_data['y_list_1d'] = torch.cat(target_1d_tensors, dim=1)
        else:
            batch_data['y_list_1d'] = torch.empty(batch_size, 0, device=device)
        assert batch_data['y_list_1d'].shape[0] == batch_size, f"y_list_1d batch size mismatch: {batch_data['y_list_1d'].shape[0]} vs {batch_size}"
        
        # Target 2D data (only y_list_columns_2d)
        target_2d_tensors = []
        for col in self.data_info['y_list_columns_2d']:
            if col in data['list_2d']:
                t = data['list_2d'][col][start_idx:end_idx]
                if t.shape[0] != batch_size:
                    t = torch.zeros(batch_size, self.data_info['matrix_rows'], self.data_info['matrix_cols'], device=device)
                target_2d_tensors.append(t.unsqueeze(1))
            else:
                target_2d_tensors.append(torch.zeros(batch_size, 1, self.data_info['matrix_rows'], self.data_info['matrix_cols'], device=device))
        if target_2d_tensors:
            batch_data['y_list_2d'] = torch.cat(target_2d_tensors, dim=1)
        else:
            batch_data['y_list_2d'] = torch.empty(batch_size, 0, 0, 0, device=device)
        assert batch_data['y_list_2d'].shape[0] == batch_size, f"y_list_2d batch size mismatch: {batch_data['y_list_2d'].shape[0]} vs {batch_size}"
        
        return batch_data
    
    def _compute_loss(self, scalar_pred, vector_pred, matrix_pred, target, y_list_1d, y_list_2d):
        """Compute the total loss."""
        total_loss = 0.0
        
        # Scalar loss (only if target has features)
        if target.shape[1] > 0:
            loss_scalar = self.criterion(scalar_pred, target)
            total_loss += self.config.scalar_loss_weight * loss_scalar
        else:
            loss_scalar = torch.tensor(0.0, device=self.device)
        
        # Vector loss (only if we have 1D list targets)
        if y_list_1d.shape[1] > 0:
            # Flatten both predictions and targets to [batch_size, -1]
            loss_vector = self.criterion(
                vector_pred.view(vector_pred.size(0), -1),
                y_list_1d.view(y_list_1d.size(0), -1)
            )
            total_loss += self.config.vector_loss_weight * loss_vector
        else:
            loss_vector = torch.tensor(0.0, device=self.device)
        
        # Matrix loss (only if we have 2D list targets)
        if y_list_2d.shape[1] > 0:
            loss_matrix = self.criterion(matrix_pred, y_list_2d)
            total_loss += self.config.matrix_loss_weight * loss_matrix
        else:
            loss_matrix = torch.tensor(0.0, device=self.device)
        
        # Log loss weights for debugging
        if hasattr(self, '_loss_logged') and not self._loss_logged:
            logger.info(f"Loss weights - Scalar: {self.config.scalar_loss_weight:.4f}, "
                       f"Vector: {self.config.vector_loss_weight:.4f}, "
                       f"Matrix: {self.config.matrix_loss_weight:.4f}")
            self._loss_logged = True
        
        return total_loss
    
    def train(self) -> Dict[str, List[float]]:
        """
        Complete training loop.
        
        Returns:
            Dictionary containing training and validation losses
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs...")
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation
            if epoch % self.config.validation_frequency == 0:
                val_loss = self.validate_epoch()
                self.val_losses.append(val_loss)
                
                # Log progress
                logger.info(f"Epoch [{epoch+1}/{self.config.num_epochs}] - "
                           f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Log loss weights
                loss_weights = self.model.get_loss_weights()
                log_msg = f"Loss weights - "
                if 'scalar' in loss_weights:
                    log_msg += f"Scalar: {loss_weights['scalar']:.4f}, "
                log_msg += f"Vector: {loss_weights['vector']:.4f}, Matrix: {loss_weights['matrix']:.4f}"
                logger.info(log_msg)
                
                # Early stopping
                if self.config.use_early_stopping:
                    if self._check_early_stopping(val_loss):
                        logger.info("Early stopping triggered")
                        break
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
        
        logger.info("Training completed")
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
        
        # Create evaluation data loader
        eval_dataset = TensorDataset(
            self.test_data['time_series'],
            self.test_data['static'],
            self.test_data['target'],
            self.test_data['list_1d_tensor'],
            self.test_data['list_2d_tensor'],
            self.test_data['y_list_1d'],
            self.test_data['y_list_2d']
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers
        )
        
        all_predictions = {
            'scalar': [],
            'vector': [],
            'matrix': []
        }
        all_targets = {
            'scalar': [],
            'vector': [],
            'matrix': []
        }
        
        with torch.no_grad():
            for time_series, static, target, list_1d, list_2d, y_list_1d, y_list_2d in eval_loader:
                # Move to device
                time_series = time_series.to(self.device, non_blocking=True)
                static = static.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                list_1d = list_1d.to(self.device, non_blocking=True)
                list_2d = list_2d.to(self.device, non_blocking=True)
                y_list_1d = y_list_1d.to(self.device, non_blocking=True)
                y_list_2d = y_list_2d.to(self.device, non_blocking=True)
                
                # Forward pass with mixed precision
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(
                            time_series, static, list_1d, list_2d
                        )
                        # Extract outputs from dictionary
                        scalar_pred = outputs['scalar']
                        vector_pred = outputs['vector']
                        matrix_pred = outputs['matrix']
                else:
                    outputs = self.model(
                        time_series, static, list_1d, list_2d
                    )
                    # Extract outputs from dictionary
                    scalar_pred = outputs['scalar']
                    vector_pred = outputs['vector']
                    matrix_pred = outputs['matrix']
                
                # Collect predictions and targets
                all_predictions['scalar'].append(scalar_pred.cpu())
                all_predictions['vector'].append(vector_pred.cpu())
                all_predictions['matrix'].append(matrix_pred.cpu())
                
                all_targets['scalar'].append(target.cpu())
                all_targets['vector'].append(y_list_1d.cpu())
                all_targets['matrix'].append(y_list_2d.cpu())
        
        # Concatenate all batches
        predictions = {
            'scalar': torch.cat(all_predictions['scalar'], dim=0),
            'vector': torch.cat(all_predictions['vector'], dim=0),
            'matrix': torch.cat(all_predictions['matrix'], dim=0)
        }
        
        targets = {
            'scalar': torch.cat(all_targets['scalar'], dim=0),
            'vector': torch.cat(all_targets['vector'], dim=0),
            'matrix': torch.cat(all_targets['matrix'], dim=0)
        }
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, targets)
        
        return predictions, metrics
    
    def _calculate_metrics(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        from sklearn.metrics import mean_squared_error
        
        metrics = {}
        
        # Scalar metrics
        test_target = targets['scalar'].cpu().numpy()
        pred_scalar = predictions['scalar'].cpu().numpy()
        mse_scalar = mean_squared_error(test_target, pred_scalar)
        metrics['scalar_rmse'] = np.sqrt(mse_scalar)
        metrics['scalar_mse'] = mse_scalar
        
        # Vector metrics - use the y_list_1d tensor directly
        if 'vector' in predictions and predictions['vector'].shape[1] > 0:
            test_y_list_1d = targets['vector'].cpu().numpy()
            pred_vector = predictions['vector'].cpu().numpy()
            
            # Squeeze pred_vector if it has extra singleton dimension
            if pred_vector.ndim == 3 and pred_vector.shape[1] == 1:
                pred_vector = pred_vector.squeeze(1)
            
            # Ensure shapes match
            min_samples = min(test_y_list_1d.shape[0], pred_vector.shape[0])
            test_y_list_1d = test_y_list_1d[:min_samples]
            pred_vector = pred_vector[:min_samples]
            
            mse_vector = mean_squared_error(
                test_y_list_1d.reshape(-1),
                pred_vector.reshape(-1)
            )
            metrics['vector_rmse'] = np.sqrt(mse_vector)
            metrics['vector_mse'] = mse_vector
        else:
            metrics['vector_rmse'] = 0.0
            metrics['vector_mse'] = 0.0
        
        # Matrix metrics - use the y_list_2d tensor directly
        if 'matrix' in predictions and predictions['matrix'].shape[1] > 0:
            test_y_list_2d = targets['matrix'].cpu().numpy()
            pred_matrix = predictions['matrix'].cpu().numpy()
            
            # Squeeze pred_matrix if it has extra singleton dimension
            if pred_matrix.ndim == 4 and pred_matrix.shape[1] == 1:
                pred_matrix = pred_matrix.squeeze(1)
            
            # Ensure both arrays have 4 dimensions
            while test_y_list_2d.ndim < 4:
                test_y_list_2d = np.expand_dims(test_y_list_2d, axis=-1)
            while pred_matrix.ndim < 4:
                pred_matrix = np.expand_dims(pred_matrix, axis=-1)
            
            # Ensure shapes match
            min_samples = min(test_y_list_2d.shape[0], pred_matrix.shape[0])
            min_channels = min(test_y_list_2d.shape[1], pred_matrix.shape[1])
            min_rows = min(test_y_list_2d.shape[2], pred_matrix.shape[2])
            min_cols = min(test_y_list_2d.shape[3], pred_matrix.shape[3])
            test_y_list_2d = test_y_list_2d[:min_samples, :min_channels, :min_rows, :min_cols]
            pred_matrix = pred_matrix[:min_samples, :min_channels, :min_rows, :min_cols]
            
            # Flatten both arrays to 1D
            test_y_flat = test_y_list_2d.reshape(-1)
            pred_flat = pred_matrix.reshape(-1)
            
            # Ensure they have the same length
            min_length = min(len(test_y_flat), len(pred_flat))
            test_y_flat = test_y_flat[:min_length]
            pred_flat = pred_flat[:min_length]
            
            if min_length > 0:
                mse_matrix = mean_squared_error(test_y_flat, pred_flat)
                metrics['matrix_rmse'] = np.sqrt(mse_matrix)
                metrics['matrix_mse'] = mse_matrix
            else:
                metrics['matrix_rmse'] = 0.0
                metrics['matrix_mse'] = 0.0
        else:
            metrics['matrix_rmse'] = 0.0
            metrics['matrix_mse'] = 0.0
        
        return metrics
    
    def save_results(self, predictions: Dict[str, np.ndarray], metrics: Dict[str, float]):
        """Save training results and predictions."""
        if not self.config.save_predictions:
            return
        
        # Create predictions directory
        predictions_dir = Path(self.config.predictions_dir)
        predictions_dir.mkdir(exist_ok=True)
        
        # Save losses
        if self.config.save_losses:
            losses_df = pd.DataFrame({
                'Epoch': list(range(1, len(self.train_losses) + 1)),
                'Train Loss': self.train_losses,
                'Validation Loss': self.val_losses
            })
            losses_df.to_csv(self.config.losses_save_path, index=False)
            logger.info(f"Losses saved to {self.config.losses_save_path}")
        
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
        # Save scalar predictions
        predictions_scalar_np = predictions['scalar'].cpu().numpy()
        # Use only the correct number of columns
        scalar_cols = self.data_info['target_columns'][:predictions_scalar_np.shape[1]]
        predictions_df = pd.DataFrame(predictions_scalar_np, columns=scalar_cols)
        predictions_df.to_csv(os.path.join(predictions_dir, 'scalar_predictions.csv'), index=False)
        
        # 1D predictions
        predictions_1d_dict = {}
        ground_truth_1d_dict = {}
        
        for idx, col in enumerate(self.data_info['x_list_columns_1d']):
            y_col = 'Y_' + col
            if y_col in self.scalers['list_1d']:
                predictions_1d_dict[y_col] = self.scalers['list_1d'][y_col].inverse_transform(
                    predictions['vector'][:, idx, :]
                )
        
        for y_col in self.test_data['list_1d']:
            ground_truth_1d_dict[y_col] = self.test_data['list_1d'][y_col].cpu().numpy()
        
        predictions_1d_df = pd.DataFrame({
            col: predictions_1d_dict[col].tolist() for col in predictions_1d_dict
        })
        ground_truth_1d_df = pd.DataFrame({
            col: ground_truth_1d_dict[col].tolist() for col in ground_truth_1d_dict
        })
        
        predictions_1d_df.to_csv(predictions_dir / "predictions_1d.csv", index=False)
        ground_truth_1d_df.to_csv(predictions_dir / "ground_truth_1d.csv", index=False)
        
        # 2D predictions
        predictions_2d_dict = {}
        ground_truth_2d_dict = {}
        
        for idx, col in enumerate(self.data_info['x_list_columns_2d']):
            y_col = 'Y_' + col
            if y_col in self.scalers['list_2d']:
                pred_reshaped = predictions['matrix'][:, idx, :, :].reshape(-1, 1)
                predictions_2d_dict[y_col] = self.scalers['list_2d'][y_col].inverse_transform(
                    pred_reshaped
                ).reshape(-1, predictions['matrix'].shape[2], predictions['matrix'].shape[3])
        
        for y_col in self.test_data['list_2d']:
            ground_truth_2d_dict[y_col] = self.test_data['list_2d'][y_col].cpu().numpy()
        
        predictions_2d_df = pd.DataFrame({
            col: predictions_2d_dict[col].tolist() for col in predictions_2d_dict
        })
        ground_truth_2d_df = pd.DataFrame({
            col: ground_truth_2d_dict[col].tolist() for col in ground_truth_2d_dict
        })
        
        predictions_2d_df.to_csv(predictions_dir / "predictions_2d.csv", index=False)
        ground_truth_2d_df.to_csv(predictions_dir / "ground_truth_2d.csv", index=False)
        
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
            
            # Training loop
            train_losses = []
            val_losses = []
            
            logger.info(f"Starting training for {self.config.num_epochs} epochs...")
            
            for epoch in range(self.config.num_epochs):
                epoch_start_time = time.time()
                
                # Train
                train_loss = self.train_epoch()
                train_losses.append(train_loss)
                
                # Validate
                val_loss = self.validate_epoch()
                val_losses.append(val_loss)
                
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
                
                # Check for early stopping
                if len(val_losses) > 10 and val_losses[-1] > min(val_losses[-10:]):
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
                'train_losses': train_losses,
                'val_losses': val_losses,
                'predictions': predictions,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise 