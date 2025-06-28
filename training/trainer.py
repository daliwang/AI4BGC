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

from config.training_config import TrainingConfig
from models.combined_model import CombinedModel, FlexibleCombinedModel

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
        
        # Setup device
        self.device = self.config.get_device()
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = self.config.get_optimizer(self.model)
        self.scheduler = self.config.get_scheduler(self.optimizer)
        
        # Setup loss functions
        self.criterion_scalar = nn.MSELoss()
        self.criterion_vector = nn.MSELoss()
        self.criterion_matrix = nn.MSELoss()
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Move data to device
        self._move_data_to_device()
        
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
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        
        train_size = self.train_data['time_series'].shape[0]
        
        for i in range(0, train_size, self.config.batch_size):
            end_idx = min(i + self.config.batch_size, train_size)
            
            # Prepare batch data
            batch_data = self._prepare_batch_data(self.train_data, i, end_idx)
            
            # Forward pass
            self.optimizer.zero_grad()
            scalar_pred, vector_pred, matrix_pred = self.model(
                batch_data['time_series'],
                batch_data['static'],
                batch_data['list_1d'],
                batch_data['list_2d']
            )
            
            # Calculate losses
            loss = self._calculate_loss(
                scalar_pred, vector_pred, matrix_pred,
                batch_data['target'], batch_data['y_list_1d'], batch_data['y_list_2d']
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
        
        return running_loss / num_batches
    
    def validate_epoch(self) -> float:
        """
        Validate for one epoch.
        
        Returns:
            Average validation loss for the epoch
        """
        self.model.eval()
        running_loss = 0.0
        num_batches = 0
        
        test_size = self.test_data['time_series'].shape[0]
        
        with torch.no_grad():
            for i in range(0, test_size, self.config.batch_size):
                end_idx = min(i + self.config.batch_size, test_size)
                
                # Prepare batch data
                batch_data = self._prepare_batch_data(self.test_data, i, end_idx)
                
                # Forward pass
                scalar_pred, vector_pred, matrix_pred = self.model(
                    batch_data['time_series'],
                    batch_data['static'],
                    batch_data['list_1d'],
                    batch_data['list_2d']
                )
                
                # Calculate losses
                loss = self._calculate_loss(
                    scalar_pred, vector_pred, matrix_pred,
                    batch_data['target'], batch_data['y_list_1d'], batch_data['y_list_2d']
                )
                
                running_loss += loss.item()
                num_batches += 1
        
        return running_loss / num_batches
    
    def _prepare_batch_data(self, data: Dict[str, Any], start_idx: int, end_idx: int) -> Dict[str, Any]:
        """Prepare batch data for training/validation."""
        batch_data = {}
        
        # Time series and static data
        batch_data['time_series'] = data['time_series'][start_idx:end_idx]
        batch_data['static'] = data['static'][start_idx:end_idx]
        batch_data['target'] = data['target'][start_idx:end_idx]
        
        # List data
        batch_data['list_1d'] = torch.cat([
            tensor[start_idx:end_idx] for tensor in data['list_1d'].values()
        ], dim=1)
        
        batch_data['list_2d'] = torch.cat([
            tensor[start_idx:end_idx].unsqueeze(1) for tensor in data['list_2d'].values()
        ], dim=1)
        
        # Target list data
        batch_data['y_list_1d'] = torch.cat([
            tensor[start_idx:end_idx] for tensor in data['list_1d'].values()
        ], dim=1).view(end_idx - start_idx, len(self.data_info['y_list_columns_1d']), -1)
        
        batch_data['y_list_2d'] = torch.cat([
            tensor[start_idx:end_idx].unsqueeze(1) for tensor in data['list_2d'].values()
        ], dim=1)
        
        return batch_data
    
    def _calculate_loss(self, scalar_pred: torch.Tensor, vector_pred: torch.Tensor, 
                       matrix_pred: torch.Tensor, target: torch.Tensor,
                       y_list_1d: torch.Tensor, y_list_2d: torch.Tensor) -> torch.Tensor:
        """Calculate total loss with weighted components."""
        loss_scalar = self.criterion_scalar(scalar_pred, target)
        loss_vector = self.criterion_vector(vector_pred, y_list_1d)
        loss_matrix = self.criterion_matrix(matrix_pred, y_list_2d)
        
        total_loss = (
            self.config.scalar_loss_weight * loss_scalar +
            self.config.vector_loss_weight * loss_vector +
            self.config.matrix_loss_weight * loss_matrix
        )
        
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
                logger.info(f"Loss weights - Scalar: {loss_weights['scalar']:.4f}, "
                           f"Vector: {loss_weights['vector']:.4f}, "
                           f"Matrix: {loss_weights['matrix']:.4f}")
                
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
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the trained model on test data.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating model on test data...")
        
        self.model.eval()
        predictions = self._get_predictions()
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions)
        
        logger.info(f"Test metrics: {metrics}")
        return metrics
    
    def _get_predictions(self) -> Dict[str, np.ndarray]:
        """Get predictions on test data."""
        predictions_scalar = []
        predictions_vector = []
        predictions_matrix = []
        
        test_size = self.test_data['time_series'].shape[0]
        
        with torch.no_grad():
            for i in range(0, test_size, self.config.batch_size):
                end_idx = min(i + self.config.batch_size, test_size)
                
                # Prepare batch data
                batch_data = self._prepare_batch_data(self.test_data, i, end_idx)
                
                # Get predictions
                scalar_pred, vector_pred, matrix_pred = self.model(
                    batch_data['time_series'],
                    batch_data['static'],
                    batch_data['list_1d'],
                    batch_data['list_2d']
                )
                
                predictions_scalar.append(scalar_pred.cpu().numpy())
                predictions_vector.append(vector_pred.cpu().numpy())
                predictions_matrix.append(matrix_pred.cpu().numpy())
        
        return {
            'scalar': np.concatenate(predictions_scalar, axis=0),
            'vector': np.concatenate(predictions_vector, axis=0),
            'matrix': np.concatenate(predictions_matrix, axis=0)
        }
    
    def _calculate_metrics(self, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {}
        
        # Scalar metrics
        test_target = self.test_data['target'].cpu().numpy()
        mse_scalar = mean_squared_error(test_target, predictions['scalar'])
        metrics['scalar_rmse'] = np.sqrt(mse_scalar)
        metrics['scalar_mse'] = mse_scalar
        
        # Vector metrics
        test_y_list_1d = torch.cat([
            tensor for tensor in self.test_data['list_1d'].values()
        ], dim=1).cpu().numpy()
        
        mse_vector = mean_squared_error(
            test_y_list_1d.reshape(-1, test_y_list_1d.shape[1]),
            predictions['vector'].reshape(-1, predictions['vector'].shape[1])
        )
        metrics['vector_rmse'] = np.sqrt(mse_vector)
        metrics['vector_mse'] = mse_vector
        
        # Matrix metrics
        test_y_list_2d = torch.cat([
            tensor.unsqueeze(1) for tensor in self.test_data['list_2d'].values()
        ], dim=1).cpu().numpy()
        
        mse_matrix = mean_squared_error(
            test_y_list_2d.reshape(-1, test_y_list_2d.shape[1] * test_y_list_2d.shape[2] * test_y_list_2d.shape[3]),
            predictions['matrix'].reshape(-1, predictions['matrix'].shape[1] * predictions['matrix'].shape[2] * predictions['matrix'].shape[3])
        )
        metrics['matrix_rmse'] = np.sqrt(mse_matrix)
        metrics['matrix_mse'] = mse_matrix
        
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
        if self.config.save_model:
            scripted_model = torch.jit.script(self.model)
            scripted_model.save(self.config.model_save_path)
            logger.info(f"Model saved to {self.config.model_save_path}")
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(predictions_dir / "test_metrics.csv", index=False)
        logger.info(f"Metrics saved to {predictions_dir / 'test_metrics.csv'}")
    
    def _save_predictions(self, predictions: Dict[str, np.ndarray], predictions_dir: Path):
        """Save predictions with inverse transformation."""
        # Scalar predictions
        predictions_scalar_np = self.scalers['target'].inverse_transform(predictions['scalar'])
        ground_truth_scalar_np = self.scalers['target'].inverse_transform(
            self.test_data['target'].cpu().numpy()
        )
        
        predictions_df = pd.DataFrame(predictions_scalar_np, columns=self.data_info['target_columns'])
        ground_truth_df = pd.DataFrame(ground_truth_scalar_np, columns=self.data_info['target_columns'])
        
        predictions_df.to_csv(predictions_dir / "predictions_scalar.csv", index=False)
        ground_truth_df.to_csv(predictions_dir / "ground_truth_scalar.csv", index=False)
        
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
        """
        Run the complete training pipeline.
        
        Returns:
            Dictionary containing training results
        """
        try:
            # Train the model
            losses = self.train()
            
            # Evaluate the model
            metrics = self.evaluate()
            
            # Get predictions
            predictions = self._get_predictions()
            
            # Save results
            self.save_results(predictions, metrics)
            
            # Plot training curves
            self.plot_training_curves()
            
            return {
                'losses': losses,
                'metrics': metrics,
                'predictions': predictions,
                'model': self.model
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise 