"""
Data loader module for model training.

This module handles data loading, preprocessing, and preparation for training,
making it easy to work with different input and output configurations.
"""

import os
import glob
import logging
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import warnings

from config.training_config import DataConfig, PreprocessingConfig

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Flexible data loader for climate model training.
    
    This class handles loading, preprocessing, and preparing data for training
    with different input and output configurations.
    """
    
    def __init__(self, data_config: DataConfig, preprocessing_config: PreprocessingConfig):
        """
        Initialize the data loader.
        
        Args:
            data_config: Data configuration
            preprocessing_config: Preprocessing configuration
        """
        self.data_config = data_config
        self.preprocessing_config = preprocessing_config
        self.df = None
        self.scalers = {}
        
        # Validate configurations
        self._validate_configs()
    
    def _validate_configs(self):
        """Validate data and preprocessing configurations."""
        if not self.data_config.data_paths:
            raise ValueError("Data paths cannot be empty")
        
        if not self.data_config.time_series_columns:
            raise ValueError("Time series columns cannot be empty")
        
        # Check for matching input/output pairs
        if len(self.data_config.x_list_columns_1d) != len(self.data_config.y_list_columns_1d):
            raise ValueError("Number of 1D input and output columns must match")
        
        if len(self.data_config.x_list_columns_2d) != len(self.data_config.y_list_columns_2d):
            raise ValueError("Number of 2D input and output columns must match")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from all specified paths.
        
        Returns:
            Combined DataFrame
        """
        logger.info("Loading data from multiple paths...")
        
        input_files = []
        for path in self.data_config.data_paths:
            pattern = os.path.join(path, self.data_config.file_pattern)
            files = sorted(glob.glob(pattern))
            input_files.extend(files)
            logger.info(f"Found {len(files)} files in {path}")
        
        if not input_files:
            raise RuntimeError("No data files found in specified paths")
        
        logger.info(f"Total files found: {len(input_files)}")
        
        # Load all files
        df_list = []
        for file in input_files:
            try:
                logger.debug(f"Loading {file}...")
                batch_df = pd.read_pickle(file)
                df_list.append(batch_df)
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
                continue
        
        if not df_list:
            raise RuntimeError("No data loaded successfully")
        
        # Combine all DataFrames
        self.df = pd.concat(df_list, ignore_index=True)
        logger.info(f"Successfully loaded {len(self.df)} samples")
        
        return self.df
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the loaded data.
        
        Returns:
            Preprocessed DataFrame
        """
        if self.df is None:
            raise RuntimeError("Data must be loaded before preprocessing")
        
        logger.info("Starting data preprocessing...")
        
        # Drop specified columns
        self._drop_columns()
        
        # Filter data if specified
        self._filter_data()
        
        # Process time series data
        self._process_time_series()
        
        # Process list columns (1D and 2D)
        self._process_list_columns()
        
        # Shuffle data
        self._shuffle_data()
        
        logger.info("Data preprocessing completed")
        return self.df
    
    def _drop_columns(self):
        """Drop specified columns from the dataset."""
        if self.data_config.columns_to_drop:
            self.df = self.df.drop(columns=self.data_config.columns_to_drop, errors='ignore')
            logger.info(f"Dropped columns: {self.data_config.columns_to_drop}")
    
    def _filter_data(self):
        """Filter data based on specified criteria."""
        if self.data_config.filter_column:
            initial_size = len(self.df)
            self.df = self.df.dropna(subset=[self.data_config.filter_column])
            final_size = len(self.df)
            logger.info(f"Filtered data: {initial_size} -> {final_size} samples")
    
    def _process_time_series(self):
        """Process time series columns."""
        logger.info("Processing time series data...")
        
        for col in self.data_config.time_series_columns:
            if col not in self.df.columns:
                logger.warning(f"Time series column {col} not found in dataset")
                continue
            
            logger.debug(f"Processing time series column: {col}")
            
            # Pad/truncate time series to specified length
            self.df[col] = self.df[col].apply(
                lambda x: self._process_time_series_item(x, col)
            )
    
    def _process_time_series_item(self, x: Any, col: str) -> np.ndarray:
        """Process a single time series item."""
        if isinstance(x, list):
            # Pad or truncate to target length
            if len(x) > self.data_config.max_time_series_length:
                x = x[:self.data_config.max_time_series_length]
            
            # Take last N elements
            if len(x) > self.data_config.time_series_length:
                x = x[-self.data_config.time_series_length:]
            
            # Pad if shorter
            if len(x) < self.data_config.time_series_length:
                x = np.pad(x, (0, self.data_config.time_series_length - len(x)), 'constant')
            
            return np.array(x, dtype=np.float32)
        else:
            return np.zeros(self.data_config.time_series_length, dtype=np.float32)
    
    def _process_list_columns(self):
        """Process 1D and 2D list columns."""
        logger.info("Processing list columns...")
        
        # Process 1D columns
        self._process_1d_columns()
        
        # Process 2D columns
        self._process_2d_columns()
    
    def _process_1d_columns(self):
        """Process 1D list columns."""
        list_columns_1d = (
            self.data_config.x_list_columns_1d + 
            self.data_config.y_list_columns_1d
        )
        
        for col in list_columns_1d:
            if col not in self.df.columns:
                logger.warning(f"1D column {col} not found in dataset")
                continue
            
            logger.debug(f"Processing 1D column: {col}")
            
            # Truncate to specified length
            self.df[col] = self.df[col].apply(
                lambda x: x[1:17] if isinstance(x, list) else x
            )
            
            # Pad to uniform length
            self.df[col] = self.df[col].apply(
                lambda x: self._pad_1d_array(x, self.data_config.max_1d_length)
            )
    
    def _process_2d_columns(self):
        """Process 2D list columns."""
        list_columns_2d = (
            self.data_config.x_list_columns_2d + 
            self.data_config.y_list_columns_2d
        )
        
        for col in list_columns_2d:
            if col not in self.df.columns:
                logger.warning(f"2D column {col} not found in dataset")
                continue
            
            logger.debug(f"Processing 2D column: {col}")
            
            # Convert to numpy arrays
            self.df[col] = self.df[col].apply(
                lambda x: np.array(x) if isinstance(x, list) else x
            )
            
            # Truncate to specified dimensions
            self.df[col] = self.df[col].apply(
                lambda x: x[:, :self.data_config.max_2d_cols] 
                if isinstance(x, np.ndarray) and x.shape[1] >= self.data_config.max_2d_cols 
                else x
            )
            
            # Pad to uniform shape
            self.df[col] = self.df[col].apply(
                lambda x: self._pad_2d_array(x, self.data_config.max_2d_rows, self.data_config.max_2d_cols)
            )
    
    def _pad_1d_array(self, x: Any, target_length: int) -> np.ndarray:
        """Pad 1D array to target length."""
        if isinstance(x, (list, np.ndarray)):
            x_array = np.array(x)
            if len(x_array) < target_length:
                return np.pad(x_array, (0, target_length - len(x_array)), mode='constant')
            else:
                return x_array[:target_length]
        else:
            return np.zeros(target_length, dtype=np.float32)
    
    def _pad_2d_array(self, x: Any, target_rows: int, target_cols: int) -> np.ndarray:
        """Pad 2D array to target shape."""
        if isinstance(x, np.ndarray) and x.ndim == 2:
            if x.shape[0] < target_rows or x.shape[1] < target_cols:
                return np.pad(
                    x, 
                    ((0, target_rows - x.shape[0]), (0, target_cols - x.shape[1])), 
                    mode='constant'
                )
            else:
                return x[:target_rows, :target_cols]
        else:
            return np.zeros((target_rows, target_cols), dtype=np.float32)
    
    def _shuffle_data(self):
        """Shuffle the dataset."""
        logger.info("Shuffling dataset...")
        self.df = shuffle(self.df, random_state=self.data_config.random_state).reset_index(drop=True)
    
    def normalize_data(self) -> Dict[str, Any]:
        """
        Normalize all data types.
        
        Returns:
            Dictionary containing normalized data and scalers
        """
        logger.info("Normalizing data...")
        
        # Get static and target columns
        static_columns = self._get_static_columns()
        target_columns = self._get_target_columns()
        
        # Normalize time series data
        time_series_data, time_series_scaler = self._normalize_time_series()
        
        # Normalize static data
        static_data, static_scaler = self._normalize_static(static_columns)
        
        # Normalize target data
        target_data, target_scaler = self._normalize_target(target_columns)
        
        # Normalize list data
        list_1d_data, list_1d_scalers = self._normalize_list_1d()
        list_2d_data, list_2d_scalers = self._normalize_list_2d()
        
        # Store all scalers
        self.scalers = {
            'time_series': time_series_scaler,
            'static': static_scaler,
            'target': target_scaler,
            'list_1d': list_1d_scalers,
            'list_2d': list_2d_scalers
        }
        
        return {
            'time_series_data': time_series_data,
            'static_data': static_data,
            'target_data': target_data,
            'list_1d_data': list_1d_data,
            'list_2d_data': list_2d_data,
            'scalers': self.scalers
        }
    
    def _get_static_columns(self) -> List[str]:
        """Get static columns (non-time series, non-list, non-target)."""
        all_columns = set(self.df.columns)
        time_series_set = set(self.data_config.time_series_columns)
        list_1d_set = set(self.data_config.x_list_columns_1d + self.data_config.y_list_columns_1d)
        list_2d_set = set(self.data_config.x_list_columns_2d + self.data_config.y_list_columns_2d)
        target_set = set(self.data_config.y_list_columns_1d + self.data_config.y_list_columns_2d)
        
        static_columns = list(
            all_columns - time_series_set - list_1d_set - list_2d_set - target_set
        )
        
        logger.info(f"Found {len(static_columns)} static columns")
        return static_columns
    
    def _get_target_columns(self) -> List[str]:
        """Get target columns (Y_ prefixed, excluding list targets)."""
        all_columns = set(self.df.columns)
        list_targets = set(self.data_config.y_list_columns_1d + self.data_config.y_list_columns_2d)
        
        target_columns = [
            col for col in all_columns 
            if col.startswith('Y_') and col not in list_targets
        ]
        
        logger.info(f"Found {len(target_columns)} target columns")
        return target_columns
    
    def _normalize_time_series(self) -> Tuple[torch.Tensor, Any]:
        """Normalize time series data."""
        logger.info("Normalizing time series data...")
        
        # Stack time series data
        time_series_data = np.stack([
            np.column_stack(self.df[col]) for col in self.data_config.time_series_columns
        ], axis=-1)
        
        # Reshape for normalization
        original_shape = time_series_data.shape
        time_series_flat = time_series_data.reshape(-1, len(self.data_config.time_series_columns))
        
        # Apply normalization
        scaler = self._get_scaler(self.preprocessing_config.time_series_normalization)
        time_series_normalized = scaler.fit_transform(time_series_flat)
        
        # Reshape back
        time_series_data = time_series_normalized.reshape(original_shape)
        
        return torch.tensor(time_series_data, dtype=self.preprocessing_config.data_type), scaler
    
    def _normalize_static(self, static_columns: List[str]) -> Tuple[torch.Tensor, Any]:
        """Normalize static data."""
        logger.info("Normalizing static data...")
        
        static_data = self.df[static_columns].values
        scaler = self._get_scaler(self.preprocessing_config.static_normalization)
        static_normalized = scaler.fit_transform(static_data)
        
        return torch.tensor(static_normalized, dtype=self.preprocessing_config.data_type), scaler
    
    def _normalize_target(self, target_columns: List[str]) -> Tuple[torch.Tensor, Any]:
        """Normalize target data."""
        logger.info("Normalizing target data...")
        
        target_data = self.df[target_columns].values
        scaler = self._get_scaler(self.preprocessing_config.target_normalization)
        target_normalized = scaler.fit_transform(target_data)
        
        return torch.tensor(target_normalized, dtype=self.preprocessing_config.data_type), scaler
    
    def _normalize_list_1d(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Normalize 1D list data."""
        logger.info("Normalizing 1D list data...")
        
        list_columns_1d = (
            self.data_config.x_list_columns_1d + 
            self.data_config.y_list_columns_1d
        )
        
        normalized_data = {}
        scalers = {}
        
        for col in list_columns_1d:
            if col not in self.df.columns:
                continue
            
            # Stack data
            col_data = np.vstack(self.df[col].values)
            
            # Normalize
            scaler = self._get_scaler(self.preprocessing_config.list_1d_normalization)
            col_normalized = scaler.fit_transform(col_data)
            
            # Convert to tensor
            normalized_data[col] = torch.tensor(
                col_normalized, dtype=self.preprocessing_config.data_type
            )
            scalers[col] = scaler
        
        return normalized_data, scalers
    
    def _normalize_list_2d(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Normalize 2D list data."""
        logger.info("Normalizing 2D list data...")
        
        list_columns_2d = (
            self.data_config.x_list_columns_2d + 
            self.data_config.y_list_columns_2d
        )
        
        normalized_data = {}
        scalers = {}
        
        for col in list_columns_2d:
            if col not in self.df.columns:
                continue
            
            # Reshape data for normalization
            col_data = np.vstack(self.df[col].values).reshape(-1, 1)
            
            # Normalize
            scaler = self._get_scaler(self.preprocessing_config.list_2d_normalization)
            col_normalized = scaler.fit_transform(col_data)
            
            # Reshape back
            col_reshaped = col_normalized.reshape(
                len(self.df), self.data_config.max_2d_rows, self.data_config.max_2d_cols
            )
            
            # Convert to tensor
            normalized_data[col] = torch.tensor(
                col_reshaped, dtype=self.preprocessing_config.data_type
            )
            scalers[col] = scaler
        
        return normalized_data, scalers
    
    def _get_scaler(self, normalization_type: str):
        """Get appropriate scaler based on normalization type."""
        if normalization_type.lower() == 'minmax':
            return MinMaxScaler(feature_range=self.preprocessing_config.minmax_range)
        elif normalization_type.lower() == 'standard':
            return StandardScaler()
        elif normalization_type.lower() == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unsupported normalization type: {normalization_type}")
    
    def split_data(self, normalized_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Split data into training and testing sets.
        
        Args:
            normalized_data: Dictionary containing normalized data
            
        Returns:
            Dictionary containing train/test splits
        """
        logger.info("Splitting data into train/test sets...")
        
        train_size = int(self.data_config.train_split * len(self.df))
        test_size = len(self.df) - train_size
        
        # Split time series data
        train_time_series = normalized_data['time_series_data'][:train_size]
        test_time_series = normalized_data['time_series_data'][train_size:]
        
        # Split static data
        train_static = normalized_data['static_data'][:train_size]
        test_static = normalized_data['static_data'][train_size:]
        
        # Split target data
        train_target = normalized_data['target_data'][:train_size]
        test_target = normalized_data['target_data'][train_size:]
        
        # Split list data
        train_list_1d = {
            col: tensor[:train_size] for col, tensor in normalized_data['list_1d_data'].items()
        }
        test_list_1d = {
            col: tensor[train_size:] for col, tensor in normalized_data['list_1d_data'].items()
        }
        
        train_list_2d = {
            col: tensor[:train_size] for col, tensor in normalized_data['list_2d_data'].items()
        }
        test_list_2d = {
            col: tensor[train_size:] for col, tensor in normalized_data['list_2d_data'].items()
        }
        
        return {
            'train': {
                'time_series': train_time_series,
                'static': train_static,
                'target': train_target,
                'list_1d': train_list_1d,
                'list_2d': train_list_2d
            },
            'test': {
                'time_series': test_time_series,
                'static': test_static,
                'target': test_target,
                'list_1d': test_list_1d,
                'list_2d': test_list_2d
            },
            'train_size': train_size,
            'test_size': test_size
        }
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded data."""
        if self.df is None:
            return {}
        
        return {
            'total_samples': len(self.df),
            'time_series_columns': self.data_config.time_series_columns,
            'static_columns': self._get_static_columns(),
            'target_columns': self._get_target_columns(),
            'x_list_columns_1d': self.data_config.x_list_columns_1d,
            'y_list_columns_1d': self.data_config.y_list_columns_1d,
            'x_list_columns_2d': self.data_config.x_list_columns_2d,
            'y_list_columns_2d': self.data_config.y_list_columns_2d,
            'available_columns': list(self.df.columns)
        } 