"""
Data loader module for model training.

This module handles data loading, preprocessing, and preparation for training,
making it easy to work with different input and output configurations.
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
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
            logger.info(f"Found {len(files)} files in {path}")
            
            # Limit number of files per path if specified
            if self.data_config.max_files is not None:
                files = files[:self.data_config.max_files]
                logger.info(f"Limited to {len(files)} files from {path}")
            
            input_files.extend(files)
        
        if not input_files:
            raise RuntimeError("No data files found in specified paths")
        
        logger.info(f"Total files to load: {len(input_files)}")
        
        # Load all files sequentially
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
        initial_size = len(self.df)
        
        # Filter by specific column if specified
        if self.data_config.filter_column:
            # Debug: Check the filter column before filtering
            if self.data_config.filter_column in self.df.columns:
                filter_col_data = self.df[self.data_config.filter_column]
                
                # Special handling for H2OSOI_10CM column (20 layers, ignore first 5)
                if self.data_config.filter_column == 'H2OSOI_10CM':
                    logger.info("Processing H2OSOI_10CM column (20 layers, ignoring first 5)")
                    
                    # Check if the column contains list/array data
                    if isinstance(filter_col_data.iloc[0], (list, np.ndarray)):
                        # Optimized vectorized filtering for large datasets
                        logger.info("Using vectorized filtering for H2OSOI_10CM...")
                        
                        # Convert to numpy arrays for faster processing
                        filter_arrays = filter_col_data.values
                        
                        # Vectorized validation check
                        def is_valid_sample(arr):
                            if isinstance(arr, (list, np.ndarray)):
                                if len(arr) >= 20:
                                    layers_5_to_19 = arr[5:20]
                                    return not np.all(np.isnan(layers_5_to_19))
                                elif len(arr) >= 15:
                                    available_layers = arr[5:]
                                    return not np.all(np.isnan(available_layers))
                            return False
                        
                        # Use numpy vectorize for faster processing
                        valid_mask = np.vectorize(is_valid_sample, otypes=[bool])(filter_arrays)
                        
                        valid_count = np.sum(valid_mask)
                        nan_count = initial_size - valid_count
                        
                        logger.info(f"Filter column '{self.data_config.filter_column}' stats (layers 5-19):")
                        logger.info(f"  - Total samples: {initial_size}")
                        logger.info(f"  - Valid (non-NaN in layers 5-19): {valid_count}")
                        logger.info(f"  - Invalid (all NaN in layers 5-19): {nan_count}")
                        logger.info(f"  - Invalid percentage: {(nan_count/initial_size)*100:.2f}%")
                        
                        # Apply the filter using boolean indexing
                        self.df = self.df[valid_mask].reset_index(drop=True)
                        
                    else:
                        # For scalar data, use original logic
                        nan_count = filter_col_data.isna().sum()
                        valid_count = filter_col_data.notna().sum()
                        logger.info(f"Filter column '{self.data_config.filter_column}' stats (scalar):")
                        logger.info(f"  - Total samples: {initial_size}")
                        logger.info(f"  - Valid (non-NaN): {valid_count}")
                        logger.info(f"  - NaN values: {nan_count}")
                        logger.info(f"  - NaN percentage: {(nan_count/initial_size)*100:.2f}%")
                        
                        # Show some sample values
                        sample_values = filter_col_data.dropna().head(3)
                        logger.info(f"  - Sample values: {sample_values.tolist()}")
                        
                        # Perform the filtering
                        self.df = self.df.dropna(subset=[self.data_config.filter_column])
                else:
                    # For other filter columns, use original logic
                    nan_count = filter_col_data.isna().sum()
                    valid_count = filter_col_data.notna().sum()
                    logger.info(f"Filter column '{self.data_config.filter_column}' stats:")
                    logger.info(f"  - Total samples: {initial_size}")
                    logger.info(f"  - Valid (non-NaN): {valid_count}")
                    logger.info(f"  - NaN values: {nan_count}")
                    logger.info(f"  - NaN percentage: {(nan_count/initial_size)*100:.2f}%")
                    
                    # Show some sample values
                    sample_values = filter_col_data.dropna().head(3)
                    logger.info(f"  - Sample values: {sample_values.tolist()}")
                    
                    # Perform the filtering
                    self.df = self.df.dropna(subset=[self.data_config.filter_column])
            else:
                logger.warning(f"Filter column '{self.data_config.filter_column}' not found in dataset")
                return
            
            final_size = len(self.df)
            logger.info(f"Filtered data: {initial_size} -> {final_size} samples")
            logger.info(f"Removed {initial_size - final_size} samples due to NaN values in '{self.data_config.filter_column}'")
            
            # Check if we have enough data for train/test split
            if final_size < 10:
                logger.warning(f"Very few samples remaining after filtering: {final_size}")
                logger.warning("This might cause issues with train/test splitting")
        else:
            logger.info("No specific column filtering applied")
        
        # Filter NaN values in time series columns if enabled
        if self.data_config.filter_time_series_nan:
            logger.info("Filtering NaN values in time series columns...")
            initial_size = len(self.df)
            
            # Check each time series column for NaN values
            for col in self.data_config.time_series_columns:
                if col in self.df.columns:
                    # Count samples with problematic time series data
                    problematic_mask = []
                    for idx, value in enumerate(self.df[col]):
                        is_problematic = (
                            not isinstance(value, (list, np.ndarray)) or 
                            (isinstance(value, (list, np.ndarray)) and len(value) == 0) or
                            (isinstance(value, (list, np.ndarray)) and np.isnan(value).any())
                        )
                        problematic_mask.append(is_problematic)
                    
                    problematic_count = sum(problematic_mask)
                    if problematic_count > 0:
                        logger.info(f"Column {col}: {problematic_count} problematic samples out of {len(self.df)}")
                        # Remove problematic samples
                        self.df = self.df[~np.array(problematic_mask)].reset_index(drop=True)
                        logger.info(f"Removed {problematic_count} samples with problematic data in {col}")
            
            final_size = len(self.df)
            logger.info(f"Time series filtering: {initial_size} -> {final_size} samples")
            logger.info(f"Removed {initial_size - final_size} samples due to NaN/invalid time series data")
        else:
            logger.info("No time series NaN filtering applied")
    
    def _process_time_series(self):
        """Process time series columns."""
        logger.info("Processing time series data...")
        
        for col in self.data_config.time_series_columns:
            if col not in self.df.columns:
                logger.warning(f"Time series column {col} not found in dataset")
                continue
            
            logger.debug(f"Processing time series column: {col}")
            
            # Check for problematic data before processing
            problematic_samples = []
            for idx, value in enumerate(self.df[col]):
                if not isinstance(value, (list, np.ndarray)) or (isinstance(value, (list, np.ndarray)) and len(value) == 0):
                    problematic_samples.append((idx, type(value), value))
                    if len(problematic_samples) <= 5:  # Only log first 5 problematic samples
                        logger.debug(f"Sample {idx} has problematic data: {type(value)} = {value}")
            
            if problematic_samples:
                logger.warning(f"Found {len(problematic_samples)} samples with problematic time series data in column {col}")
            
            # Pad/truncate time series to specified length
            self.df[col] = self.df[col].apply(
                lambda x: self._process_time_series_item(x, col)
            )
    
    def _process_time_series_item(self, x: Any, col: str) -> np.ndarray:
        """Process a single time series item."""
        try:
            if isinstance(x, list) and len(x) > 0:
                # Convert to numpy array and handle NaN values
                x_array = np.array(x, dtype=np.float32)
                
                # Handle NaN values
                if np.isnan(x_array).any():
                    x_array = np.nan_to_num(x_array, nan=0.0)
                
                # Pad or truncate to target length
                if len(x_array) > self.data_config.max_time_series_length:
                    x_array = x_array[:self.data_config.max_time_series_length]
                
                # Take last N elements
                if len(x_array) > self.data_config.time_series_length:
                    x_array = x_array[-self.data_config.time_series_length:]
                
                # Pad if shorter
                if len(x_array) < self.data_config.time_series_length:
                    x_array = np.pad(x_array, (0, self.data_config.time_series_length - len(x_array)), 'constant')
                
                return x_array
            elif isinstance(x, np.ndarray) and x.size > 0:
                # Handle numpy array input
                x_array = x.astype(np.float32)
                
                # Handle NaN values
                if np.isnan(x_array).any():
                    x_array = np.nan_to_num(x_array, nan=0.0)
                
            # Pad or truncate to target length
                if len(x_array) > self.data_config.max_time_series_length:
                    x_array = x_array[:self.data_config.max_time_series_length]
            
            # Take last N elements
                if len(x_array) > self.data_config.time_series_length:
                    x_array = x_array[-self.data_config.time_series_length:]
            
            # Pad if shorter
                if len(x_array) < self.data_config.time_series_length:
                    x_array = np.pad(x_array, (0, self.data_config.time_series_length - len(x_array)), 'constant')
            
                return x_array
        else:
                # For None, NaN, empty lists, or other invalid data
                logger.debug(f"Invalid time series data for column {col}: {type(x)} = {x}")
                return np.zeros(self.data_config.time_series_length, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Error processing time series item for column {col}: {e}, using zeros")
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
        """Get static columns using the fixed list from config to ensure consistency."""
        # Use the fixed static columns from config
        static_columns = []
        for col in self.data_config.static_columns:
            if col in self.df.columns:
                # Check if the column contains list data
                sample_values = self.df[col].dropna().head(10)
                if len(sample_values) > 0:
                    # Check if any value is a list
                    has_lists = any(isinstance(val, list) for val in sample_values)
                    if not has_lists:
                        static_columns.append(col)
                    else:
                        logger.warning(f"Column {col} contains list data but was not in list configurations. Skipping from static columns.")
                else:
                    # Column exists but has no data, still include it
                    static_columns.append(col)
            else:
                logger.warning(f"Static column {col} not found in dataset. This may cause shape mismatches.")
        
        logger.info(f"Found {len(static_columns)} static columns from fixed config: {static_columns}")
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
        
        # Create time series data with proper shape (samples, time_steps, features)
        time_series_list = []
        for col in self.data_config.time_series_columns:
            if col not in self.df.columns:
                logger.warning(f"Time series column {col} not found, using zeros")
                col_data = np.zeros((len(self.df), self.data_config.time_series_length), dtype=np.float32)
            else:
                # Extract time series data for this column
                col_data = np.vstack(self.df[col].values)
                if col_data.shape[1] != self.data_config.time_series_length:
                    logger.warning(f"Time series column {col} has unexpected shape {col_data.shape}, expected {len(self.df)}x{self.data_config.time_series_length}")
                if col_data.size == 0:
                    logger.error(f"Time series column {col} has empty data!")
                    col_data = np.zeros((len(self.df), self.data_config.time_series_length), dtype=np.float32)
                elif np.isnan(col_data).any():
                    logger.warning(f"Time series column {col} contains NaN values, filling with 0")
                    col_data = np.nan_to_num(col_data, nan=0.0)
                elif np.isinf(col_data).any():
                    logger.warning(f"Time series column {col} contains infinite values, filling with 0")
                    col_data = np.nan_to_num(col_data, nan=0.0, posinf=0.0, neginf=0.0)
            time_series_list.append(col_data)
        
        # Stack along feature dimension to get (samples, time_steps, features)
        time_series_data = np.stack(time_series_list, axis=-1)
        time_series_data = np.ascontiguousarray(time_series_data)
        expected_shape = (len(self.df), self.data_config.time_series_length, len(self.data_config.time_series_columns))
        if time_series_data.shape != expected_shape:
            logger.error(f"Time series data has wrong shape: {time_series_data.shape}, expected {expected_shape}")
            raise ValueError(f"Time series data has wrong shape: {time_series_data.shape}, expected {expected_shape}")
        if time_series_data.shape[0] == 0 or time_series_data.shape[1] == 0:
            logger.error(f"Time series data has invalid shape: {time_series_data.shape}")
            raise ValueError(f"Time series data has invalid shape: {time_series_data.shape}")
        
        # Reshape for normalization: (samples * time_steps, features)
        original_shape = time_series_data.shape
        time_series_flat = time_series_data.reshape(-1, len(self.data_config.time_series_columns))
        scaler = self._get_scaler(self.preprocessing_config.time_series_normalization)
        time_series_normalized = scaler.fit_transform(time_series_flat)
        time_series_data = time_series_normalized.reshape(original_shape)
        time_series_data = np.ascontiguousarray(time_series_data)
        return torch.tensor(time_series_data, dtype=self.preprocessing_config.data_type), scaler
    
    def _normalize_static(self, static_columns: List[str]) -> Tuple[torch.Tensor, Any]:
        """Normalize static data."""
        logger.info("Normalizing static data...")
        
        if not static_columns:
            logger.warning("No static columns found. Creating empty tensor.")
            return torch.empty((len(self.df), 0), dtype=self.preprocessing_config.data_type), None
        
        # Filter out any columns that might contain list data
        valid_static_columns = []
        for col in static_columns:
            if col in self.df.columns:
                # Double-check that the column doesn't contain list data
                sample_values = self.df[col].dropna().head(5)
                if len(sample_values) > 0:
                    has_lists = any(isinstance(val, list) for val in sample_values)
                    if not has_lists:
                        valid_static_columns.append(col)
                    else:
                        logger.warning(f"Skipping column {col} from static normalization - contains list data")
        
        if not valid_static_columns:
            logger.warning("No valid static columns found after filtering. Creating empty tensor.")
            return torch.empty((len(self.df), 0), dtype=self.preprocessing_config.data_type), None
        
        logger.info(f"Normalizing {len(valid_static_columns)} static columns: {valid_static_columns}")
        
        # Extract static data and ensure it's numeric
        static_data = self.df[valid_static_columns].values
        
        # Check for any non-numeric data
        if not np.issubdtype(static_data.dtype, np.number):
            logger.warning("Static data contains non-numeric values. Attempting to convert...")
            try:
                static_data = static_data.astype(np.float64)
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to convert static data to numeric: {e}")
                # Return empty tensor if conversion fails
                return torch.empty((len(self.df), 0), dtype=self.preprocessing_config.data_type), None
        
        # Handle NaN values
        if np.isnan(static_data).any():
            logger.warning("Static data contains NaN values. Filling with 0.")
            static_data = np.nan_to_num(static_data, nan=0.0)
        
        scaler = self._get_scaler(self.preprocessing_config.static_normalization)
        static_normalized = scaler.fit_transform(static_data)
        
        return torch.tensor(static_normalized, dtype=self.preprocessing_config.data_type), scaler
    
    def _normalize_target(self, target_columns: List[str]) -> Tuple[torch.Tensor, Any]:
        """Normalize target data."""
        logger.info("Normalizing target data...")
        
        if not target_columns:
            logger.warning("No target columns found. Creating empty tensor.")
            return torch.empty((len(self.df), 0), dtype=self.preprocessing_config.data_type), None
        
        # Filter out any columns that might contain list data
        valid_target_columns = []
        for col in target_columns:
            if col in self.df.columns:
                # Check if the column contains list data
                sample_values = self.df[col].dropna().head(5)
                if len(sample_values) > 0:
                    has_lists = any(isinstance(val, list) for val in sample_values)
                    if not has_lists:
                        valid_target_columns.append(col)
                    else:
                        logger.warning(f"Skipping column {col} from target normalization - contains list data")
        
        if not valid_target_columns:
            logger.warning("No valid target columns found after filtering. Creating empty tensor.")
            return torch.empty((len(self.df), 0), dtype=self.preprocessing_config.data_type), None
        
        logger.info(f"Normalizing {len(valid_target_columns)} target columns: {valid_target_columns}")
        
        # Extract target data and ensure it's numeric
        target_data = self.df[valid_target_columns].values
        
        # Check for any non-numeric data
        if not np.issubdtype(target_data.dtype, np.number):
            logger.warning("Target data contains non-numeric values. Attempting to convert...")
            try:
                target_data = target_data.astype(np.float64)
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to convert target data to numeric: {e}")
                # Return empty tensor if conversion fails
                return torch.empty((len(self.df), 0), dtype=self.preprocessing_config.data_type), None
        
        # Handle NaN values
        if np.isnan(target_data).any():
            logger.warning("Target data contains NaN values. Filling with 0.")
            target_data = np.nan_to_num(target_data, nan=0.0)
        
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
        
        total_samples = len(self.df)
        train_size = int(self.data_config.train_split * total_samples)
        test_size = total_samples - train_size
        
        logger.info(f"Data splitting details:")
        logger.info(f"  - Total samples: {total_samples}")
        logger.info(f"  - Train split ratio: {self.data_config.train_split}")
        logger.info(f"  - Train size: {train_size}")
        logger.info(f"  - Test size: {test_size}")
        
        if test_size == 0:
            logger.error("Test size is 0! This will cause evaluation issues.")
            logger.error("Consider reducing train_split ratio or increasing dataset size.")
        
        # Split time series data
        train_time_series = normalized_data['time_series_data'][:train_size, :, :]
        test_time_series = normalized_data['time_series_data'][train_size:, :, :]
        
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
        
        logger.info(f"Split completed:")
        logger.info(f"  - Train time_series shape: {train_time_series.shape}")
        logger.info(f"  - Test time_series shape: {test_time_series.shape}")
        logger.info(f"  - Train static shape: {train_static.shape}")
        logger.info(f"  - Test static shape: {test_static.shape}")
        
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
            'available_columns': list(self.df.columns),
            'matrix_rows': self.data_config.max_2d_rows,
            'matrix_cols': self.data_config.max_2d_cols
        } 