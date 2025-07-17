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
        # Check for matching input/output pairs for 1D
        if len(self.data_config.variables_1d_pft) != len(self.data_config.y_list_columns_1d):
            raise ValueError("Number of 1D input (variables_1d_pft) and output columns must match")
        # Relaxed check for 2D: all outputs must be in inputs, but inputs can have extras
        def strip_y(col):
            return col[2:] if col.startswith('Y_') else col
        x2d_set = set(self.data_config.x_list_columns_2d)
        missing_outputs = [col for col in self.data_config.y_list_columns_2d if strip_y(col) not in x2d_set]
        if missing_outputs:
            raise ValueError(f"The following 2D output columns (after removing 'Y_') are not present in 2D input columns: {missing_outputs}")
        if len(self.data_config.x_list_columns_2d) != len(self.data_config.y_list_columns_2d):
            logger.warning(f"Number of 2D input columns ({len(self.data_config.x_list_columns_2d)}) does not match number of 2D output columns ({len(self.data_config.y_list_columns_2d)}). This is allowed if some 2D inputs are input-only.")
    
    def check_nans(self):
        """Check for NaN values in the loaded DataFrame and log the count per column."""
        if self.df is None:
            logger.warning("No data loaded to check for NaNs.")
            return
        nan_counts = self.df.isna().sum()
        total_nans = nan_counts.sum()
        logger.info(f"Total NaN values in DataFrame: {total_nans}")
        logger.info("NaN count per column:")
        for col, count in nan_counts.items():
            if count > 0:
                logger.info(f"  {col}: {count}")

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
        # Check for NaNs after loading
        self.check_nans()
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
            self.data_config.variables_1d_pft + 
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
            # Impute NaNs with 0.0 in all arrays
            # count the number of NaNs in the array
            self.df[col] = self.df[col].apply(
                lambda x: np.isnan(x).sum() if isinstance(x, np.ndarray) else 0
            )
            # print the number of NaNs in the array
            logger.info(f"Number of NaNs in {col}: {self.df[col].sum()}")
            self.df[col] = self.df[col].apply(
                lambda x: np.nan_to_num(x, nan=0.0) if isinstance(x, np.ndarray) and np.isnan(x).any() else x
            )
            # Special handling for PFT parameter columns (should be 1D of length 17)
            if col in self.data_config.x_list_columns_2d and col in self.data_config.x_list_columns_2d[-44:]:
                # If 1D, reshape to (17, 1)
                self.df[col] = self.df[col].apply(
                    lambda x: x.reshape(17, 1) if isinstance(x, np.ndarray) and x.ndim == 1 and x.shape[0] == 17 else x
                )
            # Debug: Check array shapes
            sample_values = self.df[col].dropna().head(5)
            for i, val in enumerate(sample_values):
                if isinstance(val, np.ndarray):
                    logger.debug(f"Column {col}, sample {i}: shape={val.shape}, ndim={val.ndim}")
                else:
                    logger.debug(f"Column {col}, sample {i}: type={type(val)}, value={val}")
            # Truncate to specified dimensions
            self.df[col] = self.df[col].apply(
                lambda x: x[:, :self.data_config.max_2d_cols] 
                if isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] >= self.data_config.max_2d_cols 
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
        Normalize all data types using config-defined column order.
        Returns:
            Dictionary containing normalized data and scalers
        """
        logger.info("Normalizing data...")
        # Print config lists and DataFrame columns for debugging
        # print('--- DEBUG: Config variable lists ---')
        # print('x_list_scalar_columns:', self.data_config.x_list_scalar_columns)
        # print('y_list_scalar_columns:', self.data_config.y_list_scalar_columns)
        # print('variables_1d_pft:', self.data_config.variables_1d_pft)
        # print('y_list_columns_1d:', self.data_config.y_list_columns_1d)
        # print('x_list_columns_2d:', self.data_config.x_list_columns_2d)
        # print('y_list_columns_2d:', self.data_config.y_list_columns_2d)
        # print('pft_param_columns:', self.data_config.pft_param_columns)
        # print('DataFrame columns:', list(self.df.columns))
        # print('--- END DEBUG ---')

        # Time series
        time_series_data, time_series_scaler = self._normalize_time_series()
        
        # Static
        static_data, static_scaler = self._normalize_static(self.data_config.static_columns)

        # Scalar
        scalar_data, scalar_scaler = self._normalize_scalar()
        y_scalar_data, y_scalar_scaler = self._normalize_y_scalar()

        # 1D PFT
        pft_1d_data, pft_1d_scaler = self._normalize_list_1d(self.data_config.variables_1d_pft)
        y_pft_1d_data, y_pft_1d_scaler = self._normalize_list_1d(self.data_config.y_list_columns_1d)

        # 2D Soil
        variables_2d_soil, variables_2d_soil_scaler = self._normalize_list_2d(self.data_config.x_list_columns_2d)
        y_soil_2d, y_soil_2d_scaler = self._normalize_list_2d(self.data_config.y_list_columns_2d)

        # PFT param
        pft_param_data, pft_param_scaler = self._normalize_pft_param()

        # Water (if present)
        water_tensor = None
        y_water_tensor = None
        if hasattr(self.data_config, 'x_list_water_columns') and self.data_config.x_list_water_columns:
            water_tensor, water_scaler = self._normalize_list_1d(self.data_config.x_list_water_columns)
        if hasattr(self.data_config, 'y_list_water_columns') and self.data_config.y_list_water_columns:
            y_water_tensor, y_water_scaler = self._normalize_list_1d(self.data_config.y_list_water_columns)

        # Print tensor shapes for debugging
        # print('--- DEBUG: Normalized tensor shapes ---')
        # print('time_series_data:', time_series_data.shape)
        # print('static_data:', static_data.shape)
        # print('scalar_data:', scalar_data.shape)
        # print('y_scalar_data:', y_scalar_data.shape)
        # print('pft_1d_data:', pft_1d_data.shape)
        # print('y_pft_1d_data:', y_pft_1d_data.shape)
        # print('variables_2d_soil:', variables_2d_soil.shape)
        # print('y_soil_2d:', y_soil_2d.shape)
        # print('pft_param_data:', pft_param_data.shape)
        # if water_tensor is not None:
        #     print('water_tensor:', water_tensor.shape)
        # if y_water_tensor is not None:
        #     print('y_water_tensor:', y_water_tensor.shape)
        # print('--- END DEBUG ---')

        # Assert lists are not empty
        assert len(self.data_config.variables_1d_pft) > 0, 'variables_1d_pft list is empty!'
        assert pft_1d_data.shape[1] == len(self.data_config.variables_1d_pft), 'Mismatch in 1D PFT feature count!'
        assert scalar_data.shape[1] == len(self.data_config.x_list_scalar_columns), 'Mismatch in scalar feature count!'
        assert variables_2d_soil.shape[1] == len(self.data_config.x_list_columns_2d), 'Mismatch in 2D soil feature count!'
        assert pft_param_data.shape[1] == len(self.data_config.pft_param_columns), 'Mismatch in PFT param feature count!'
        assert y_scalar_data.shape[1] == len(self.data_config.y_list_scalar_columns), 'Mismatch in y_scalar feature count!'
        assert y_pft_1d_data.shape[1] == len(self.data_config.y_list_columns_1d), 'Mismatch in y_pft_1d feature count!'
        assert y_soil_2d.shape[1] == len(self.data_config.y_list_columns_2d), 'Mismatch in y_soil_2d feature count!'

        # Store all scalers
        self.scalers = {
            'time_series': time_series_scaler,
            'static': static_scaler,
            'scalar': scalar_scaler,
            'y_scalar': y_scalar_scaler,
            'pft_1d': pft_1d_scaler,
            'y_pft_1d': y_pft_1d_scaler,
            'variables_2d_soil': variables_2d_soil_scaler,
            'y_soil_2d': y_soil_2d_scaler,
            'pft_param': pft_param_scaler,
            'water': water_scaler if 'water_scaler' in locals() else None,
            'y_water': y_water_scaler if 'y_water_scaler' in locals() else None
        }
        
        return {
            'time_series_data': time_series_data,
            'static_data': static_data,
            'pft_param_data': pft_param_data,
            'scalar_data': scalar_data,
            'variables_1d_pft': pft_1d_data,
            'variables_2d_soil': variables_2d_soil,
            'y_scalar': y_scalar_data,
            'y_pft_1d': y_pft_1d_data,
            'y_soil_2d': y_soil_2d,
            'water': water_tensor,
            'y_water': y_water_tensor,
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
        """Normalize static data in the order defined by static_columns."""
        logger.info(f"Normalizing static data with columns: {static_columns}")
        # Enforce order
        for i, col in enumerate(static_columns):
            assert col in self.df.columns, f"Static column '{col}' missing in DataFrame!"
        static_data = self.df[static_columns].values
        scaler = self._get_scaler(self.preprocessing_config.static_normalization)
        static_normalized = scaler.fit_transform(static_data)
        return torch.tensor(static_normalized, dtype=self.preprocessing_config.data_type), scaler
    
    def _normalize_scalar(self) -> Tuple[torch.Tensor, Any]:
        """Normalize scalar variables in the order defined by x_list_scalar_columns."""
        scalar_columns = self.data_config.x_list_scalar_columns
        logger.info(f"Normalizing scalar data with columns: {scalar_columns}")
        for i, col in enumerate(scalar_columns):
            assert col in self.df.columns, f"Scalar column '{col}' missing in DataFrame!"
        scalar_data = self.df[scalar_columns].values
        scaler = self._get_scaler(self.preprocessing_config.static_normalization)
        scalar_normalized = scaler.fit_transform(scalar_data)
        return torch.tensor(scalar_normalized, dtype=self.preprocessing_config.data_type), scaler

    def _normalize_y_scalar(self) -> Tuple[torch.Tensor, Any]:
        """Normalize y_scalar variables in the order defined by y_list_scalar_columns."""
        y_scalar_columns = self.data_config.y_list_scalar_columns
        logger.info(f"Normalizing y_scalar data with columns: {y_scalar_columns}")
        for i, col in enumerate(y_scalar_columns):
            assert col in self.df.columns, f"y_scalar column '{col}' missing in DataFrame!"
        y_scalar_data = self.df[y_scalar_columns].values
        scaler = self._get_scaler(self.preprocessing_config.static_normalization)
        y_scalar_normalized = scaler.fit_transform(y_scalar_data)
        return torch.tensor(y_scalar_normalized, dtype=self.preprocessing_config.data_type), scaler

    def _normalize_list_1d(self, columns: List[str]) -> Tuple[torch.Tensor, Any]:
        """Normalize 1D list data in the order defined by columns."""
        logger.info(f"Normalizing 1D list data with columns: {columns}")
        for i, col in enumerate(columns):
            assert col in self.df.columns, f"1D column '{col}' missing in DataFrame!"
        col_data = [np.vstack(self.df[col].values) for col in columns]
        data = np.stack(col_data, axis=1)  # shape: (samples, features, length)
        n_samples, n_features, n_length = data.shape
        data_reshaped = data.reshape(n_samples, -1)
        scaler = self._get_scaler(self.preprocessing_config.list_1d_normalization)
        data_normalized = scaler.fit_transform(data_reshaped)
        data_normalized = data_normalized.reshape(n_samples, n_features, n_length)
        return torch.tensor(data_normalized, dtype=self.preprocessing_config.data_type), scaler

    def _normalize_list_2d(self, columns: List[str]) -> Tuple[torch.Tensor, Any]:
        """Normalize 2D list data in the order defined by columns."""
        logger.info(f"Normalizing 2D list data with columns: {columns}")
        for i, col in enumerate(columns):
            assert col in self.df.columns, f"2D column '{col}' missing in DataFrame!"
        col_data = [np.stack(self.df[col].values) for col in columns]
        data = np.stack(col_data, axis=1)  # shape: (samples, features, rows, cols)
        n_samples, n_features, n_rows, n_cols = data.shape
        data_reshaped = data.reshape(n_samples, -1)
        scaler = self._get_scaler(self.preprocessing_config.list_2d_normalization)
        data_normalized = scaler.fit_transform(data_reshaped)
        data_normalized = data_normalized.reshape(n_samples, n_features, n_rows, n_cols)
        return torch.tensor(data_normalized, dtype=self.preprocessing_config.data_type), scaler

    def _normalize_pft_param(self) -> Tuple[torch.Tensor, Any]:
        """Normalize and stack pft_param data as [batch, 44, 17] in config order."""
        pft_param_columns = self.data_config.pft_param_columns

        num_params = len(pft_param_columns)
        num_pfts = 17  # Always use 17 PFTs
        logger.info(f"Normalizing pft_param data with columns: {pft_param_columns}")
        # Enforce order and presence
        for col in pft_param_columns:
            assert col in self.df.columns, f"PFT param column '{col}' missing in DataFrame!"
        # Stack in config order
        param_matrix = []
        for idx, row in self.df.iterrows():
            row_vectors = []
            for col in pft_param_columns:
                val = row[col]
                if isinstance(val, (list, np.ndarray)) and len(val) == num_pfts:
                    row_vectors.append(np.array(val, dtype=np.float32))
                else:
                    row_vectors.append(np.zeros(num_pfts, dtype=np.float32))
            row_matrix = np.stack(row_vectors, axis=0)  # [44, 17]
            param_matrix.append(row_matrix)
        param_matrix = np.stack(param_matrix, axis=0)  # [batch, 44, 17]
        assert param_matrix.shape[1:] == (num_params, num_pfts), f"pft_param_data shape {param_matrix.shape} does not match [batch, 44, 17]"
        # Flatten for normalization
        flat_param_matrix = param_matrix.reshape(param_matrix.shape[0], -1)
        scaler = self._get_scaler(self.preprocessing_config.list_1d_normalization)
        flat_param_matrix_norm = scaler.fit_transform(flat_param_matrix)
        param_matrix_norm = flat_param_matrix_norm.reshape(param_matrix.shape)
        pft_param_data = torch.tensor(param_matrix_norm, dtype=self.preprocessing_config.data_type)
        return pft_param_data, scaler


    def _normalize_list_scalar(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Normalize scalar data."""
        logger.info("Normalizing scalar data...")
        
        list_columns_scalar = (
            self.data_config.x_list_scalar_columns + 
            self.data_config.y_list_scalar_columns) 
        
        normalized_data = {}
        scalers = {}
        
        for col in list_columns_scalar:
            if col not in self.df.columns:
                continue
            # Stack data
            col_data = np.vstack(self.df[col].values)
            scaler = self._get_scaler(self.preprocessing_config.list_1d_normalization)
            col_normalized = scaler.fit_transform(col_data)
            normalized_data[col] = torch.tensor(col_normalized, dtype=self.preprocessing_config.data_type)
            scalers[col] = scaler
        return normalized_data, scalers
    
    def _concat_list_columns_2d(self, list_data: Dict[str, torch.Tensor], column_names: List[str]) -> torch.Tensor:
        """
        Concatenates 2D list data for a given set of column names.
        This is useful when you have multiple 2D outputs and want to stack them.
        """
        concatenated_data = []
        for col_name in column_names:
            if col_name in list_data:
                # Ensure the tensor is 2D and has the correct shape
                tensor = list_data[col_name]
                if tensor.ndim == 2 and tensor.shape[1] == 1: # Assuming 1D of length 1
                    concatenated_data.append(tensor.squeeze(1)) # Remove the extra dimension
                else:
                    concatenated_data.append(tensor)
            else:
                logger.warning(f"Column '{col_name}' not found in list_data for concatenation.")
                # Optionally, you could append a zero tensor or raise an error
                concatenated_data.append(torch.zeros((len(self.df), self.data_config.max_2d_rows, self.data_config.max_2d_cols), dtype=self.preprocessing_config.data_type))
        
        # Stack the concatenated tensors along the first dimension (batch)
        return torch.stack(concatenated_data, dim=1) # Stack along the 2nd dimension (features)
    
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
        Split normalized data into train and test sets.
        """
        logger.info("Splitting data into train/test sets...")

        train_data = {}
        test_data = {}
        
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
        train_data['time_series'] = train_time_series
        test_data['time_series'] = test_time_series
        
        # Split static data
        train_static = normalized_data['static_data'][:train_size]
        test_static = normalized_data['static_data'][train_size:]
        train_data['static'] = train_static
        test_data['static'] = test_static
        
        # Split pft_param data
        train_pft_param = normalized_data['pft_param_data'][:train_size]
        test_pft_param = normalized_data['pft_param_data'][train_size:]
        train_data['pft_param'] = train_pft_param
        test_data['pft_param'] = test_pft_param

        # Split scalar data (input)
        train_list_scalar = normalized_data['scalar_data'][:train_size]
        test_list_scalar = normalized_data['scalar_data'][train_size:]
        train_data['scalar'] = train_list_scalar
        test_data['scalar'] = test_list_scalar 

        # Split y_scalar (target)
        y_scalar = normalized_data['y_scalar']
        train_data['y_scalar'] = y_scalar[:train_size]
        test_data['y_scalar'] = y_scalar[train_size:]

        # Split variables_1d_pft (input)
        variables_1d_pft = normalized_data['variables_1d_pft']
        train_data['variables_1d_pft'] = variables_1d_pft[:train_size]
        test_data['variables_1d_pft'] = variables_1d_pft[train_size:]
        
        # Split y_pft_1d (target)
        y_pft_1d = normalized_data['y_pft_1d']
        train_data['y_pft_1d'] = y_pft_1d[:train_size]
        test_data['y_pft_1d'] = y_pft_1d[train_size:]

        # Split y_soil_2d (target)
        y_soil_2d = normalized_data['y_soil_2d']
        train_data['y_soil_2d'] = y_soil_2d[:train_size]
        test_data['y_soil_2d'] = y_soil_2d[train_size:]

        # Split variables_2d_soil (input)
        variables_2d_soil = normalized_data['variables_2d_soil']
        train_data['variables_2d_soil'] = variables_2d_soil[:train_size]
        test_data['variables_2d_soil'] = variables_2d_soil[train_size:]
        
        # Split water data if present
        if 'water' in normalized_data and normalized_data['water'] is not None:
            train_data['water'] = normalized_data['water'][:train_size]
            test_data['water'] = normalized_data['water'][train_size:]
        if 'y_water' in normalized_data and normalized_data['y_water'] is not None:
            train_data['y_water'] = normalized_data['y_water'][:train_size]
            test_data['y_water'] = normalized_data['y_water'][train_size:]
        
        logger.info(f"Split completed:")
        logger.info(f"  - Train time_series shape: {train_time_series.shape}")
        logger.info(f"  - Test time_series shape: {test_time_series.shape}")
        logger.info(f"  - Train static shape: {train_static.shape}")
        logger.info(f"  - Test static shape: {test_static.shape}")
        
        # Only keep final keys in output
        final_keys = [
            'time_series', 'static', 'pft_param', 'scalar',
            'variables_1d_pft', 'variables_2d_soil',
            'y_scalar', 'y_pft_1d', 'y_soil_2d'
        ]   

        # we need to make sure water is optional 
        if 'water' in train_data:
            final_keys.append('water')
            final_keys.append('y_water')

        train_data = {k: v for k, v in train_data.items() if k in final_keys}
        test_data = {k: v for k, v in test_data.items() if k in final_keys}
        
        return {
            'train': train_data,
            'test': test_data,
            'train_size': train_size,
            'test_size': test_size
        }
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded data, ensuring all keys are present and correct."""
        if self.df is None:
            return {}
        data_info = {
            'total_samples': len(self.df),
            'time_series_columns': self.data_config.time_series_columns,
            'static_columns': self.data_config.static_columns,
            'pft_param_columns': self.data_config.pft_param_columns,
            'variables_1d_pft': self.data_config.variables_1d_pft,  # Canonical 1D PFT variable list
            'y_list_columns_1d': self.data_config.y_list_columns_1d,
            'x_list_scalar_columns': self.data_config.x_list_scalar_columns,
            'y_list_scalar_columns': self.data_config.y_list_scalar_columns,
            'x_list_columns_2d': self.data_config.x_list_columns_2d,
            'y_list_columns_2d': self.data_config.y_list_columns_2d
        } 
        # print('[DEBUG] get_data_info: variables_1d_pft =', data_info['variables_1d_pft'])
        return data_info 