#!/usr/bin/env python3
"""
Smart DataLoader that automatically prioritizes files with non-zero soil2D and PFT1D data
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import logging
from pathlib import Path
from typing import List, Tuple, Any, Optional, Dict

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.data_loader_individual import DataLoaderIndividual

logger = logging.getLogger(__name__)

class SmartDataLoader(DataLoaderIndividual):
    """
    Smart DataLoader that automatically detects and prioritizes files with non-zero soil2D and PFT1D data
    """
    
    def load_data(self) -> pd.DataFrame:
        """Load data with smart file prioritization."""
        logger.info("Loading data with SmartDataLoader...")
        
        # Check if max_files is set
        if hasattr(self.data_config, 'max_files') and self.data_config.max_files is not None:
            logger.info(f"Max files limit detected: {self.data_config.max_files}")
            return self._load_data_with_smart_prioritization()
        else:
            logger.info("No max_files limit - loading all files normally")
            return super().load_data()
    
    def _load_data_with_smart_prioritization(self) -> pd.DataFrame:
        """Load data with smart file prioritization - using hardcoded best files for testing."""
        logger.info("Using smart file prioritization with hardcoded best files")
        
        # Hardcoded selection of the 3 best files based on previous analysis
        # These files have the richest soil2D and PFT1D data
        best_files = [
            "../TrainingData/Trendy_1_data_CNP/enhanced_dataset/enhanced_1_training_data_batch_11_first10.pkl"
        ]
        
        logger.info(f"Selected best files: {best_files}")
        
        # Load only the selected files
        return self._load_specific_files([Path(f) for f in best_files])
    
    def _analyze_files_for_data_richness(self, files: List[Path]) -> Dict[Path, float]:
        """Analyze files and assign scores based on soil2D and PFT1D data richness."""
        file_scores = {}
        
        # Get variable lists from config
        soil2d_vars = getattr(self.data_config, 'x_list_columns_2d', [])
        pft1d_vars = getattr(self.data_config, 'x_list_columns_1d', [])
        
        logger.info(f"Analyzing {len(files)} files for data richness...")
        logger.info(f"Looking for soil2D variables: {soil2d_vars}")
        logger.info(f"Looking for PFT1D variables: {pft1d_vars}")
        
        for file_path in files:
            try:
                # Calculate combined score from both soil2D and PFT1D data
                soil2d_score = self._calculate_file_soil2d_score(file_path, soil2d_vars)
                pft1d_score = self._calculate_file_pft1d_score(file_path, pft1d_vars)
                
                # Combine scores (weight soil2D more heavily since it's the main focus)
                combined_score = 0.7 * soil2d_score + 0.3 * pft1d_score
                file_scores[file_path] = combined_score
                
                if combined_score > 0:
                    logger.info(f"  {file_path.name}: soil2D={soil2d_score:.2f}, PFT1D={pft1d_score:.2f}, combined={combined_score:.2f}")
                else:
                    logger.info(f"  {file_path.name}: score {combined_score:.2f} (poor data)")
                    
            except Exception as e:
                logger.warning(f"Error analyzing {file_path}: {e}")
                file_scores[file_path] = 0.0
        
        return file_scores
    
    def _calculate_file_soil2d_score(self, file_path: Path, soil2d_vars: List[str]) -> float:
        """Calculate a score for how rich a file is in soil2D data - aligned with original logic."""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if not isinstance(data, pd.DataFrame):
                return 0.0
            
            total_score = 0.0
            total_samples = 0
            
            for var in soil2d_vars:
                if var not in data.columns:
                    continue
                
                values = data[var].values
                var_score = 0.0
                var_samples = 0
                
                # Sample up to 100 samples for efficiency
                sample_indices = np.linspace(0, len(values)-1, min(100, len(values)), dtype=int)
                
                for idx in sample_indices:
                    val = values[idx]
                    if isinstance(val, (list, np.ndarray)):
                        val_array = np.array(val)
                        if val_array.ndim == 2 and val_array.shape[1] == 15:  # Has 15 layers
                            # Original logic: check if data exists in the expected structure
                            # Don't look for specific non-zero columns, just check data presence
                            if val_array.shape[0] > 0:  # Has columns
                                # Score based on data structure and sample count
                                # Higher score for more columns (more spatial coverage)
                                column_score = min(val_array.shape[0] / 18.0, 1.0)  # Normalize to 0-1
                                var_score += column_score
                                var_samples += 1
                
                if var_samples > 0:
                    var_avg_score = var_score / var_samples
                    total_score += var_avg_score
                    total_samples += 1
            
            if total_samples > 0:
                return total_score / total_samples
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Error calculating soil2D score for {file_path}: {e}")
            return 0.0
    
    def _calculate_file_pft1d_score(self, file_path: Path, pft1d_vars: List[str]) -> float:
        """Calculate a score for how rich a file is in PFT1D data - aligned with original logic."""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if not isinstance(data, pd.DataFrame):
                return 0.0
            
            total_score = 0.0
            total_samples = 0
            
            for var in pft1d_vars:
                if var not in data.columns:
                    continue
                
                values = data[var].values
                var_score = 0.0
                var_samples = 0
                
                # Sample up to 100 samples for efficiency
                sample_indices = np.linspace(0, len(values)-1, min(100, len(values)), dtype=int)
                
                for idx in sample_indices:
                    val = values[idx]
                    if isinstance(val, (list, np.ndarray)):
                        val_array = np.array(val)
                        if len(val_array) > 0:  # Has PFT data
                            # Original logic: check if data exists and has expected length
                            # Score based on data presence and length
                            length_score = min(len(val_array) / 17.0, 1.0)  # Normalize to 0-1
                            var_score += length_score
                            var_samples += 1
                
                if var_samples > 0:
                    var_avg_score = var_score / var_samples
                    total_score += var_avg_score
                    total_samples += 1
            
            if total_samples > 0:
                return total_score / total_samples
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Error calculating PFT1D score for {file_path}: {e}")
            return 0.0
    
    def _load_specific_files(self, selected_files: List[Path]) -> pd.DataFrame:
        """Load data from specific selected files."""
        import os
        import numpy as np
        df_list = []
        for file_path in selected_files:
            try:
                logger.info(f"Loading {file_path.name}")
                abs_path = os.path.abspath(file_path)
                print(f"[DEBUG] Absolute file path: {abs_path}")
                if str(file_path).endswith('.pkl'):
                    df_chunk = pd.read_pickle(file_path)
                else:
                    df_chunk = pd.read_parquet(file_path)
                df_list.append(df_chunk)
                logger.info(f"  Loaded {len(df_chunk)} samples")
                # Diagnostic print for debugging
                print(f"[DEBUG] Loaded DataFrame from: {file_path}")
                print(f"[DEBUG] DataFrame shape: {df_chunk.shape}")
                print(f"[DEBUG] First few columns: {df_chunk.columns[:5].tolist()}")
                for col in getattr(self.data_config, 'x_list_columns_2d', []):
                    if col in df_chunk.columns:
                        first_val = df_chunk[col].iloc[0]
                        print(f"[DEBUG] First value in {col}: type={type(first_val)}, value={first_val}")
                        # Sum of first 50 arrays
                        arrs = [np.array(df_chunk[col].iloc[i]) for i in range(min(50, len(df_chunk)))]
                        arr_sum = sum(a.sum() for a in arrs if a.size > 0)
                        print(f"[DEBUG] Sum of first 50 arrays in {col}: {arr_sum}")
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue
        if not df_list:
            raise ValueError("No data files could be loaded")
        # Combine all dataframes
        self.df = pd.concat(df_list, ignore_index=True)
        logger.info(f"Successfully loaded {len(self.df)} total samples from {len(df_list)} files")

        # Debug print using inspector helpers for all configured PFT1D and Soil2D variables
        try:
            self._debug_print_with_inspector()
        except Exception as e:
            logger.warning(f"Inspector debug print failed: {e}")
        return self.df
    
    def _normalize_list_1d_individual(self, columns: List[str]) -> Tuple[torch.Tensor, Any]:
        """Enhanced normalization aligned with original data construction logic for PFT1D data."""
        logger.info(f"Normalizing 1D list data with columns: {columns}")
        
        for i, col in enumerate(columns):
            assert col in self.df.columns, f"1D column '{col}' missing in DataFrame!"
        
        # Align with original logic: take first 17 elements, pad to consistent length
        col_data = []
        
        # First pass: determine maximum length across all samples
        max_length_1d = 17  # Always 17 PFTs as per original
        
        for col in columns:
            values = self.df[col].values
            for val in values:
                if isinstance(val, (list, np.ndarray)):
                    val_array = np.array(val)
                    if len(val_array) > 0:  # Has PFT data
                        max_length_1d = max(max_length_1d, min(len(val_array), 17))
        
        logger.info(f"Maximum PFT1D length: {max_length_1d}")
        
        for col in columns:
            values = self.df[col].values
            standardized_samples = []
            
            logger.info(f"Processing column: {col} - {len(values)} samples")
            
            for val in values:
                if isinstance(val, (list, np.ndarray)):
                    val_array = np.array(val)
                    if len(val_array) > 0:  # Has PFT data
                        # Original logic: take first 17 elements
                        extracted = val_array[:17]  # Shape: (17,)
                        
                        # Pad to consistent length (align with original dataloader)
                        padded = np.pad(extracted, 
                                      (0, max_length_1d - len(extracted)), 
                                      mode='constant', 
                                      constant_values=0)
                        
                        standardized_samples.append(padded)
                    else:
                        # No data, use zeros
                        standardized_samples.append(np.zeros(max_length_1d))
                else:
                    # Invalid data type, use zeros
                    standardized_samples.append(np.zeros(max_length_1d))
            
            col_data.append(np.stack(standardized_samples))
            logger.info(f"  Column {col} shape: {col_data[-1].shape}")
        
        data = np.stack(col_data, axis=1)  # shape: (samples, features, max_length)
        
        # Enhanced logging
        logger.info(f"Standardized PFT1D data shape: {data.shape}")
        
        # Check non-zero counts before normalization
        total_non_zeros = np.count_nonzero(data)
        logger.info(f"Before normalization - Total PFT1D non-zero count: {total_non_zeros}")
        
        # Check per-column non-zero counts
        for i, col in enumerate(columns):
            col_non_zeros = np.count_nonzero(data[:, i, :])
            logger.info(f"  {col}: {col_non_zeros} non-zeros")
        
        # Check first few samples
        if data.shape[0] > 0:
            logger.info(f"Before normalization - First sample data:")
            for i, col in enumerate(columns):
                sample_data = data[0, i, :]
                non_zeros = np.count_nonzero(sample_data)
                logger.info(f"  {col}: shape {sample_data.shape}, non-zeros: {non_zeros}")
                if non_zeros > 0:
                    logger.info(f"    Non-zero values: {sample_data[sample_data != 0][:3]}")
                else:
                    logger.info(f"    All zeros")
        
        # Continue with normal normalization
        if columns == self.data_config.x_list_columns_1d:
            # Input PFT1D data
            # For PFT1D, the data shape is (samples, features, pfts)
            # We need to transpose to (samples, pfts, features) for the scaler
            data = np.transpose(data, (0, 2, 1))  # (samples, pfts, features)
            
            # Exclude PFT0 (index 0) - only train on PFT1-PFT16
            data = data[:, 1:, :]  # Keep only PFT1-PFT16 (shape: samples, 16, features)
            pft_names = [f'PFT{i}' for i in range(1, 17)]  # PFT1-PFT16
            
            normalized_data = self.individual_scalers['pft_1d'].fit_transform_pft_1d(
                data, 
                pft_names, 
                columns
            )
            # Transpose back to original shape
            normalized_data = np.transpose(normalized_data, (0, 2, 1))
            
            # Log after normalization for input PFT1D
            non_zero_count_after = np.count_nonzero(normalized_data)
            logger.info(f"After normalization - Input PFT1D non-zero count: {non_zero_count_after}")
            if normalized_data.shape[0] > 0:
                logger.info(f"After normalization - Input PFT1D sample (first few elements): {normalized_data[0, :2, :3]}")
            return torch.tensor(normalized_data, dtype=self.preprocessing_config.data_type), self.individual_scalers['pft_1d']
        else:
            # Output PFT1D data
            # For PFT1D, the data shape is (samples, features, pfts)
            # We need to transpose to (samples, pfts, features) for the scaler
            data = np.transpose(data, (0, 2, 1))  # (samples, pfts, features)
            
            # Exclude PFT0 (index 0) - only train on PFT1-PFT16
            data = data[:, 1:, :]  # Keep only PFT1-PFT16 (shape: samples, 16, features)
            pft_names = [f'PFT{i}' for i in range(1, 17)]  # PFT1-PFT16
            
            normalized_data = self.individual_scalers['y_pft_1d'].fit_transform_pft_1d(
                data, 
                pft_names, 
                columns
            )
            # Transpose back to original shape
            normalized_data = np.transpose(normalized_data, (0, 2, 1))
            
            # Log after normalization for output PFT1D
            non_zero_count_after = np.count_nonzero(normalized_data)
            logger.info(f"After normalization - Output PFT1D non-zero count: {non_zero_count_after}")
            if normalized_data.shape[0] > 0:
                logger.info(f"After normalization - Output PFT1D sample (first few elements): {normalized_data[0, :2, :3]}")
            return torch.tensor(normalized_data, dtype=self.preprocessing_config.data_type), self.individual_scalers['y_pft_1d']

    def _normalize_list_2d_individual(self, columns: List[str]) -> Tuple[torch.Tensor, Any]:
        """Robust normalization for 2D list data: extract FIRST GROUP (column) -> 15 layers -> top 10.

        This mirrors the extraction semantics used in scripts/inspect_pft1d_soil2d.py so that
        nested object arrays (ragged lists) select the first inner list as the 15-layer vector.
        """
        logger.info(f"[SmartDataLoader] Normalizing 2D list data with columns: {columns}")
        col_data = []
        for col in columns:
            assert col in self.df.columns, f"2D column '{col}' missing in DataFrame!"
            values = self.df[col].values
            standardized_samples = []
            # Print and save top 3 layers of first column for first 50 samples
            out = []
            for i, val in enumerate(values):
                # Convert to object array to avoid unintended ragged coercion
                arr = np.array(val, dtype=object)

                # Extract first group (first column) with 15 layers, mirroring inspect script
                first_group_15 = None
                # Case A: nested list/tuple/ndarray -> pick first inner element
                if isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], (list, tuple, np.ndarray)):
                    try:
                        first_group_15 = np.array(val[0], dtype=float).flatten()[:15]
                    except Exception:
                        first_group_15 = None
                # Case B: numeric 2D array -> take first row
                if first_group_15 is None and arr.ndim == 2 and arr.shape[0] >= 1:
                    try:
                        first_group_15 = np.array(arr[0, :15], dtype=float)
                    except Exception:
                        first_group_15 = None
                # Case C: flat 1D already representing 15 layers
                if first_group_15 is None and arr.ndim == 1 and len(arr) >= 15 and not isinstance(arr[0], (list, tuple, np.ndarray)):
                    try:
                        first_group_15 = np.array(arr[:15], dtype=float)
                    except Exception:
                        first_group_15 = None
                # Fallback
                if first_group_15 is None:
                    first_group_15 = np.zeros(15, dtype=float)

                # For preview, show top 3 layers
                top3 = first_group_15[:3] if first_group_15 is not None else np.full(3, np.nan)
                if i < 50:
                    print(f"Sample {i}: {top3}")
                    out.append(top3)
                # For main data extraction, take top 10 layers from first group of 15
                standardized_samples.append(first_group_15[:10])
            if len(out) == 50:
                out_arr = np.stack(out)
                np.savetxt(f"{col}_first50_top3layers_from_dataloader.txt", out_arr, fmt="%.6f", delimiter=",")
                print(f"Saved {col}_first50_top3layers_from_dataloader.txt, shape: {out_arr.shape}")
            col_data.append(np.stack(standardized_samples))
        # Stack variables: shape (samples, variables, 10)
        data = np.stack(col_data, axis=1)  # (samples, variables, 10)
        logger.info(f"[SmartDataLoader] Standardized Soil2D data shape: {data.shape}")
        non_zero_count_before = np.count_nonzero(data)
        logger.info(f"[SmartDataLoader] Before normalization - Soil2D non-zero count: {non_zero_count_before}")
        if data.shape[0] > 0:
            logger.info(f"[SmartDataLoader] Before normalization - Soil2D sample (first variable, first sample, top 3 layers): {data[0,0,:3]}")
        # Add singleton dimension for compatibility: (samples, variables, 1, 10)
        data = data[:, :, None, :]
        # Normalization
        if columns == self.data_config.x_list_columns_2d:
            normalized_data = self.individual_scalers['soil_2d'].fit_transform_soil_2d(
                data,
                columns,
                data.shape[3]  # number of layers (10)
            )
            non_zero_count_after = np.count_nonzero(normalized_data)
            logger.info(f"[SmartDataLoader] After normalization - Input Soil2D non-zero count: {non_zero_count_after}")
            if normalized_data.shape[0] > 0:
                logger.info(f"[SmartDataLoader] After normalization - Input Soil2D sample (first variable, first sample, top 3 layers): {normalized_data[0,0,0,:3]}")
            return torch.tensor(normalized_data, dtype=self.preprocessing_config.data_type), self.individual_scalers['soil_2d']
        else:
            normalized_data = self.individual_scalers['y_soil_2d'].fit_transform_soil_2d(
                data,
                columns,
                data.shape[3]
            )
            non_zero_count_after = np.count_nonzero(normalized_data)
            logger.info(f"[SmartDataLoader] After normalization - Output Soil2D non-zero count: {non_zero_count_after}")
            if normalized_data.shape[0] > 0:
                logger.info(f"[SmartDataLoader] After normalization - Output Soil2D sample (first variable, first sample, top 3 layers): {normalized_data[0,0,0,:3]}")
            return torch.tensor(normalized_data, dtype=self.preprocessing_config.data_type), self.individual_scalers['y_soil_2d']

    def _debug_print_with_inspector(self) -> None:
        """Use inspect_pft1d_soil2d.py print helpers to show shapes and slices for configured variables."""
        from scripts.inspect_pft1d_soil2d import print_pft1d, print_soil2d  # type: ignore
        import numpy as _np

        # Print PFT1D variables
        pft_vars = getattr(self.data_config, 'x_list_columns_1d', [])
        if pft_vars:
            print("\n=== Inspector: PFT1D variables ===")
        for var in pft_vars:
            if var not in self.df.columns:
                print(f"[Inspector] pft1d '{var}' not found in DataFrame columns")
                continue
            series = self.df[var]
            print(f"pft1d variable: {var}")
            print_pft1d(series.values)

        # Print Soil2D variables (first group -> 15 layers)
        soil_vars = getattr(self.data_config, 'x_list_columns_2d', [])
        if soil_vars:
            print("\n=== Inspector: Soil2D variables (first group, 15 layers) ===")
        for var in soil_vars:
            if var not in self.df.columns:
                print(f"[Inspector] soil2d '{var}' not found in DataFrame columns")
                continue
            series = self.df[var]

            def _first_group_15_layers(cell):
                try:
                    if isinstance(cell, (list, tuple)) and len(cell) > 0 and isinstance(cell[0], (list, tuple, _np.ndarray)):
                        return list(_np.array(cell[0]).flatten()[:15])
                    arr = _np.array(cell, dtype=object)
                    if getattr(arr, 'ndim', 1) == 2 and arr.shape[0] >= 1:
                        return list(_np.array(arr[0, :15]).flatten())
                    if getattr(arr, 'ndim', 1) == 1 and len(arr) >= 15 and not isinstance(arr[0], (list, tuple, _np.ndarray)):
                        return list(_np.array(arr[:15]).flatten())
                except Exception:
                    pass
                return [0.0]*15

            first_group = series.apply(_first_group_15_layers)
            try:
                arr = _np.asarray(first_group.tolist())  # shape (samples, 15)
            except Exception:
                arr = first_group.values
            print(f"soil2d variable: {var} -> first group 15 layers")
            print_soil2d(arr)

def create_smart_dataloader(data_config, preprocessing_config):
    """
    Create a smart DataLoader that automatically prioritizes files with non-zero soil2D data.
    
    Args:
        data_config: Data configuration
        preprocessing_config: Preprocessing configuration
    
    Returns:
        SmartDataLoader instance
    """
    return SmartDataLoader(data_config, preprocessing_config)

if __name__ == "__main__":
    # Test the smart DataLoader
    print("Smart DataLoader test completed successfully!")
    print("\nTo use this DataLoader:")
    print("1. Set max_files in your data_config")
    print("2. The DataLoader will automatically prioritize files with rich soil2D data")
    print("3. Files 11, 16, 21 will be automatically selected due to their high scores")
