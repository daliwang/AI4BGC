#!/usr/bin/env python3
"""
Pandas-based DataLoader that robustly extracts all variables per-row from DataFrame, matching original dataset construction logic.
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

class PandasDataLoader(DataLoaderIndividual):
    """
    DataLoader that robustly extracts all variables per-row from DataFrame, matching original dataset construction logic.
    """
    def _extract_scalar(self, columns: List[str]) -> np.ndarray:
        # Extract scalar variables per-row
        data = np.array([[self.df[col].iloc[i] for col in columns] for i in range(len(self.df))])
        logger.info(f"Extracted scalar data shape: {data.shape}")
        return data

    def _extract_time_series(self, columns: List[str]) -> np.ndarray:
        # Extract time series variables per-row
        data = np.array([[np.array(self.df[col].iloc[i]) for col in columns] for i in range(len(self.df))])
        logger.info(f"Extracted time series data shape: {data.shape}")
        return data

    def load_data(self) -> pd.DataFrame:
        df = super().load_data()
        # Debug: structured print of Y columns using inspector helpers
        try:
            from scripts.inspect_pft1d_soil2d import print_pft1d, print_soil2d  # type: ignore
            import numpy as _np

            # Scalars (unchanged)
            if getattr(self.data_config, 'y_list_scalar_columns', None):
                print("First Y_scalar values:", [df[col].iloc[0] for col in self.data_config.y_list_scalar_columns])

            # PFT1D
            y_pft_cols = getattr(self.data_config, 'y_list_columns_1d', [])
            if y_pft_cols:
                print("\n=== Inspector (Y): PFT1D variables ===")
            for col in y_pft_cols:
                if col not in df.columns:
                    print(f"[Inspector] Y pft1d '{col}' not found in DataFrame")
                    continue
                print(f"Y pft1d variable: {col}")
                print_pft1d(df[col].values)

            # Soil2D (use first-group 15-layer extraction like inspector)
            y_soil_cols = getattr(self.data_config, 'y_list_columns_2d', [])
            if y_soil_cols:
                print("\n=== Inspector (Y): Soil2D variables (first group, 15 layers) ===")
            for col in y_soil_cols:
                if col not in df.columns:
                    print(f"[Inspector] Y soil2d '{col}' not found in DataFrame")
                    continue
                series = df[col]
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
                    arr = _np.asarray(first_group.tolist())
                except Exception:
                    arr = first_group.values
                print(f"Y soil2d variable: {col} -> first group 15 layers")
                print_soil2d(arr)
        except Exception as e:
            print("[DEBUG] Error printing Y columns with inspector:", e)
        return df

    def _normalize_scalar_individual(self) -> Tuple[torch.Tensor, Any]:
        scalar_columns = self.data_config.x_list_scalar_columns
        logger.info(f"Normalizing scalar data with columns: {scalar_columns}")
        data = np.array([[self.df[col].iloc[i] for col in scalar_columns] for i in range(len(self.df))])
        normalized_data = self.individual_scalers['scalar'].fit_transform_scalar(data, scalar_columns)
        return torch.tensor(normalized_data, dtype=self.preprocessing_config.data_type), self.individual_scalers['scalar']

    def _normalize_y_scalar_individual(self) -> Tuple[torch.Tensor, Any]:
        y_scalar_columns = self.data_config.y_list_scalar_columns
        logger.info(f"Normalizing y_scalar data with columns: {y_scalar_columns}")
        data = np.array([[self.df[col].iloc[i] for col in y_scalar_columns] for i in range(len(self.df))])
        normalized_data = self.individual_scalers['y_scalar'].fit_transform_scalar(data, y_scalar_columns)
        tensor = torch.tensor(normalized_data, dtype=self.preprocessing_config.data_type)
        print("y_scalar_data stats:", tensor.min().item(), tensor.max().item(), tensor.mean().item())
        return tensor, self.individual_scalers['y_scalar']

    def _normalize_static(self, static_columns: List[str]) -> Tuple[torch.Tensor, Any]:
        logger.info(f"Normalizing static data with columns: {static_columns}")
        data = np.array([[self.df[col].iloc[i] for col in static_columns] for i in range(len(self.df))])
        scaler = self._get_scaler(self.preprocessing_config.static_normalization)
        static_normalized = scaler.fit_transform(data)
        return torch.tensor(static_normalized, dtype=self.preprocessing_config.data_type), scaler

    def _normalize_time_series(self) -> Tuple[torch.Tensor, Any]:
        logger.info("Normalizing time series data...")
        time_series_list = []
        for col in self.data_config.time_series_columns:
            col_data = [np.array(self.df[col].iloc[i], dtype=np.float32) if isinstance(self.df[col].iloc[i], (list, np.ndarray)) else np.zeros(self.data_config.time_series_length, dtype=np.float32) for i in range(len(self.df))]
            col_data = np.stack(col_data)
            time_series_list.append(col_data)
        time_series_data = np.stack(time_series_list, axis=-1)  # (samples, time_steps, features)
        time_series_data = np.ascontiguousarray(time_series_data)
        original_shape = time_series_data.shape
        time_series_flat = time_series_data.reshape(-1, len(self.data_config.time_series_columns))
        scaler = self._get_scaler(self.preprocessing_config.time_series_normalization)
        time_series_normalized = scaler.fit_transform(time_series_flat)
        time_series_data = time_series_normalized.reshape(original_shape)
        time_series_data = np.ascontiguousarray(time_series_data)
        return torch.tensor(time_series_data, dtype=self.preprocessing_config.data_type), scaler

    def _normalize_pft_param(self) -> Tuple[torch.Tensor, Any]:
        pft_param_columns = self.data_config.pft_param_columns
        num_params = len(pft_param_columns)
        num_pfts = 17
        logger.info(f"Normalizing pft_param data with columns: {pft_param_columns}")
        param_matrix = []
        for idx, row in self.df.iterrows():
            row_vectors = []
            for col in pft_param_columns:
                val = row[col]
                if isinstance(val, (list, np.ndarray)) and len(val) == num_pfts:
                    row_vectors.append(np.array(val, dtype=np.float32))
                else:
                    row_vectors.append(np.zeros(num_pfts, dtype=np.float32))
            row_matrix = np.stack(row_vectors, axis=0)
            param_matrix.append(row_matrix)
        param_matrix = np.stack(param_matrix, axis=0)
        flat_param_matrix = param_matrix.reshape(param_matrix.shape[0], -1)
        scaler = self._get_scaler(self.preprocessing_config.list_1d_normalization)
        flat_param_matrix_norm = scaler.fit_transform(flat_param_matrix)
        param_matrix_norm = flat_param_matrix_norm.reshape(param_matrix.shape)
        pft_param_data = torch.tensor(param_matrix_norm, dtype=self.preprocessing_config.data_type)
        return pft_param_data, scaler

    def _normalize_list_1d_individual(self, columns: List[str]) -> Tuple[torch.Tensor, Any]:
        logger.info(f"Normalizing 1D list data with columns: {columns}")
        col_data = []
        max_length_1d = 17
        for col in columns:
            standardized_samples = []
            for i in range(len(self.df)):
                val = self.df[col].iloc[i]
                arr = np.array(val)
                if arr.shape[0] >= 17:
                    pfts = arr[:17]
                else:
                    pfts = np.pad(arr, (0, 17 - arr.shape[0]), mode='constant')
                standardized_samples.append(pfts)
            col_data.append(np.stack(standardized_samples))
        data = np.stack(col_data, axis=1)  # (samples, variables, 17)
        logger.info(f"Standardized PFT1D data shape: {data.shape}")
        data = data[:, :, 1:]              # (samples, variables, 16)
        logger.info(f"After dropping PFT0: {data.shape}")
        data = np.transpose(data, (0, 2, 1))  # (samples, 16, variables)
        logger.info(f"After transpose: {data.shape}")
        # Normalization
        if columns == self.data_config.x_list_columns_1d:
            normalized_data = self.individual_scalers['pft_1d'].fit_transform_pft_1d(
                data, [f'PFT{i}' for i in range(1, 17)], columns)
            # Transpose back to (samples, variables, 16)
            normalized_data = np.transpose(normalized_data, (0, 2, 1))
            tensor = torch.tensor(normalized_data, dtype=self.preprocessing_config.data_type)
            return tensor, self.individual_scalers['pft_1d']
        else:
            normalized_data = self.individual_scalers['y_pft_1d'].fit_transform_pft_1d(
                data, [f'PFT{i}' for i in range(1, 17)], columns)
            # Transpose back to (samples, variables, 16)
            normalized_data = np.transpose(normalized_data, (0, 2, 1))
            tensor = torch.tensor(normalized_data, dtype=self.preprocessing_config.data_type)
            print("y_pft_1d_data stats:", tensor.min().item(), tensor.max().item(), tensor.mean().item())
            return tensor, self.individual_scalers['y_pft_1d']

    def _normalize_list_2d_individual(self, columns: List[str]) -> Tuple[torch.Tensor, Any]:
        logger.info(f"Normalizing 2D list data with columns: {columns}")
        col_data = []
        fallback_counts = [0] * len(columns)

        def _first_group_15_layers(cell):
            try:
                if isinstance(cell, (list, tuple)) and len(cell) > 0 and isinstance(cell[0], (list, tuple, np.ndarray)):
                    return list(np.array(cell[0], dtype=float).flatten()[:15])
                arr = np.array(cell, dtype=object)
                if getattr(arr, 'ndim', 1) == 2 and arr.shape[0] >= 1:
                    return list(np.array(arr[0, :15], dtype=float).flatten())
                if getattr(arr, 'ndim', 1) == 1 and len(arr) >= 15 and not isinstance(arr[0], (list, tuple, np.ndarray)):
                    return list(np.array(arr[:15], dtype=float).flatten())
            except Exception:
                pass
            return [0.0]*15

        for col_idx, col in enumerate(columns):
            series = self.df[col]
            first_group = series.apply(_first_group_15_layers)
            mat = np.asarray(first_group.tolist(), dtype=float)  # (samples, 15)
            if mat.ndim != 2 or mat.shape[1] < 10:
                logger.warning(f"Soil2D column {col} produced unexpected shape {mat.shape}; padding/truncating to 10")
            ten = np.zeros((len(series), 10), dtype=float)
            take = min(10, mat.shape[1] if mat.ndim == 2 else 0)
            if take > 0:
                ten[:, :take] = mat[:, :take]
            nz = int(np.count_nonzero(ten))
            if nz == 0 and take == 0:
                fallback_counts[col_idx] += 1
            col_data.append(ten)
        data = np.stack(col_data, axis=1)  # (samples, variables, 10)
        # Pre-scaler non-zero diagnostics
        try:
            per_var_nonzeros = [(columns[v], int(np.count_nonzero(data[:, v, :]))) for v in range(len(columns))]
            total_nonzeros = int(np.count_nonzero(data))
            logger.info(f"Soil2D pre-scaler non-zeros per var: {per_var_nonzeros}; total_nonzeros={total_nonzeros}; fallback_counts={fallback_counts}")
        except Exception as _e:
            logger.warning(f"Failed pre-scaler Soil2D diagnostics: {_e}")
        logger.info(f"Standardized Soil2D data shape: {data.shape}")
        data = data[:, :, None, :]
        if columns == self.data_config.x_list_columns_2d:
            normalized_data = self.individual_scalers['soil_2d'].fit_transform_soil_2d(
                data, columns, data.shape[3])
            tensor = torch.tensor(normalized_data, dtype=self.preprocessing_config.data_type)
            return tensor, self.individual_scalers['soil_2d']
        else:
            normalized_data = self.individual_scalers['y_soil_2d'].fit_transform_soil_2d(
                data, columns, data.shape[3])
            tensor = torch.tensor(normalized_data, dtype=self.preprocessing_config.data_type)
            print("y_soil_2d stats:", tensor.min().item(), tensor.max().item(), tensor.mean().item())
            return tensor, self.individual_scalers['y_soil_2d']

    # Add similar per-row extraction for water and any other groups as needed.
