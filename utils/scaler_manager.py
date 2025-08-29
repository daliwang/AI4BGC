#!/usr/bin/env python3
"""
Scaler Management Utility

This module provides utilities for loading saved scalers and applying inverse transformations
to normalized predictions, restoring them to their original physical units.
"""

import os
import pickle
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

logger = logging.getLogger(__name__)


class ScalerManager:
    """
    Utility class for managing scalers and applying inverse transformations.
    """
    
    def __init__(self, scalers_dir: str):
        """
        Initialize the scaler manager.
        
        Args:
            scalers_dir: Directory containing saved scalers
        """
        self.scalers_dir = Path(scalers_dir)
        self.scalers = {}
        self.scaler_metadata = {}
        self.load_scalers()
    
    def load_scalers(self):
        """Load all saved scalers from disk."""
        if not self.scalers_dir.exists():
            raise FileNotFoundError(f"Scalers directory not found: {self.scalers_dir}")
        
        # Load scaler metadata
        metadata_file = self.scalers_dir / "scaler_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.scaler_metadata = json.load(f)
            logger.info(f"Loaded scaler metadata: {list(self.scaler_metadata.keys())}")
        
        # Load each scaler
        for scaler_name in self.scaler_metadata.keys():
            scaler_file = self.scalers_dir / f"{scaler_name}_scaler.pkl"
            if scaler_file.exists():
                with open(scaler_file, 'rb') as f:
                    self.scalers[scaler_name] = pickle.load(f)
                logger.info(f"Loaded {scaler_name} scaler")
            else:
                logger.warning(f"Scaler file not found: {scaler_file}")
        
        logger.info(f"Loaded {len(self.scalers)} scalers from {self.scalers_dir}")
    
    def inverse_transform_scalar(self, normalized_data: np.ndarray, scaler_name: str = 'y_scalar') -> np.ndarray:
        """
        Apply inverse transformation to scalar predictions.
        
        Args:
            normalized_data: Normalized predictions (n_samples, n_features)
            scaler_name: Name of the scaler to use
            
        Returns:
            Denormalized predictions in original units
        """
        if scaler_name not in self.scalers:
            raise ValueError(f"Scaler '{scaler_name}' not found. Available scalers: {list(self.scalers.keys())}")
        
        scaler = self.scalers[scaler_name]
        if scaler is None:
            logger.warning(f"Scaler '{scaler_name}' is None, returning original data")
            return normalized_data
        
        try:
            denormalized = scaler.inverse_transform(normalized_data)
            logger.info(f"Applied inverse transformation to scalar data using {scaler_name}")
            return denormalized
        except Exception as e:
            logger.error(f"Failed to apply inverse transformation: {e}")
            return normalized_data
    
    def inverse_transform_pft_1d(self, normalized_data: np.ndarray, scaler_name: str = 'y_pft_1d') -> np.ndarray:
        """
        Apply inverse transformation to PFT 1D predictions.
        
        Args:
            normalized_data: Normalized predictions (n_samples, n_features, n_pfts)
            scaler_name: Name of the scaler to use
            
        Returns:
            Denormalized predictions in original units
        """
        if scaler_name not in self.scalers:
            raise ValueError(f"Scaler '{scaler_name}' not found. Available scalers: {list(self.scalers.keys())}")
        
        scaler = self.scalers[scaler_name]
        if scaler is None:
            logger.warning(f"Scaler '{scaler_name}' is None, returning original data")
            return normalized_data
        
        try:
            # Reshape for inverse transform
            original_shape = normalized_data.shape
            normalized_flat = normalized_data.reshape(normalized_data.shape[0], -1)
            denormalized_flat = scaler.inverse_transform(normalized_flat)
            denormalized = denormalized_flat.reshape(original_shape)
            logger.info(f"Applied inverse transformation to PFT 1D data using {scaler_name}")
            return denormalized
        except Exception as e:
            logger.error(f"Failed to apply inverse transformation: {e}")
            return normalized_data
    
    def inverse_transform_soil_2d(self, normalized_data: np.ndarray, scaler_name: str = 'y_soil_2d') -> np.ndarray:
        """
        Apply inverse transformation to soil 2D predictions.
        
        Args:
            normalized_data: Normalized predictions (n_samples, n_features, n_cols, n_layers)
            scaler_name: Name of the scaler to use
            
        Returns:
            Denormalized predictions in original units
        """
        if scaler_name not in self.scalers:
            raise ValueError(f"Scaler '{scaler_name}' not found. Available scalers: {list(self.scalers.keys())}")
        
        scaler = self.scalers[scaler_name]
        if scaler is None:
            logger.warning(f"Scaler '{scaler_name}' is None, returning original data")
            return normalized_data
        
        try:
            # Reshape for inverse transform
            original_shape = normalized_data.shape
            normalized_flat = normalized_data.reshape(normalized_data.shape[0], -1)
            denormalized_flat = scaler.inverse_transform(normalized_flat)
            denormalized = denormalized_flat.reshape(original_shape)
            logger.info(f"Applied inverse transformation to soil 2D data using {scaler_name}")
            return denormalized
        except Exception as e:
            logger.error(f"Failed to apply inverse transformation: {e}")
            return normalized_data
    
    def get_scaler_info(self, scaler_name: str) -> Dict[str, Any]:
        """
        Get information about a specific scaler.
        
        Args:
            scaler_name: Name of the scaler
            
        Returns:
            Dictionary containing scaler information
        """
        if scaler_name not in self.scaler_metadata:
            return {}
        
        return self.scaler_metadata[scaler_name]
    
    def list_available_scalers(self) -> list:
        """Get list of available scaler names."""
        return list(self.scalers.keys())
    
    def validate_scaler_compatibility(self, data_shape: tuple, scaler_name: str) -> bool:
        """
        Validate that data shape is compatible with a scaler.
        
        Args:
            data_shape: Shape of the data to be transformed
            scaler_name: Name of the scaler to validate against
            
        Returns:
            True if compatible, False otherwise
        """
        if scaler_name not in self.scalers:
            return False
        
        scaler = self.scalers[scaler_name]
        if scaler is None:
            return False
        
        # Check if the scaler has been fitted
        if not hasattr(scaler, 'n_features_in_'):
            return False
        
        # For 2D data, check if the flattened size matches
        if len(data_shape) > 2:
            flattened_size = np.prod(data_shape[1:])
            return flattened_size == scaler.n_features_in_
        else:
            return data_shape[1] == scaler.n_features_in_


def load_and_transform_predictions(
    predictions_dir: str,
    output_dir: str = None,
    save_denormalized: bool = True
) -> Dict[str, np.ndarray]:
    """
    Load normalized predictions and apply inverse transformation.
    
    Args:
        predictions_dir: Directory containing normalized predictions
        output_dir: Directory to save denormalized predictions (optional)
        save_denormalized: Whether to save denormalized predictions
        
    Returns:
        Dictionary containing denormalized predictions
    """
    predictions_dir = Path(predictions_dir)
    
    # Check if scalers directory exists
    scalers_dir = predictions_dir / "scalers"
    if not scalers_dir.exists():
        raise FileNotFoundError(f"Scalers directory not found: {scalers_dir}")
    
    # Initialize scaler manager
    scaler_manager = ScalerManager(str(scalers_dir))
    
    # Load normalized predictions
    denormalized_predictions = {}
    
    # Process scalar predictions
    scalar_file = predictions_dir / "predictions_scalar.csv"
    if scalar_file.exists():
        normalized_scalar = pd.read_csv(scalar_file).values
        denormalized_scalar = scaler_manager.inverse_transform_scalar(normalized_scalar)
        denormalized_predictions['scalar'] = denormalized_scalar
        
        if save_denormalized and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            scalar_cols = pd.read_csv(scalar_file).columns
            denormalized_df = pd.DataFrame(denormalized_scalar, columns=scalar_cols)
            denormalized_df.to_csv(output_dir / "predictions_scalar_denormalized.csv", index=False)
            logger.info(f"Saved denormalized scalar predictions to {output_dir}")
    
    # Process PFT 1D predictions
    pft_1d_dir = predictions_dir / "pft_1d_predictions"
    if pft_1d_dir.exists():
        pft_1d_predictions = {}
        for pred_file in pft_1d_dir.glob("predictions_*.csv"):
            var_name = pred_file.stem.replace("predictions_", "")
            normalized_pft = pd.read_csv(pred_file).values
            # Reshape to (n_samples, 1, n_pfts) for inverse transform
            normalized_pft_reshaped = normalized_pft.reshape(normalized_pft.shape[0], 1, -1)
            denormalized_pft = scaler_manager.inverse_transform_pft_1d(normalized_pft_reshaped)
            pft_1d_predictions[var_name] = denormalized_pft
        
        if pft_1d_predictions:
            denormalized_predictions['pft_1d'] = pft_1d_predictions
            
            if save_denormalized and output_dir:
                output_pft_dir = output_dir / "pft_1d_predictions_denormalized"
                output_pft_dir.mkdir(exist_ok=True)
                for var_name, predictions in pft_1d_predictions.items():
                    # Reshape back to 2D for CSV
                    predictions_2d = predictions.reshape(predictions.shape[0], -1)
                    columns = [f'{var_name}_pft{p+1}' for p in range(predictions.shape[2])]
                    denormalized_df = pd.DataFrame(predictions_2d, columns=columns)
                    denormalized_df.to_csv(output_pft_dir / f"predictions_{var_name}_denormalized.csv", index=False)
                logger.info(f"Saved denormalized PFT 1D predictions to {output_pft_dir}")
    
    # Process soil 2D predictions
    soil_2d_dir = predictions_dir / "soil_2d_predictions"
    if soil_2d_dir.exists():
        soil_2d_predictions = {}
        for pred_file in soil_2d_dir.glob("predictions_*.csv"):
            var_name = pred_file.stem.replace("predictions_", "")
            df = pd.read_csv(pred_file)
            normalized_soil = df.values
            n_samples = normalized_soil.shape[0]
            # Infer columns and layers from headers like Y_var_col{c}_layer{l}
            headers = list(df.columns)
            cols = []
            layers = []
            for h in headers:
                if "_col" in h and "_layer" in h:
                    try:
                        after_col = h.split("_col")[-1]
                        c_str, l_part = after_col.split("_layer")
                        cols.append(int(c_str))
                        layers.append(int(l_part))
                    except Exception:
                        continue
            if cols and layers:
                n_cols = max(cols)
                n_layers = max(layers)
            else:
                # Fallback to 1 column x 10 layers (first column, top 10 layers)
                n_cols, n_layers = 1, 10
            per_var = n_cols * n_layers
            if normalized_soil.shape[1] % per_var != 0:
                logger.warning(f"Unexpected soil 2D width {normalized_soil.shape[1]} not divisible by cols*layers {per_var} for {var_name}; skipping")
                continue
            normalized_soil_reshaped = normalized_soil.reshape(n_samples, 1, n_cols, n_layers)
            denormalized_soil = scaler_manager.inverse_transform_soil_2d(normalized_soil_reshaped)
            soil_2d_predictions[var_name] = denormalized_soil
        
        if soil_2d_predictions:
            denormalized_predictions['soil_2d'] = soil_2d_predictions
            
            if save_denormalized and output_dir:
                output_soil_dir = output_dir / "soil_2d_predictions_denormalized"
                output_soil_dir.mkdir(exist_ok=True)
                for var_name, predictions in soil_2d_predictions.items():
                    # Reshape back to 2D for CSV
                    predictions_2d = predictions.reshape(predictions.shape[0], -1)
                    columns = [f'{var_name}_col{c+1}_layer{l+1}' for c in range(predictions.shape[2]) for l in range(predictions.shape[3])]
                    denormalized_df = pd.DataFrame(predictions_2d, columns=columns)
                    denormalized_df.to_csv(output_soil_dir / f"predictions_{var_name}_denormalized.csv", index=False)
                logger.info(f"Saved denormalized soil 2D predictions to {output_soil_dir}")
    
    logger.info(f"Successfully processed {len(denormalized_predictions)} prediction types")
    return denormalized_predictions


def create_denormalization_report(
    predictions_dir: str,
    output_file: str = None
) -> Dict[str, Any]:
    """
    Create a comprehensive report about the denormalization process.
    
    Args:
        predictions_dir: Directory containing predictions and scalers
        output_file: File to save the report (optional)
        
    Returns:
        Dictionary containing the report
    """
    predictions_dir = Path(predictions_dir)
    
    # Load scaler metadata
    scalers_dir = predictions_dir / "scalers"
    metadata_file = scalers_dir / "scaler_metadata.json"
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"Scaler metadata not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        scaler_metadata = json.load(f)
    
    # Load original data ranges if available
    ranges_file = predictions_dir / "original_data_ranges.json"
    original_ranges = {}
    if ranges_file.exists():
        with open(ranges_file, 'r') as f:
            original_ranges = json.load(f)
    
    # Create report
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'predictions_directory': str(predictions_dir),
        'available_scalers': list(scaler_metadata.keys()),
        'scaler_details': {},
        'original_data_ranges': original_ranges,
        'recommendations': []
    }
    
    # Analyze each scaler
    for scaler_name, scaler_info in scaler_metadata.items():
        report['scaler_details'][scaler_name] = {
            'type': scaler_info.get('type', 'Unknown'),
            'n_samples_seen': scaler_info.get('n_samples_seen_', 'Unknown'),
            'n_features': len(scaler_info.get('feature_names_in_', [])) if scaler_info.get('feature_names_in_') else 'Unknown'
        }
        
        # Add specific recommendations based on scaler type
        if scaler_info.get('type') == 'MinMaxScaler':
            report['recommendations'].append(
                f"{scaler_name}: MinMaxScaler - Data normalized to [0,1] range. "
                "Inverse transformation will restore original scale."
            )
        elif scaler_info.get('type') == 'StandardScaler':
            report['recommendations'].append(
                f"{scaler_name}: StandardScaler - Data standardized to mean=0, std=1. "
                "Inverse transformation will restore original scale and units."
            )
        elif scaler_info.get('type') == 'RobustScaler':
            report['recommendations'].append(
                f"{scaler_name}: RobustScaler - Data normalized using robust statistics. "
                "Inverse transformation will restore original scale."
            )
    
    # Add general recommendations
    if not report['recommendations']:
        report['recommendations'].append(
            "No specific scaler information available. "
            "Check that scalers were properly saved during training."
        )
    
    report['recommendations'].extend([
        "Use the ScalerManager class to apply inverse transformations to new predictions.",
        "Always validate scaler compatibility before applying transformations.",
        "Save both normalized and denormalized predictions for comparison."
    ])
    
    # Save report if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Denormalization report saved to {output_file}")
    
    return report


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Scaler Management Utility")
    parser.add_argument("--predictions-dir", required=True, help="Directory containing predictions and scalers")
    parser.add_argument("--output-dir", help="Directory to save denormalized predictions")
    parser.add_argument("--create-report", action="store_true", help="Create denormalization report")
    
    args = parser.parse_args()
    
    try:
        if args.create_report:
            report = create_denormalization_report(args.predictions_dir)
            print("Denormalization Report:")
            print(json.dumps(report, indent=2))
        else:
            denormalized = load_and_transform_predictions(
                args.predictions_dir, 
                args.output_dir
            )
            print(f"Successfully processed {len(denormalized)} prediction types")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)
