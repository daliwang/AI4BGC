"""
Flexible combined model architecture for climate data prediction.

This module provides a flexible neural network model that can handle
different input and output configurations for climate data prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import logging

from config.training_config import ModelConfig
from models.cnp_combined_model import CNPCombinedModel

logger = logging.getLogger(__name__)


class CombinedModel(CNPCombinedModel):
    """
    Simple default model: CombinedModel is now just a wrapper for CNPCombinedModel.
    """
    pass


class FlexibleCombinedModel(CombinedModel):
    """
    More flexible version of CombinedModel that can handle variable input/output sizes.
    """
    
    def __init__(self, model_config: ModelConfig, data_info: Dict[str, Any], 
                 custom_output_sizes: Optional[Dict[str, int]] = None):
        """
        Initialize flexible combined model.
        
        Args:
            model_config: Model configuration
            data_info: Information about the data structure
            custom_output_sizes: Custom output sizes for different prediction types
        """
        # Override output sizes if provided
        if custom_output_sizes:
            for key, value in custom_output_sizes.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
        
        super().__init__(model_config, data_info)
    
    def add_output_head(self, name: str, output_size: int, input_size: Optional[int] = None):
        """
        Add a new output head dynamically.
        
        Args:
            name: Name of the output head
            output_size: Size of the output
            input_size: Input size (if None, uses fused feature size)
        """
        if input_size is None:
            input_size = self.token_dim * self.model_config.num_tokens
        
        # Create the new output head
        setattr(self, f'fc_{name}_branch', nn.Linear(input_size, 128))
        setattr(self, f'fc_{name}', nn.Linear(128, output_size))
        setattr(self, f'log_sigma_{name}', nn.Parameter(torch.zeros(1)))
        
        logger.info(f"Added output head '{name}' with output size {output_size}")
    
    def forward(self, time_series_data: torch.Tensor, static_data: torch.Tensor,
                list_1d_data: torch.Tensor, list_2d_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with flexible outputs.
        
        Returns:
            Dictionary containing all outputs
        """
        batch_size = time_series_data.shape[0]
        
        # Get base outputs
        scalar_output, vector_output, matrix_output = super().forward(
            time_series_data, static_data, list_1d_data, list_2d_data
        )
        
        outputs = {
            'scalar': scalar_output,
            'vector': vector_output,
            'matrix': matrix_output
        }
        
        # Add any additional output heads
        fused_features = self._get_fused_features(time_series_data, static_data, list_1d_data, list_2d_data)
        
        for name in dir(self):
            if name.startswith('fc_') and name.endswith('_branch') and name not in ['fc_scalar_branch', 'fc_vector_branch', 'fc_matrix_branch']:
                output_name = name[3:-6]  # Remove 'fc_' prefix and '_branch' suffix
                branch_layer = getattr(self, name)
                output_layer = getattr(self, f'fc_{output_name}')
                
                features = F.relu(branch_layer(fused_features))
                outputs[output_name] = output_layer(features)
        
        return outputs
    
    def _get_fused_features(self, time_series_data: torch.Tensor, static_data: torch.Tensor,
                           list_1d_data: torch.Tensor, list_2d_data: torch.Tensor) -> torch.Tensor:
        """Get fused features for additional output heads."""
        batch_size = time_series_data.shape[0]
        
        # Process time series with LSTM
        lstm_out, _ = self.lstm(time_series_data)
        lstm_out = lstm_out[:, -1, :]
        
        # Process static features
        if self.fc_static is not None:
            static_out = F.relu(self.fc_static(static_data))
        else:
            # Create zero tensor for static features
            static_out = torch.zeros(batch_size, self.model_config.static_fc_size, device=time_series_data.device)
        
        # Process 1D list features
        list_1d_out = F.relu(self.fc_1d(list_1d_data))
        
        # Process 2D list features
        list_2d_out = self.conv2d(list_2d_data)
        
        # Combine all features
        combined = torch.cat((lstm_out, static_out, list_1d_out, list_2d_out), dim=1)
        
        # Project to token dimension
        combined_projected = self.feature_projection(combined)
        
        # Reshape for transformer
        combined_tokens = combined_projected.view(batch_size, self.model_config.num_tokens, self.token_dim)
        
        # Apply transformer fusion
        fused_tokens = self.feature_fusion(combined_tokens)
        
        # Flatten fused features
        return fused_tokens.view(batch_size, -1) 