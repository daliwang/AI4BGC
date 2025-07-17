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

logger = logging.getLogger(__name__)


class CombinedModel(nn.Module):
    """
    Flexible combined model for climate data prediction.
    
    This model combines LSTM for time series, CNN for 2D data,
    and fully connected layers for static and 1D data.
    """
    
    def __init__(self, model_config: ModelConfig, data_info: Dict[str, Any], 
                 actual_1d_size: Optional[int] = None, actual_2d_channels: Optional[int] = None):
        """
        Initialize the combined model.
        
        Args:
            model_config: Model configuration
            data_info: Information about the data structure
            actual_1d_size: Actual size of 1D input data (if different from config)
            actual_2d_channels: Actual number of 2D channels (if different from config)
        """
        super(CombinedModel, self).__init__()
        
        self.model_config = model_config
        self.data_info = data_info
        self.actual_1d_size = actual_1d_size
        self.actual_2d_channels = actual_2d_channels
        
        # Calculate input dimensions
        self._calculate_input_dimensions()
        
        # Build model components
        self._build_lstm()
        self._build_static_encoder()
        self._build_1d_encoder()
        self._build_2d_encoder()
        self._build_feature_fusion()
        self._build_output_heads()
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Model initialized with {self._count_parameters()} parameters")
    
    def _calculate_input_dimensions(self):
        """Calculate input dimensions based on data info."""
        # Time series input size
        self.lstm_input_size = len(self.data_info['time_series_columns'])
        # Static input size
        self.static_input_size = len(self.data_info['static_columns'])
        # 1D input size - use actual size if provided, otherwise calculate from config
        if self.actual_1d_size is not None:
            self.list_1d_input_size = self.actual_1d_size
            logger.info(f"Using actual 1D input size: {self.actual_1d_size}")
        else:
            self.list_1d_input_size = len(self.data_info.get('variables_1d_pft', [])) * self.model_config.vector_length
        # 2D input size - use actual channels if provided, otherwise calculate from config
        if self.actual_2d_channels is not None:
            self.list_2d_input_channels = self.actual_2d_channels
            logger.info(f"Using actual 2D channels: {self.actual_2d_channels}")
        else:
            self.list_2d_input_channels = len(self.data_info['x_list_columns_2d'])
        # 2D total size
        self.list_2d_input_size = self.list_2d_input_channels * self.model_config.matrix_rows * self.model_config.matrix_cols
        logger.info(f"Input dimensions - LSTM: {self.lstm_input_size}, Static: {self.static_input_size}, "
                   f"1D: {self.list_1d_input_size}, 2D: {self.list_2d_input_size} (channels: {self.list_2d_input_channels})")
        # Log the actual data structure for debugging
        logger.info(f"1D columns: {self.data_info.get('variables_1d_pft', [])}")
        logger.info(f"2D columns: {self.data_info['x_list_columns_2d']}")
        logger.info(f"Vector length config: {self.model_config.vector_length}")
    
    def _build_lstm(self):
        """Build LSTM component for time series processing."""
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.model_config.lstm_hidden_size,
            batch_first=True
        )
    
    def _build_static_encoder(self):
        """Build encoder for static features."""
        if self.static_input_size > 0:
            self.fc_static = nn.Linear(self.static_input_size, self.model_config.static_fc_size)
        else:
            # If no static features, create a dummy layer that outputs zeros
            self.fc_static = None
            logger.warning("No static features found. Static encoder will output zeros.")
    
    def _build_1d_encoder(self):
        """Build encoder for 1D list features."""
        self.fc_1d = nn.Linear(self.list_1d_input_size, self.model_config.fc_hidden_size)
    
    def _build_2d_encoder(self):
        """Build CNN encoder for 2D list features."""
        layers = []
        in_channels = self.list_2d_input_channels
        
        for i, out_channels in enumerate(self.model_config.conv_channels):
            layers.extend([
                nn.Conv2d(
                    in_channels, out_channels, 
                    kernel_size=self.model_config.conv_kernel_size,
                    padding=self.model_config.conv_padding
                ),
                nn.ReLU()
            ])
            
            if i < len(self.model_config.conv_channels) - 1:
                layers.append(nn.MaxPool2d(2))
            
            in_channels = out_channels
        
        # Calculate the size after convolutions and pooling
        conv_output_size = self._calculate_conv_output_size()
        
        layers.extend([
            nn.Flatten(),
            nn.Linear(conv_output_size, self.model_config.fc_hidden_size)
        ])
        
        self.conv2d = nn.Sequential(*layers)
    
    def _calculate_conv_output_size(self) -> int:
        """Calculate the output size after convolutions."""
        # Start with input size
        h, w = self.model_config.matrix_rows, self.model_config.matrix_cols
        
        # Apply convolutions and pooling
        for i, out_channels in enumerate(self.model_config.conv_channels):
            # Convolution doesn't change size due to padding
            if i < len(self.model_config.conv_channels) - 1:
                # MaxPool2d reduces size by 2
                h = h // 2
                w = w // 2
        
        return self.model_config.conv_channels[-1] * h * w
    
    def _build_feature_fusion(self):
        """Build feature fusion component using transformer."""
        # Calculate total feature size after encoding
        total_features = (
            self.model_config.lstm_hidden_size +
            self.model_config.static_fc_size +
            self.model_config.fc_hidden_size +
            self.model_config.fc_hidden_size
        )
        
        self.token_dim = total_features // self.model_config.num_tokens
        
        # Ensure token_dim is divisible by number of heads
        if self.token_dim % self.model_config.transformer_heads != 0:
            self.token_dim = (self.token_dim // self.model_config.transformer_heads) * self.model_config.transformer_heads
        
        # Adjust total_features to be divisible by num_tokens
        total_features = self.token_dim * self.model_config.num_tokens
        
        # Feature projection to match token_dim
        self.feature_projection = nn.Linear(
            self.model_config.lstm_hidden_size + self.model_config.static_fc_size + 
            2 * self.model_config.fc_hidden_size, 
            total_features
        )
        
        # Transformer encoder
        self.feature_fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.token_dim, 
                nhead=self.model_config.transformer_heads, 
                batch_first=True
            ),
            num_layers=self.model_config.transformer_layers
        )
    
    def _build_output_heads(self):
        """Build output heads for different prediction types."""
        # Calculate fused feature size
        fused_feature_size = self.token_dim * self.model_config.num_tokens
        
        # Scalar output head (only if scalar_output_size > 0)
        if self.model_config.scalar_output_size > 0:
            self.fc_scalar_branch = nn.Linear(fused_feature_size, 128)
            self.fc_scalar = nn.Linear(128, self.model_config.scalar_output_size)
            self.log_sigma_scalar = nn.Parameter(torch.zeros(1))
        else:
            self.fc_scalar_branch = None
            self.fc_scalar = None
            self.log_sigma_scalar = None
            logger.warning("No scalar outputs configured. Scalar output head will be disabled.")
        
        # Vector output head (1D predictions)
        self.fc_vector_branch = nn.Linear(fused_feature_size, 128)
        self.fc_vector = nn.Linear(128, self.model_config.vector_output_size * self.model_config.vector_length)
        
        # Matrix output head (2D predictions)
        self.fc_matrix_branch = nn.Linear(fused_feature_size, 128)
        self.fc_matrix = nn.Linear(
            128, 
            self.model_config.matrix_output_size * self.model_config.matrix_rows * self.model_config.matrix_cols
        )
        
        # Learnable loss weights
        self.log_sigma_vector = nn.Parameter(torch.zeros(1))
        self.log_sigma_matrix = nn.Parameter(torch.zeros(1))
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def _count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, time_series_data: torch.Tensor, static_data: torch.Tensor, 
                list_1d_data: torch.Tensor, list_2d_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            time_series_data: Time series input [batch_size, seq_len, features]
            static_data: Static features [batch_size, static_features]
            list_1d_data: 1D list features [batch_size, 1d_features]
            list_2d_data: 2D list features [batch_size, channels, height, width]
            
        Returns:
            Tuple of (scalar_output, vector_output, matrix_output)
        """
        batch_size = time_series_data.shape[0]
        
        # Process time series with LSTM
        lstm_out, _ = self.lstm(time_series_data)
        lstm_out = lstm_out[:, -1, :]  # Take last hidden state
        
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
        combined_fused = fused_tokens.view(batch_size, -1)
        
        # Generate outputs
        if self.fc_scalar is not None:
            scalar_features = F.relu(self.fc_scalar_branch(combined_fused))
            scalar_output = self.fc_scalar(scalar_features)
        else:
            # Create dummy scalar output if disabled
            scalar_output = torch.zeros(batch_size, 1, device=time_series_data.device)
        
        vector_features = F.relu(self.fc_vector_branch(combined_fused))
        matrix_features = F.relu(self.fc_matrix_branch(combined_fused))
        
        vector_output = F.softplus(self.fc_vector(vector_features)).view(
            batch_size, self.model_config.vector_output_size, self.model_config.vector_length
        )
        matrix_output = F.softplus(self.fc_matrix(matrix_features)).view(
            batch_size, self.model_config.matrix_output_size, 
            self.model_config.matrix_rows, self.model_config.matrix_cols
        )
        
        return scalar_output, vector_output, matrix_output
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        weights = {}
        
        if self.log_sigma_scalar is not None:
            weights['scalar'] = (1 / (2 * torch.exp(self.log_sigma_scalar)**2)).item()
        
        weights['vector'] = (1 / (2 * torch.exp(self.log_sigma_vector)**2)).item()
        weights['matrix'] = (1 / (2 * torch.exp(self.log_sigma_matrix)**2)).item()
        
        return weights
    
    def predict(self, time_series_data: torch.Tensor, static_data: torch.Tensor,
                list_1d_data: torch.Tensor, list_2d_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make predictions with the model.
        
        Args:
            time_series_data: Time series input
            static_data: Static features
            list_1d_data: 1D list features
            list_2d_data: 2D list features
            
        Returns:
            Dictionary containing predictions
        """
        self.eval()
        with torch.no_grad():
            scalar_pred, vector_pred, matrix_pred = self.forward(
                time_series_data, static_data, list_1d_data, list_2d_data
            )
        
        return {
            'scalar': scalar_pred,
            'vector': vector_pred,
            'matrix': matrix_pred
        }


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