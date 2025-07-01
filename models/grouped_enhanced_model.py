"""
Enhanced Grouped Model Architecture for Climate Data Prediction.

This module provides an enhanced neural network model that groups input data
into logical categories, applies separate embedding layers for each group,
and then concatenates them for transformer-based fusion with additional
transformer heads for LSTM and 1D CNN pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import logging

from config.training_config import ModelConfig

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Residual block for enhanced CNN processing."""
    
    def __init__(self, block: nn.Module):
        super(ResidualBlock, self).__init__()
        self.block = block
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x))


class GroupEmbeddingLayer(nn.Module):
    """Simple embedding layer for each data group using MLPs."""
    
    def __init__(self, input_size: int, embedding_size: int, group_name: str):
        super(GroupEmbeddingLayer, self).__init__()
        self.group_name = group_name
        self.embedding_size = embedding_size
        
        # Simple MLP embedding - more efficient than attention for small groups
        self.embedding = nn.Sequential(
            nn.Linear(input_size, embedding_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_size * 2, embedding_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(embedding_size)  # Layer normalization for stability
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply simple MLP embedding
        return self.embedding(x)


class LSTMTransformerHead(nn.Module):
    """Additional transformer head specifically for LSTM features."""
    
    def __init__(self, input_size: int, hidden_size: int = 128):
        super(LSTMTransformerHead, self).__init__()
        self.hidden_size = hidden_size
        
        # LSTM-specific transformer
        self.lstm_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_size,
                nhead=8,
                dim_feedforward=hidden_size * 2,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Output projection
        self.output_projection = nn.Linear(input_size, hidden_size)
    
    def forward(self, lstm_features: torch.Tensor) -> torch.Tensor:
        # Reshape for transformer (add sequence dimension if needed)
        if lstm_features.dim() == 2:
            lstm_features = lstm_features.unsqueeze(1)  # [batch, 1, features]
        
        # Apply LSTM-specific transformer
        transformed = self.lstm_transformer(lstm_features)
        
        # Global average pooling
        pooled = torch.mean(transformed, dim=1)  # [batch, features]
        
        # Project to output size
        output = self.output_projection(pooled)
        
        return output


class CNN1DTransformerHead(nn.Module):
    """Additional transformer head specifically for 1D CNN features."""
    
    def __init__(self, input_size: int, hidden_size: int = 128):
        super(CNN1DTransformerHead, self).__init__()
        self.hidden_size = hidden_size
        
        # 1D CNN-specific transformer
        self.cnn_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_size,
                nhead=8,
                dim_feedforward=hidden_size * 2,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Output projection
        self.output_projection = nn.Linear(input_size, hidden_size)
    
    def forward(self, cnn_features: torch.Tensor) -> torch.Tensor:
        # Reshape for transformer (add sequence dimension if needed)
        if cnn_features.dim() == 2:
            cnn_features = cnn_features.unsqueeze(1)  # [batch, 1, features]
        
        # Apply CNN-specific transformer
        transformed = self.cnn_transformer(cnn_features)
        
        # Global average pooling
        pooled = torch.mean(transformed, dim=1)  # [batch, features]
        
        # Project to output size
        output = self.output_projection(pooled)
        
        return output


class GroupedEnhancedModel(nn.Module):
    """
    Enhanced grouped model for climate data prediction with separate embedding layers.
    
    This model groups input data into logical categories and applies separate
    embedding layers for each group, then concatenates them for transformer-based
    fusion with additional transformer heads for LSTM and 1D CNN pooling.
    
    Groups:
    1. Time Series Group: Atmospheric forcing variables
    2. Static Surface Group: Geographic, PFT, soil properties
    3. Water Group: Water-related variables
    4. Temperature Group: Temperature-related variables
    5. 1D CNP Group: Carbon/Nitrogen/Phosphorus pools (1D)
    6. 2D CNP Group: Soil and litter CNP pools (2D)
    """
    
    def __init__(self, model_config: ModelConfig, data_info: Dict[str, Any]):
        """
        Initialize the grouped enhanced model.
        
        Args:
            model_config: Model configuration
            data_info: Information about the data structure
        """
        super(GroupedEnhancedModel, self).__init__()
        
        self.model_config = model_config
        self.data_info = data_info
        
        # Calculate input dimensions for each group
        self._calculate_group_input_dimensions()
        
        # Build group embedding layers
        self._build_group_embeddings()
        
        # Build LSTM for time series
        self._build_lstm_encoder()
        
        # Build CNN for 2D data
        self._build_2d_cnn_encoder()
        

        
        # Build main transformer fusion
        self._build_main_transformer_fusion()
        
        # Build output heads
        self._build_output_heads()
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Grouped enhanced model initialized with {self._count_parameters()} parameters")
    
    def _calculate_group_input_dimensions(self):
        """Calculate input dimensions for each data group."""
        # Time series group
        self.time_series_input_size = len(self.data_info['time_series_columns'])
        
        # Static surface group
        self.static_surface_input_size = len(self.data_info.get('static_surface_columns', []))
        
        # Water group
        self.water_group_input_size = len(self.data_info.get('water_group_columns', []))
        
        # Temperature group
        self.temperature_group_input_size = len(self.data_info.get('temperature_group_columns', []))
        
        # 1D CNP group
        self.cnp_1d_input_size = len(self.data_info['x_list_columns_1d']) * self.model_config.vector_length
        
        # 2D CNP group
        self.cnp_2d_input_channels = len(self.data_info['x_list_columns_2d'])
        
        logger.info(f"Group input dimensions:")
        logger.info(f"  Time Series: {self.time_series_input_size}")
        logger.info(f"  Static Surface: {self.static_surface_input_size}")
        logger.info(f"  Water Group: {self.water_group_input_size}")
        logger.info(f"  Temperature Group: {self.temperature_group_input_size}")
        logger.info(f"  1D CNP: {self.cnp_1d_input_size}")
        logger.info(f"  2D CNP: {self.cnp_2d_input_channels} channels")
    
    def _build_group_embeddings(self):
        """Build separate embedding layers for each group."""
        embedding_size = 128  # Fixed embedding size for all groups
        
        # Static surface group embedding
        if self.static_surface_input_size > 0:
            self.static_surface_embedding = GroupEmbeddingLayer(
                self.static_surface_input_size, embedding_size, "static_surface"
            )
        else:
            self.static_surface_embedding = None
        
        # Water group embedding
        if self.water_group_input_size > 0:
            self.water_group_embedding = GroupEmbeddingLayer(
                self.water_group_input_size, embedding_size, "water"
            )
        else:
            self.water_group_embedding = None
        
        # Temperature group embedding
        if self.temperature_group_input_size > 0:
            self.temperature_group_embedding = GroupEmbeddingLayer(
                self.temperature_group_input_size, embedding_size, "temperature"
            )
        else:
            self.temperature_group_embedding = None
        
        # 1D CNP group embedding
        if self.cnp_1d_input_size > 0:
            self.cnp_1d_embedding = GroupEmbeddingLayer(
                self.cnp_1d_input_size, embedding_size, "cnp_1d"
            )
        else:
            self.cnp_1d_embedding = None
    
    def _build_lstm_encoder(self):
        """Build enhanced LSTM encoder for time series data."""
        # Use a larger LSTM instead of LSTM + transformer head
        self.lstm = nn.LSTM(
            input_size=self.time_series_input_size,
            hidden_size=256,  # Larger hidden size
            num_layers=3,     # More layers
            batch_first=True,
            dropout=0.1,
            bidirectional=True  # Bidirectional for better feature capture
        )
        
        # Direct output projection (no intermediate transformer needed)
        self.lstm_projection = nn.Linear(256 * 2, 128)  # *2 for bidirectional
    
    def _build_2d_cnn_encoder(self):
        """Build enhanced CNN encoder for 2D CNP data."""
        layers = []
        in_channels = self.cnp_2d_input_channels
        
        # Enhanced CNN with residual connections
        for i, out_channels in enumerate(self.model_config.conv_channels):
            # Residual block
            residual_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels)
            )
            
            # Skip connection
            if in_channels == out_channels:
                layers.append(ResidualBlock(residual_block))
            else:
                layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    ResidualBlock(residual_block)
                ))
            
            if i < len(self.model_config.conv_channels) - 1:
                layers.append(nn.MaxPool2d(2))
            
            in_channels = out_channels
        
        # Calculate output size
        conv_output_size = self._calculate_conv_output_size()
        
        layers.extend([
            nn.Flatten(),
            nn.Linear(conv_output_size, 128)
        ])
        
        self.cnp_2d_encoder = nn.Sequential(*layers)
    
    def _calculate_conv_output_size(self) -> int:
        """Calculate the output size after convolutions."""
        h, w = self.model_config.matrix_rows, self.model_config.matrix_cols
        
        for i, out_channels in enumerate(self.model_config.conv_channels):
            if i < len(self.model_config.conv_channels) - 1:
                h = h // 2
                w = w // 2
        
        return self.model_config.conv_channels[-1] * h * w
    

    
    def _build_main_transformer_fusion(self):
        """Build main transformer fusion for all groups."""
        # Calculate total feature size from all groups
        total_features = 0
        group_sizes = []
        
        # LSTM features (after projection)
        total_features += 128
        group_sizes.append(128)
        
        # Static surface features
        if self.static_surface_embedding is not None:
            total_features += 128
            group_sizes.append(128)
        
        # Water group features
        if self.water_group_embedding is not None:
            total_features += 128
            group_sizes.append(128)
        
        # Temperature group features
        if self.temperature_group_embedding is not None:
            total_features += 128
            group_sizes.append(128)
        
        # 1D CNP features (after embedding)
        if self.cnp_1d_embedding is not None:
            total_features += 128
            group_sizes.append(128)
        
        # 2D CNP features
        total_features += 128
        group_sizes.append(128)
        
        self.total_features = total_features
        self.group_sizes = group_sizes
        
        # Main transformer fusion
        self.main_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,  # Fixed token dimension
                nhead=16,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6
        )
        
        # Feature projection to align all groups to same dimension
        self.feature_projection = nn.Linear(total_features, 128 * len(group_sizes))
    
    def _build_output_heads(self):
        """Build output heads for different prediction types."""
        fused_feature_size = 128 * len(self.group_sizes)
        
        # Scalar output head
        if self.model_config.scalar_output_size > 0:
            self.scalar_head = nn.Sequential(
                nn.Linear(fused_feature_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, self.model_config.scalar_output_size)
            )
        else:
            self.scalar_head = None
        
        # Vector output head
        self.vector_head = nn.Sequential(
            nn.Linear(fused_feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.model_config.vector_output_size * self.model_config.vector_length)
        )
        
        # Matrix output head
        self.matrix_head = nn.Sequential(
            nn.Linear(fused_feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.model_config.matrix_output_size * self.model_config.matrix_rows * self.model_config.matrix_cols)
        )
    
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
    
    def forward(self, time_series_data: torch.Tensor, static_surface_data: torch.Tensor,
                water_data: torch.Tensor, temperature_data: torch.Tensor,
                cnp_1d_data: torch.Tensor, cnp_2d_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the grouped enhanced model.
        
        Args:
            time_series_data: Time series input [batch_size, seq_len, features]
            static_surface_data: Static surface features [batch_size, static_features]
            water_data: Water group features [batch_size, water_features]
            temperature_data: Temperature group features [batch_size, temp_features]
            cnp_1d_data: 1D CNP features [batch_size, 1d_features]
            cnp_2d_data: 2D CNP features [batch_size, channels, height, width]
            
        Returns:
            Tuple of (scalar_output, vector_output, matrix_output)
        """
        batch_size = time_series_data.shape[0]
        
        # 1. Process time series with enhanced LSTM
        lstm_out, _ = self.lstm(time_series_data)
        lstm_out = lstm_out[:, -1, :]  # Take last hidden state
        lstm_projected = self.lstm_projection(lstm_out)
        
        # 3. Process static surface group
        if self.static_surface_embedding is not None:
            static_surface_embedded = self.static_surface_embedding(static_surface_data)
        else:
            static_surface_embedded = torch.zeros(batch_size, 128, device=time_series_data.device)
        
        # 4. Process water group
        if self.water_group_embedding is not None:
            water_embedded = self.water_group_embedding(water_data)
        else:
            water_embedded = torch.zeros(batch_size, 128, device=time_series_data.device)
        
        # 5. Process temperature group
        if self.temperature_group_embedding is not None:
            temperature_embedded = self.temperature_group_embedding(temperature_data)
        else:
            temperature_embedded = torch.zeros(batch_size, 128, device=time_series_data.device)
        
        # 6. Process 1D CNP group
        if self.cnp_1d_embedding is not None:
            cnp_1d_embedded = self.cnp_1d_embedding(cnp_1d_data)
        else:
            cnp_1d_embedded = torch.zeros(batch_size, 128, device=time_series_data.device)
        
        # 7. Process 2D CNP group
        cnp_2d_encoded = self.cnp_2d_encoder(cnp_2d_data)
        
        # 8. Concatenate all group features
        group_features = []
        group_features.append(lstm_projected)
        
        if self.static_surface_embedding is not None:
            group_features.append(static_surface_embedded)
        
        if self.water_group_embedding is not None:
            group_features.append(water_embedded)
        
        if self.temperature_group_embedding is not None:
            group_features.append(temperature_embedded)
        
        if self.cnp_1d_embedding is not None:
            group_features.append(cnp_1d_embedded)
        
        group_features.append(cnp_2d_encoded)
        
        # Concatenate all features
        combined_features = torch.cat(group_features, dim=1)
        
        # 9. Apply main transformer fusion
        # Project to token dimension
        projected_features = self.feature_projection(combined_features)
        
        # Reshape for transformer (each group becomes a token)
        num_groups = len(group_features)
        token_features = projected_features.view(batch_size, num_groups, 128)
        
        # Apply main transformer
        fused_tokens = self.main_transformer(token_features)
        
        # Flatten fused features
        fused_features = fused_tokens.view(batch_size, -1)
        
        # 10. Generate outputs
        scalar_output = None
        if self.scalar_head is not None:
            scalar_output = self.scalar_head(fused_features)
        
        vector_output = self.vector_head(fused_features).view(
            batch_size, self.model_config.vector_output_size, self.model_config.vector_length
        )
        
        matrix_output = self.matrix_head(fused_features).view(
            batch_size, self.model_config.matrix_output_size,
            self.model_config.matrix_rows, self.model_config.matrix_cols
        )
        
        return scalar_output, vector_output, matrix_output
    
    def predict(self, time_series_data: torch.Tensor, static_surface_data: torch.Tensor,
                water_data: torch.Tensor, temperature_data: torch.Tensor,
                cnp_1d_data: torch.Tensor, cnp_2d_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate predictions using the model.
        
        Args:
            time_series_data: Time series input
            static_surface_data: Static surface features
            water_data: Water group features
            temperature_data: Temperature group features
            cnp_1d_data: 1D CNP features
            cnp_2d_data: 2D CNP features
            
        Returns:
            Dictionary containing predictions
        """
        self.eval()
        with torch.no_grad():
            scalar_output, vector_output, matrix_output = self.forward(
                time_series_data, static_surface_data, water_data, temperature_data,
                cnp_1d_data, cnp_2d_data
            )
        
        predictions = {
            'vector_output': vector_output,
            'matrix_output': matrix_output
        }
        
        if scalar_output is not None:
            predictions['scalar_output'] = scalar_output
        
        return predictions 