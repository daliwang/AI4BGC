"""
Enhanced combined model architecture for dataset3 with reorganized input groups.

This module provides an enhanced neural network model with reorganized input groups:
- Time Series: Atmospheric forcing variables (unchanged)
- Static Surface: Geographic, PFT, soil properties, and auxiliary variables
- Water Group: Separate processing for water-related variables
- Temperature Group: Separate processing for temperature-related variables
- 1D CNP Pools: Extended 1D list processing for carbon/nitrogen/phosphorus pools
- 2D CNP Pools: Extended 2D list processing for soil and litter CNP pools
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import logging

from config.training_config import ModelConfig

logger = logging.getLogger(__name__)


class EnhancedCombinedModel(nn.Module):
    """
    Enhanced combined model for dataset3 climate data prediction with reorganized input groups.
    
    This model is specifically designed for dataset3's rich feature set with separate processing:
    - Time Series: Atmospheric forcing (FLDS, PSRF, FSDS, QBOT, PRECTmms, TBOT)
    - Static Surface: Geographic, PFT, soil properties, and auxiliary variables
    - Water Group: Water variables (H2OCAN, H2OSFC, H2OSNO, H2OSOI_LIQ, H2OSOI_ICE, H2OSOI_10CM)
    - Temperature Group: Temperature variables (T_VEG, T10_VALUE, TH2OSFC, T_GRND, etc.)
    - 1D CNP Pools: Carbon/Nitrogen/Phosphorus pools (deadcrootc, deadstemc, tlai, leafc, frootc, totlitc)
    - 2D CNP Pools: Soil and litter CNP pools (soil1-4c_vr, litr1-3c_vr)
    """
    
    def __init__(self, model_config: ModelConfig, data_info: Dict[str, Any]):
        """
        Initialize the enhanced combined model with reorganized input groups.
        
        Args:
            model_config: Model configuration
            data_info: Information about the data structure
        """
        super(EnhancedCombinedModel, self).__init__()
        
        self.model_config = model_config
        self.data_info = data_info
        
        # Calculate input dimensions for reorganized groups
        self._calculate_reorganized_input_dimensions()
        
        # Build reorganized model components
        self._build_time_series_encoder()
        self._build_static_surface_encoder()
        self._build_water_group_encoder()
        self._build_temperature_group_encoder()
        self._build_1d_cnp_encoder()
        self._build_2d_cnp_encoder()
        self._build_enhanced_feature_fusion()
        self._build_enhanced_output_heads()
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Enhanced reorganized model initialized with {self._count_parameters()} parameters")
    
    def _calculate_reorganized_input_dimensions(self):
        """Calculate input dimensions for reorganized input groups."""
        # Time series input size (unchanged)
        self.time_series_input_size = len(self.data_info['time_series_columns'])
        
        # Reorganized static surface input size
        self.static_surface_input_size = len(self.data_info['static_surface_columns'])
        
        # Water group input size
        self.water_group_input_size = len(self.data_info['water_group_columns'])
        
        # Temperature group input size
        self.temperature_group_input_size = len(self.data_info['temperature_group_columns'])
        
        # Enhanced 1D CNP input size
        self.list_1d_cnp_input_size = len(self.data_info['x_list_columns_1d']) * self.model_config.vector_length
        
        # Enhanced 2D CNP input size
        self.list_2d_cnp_input_channels = len(self.data_info['x_list_columns_2d'])
        self.list_2d_cnp_input_size = self.list_2d_cnp_input_channels * self.model_config.matrix_rows * self.model_config.matrix_cols
        
        logger.info(f"Reorganized input dimensions:")
        logger.info(f"  - Time Series: {self.time_series_input_size}")
        logger.info(f"  - Static Surface: {self.static_surface_input_size}")
        logger.info(f"  - Water Group: {self.water_group_input_size}")
        logger.info(f"  - Temperature Group: {self.temperature_group_input_size}")
        logger.info(f"  - 1D CNP: {self.list_1d_cnp_input_size}")
        logger.info(f"  - 2D CNP: {self.list_2d_cnp_input_size} (channels: {self.list_2d_cnp_input_channels})")
        
        # Log the reorganized data structure
        logger.info(f"Time series columns: {self.data_info['time_series_columns']}")
        logger.info(f"Static surface columns: {self.data_info['static_surface_columns']}")
        logger.info(f"Water group columns: {self.data_info['water_group_columns']}")
        logger.info(f"Temperature group columns: {self.data_info['temperature_group_columns']}")
        logger.info(f"1D CNP columns: {self.data_info['x_list_columns_1d']}")
        logger.info(f"2D CNP columns: {self.data_info['x_list_columns_2d']}")
    
    def _build_time_series_encoder(self):
        """Build time series encoder (unchanged from original)."""
        self.time_series_lstm = nn.LSTM(
            input_size=self.time_series_input_size,
            hidden_size=self.model_config.lstm_hidden_size,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism for time series
        self.time_series_attention = nn.MultiheadAttention(
            embed_dim=self.model_config.lstm_hidden_size * 2,  # Bidirectional
            num_heads=8,
            batch_first=True
        )
        
        # Final LSTM output projection
        self.time_series_projection = nn.Linear(
            self.model_config.lstm_hidden_size * 2,  # Bidirectional
            self.model_config.lstm_hidden_size
        )
    
    def _build_static_surface_encoder(self):
        """Build static surface encoder for geographic, PFT, soil, and auxiliary variables."""
        if self.static_surface_input_size > 0:
            # Group static surface features by category
            self._define_static_surface_groups()
            
            # Process each group separately
            self.static_surface_encoders = nn.ModuleDict()
            for group_name, group_size in self.static_surface_groups.items():
                self.static_surface_encoders[group_name] = nn.Sequential(
                    nn.Linear(group_size, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 64)
                )
            
            # Combine all group features
            total_group_features = len(self.static_surface_groups) * 64
            self.static_surface_combiner = nn.Sequential(
                nn.Linear(total_group_features, self.model_config.static_fc_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
            self.static_surface_encoders = None
            self.static_surface_combiner = None
            logger.warning("No static surface features found.")
    
    def _define_static_surface_groups(self):
        """Define groups for static surface features."""
        static_surface_columns = self.data_info['static_surface_columns']
        
        # Define feature groups
        self.static_surface_groups = {}
        
        # Geographic features
        geo_features = [col for col in static_surface_columns if col in ['lat', 'lon', 'area', 'landfrac']]
        if geo_features:
            self.static_surface_groups['geographic'] = len(geo_features)
        
        # PFT features
        pft_features = [col for col in static_surface_columns if 'PFT' in col or 'PCT_NAT' in col]
        if pft_features:
            self.static_surface_groups['pft'] = len(pft_features)
        
        # Soil features
        soil_features = [col for col in static_surface_columns if 'CLAY' in col or 'SAND' in col or 'SCALARAVG' in col or 'SOIL_' in col]
        if soil_features:
            self.static_surface_groups['soil'] = len(soil_features)
        
        # Auxiliary features (other surface properties)
        aux_features = [col for col in static_surface_columns if col not in sum([list(self.static_surface_groups.keys())], [])]
        if aux_features:
            self.static_surface_groups['auxiliary'] = len(aux_features)
        
        logger.info(f"Static surface feature groups: {self.static_surface_groups}")
    
    def _build_water_group_encoder(self):
        """Build dedicated encoder for water group variables."""
        if self.water_group_input_size > 0:
            self.water_encoder = nn.Sequential(
                nn.Linear(self.water_group_input_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, self.model_config.fc_hidden_size)
            )
        else:
            self.water_encoder = None
            logger.warning("No water group features found.")
    
    def _build_temperature_group_encoder(self):
        """Build dedicated encoder for temperature group variables."""
        if self.temperature_group_input_size > 0:
            self.temperature_encoder = nn.Sequential(
                nn.Linear(self.temperature_group_input_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, self.model_config.fc_hidden_size)
            )
        else:
            self.temperature_encoder = None
            logger.warning("No temperature group features found.")
    
    def _build_1d_cnp_encoder(self):
        """Build enhanced 1D CNP encoder with attention mechanism."""
        self.fc_1d_cnp = nn.Sequential(
            nn.Linear(self.list_1d_cnp_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.model_config.fc_hidden_size)
        )
        
        # Attention mechanism for 1D CNP features
        self.attention_1d_cnp = nn.MultiheadAttention(
            embed_dim=self.model_config.fc_hidden_size,
            num_heads=4,
            batch_first=True
        )
    
    def _build_2d_cnp_encoder(self):
        """Build enhanced 2D CNP encoder with residual connections."""
        layers = []
        in_channels = self.list_2d_cnp_input_channels
        
        # Enhanced CNN with residual connections for CNP pools
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
        
        # Calculate the size after convolutions
        conv_output_size = self._calculate_enhanced_conv_output_size()
        
        layers.extend([
            nn.Flatten(),
            nn.Linear(conv_output_size, self.model_config.fc_hidden_size)
        ])
        
        self.conv2d_cnp = nn.Sequential(*layers)
    
    def _calculate_enhanced_conv_output_size(self) -> int:
        """Calculate the output size after enhanced convolutions."""
        h, w = self.model_config.matrix_rows, self.model_config.matrix_cols
        
        for i, out_channels in enumerate(self.model_config.conv_channels):
            if i < len(self.model_config.conv_channels) - 1:
                h = h // 2
                w = w // 2
        
        return self.model_config.conv_channels[-1] * h * w
    
    def _build_enhanced_feature_fusion(self):
        """Build enhanced feature fusion with hierarchical attention for all 6 groups."""
        # Calculate total feature size from all 6 groups
        total_features = (
            self.model_config.lstm_hidden_size +  # Time series
            self.model_config.static_fc_size +    # Static surface
            self.model_config.fc_hidden_size +    # Water group
            self.model_config.fc_hidden_size +    # Temperature group
            self.model_config.fc_hidden_size +    # 1D CNP
            self.model_config.fc_hidden_size      # 2D CNP
        )
        
        # Enhanced transformer with larger capacity
        self.token_dim = 128  # Fixed token dimension for better stability
        self.num_tokens = total_features // self.token_dim
        
        # Ensure proper divisibility
        if total_features % self.token_dim != 0:
            self.num_tokens = (total_features // self.token_dim) + 1
            total_features = self.num_tokens * self.token_dim
        
        # Feature projection for all 6 groups
        self.feature_projection = nn.Linear(
            self.model_config.lstm_hidden_size + self.model_config.static_fc_size + 
            4 * self.model_config.fc_hidden_size,  # 4 groups: water, temp, 1D CNP, 2D CNP
            total_features
        )
        
        # Enhanced transformer encoder
        self.feature_fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.token_dim,
                nhead=16,  # More attention heads
                dim_feedforward=512,  # Larger feedforward
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6  # More transformer layers
        )
    
    def _build_enhanced_output_heads(self):
        """Build enhanced output heads with more flexible configurations."""
        fused_feature_size = self.token_dim * self.num_tokens
        
        # Enhanced scalar output head
        self.scalar_head = nn.Sequential(
            nn.Linear(fused_feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.model_config.scalar_output_size)
        )
        
        # Enhanced vector output head
        self.vector_head = nn.Sequential(
            nn.Linear(fused_feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.model_config.vector_output_size * self.model_config.vector_length)
        )
        
        # Enhanced matrix output head
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
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def _count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, time_series_data: torch.Tensor, static_surface_data: torch.Tensor, 
                water_data: torch.Tensor, temperature_data: torch.Tensor,
                list_1d_cnp_data: torch.Tensor, list_2d_cnp_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the enhanced reorganized model.
        
        Args:
            time_series_data: Time series input (batch_size, time_steps, features)
            static_surface_data: Static surface features input (batch_size, static_surface_features)
            water_data: Water group features input (batch_size, water_features)
            temperature_data: Temperature group features input (batch_size, temperature_features)
            list_1d_cnp_data: 1D CNP features input (batch_size, 1d_cnp_features)
            list_2d_cnp_data: 2D CNP features input (batch_size, channels, height, width)
            
        Returns:
            Tuple of (scalar_output, vector_output, matrix_output)
        """
        batch_size = time_series_data.size(0)
        
        # 1. Time Series Processing (unchanged)
        time_series_out, _ = self.time_series_lstm(time_series_data)
        time_series_out, _ = self.time_series_attention(time_series_out, time_series_out, time_series_out)
        time_series_out = self.time_series_projection(time_series_out.mean(dim=1))  # Global average pooling
        
        # 2. Static Surface Processing with grouped features
        if self.static_surface_encoders is not None:
            static_surface_features = []
            start_idx = 0
            
            for group_name, group_size in self.static_surface_groups.items():
                group_data = static_surface_data[:, start_idx:start_idx + group_size]
                group_features = self.static_surface_encoders[group_name](group_data)
                static_surface_features.append(group_features)
                start_idx += group_size
            
            static_surface_features = torch.cat(static_surface_features, dim=1)
            static_surface_out = self.static_surface_combiner(static_surface_features)
        else:
            static_surface_out = torch.zeros(batch_size, self.model_config.static_fc_size, device=time_series_data.device)
        
        # 3. Water Group Processing
        if self.water_encoder is not None:
            water_out = self.water_encoder(water_data)
        else:
            water_out = torch.zeros(batch_size, self.model_config.fc_hidden_size, device=time_series_data.device)
        
        # 4. Temperature Group Processing
        if self.temperature_encoder is not None:
            temperature_out = self.temperature_encoder(temperature_data)
        else:
            temperature_out = torch.zeros(batch_size, self.model_config.fc_hidden_size, device=time_series_data.device)
        
        # 5. 1D CNP Processing with attention
        list_1d_cnp_out = self.fc_1d_cnp(list_1d_cnp_data)
        list_1d_cnp_out = list_1d_cnp_out.unsqueeze(1)  # Add sequence dimension for attention
        list_1d_cnp_out, _ = self.attention_1d_cnp(list_1d_cnp_out, list_1d_cnp_out, list_1d_cnp_out)
        list_1d_cnp_out = list_1d_cnp_out.squeeze(1)
        
        # 6. 2D CNP Processing
        list_2d_cnp_out = self.conv2d_cnp(list_2d_cnp_data)
        
        # 7. Enhanced Feature Fusion for all 6 groups
        combined_features = torch.cat([
            time_series_out, static_surface_out, water_out, 
            temperature_out, list_1d_cnp_out, list_2d_cnp_out
        ], dim=1)
        projected_features = self.feature_projection(combined_features)
        
        # Reshape for transformer
        token_features = projected_features.view(batch_size, self.num_tokens, self.token_dim)
        fused_features = self.feature_fusion(token_features)
        fused_features = fused_features.view(batch_size, -1)
        
        # 8. Enhanced Output Heads
        scalar_output = self.scalar_head(fused_features)
        vector_output = self.vector_head(fused_features).view(
            batch_size, self.model_config.vector_output_size, self.model_config.vector_length
        )
        matrix_output = self.matrix_head(fused_features).view(
            batch_size, self.model_config.matrix_output_size, 
            self.model_config.matrix_rows, self.model_config.matrix_cols
        )
        
        return scalar_output, vector_output, matrix_output
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get loss weights for different output types."""
        return {
            'scalar': self.model_config.scalar_loss_weight,
            'vector': self.model_config.vector_loss_weight,
            'matrix': self.model_config.matrix_loss_weight
        }


class ResidualBlock(nn.Module):
    """Residual block for enhanced 2D processing."""
    
    def __init__(self, layers: nn.Module):
        super(ResidualBlock, self).__init__()
        self.layers = layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.layers(x)) 