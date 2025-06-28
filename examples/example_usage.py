#!/usr/bin/env python3
"""
Example Usage of Refactored Climate Model Training Framework

This script demonstrates how to use the refactored framework with different
configurations and customizations.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from config.training_config import (
    get_default_config, get_minimal_config, get_extended_config,
    TrainingConfigManager
)
from data.data_loader import DataLoader
from models.combined_model import CombinedModel, FlexibleCombinedModel
from training.trainer import ModelTrainer


def example_1_basic_training():
    """Example 1: Basic training with default configuration."""
    print("=== Example 1: Basic Training ===")
    
    # Get default configuration
    config = get_default_config()
    
    # Initialize data loader
    data_loader = DataLoader(
        config.data_config,
        config.preprocessing_config
    )
    
    # Load and preprocess data
    df = data_loader.load_data()
    df = data_loader.preprocess_data()
    
    # Get data information
    data_info = data_loader.get_data_info()
    print(f"Data info: {data_info}")
    
    # Normalize data
    normalized_data = data_loader.normalize_data()
    
    # Split data
    split_data = data_loader.split_data(normalized_data)
    
    # Create model
    model = CombinedModel(
        config.model_config,
        data_info
    )
    
    # Initialize trainer
    trainer = ModelTrainer(
        config.training_config,
        model,
        split_data['train'],
        split_data['test'],
        normalized_data['scalers'],
        data_info
    )
    
    # Run training (commented out for demo)
    # results = trainer.run_training_pipeline()
    print("Training pipeline ready to run!")


def example_2_custom_configuration():
    """Example 2: Custom configuration for specific use case."""
    print("\n=== Example 2: Custom Configuration ===")
    
    # Create custom configuration
    config = TrainingConfigManager()
    
    # Focus on soil-related features only
    config.update_data_config(
        x_list_columns_2d=['soil3c_vr', 'soil4c_vr'],
        y_list_columns_2d=['Y_soil3c_vr', 'Y_soil4c_vr'],
        x_list_columns_1d=['deadcrootc'],
        y_list_columns_1d=['Y_deadcrootc']
    )
    
    # Use only essential time series features
    config.update_data_config(
        time_series_columns=['FLDS', 'PSRF', 'FSDS', 'TBOT']
    )
    
    # Customize model architecture
    config.update_model_config(
        lstm_hidden_size=64,
        fc_hidden_size=32,
        transformer_layers=2
    )
    
    # Customize training parameters
    config.update_training_config(
        num_epochs=20,
        batch_size=16,
        learning_rate=0.001,
        scalar_loss_weight=1.0,
        vector_loss_weight=0.8,
        matrix_loss_weight=0.8
    )
    
    print("Custom configuration created:")
    print(f"- 2D inputs: {config.data_config.x_list_columns_2d}")
    print(f"- 2D outputs: {config.data_config.y_list_columns_2d}")
    print(f"- 1D inputs: {config.data_config.x_list_columns_1d}")
    print(f"- 1D outputs: {config.data_config.y_list_columns_1d}")
    print(f"- Time series: {config.data_config.time_series_columns}")
    print(f"- Training epochs: {config.training_config.num_epochs}")


def example_3_experiment_comparison():
    """Example 3: Comparing different experiment configurations."""
    print("\n=== Example 3: Experiment Comparison ===")
    
    # Experiment 1: Soil only
    config1 = get_default_config()
    config1.update_data_config(
        x_list_columns_2d=['soil3c_vr'],
        y_list_columns_2d=['Y_soil3c_vr'],
        x_list_columns_1d=['deadcrootc'],
        y_list_columns_1d=['Y_deadcrootc']
    )
    config1.update_training_config(num_epochs=10, batch_size=8)
    
    # Experiment 2: All features
    config2 = get_default_config()
    config2.update_training_config(num_epochs=10, batch_size=8)
    
    # Experiment 3: Extended model
    config3 = get_extended_config()
    config3.update_training_config(num_epochs=10, batch_size=8)
    
    experiments = [
        ("Soil Only", config1),
        ("All Features", config2),
        ("Extended Model", config3)
    ]
    
    for name, config in experiments:
        print(f"\n{name} Experiment:")
        print(f"- 2D features: {len(config.data_config.x_list_columns_2d)}")
        print(f"- 1D features: {len(config.data_config.x_list_columns_1d)}")
        print(f"- Time series: {len(config.data_config.time_series_columns)}")
        print(f"- Model complexity: {config.model_config.lstm_hidden_size} hidden units")


def example_4_flexible_model():
    """Example 4: Using the flexible model with dynamic outputs."""
    print("\n=== Example 4: Flexible Model ===")
    
    config = get_default_config()
    
    # Create flexible model
    data_info = {
        'time_series_columns': ['FLDS', 'PSRF', 'FSDS', 'TBOT'],
        'static_columns': ['col1', 'col2'],
        'target_columns': ['Y_GPP', 'Y_NPP'],
        'x_list_columns_1d': ['deadcrootc'],
        'y_list_columns_1d': ['Y_deadcrootc'],
        'x_list_columns_2d': ['soil3c_vr'],
        'y_list_columns_2d': ['Y_soil3c_vr']
    }
    
    model = FlexibleCombinedModel(
        config.model_config,
        data_info
    )
    
    # Add custom output head
    model.add_output_head("custom_output", output_size=3)
    
    print("Flexible model created with custom output head")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")


def example_5_data_analysis():
    """Example 5: Data analysis and exploration."""
    print("\n=== Example 5: Data Analysis ===")
    
    config = get_default_config()
    
    # Initialize data loader
    data_loader = DataLoader(
        config.data_config,
        config.preprocessing_config
    )
    
    # Load data
    df = data_loader.load_data()
    
    # Get data information
    data_info = data_loader.get_data_info()
    
    print("Data Analysis Results:")
    print(f"- Total samples: {data_info['total_samples']}")
    print(f"- Available columns: {len(data_info['available_columns'])}")
    print(f"- Time series features: {len(data_info['time_series_columns'])}")
    print(f"- Static features: {len(data_info['static_columns'])}")
    print(f"- Target features: {len(data_info['target_columns'])}")
    print(f"- 1D input features: {len(data_info['x_list_columns_1d'])}")
    print(f"- 1D output features: {len(data_info['y_list_columns_1d'])}")
    print(f"- 2D input features: {len(data_info['x_list_columns_2d'])}")
    print(f"- 2D output features: {len(data_info['y_list_columns_2d'])}")
    
    # Show sample of available columns
    print(f"\nSample available columns: {data_info['available_columns'][:10]}")


def example_6_training_variations():
    """Example 6: Different training variations."""
    print("\n=== Example 6: Training Variations ===")
    
    # Variation 1: Different loss weights
    config1 = get_default_config()
    config1.update_training_config(
        scalar_loss_weight=1.0,
        vector_loss_weight=0.5,
        matrix_loss_weight=0.3
    )
    
    # Variation 2: Different learning rates
    config2 = get_default_config()
    config2.update_training_config(
        learning_rate=0.0001,
        use_scheduler=True,
        scheduler_type='cosine'
    )
    
    # Variation 3: Early stopping
    config3 = get_default_config()
    config3.update_training_config(
        use_early_stopping=True,
        patience=5,
        min_delta=0.001
    )
    
    variations = [
        ("Weighted Loss", config1),
        ("Cosine Scheduler", config2),
        ("Early Stopping", config3)
    ]
    
    for name, config in variations:
        print(f"\n{name} Configuration:")
        print(f"- Learning rate: {config.training_config.learning_rate}")
        print(f"- Loss weights: S={config.training_config.scalar_loss_weight}, "
              f"V={config.training_config.vector_loss_weight}, "
              f"M={config.training_config.matrix_loss_weight}")
        print(f"- Early stopping: {config.training_config.use_early_stopping}")
        print(f"- Scheduler: {config.training_config.scheduler_type if config.training_config.use_scheduler else 'None'}")


def main():
    """Run all examples."""
    print("Climate Model Training Framework - Example Usage")
    print("=" * 50)
    
    try:
        example_1_basic_training()
        example_2_custom_configuration()
        example_3_experiment_comparison()
        example_4_flexible_model()
        example_5_data_analysis()
        example_6_training_variations()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nTo run actual training, use:")
        print("python model_training_refactored.py default")
        print("python model_training_refactored.py minimal")
        print("python model_training_refactored.py extended")
        print("python model_training_refactored.py custom")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure your data files are available and paths are correct.")


if __name__ == "__main__":
    main() 