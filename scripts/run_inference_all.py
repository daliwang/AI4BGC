#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
import textwrap
import shutil

import torch
from netCDF4 import Dataset

# Project imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.training_config import get_cnp_combined_config, parse_cnp_io_list
from data.data_loader import DataLoader
from models.cnp_combined_model import CNPCombinedModel
from training.trainer import ModelTrainer


def run_inference_all(
    variable_list_path: str,
    model_path: str,
    data_paths: List[str],
    file_pattern: str,
    device: str,
    output_dir: Path,
) -> Path:
    """Run inference using the trained model on the ENTIRE dataset (train_split=0.0).
    Returns predictions_dir path.
    """
    # For inference on entire dataset, we need to disable data filtering
    # Create a custom config that doesn't apply the H2OSOI_10CM filter
    config = get_cnp_combined_config(
        use_trendy1=True,
        use_trendy05=False,
        max_files=None,
        include_water=False,
        variable_list_path=variable_list_path,
    )
    
    # Explicitly disable the filter column to preserve all samples
    config.data_config.filter_column = None
    print(f"  Filter column disabled: {config.data_config.filter_column}")
    
    # Configure data and device - use entire dataset for inference
    # Convert single data_paths string to list if needed
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    config.update_data_config(data_paths=data_paths, file_pattern=file_pattern, train_split=0.0)
    config.update_training_config(device=device, predictions_dir=str(output_dir / "cnp_predictions"))

    print(f"Running inference on entire dataset...")
    print(f"  Data paths: {data_paths}")
    print(f"  Model: {model_path}")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")

    # Load and preprocess data
    loader = DataLoader(config.data_config, config.preprocessing_config)
    loader.load_data()
    print(f"  Total samples loaded: {len(loader.df)}")
    loader.preprocess_data()
    print(f"  Samples after preprocessing: {len(loader.df)}")
    data_info = loader.get_data_info()
    print(f"  Starting normalization...")
    normalized = loader.normalize_data()
    print(f"  Samples after normalization: {len(normalized['time_series_data'])}")
    print(f"  Normalized data shapes:")
    print(f"    - time_series_data: {normalized['time_series_data'].shape}")
    print(f"    - static_data: {normalized['static_data'].shape}")
    print(f"    - scalar_data: {normalized['scalar_data'].shape}")
    print(f"  Starting data split...")
    split = loader.split_data(normalized)
    
    # With train_split=0.0, all data goes to test set
    train_data = split['train']
    test_data = split['test']
    
    print(f"  Dataset split: {len(train_data)} train, {len(test_data)} test samples")
    print(f"  Actual sample counts:")
    print(f"    - Train samples: {train_data['time_series'].shape[0] if 'time_series' in train_data else 0}")
    print(f"    - Test samples: {test_data['time_series'].shape[0] if 'time_series' in test_data else 0}")
    print(f"  Total samples for inference: {test_data['time_series'].shape[0] if 'time_series' in test_data else 0}")
    print(f"  Split details:")
    print(f"    - train_size: {split['train_size']}")
    print(f"    - test_size: {split['test_size']}")
    print(f"    - Total in split: {split['train_size'] + split['test_size']}")
    
    scalers = normalized['scalers']

    # Build model and load weights
    model = CNPCombinedModel(config.model_config, data_info, include_water=False,
                             use_learnable_loss_weights=config.training_config.use_learnable_loss_weights)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)

    # Evaluate and save predictions (uses test_data which contains all data)
    trainer = ModelTrainer(config.training_config, model, train_data, test_data, scalers, data_info)
    predictions, metrics = trainer.evaluate()
    trainer.save_results(predictions, metrics)

    print(f"Inference complete! Results saved to: {config.training_config.predictions_dir}")
    return Path(config.training_config.predictions_dir)


def main():
    examples = textwrap.dedent(
        """
        Examples:
          # Run inference on entire dataset for map comparison
          python scripts/run_inference_all.py \
            --variable-list CNP_IO_list_general.txt \
            --model-path cnp_results/run_20250807_215454/cnp_predictions/model.pth \
            --data-paths /path/to/Trendy_1_data_CNP/enhanced_dataset \
            --file-pattern 'enhanced*1_training_data_batch_*.pkl' \
            --device cuda \
            --out-dir my_inference_results

          # Use default output directory (data_path_basename_inference)
          python scripts/run_inference_all.py \
            --variable-list CNP_IO_list_general.txt \
            --model-path cnp_results/run_20250807_215454/cnp_predictions/model.pth \
            --data-paths /path/to/Trendy_1_data_CNP/enhanced_dataset \
            --device cuda
        """
    )

    p = argparse.ArgumentParser(
        description='Run CNP model inference on the ENTIRE dataset for map comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples
    )
    p.add_argument('--variable-list', required=True, help='Path to CNP_IO_list_general.txt')
    p.add_argument('--model-path', default='./cnp_predictions/model.pth', help='Path to trained model state_dict (model.pth) for inference')
    p.add_argument('--data-paths', default='/mnt/proj-shared/AI4BGC_7xw/TrainingData/Trendy_1_data_CNP/enhanced_dataset/', help='Data directories containing PKL batches for inference')
    p.add_argument('--file-pattern', default='enhanced_*1_training_data_batch_*.pkl', help='Glob pattern for PKL files')
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device for inference')
    p.add_argument('--out-dir', default='cnp_inference_entire_dataset', help='Directory to store inference outputs; defaults to cnp_inference_entire_dataset')
    p.add_argument('--examples', action='store_true', help='Show example usage and exit')
    args = p.parse_args()

    if args.examples:
        print(p.format_help())
        sys.exit(0)

    # Determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        # Use default directory
        out_dir = Path('cnp_inference_entire_dataset')
    
    print(f"Output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference on entire dataset
    predictions_dir = run_inference_all(
        variable_list_path=args.variable_list,
        model_path=args.model_path,
        data_paths=args.data_paths,
        file_pattern=args.file_pattern,
        device=args.device,
        output_dir=out_dir,
    )
    
    # Save metadata
    try:
        parsed = parse_cnp_io_list(args.variable_list)
        with open(out_dir / 'inference_config.json', 'w') as f:
            json.dump({
                'variable_list': args.variable_list,
                'model_path': args.model_path,
                'data_paths': args.data_paths,
                'file_pattern': args.file_pattern,
                'device': args.device,
                'train_split': 0.0,
                'predictions_dir': str(predictions_dir),
                'variables': parsed
            }, f, indent=2)
        print(f"Configuration saved to: {out_dir / 'inference_config.json'}")
    except Exception as e:
        print(f"Warning: Could not save configuration: {e}")
    
    print(f"\nInference results ready for restart file update!")
    print(f"Use the following command to update your restart file:")
    print(f"python scripts/update_restart_file.py \\")
    print(f"  --variable-list {args.variable_list} \\")
    print(f"  --run-dir {out_dir} \\")
    print(f"  --input-nc your_input_restart.nc \\")
    print(f"  --output-nc your_updated_restart.nc")


if __name__ == '__main__':
    main()
