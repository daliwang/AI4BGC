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


def run_inference(
    variable_list_path: str,
    model_path: str,
    data_paths: List[str],
    file_pattern: str,
    device: str,
    output_dir: Path,
    inference_all: bool = False,
) -> Path:
    """Run inference using the trained model on data from data_paths; returns predictions_dir.
    If inference_all is True, use the entire dataset (train_split=0.0) for evaluation output.
    """
    config = get_cnp_combined_config(
        use_trendy1=True,
        use_trendy05=False,
        max_files=None,
        include_water=False,
        variable_list_path=variable_list_path,
    )
    # Configure data and device
    # If inference_all, push all samples into the 'test' split so predictions cover the full dataset
    split_ratio = 0.0 if inference_all else 0.7
    config.update_data_config(data_paths=data_paths, file_pattern=file_pattern, train_split=split_ratio)
    config.update_training_config(device=device, predictions_dir=str(output_dir / "cnp_predictions"))

    # Load and preprocess data
    loader = DataLoader(config.data_config, config.preprocessing_config)
    loader.load_data()
    loader.preprocess_data()
    data_info = loader.get_data_info()
    normalized = loader.normalize_data()
    split = loader.split_data(normalized)
    train_data = split['train']
    test_data = split['test']
    scalers = normalized['scalers']

    # Build model and load weights
    model = CNPCombinedModel(config.model_config, data_info, include_water=False,
                             use_learnable_loss_weights=config.training_config.use_learnable_loss_weights)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)

    # Evaluate and save predictions (uses test_data by design)
    trainer = ModelTrainer(config.training_config, model, train_data, test_data, scalers, data_info)
    predictions, metrics = trainer.evaluate()
    trainer.save_results(predictions, metrics)

    return Path(config.training_config.predictions_dir)


def load_predictions(predictions_dir: Path) -> Dict[str, Any]:
    """Load predictions saved by trainer.save_results into memory."""
    preds: Dict[str, Any] = {}
    # Scalars
    scalar_path = predictions_dir / 'predictions_scalar.csv'
    if scalar_path.exists():
        preds['scalar'] = pd.read_csv(scalar_path)
    # 1D PFT
    pft_dir = predictions_dir / 'pft_1d_predictions'
    if pft_dir.exists():
        preds['pft_1d'] = {p.stem.replace('predictions_', ''): pd.read_csv(p) for p in sorted(pft_dir.glob('predictions_*.csv'))}
    # 2D soil
    soil_dir = predictions_dir / 'soil_2d_predictions'
    if soil_dir.exists():
        preds['soil_2d'] = {p.stem.replace('predictions_', ''): pd.read_csv(p) for p in sorted(soil_dir.glob('predictions_*.csv'))}
    # Optional: static inverse for mapping (Longitude/Latitude)
    static_inv = predictions_dir / 'test_static_inverse.csv'
    if static_inv.exists():
        preds['test_static_inverse'] = pd.read_csv(static_inv)
    return preds


def update_restart_with_predictions(
    input_nc: str,
    output_nc: str,
    preds: Dict[str, Any],
    special_soil_vars_all_columns: Optional[List[str]] = None
) -> None:
    """Write predicted variables into a copy of the restart NetCDF file."""
    special = set(special_soil_vars_all_columns or ['labilep_vr', 'occlp_vr', 'primp_vr', 'secondp_vr'])

    with Dataset(input_nc, 'r') as src, Dataset(output_nc, 'w', format='NETCDF4') as dst:
        # Copy dimensions
        for name, dim in src.dimensions.items():
            dst.createDimension(name, (len(dim) if not dim.isunlimited() else None))
        # Copy variables, then selectively overwrite below
        for name, var in src.variables.items():
            out = dst.createVariable(name, var.dtype, var.dimensions)
            out.setncatts({a: var.getncattr(a) for a in var.ncattrs()})
            out[:] = var[:]

        # Load mapping indices if present
        pft_grid_idx = src.variables['pfts1d_gridcell_index'][:] if 'pfts1d_gridcell_index' in src.variables else None
        col_grid_idx = src.variables['cols1d_gridcell_index'][:] if 'cols1d_gridcell_index' in src.variables else None

        # 1D PFT predictions: preds['pft_1d'] maps var name -> DataFrame columns like var_pft1..16
        if 'pft_1d' in preds and pft_grid_idx is not None:
            # Build gridcell->pft indices map
            from collections import defaultdict
            grid_to_pfts = defaultdict(list)
            for pft_i, grid_i in enumerate(pft_grid_idx):
                grid_to_pfts[int(grid_i - 1)].append(pft_i)

            # Build optional mapping from prediction row to gridcell using static inverse if available
            row_to_grid = None
            if 'test_static_inverse' in preds:
                df_static = preds['test_static_inverse']
                # Heuristic: find nearest gridcell by Longitude/Latitude if present
                if {'Longitude', 'Latitude'}.issubset(df_static.columns):
                    try:
                        # Index gridcell lat/lon
                        g_lat = src.variables['grid1d_lat'][:] if 'grid1d_lat' in src.variables else None
                        g_lon = src.variables['grid1d_lon'][:] if 'grid1d_lon' in src.variables else None
                        if g_lat is not None and g_lon is not None:
                            from scipy.spatial import cKDTree
                            tree = cKDTree(np.column_stack([g_lon, g_lat]))
                            q = np.column_stack([df_static['Longitude'].values, df_static['Latitude'].values])
                            dists, idxs = tree.query(q, k=1)
                            row_to_grid = idxs.astype(int)
                    except Exception:
                        row_to_grid = None

            for y_var, df in preds['pft_1d'].items():
                # y_var is "Y_xyz"; NetCDF variable may be without Y_
                nc_name = y_var.replace('Y_', '')
                if nc_name not in dst.variables:
                    continue
                data = dst.variables[nc_name][:]
                # Expect columns named like f"{y_var}_pft1"..pft16 (1-based). Sort numerically.
                pft_cols = [c for c in df.columns if c.startswith(y_var + '_pft')]
                if not pft_cols:
                    continue
                def _pft_num(col: str) -> int:
                    try:
                        return int(col.split('_pft')[-1])
                    except Exception:
                        return 999
                pft_cols = sorted(pft_cols, key=_pft_num)
                vals = df[pft_cols].to_numpy()  # shape: (num_gridcells_like, num_pft_cols)
                # Assign per prediction row -> matched gridcell
                for row_idx in range(vals.shape[0]):
                    # Choose target gridcell
                    if row_to_grid is not None and row_idx < len(row_to_grid):
                        grid_idx = int(row_to_grid[row_idx])
                    else:
                        grid_idx = row_idx
                    if grid_idx not in grid_to_pfts:
                        continue
                    pft_indices = grid_to_pfts[grid_idx]
                    # Ensure we have at least 2 indices: pft0 plus at least pft1
                    if len(pft_indices) <= 1:
                        continue
                    # Determine how many PFTs we can update (up to 16, or available - 1 for pft0)
                    max_update = min(16, len(pft_indices) - 1, vals.shape[1])
                    # Map k in [0..max_update-1] -> PFT index pft_indices[k+1]
                    for k in range(max_update):
                        try:
                            data[pft_indices[k + 1]] = vals[row_idx, k]
                        except Exception:
                            # If variable has extra dims incompatible with direct 1D assignment, skip
                            pass
                dst.variables[nc_name][:] = data

        # 2D soil predictions: per variable DataFrame with columns like var_col{1..18}_layer{1..10}
        if 'soil_2d' in preds and col_grid_idx is not None:
            from collections import defaultdict
            grid_to_cols = defaultdict(list)
            for col_i, grid_i in enumerate(col_grid_idx):
                grid_to_cols[int(grid_i - 1)].append(col_i)

            for y_var, df in preds['soil_2d'].items():
                nc_name = y_var.replace('Y_', '')
                if nc_name not in dst.variables:
                    continue
                var = dst.variables[nc_name]
                data = var[:]
                # Determine columns/layers from headers
                col_layer_cols = [c for c in df.columns if c.startswith(y_var + '_col')]
                if not col_layer_cols:
                    continue
                # Parse max col/layer
                cols = []
                layers = []
                for c in col_layer_cols:
                    # format: Y_var_col{C}_layer{L}
                    try:
                        tail = c.replace(y_var + '_col', '')
                        c_idx, l_part = tail.split('_layer')
                        cols.append(int(c_idx))
                        layers.append(int(l_part))
                    except Exception:
                        continue
                max_col = max(cols) if cols else 0
                max_layer = max(layers) if layers else 0
                V = df[col_layer_cols].to_numpy().reshape(df.shape[0], max_col, max_layer)

                for grid_idx, col_indices in grid_to_cols.items():
                    if not col_indices:
                        continue
                    # Determine which prediction row maps to this gridcell
                    if 'test_static_inverse' in preds and row_to_grid is not None:
                        # Find first row mapped to this grid
                        rows = np.where(row_to_grid == grid_idx)[0]
                        if rows.size == 0:
                            continue
                        row_sel = int(rows[0])
                    else:
                        row_sel = grid_idx
                    if row_sel >= V.shape[0]:
                        continue
                    # Always update only the first column for now
                    first_col = col_indices[0]
                    if data.ndim == 2:
                        # shape: (num_cols, num_layers)
                        try:
                            data[first_col, :min(max_layer, data.shape[1])] = V[row_sel, 0, :min(max_layer, data.shape[1])]
                        except Exception:
                            pass
                    elif data.ndim == 1:
                        # scalar per column
                        try:
                            data[first_col] = V[row_sel, 0, 0]
                        except Exception:
                            pass
                dst.variables[nc_name][:] = data

        # Scalars: if present and names exist, overwrite (these may be grid-level scalars; use index-wise assignment)
        if 'scalar' in preds:
            for y_name in preds['scalar'].columns:
                nc_name = y_name.replace('Y_', '')
                if nc_name in dst.variables:
                    try:
                        dst.variables[nc_name][:] = preds['scalar'][y_name].to_numpy()
                    except Exception:
                        pass


def main():
    examples = textwrap.dedent(
        """
        Examples:
          # Reuse predictions from an existing run directory to update restart file
          python scripts/update_restart_with_ai.py \
            --variable-list CNP_IO_list_general.txt \
            --run-dir cnp_results/run_20250807_215454 \
            --input-nc original_20250408_trendytest_ICB1850CNPRDCTCBC.elm.r.0021-01-01-00000.nc \
            --output-nc updated_20250408_trendytest_ICB1850CNPRDCTCBC.elm.r.0021-01-01-00000.nc

          # Run inference with a trained model on ALL data, then update restart file
          python scripts/update_restart_with_ai.py \
            --variable-list CNP_IO_list_general.txt \
            --model-path cnp_results/run_20250807_215454/cnp_predictions/model.pth \
            --data-paths /path/to/Trendy_1_data_CNP \
            --file-pattern '1_training_data_batch_*.pkl' \
            --device cuda \
            --inference-all \
            --output-nc updated_20250408_trendytest_ICB1850CNPRDCTCBC.elm.r.0021-01-01-00000.nc
        """
    )

    p = argparse.ArgumentParser(
        description='Run CNP model inference and update a restart NetCDF with AI predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples
    )
    p.add_argument('--variable-list', required=True, help='Path to CNP_IO_list_general.txt')
    p.add_argument('--input-nc', default='original_20250408_trendytest_ICB1850CNPRDCTCBC.elm.r.0021-01-01-00000.nc', help='Path to input restart NetCDF')
    p.add_argument('--output-nc', required=True, help='Path to output restart NetCDF')
    # Inference options
    p.add_argument('--run-dir', help='Existing run directory containing cnp_predictions to use directly')
    p.add_argument('--model-path', help='Path to trained model state_dict (model.pth) for inference')
    p.add_argument('--data-paths', nargs='*', help='Data directories containing PKL batches for inference')
    p.add_argument('--file-pattern', default='1_training_data_batch_*.pkl', help='Glob pattern for PKL files')
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device for inference')
    p.add_argument('--out-dir', default=None, help='Directory to store inference outputs; if --inference-all and not set, defaults to {data_path_basename}_inference')
    p.add_argument('--inference-all', action='store_true', help='Use the entire dataset for inference outputs (train_split=0.0)')
    p.add_argument('--examples', action='store_true', help='Show example usage and exit')
    args = p.parse_args()

    if args.examples:
        print(p.format_help())
        sys.exit(0)

    if not args.run_dir and not (args.model_path and args.data_paths):
        p.error('Provide either --run-dir (to reuse predictions) OR both --model-path and --data-paths to run inference')

    if args.run_dir:
        predictions_dir = Path(args.run_dir) / 'cnp_predictions'
        if not predictions_dir.exists():
            p.error(f'Predictions directory not found: {predictions_dir}')
    else:
        # Determine output directory
        if args.out_dir:
            out_dir = Path(args.out_dir)
        else:
            # Default to {first_data_path_basename}_inference when inferring over all data; otherwise cnp_infer
            if args.inference_all and args.data_paths:
                base_name = Path(args.data_paths[0]).name
                out_dir = Path(f"{base_name}_inference")
            else:
                out_dir = Path('cnp_infer')
        out_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir = run_inference(
            variable_list_path=args.variable_list,
            model_path=args.model_path,
            data_paths=args.data_paths,
            file_pattern=args.file_pattern,
            device=args.device,
            output_dir=out_dir,
            inference_all=bool(args.inference_all),
        )

    preds = load_predictions(predictions_dir)
    # Persist the IO list used alongside the output NetCDF (if provided)
    out_meta_dir = Path(args.output_nc).parent
    out_meta_dir.mkdir(parents=True, exist_ok=True)
    if args.variable_list and Path(args.variable_list).exists():
        try:
            dest_txt = out_meta_dir / 'CNP_IO_list_used.txt'
            shutil.copyfile(args.variable_list, dest_txt)
        except Exception:
            pass
        try:
            parsed = parse_cnp_io_list(args.variable_list)
            with open(out_meta_dir / 'CNP_IO_list_used.json', 'w') as f:
                json.dump(parsed, f, indent=2)
        except Exception:
            pass
    update_restart_with_predictions(args.input_nc, args.output_nc, preds)
    print(f"Updated restart file written to: {args.output_nc}")


if __name__ == '__main__':
    main() 