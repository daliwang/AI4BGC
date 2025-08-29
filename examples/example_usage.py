#!/usr/bin/env python3
"""
Example Usage for AI4BGC Model Training

This script demonstrates how to use the main training scripts from the command line.
"""

print("""
AI4BGC Example Usage
====================

# 1. Quick Start: Default Model (Mini Dataset)

To train the default (quick start) model on the small example dataset:

    python train_model.py

- Uses only a minimal set of variables and 3 small files from data/example_dataset/
- Fast, suitable for testing and development

# 2. Full CNP Model Training (Production/Research)

To train the full CNP model on all available data and variables:

    python train_cnp_model.py (use Trendy_1_data_CNP as default, same as --use-trendy05)

- Uses all variables and all data files (Trendy_1_data_CNP and Trendy_05_data_CNP (--use-trendy05))
- Requires more memory and time

# 3. Optional Input Flags (if supported)

Both scripts may support optional CLI flags for advanced usage. For example:

    python train_model.py --epochs 20 --batch_size 64
    python train_cnp_model.py --epochs 150 --batch_size 128 --learning-rate 0.0001 --use-trendy05

Check the script's help for available options:

    python train_model.py (does not support --help)
    python train_cnp_model.py --help

# 4. Regenerating the Example Dataset

If you need to regenerate the small example dataset:

    cd data
    python create_example_dataset.py

# 5. More Information

- See the README.md for details on model modes, configuration, and dataset structure.
- All configuration options are in config/training_config.py.

""") 