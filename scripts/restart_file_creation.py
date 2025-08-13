#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
import subprocess
import re

# Defaults derived from previous shell script
DEFAULT_AI4BGC_DIR = "/mnt/proj-shared/AI4BGC_7xw"
DEFAULT_DATA_PATH = f"{DEFAULT_AI4BGC_DIR}/TrainingData/Trendy_1_data_CNP/enhanced_dataset"
DEFAULT_INPUT_NC = f"{DEFAULT_AI4BGC_DIR}/AI4BGC/ELM_data/original_20250408_trendytest_ICB1850CNPRDCTCBC.elm.r.0021-01-01-00000.nc"
DEFAULT_MODEL_PATH = "./cnp_predictions/model.pth"

# Fallback output name if no training log timestamp is discoverable
FALLBACK_OUTPUT_NC = "./restart_file_updated.nc"


def _discover_training_timestamp(cwd: Path) -> str:
    pattern = re.compile(r"^cnp_training_(\d{8}_\d{6})\.log$")
    timestamps = []
    for p in cwd.glob("cnp_training_*.log"):
        m = pattern.match(p.name)
        if m:
            timestamps.append(m.group(1))
    if not timestamps:
        return ""
    # Choose the latest by lexicographic order (safe for YYYYMMDD_HHMMSS)
    return sorted(timestamps)[-1]


def _sanitize_component(name: str) -> str:
    # Keep alphanum, dash, underscore; replace others with underscore
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def _build_default_output_nc(variable_list_path: str) -> str:
    ts = _discover_training_timestamp(Path.cwd())
    var_stem = _sanitize_component(Path(variable_list_path).stem) if variable_list_path else "vars"
    if ts:
        return f"./restart_file_{var_stem}_{ts}_updated.nc"
    return f"./restart_file_{var_stem}_updated.nc"


SCRIPT_PATH = f"{DEFAULT_AI4BGC_DIR}/AI4BGC/scripts/update_restart_with_aiprediction.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create an updated restart NetCDF using AI predictions (wrapper)",
    )
    p.add_argument("--variable-list", required=True, help="Path to CNP IO variable list (e.g., CNP_IO_default.txt)")
    p.add_argument("--input-nc", default=DEFAULT_INPUT_NC, help="Path to input restart NetCDF")
    p.add_argument("--output-nc", default=None, help="Path to output restart NetCDF (default: ./restart_file_<variable-list>_<timestamp>_updated.nc)")
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to trained model state_dict (model.pth)")
    p.add_argument("--data-paths", nargs="*", default=[DEFAULT_DATA_PATH], help=f"Data directories for inference (default: {DEFAULT_DATA_PATH})")
    p.add_argument("--file-pattern", default="*1_training_data_batch_*.pkl", help="Glob pattern for PKL files")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for inference")
    p.add_argument("--inference-all", action="store_true", default=True, help="Use the entire dataset for inference outputs (default: enabled)")
    p.add_argument("--run-dir", help="Existing run directory containing cnp_predictions (alternative to providing model/data)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.output_nc is None:
        args.output_nc = _build_default_output_nc(args.variable_list)

    if not Path(SCRIPT_PATH).exists():
        print(f"[ERROR] Backend script not found: {SCRIPT_PATH}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        SCRIPT_PATH,
        "--variable-list", args.variable_list,
        "--input-nc", args.input_nc,
        "--output-nc", args.output_nc,
    ]

    # Prefer run-dir if provided
    if args.run_dir:
        cmd.extend(["--run-dir", args.run_dir])
    else:
        cmd.extend([
            "--model-path", args.model_path,
            "--file-pattern", args.file_pattern,
            "--device", args.device,
        ])
        if args.data_paths:
            cmd.append("--data-paths")
            cmd.extend(args.data_paths)
        if args.inference_all:
            cmd.append("--inference-all")

    print("Running:", " ".join([repr(c) if " " in c else c for c in cmd]))
    proc = subprocess.run(cmd)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main()) 