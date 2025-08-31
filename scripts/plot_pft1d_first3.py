#!/usr/bin/env python3
"""
Plot the first three PFT columns (after Longitude, Latitude) on a global map.

Default behavior: plot BOTH prediction and ground-truth files.

Usage (both by default):
  python scripts/plot_pft1d_first3.py \
    [--pred-csv ./cnp_inference_entire_dataset/cnp_predictions/pft_1d_predictions/predictions_Y_deadcrootc.csv] \
    [--gt-csv ./cnp_inference_entire_dataset/cnp_predictions/pft_1d_ground_truth/ground_truth_Y_tlai.csv] \
    [--output-dir ./cnp_inference_entire_dataset/cnp_predictions/ai_model_comparison_plots/] \
    [--cartopy]

Single-file mode:
  python scripts/plot_pft1d_first3.py --only pred  # or --only gt

Assumes CSV columns are:
  Longitude, Latitude, <pft1>, <pft2>, <pft3>, ... up to 16 PFTs.
"""

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot first 3 PFT 1D columns globally")
    parser.add_argument("--pred-csv", default="./cnp_inference_entire_dataset/cnp_predictions/pft_1d_predictions/predictions_Y_deadcrootc.csv", help="Path to predictions CSV")
    parser.add_argument("--gt-csv", default="./cnp_inference_entire_dataset/cnp_predictions/pft_1d_ground_truth/ground_truth_Y_tlai.csv", help="Path to ground-truth CSV")
    parser.add_argument("--output-dir", default="./cnp_inference_entire_dataset/cnp_predictions/ai_model_comparison_plots/", help="Directory to save output images")
    parser.add_argument("--only", choices=["pred", "gt"], help="If set, only generate one of: pred or gt")
    parser.add_argument("--cartopy", action="store_true", help="Use cartopy for a proper map projection")
    parser.add_argument("--marker-size", type=float, default=6.0, help="Scatter marker size")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    return parser.parse_args()


def wrap_longitudes(longitudes: np.ndarray) -> np.ndarray:
    lon = np.asarray(longitudes, dtype=float)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    return lon


def pick_value_columns(header: List[str]) -> List[str]:
    lower = [h.lower() for h in header]
    try:
        lon_idx = lower.index("longitude")
        lat_idx = lower.index("latitude")
    except ValueError:
        raise ValueError("CSV must have 'Longitude' and 'Latitude' columns")
    start_idx = max(lon_idx, lat_idx) + 1
    value_cols = header[start_idx: start_idx + 3]
    if len(value_cols) < 3:
        raise ValueError("CSV does not have at least 3 PFT columns after coordinates")
    return value_cols


def plot_with_matplotlib(ax, lon, lat, values, title: str, marker_size: float, vmin: float, vmax: float) -> None:
    ax.scatter(
        lon,
        lat,
        c=values,
        s=marker_size,
        cmap="viridis",
        alpha=0.9,
        edgecolors="none",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(True, linestyle=":", alpha=0.5)


def plot_with_cartopy(ax, lon, lat, values, title: str, marker_size: float, vmin: float, vmax: float):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ax.set_global()
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2)
    sc = ax.scatter(
        lon,
        lat,
        c=values,
        s=marker_size,
        cmap="viridis",
        alpha=0.9,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        edgecolors="none",
    )
    ax.set_title(title)
    return sc


def generate_map_for_csv(csv_path: str, output_path: str, args: argparse.Namespace, colorbar_label: str = "Value") -> None:
    df = pd.read_csv(csv_path)
    header = list(df.columns)
    value_cols = pick_value_columns(header)
    lon = wrap_longitudes(df[header[0]].to_numpy())
    lat = df[header[1]].to_numpy()

    fig_kwargs = dict(figsize=(14, 4.5), dpi=args.dpi)

    if args.cartopy:
        try:
            import cartopy.crs as ccrs  # noqa: F401
        except Exception as exc:
            print("Cartopy requested but not importable. Install cartopy or omit --cartopy.")
            print(f"Import error: {exc}")
            raise
        proj = ccrs.Robinson()
        all_values = np.concatenate([df[col].to_numpy() for col in value_cols])
        vmin, vmax = float(np.nanmin(all_values)), float(np.nanmax(all_values))
        fig, axes = plt.subplots(1, 3, subplot_kw=dict(projection=proj), **fig_kwargs)
        for ax, col in zip(axes, value_cols):
            sc = plot_with_cartopy(
                ax,
                lon,
                lat,
                df[col].to_numpy(),
                title=col,
                marker_size=args.marker_size,
                vmin=vmin,
                vmax=vmax,
            )
        cb = fig.colorbar(sc, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.05, pad=0.08)
        cb.set_label(colorbar_label)
    else:
        fig, axes = plt.subplots(1, 3, **fig_kwargs)
        all_values = np.concatenate([df[col].to_numpy() for col in value_cols])
        vmin, vmax = float(np.nanmin(all_values)), float(np.nanmax(all_values))
        for ax, col in zip(axes, value_cols):
            plot_with_matplotlib(
                ax,
                lon,
                lat,
                df[col].to_numpy(),
                title=col,
                marker_size=args.marker_size,
                vmin=vmin,
                vmax=vmax,
            )
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.05, pad=0.08)
        cb.set_label(colorbar_label)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pred_out = os.path.join(args.output_dir, "Y_tlai_pred_first3_global.png")
    gt_out = os.path.join(args.output_dir, "Y_tlai_GT_first3_global.png")

    try:
        if args.only == "pred":
            generate_map_for_csv(args.pred_csv, pred_out, args, colorbar_label="Value")
        elif args.only == "gt":
            generate_map_for_csv(args.gt_csv, gt_out, args, colorbar_label="Value")
        else:
            generate_map_for_csv(args.pred_csv, pred_out, args, colorbar_label="Value")
            generate_map_for_csv(args.gt_csv, gt_out, args, colorbar_label="Value")
    except Exception as exc:
        print(f"Error while generating map(s): {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


