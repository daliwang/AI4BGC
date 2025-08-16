#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import argparse
from glob import glob

def create_world_map(ax, title):
    """Set up a world map with coastlines and gridlines"""
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.2)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_global()
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle='--', linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}

def plot_scalar_variables(static_data, predictions_dir, output_dir):
    """Plot scalar variables (grid-level predictions)"""
    scalar_file = os.path.join(predictions_dir, 'predictions_scalar.csv')
    if not os.path.exists(scalar_file):
        print("Scalar predictions file not found, skipping scalar maps")
        return
    
    scalar_pred = pd.read_csv(scalar_file)
    print(f"Found {len(scalar_pred.columns)} scalar variables")
    
    # Create subplots for scalar variables
    num_scalars = min(len(scalar_pred.columns), 4)  # Limit to 4 for display
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), 
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    
    for i, var_name in enumerate(scalar_pred.columns[:num_scalars]):
        if i >= 4:  # Safety check
            break
            
        ax = axes[i]
        create_world_map(ax, f'{var_name} Predictions')
        
        # Get prediction values
        values = scalar_pred[var_name].values
        
        # Create scatter plot
        scatter = ax.scatter(static_data['Longitude'], static_data['Latitude'], 
                            c=values, s=3, cmap='viridis', alpha=0.7,
                            transform=ccrs.PlateCarree())
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.05)
        
        # Print statistics
        print(f"  {var_name}: min={np.nanmin(values):.4g}, max={np.nanmax(values):.4g}")
    
    # Hide unused subplots
    for i in range(num_scalars, 4):
        axes[i].set_visible(False)
    
    plt.suptitle('Scalar Variable Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scalar_variables_map.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Scalar maps saved to: {output_dir}/scalar_variables_map.png")

def plot_pft_variables(static_data, predictions_dir, output_dir, max_pfts=2):
    """Plot first N PFT variables"""
    pft_dir = os.path.join(predictions_dir, 'pft_1d_predictions')
    if not os.path.exists(pft_dir):
        print("PFT predictions directory not found, skipping PFT maps")
        return
    
    pft_files = glob(os.path.join(pft_dir, 'predictions_Y_*.csv'))
    if not pft_files:
        print("No PFT prediction files found")
        return
    
    print(f"Found {len(pft_files)} PFT variables")
    
    # Limit to first few variables for demonstration
    num_vars = min(len(pft_files), 2)
    
    for var_idx, pft_file in enumerate(pft_files[:num_vars]):
        var_name = os.path.basename(pft_file).replace('predictions_Y_', '').replace('.csv', '')
        pft_pred = pd.read_csv(pft_file)
        
        print(f"Processing PFT variable: {var_name}")
        
        # Create subplots for first 2 PFTs
        fig, axes = plt.subplots(1, max_pfts, figsize=(12, 5), 
                                 subplot_kw={'projection': ccrs.PlateCarree()})
        
        if max_pfts == 1:
            axes = [axes]
        
        for pft_idx in range(max_pfts):
            if pft_idx >= pft_pred.shape[1]:
                break
                
            ax = axes[pft_idx]
            create_world_map(ax, f'{var_name} - PFT {pft_idx+1}')
            
            # Get PFT prediction values
            values = pft_pred.iloc[:, pft_idx].values
            
            # Create scatter plot
            scatter = ax.scatter(static_data['Longitude'], static_data['Latitude'], 
                                c=values, s=3, cmap='viridis', alpha=0.7,
                                transform=ccrs.PlateCarree())
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.05)
            
            # Print statistics
            print(f"    PFT {pft_idx+1}: min={np.nanmin(values):.4g}, max={np.nanmax(values):.4g}")
        
        plt.suptitle(f'{var_name} - First {max_pfts} PFTs', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pft_{var_name}_map.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"PFT maps saved to: {output_dir}/")

def plot_soil_variables(static_data, predictions_dir, output_dir, max_layers=2):
    """Plot first N soil layers for soil variables"""
    soil_dir = os.path.join(predictions_dir, 'soil_2d_predictions')
    if not os.path.exists(soil_dir):
        print("Soil predictions directory not found, skipping soil maps")
        return
    
    soil_files = glob(os.path.join(soil_dir, 'predictions_Y_*.csv'))
    if not soil_files:
        print("No soil prediction files found")
        return
    
    print(f"Found {len(soil_files)} soil variables")
    
    # Limit to first few variables for demonstration
    num_vars = min(len(soil_files), 2)
    
    for var_idx, soil_file in enumerate(soil_files[:num_vars]):
        var_name = os.path.basename(soil_file).replace('predictions_Y_', '').replace('.csv', '')
        soil_pred = pd.read_csv(soil_file)
        
        print(f"Processing soil variable: {var_name}")
        
        # Create subplots for first 2 layers
        fig, axes = plt.subplots(1, max_layers, figsize=(12, 5), 
                                 subplot_kw={'projection': ccrs.PlateCarree()})
        
        if max_layers == 1:
            axes = [axes]
        
        for layer_idx in range(max_layers):
            if layer_idx >= soil_pred.shape[1]:
                break
                
            ax = axes[layer_idx]
            create_world_map(ax, f'{var_name} - Layer {layer_idx+1}')
            
            # Get soil layer prediction values
            values = soil_pred.iloc[:, layer_idx].values
            
            # Create scatter plot
            scatter = ax.scatter(static_data['Longitude'], static_data['Latitude'], 
                                c=values, s=3, cmap='viridis', alpha=0.7,
                                transform=ccrs.PlateCarree())
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.05)
            
            # Print statistics
            print(f"    Layer {layer_idx+1}: min={np.nanmin(values):.4g}, max={np.nanmax(values):.4g}")
        
        plt.suptitle(f'{var_name} - First {max_layers} Layers', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'soil_{var_name}_map.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Soil maps saved to: {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Create world maps from CNP inference results')
    parser.add_argument('predictions_dir', help='Path to cnp_predictions directory')
    parser.add_argument('--output-dir', default='./inference_maps', help='Output directory for maps')
    parser.add_argument('--max-pfts', type=int, default=2, help='Maximum number of PFTs to plot (default: 2)')
    parser.add_argument('--max-layers', type=int, default=2, help='Maximum number of soil layers to plot (default: 2)')
    
    args = parser.parse_args()
    
    # Check if predictions directory exists
    if not os.path.exists(args.predictions_dir):
        print(f"Error: Predictions directory not found: {args.predictions_dir}")
        return
    
    # Load static inverse data (contains lat/lon coordinates)
    static_file = os.path.join(args.predictions_dir, 'test_static_inverse.csv')
    if not os.path.exists(static_file):
        print(f"Error: test_static_inverse.csv not found in {args.predictions_dir}")
        return
    
    print(f"Loading coordinates from: {static_file}")
    static_data = pd.read_csv(static_file)
    print(f"Loaded {len(static_data)} geographic points")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Create maps
    print("\n=== Creating Scalar Variable Maps ===")
    plot_scalar_variables(static_data, args.predictions_dir, args.output_dir)
    
    print("\n=== Creating PFT Variable Maps ===")
    plot_pft_variables(static_data, args.predictions_dir, args.output_dir, args.max_pfts)
    
    print("\n=== Creating Soil Variable Maps ===")
    plot_soil_variables(static_data, args.predictions_dir, args.output_dir, args.max_layers)
    
    print(f"\nAll maps created successfully in: {args.output_dir}")
    print("Files created:")
    for file in os.listdir(args.output_dir):
        if file.endswith('.png'):
            print(f"  - {file}")

if __name__ == '__main__':
    main()
