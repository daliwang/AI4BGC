# AI Model Comparison Plot Tool

## Overview

The `ai_model_comparison_plot.py` script is a powerful utility for comparing AI model predictions with numerical model results using tri-panel visualization plots. It automatically discovers common variables between datasets and generates professional-quality comparison maps.

## Features

- **Auto-discovery**: Automatically finds variables that exist in both AI predictions and model results
- **Tri-panel plots**: Shows AI predictions, model results, differences, and percent difference categories
- **Multiple variable types**: Handles column-based (soil), PFT-based, and gridcell-based variables
- **Flexible input**: Supports custom file paths and variable selection
- **Professional output**: Generates high-resolution maps with proper geographic projections

## Requirements

- Python 3.7+
- Required packages:
  - `xarray`
  - `numpy`
  - `matplotlib`
  - `cartopy`
  - `pandas`

## Installation

1. Ensure you have the required Python packages installed:
```bash
pip install xarray numpy matplotlib cartopy pandas
```

2. Make the script executable:
```bash
chmod +x ai_model_comparison_plot.py
```

## Usage

### Basic Usage

```bash
# Use default settings (auto-discovers variables)
python ai_model_comparison_plot.py
```

### Command Line Options

```bash
python ai_model_comparison_plot.py [OPTIONS]
```

**Options:**
- `--ai-predictions PATH`: Path to AI predictions NetCDF file
- `--model PATH`: Path to model results NetCDF file  
- `--output-dir PATH`: Output directory for plots
- `--variables [VARS ...]`: Specific variables to plot
- `--layers [LAYERS ...]`: Soil layers to plot for column variables
- `--pfts [PFTS ...]`: PFTs to plot for PFT variables
- `-h, --help`: Show help message

### Examples

#### 1. Default Comparison
```bash
# Uses default paths and auto-discovers variables
python ai_model_comparison_plot.py
```

#### 2. Custom File Paths
```bash
python ai_model_comparison_plot.py \
  --ai-predictions ./my_ai_predictions.nc \
  --model ./my_model_results.nc \
  --output-dir ./my_comparison_plots
```

#### 3. Specific Variables Only
```bash
# Plot only specific variables
python ai_model_comparison_plot.py --variables cwdc_vr tlai
```

#### 4. Custom Layers and PFTs
```bash
# Plot specific soil layers and PFTs
python ai_model_comparison_plot.py \
  --layers 0 2 5 9 \
  --pfts 1 3 5 7
```

## Default Settings

- **AI predictions**: `./comparison_results/ai_predictions_for_plotting.nc`
- **Model results**: `/mnt/proj-shared/AI4BGC_7xw/AI4BGC/ELM_data/original_780_spinup_from_modelsimulation.nc`
- **Output directory**: `./ai_model_comparison_plots`
- **Default variables**: Auto-discovered from both datasets
- **Default layers**: `[0, 4, 9]` (for soil variables)
- **Default PFTs**: `[1, 2, 3, 4, 5]` (for PFT variables)

## Output

### Generated Plots

The script creates tri-panel comparison plots for each variable:

1. **AI Predictions**: Shows AI model output
2. **Model Results**: Shows numerical model output
3. **Difference**: Shows (AI - Model) differences
4. **Percent Difference**: Categorized percent differences

### File Naming Convention

- **Column variables**: `{variable}_lev{layer}.png` (e.g., `cwdc_vr_lev0.png`)
- **PFT variables**: `{variable}_pft{pft_number}.png` (e.g., `tlai_pft2.png`)

### Statistics

Each plot includes statistical metrics:
- **RMSE**: Root Mean Square Error
- **NRMSE**: Normalized RMSE
- **R²**: Coefficient of determination
- **Data ranges**: Min/max values for both datasets

## Supported Variable Types

### 1. Column Variables (Soil Variables)
- **Dimensions**: `(column, levgrnd)` or `(column, levgrnd, gridcell)`
- **Examples**: `cwdc_vr`, `cwdn_vr`, `cwdp_vr`
- **Plotting**: Multiple soil layers (default: 0, 4, 9)

### 2. PFT Variables (Plant Functional Type)
- **Dimensions**: `(pft,)` or `(pft, gridcell)`
- **Examples**: `tlai`, `GPP`, `NPP`
- **Plotting**: Multiple PFTs (default: 1, 2, 3, 4, 5)

### 3. Gridcell Variables (Scalar)
- **Dimensions**: `(gridcell,)`
- **Examples**: `GPP`, `NPP`, `AR`, `HR`
- **Plotting**: Direct gridcell comparison

## Data Structure Requirements

### AI Predictions File
- Must contain variables to compare
- Should have coordinate variables: `grid1d_lon`, `grid1d_lat`
- For column variables: dimensions should include `column` and `levgrnd`
- For PFT variables: dimensions should include `pft`

### Model Results File
- Must contain the same variables as AI predictions
- Should have coordinate variables: `grid1d_lon`, `grid1d_lat`
- Should have mapping variables: `cols1d_gridcell_index`, `pfts1d_gridcell_index`
- For column variables: dimensions should include `column` and `levgrnd`
- For PFT variables: dimensions should include `pft`

## Troubleshooting

### Common Issues

#### 1. "No common variables found"
- **Cause**: Variables don't exist in both datasets
- **Solution**: Check variable names in both files using `ncdump -h` or `xarray`

#### 2. "Index out of bounds" errors
- **Cause**: Dimension mismatch between AI and model data
- **Solution**: Ensure both datasets have compatible dimensions

#### 3. All values are 0 or NaN
- **Cause**: Data extraction issues or missing data
- **Solution**: Check data values directly in the NetCDF files

#### 4. "Gridcell count mismatch" warnings
- **Cause**: Different numbers of gridcells between datasets
- **Solution**: The script will automatically handle this by using the minimum count

### Debug Mode

The script provides detailed debugging information:
- Variable shapes and dimensions
- Grid mapping information
- Sample data values
- Processing statistics

## Example Output

```
============================================================
AI vs Model Comparison
============================================================
AI predictions: ./comparison_results/ai_predictions_for_plotting.nc
Model results: /mnt/proj-shared/AI4BGC_7xw/AI4BGC/ELM_data/original_780_spinup_from_modelsimulation.nc
Output directory: ./ai_model_comparison_plots
Variables to plot: ['cwdc_vr', 'cwdn_vr', 'cwdp_vr', 'tlai']
Layers to plot: [0, 4, 9]
PFTs to plot: [1, 2, 3, 4, 5]
============================================================

Discovered 4 common variables: ['cwdc_vr', 'cwdn_vr', 'cwdp_vr', 'tlai']
Total gridcells: 20975 | total columns: 253641 | total pfts: 589241

Start plotting: 4 variables

Variable cwdc_vr, dims: ('column', 'levgrnd', 'gridcell')
  AI shape: (1, 10, 20975)
  Model shape: (253641, 15)
Stats for cwdc_vr_lev0:
  AI Predictions: sum=1441.99 min=0 max=0.785969
  Model Results: sum=1.41128e+08 min=0 max=90010.1
  Metrics (AI vs Model): n=20975 rmse=13278.5 nrmse=193147 r2=-1.30739e+10
  Saved: ./ai_model_comparison_plots/cwdc_vr_lev0.png
```

## Advanced Usage

### Batch Processing

```bash
# Process multiple variables in sequence
for var in cwdc_vr cwdn_vr cwdp_vr tlai; do
    python ai_model_comparison_plot.py --variables $var
done
```

### Custom Plotting

The script can be modified to:
- Change color schemes
- Adjust plot layouts
- Add custom statistical metrics
- Modify geographic projections

## File Structure

```
run_20250814_193455/
├── ai_model_comparison_plot.py          # Main script
├── README_ai_model_comparison.md        # This README
├── comparison_results/                   # AI predictions
│   └── ai_predictions_for_plotting.nc
└── ai_model_comparison_plots/           # Generated plots
    ├── cwdc_vr_lev0.png
    ├── cwdc_vr_lev4.png
    ├── cwdc_vr_lev9.png
    ├── cwdn_vr_lev0.png
    ├── cwdn_vr_lev4.png
    ├── cwdn_vr_lev9.png
    ├── cwdp_vr_lev0.png
    ├── cwdp_vr_lev4.png
    ├── cwdp_vr_lev9.png
    ├── tlai_pft2.png
    ├── tlai_pft3.png
    ├── tlai_pft4.png
    ├── tlai_pft5.png
    └── tlai_pft6.png
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your data files have the required structure
3. Ensure all required Python packages are installed
4. Check the script's debug output for detailed information

## Version History

- **v1.0**: Initial release with basic comparison functionality
- **v1.1**: Added auto-discovery of common variables
- **v1.2**: Improved gridcell mapping and data extraction
- **v1.3**: Added comprehensive error handling and debugging
