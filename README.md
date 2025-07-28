# AI4BGC: Artificial Intelligence for Biogeochemical Cycle Simulation

Welcome to the **AI4BGC** pre-release! This repository provides a modern, flexible, and high-performance deep learning framework for modeling terrestrial biogeochemical cycles using neural networks.

---

## 🚀 Quick Start

### 1. **Clone the Repository**
```bash
git clone <your-repo-url>
cd AI4BGC
git checkout pre_release_branch
```

### 2. **Install Dependencies**
We recommend using a conda environment:
```bash
conda create -n ai4bgc_env python=3.11
conda activate ai4bgc_env
pip install -r requirements.txt
```

### 3. **Run a Demo Training (Default Model)**
```bash
python train_model.py
```
- Trains a simple model on a small subset of the data (3 files, minimal variables)
- Fast: completes in minutes
- Great for testing, learning, and development

### 4. **Run Full CNP Model Training (Production/Research)**
```bash
python train_cnp_model.py
```
- Requires dedicated CNP datasets (not included in this release) 
- Trains the full CNP model on all available data and variables
- For research, publications, and production use
- Requires more memory and time
- Support --help option

---

## 🏗️ Model Modes

### **Default Model (`train_model.py`)**
- **Purpose:** Quick start, demonstration, and development
- **Inputs:** Minimal set of variables (forcing, static, 1 PFT param, 1 1D PFT, 1 soil 2D)
- **Data:** 3 mini files from data/example_dataset
- **Speed:** Fast (minutes)
- **Resources:** Low (can run on laptop or small GPU)

### **CNP Model (`train_cnp_model.py`)**
- **Purpose:** Full production/research model
- **Inputs:** All available variables (forcing, static, 44 PFT params, 14 1D PFT, 28 soil 2D, etc.)
- **Data:** All files from Trendy_1_data_CNP and Trendy_05_data_CNP
- **Speed:** Slower (hours+)
- **Resources:** High (requires large GPU/cluster)

---

## 🔑 Key Features
- **Unified architecture:** LSTM, CNN, Transformer fusion (CNPCombinedModel)
- **Flexible data pipeline:** Handles time series, static, 1D/2D/param groups
- **Configurable:** All variables, paths, and training settings in `config/training_config.py`
- **Clean logs:** Only essential INFO-level output for release
- **No debug/NaN spam:** All debug and expensive NaN checks removed for performance
- **Ready for research:** Full CNP model for publications and production
- **Easy to extend:** Add new variables, datasets, or architectures as needed

---

## 📁 Directory Structure
```
AI4BGC_pre_release/
├── config/                # Configuration files (training, model, data)
├── data/                  # Data loading and preprocessing
├── models/                # Model architectures (CNP, enhanced, etc.)
├── training/              # Training pipeline and utilities
├── utils/                 # Utility functions (GPU monitoring, etc.)
├── train_model.py         # Default (quick start) training script
├── train_cnp_model.py     # Full CNP model training script
├── requirements.txt       # Python dependencies
└── README.md              # This file
```.

---

## 🧪 Example Mini Dataset (Quick Testing)

A minimal example dataset is available for fast testing and development:

- **Location:** `data/example_dataset/`
- **Files:** `mini_data_1.pkl`, `mini_data_2.pkl`, `mini_data_3.pkl`
- **Contents:** Only the variables used in the default model (see below)

### Variables Included
- **Time series (20 timesteps):** `FLDS`, `PSRF`, `FSDS`, `QBOT`, `PRECTmms`, `TBOT` 
- **Static:** `Latitude`, `Longitude`
- **PFT param:** `pft_leafcn`
- **1D PFT:** `deadcrootc`, `Y_deadcrootc`
- **Scalar:** `GPP`, `NPP`, `Y_GPP`, `Y_NPP`
- **2D:** `soil1c_vr`, `Y_soil1c_vr`

---

## 🤝 Contributing
We welcome feedback, bug reports, and contributions! Please open issues or pull requests as needed.

---

**AI4BGC: Accelerating Earth System Science with Deep Learning** 
