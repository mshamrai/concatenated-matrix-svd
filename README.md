# Concatenated Matrix SVD

Official implementation of **"Concatenated Matrix SVD: Compression Bounds, Incremental Approximation, and Error-Constrained Clustering"**.

This repository provides algorithms for clustering matrix blocks under error constraints, with applications to data compression, neural network weight compression, and scientific data analysis. The methods enable efficient low-rank approximation of concatenated matrices without explicitly forming the full matrix.

## Overview

The concatenated matrix SVD problem arises when we have a collection of matrices (data blocks) that we want to cluster and compress using low-rank approximations, subject to reconstruction error constraints. This work presents:

- **Theoretical compression bounds** for concatenated matrix SVD
- **Incremental approximation algorithms** that avoid forming the full concatenated matrix
- **Error-constrained clustering** methods based on different error estimation strategies

### Key Features

- **Incremental SVD updates**: Efficiently compute approximate singular values when adding new blocks
- **Multiple clustering strategies**: Max-norm, residual-based, and incremental approximation methods
- **Error guarantees**: Theoretical bounds on reconstruction error
- **Scalability**: Process large collections of matrices without full concatenation
- **Multiple datasets**: Tested on wireless channel data, satellite imagery, PDE simulations, and neural network weights

## Installation

### Requirements

- Python 3.8+
- NumPy, SciPy, scikit-learn
- PyTorch, Transformers (for neural network experiments)
- HDBSCAN, TensorLy, Rasterio (for specific datasets)

### Setup

```bash
# Clone the repository
git clone ...
cd concatenated-matrix-svd

# Install dependencies
pip install -r requirements.txt
```

## Algorithms

This repository implements several clustering algorithms:

### 1. **Max-Norm Algorithm**
Uses Weyl's inequality to estimate singular values based on matrix norms. Fast but conservative.

```bash
python -m src.main --dataset qualcomm --data_path <path> --algorithm max_norm --r_target 20 --eps 0.05
```

### 2. **Residuals Algorithm**
Computes residual norms after removing top singular components. More accurate than max-norm.

```bash
python -m src.main --dataset qualcomm --data_path <path> --algorithm residuals --sorting_strategy norm --r_target 20 --eps 0.05
```

### 3. **Approximate Algorithm** (Recommended)
Incrementally updates approximate singular values without forming the full matrix. Most efficient for large datasets.

```bash
python -m src.main --dataset qualcomm --data_path <path> --algorithm approximate --sorting_strategy residual --r_target 20 --eps 0.05
```

### Baseline Methods

- **Random**: Random clustering
- **K-Means**: Clustering based on matrix norms
- **HDBSCAN**: Density-based clustering

## Datasets

In our experiments we use 4 datasets:

### 1. Qualcomm Massive MIMO Dataset
Wireless channel matrices from massive MIMO systems.
- **Download**: [Qualcomm Dataset](https://www.qualcomm.com/developer/software/massive-mimo-spatial-channel-model-dataset)
- **Format**: HDF5 files with channel matrices (batch 17 used in the experiments)

### 2. BigEarthNet v2 S1
Sentinel-1 SAR satellite imagery for Earth observation.
- **Download**: [BigEarthNet](https://bigearth.net)
- **Format**: GeoTIFF files with multi-channel images

### 3. PDEBench (Advection)
Simulated solutions to partial differential equations.
- **Download**: [PDEBench](https://github.com/pdebench/PDEBench/tree/main/pdebench/data_download)
- **Format**: HDF5 files with 1D advection equation solutions

### 4. SmolVLM2-256M
Neural network weight matrices from vision-language model.
- **Download**: [SmolVLM2-256M](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct)
- **Format**: Pretrained PyTorch model weights

## Usage

### Basic Example

```python
from src.clustering import ApproximateClusterAlgorithm
from src.data import get_data_reader

# Load data
reader = get_data_reader('qualcomm', data_path='data/QDNData')
data = reader.read()

# Convert to blocks
blocks = [data[i] for i in range(len(data))]

# Cluster with error constraint
algorithm = ApproximateClusterAlgorithm(
    eps=0.05,              # 5% error threshold
    r_target=20,           # Target rank
    sorting_strategy='residual'
)
algorithm.fit(blocks)

# Get results
clusters = algorithm.clusters_
labels = algorithm.labels_
```

### Running Experiments

The repository includes shell scripts to reproduce paper experiments:

```bash
# Run main experiments (Table results)
bash run_experiments_table.sh

# Run baseline comparisons
bash run_baselines.sh

# Run error estimation analysis
bash run_error_estimations.sh

# Run scatter plot experiments
bash run_experiments_scatter.sh
```

### Command-Line Arguments

Key parameters:
- `--dataset`: Dataset to use (`qualcomm`, `bigearth`, `pdebench`, `smolvlm`)
- `--algorithm`: Clustering method (`approximate`, `max_norm`, `residuals`, `kmeans`, `hdbscan`, `random`)
- `--eps`: Error threshold (e.g., 0.05 for 5% relative error)
- `--r_target`: Target rank for low-rank approximation
- `--sorting_strategy`: Block ordering strategy (`norm` or `residual`)
- `--data_path`: Path to dataset files

Full options:
```bash
python -m src.main --help
```

## Results

The algorithms produce:
- **Cluster assignments**: Which blocks belong to each cluster
- **Reconstruction errors**: Per-cluster relative Frobenius norm errors
- **Compression ratios**: Effective compression achieved
- **Runtime statistics**: Processing time per algorithm

Results are logged to the console and saved to `logs/` directory.
