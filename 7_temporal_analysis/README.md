# 7. Temporal Analysis: Topological Turning Points

This directory contains scripts for temporal analysis of connectomics data, adapted from the lifespan_topological_turning_points pipeline.

## Overview

This module extends the bctpy_mrtrix pipeline with:
1. **Small-Worldness Calculation** - Network topology metric (sigma)
2. **Generalized Additive Models (GAM)** - Smooth modeling of metrics across timepoints
3. **UMAP Dimensionality Reduction** - Visualization of network state trajectories
4. **Turning Point Detection** - Identification of critical transitions

## Study Design

- **Data**: /Volumes/Evo/data/129/connectomics/bct_input/Schaefer200
- **Atlas**: Schaefer 200
- **Timepoints**: 3 (ses-1, ses-2, ses-3)
- **Groups**: 3 (ctrl, 2w, 4w) - neglecting social/solo differentiation
- **Subjects**: ~146+ subjects

## Directory Structure

```
7_temporal_analysis/
├── scripts/
│   ├── small_worldness.py      # Calculate small-worldness (sigma)
│   ├── 01_gam_models.R          # GAM modeling in R
│   ├── 02_umap_projection.py    # UMAP dimensionality reduction
│   └── 03_turning_points.py     # Turning point detection
├── tests/
│   ├── test_small_worldness.py
│   ├── test_gam_models.R
│   ├── test_umap_projection.py
│   └── test_turning_points.py
├── config/
│   └── run_spec_temporal.json
├── outputs/
│   ├── gam_results/
│   ├── umap_results/
│   ├── turning_points/
│   └── small_worldness/
└── README.md
```

## Requirements

### Python
- numpy
- pandas
- scipy
- umap-learn
- scikit-learn
- matplotlib
- seaborn
- bctpy (for graph metrics)

### R
- mgcv (for GAM)
- tidyverse
- R.matlab (for .mat file I/O)

## Usage

Run scripts in order:
1. `python scripts/small_worldness.py` - Calculate sigma for all connectomes
2. `Rscript scripts/01_gam_models.R` - Fit GAM models
3. `python scripts/02_umap_projection.py` - Create UMAP embeddings
4. `python scripts/03_turning_points.py` - Detect turning points

## Inputs

- Connectomes: CSV matrices from /Volumes/Evo/data/129/connectomics/bct_input/Schaefer200
- Metadata: CSV with columns: participant_id, session, group, age, sex

## Outputs

- small_worldness: sigma values per subject/session
- gam_results: GAM model fits and predictions
- umap_results: UMAP coordinates and plots
- turning_points: Identified critical transitions
