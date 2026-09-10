# 7. Temporal Analysis: Group x Time Network Trajectories

This module analyzes how graph-theory network metrics change across the
study's 3 timepoints and 5 study arms, for each of 5 parcellation
atlases.

## Overview

1. **Small-Worldness Calculation** - Network topology metric (sigma), per atlas
2. **Group x Time Mixed-Effects Models** - `lmer(metric ~ group*session + (1|participant_id))`,
   with `emmeans` pairwise group contrasts per timepoint
3. **UMAP Dimensionality Reduction** - Exploratory visualization of
   per-group network-state trajectories across timepoints

Adapted from the small-worldness/GAM/UMAP/turning-points chain in
`lifespan_topological_turning_points` (Mousley et al. 2025), but the
age-based GAM and turning-point detection are replaced by mixed-effects
models: the source method requires a continuous covariate with many
distinct values (91 ages, n~3800 cross-sectional subjects) to have
statistical meaning, and doesn't apply to a 3-timepoint repeated-measures
design. `turning_points.py` is kept in the repo but not called by the
active pipeline.

## Study Design

- **Data**: `/Volumes/Evo/data/129/connectomics/bct_input/<atlas>`
- **Atlases**: AAL3 (166 nodes), Gordon333 (333), HCP-MMP (360),
  Schaefer200 (200), Schaefer400 (400) - all 5 processed by default
- **Timepoints**: 3 (ses-1, ses-2, ses-3)
- **Groups**: 5 study arms - `ctrl`, `g1_2w`, `g1_4w`, `g2_2w`, `g2_4w`.
  `g1` vs `g2` is deliberately kept blinded (double-blind analysis); `2w`/`4w`
  is duration.
- **Subjects**: 126

## Directory Structure

```
7_temporal_analysis/
├── scripts/
│   ├── small_worldness.py      # Calculate small-worldness (sigma)
│   ├── mixed_models.R          # Group x Time mixed-effects models
│   ├── umap_projection.py      # UMAP dimensionality reduction
│   └── turning_points.py       # Retired from the active pipeline; not called
├── tests/
├── config/
│   └── run_spec_temporal.json
├── outputs/
│   └── <atlas>/
│       ├── small_worldness/
│       ├── mixed_model_results/
│       └── umap_results/
└── README.md
```

## Requirements

See `0_installation/README.md` — `install.sh` provisions both the Python
(`uv sync`) and R (`renv`) sides automatically.

## Usage

```bash
export CONNECTOMES_DIR=/path/to/your/bct_input
./0_installation/run_temporal_analysis.sh
```

Runs all 5 atlases in sequence; per-atlas output lands under
`outputs/temporal_analysis/<atlas>/`.
