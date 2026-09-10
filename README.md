# bctpy_mrtrix

Python (bctpy) reimplementation of a brain-connectivity graph-theory
analysis pipeline, adapted for a repeated-measures intervention study
(3 timepoints, 5 study arms) across 5 parcellation atlases.

## Setup

```bash
./0_installation/install.sh
```

This installs the Python environment via `uv` and the R environment via
`renv` (see `0_installation/README.md` for details, including one-time R
lockfile generation and troubleshooting).

## Verify

```bash
source .venv/bin/activate
python 0_installation/preflight_check.py run_spec.json --temporal --r
```

## Pipeline stages

See the numbered directories `1_utilities/` through `7_temporal_analysis/`,
each with its own README. Stage 7 (`7_temporal_analysis/`) is documented
separately since it has the most moving parts (Python + R).

## Data

Real subject data and results are never committed to this repository —
see `.gitignore`. Point the pipeline at your own data via `run_spec.json`
/ the `CONNECTOMES_DIR` environment variable documented in
`0_installation/run_temporal_analysis.sh`.
