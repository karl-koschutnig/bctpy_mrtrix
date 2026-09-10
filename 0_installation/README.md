# Installation & Environment Setup

This directory contains scripts for setting up the bctpy_mrtrix Python environment using **UV** (astral-sh/uv), a fast Python package manager.

## Quick Start

### 1. Install UV

UV is required for dependency management. Install it first:

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Verify installation:
```bash
uv --version
```

### 2. Set Up Virtual Environment

Run the installation script:
```bash
./0_installation/install.sh
```

This will:
- Detect compatible Python (3.9, 3.10, or 3.11)
- Create a virtual environment at `.venv/`
- Install all Python dependencies via `uv sync --all-extras` (core +
  temporal analysis extras, from `pyproject.toml`/`uv.lock`)
- Install/restore the R environment via `renv` (see [R
  Dependencies](#r-dependencies) below)

### 3. Activate Environment

```bash
source .venv/bin/activate
```

### 4. Verify Installation

Check core packages:
```bash
python 0_installation/preflight_check.py run_spec.json
```

Check temporal analysis packages:
```bash
python 0_installation/preflight_check.py run_spec.json --temporal
```

Check UV lock file:
```bash
python 0_installation/preflight_check.py run_spec.json --uv-lock
```

## Manual Installation

### Using UV Directly

```bash
# Create venv
uv venv .venv --python python3.11

# Install all dependencies (including temporal analysis) from pyproject.toml/uv.lock
uv sync --all-extras
```

`uv sync` reads `pyproject.toml` and creates/updates `uv.lock` automatically
if it's missing or out of date — there's no separate manual lock-generation
step.

### Using pip (Alternative)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[default]
```

## R Dependencies

R packages are managed with [renv](https://rstudio.github.io/renv/),
scoped to this repo (not your global R library). `install.sh` handles
this automatically:

- First run ever: generates `renv.lock` by installing the required
  packages (`mgcv`, `lme4`, `lmerTest`, `emmeans`, `tidyverse`,
  `jsonlite`, `argparse`, `R.matlab`, `arrow`) — can take several minutes.
- Every subsequent run: restores the exact locked versions via
  `renv::restore()` — fast.

To add a new R package: install it normally inside the project (`Rscript
-e "install.packages('somepkg')"` from the repo root — renv intercepts
this automatically once `.Rprofile` is present), then refresh the lock
file with `Rscript 0_installation/init_r_env.R`, then commit the updated
`renv.lock`.

## Environment Configuration

### pyproject.toml

The main configuration file defines:
- Core dependencies (required for all pipelines)
- Optional dependencies (for specific features)
- Temporal analysis extras

```toml
[project.optional-dependencies]
temporal = ["umap-learn>=0.5.0", "scikit-learn>=1.0.0"]
default = [...]  # All dependencies
```

### uv.lock

The `uv.lock` file at the **repository root** (not in `0_installation/`) provides **reproducible dependency resolution**. It's generated and kept up to date automatically by `uv sync`/`uv sync --all-extras` (run by `install.sh`) — there's no separate manual command to generate it.

**Important:** This is a UV-format lock file, not a pip-format requirements file. Do not use it with `pip install -r uv.lock`.

To install from the lock file:
```bash
uv sync  # Installs exact versions from uv.lock
# OR
uv pip sync uv.lock
```

This ensures everyone uses the exact same package versions.

## Troubleshooting

### "Couldn't parse requirement in uv.lock"

**Problem:** You tried to use `pip install -r uv.lock`

**Solution:** uv.lock is in **UV format**, not pip format. Use:
```bash
uv sync  # Correct way to install from uv.lock
# OR
uv pip sync uv.lock
```

### UV Not Found

```
ERROR: uv is required but not installed.
Install uv first: https://docs.astral.sh/uv/
```

**Solution:** Install UV as shown above.

### Python Not Found

```
ERROR: No compatible Python found.
Required: Python 3.9, 3.10, or 3.11
```

**Solution:** Install Python 3.9, 3.10, or 3.11 (recommended: 3.11).

### Missing Packages

```
✗ Missing required packages: bct, umap
```

**Solution:** Run `install.sh` or manually install with `uv sync --all-extras`.

## Files

| File | Purpose |
|------|---------|
| `install.sh` | Create venv, install all Python deps (`uv sync`), set up R (`renv`) |
| `preflight_check.py` | Validate environment before running pipeline |
| `init_r_env.R` | One-time: initialize renv and generate `renv.lock` |
| `setup_r_env.R` | Restore the R environment from `renv.lock` |
| `README.md` | This file |
