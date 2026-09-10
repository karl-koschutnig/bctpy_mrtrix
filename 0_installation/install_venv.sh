#!/usr/bin/env bash
set -euo pipefail

# Create venv and install packages with uv
# This script sets up a reproducible Python environment for bctpy_mrtrix
# including the new temporal analysis pipeline

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

# ============================================================================
# CHECK DEPENDENCIES
# ============================================================================

echo "=== bctpy_mrtrix Installation ==="
echo ""

# Check for uv
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required but not installed."
  echo ""
  echo "Install uv first:"
  echo "  macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "  Windows: powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\""
  echo ""
  echo "Or visit: https://docs.astral.sh/uv/"
  exit 1
fi

echo "✓ uv found: $(uv --version)"

# Check Python
PYTHON_BIN=""
if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
elif command -v python3.10 >/dev/null 2>&1; then
  PYTHON_BIN="python3.10"
elif command -v python3.9 >/dev/null 2>&1; then
  PYTHON_BIN="python3.9"
else
  echo "ERROR: No compatible Python found."
  echo "Required: Python 3.9, 3.10, or 3.11"
  echo "Recommended: Python 3.11 for best compatibility"
  exit 1
fi

echo "✓ Python found: $PYTHON_BIN"
echo ""

# ============================================================================
# CREATE VIRTUAL ENVIRONMENT
# ============================================================================

echo "Creating virtual environment at: $VENV_DIR"
uv venv "$VENV_DIR" --python "$PYTHON_BIN"
echo "✓ Virtual environment created"
echo ""

# ============================================================================
# INSTALL CORE DEPENDENCIES
# ============================================================================

echo "Installing core dependencies..."
uv pip install --python "$VENV_DIR/bin/python" \
  numpy \
  pandas \
  bctpy \
  scipy \
  matplotlib \
  seaborn \
  openpyxl \
  h5py \
  pyarrow \
  statsmodels \
  flask \
  waitress

echo "✓ Core dependencies installed"
echo ""

# ============================================================================
# INSTALL TEMPORAL ANALYSIS DEPENDENCIES
# ============================================================================

echo "Installing temporal analysis dependencies..."
uv pip install --python "$VENV_DIR/bin/python" \
  umap-learn \
  scikit-learn

echo "✓ Temporal analysis dependencies installed"
echo ""

# ============================================================================
# INSTALL R DEPENDENCIES (for GAM modeling)
# ============================================================================

echo "R packages for GAM modeling (install manually if needed):"
echo "  Rscript -e \"install.packages(c('mgcv', 'tidyverse', 'jsonlite', 'argparse', 'R.matlab'), repos='https://cloud.r-project.org/')\""
echo ""

# ============================================================================
# FINALIZE
# ============================================================================

echo "=== Installation Complete ==="
echo ""
echo "Virtual environment: $VENV_DIR"
echo ""
echo "To activate:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run pipeline:"
echo "  source $VENV_DIR/bin/activate"
echo "  python 1_utilities/..."
echo ""
echo "To run temporal analysis:"
echo "  source $VENV_DIR/bin/activate"
echo "  python 7_temporal_analysis/scripts/small_worldness.py --data-dir /path/to/data --metadata-file metadata.csv"
echo ""

# Generate uv.lock for reproducibility
echo "Generating uv.lock for reproducible installs..."
cd "$ROOT_DIR"
uv pip compile pyproject.toml --all-extras -o uv.lock
echo "✓ uv.lock updated"
