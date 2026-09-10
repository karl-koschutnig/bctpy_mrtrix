#!/usr/bin/env bash
# =============================================================================
# Temporal Analysis Workflow Script
# =============================================================================
#
# This script runs the complete temporal analysis pipeline for your
# interventional connectomics study.
#
# Requirements:
#   - Python environment activated (run install.sh first)
#   - R installed for GAM modeling
#   - Metadata file created
#
# Usage:
#   ./0_installation/run_temporal_analysis.sh
#
# Author: bctpy_mrtrix Team
# Date: 2026-09-10
# =============================================================================

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data/processed"
METADATA_FILE="${DATA_DIR}/metadata.csv"
CONNECTOMES_DIR="${ROOT_DIR}/data/connectomes"
OUTPUTS_DIR="${ROOT_DIR}/outputs/temporal_analysis"

# Atlas settings
ATLAS="Schaefer200"
N_NODES=200

# Study settings
TIMEPOINT_COL="session"
GROUP_COL="group"
N_TIMEPOINTS=3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ============================================================================
# FUNCTIONS
# ============================================================================

echo_header() {
    echo -e "${BLUE}========================================================================${NC}"
    echo -e "${BLUE}  Temporal Analysis Pipeline${NC}"
    echo -e "${BLUE}========================================================================${NC}"
    echo ""
}

echo_step() {
    echo -e "${YELLOW}=== $1 ===${NC}"
}

echo_success() {
    echo -e "${GREEN}✓ ${1}${NC}"
}

echo_error() {
    echo -e "${RED}✗ ${1}${NC}"
}

echo_info() {
    echo "  ${1}"
}

echo_command() {
    echo -e "${BLUE}  $ ${1}${NC}"
}

check_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo_error "$1 is not installed or not in PATH"
        return 1
    fi
    return 0
}

# ============================================================================
# CHECK ENVIRONMENT
# ============================================================================

check_environment() {
    echo_step "Checking Environment"
    
    # Check Python
    if ! check_command python; then
        echo_error "Python not found. Please activate the virtual environment."
        echo_info "Run: source ${ROOT_DIR}/.venv/bin/activate"
        exit 1
    fi
    echo_success "Python: $(python --version 2>&1)"
    
    # Check UV
    if ! check_command uv; then
        echo_error "UV not found. Please install UV first."
        echo_info "Run: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    echo_success "UV: $(uv --version)"
    
    # Check if venv is activated
    if [[ -z "$VIRTUAL_ENV" ]]; then
        echo_error "Virtual environment not activated"
        echo_info "Run: source ${ROOT_DIR}/.venv/bin/activate"
        exit 1
    fi
    echo_success "Virtual environment active: $VIRTUAL_ENV"
    
    # Check R for GAM
    if ! check_command Rscript; then
        echo_error "R not found. GAM modeling requires R."
        echo_info "Install R from: https://cran.r-project.org/"
        echo_info "Then install packages: Rscript -e \"install.packages(c('mgcv', 'tidyverse', 'jsonlite', 'argparse'), repos='https://cloud.r-project.org/')\""
        read -p "Continue without R? (y/n) [n]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo_success "R: $(Rscript --version 2>&1 | head -1)"
    fi
    
    echo ""
}

# ============================================================================
# CREATE METADATA FILE (if not exists)
# ============================================================================

create_metadata() {
    echo_step "Checking Metadata File"
    
    if [[ ! -f "$METADATA_FILE" ]]; then
        echo_info "Metadata file not found: $METADATA_FILE"
        echo_info "Creating template metadata.csv..."
        
        mkdir -p "$DATA_DIR"
        
        # Create a template metadata file
        cat > "$METADATA_FILE" << 'EOF'
participant_id,session,group,age,sex
sub-001,ses-1,ctrl,25,0
sub-001,ses-2,ctrl,25,0
sub-001,ses-3,ctrl,25,0
sub-002,ses-1,2w,30,1
sub-002,ses-2,2w,30,1
sub-002,ses-3,2w,30,1
sub-003,ses-1,4w,28,0
sub-003,ses-2,4w,28,0
sub-003,ses-3,4w,28,0
EOF
        
        echo_success "Template metadata.csv created at: $METADATA_FILE"
        echo_info "Please edit this file with your actual participant data."
        echo_info "Format: participant_id,session,group,age,sex"
        echo_info "Groups: ctrl (control), 2w (2-week), 4w (4-week)"
        echo ""
    else
        echo_success "Metadata file found: $METADATA_FILE"
        echo_info "Participants: $(csvcut -c participant_id $METADATA_FILE | tail -n +2 | sort -u | wc -l)"
        echo_info "Sessions: $(csvcut -c session $METADATA_FILE | tail -n +2 | sort -u | tr '\n' ',' | sed 's/,$//')"
        echo_info "Groups: $(csvcut -c group $METADATA_FILE | tail -n +2 | sort -u | tr '\n' ',' | sed 's/,$//')"
        echo ""
    fi
}

# ============================================================================
# RUN TEMPORAL ANALYSIS PIPELINE
# ============================================================================

run_pipeline() {
    echo_step "Running Temporal Analysis Pipeline"
    
    # Step 1: Small-worldness calculation
    echo_info "Step 1/4: Calculating small-worldness..."
    echo_command "python 7_temporal_analysis/scripts/small_worldness.py \\"
    echo_command "  --data-dir ${CONNECTOMES_DIR}/${ATLAS} \\"
    echo_command "  --metadata-file ${METADATA_FILE} \\"
    echo_command "  --output-dir ${OUTPUTS_DIR}/small_worldness \\"
    echo_command "  --n-nodes ${N_NODES}"
    
    python 7_temporal_analysis/scripts/small_worldness.py \
        --data-dir "${CONNECTOMES_DIR}/${ATLAS}" \
        --metadata-file "${METADATA_FILE}" \
        --output-dir "${OUTPUTS_DIR}/small_worldness" \
        --n-nodes ${N_NODES} || {
        echo_error "Small-worldness calculation failed"
        exit 1
    }
    echo_success "Small-worldness complete"
    echo ""
    
    # Step 2: GAM models (if R is available)
    if check_command Rscript; then
        echo_info "Step 2/4: Fitting GAM models..."
        echo_command "Rscript 7_temporal_analysis/scripts/gam_models.R \\"
        echo_command "  --input-file ${OUTPUTS_DIR}/small_worldness/small_worldness_results.csv \\"
        echo_command "  --output-dir ${OUTPUTS_DIR}/gam_results \\"
        echo_command "  --timepoint-col ${TIMEPOINT_COL} \\"
        echo_command "  --group-col ${GROUP_COL}"
        
        Rscript 7_temporal_analysis/scripts/gam_models.R \
            --input-file "${OUTPUTS_DIR}/small_worldness/small_worldness_results.csv" \
            --output-dir "${OUTPUTS_DIR}/gam_results" \
            --timepoint-col "${TIMEPOINT_COL}" \
            --group-col "${GROUP_COL}" || {
            echo_error "GAM modeling failed"
            exit 1
        }
        echo_success "GAM models complete"
        echo ""
    else
        echo_info "Skipping GAM models (R not available)"
        echo ""
    fi
    
    # Step 3: UMAP projection
    echo_info "Step 3/4: Running UMAP projection..."
    echo_command "python 7_temporal_analysis/scripts/umap_projection.py \\"
    echo_command "  --input-file ${OUTPUTS_DIR}/small_worldness/small_worldness_results.csv \\"
    echo_command "  --output-dir ${OUTPUTS_DIR}/umap_results \\"
    echo_command "  --timepoint-col ${TIMEPOINT_COL} \\"
    echo_command "  --group-col ${GROUP_COL}"
    
    python 7_temporal_analysis/scripts/umap_projection.py \
        --input-file "${OUTPUTS_DIR}/small_worldness/small_worldness_results.csv" \
        --output-dir "${OUTPUTS_DIR}/umap_results" \
        --timepoint-col "${TIMEPOINT_COL}" \
        --group-col "${GROUP_COL}" || {
        echo_error "UMAP projection failed"
        exit 1
    }
    echo_success "UMAP projection complete"
    echo ""
    
    # Step 4: Turning point detection
    echo_info "Step 4/4: Detecting turning points..."
    echo_command "python 7_temporal_analysis/scripts/turning_points.py \\"
    echo_command "  --input-file ${OUTPUTS_DIR}/umap_results/umap_coordinates.csv \\"
    echo_command "  --output-dir ${OUTPUTS_DIR}/turning_points \\"
    echo_command "  --timepoint-col ${TIMEPOINT_COL} \\"
    echo_command "  --group-col ${GROUP_COL}"
    
    python 7_temporal_analysis/scripts/turning_points.py \
        --input-file "${OUTPUTS_DIR}/umap_results/umap_coordinates.csv" \
        --output-dir "${OUTPUTS_DIR}/turning_points" \
        --timepoint-col "${TIMEPOINT_COL}" \
        --group-col "${GROUP_COL}" || {
        echo_error "Turning point detection failed"
        exit 1
    }
    echo_success "Turning point detection complete"
    echo ""
}

# ============================================================================
# SHOW RESULTS
# ============================================================================

show_results() {
    echo_step "Pipeline Complete!"
    echo ""
    
    echo_success "Results saved to: ${OUTPUTS_DIR}/"
    echo ""
    
    echo_info "Output directories:"
    for dir in small_worldness gam_results umap_results turning_points; do
        if [[ -d "${OUTPUTS_DIR}/${dir}" ]]; then
            echo_info "  ✓ ${OUTPUTS_DIR}/${dir}/"
        fi
    done
    echo ""
    
    echo_info "Key output files:"
    if [[ -f "${OUTPUTS_DIR}/small_worldness/small_worldness_results.csv" ]]; then
        echo_info "  ✓ small_worldness_results.csv"
    fi
    if [[ -f "${OUTPUTS_DIR}/gam_results/gam_model_statistics.csv" ]]; then
        echo_info "  ✓ gam_model_statistics.csv"
    fi
    if [[ -f "${OUTPUTS_DIR}/umap_results/umap_coordinates.csv" ]]; then
        echo_info "  ✓ umap_coordinates.csv"
    fi
    if [[ -f "${OUTPUTS_DIR}/turning_points/turning_points.json" ]]; then
        echo_info "  ✓ turning_points.json"
    fi
    if [[ -f "${OUTPUTS_DIR}/umap_results/umap_trajectory_averages.png" ]]; then
        echo_info "  ✓ umap_trajectory_averages.png"
    fi
    if [[ -f "${OUTPUTS_DIR}/turning_points/trajectory_with_turning_points.png" ]]; then
        echo_info "  ✓ trajectory_with_turning_points.png"
    fi
    echo ""
    
    echo_info "To view results:"
    echo_info "  open ${OUTPUTS_DIR}/umap_results/umap_trajectory_averages.png"
    echo_info "  open ${OUTPUTS_DIR}/turning_points/trajectory_with_turning_points.png"
    echo ""
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    echo_header
    
    check_environment
    create_metadata
    run_pipeline
    show_results
    
    echo_header
}

main "$@"
