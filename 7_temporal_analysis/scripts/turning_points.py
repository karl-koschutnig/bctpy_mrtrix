#!/usr/bin/env python3
"""
Turning Point Detection
======================

Identify critical turning points in network state trajectories.

This script:
1. Loads UMAP coordinates and timepoint information
2. Detects turning points where the trajectory changes direction
3. Divides the trajectory into epochs based on turning points
4. Performs epoch-specific analysis (PCA, LASSO)

Usage:
    python turning_points.py --input-file umap_coordinates.csv --output-dir outputs/turning_points

Author: TDD Implementation
Date: 2026-09-10
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys


# ============================================================================
# OPTIONAL IMPORTS
# ============================================================================

# Optional matplotlib for visualization
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available, skipping visualization")

# Optional sklearn for PCA and LASSO
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available, skipping PCA and LASSO")


# ============================================================================
# TURNING POINT DETECTION
# ============================================================================

def detect_turning_points(embedding, gradient_threshold=0.8, window_size=1):
    """
    Detect turning points in a 2D trajectory.
    
    Args:
        embedding: n x 2 numpy array of coordinates
        gradient_threshold: angle threshold (in radians) for detecting turns
        window_size: number of points to look ahead/behind
    
    Returns:
        turning_indices: list of indices where turning points occur
    """
    if embedding.shape[0] < 3:
        return []
    
    if embedding.shape[1] < 2:
        return []
    
    n_points = embedding.shape[0]
    turning_indices = []
    
    # Calculate velocity vectors (first differences)
    velocities = np.diff(embedding, axis=0)
    
    # For each point (except first and last), check if it's a turning point
    for i in range(1, n_points - 1):
        # Get incoming and outgoing velocity vectors
        v_in = velocities[i - 1]  # velocity into point i
        v_out = velocities[i]      # velocity out of point i
        
        # Calculate angle between vectors
        norm_in = np.linalg.norm(v_in)
        norm_out = np.linalg.norm(v_out)
        
        if norm_in > 0 and norm_out > 0:
            # Dot product and angle
            dot_product = np.dot(v_in, v_out)
            cos_angle = dot_product / (norm_in * norm_out)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            
            # Check if angle exceeds threshold
            if angle > gradient_threshold:
                turning_indices.append(i)
    
    return turning_indices


def identify_epochs(turning_indices, n_timepoints):
    """
    Identify epochs based on turning points.
    
    Turning points mark the END of an epoch, not the start of a new one.
    
    Args:
        turning_indices: list of turning point indices (indices in the timepoint list)
        n_timepoints: total number of timepoints
    
    Returns:
        epochs: list of lists, where each sublist contains timepoint indices for an epoch
    """
    # Sort turning indices
    turning_indices = sorted(turning_indices)
    
    # Add boundary points
    # Turning points divide epochs, so we include them in the first epoch that ends at them
    boundaries = [0] + [tp + 1 for tp in turning_indices] + [n_timepoints]
    
    # Create epochs
    epochs = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        epochs.append(list(range(start, end)))
    
    return epochs


# ============================================================================
# EPOCH ANALYSIS
# ============================================================================

def pca_per_epoch(data, epochs, timepoint_col='session', metric_cols=None):
    """
    Perform PCA separately for each epoch.
    
    Args:
        data: DataFrame with metrics and timepoint information
        epochs: list of epoch definitions (each is a list of timepoint values)
        timepoint_col: column name for timepoints
        metric_cols: list of metric columns to use
    
    Returns:
        pca_results: dict with PCA results per epoch
    """
    if not SKLEARN_AVAILABLE:
        warnings.warn("scikit-learn not available, skipping PCA")
        return {}
    
    if metric_cols is None:
        metric_cols = [col for col in data.columns if col != timepoint_col]
    
    pca_results = {}
    
    for epoch_idx, epoch_timepoints in enumerate(epochs):
        # Get data for this epoch
        epoch_data = data[data[timepoint_col].isin(epoch_timepoints)]
        
        if len(epoch_data) < 2:
            continue
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(epoch_data[metric_cols])
        
        # PCA
        n_components = min(5, len(metric_cols), len(epoch_data) - 1)
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        
        # Get explained variance
        explained_variance = pca.explained_variance_ratio_
        
        pca_results[f'epoch_{epoch_idx}'] = {
            'timepoints': epoch_timepoints,
            'n_samples': len(epoch_data),
            'explained_variance': explained_variance.tolist(),
            'components': pca.components_.tolist()
        }
    
    return pca_results


def lasso_per_epoch(data, epochs, timepoint_col='session', metric_cols=None, 
                   group_col='group'):
    """
    Perform LASSO regression for each epoch to identify important metrics.
    
    Args:
        data: DataFrame with metrics and timepoint information
        epochs: list of epoch definitions
        timepoint_col: column name for timepoints
        metric_cols: list of metric columns to use as features
        group_col: column name for group (target variable)
    
    Returns:
        lasso_results: dict with LASSO results per epoch
    """
    if not SKLEARN_AVAILABLE:
        warnings.warn("scikit-learn not available, skipping LASSO")
        return {}
    
    if group_col not in data.columns:
        warnings.warn(f"Group column '{group_col}' not found in data")
        return {}
    
    if metric_cols is None:
        metric_cols = [col for col in data.columns 
                     if col not in [timepoint_col, group_col, 'participant_id', 'age', 'sex']]
    
    lasso_results = {}
    
    for epoch_idx, epoch_timepoints in enumerate(epochs):
        # Get data for this epoch
        epoch_data = data[data[timepoint_col].isin(epoch_timepoints)]
        
        if len(epoch_data) < 2:
            continue
        
        # Get features and target
        X = epoch_data[metric_cols].values
        y = epoch_data[group_col].astype('category').cat.codes.values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # LASSO with cross-validation
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X_scaled, y)
        
        # Get coefficients
        coefs = dict(zip(metric_cols, lasso.coef_))
        
        # Get non-zero coefficients (selected features)
        selected_features = [col for col, coef in coefs.items() if abs(coef) > 1e-5]
        
        lasso_results[f'epoch_{epoch_idx}'] = {
            'timepoints': epoch_timepoints,
            'n_samples': len(epoch_data),
            'alpha': lasso.alpha_,
            'n_selected_features': len(selected_features),
            'selected_features': selected_features,
            'coefficients': coefs
        }
    
    return lasso_results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_trajectory_with_turning_points(data, umap_cols=['umap_1', 'umap_2'],
                                       timepoint_col='session',
                                       turning_points=None, output_path=None):
    """
    Plot UMAP trajectory with turning points marked.
    
    Args:
        data: DataFrame with UMAP coordinates
        umap_cols: names of UMAP coordinate columns
        timepoint_col: column name for timepoints
        turning_points: list of turning point timepoint values
        output_path: path to save plot
    """
    if not MATPLOTLIB_AVAILABLE:
        warnings.warn("Matplotlib not available, skipping plot")
        return
    
    if turning_points is None:
        # Calculate turning points from data
        embedding = data[umap_cols].values
        turning_indices = detect_turning_points(embedding)
        turning_points = [data[timepoint_col].iloc[idx] for idx in turning_indices]
    
    plt.figure(figsize=(12, 10))
    
    # Plot all points
    unique_timepoints = data[timepoint_col].unique()
    colors = sns.color_palette('husl', len(unique_timepoints))
    
    for i, tp in enumerate(unique_timepoints):
        subset = data[data[timepoint_col] == tp]
        plt.scatter(
            subset[umap_cols[0]],
            subset[umap_cols[1]],
            c=[colors[i]],
            label=str(tp),
            s=100,
            alpha=0.7
        )
    
    # Mark turning points with larger, different color
    for tp in turning_points:
        subset = data[data[timepoint_col] == tp]
        plt.scatter(
            subset[umap_cols[0]],
            subset[umap_cols[1]],
            c='red',
            s=300,
            marker='*',
            edgecolor='black',
            linewidth=2,
            label='Turning Point' if tp == turning_points[0] else ""
        )
    
    # Add arrows showing trajectory
    unique_timepoints_sorted = sorted(unique_timepoints)
    for i in range(len(unique_timepoints_sorted) - 1):
        tp1 = unique_timepoints_sorted[i]
        tp2 = unique_timepoints_sorted[i + 1]
        
        subset1 = data[data[timepoint_col] == tp1]
        subset2 = data[data[timepoint_col] == tp2]
        
        x1 = subset1[umap_cols[0]].mean()
        y1 = subset1[umap_cols[1]].mean()
        x2 = subset2[umap_cols[0]].mean()
        y2 = subset2[umap_cols[1]].mean()
        
        plt.arrow(x1, y1, x2 - x1, y2 - y1,
                  head_width=0.05, head_length=0.1,
                  fc='gray', ec='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel(f'UMAP {umap_cols[0]}')
    plt.ylabel(f'UMAP {umap_cols[1]}')
    plt.title('Network State Trajectory with Turning Points')
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_epochs(data, epochs, umap_cols=['umap_1', 'umap_2'],
                timepoint_col='session', output_path=None):
    """
    Plot trajectory divided into epochs.
    
    Args:
        data: DataFrame with UMAP coordinates
        epochs: list of epoch definitions
        umap_cols: names of UMAP coordinate columns
        timepoint_col: column name for timepoints
        output_path: path to save plot
    """
    if not MATPLOTLIB_AVAILABLE:
        warnings.warn("Matplotlib not available, skipping plot")
        return
    
    plt.figure(figsize=(12, 10))
    
    colors = sns.color_palette('Set1', len(epochs))
    
    # Flatten all timepoints and assign epoch colors
    all_timepoints = []
    all_colors = []
    for epoch_idx, epoch_timepoints in enumerate(epochs):
        all_timepoints.extend(epoch_timepoints)
        all_colors.extend([colors[epoch_idx]] * len(epoch_timepoints))
    
    # Plot each timepoint
    for i, tp in enumerate(all_timepoints):
        subset = data[data[timepoint_col] == tp]
        plt.scatter(
            subset[umap_cols[0]],
            subset[umap_cols[1]],
            c=[all_colors[i]],
            s=200,
            alpha=0.7,
            edgecolor='black'
        )
    
    # Add epoch labels
    for epoch_idx, epoch_timepoints in enumerate(epochs):
        # Get center of epoch
        epoch_data = data[data[timepoint_col].isin(epoch_timepoints)]
        center_x = epoch_data[umap_cols[0]].mean()
        center_y = epoch_data[umap_cols[1]].mean()
        plt.text(center_x, center_y, f'Epoch {epoch_idx + 1}',
                 fontsize=14, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.xlabel(f'UMAP {umap_cols[0]}')
    plt.ylabel(f'UMAP {umap_cols[1]}')
    plt.title('Network State Trajectory by Epoch')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_turning_points(input_file, output_dir, timepoint_col='session',
                           umap_cols=None, group_col='group',
                           gradient_threshold=0.8):
    """
    Main processing function for turning point detection.
    
    Args:
        input_file: path to input CSV/parquet file with UMAP coordinates
        output_dir: directory to save results
        timepoint_col: column name for timepoints
        umap_cols: names of UMAP coordinate columns (None = auto-detect)
        group_col: column name for groups
        gradient_threshold: threshold for detecting turning points
    
    Returns:
        results: dict with turning points, epochs, and analysis results
    """
    # Load data
    if input_file.endswith('.parquet'):
        try:
            data = pd.read_parquet(input_file)
        except ImportError:
            data = pd.read_csv(input_file)
    else:
        data = pd.read_csv(input_file)
    
    # Auto-detect UMAP columns
    if umap_cols is None:
        umap_cols = [col for col in data.columns if col.startswith('umap_')]
        if len(umap_cols) < 2:
            umap_cols = ['umap_1', 'umap_2']  # default
    
    # Ensure we have 2D coordinates
    if len(umap_cols) < 2:
        raise ValueError(f"Need at least 2 UMAP columns, found: {umap_cols}")
    
    # Get unique timepoints
    unique_timepoints = sorted(data[timepoint_col].unique())
    n_timepoints = len(unique_timepoints)
    
    # Calculate average UMAP coordinates per timepoint
    timepoint_avgs = data.groupby(timepoint_col)[umap_cols].mean().reset_index()
    embedding = timepoint_avgs[umap_cols].values
    
    # Detect turning points
    turning_indices = detect_turning_points(embedding, gradient_threshold=gradient_threshold)
    turning_timepoints = [timepoint_avgs[timepoint_col].iloc[idx] for idx in turning_indices]
    
    # Identify epochs
    epochs = identify_epochs(turning_indices, n_timepoints)
    
    # Map epoch indices to timepoints
    epoch_assignments = {}
    for epoch_idx, epoch_tps in enumerate(epochs):
        for tp in epoch_tps:
            epoch_assignments[tp] = epoch_idx
    
    # Add epoch column to data
    data['epoch'] = data[timepoint_col].map(epoch_assignments)
    
    # Perform epoch analysis
    pca_results = {}
    lasso_results = {}
    
    if SKLEARN_AVAILABLE:
        # Get metric columns
        metric_cols = [col for col in data.columns 
                     if col not in [timepoint_col, group_col, 'participant_id', 
                                    'age', 'sex', 'epoch'] + umap_cols]
        
        # PCA per epoch
        pca_results = pca_per_epoch(
            data, epochs, timepoint_col=timepoint_col, metric_cols=metric_cols
        )
        
        # LASSO per epoch
        lasso_results = lasso_per_epoch(
            data, epochs, timepoint_col=timepoint_col, 
            metric_cols=metric_cols, group_col=group_col
        )
    
    # Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save turning points
    turning_points_file = output_dir / "turning_points.json"
    with open(turning_points_file, 'w') as f:
        json.dump({
            'turning_timepoints': turning_timepoints,
            'turning_indices': turning_indices,
            'epochs': epochs,
            'n_timepoints': n_timepoints
        }, f, indent=2)
    
    # Save data with epoch assignments
    data_file = output_dir / "data_with_epochs.csv"
    data.to_csv(data_file, index=False)
    
    # Save parquet if available
    try:
        data.to_parquet(output_dir / "data_with_epochs.parquet", index=False)
    except ImportError:
        pass
    
    # Save PCA results
    pca_file = output_dir / "pca_results.json"
    with open(pca_file, 'w') as f:
        json.dump(pca_results, f, indent=2)
    
    # Save LASSO results
    lasso_file = output_dir / "lasso_results.json"
    with open(lasso_file, 'w') as f:
        json.dump(lasso_results, f, indent=2)
    
    # Create visualizations
    if MATPLOTLIB_AVAILABLE:
        # Plot with turning points
        plot_trajectory_with_turning_points(
            data,
            umap_cols=umap_cols,
            timepoint_col=timepoint_col,
            turning_points=turning_timepoints,
            output_path=output_dir / "trajectory_with_turning_points.png"
        )
        
        # Plot epochs
        plot_epochs(
            data,
            epochs=epochs,
            umap_cols=umap_cols,
            timepoint_col=timepoint_col,
            output_path=output_dir / "trajectory_by_epoch.png"
        )
    
    return {
        'turning_points': turning_timepoints,
        'turning_indices': turning_indices,
        'epochs': epochs,
        'pca_results': pca_results,
        'lasso_results': lasso_results,
        'data': data
    }


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface for turning point detection."""
    parser = argparse.ArgumentParser(
        description='Turning point detection for UMAP trajectories'
    )
    parser.add_argument('--input-file', type=str, required=True,
                        help='Input CSV/parquet file with UMAP coordinates')
    parser.add_argument('--output-dir', type=str, default='outputs/turning_points',
                        help='Directory to save turning point results')
    parser.add_argument('--timepoint-col', type=str, default='session',
                        help='Column name for timepoints')
    parser.add_argument('--umap-cols', type=str, nargs='+', default=None,
                        help='Names of UMAP coordinate columns')
    parser.add_argument('--group-col', type=str, default='group',
                        help='Column name for groups')
    parser.add_argument('--gradient-threshold', type=float, default=0.8,
                        help='Threshold for detecting turning points (radians)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration JSON file')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        
        if 'temporal' in config and 'timepoint_col' in config['temporal']:
            args.timepoint_col = config['temporal']['timepoint_col']
        if 'temporal' in config and 'group_col' in config['temporal']:
            args.group_col = config['temporal']['group_col']
    
    # Process
    results = process_turning_points(
        input_file=args.input_file,
        output_dir=args.output_dir,
        timepoint_col=args.timepoint_col,
        umap_cols=args.umap_cols,
        group_col=args.group_col,
        gradient_threshold=args.gradient_threshold
    )
    
    print(f"Turning point detection complete")
    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_dir}")
    print(f"Timepoints: {results['data'][args.timepoint_col].nunique()}")
    print(f"Turning points detected: {len(results['turning_points'])}")
    print(f"Turning timepoints: {results['turning_points']}")
    print(f"Epochs identified: {len(results['epochs'])}")


if __name__ == "__main__":
    main()
