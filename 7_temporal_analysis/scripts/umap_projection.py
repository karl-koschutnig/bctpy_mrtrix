#!/usr/bin/env python3
"""
UMAP Projection and Turning Point Detection
===========================================

Perform UMAP dimensionality reduction on network metrics to visualize
network state trajectories across timepoints.

This script:
1. Loads network metrics (density, efficiency, etc.) from CSV/parquet
2. Performs UMAP dimensionality reduction
3. Detects turning points in the trajectory
4. Saves UMAP coordinates and visualizations

Usage:
    python 02_umap_projection.py --input-file metrics.csv --output-dir outputs/umap --timepoint-col session

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

# Optional tqdm for progress bar
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Optional umap
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available, using PCA fallback")

# Optional matplotlib for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available, skipping visualization")

# Optional sklearn for PCA fallback
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================================
# FALLBACK IMPLEMENTATIONS
# ============================================================================

def pca_fallback(X, n_components=2):
    """Fallback to PCA if UMAP not available."""
    if not SKLEARN_AVAILABLE:
        # Very simple manual PCA (for testing only)
        # Center the data
        X_centered = X - X.mean(axis=0)
        # Covariance matrix
        if X_centered.shape[0] > 1:
            cov = np.cov(X_centered, rowvar=False)
        else:
            return np.zeros((X.shape[0], n_components))
        # Eigen decomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Sort by eigenvalue (descending)
            idx = np.argsort(eigenvalues)[::-1][:n_components]
            components = eigenvectors[:, idx]
            return X_centered @ components
        except:
            # If SVD fails, return zeros
            return np.zeros((X.shape[0], n_components))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X_scaled)


# ============================================================================
# UMAP FUNCTIONS
# ============================================================================

def run_umap(data, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Run UMAP dimensionality reduction on the input data.
    
    Args:
        data: DataFrame or numpy array of features
        n_components: number of UMAP dimensions
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: random seed for reproducibility
    
    Returns:
        embedding: n_samples x n_components array
    """
    if isinstance(data, pd.DataFrame):
        X = data.values
    else:
        X = data
    
    # Standardize
    if SKLEARN_AVAILABLE:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Run UMAP or PCA fallback
    if UMAP_AVAILABLE:
        reducer = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            metric='euclidean'
        )
        embedding = reducer.fit_transform(X_scaled)
    else:
        embedding = pca_fallback(X_scaled, n_components=n_components)
    
    return embedding


def calculate_timepoint_averages(data, timepoint_col='session', metric_cols=None):
    """
    Calculate average metrics for each timepoint.
    
    Args:
        data: DataFrame with metrics and timepoint column
        timepoint_col: name of the timepoint column
        metric_cols: list of metric columns to average (None = all numeric columns)
    
    Returns:
        averages: DataFrame with one row per timepoint, containing average metrics
    """
    if metric_cols is None:
        # Get all numeric columns except timepoint_col
        metric_cols = [col for col in data.select_dtypes(include=[np.number]).columns 
                      if col != timepoint_col]
    
    # Group by timepoint and calculate mean
    averages = data.groupby(timepoint_col)[metric_cols].mean().reset_index()
    
    return averages


def detect_turning_points(embedding, gradient_threshold=0.8):
    """
    Detect turning points in the UMAP trajectory.
    
    A turning point is identified where the direction of the trajectory
    changes significantly (gradient > threshold).
    
    Args:
        embedding: n_samples x 2 array of UMAP coordinates
        gradient_threshold: threshold for detecting turning points
    
    Returns:
        turning_indices: list of indices where turning points occur
    """
    if embedding.shape[0] < 3:
        return []  # Need at least 3 points to detect turning points
    
    # Calculate differences between consecutive points
    diffs = np.diff(embedding, axis=0)
    
    # Calculate angles between consecutive differences (direction changes)
    angles = []
    for i in range(len(diffs) - 1):
        v1 = diffs[i]
        v2 = diffs[i + 1]
        
        # Normalize vectors
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 > 0 and norm_v2 > 0:
            # Cosine of angle between vectors
            cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
            # Angle in radians
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angles.append(angle)
        else:
            angles.append(0)
    
    # Find indices where angle > threshold
    turning_indices = []
    for i, angle in enumerate(angles):
        if angle > gradient_threshold:
            # Turning point is at position i+1 (between i and i+2)
            turning_indices.append(i + 1)
    
    return turning_indices


def plot_umap_trajectory(data, umap_cols=['umap_1', 'umap_2'], color_col=None,
                          output_path=None, show_labels=True):
    """
    Create a scatter plot of UMAP trajectory colored by timepoint or group.
    
    Args:
        data: DataFrame with UMAP coordinates and metadata
        umap_cols: names of UMAP coordinate columns
        color_col: column to use for coloring (default: session)
        output_path: path to save plot (optional)
        show_labels: whether to show timepoint labels
    """
    if not MATPLOTLIB_AVAILABLE:
        warnings.warn("Matplotlib not available, skipping plot")
        return
    
    if color_col is None:
        color_col = 'session' if 'session' in data.columns else 'group'
    
    plt.figure(figsize=(10, 8))
    
    # Get unique timepoints for coloring
    if color_col in data.columns:
        unique_values = data[color_col].unique()
        colors = sns.color_palette('husl', len(unique_values))
        
        for i, value in enumerate(unique_values):
            subset = data[data[color_col] == value]
            plt.scatter(
                subset[umap_cols[0]],
                subset[umap_cols[1]],
                c=[colors[i]],
                label=str(value),
                s=100,
                alpha=0.7
            )
            
            # Add labels if requested
            if show_labels:
                for _, row in subset.iterrows():
                    plt.text(
                        row[umap_cols[0]],
                        row[umap_cols[1]],
                        str(row.get('participant_id', '')),
                        fontsize=8,
                        alpha=0.5
                    )
    else:
        plt.scatter(
            data[umap_cols[0]],
            data[umap_cols[1]],
            c='blue',
            s=100,
            alpha=0.7
        )
    
    plt.xlabel(f'UMAP {umap_cols[0]}')
    plt.ylabel(f'UMAP {umap_cols[1]}')
    plt.title('Network State Trajectory (UMAP)')
    
    if color_col in data.columns:
        plt.legend(title=color_col)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_timepoint_averages(averages, umap_cols=['umap_1', 'umap_2'], 
                           color_col='session', output_path=None):
    """
    Plot UMAP trajectory of timepoint averages.
    
    Args:
        averages: DataFrame with timepoint averages and UMAP coordinates
        umap_cols: names of UMAP coordinate columns
        color_col: column to use for coloring
        output_path: path to save plot
    """
    if not MATPLOTLIB_AVAILABLE:
        warnings.warn("Matplotlib not available, skipping plot")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Plot trajectory with arrows
    if umap_cols[0] in averages.columns and umap_cols[1] in averages.columns:
        x = averages[umap_cols[0]].values
        y = averages[umap_cols[1]].values
        
        # Plot points
        if color_col in averages.columns:
            unique_values = averages[color_col].unique()
            colors = sns.color_palette('husl', len(unique_values))
            
            for i, value in enumerate(unique_values):
                idx = averages[color_col] == value
                plt.scatter(x[idx], y[idx], c=[colors[i]], label=str(value), s=300, alpha=0.8)
                
                # Add text label
                plt.text(x[idx][0], y[idx][0], str(value), fontsize=12, ha='center')
        
        # Plot arrows between timepoints
        for i in range(len(x) - 1):
            plt.arrow(x[i], y[i], x[i+1] - x[i], y[i+1] - y[i],
                      head_width=0.05, head_length=0.1, fc='gray', ec='gray',
                      linestyle='--', alpha=0.5)
        
        plt.xlabel(f'UMAP {umap_cols[0]}')
        plt.ylabel(f'UMAP {umap_cols[1]}')
        plt.title('Network State Trajectory - Timepoint Averages')
        
        if color_col in averages.columns:
            plt.legend(title=color_col)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_umap_projection(input_file, output_dir, timepoint_col='session',
                           metric_cols=None, n_components=2, n_neighbors=15,
                           min_dist=0.1, random_state=42, color_col=None):
    """
    Main processing function for UMAP projection.
    
    Args:
        input_file: path to input CSV/parquet file with metrics
        output_dir: directory to save outputs
        timepoint_col: column name for timepoints
        metric_cols: list of metric columns to use (None = auto-detect)
        n_components: number of UMAP dimensions
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: random seed
        color_col: column to use for coloring
    
    Returns:
        results: DataFrame with UMAP coordinates
    """
    # Load data
    if input_file.endswith('.parquet'):
        data = pd.read_parquet(input_file)
    else:
        data = pd.read_csv(input_file)
    
    # Auto-detect metric columns
    if metric_cols is None:
        exclude_cols = ['participant_id', 'session', 'group', 'age', 'sex', timepoint_col]
        metric_cols = [col for col in data.columns if col not in exclude_cols]
    
    # Run UMAP
    embedding = run_umap(
        data[metric_cols],
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    
    # Add UMAP coordinates to data
    for i in range(n_components):
        data[f'umap_{i+1}'] = embedding[:, i]
    
    # Calculate timepoint averages
    averages = calculate_timepoint_averages(
        data[metric_cols + [timepoint_col]],
        timepoint_col=timepoint_col,
        metric_cols=metric_cols
    )
    
    # Add UMAP coordinates to averages
    for i in range(n_components):
        # Calculate average UMAP coordinates per timepoint
        umap_avgs = data.groupby(timepoint_col)[f'umap_{i+1}'].mean().reset_index()
        averages[f'umap_{i+1}'] = umap_avgs[f'umap_{i+1}']
    
    # Detect turning points on timepoint averages
    if n_components >= 2:
        embedding_2d = averages[[f'umap_{i+1}' for i in range(min(2, n_components))]].values
        turning_indices = detect_turning_points(embedding_2d)
        turning_timepoints = [averages[timepoint_col].iloc[idx] for idx in turning_indices]
    else:
        turning_timepoints = []
    
    # Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save UMAP coordinates
    umap_coords_file = output_dir / "umap_coordinates.csv"
    data.to_csv(umap_coords_file, index=False)
    
    # Save parquet if available
    try:
        data.to_parquet(output_dir / "umap_coordinates.parquet", index=False)
    except ImportError:
        pass
    
    # Save timepoint averages
    averages_file = output_dir / "umap_timepoint_averages.csv"
    averages.to_csv(averages_file, index=False)
    
    # Save turning points
    turning_file = output_dir / "turning_points.json"
    with open(turning_file, 'w') as f:
        json.dump({
            'turning_timepoints': turning_timepoints,
            'turning_indices': turning_indices
        }, f, indent=2)
    
    # Create visualizations
    if MATPLOTLIB_AVAILABLE:
        # Plot all points
        plot_umap_trajectory(
            data,
            umap_cols=[f'umap_{i+1}' for i in range(min(2, n_components))],
            color_col=color_col or timepoint_col,
            output_path=output_dir / "umap_trajectory_all.png",
            show_labels=False
        )
        
        # Plot timepoint averages with arrows
        plot_timepoint_averages(
            averages,
            umap_cols=[f'umap_{i+1}' for i in range(min(2, n_components))],
            color_col=color_col or timepoint_col,
            output_path=output_dir / "umap_trajectory_averages.png"
        )
    
    return data, averages, turning_timepoints


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface for UMAP projection."""
    parser = argparse.ArgumentParser(
        description='UMAP dimensionality reduction for network metrics'
    )
    parser.add_argument('--input-file', type=str, required=True,
                        help='Input CSV/parquet file with network metrics')
    parser.add_argument('--output-dir', type=str, default='outputs/umap_results',
                        help='Directory to save UMAP results')
    parser.add_argument('--timepoint-col', type=str, default='session',
                        help='Column name for timepoints')
    parser.add_argument('--group-col', type=str, default='group',
                        help='Column name for groups (for coloring)')
    parser.add_argument('--n-components', type=int, default=2,
                        help='Number of UMAP dimensions')
    parser.add_argument('--n-neighbors', type=int, default=15,
                        help='UMAP n_neighbors parameter')
    parser.add_argument('--min-dist', type=float, default=0.1,
                        help='UMAP min_dist parameter')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration JSON file')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        
        if 'temporal' in config and 'timepoint_col' in config['temporal']:
            args.timepoint_col = config['temporal']['timepoint_col']
        if 'umap' in config:
            if 'n_components' in config['umap']:
                args.n_components = config['umap']['n_components']
            if 'n_neighbors' in config['umap']:
                args.n_neighbors = config['umap']['n_neighbors']
            if 'min_dist' in config['umap']:
                args.min_dist = config['umap']['min_dist']
            if 'random_state' in config['umap']:
                args.random_state = config['umap']['random_state']
    
    # Run UMAP
    data, averages, turning_points = process_umap_projection(
        input_file=args.input_file,
        output_dir=args.output_dir,
        timepoint_col=args.timepoint_col,
        color_col=args.group_col,
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state
    )
    
    print(f"UMAP projection complete")
    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_dir}")
    print(f"Samples: {len(data)}")
    print(f"Dimensions: {args.n_components}")
    print(f"Timepoints: {data[args.timepoint_col].nunique()}")
    print(f"Turning points detected: {len(turning_points)}")
    print(f"Turning timepoints: {turning_points}")


if __name__ == "__main__":
    main()
