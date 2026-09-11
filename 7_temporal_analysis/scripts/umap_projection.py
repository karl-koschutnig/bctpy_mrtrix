#!/usr/bin/env python3
"""
UMAP Projection and Multi-Metric Trajectory Analysis
====================================================

Condense the full organizational-measure battery into a single 2D manifold
(à la Mousley et al.) and read each group's network-state trajectory across
the three study timepoints.

This script:
1. Loads network metrics (density, efficiency, etc.) from CSV/parquet
2. Performs UMAP dimensionality reduction on all measures at once
3. Draws one manifold with every group's mean trajectory (ses1->ses2->ses3)
4. Per subject, measures the mid-study "turn" at ses-2 (angle between the
   ses1->ses2 and ses2->ses3 displacement vectors) and tests it by group.
   With only 3 timepoints a trajectory has exactly one interior angle, so
   the source paper's multi-turning-point curve fitting does not transfer -
   this is its scaled-down analogue.
5. Saves UMAP coordinates and visualizations

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

# Optional scipy for the turn-angle group tests
try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


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


def calculate_timepoint_averages(data, timepoint_col='session', group_col=None, metric_cols=None):
    """
    Calculate average metrics for each (group, timepoint) combination.

    Args:
        data: DataFrame with metrics and timepoint (and optionally group) columns
        timepoint_col: name of the timepoint column
        group_col: name of the group column; if given, averages are computed
            per (group, timepoint) instead of collapsing across groups
        metric_cols: list of metric columns to average (None = all numeric columns)

    Returns:
        averages: DataFrame with one row per (group, timepoint) [or per
            timepoint if group_col is None], containing average metrics
    """
    group_keys = [group_col, timepoint_col] if group_col else [timepoint_col]

    if metric_cols is None:
        exclude = set(group_keys)
        metric_cols = [col for col in data.select_dtypes(include=[np.number]).columns
                      if col not in exclude]

    averages = data.groupby(group_keys)[metric_cols].mean().reset_index()
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


def compute_subject_trajectories(data, participant_col='participant_id',
                                 timepoint_col='session', group_col='group',
                                 umap_cols=('umap_1', 'umap_2')):
    """
    Per subject, walk the UMAP trajectory through its ordered timepoints and
    measure how sharply it turns.

    For a subject with k ordered points there are k-2 interior angles; with
    the study's 3 timepoints that is exactly one, the turn at ses-2. Angle
    0 = the network kept moving the same direction; angle pi = it fully
    reversed.

    Returns one row per subject that has >= 3 timepoints present:
        participant, group, n_timepoints, seg_len_mean, path_length,
        net_displacement, turn_angle_rad, turn_angle_deg
    """
    umap_cols = list(umap_cols)
    rows = []
    for pid, sub in data.groupby(participant_col):
        sub = sub.sort_values(timepoint_col)
        pts = sub[umap_cols].to_numpy(dtype=float)
        if len(pts) < 3:
            continue
        diffs = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        angles = []
        for i in range(len(diffs) - 1):
            n1, n2 = seg_lens[i], seg_lens[i + 1]
            if n1 > 0 and n2 > 0:
                cos_a = np.dot(diffs[i], diffs[i + 1]) / (n1 * n2)
                angles.append(np.arccos(np.clip(cos_a, -1.0, 1.0)))
        if not angles:
            continue
        turn = float(np.mean(angles))
        rows.append({
            participant_col: pid,
            group_col: sub[group_col].iloc[0],
            "n_timepoints": len(pts),
            "seg_len_mean": float(np.mean(seg_lens)),
            "path_length": float(np.sum(seg_lens)),
            "net_displacement": float(np.linalg.norm(pts[-1] - pts[0])),
            "turn_angle_rad": turn,
            "turn_angle_deg": float(np.degrees(turn)),
        })
    return pd.DataFrame(rows)


def test_trajectory_by_group(traj, group_col='group',
                             measures=('turn_angle_deg', 'path_length', 'net_displacement')):
    """
    One-way group comparison (Kruskal-Wallis + one-way ANOVA) for each
    trajectory measure. Returns a tidy DataFrame; empty if scipy is missing
    or fewer than two groups have >= 2 subjects.
    """
    if not SCIPY_AVAILABLE or traj.empty:
        return pd.DataFrame()
    out = []
    for m in measures:
        groups = [g[m].dropna().to_numpy() for _, g in traj.groupby(group_col)]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            continue
        kw_h, kw_p = scipy_stats.kruskal(*groups)
        f_stat, f_p = scipy_stats.f_oneway(*groups)
        out.append({
            "measure": m, "n_groups": len(groups),
            "kruskal_H": float(kw_h), "kruskal_p": float(kw_p),
            "anova_F": float(f_stat), "anova_p": float(f_p),
        })
    return pd.DataFrame(out)


def plot_manifold_trajectories(data, averages, umap_cols=('umap_1', 'umap_2'),
                               group_col='group', timepoint_col='session',
                               output_path=None):
    """
    One manifold: every subject-session as a faint point, plus each group's
    mean trajectory drawn as a connected path through the ordered timepoints.
    """
    if not MATPLOTLIB_AVAILABLE:
        warnings.warn("Matplotlib not available, skipping manifold plot")
        return
    x, y = list(umap_cols)
    groups = list(averages[group_col].unique())
    palette = sns.color_palette('husl', len(groups))

    def draw(ax):
        ax.scatter(data[x], data[y], s=12, c='0.8', alpha=0.5, linewidths=0, zorder=1)
        for gi, gval in enumerate(groups):
            gdf = averages[averages[group_col] == gval].sort_values(timepoint_col)
            gx, gy = gdf[x].to_numpy(), gdf[y].to_numpy()
            ax.plot(gx, gy, '-o', color=palette[gi], lw=2, ms=9,
                    label=str(gval), zorder=3)
            for j in range(len(gx) - 1):
                ax.annotate('', xy=(gx[j + 1], gy[j + 1]), xytext=(gx[j], gy[j]),
                            arrowprops=dict(arrowstyle='-|>', color=palette[gi], lw=2),
                            zorder=3)
            ax.text(gx[0], gy[0], f' {gval}', color=palette[gi], fontsize=9,
                    va='center', zorder=4)
        ax.set_xlabel(f'UMAP {x}')
        ax.set_ylabel(f'UMAP {y}')

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(18, 9))
    draw(ax0)
    ax0.set_title('Full manifold (all subject-sessions)')
    ax0.legend(title=group_col, fontsize=8)
    draw(ax1)
    # zoom the second panel to the group-trajectory extent + 20% margin
    tx, ty = averages[x].to_numpy(), averages[y].to_numpy()
    mx = 0.2 * (tx.max() - tx.min() or 1.0)
    my = 0.2 * (ty.max() - ty.min() or 1.0)
    ax1.set_xlim(tx.min() - mx, tx.max() + mx)
    ax1.set_ylim(ty.min() - my, ty.max() + my)
    ax1.set_title('Zoom: mean trajectory per group (ses1 -> ses2 -> ses3)')
    fig.suptitle('Multi-metric network manifold', fontsize=14)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


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
                           group_col=None, metric_cols=None, n_components=2, n_neighbors=15,
                           min_dist=0.1, random_state=42, color_col=None,
                           participant_col='participant_id'):
    """
    Main processing function for UMAP projection.

    Args:
        input_file: path to input CSV/parquet file with metrics
        output_dir: directory to save outputs
        timepoint_col: column name for timepoints
        group_col: column name for groups; when given, trajectory averages
            are computed per (group, timepoint) instead of collapsing
            across groups
        metric_cols: list of metric columns to use (None = auto-detect)
        n_components: number of UMAP dimensions
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: random seed
        color_col: column to use for coloring in plots (defaults to group_col or timepoint_col)

    Returns:
        (data, averages): DataFrames with per-subject and per-(group,timepoint)
            UMAP coordinates respectively. Turning-point detection is not run
            here - it requires a continuous covariate with many distinct
            values, which a fixed-timepoint design doesn't have.
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

    # Drop rows with NaN metrics before projection. This module is
    # exploratory-only visualization (not the primary statistical analysis),
    # so dropping is safe here - unlike mixed_models.R, which needs the full
    # decision about missing data left to the study team.
    n_before = len(data)
    data = data.dropna(subset=metric_cols).reset_index(drop=True)
    n_dropped = n_before - len(data)
    if n_dropped > 0:
        print(f"Dropped {n_dropped}/{n_before} rows with NaN metrics before UMAP projection")

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

    # Calculate group x timepoint averages (exploratory trajectory visualization)
    group_keys = [group_col, timepoint_col] if group_col else [timepoint_col]
    averages = calculate_timepoint_averages(
        data[metric_cols + group_keys],
        timepoint_col=timepoint_col,
        group_col=group_col,
        metric_cols=metric_cols
    )

    # Add average UMAP coordinates per (group, timepoint)
    for i in range(n_components):
        umap_avgs = data.groupby(group_keys)[f'umap_{i+1}'].mean().reset_index()
        averages = averages.merge(umap_avgs, on=group_keys, how='left')

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

    # Save group x timepoint averages
    averages_file = output_dir / "umap_group_timepoint_averages.csv"
    averages.to_csv(averages_file, index=False)

    # Multi-metric trajectory condensation: per-subject mid-study turn angle
    # + group tests (see module docstring). Needs a group and participant col.
    umap_cols = [f'umap_{i+1}' for i in range(min(2, n_components))]
    traj = pd.DataFrame()
    if group_col and participant_col in data.columns and group_col in data.columns:
        traj = compute_subject_trajectories(
            data, participant_col=participant_col, timepoint_col=timepoint_col,
            group_col=group_col, umap_cols=umap_cols,
        )
        if not traj.empty:
            traj.to_csv(output_dir / "subject_trajectories.csv", index=False)
            tests = test_trajectory_by_group(traj, group_col=group_col)
            if not tests.empty:
                tests.to_csv(output_dir / "trajectory_turn_tests.csv", index=False)
        plot_manifold_trajectories(
            data, averages, umap_cols=umap_cols, group_col=group_col,
            timepoint_col=timepoint_col,
            output_path=output_dir / "umap_manifold_trajectories.png",
        )

    # Create visualizations
    if MATPLOTLIB_AVAILABLE:
        # Plot all points
        plot_umap_trajectory(
            data,
            umap_cols=[f'umap_{i+1}' for i in range(min(2, n_components))],
            color_col=color_col or group_col or timepoint_col,
            output_path=output_dir / "umap_trajectory_all.png",
            show_labels=False
        )

        # Plot timepoint-average trajectory with arrows, one plot per group if available
        if group_col and group_col in averages.columns:
            for group_value, group_df in averages.groupby(group_col):
                plot_timepoint_averages(
                    group_df,
                    umap_cols=[f'umap_{i+1}' for i in range(min(2, n_components))],
                    color_col=timepoint_col,
                    output_path=output_dir / f"umap_trajectory_averages_{group_value}.png"
                )
        else:
            plot_timepoint_averages(
                averages,
                umap_cols=[f'umap_{i+1}' for i in range(min(2, n_components))],
                color_col=color_col or timepoint_col,
                output_path=output_dir / "umap_trajectory_averages.png"
            )

    return data, averages


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
    parser.add_argument('--participant-col', type=str, default='participant_id',
                        help='Column name for participant IDs (trajectory analysis)')
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
    data, averages = process_umap_projection(
        input_file=args.input_file,
        output_dir=args.output_dir,
        timepoint_col=args.timepoint_col,
        group_col=args.group_col,
        color_col=args.group_col,
        participant_col=args.participant_col,
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
    if args.group_col in data.columns:
        print(f"Groups: {data[args.group_col].nunique()}")


if __name__ == "__main__":
    main()
