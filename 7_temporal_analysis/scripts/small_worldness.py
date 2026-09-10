#!/usr/bin/env python3
"""
Small-Worldness Calculation (sigma)
=====================================

Calculate small-worldness coefficient for connectome matrices.

sigma = (C / C_random) / (L / L_random)

where:
- C = clustering coefficient
- L = characteristic path length
- C_random = clustering coefficient of random graph
- L_random = characteristic path length of random graph

For small-world networks: sigma > 1
For random networks: sigma ~ 1
For lattice networks: sigma >> 1

Usage:
    python small_worldness.py --data-dir /path/to/connectomes --metadata-file metadata.csv --output-dir outputs/

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

# Optional tqdm for progress bar
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ============================================================================
# BCT IMPORTS (fallback to internal implementations if bct not available)
# ============================================================================

BCT_AVAILABLE = False
try:
    import bct
    BCT_AVAILABLE = True
except ImportError:
    pass


# ============================================================================
# INTERNAL BCT IMPLEMENTATIONS (fallback)
# ============================================================================

def clustering_coef_wu_fallback(W):
    """Fallback implementation of weighted clustering coefficient."""
    n = W.shape[0]
    C = np.zeros(n)
    
    for i in range(n):
        neighbors = np.where(W[i, :] > 0)[0]
        if len(neighbors) < 2:
            C[i] = 0
            continue
        
        # Get subgraph of neighbors
        subgraph = W[neighbors[:, None], neighbors]
        # Number of triangles
        k = len(neighbors)
        num_triangles = np.trace(subgraph @ subgraph @ subgraph) / 6
        # Number of possible triangles
        possible_triangles = k * (k - 1) / 2
        
        if possible_triangles > 0:
            C[i] = num_triangles / possible_triangles
    
    return C


def charpath_fallback(D):
    """Fallback implementation of characteristic path length."""
    n = D.shape[0]
    n_nodes = n * (n - 1) / 2
    L = np.sum(D[D < np.inf]) / n_nodes
    return L


def distance_bin(W):
    """Convert weighted matrix to binary distance matrix."""
    B = W > 0
    B = B.astype(float)
    np.fill_diagonal(B, 0)
    
    # Distance matrix: shortest path lengths
    n = B.shape[0]
    D = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                D[i, j] = 0
            elif B[i, j] > 0:
                D[i, j] = 1
            else:
                D[i, j] = np.inf
    
    # Floyd-Warshall for shortest paths
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if D[i, k] + D[k, j] < D[i, j]:
                    D[i, j] = D[i, k] + D[k, j]
    
    return D


# ============================================================================
# MAIN SMALL-WORLDNESS FUNCTION
# ============================================================================

def calculate_clustering_coefficient(W, n_null=100):
    """
    Calculate clustering coefficient for a weighted connectivity matrix.
    
    Args:
        W: n x n weighted connectivity matrix (symmetric, zero diagonal)
        n_null: number of random null models to average
    
    Returns:
        C: mean clustering coefficient
    """
    if BCT_AVAILABLE:
        C = bct.clustering_coef_wu(W)
    else:
        C = clustering_coef_wu_fallback(W)
    
    return np.mean(C)


def calculate_path_length(W, n_null=100):
    """
    Calculate characteristic path length for a weighted connectivity matrix.
    
    Args:
        W: n x n weighted connectivity matrix (symmetric, zero diagonal)
        n_null: number of random null models to average
    
    Returns:
        L: characteristic path length
    """
    # Convert to binary for distance calculation
    if BCT_AVAILABLE:
        D = bct.distance_bin(W)
        L, _, _, _, _ = bct.charpath(D)
    else:
        D = distance_bin(W)
        L = charpath_fallback(D)
    
    return L


def make_random_graph(n_nodes, density=None):
    """
    Generate a random Erdős–Rényi graph with similar properties.
    
    Args:
        n_nodes: number of nodes
        density: target connection density (if None, use 10%)
    
    Returns:
        W_random: random connectivity matrix
    """
    if density is None:
        density = 0.1
    
    W = np.random.rand(n_nodes, n_nodes) < density
    W = W.astype(float)
    W = (W + W.T) / 2  # Make symmetric
    np.fill_diagonal(W, 0)
    
    return W


def calculate_small_worldness(W, n_null=100, random_state=42):
    """
    Calculate small-worldness coefficient sigma for a connectivity matrix.
    
    sigma = (C / C_random) / (L / L_random)
    
    Args:
        W: n x n weighted connectivity matrix (symmetric, zero diagonal)
        n_null: number of random null models to average
        random_state: random seed for reproducibility
    
    Returns:
        sigma: small-worldness coefficient
    
    Raises:
        ValueError: if matrix is not square, not symmetric, or diagonal not zero
    """
    n_nodes = W.shape[0]
    
    # Validate matrix
    n_nodes = W.shape[0]
    expected_nodes = 200  # Schaefer200
    
    if W.shape[0] != W.shape[1]:
        raise ValueError(f"Matrix must be square, got {W.shape}")
    
    if n_nodes != expected_nodes:
        raise ValueError(f"Matrix must be {expected_nodes}x{expected_nodes}, got {W.shape}")
    
    if not np.allclose(W, W.T):
        raise ValueError("Matrix must be symmetric")
    
    if not np.allclose(np.diag(W), 0):
        warnings.warn("Matrix diagonal is not zero - setting to zero")
        W = W.copy()
        np.fill_diagonal(W, 0)
    
    # Calculate actual network metrics
    C = calculate_clustering_coefficient(W)
    L = calculate_path_length(W)
    
    # Handle disconnected graphs
    if np.isinf(L):
        return np.nan
    
    # Calculate random network metrics (average over null models)
    np.random.seed(random_state)
    C_randoms = []
    L_randoms = []
    
    for _ in range(n_null):
        W_random = make_random_graph(n_nodes, density=0.1)
        C_random = calculate_clustering_coefficient(W_random)
        L_random = calculate_path_length(W_random)
        
        if not np.isinf(L_random):
            C_randoms.append(C_random)
            L_randoms.append(L_random)
    
    # If we couldn't generate valid random graphs
    if len(C_randoms) == 0 or len(L_randoms) == 0:
        return np.nan
    
    C_random = np.mean(C_randoms)
    L_random = np.mean(L_randoms)
    
    # Handle division by zero
    if C_random == 0 or L_random == 0:
        return np.nan
    
    # Small-worldness
    sigma = (C / C_random) / (L / L_random)
    
    return sigma


# ============================================================================
# DIRECTORY PROCESSING
# ============================================================================

def load_connectome(filepath, n_nodes=200):
    """
    Load a connectivity matrix from CSV file.
    
    Args:
        filepath: path to CSV file
        n_nodes: expected number of nodes
    
    Returns:
        W: n_nodes x n_nodes connectivity matrix
    """
    df = pd.read_csv(filepath, header=None)
    W = df.values
    
    if W.shape[0] != n_nodes or W.shape[1] != n_nodes:
        raise ValueError(f"Expected {n_nodes}x{n_nodes} matrix, got {W.shape}")
    
    return W.astype(float)


def find_connectome_files(data_dir, metadata, extension='.csv'):
    """
    Find connectome files matching metadata entries.
    
    Args:
        data_dir: directory containing connectome files
        metadata: DataFrame with participant_id and session columns
        extension: file extension to look for
    
    Returns:
        list of (participant_id, session, filepath) tuples
    """
    data_dir = Path(data_dir)
    files = []
    
    for _, row in metadata.iterrows():
        participant_id = row['participant_id']
        session = row['session']
        
        # Try different naming patterns
        patterns = [
            f"{participant_id}_{session}{extension}",
            f"{participant_id}_{session}.roi_normalized{extension}",
            f"{participant_id}_{session}.count{extension}",
            f"{participant_id}_{session}.count.roi_normalized{extension}",
        ]
        
        for pattern in patterns:
            filepath = data_dir / pattern
            if filepath.exists():
                files.append((participant_id, session, str(filepath)))
                break
    
    return files


def process_connectome_directory(data_dir, metadata, n_nodes=200, output_dir=None, 
                                  n_null=100, random_state=42):
    """
    Process all connectomes in a directory and calculate small-worldness.
    
    Args:
        data_dir: directory containing connectome CSV files
        metadata: DataFrame with participant_id, session, and other columns
        n_nodes: expected number of nodes
        output_dir: directory to save results (optional)
        n_null: number of null models for small-worldness
        random_state: random seed
    
    Returns:
        results: DataFrame with small-worldness values
    """
    # Find connectome files
    files = find_connectome_files(data_dir, metadata)
    
    if len(files) == 0:
        warnings.warn(f"No connectome files found in {data_dir}")
        return pd.DataFrame()
    
    results = []
    
    for participant_id, session, filepath in tqdm(files, desc="Processing connectomes"):
        try:
            W = load_connectome(filepath, n_nodes)
            sigma = calculate_small_worldness(W, n_null=n_null, random_state=random_state)
            
            results.append({
                'participant_id': participant_id,
                'session': session,
                'small_worldness': sigma
            })
        except Exception as e:
            warnings.warn(f"Failed to process {filepath}: {e}")
            results.append({
                'participant_id': participant_id,
                'session': session,
                'small_worldness': np.nan
            })
    
    results_df = pd.DataFrame(results)
    
    # Merge with metadata to preserve all rows
    results_df = metadata.merge(results_df, on=['participant_id', 'session'], how='left')
    
    # Save to output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "small_worldness_results.csv"
        results_df.to_csv(output_file, index=False)
        
        # Also save as parquet (if available)
        try:
            output_file_parquet = output_dir / "small_worldness_results.parquet"
            results_df.to_parquet(output_file_parquet, index=False)
        except ImportError:
            # pyarrow/fastparquet not available, skip parquet
            pass
    
    return results_df


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface for small-worldness calculation."""
    parser = argparse.ArgumentParser(
        description='Calculate small-worldness for connectome matrices'
    )
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing connectome CSV files')
    parser.add_argument('--metadata-file', type=str, required=True,
                        help='CSV file with metadata (participant_id, session, group, etc.)')
    parser.add_argument('--output-dir', type=str, default='outputs/small_worldness',
                        help='Directory to save results')
    parser.add_argument('--n-nodes', type=int, default=200,
                        help='Number of nodes in the atlas')
    parser.add_argument('--n-null', type=int, default=100,
                        help='Number of null models for small-worldness')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration JSON file')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        
        # Override args with config values
        if 'temporal' in config and 'n_nodes' in config['temporal']:
            args.n_nodes = config['temporal']['n_nodes']
        if 'small_worldness' in config:
            if 'n_null_models' in config['small_worldness']:
                args.n_null = config['small_worldness']['n_null_models']
            if 'random_state' in config['small_worldness']:
                args.random_state = config['small_worldness']['random_state']
    
    # Load metadata
    metadata = pd.read_csv(args.metadata_file)
    
    # Process connectomes
    results = process_connectome_directory(
        data_dir=args.data_dir,
        metadata=metadata,
        n_nodes=args.n_nodes,
        output_dir=args.output_dir,
        n_null=args.n_null,
        random_state=args.random_state
    )
    
    print(f"Processed {len(results)} connectomes")
    print(f"Results saved to {args.output_dir}")
    print(f"\nSmall-worldness summary:")
    print(results['small_worldness'].describe())


if __name__ == "__main__":
    main()
