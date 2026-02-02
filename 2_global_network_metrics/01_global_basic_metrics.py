#!/usr/bin/env python3
"""
SCRIPT: Global Network Metrics Analysis
========================================

PURPOSE:
    Compute global (whole-brain) network metrics from connectivity matrices.
    
    Metrics computed (defaults):
    - Density
    - Path length (average shortest path between all nodes)
    - Global efficiency (inverse of path length)
    - Clustering coefficient (tendency to form local clusters)
    - Transitivity
    - Modularity (Q) + number of communities
    - Participation coefficient (mean across nodes)
    - Local efficiency (mean across nodes)
    - Betweenness (mean across nodes)
    - Small-worldness (approximate, C/L)
    - UMAP trajectories (dimensionality reduction for visualization)

INPUT REQUIREMENTS:
    - Directory with connectivity matrices (format: NxN arrays, N=number of nodes)
    - Participant metadata (CSV/TSV with columns: subject, session, group, etc.)
    - Optionally: timepoint labels for trajectory analysis

OUTPUT FILES:
    - global_metrics.parquet     (Table: subject × timepoint → metrics)
    - global_metrics.csv         (Same table in CSV)
    - umap_coordinates.parquet   (Table: subject × timepoint → UMAP coordinates)
    - plots/                     (Folder: visualizations)

CONFIGURATION:
    Edit the CONFIG section below to match your data structure.
    Key parameters:
    - data_dir: Location of connectivity matrices
    - metadata_file: Participant information file
    - output_dir: Where to save results
    - n_nodes: Number of brain regions in your atlas
    - file_pattern: How connectivity files are named in your study
    - include_metrics: Which metrics to compute (default: "all")
    - exclude_metrics: List of metrics to skip

USAGE:
    python 01_global_basic_metrics.py
    
    Or for background execution:
    nohup python 01_global_basic_metrics.py > analysis.log 2>&1 &

AUTHOR: Analysis Pipeline
DATE: January 2026
VERSION: 2.0 (Generic, reusable version)
"""

import os
import numpy as np
import pandas as pd
import bct
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Disable GPU if using for reproducibility
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for remote server
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# UMAP will be imported lazily if needed (can be slow due to Numba JIT compilation)
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except (ImportError, RuntimeError):
    UMAP_AVAILABLE = False

# ======================== CONFIGURATION ========================
DATA_DIR = Path("/data/local/129_PK01/derivatives/dsistudio_connectomics/connectivity")
BCT_DIR = Path("/data/local/129_PK01/derivatives/bct")
OUTPUT_DIR = BCT_DIR / "comprehensive_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

ATLAS = "Brainnectome"
N_NODES = 246

print("="*70)
print("COMPREHENSIVE ANALYSIS: METRICS + UMAP + TRAJECTORIES")
print("="*70)
print(f"Output directory: {OUTPUT_DIR}")
print()

# ======================== LOAD DATA ========================
print("Loading participant metadata...")
participants_file = BCT_DIR / "participants_5groups.tsv"
participants_df = pd.read_csv(participants_file, sep='\t')
print(f"Loaded {len(participants_df)} participant records")

# ============================================================================
# CONFIGURATION - EDIT THIS SECTION FOR YOUR DATA
# ============================================================================

# Default configuration (can be overridden by injected CONFIG from pipeline)
DEFAULT_CONFIG = {
    # ---- DATA LOCATIONS ----
    "data_dir": Path("/data/local/129_PK01/derivatives/dsistudio_connectomics/connectivity"),
    "metadata_file": Path("/data/local/129_PK01/derivatives/bct/participants_5groups.tsv"),
    "output_dir": Path("/data/local/129_PK01/derivatives/bct/global_metrics"),
    
    # ---- DATA STRUCTURE ----
    "n_nodes": None,  # Auto-detected if not provided
    "atlas_name": "Brainnectome",  # Name of your atlas
    
    # ---- FILE NAMING PATTERN ----
    # How to find connectivity files. Example patterns:
    # "{subject}_ses-{session}*/tracks_*/by_atlas/{atlas}/*.connectivity.mat"
    # "connectivity/{subject}/ses-{session}/connectogram.npy"
    "file_pattern": "{subject}_ses-{session}*/tracks_*/by_atlas/{atlas}/*.connectivity.mat",
    
    # ---- PARTICIPANTS METADATA ----
    # Columns expected in metadata file:
    "subject_col": "participant_id",
    "session_col": "session_id",
    "group_col": "group",
    "sex_col": "sex",
    
    # ---- PROCESSING OPTIONS ----
    "binarize": False,  # Convert to binary (0/1) connectivity?
    "weight_type": "weighted",  # 'weighted' or 'binary'
    "use_umap": False,  # Use UMAP for dimensionality reduction (slower, requires numba)
    "umap_n_neighbors": 15,
    "umap_min_dist": 0.1,
    "umap_metric": "euclidean",
    "include_metrics": "all",  # or list of metric keys to compute
    "exclude_metrics": [],  # list of metric keys to skip
}

# Merge with injected CONFIG (if provided by pipeline_execution.py)
if 'CONFIG' in globals():
    # If CONFIG was injected, update it with defaults for missing keys
    for key, value in DEFAULT_CONFIG.items():
        if key not in CONFIG:
            CONFIG[key] = value
else:
    # Otherwise use default config
    CONFIG = DEFAULT_CONFIG.copy()

# Ensure paths are Path objects
CONFIG["data_dir"] = Path(CONFIG["data_dir"])
CONFIG["metadata_file"] = Path(CONFIG["metadata_file"])
CONFIG["output_dir"] = Path(CONFIG["output_dir"])

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def auto_detect_n_nodes() -> int:
    """
    Auto-detect n_nodes by loading the first available connectivity matrix.
    
    RETURNS:
        int: Number of nodes detected from the first available file
    
    RAISES:
        ValueError: If no connectivity files can be found or loaded
    """
    print("Auto-detecting n_nodes from connectivity matrices...")
    metadata = pd.read_csv(CONFIG["metadata_file"], sep='\t')
    
    for idx, row in metadata.iterrows():
        subject = row[CONFIG["subject_col"]]
        session = row[CONFIG["session_col"]]
        
        # Ensure subject has 'sub-' prefix
        if not subject.startswith('sub-'):
            subject = f'sub-{subject}'
        
        # Handle session - check if it already has 'ses-' prefix
        if isinstance(session, str) and session.startswith('ses-'):
            session_num = session.replace('ses-', '')
        else:
            session_num = str(session)
        
        # Try the pattern with the session number (without 'ses-' prefix, will be added by pattern)
        pattern = CONFIG["file_pattern"].format(
            subject=subject,
            session=session_num,
            atlas=CONFIG["atlas_name"]
        )
        
        matches = list(CONFIG["data_dir"].glob(pattern))
        if matches:
            try:
                filepath = matches[0]
                if filepath.suffix == '.mat':
                    mat_data = loadmat(filepath)
                    if 'connectivity' in mat_data:
                        A = mat_data['connectivity']
                    else:
                        keys = [k for k in mat_data.keys() if not k.startswith('__')]
                        A = mat_data[keys[0]] if keys else None
                elif filepath.suffix == '.npy':
                    A = np.load(filepath)
                else:
                    continue
                
                if A is not None:
                    A = np.array(A, dtype=float)
                    n_nodes = A.shape[0]
                    print(f"✓ Detected n_nodes = {n_nodes} from {filepath.name}")
                    return n_nodes
            except Exception as e:
                continue
    
    raise ValueError(f"Could not auto-detect n_nodes: no valid connectivity matrices found in {CONFIG['data_dir']} with pattern {CONFIG['file_pattern']}")


def load_connectivity_matrix(subject: str, session: str) -> np.ndarray:
    """
    Load connectivity matrix for a subject and session.
    
    PARAMETERS:
        subject (str): Subject identifier (e.g., "sub-119BPAF161001" or "119BPAF161001")
        session (str): Session identifier (e.g., "ses-1" or "1")
    
    RETURNS:
        np.ndarray: Connectivity matrix (N×N), or None if not found
    
    NOTES:
        - Handles both .mat and .npy formats
        - Validates matrix shape matches N_NODES
        - Returns None if file not found or invalid
    """
    # Ensure subject has 'sub-' prefix
    if not subject.startswith('sub-'):
        subject = f'sub-{subject}'
    
    # Handle session - if it has 'ses-' prefix, extract just the number
    if isinstance(session, str) and session.startswith('ses-'):
        session_num = session.replace('ses-', '')
    else:
        session_num = str(session)
    
    # Format the search pattern
    pattern = CONFIG["file_pattern"].format(
        subject=subject,
        session=session_num,
        atlas=CONFIG["atlas_name"]
    )
    
    matches = list(CONFIG["data_dir"].glob(pattern))
    if not matches:
        return None
    
    try:
        filepath = matches[0]
        
        # Load based on file extension
        if filepath.suffix == '.mat':
            mat_data = loadmat(filepath)
            if 'connectivity' in mat_data:
                A = mat_data['connectivity']
            else:
                keys = [k for k in mat_data.keys() if not k.startswith('__')]
                A = mat_data[keys[0]] if keys else None
        elif filepath.suffix == '.npy':
            A = np.load(filepath)
        else:
            return None
        
        if A is None:
            return None
        
        A = np.array(A, dtype=float)
        
        # Validate shape
        if A.shape[0] != CONFIG["n_nodes"] or A.shape[1] != CONFIG["n_nodes"]:
            print(f"  ⚠ Invalid shape for {subject} ses-{session}: {A.shape}")
            return None
        
        return A
    
    except Exception as e:
        print(f"  ⚠ Error loading {subject} ses-{session}: {e}")
        return None


def compute_global_metrics(A: np.ndarray) -> dict:
    """
    Compute global network metrics from connectivity matrix.
    
    PARAMETERS:
        A (np.ndarray): Connectivity matrix (N×N)
    
    RETURNS:
        dict: Dictionary with global metrics. Defaults include:
            - density
            - path_length
            - global_efficiency
            - clustering_coef
            - transitivity
            - modularity
            - n_communities
            - participation_coef_mean
            - local_efficiency_mean
            - betweenness_mean
            - small_worldness (approximate)
    
    NOTES:
        - Binarizes if configured
        - Handles disconnected networks
        - Returns NaN for invalid networks
    """
    try:
        # Binarize if needed
        if CONFIG["binarize"]:
            A_proc = (A > 0).astype(float)
        else:
            A_proc = A.copy()
        
        # Check if connected
        if not np.any(A_proc):
            return {
                'path_length': np.nan,
                'global_efficiency': np.nan,
                'clustering_coef': np.nan,
                'small_worldness': np.nan,
            }
        
        # Compute metrics using BCT
        A_bin = (A_proc > 0).astype(float)
        density_result = bct.density_und(A_bin)
        # Handle cases where function returns tuple
        density = float(density_result[0]) if isinstance(density_result, tuple) else float(density_result)

        L = bct.distance_wei(1 / (A_proc + 1e-10))[0]  # Path length matrix
        path_length = np.mean(L[L > 0])  # Average, excluding zeros
        global_efficiency = 1 / path_length if path_length > 0 else np.nan

        C = bct.clustering_coef_wu(A_proc)
        clustering_coef = np.mean(C)
        transitivity = bct.transitivity_wu(A_proc)

        # Modularity + community structure
        Ci, Q = bct.community_louvain(A_bin)
        n_communities = len(np.unique(Ci[~np.isnan(Ci)]))

        # Participation coefficient (mean across nodes)
        participation = bct.participation_coef(A_proc, Ci)
        participation_mean = np.nanmean(participation)

        # Local efficiency (mean across nodes)
        local_eff = bct.efficiency_wei(A_proc, local=True)
        local_eff_mean = np.nanmean(local_eff)

        # Betweenness (mean across nodes)
        betweenness = bct.betweenness_wei(A_proc)
        betweenness_mean = np.nanmean(betweenness)

        # Small-worldness (approximate, C/L)
        small_worldness = clustering_coef / path_length if path_length > 0 else np.nan

        metrics = {
            'density': float(density),
            'path_length': float(path_length),
            'global_efficiency': float(global_efficiency),
            'clustering_coef': float(clustering_coef),
            'transitivity': float(transitivity),
            'modularity': float(Q),
            'n_communities': int(n_communities),
            'participation_coef_mean': float(participation_mean),
            'local_efficiency_mean': float(local_eff_mean),
            'betweenness_mean': float(betweenness_mean),
            'small_worldness': float(small_worldness),
        }

        # Apply include/exclude filters
        include = CONFIG.get("include_metrics", "all")
        exclude = set(CONFIG.get("exclude_metrics", []))

        if include != "all":
            include = set(include)
            metrics = {k: v for k, v in metrics.items() if k in include}

        if exclude:
            metrics = {k: v for k, v in metrics.items() if k not in exclude}

        return metrics
    
    except Exception as e:
        print(f"  ⚠ Error computing metrics: {e}")
        return {
            'density': np.nan,
            'path_length': np.nan,
            'global_efficiency': np.nan,
            'clustering_coef': np.nan,
            'transitivity': np.nan,
            'modularity': np.nan,
            'n_communities': np.nan,
            'participation_coef_mean': np.nan,
            'local_efficiency_mean': np.nan,
            'betweenness_mean': np.nan,
            'small_worldness': np.nan,
        }


# ============================================================================
# INITIALIZE
# ============================================================================

# Create output directory
CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

# Auto-detect n_nodes if not provided
if "n_nodes" not in CONFIG or CONFIG["n_nodes"] is None:
    CONFIG["n_nodes"] = auto_detect_n_nodes()
else:
    print(f"Using provided n_nodes: {CONFIG['n_nodes']}")

print("="*75)
print("GLOBAL NETWORK METRICS ANALYSIS")
print("="*75)
print(f"Data directory:      {CONFIG['data_dir']}")
print(f"Metadata file:       {CONFIG['metadata_file']}")
print(f"Output directory:    {CONFIG['output_dir']}")
print(f"Atlas:               {CONFIG['atlas_name']} ({CONFIG['n_nodes']} nodes)")
print("="*75)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading participant metadata...")
if not CONFIG["metadata_file"].exists():
    raise FileNotFoundError(f"Metadata file not found: {CONFIG['metadata_file']}")

metadata = pd.read_csv(CONFIG["metadata_file"], sep='\t')
print(f"✓ Loaded {len(metadata)} records")
print(f"  Columns: {list(metadata.columns)}")
print()

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("Computing global metrics for all subjects...")
results = []

for idx, row in metadata.iterrows():
    subject = row[CONFIG["subject_col"]]
    session = row[CONFIG["session_col"]]
    
    # Load connectivity matrix
    A = load_connectivity_matrix(str(subject), str(session))
    if A is None:
        print(f"  ⚠ Skipping {subject} ses-{session} (file not found)")
        continue
    
    # Compute metrics
    metrics = compute_global_metrics(A)
    
    # Store with metadata
    record = {
        'subject': subject,
        'session': session,
        **{col: row[col] for col in [CONFIG["group_col"], CONFIG["sex_col"]] if col in row},
        **metrics,
    }
    results.append(record)
    
    if (idx + 1) % 10 == 0:
        print(f"  ✓ Processed {idx + 1} records")

print(f"✓ Computed metrics for {len(results)} subjects")
print()

# ============================================================================
# SAVE RESULTS
# ============================================================================

results_df = pd.DataFrame(results)

# Save main metrics table
metrics_file = CONFIG["output_dir"] / "global_metrics.parquet"
results_df.to_parquet(metrics_file, index=False)
print(f"✓ Saved metrics to: {metrics_file}")

# Save CSV version
csv_file = CONFIG["output_dir"] / "global_metrics.csv"
results_df.to_csv(csv_file, index=False)
print(f"✓ Saved CSV to: {csv_file}")

# Compute summary statistics
print("\nSummary Statistics:")
print("="*75)
for group in results_df[CONFIG["group_col"]].unique():
    group_data = results_df[results_df[CONFIG["group_col"]] == group]
    print(f"\n{group} (N={len(group_data)}):")
    print(f"  Global Efficiency:  {group_data['global_efficiency'].mean():.3f} ± {group_data['global_efficiency'].std():.3f}")
    print(f"  Clustering Coef:    {group_data['clustering_coef'].mean():.3f} ± {group_data['clustering_coef'].std():.3f}")
    print(f"  Path Length:        {group_data['path_length'].mean():.3f} ± {group_data['path_length'].std():.3f}")

print("\n" + "="*75)

# ============================================================================
# DIMENSIONALITY REDUCTION (PCA or optional UMAP)
# ============================================================================

print("\nPerforming dimensionality reduction...")

# Prepare data for dimensionality reduction
metadata_cols = {'subject', 'session', CONFIG["group_col"], CONFIG["sex_col"]}
metric_cols = [c for c in results_df.columns if c not in metadata_cols]
X = results_df[metric_cols].values
X = np.nan_to_num(X, nan=0)  # Handle NaN values
X_scaled = StandardScaler().fit_transform(X)

# Use UMAP if requested and available, otherwise use PCA
if CONFIG.get("use_umap", False) and UMAP_AVAILABLE:
    print("Using UMAP for dimensionality reduction...")
    umap_model = UMAP(
        n_neighbors=CONFIG["umap_n_neighbors"],
        min_dist=CONFIG["umap_min_dist"],
        metric=CONFIG["umap_metric"],
        random_state=42
    )
    coords = umap_model.fit_transform(X_scaled)
    coord_names = ['umap_1', 'umap_2']
else:
    print("Using PCA for dimensionality reduction...")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    coord_names = ['pca_1', 'pca_2']

# Add to results
results_df[coord_names[0]] = coords[:, 0]
results_df[coord_names[1]] = coords[:, 1]

# Save coordinates
coord_file = CONFIG["output_dir"] / "dimensionality_reduction_coordinates.parquet"
results_df[['subject', 'session', CONFIG["group_col"]] + coord_names].to_parquet(coord_file, index=False)
print(f"✓ Saved dimensionality reduction coordinates to: {coord_file}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\nCreating visualizations...")

plot_dir = CONFIG["output_dir"] / "plots"
plot_dir.mkdir(exist_ok=True)

# Plot 1: Global metrics by group
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
metric_cols = ['path_length', 'global_efficiency', 'clustering_coef', 'small_worldness']

for ax, metric in zip(axes.flat, metric_cols):
    sns.boxplot(data=results_df, x=CONFIG["group_col"], y=metric, ax=ax)
    ax.set_title(f"{metric.replace('_', ' ').title()}")
    ax.set_ylabel("Value")
    ax.set_xlabel("Group")

plt.tight_layout()
plt.savefig(plot_dir / "metrics_by_group.png", dpi=150)
print(f"✓ Saved: metrics_by_group.png")

# Plot 2: Dimensionality reduction trajectory (PCA or UMAP)
fig, ax = plt.subplots(figsize=(10, 8))
groups = results_df[CONFIG["group_col"]].unique()
colors = plt.cm.Set1(np.linspace(0, 1, len(groups)))

for group, color in zip(groups, colors):
    group_data = results_df[results_df[CONFIG["group_col"]] == group]
    ax.scatter(group_data[coord_names[0]], group_data[coord_names[1]], label=group, color=color, s=100, alpha=0.6)

ax.set_xlabel(coord_names[0].replace('_', ' ').upper())
ax.set_ylabel(coord_names[1].replace('_', ' ').upper())
ax.set_title(f"{coord_names[0].split('_')[0].upper()}: Global Metrics Trajectories")
ax.legend()
plt.tight_layout()
plt.savefig(plot_dir / "trajectories.png", dpi=150)
print(f"✓ Saved: trajectories.png")

plt.close('all')

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*75)
print("ANALYSIS COMPLETE")
print("="*75)
print(f"Subjects processed:  {len(results)}")
print(f"Output directory:    {CONFIG['output_dir']}")
print(f"Main output files:")
print(f"  - global_metrics.parquet")
print(f"  - umap_coordinates.parquet")
print(f"  - plots/metrics_by_group.png")
print(f"  - plots/umap_trajectories.png")
print("="*75)
print()

# ======================== HELPER FUNCTIONS ========================

def load_connectivity_matrix(subject, session, atlas=ATLAS):
    """Load connectivity matrix from DSI Studio .mat file"""
    if not subject.startswith('sub-'):
        subject = f'sub-{subject}'
    
    pattern = f"{subject}_ses-{session}*/tracks_1000k_streamline/by_atlas/{atlas}/*.connectivity.mat"
    matches = list(DATA_DIR.glob(pattern))
    
    if not matches:
        return None
    
    try:
        mat_data = loadmat(matches[0])
        if 'connectivity' in mat_data:
            A = mat_data['connectivity']
        else:
            keys = [k for k in mat_data.keys() if not k.startswith('__')]
            A = mat_data[keys[0]] if keys else None
        
        if A is None:
            return None
            
        A = np.array(A, dtype=float)
        
        if A.shape[0] != N_NODES or A.shape[1] != N_NODES:
            return None
        return A
    except Exception as e:
        return None


def compute_small_worldness(A):
    """
    Compute small-worldness coefficient
    σ = (C/C_random) / (L/L_random)
    where C = clustering coef, L = characteristic path length
    """
    try:
        A_bin = (A > 0).astype(int)
        
        # Real network metrics
        clustering = bct.clustering_coef_bu(A_bin)
        C_real = np.mean(clustering)
        
        # Path length
        D = bct.distance_bin(A_bin)
        if np.all(np.isfinite(D)):
            C_real_path, charpath_real = bct.charpath(D)
        else:
            return np.nan
        
        # Null model (random network with same degree distribution)
        # Use 10 iterations
        C_random_list = []
        L_random_list = []
        
        for _ in range(10):
            A_rand = bct.null_model_und_degree(A_bin)
            c_rand = np.mean(bct.clustering_coef_bu(A_rand))
            D_rand = bct.distance_bin(A_rand)
            
            if np.all(np.isfinite(D_rand)):
                _, l_rand = bct.charpath(D_rand)
                C_random_list.append(c_rand)
                L_random_list.append(l_rand)
        
        if len(C_random_list) == 0:
            return np.nan
            
        C_random = np.mean(C_random_list)
        L_random = np.mean(L_random_list)
        
        # Small-worldness
        if C_random > 0 and L_random > 0:
            sigma = (C_real / C_random) / (charpath_real / L_random)
            return sigma
        else:
            return np.nan
            
    except Exception as e:
        return np.nan


def compute_global_efficiency(A):
    """Global efficiency (BCT function wrapper)"""
    try:
        eff = bct.efficiency_wei(A)
        return eff
    except:
        return np.nan


def compute_characteristic_path_length(A):
    """Characteristic path length from weighted network"""
    try:
        A_bin = (A > 0).astype(int)
        D = bct.distance_bin(A_bin)
        
        if not np.all(np.isfinite(D)):
            return np.nan
            
        C, L = bct.charpath(D)
        return L
    except:
        return np.nan


# ======================== MAIN ANALYSIS LOOP ========================
print("Computing connectivity metrics for all subjects/sessions...")
print()

all_data = []
unique_subjects = participants_df['participant_id'].unique()

for idx, subject in enumerate(unique_subjects):
    if idx % 20 == 0:
        print(f"Progress: {idx}/{len(unique_subjects)} subjects...")
    
    subj_meta = participants_df[participants_df['participant_id'] == subject].iloc[0]
    
    for session in [1, 2, 3]:
        A = load_connectivity_matrix(subject, session)
        
        if A is None:
            continue
        
        # ===== EXISTING METRICS (aggregate) =====
        A_bin = (A > 0).astype(int)
        
        metrics = {
            'subject': subject,
            'session': session,
            'atlas': ATLAS,
            'group': subj_meta.get('group', 'unknown'),
            'sex': subj_meta.get('sex', 'unknown'),
            'age': subj_meta.get('age', np.nan),
        }
        
        # Degree and strength
        try:
            degree_vec = bct.degrees_und(A_bin)
            metrics['degree_mean'] = np.mean(degree_vec)
            metrics['degree_std'] = np.std(degree_vec)
            metrics['degree_max'] = np.max(degree_vec)
            
            strength_vec = bct.strengths_und(A)
            metrics['strength_mean'] = np.mean(strength_vec)
            metrics['strength_std'] = np.std(strength_vec)
            metrics['strength_max'] = np.max(strength_vec)
        except:
            pass
        
        # Clustering
        try:
            clustering = bct.clustering_coef_wu(A)
            metrics['clustering_mean'] = np.mean(clustering)
            metrics['clustering_std'] = np.std(clustering)
        except:
            pass
        
        # Modularity
        try:
            Ci, Q = bct.community_louvain(A_bin)
            metrics['modularity'] = Q
            metrics['n_communities'] = len(np.unique(Ci[~np.isnan(Ci)]))
        except:
            metrics['modularity'] = np.nan
            metrics['n_communities'] = np.nan
        
        # Participation coefficient
        try:
            pc = bct.participation_coef(A, Ci)
            metrics['participation_coef_mean'] = np.mean(pc)
            metrics['participation_coef_std'] = np.std(pc)
        except:
            pass
        
        # Density
        try:
            metrics['density'] = bct.density_und(A_bin)
        except:
            pass
        
        # ===== NEW METRICS (3 graph theory measures) =====
        
        # 1. Characteristic Path Length
        charpath = compute_characteristic_path_length(A)
        metrics['characteristic_path_length'] = charpath
        
        # 2. Global Efficiency
        global_eff = compute_global_efficiency(A)
        metrics['global_efficiency'] = global_eff
        
        # 3. Small-Worldness (requires null model - computationally expensive)
        sigma = compute_small_worldness(A)
        metrics['small_worldness'] = sigma
        
        # Local efficiency
        try:
            local_eff = bct.efficiency_wei(A, local=True)
            metrics['local_efficiency_mean'] = np.mean(local_eff)
        except:
            pass
        
        # Betweenness centrality
        try:
            betweenness = bct.betweenness_wei(A)
            metrics['betweenness_mean'] = np.mean(betweenness)
        except:
            pass
        
        all_data.append(metrics)

print(f"Completed: {len(unique_subjects)} subjects")
print(f"Total records: {len(all_data)}")
print()

# ======================== CREATE ANALYSIS DATAFRAME ========================
print("Creating analysis dataframe...")
analysis_df = pd.DataFrame(all_data)

# Save full metrics
metrics_output = OUTPUT_DIR / "complete_metrics_with_graph_theory.parquet"
analysis_df.to_parquet(metrics_output, index=False)
print(f"✅ Metrics saved: {metrics_output}")
print()

# ======================== UMAP ANALYSIS ========================
print("Running dimensionality reduction...")

# Select features for dimensionality reduction (standardize column names)
feature_cols = [
    'degree_mean', 'degree_std', 'strength_mean', 'strength_std',
    'clustering_mean', 'modularity', 'participation_coef_mean',
    'density', 'characteristic_path_length', 'global_efficiency', 
    'small_worldness', 'local_efficiency_mean', 'betweenness_mean'
]

# Filter to available columns
available_cols = [c for c in feature_cols if c in analysis_df.columns]
print(f"Using {len(available_cols)} features for dimensionality reduction:")
for col in available_cols:
    print(f"  - {col}")

# Prepare data - replace NaN with column median or 0
umap_data = analysis_df[available_cols].copy()
print(f"Samples before NaN handling: {len(umap_data)}")

# Ensure all columns are numeric
for col in umap_data.columns:
    if umap_data[col].dtype == 'object':
        try:
            umap_data[col] = pd.to_numeric(umap_data[col], errors='coerce')
        except:
            print(f"Warning: Could not convert {col} to numeric, dropping column")
            umap_data = umap_data.drop(columns=[col])
            continue

# Replace NaN with column median (or 0 if all NaN)
for col in umap_data.columns:
    if umap_data[col].isnull().any():
        median_val = umap_data[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        umap_data[col].fillna(median_val, inplace=True)

# Remove infinite values and replace with median
for col in umap_data.columns:
    col_data = umap_data[col].values
    if np.any(np.isinf(col_data)):
        finite_mask = np.isfinite(col_data)
        if np.any(finite_mask):
            median_val = np.median(col_data[finite_mask])
        else:
            median_val = 0.0
        umap_data[col] = np.where(np.isinf(col_data), median_val, col_data)

umap_df_full = analysis_df.copy()
umap_data_values = umap_data.values

print(f"Valid samples for dimensionality reduction: {len(umap_data_values)}")
print()

# Standardize
scaler = StandardScaler()
features_scaled = scaler.fit_transform(umap_data_values)

# Use PCA or UMAP based on config
if CONFIG.get("use_umap", False) and UMAP_AVAILABLE:
    print("Using UMAP (n_neighbors=15, min_dist=0.1)...")
    try:
        reducer = UMAP(
            n_neighbors=min(15, len(umap_data_values) - 1),
            min_dist=0.1,
            n_components=2,
            metric='euclidean',
            random_state=42,
            verbose=0
        )
        embedding = reducer.fit_transform(features_scaled)
        coord_names = ['umap_1', 'umap_2']
    except Exception as e:
        print(f"UMAP failed: {e}. Falling back to PCA...")
        pca = PCA(n_components=2, random_state=42)
        embedding = pca.fit_transform(features_scaled)
        coord_names = ['pca_1', 'pca_2']
else:
    print("Using PCA for dimensionality reduction...")
    pca = PCA(n_components=2, random_state=42)
    embedding = pca.fit_transform(features_scaled)
    coord_names = ['pca_1', 'pca_2']
print("✅ Dimensionality reduction complete")
print()

# Create dimensionality reduction dataframe
dr_df = umap_df_full.copy()
dr_df[coord_names[0]] = embedding[:, 0]
dr_df[coord_names[1]] = embedding[:, 1]

# Save dimensionality reduction results
dr_output = OUTPUT_DIR / "dimensionality_reduction_embedding.parquet"
dr_df.to_parquet(dr_output)
print(f"✅ Saved dimensionality reduction embedding: {dr_output}")
print()

# ======================== TRAJECTORY ANALYSIS ========================
print("Analyzing trajectories in dimensionality reduction space...")

trajectory_metrics = []

for subject in dr_df['subject'].unique():
    subj_data = dr_df[dr_df['subject'] == subject].sort_values('session')
    
    if len(subj_data) < 3:
        continue
    
    # Get dimensionality reduction positions for each session
    sessions = []
    positions = []
    for _, row in subj_data.iterrows():
        sessions.append(row['session'])
        positions.append([row[coord_names[0]], row[coord_names[1]]])
    
    if len(sessions) != 3:
        continue
    
    s1 = np.array(positions[0])
    s2 = np.array(positions[1])
    s3 = np.array(positions[2])
    
    # Calculate distances
    dist_1_2 = np.linalg.norm(s2 - s1)
    dist_2_3 = np.linalg.norm(s3 - s2)
    total_distance = dist_1_2 + dist_2_3
    
    # Acceleration metric
    if dist_1_2 > 0.01:
        acceleration = dist_2_3 / dist_1_2
    else:
        acceleration = np.nan
    
    # Trajectory direction (vector from session 1 to 3)
    trajectory_vec = s3 - s1
    trajectory_angle = np.linalg.norm(trajectory_vec)
    
    trajectory_metrics.append({
        'subject': subject,
        'group': subj_data.iloc[0]['group'],
        'sex': subj_data.iloc[0]['sex'],
        'age': subj_data.iloc[0]['age'],
        'early_change': dist_1_2,      # Session 1→2
        'late_change': dist_2_3,       # Session 2→3
        'total_distance': total_distance,
        'acceleration': acceleration,   # late/early ratio
        'final_position_distance': trajectory_angle
    })

trajectory_df = pd.DataFrame(trajectory_metrics)

# Classify response types
trajectory_df['response_type'] = pd.cut(
    trajectory_df['acceleration'], 
    bins=[0, 0.6, 1.4, np.inf], 
    labels=['Decelerating', 'Linear', 'Accelerating'],
    include_lowest=True
)

trajectory_df['response_magnitude'] = pd.cut(
    trajectory_df['total_distance'],
    bins=[0, trajectory_df['total_distance'].quantile(0.33), 
          trajectory_df['total_distance'].quantile(0.67), np.inf],
    labels=['Low', 'Medium', 'High'],
    include_lowest=True
)

# Save trajectories
trajectory_output = OUTPUT_DIR / "trajectory_analysis.parquet"
trajectory_df.to_parquet(trajectory_output, index=False)
print(f"✅ Trajectory analysis saved: {trajectory_output}")
print()

# ======================== SUMMARY STATISTICS ========================
print("="*70)
print("TRAJECTORY SUMMARY")
print("="*70)
print("\nResponse Type Distribution:")
print(trajectory_df['response_type'].value_counts())

print("\nResponse Type by Intervention Group:")
print(pd.crosstab(trajectory_df['group'], trajectory_df['response_type'], margins=True))

print("\nResponse Magnitude by Intervention Group:")
print(pd.crosstab(trajectory_df['group'], trajectory_df['response_magnitude'], margins=True))

print("\nMean Trajectory Metrics by Group:")
group_stats = trajectory_df.groupby('group')[
    ['early_change', 'late_change', 'total_distance', 'acceleration']
].mean()
print(group_stats)

print("\nMean Trajectory Metrics by Sex:")
sex_stats = trajectory_df.groupby('sex')[
    ['early_change', 'late_change', 'total_distance', 'acceleration']
].mean()
print(sex_stats)

print()

# ======================== VISUALIZATIONS ========================
print("Creating visualizations...")

colors_map = {
    'alone_2w': '#FF6B6B', 'alone_4w': '#FFA07A',
    'group_2w': '#4169E1', 'group_4w': '#87CEEB',
    'control': '#808080',
    1: '#FF6B6B', 2: '#FFA07A',
    3: '#4169E1', 4: '#87CEEB',
    5: '#808080'
}

response_colors = {
    'Decelerating': '#FF6B6B',
    'Linear': '#FFD700',
    'Accelerating': '#4169E1'
}

# 2D projections
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel A: By group
for group in dr_df['group'].unique():
    mask = dr_df['group'] == group
    axes[0].scatter(dr_df[mask][coord_names[0]], dr_df[mask][coord_names[1]],
                   label=str(group), s=60, alpha=0.6,
                   color=colors_map.get(group, 'gray'))

axes[0].set_xlabel(coord_names[0].replace('_', ' ').upper(), fontsize=11)
axes[0].set_ylabel(coord_names[1].replace('_', ' ').upper(), fontsize=11)
axes[0].set_title(f'{coord_names[0].split("_")[0].upper()} projection - By Group', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Panel B: By response type
for rtype in trajectory_df['response_type'].unique():
    subjs_with_type = trajectory_df[trajectory_df['response_type'] == rtype]['subject'].values
    mask = dr_df['subject'].isin(subjs_with_type)
    axes[1].scatter(dr_df[mask][coord_names[0]], dr_df[mask][coord_names[1]],
                   label=rtype, s=60, alpha=0.6,
                   color=response_colors.get(rtype, 'gray'))

axes[1].set_xlabel(coord_names[0].replace('_', ' ').upper(), fontsize=11)
axes[1].set_ylabel(coord_names[1].replace('_', ' ').upper(), fontsize=11)
axes[1].set_title(f'{coord_names[0].split("_")[0].upper()} projection - By Response Type', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'umap_2d_projections.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ UMAP 2D projections saved")

# 4. Trajectory scatter plot
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Early change vs late change
for group in trajectory_df['group'].unique():
    mask = trajectory_df['group'] == group
    axes[0, 0].scatter(trajectory_df[mask]['early_change'],
                       trajectory_df[mask]['late_change'],
                       label=str(group), s=80, alpha=0.6,
                       color=colors_map.get(group, 'gray'))

axes[0, 0].set_xlabel('Early Change (Session 1→2)', fontsize=11)
axes[0, 0].set_ylabel('Late Change (Session 2→3)', fontsize=11)
axes[0, 0].set_title('Trajectory Pattern: Early vs Late Response', fontsize=12, fontweight='bold')
axes[0, 0].axline((0, 0), slope=1, color='red', linestyle='--', alpha=0.5, label='Equal')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(alpha=0.3)

# Acceleration by group
trajectory_df.boxplot(column='acceleration', by='group', ax=axes[0, 1])
axes[0, 1].set_xlabel('Group', fontsize=11)
axes[0, 1].set_ylabel('Acceleration (Late/Early)', fontsize=11)
axes[0, 1].set_title('Acceleration by Group', fontsize=12, fontweight='bold')
plt.sca(axes[0, 1])
plt.xticks(rotation=45)

# Total distance by group
trajectory_df.boxplot(column='total_distance', by='group', ax=axes[1, 0])
axes[1, 0].set_xlabel('Group', fontsize=11)
axes[1, 0].set_ylabel('Total Distance (UMAP units)', fontsize=11)
axes[1, 0].set_title('Total Response Magnitude by Group', fontsize=12, fontweight='bold')
plt.sca(axes[1, 0])
plt.xticks(rotation=45)

# Response type distribution
response_counts = trajectory_df.groupby(['group', 'response_type']).size().unstack(fill_value=0)
response_counts.plot(kind='bar', ax=axes[1, 1], color=['#FF6B6B', '#4169E1', '#FFD700'])
axes[1, 1].set_xlabel('Group', fontsize=11)
axes[1, 1].set_ylabel('Count', fontsize=11)
axes[1, 1].set_title('Response Type Distribution by Group', fontsize=12, fontweight='bold')
axes[1, 1].legend(title='Response Type', fontsize=10)
plt.sca(axes[1, 1])
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'trajectory_analysis_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Trajectory analysis plots saved")

print()
print("="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print(f"\nOutput files saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
print(f"  - complete_metrics_with_graph_theory.parquet")
print(f"  - umap_embedding.parquet")
print(f"  - trajectory_analysis.parquet")
print(f"  - umap_3d_by_group.png")
print(f"  - umap_3d_by_response_type.png")
print(f"  - umap_2d_projections.png")
print(f"  - trajectory_analysis_plots.png")
print()
print("="*70)
