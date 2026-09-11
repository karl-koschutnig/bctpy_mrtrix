#!/usr/bin/env python3
"""
Organizational (whole-brain graph-theory) measures
==================================================

Computes the battery of global network-organization measures from the source
pipeline's D_calculate_organization_measures.m, adapted for this study:

    density, characteristic path length, global efficiency, modularity,
    core/periphery-ness, k-core size (k=6), s-core size (s=0.6),
    mean strength, mean local efficiency, mean clustering coefficient,
    mean betweenness centrality, mean subgraph centrality

Small-worldness is NOT recomputed here - it has its own validated pipeline
(small_worldness.py, with degree-preserving null models). If a
small_worldness_results.csv exists alongside the output, it is merged in so
the statistics stage sees the full measure set from one file.

Weighting and matrix conventions:
  * ROI-normalized edge weights (region-size corrected), used directly for
    every weighted measure. Rescaling to [0, 1] was tried and rejected -
    dividing by a single (often outlier) max edge squashes clustering and
    local efficiency to implausible near-zero values; the ROI-normalized
    scale gives literature-typical magnitudes and matches what
    small_worldness.py was validated against.
  * A binarized copy for subgraph centrality, k-core, and the (hop-count)
    path-length / global-efficiency calculation - the source computes those
    on binary topology via distance_bin, not weighted distances.
  * Betweenness centrality is computed on a connection-LENGTH matrix
    (inverted weights, zeros preserved), per BCT's documented convention -
    a deliberate deviation from the source's betweenness_wei(W) call, which
    passes a weight matrix where a length matrix is expected.
  * Modularity uses community_louvain (seeded, weighted) - bct.modularity_und
    (the source's Newman method) is broken in this bctpy/numpy combination.
  * k-core (k=6) and s-core (s=0.6) keep the source's fixed thresholds. Both
    were calibrated to a different weight scale / network density and tend to
    saturate on this dataset's denser connectomes - retained for
    completeness, likely uninformative here.

Node exclusion (--exclude-nodes) works exactly as in small_worldness.py.

Usage:
    python organizational_measures.py --data-dir /path/to/atlas \\
        --metadata-file metadata.csv --output-dir outputs/ \\
        --n-nodes 200 [--exclude-nodes 16,124,185]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

import bct

# Reuse the file-discovery / loading logic (ROI-normalized preference,
# dimension validation) from the small-worldness script. Works both under
# pytest (7_temporal_analysis on pythonpath) and as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.small_worldness import find_connectome_files, load_connectome


# ============================================================================
# MATRIX PREPARATION
# ============================================================================

def weights_to_lengths(W: np.ndarray) -> np.ndarray:
    """Connection-length matrix: 1/w on existing edges, 0 elsewhere.

    Zeros in the input mean 'no edge' and must stay 0 (not become a huge
    finite length) so shortest-path routines treat them as absent.
    """
    L = np.zeros_like(W, dtype=float)
    mask = W > 0
    L[mask] = 1.0 / W[mask]
    return L


# ============================================================================
# MEASURES
# ============================================================================

def compute_organizational_measures(W_roi: np.ndarray) -> dict:
    """Compute all organizational measures for one connectome.

    Args:
        W_roi: ROI-normalized weighted connectivity matrix (symmetric, zero
            diagonal), already node-excluded.

    Returns:
        dict of measure name -> scalar (np.nan on failure / disconnection).
    """
    keys = [
        "density", "path_length", "global_efficiency", "modularity",
        "core_periphery", "kcore_size", "score_size", "strength_mean",
        "local_efficiency_mean", "clustering_coef_mean", "betweenness_mean",
        "subgraph_centrality_log10_mean",
    ]
    out = {k: np.nan for k in keys}

    W_norm = W_roi.copy()                  # ROI-normalized weights, used directly
    np.fill_diagonal(W_norm, 0)
    if not np.any(W_norm):
        return out

    W_bin = (W_norm > 0).astype(float)     # binary topology

    # --- density (weight-agnostic) ---
    out["density"] = bct.density_und(W_bin)[0]

    # --- path length + global efficiency, on hop-count topology ---
    try:
        D = bct.distance_bin(W_bin)
        lambda_, efficiency, *_ = bct.charpath(D, include_diagonal=False,
                                               include_infinite=True)
        if np.isfinite(lambda_):
            out["path_length"] = float(lambda_)
            out["global_efficiency"] = float(efficiency)
        else:
            warnings.warn("Disconnected graph (infinite characteristic path "
                          "length) - path length / global efficiency set to NaN")
    except Exception as e:  # pragma: no cover - defensive
        warnings.warn(f"path length / efficiency failed: {e}")

    # --- weighted whole-brain measures ---
    try:
        # community_louvain (seeded) - bct.modularity_und is broken here.
        out["modularity"] = float(bct.community_louvain(W_norm, seed=42)[1])
    except Exception as e:  # pragma: no cover
        warnings.warn(f"modularity failed: {e}")

    try:
        _, coreness = bct.core_periphery_dir(W_norm)
        out["core_periphery"] = float(coreness)
    except Exception as e:  # pragma: no cover
        warnings.warn(f"core/periphery failed: {e}")

    try:
        out["kcore_size"] = float(bct.kcore_bu(W_bin, 6)[1])
    except Exception as e:  # pragma: no cover
        warnings.warn(f"k-core failed: {e}")

    try:
        out["score_size"] = float(bct.score_wu(W_norm, 0.6)[1])
    except Exception as e:  # pragma: no cover
        warnings.warn(f"s-core failed: {e}")

    # --- nodal measures, averaged ---
    try:
        out["strength_mean"] = float(np.mean(bct.strengths_und(W_norm)))
    except Exception as e:  # pragma: no cover
        warnings.warn(f"strength failed: {e}")

    try:
        out["local_efficiency_mean"] = float(
            np.nanmean(bct.efficiency_wei(W_norm, local=True)))
    except Exception as e:  # pragma: no cover
        warnings.warn(f"local efficiency failed: {e}")

    try:
        out["clustering_coef_mean"] = float(
            np.mean(bct.clustering_coef_wu(W_norm)))
    except Exception as e:  # pragma: no cover
        warnings.warn(f"clustering coefficient failed: {e}")

    try:
        L = weights_to_lengths(W_norm)
        out["betweenness_mean"] = float(np.mean(bct.betweenness_wei(L)))
    except Exception as e:  # pragma: no cover
        warnings.warn(f"betweenness failed: {e}")

    try:
        # Subgraph centrality grows exponentially with the largest eigenvalue
        # (values here span 1e10-1e20) - store log10 of the mean so the
        # statistics stage isn't dominated by raw scale.
        sc = float(np.mean(bct.subgraph_centrality(W_bin)))
        out["subgraph_centrality_log10_mean"] = np.log10(sc) if sc > 0 else np.nan
    except Exception as e:  # pragma: no cover
        warnings.warn(f"subgraph centrality failed: {e}")

    return out


# ============================================================================
# DIRECTORY PROCESSING
# ============================================================================

def process_directory(data_dir, metadata, n_nodes, output_dir=None,
                      exclude_nodes=None):
    exclude_nodes = sorted(set(exclude_nodes or []))
    keep_idx = [i for i in range(n_nodes) if i not in exclude_nodes]

    files = find_connectome_files(data_dir, metadata)
    if len(files) == 0:
        warnings.warn(f"No connectome files found in {data_dir}")
        return pd.DataFrame()

    rows = []
    n_disconnected = 0
    for participant_id, session, filepath in tqdm(files, desc="Organizational measures"):
        try:
            W = load_connectome(filepath, n_nodes)
            if exclude_nodes:
                W = W[np.ix_(keep_idx, keep_idx)]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                measures = compute_organizational_measures(W)
            if any("Disconnected graph" in str(w.message) for w in caught):
                n_disconnected += 1
            for w in caught:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        except Exception as e:
            warnings.warn(f"Failed to process {filepath}: {e}")
            measures = {k: np.nan for k in compute_organizational_measures(
                np.ones((3, 3)) - np.eye(3)).keys()}
        rows.append({"participant_id": participant_id, "session": session, **measures})

    if rows:
        print(f"Organizational measures: {n_disconnected}/{len(rows)} connectomes "
              f"had a disconnected graph (path length / efficiency NaN).")

    results = pd.DataFrame(rows)
    results = metadata.merge(results, on=["participant_id", "session"], how="left")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Merge in small-worldness if it was computed alongside.
        sw_path = output_dir.parent / "small_worldness" / "small_worldness_results.csv"
        if sw_path.exists():
            sw = pd.read_csv(sw_path)[["participant_id", "session", "small_worldness"]]
            results = results.merge(sw, on=["participant_id", "session"], how="left")

        results.to_csv(output_dir / "organizational_measures_results.csv", index=False)
        try:
            results.to_parquet(output_dir / "organizational_measures_results.parquet", index=False)
        except ImportError:
            pass

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--metadata-file", required=True)
    parser.add_argument("--output-dir", default="outputs/organizational_measures")
    parser.add_argument("--n-nodes", type=int, required=True)
    parser.add_argument("--exclude-nodes", type=str, default=None,
                        help="Comma-separated 0-indexed node indices to drop "
                             "(same semantics as small_worldness.py)")
    args = parser.parse_args()

    exclude_nodes = None
    if args.exclude_nodes:
        exclude_nodes = [int(x) for x in args.exclude_nodes.split(",") if x.strip()]

    metadata = pd.read_csv(args.metadata_file)
    results = process_directory(
        data_dir=args.data_dir, metadata=metadata, n_nodes=args.n_nodes,
        output_dir=args.output_dir, exclude_nodes=exclude_nodes,
    )

    print(f"\nProcessed {len(results)} subject-sessions -> {args.output_dir}")
    measure_cols = [c for c in results.columns
                    if c not in {"participant_id", "session", "group", "age", "sex"}]
    if len(results):
        print(results[measure_cols].describe().T[["mean", "std", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
