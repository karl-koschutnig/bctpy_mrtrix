#!/usr/bin/env python3
"""
Hub / rich-club measures for the temporal analysis
=================================================

A focused hypothesis family: where a short exercise intervention touches a
structural network, it is more likely to show in the high-degree hubs and
the rich-club edges than in whole-brain averages.

The hub set is fixed **per atlas** from the group-mean ses-1 connectome
(top `--hub-frac` of nodes by strength, after node exclusion) and written to
`hub_nodes.json`. The same nodes are used for every subject and session, so
a change over time is a change in the hubs' role, not a change in which
nodes are called hubs. The hub set is computed across all baseline subjects
regardless of arm - blind to group.

Per subject-session scalars (ROI-normalized weighted graph):
    richclub_coef        raw binary rich-club coefficient at the hub degree
    hub_strength_frac    share of total node strength carried by hub nodes
    richclub_edge_frac   share of total edge weight on hub-hub edges
    feeder_edge_frac     share on hub-nonhub edges
    local_edge_frac      share on nonhub-nonhub edges  (the three sum to 1)
    connector_hub_count  hub nodes with participation coefficient > 0.30
    hub_participation_mean   mean participation coefficient of the hub set

Node exclusion (--exclude-nodes) works exactly as in small_worldness.py.

Usage:
    python hub_measures.py --data-dir /path/to/atlas \
        --metadata-file metadata.csv --output-dir outputs/.../hub_measures \
        --n-nodes 200 [--exclude-nodes 16,124,185] [--hub-frac 0.15]
"""
from __future__ import annotations

import argparse
import json
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.small_worldness import find_connectome_files, load_connectome

HUB_KEYS = [
    "richclub_coef", "hub_strength_frac", "richclub_edge_frac",
    "feeder_edge_frac", "local_edge_frac", "connector_hub_count",
    "hub_participation_mean",
]
PARTICIPATION_CONNECTOR_THRESHOLD = 0.30


def _prep(W_raw: np.ndarray, keep_idx: list[int] | None) -> np.ndarray:
    W = np.array(W_raw, dtype=float)
    if keep_idx is not None:
        W = W[np.ix_(keep_idx, keep_idx)]
    np.fill_diagonal(W, 0.0)
    W[W < 0] = 0.0
    return W


def compute_hub_set(files, n_nodes, keep_idx, hub_frac):
    """Mean ses-1 connectome -> indices (into the kept matrix) of the top
    hub_frac nodes by strength."""
    acc, n = None, 0
    for participant_id, session, filepath in files:
        if session not in ("ses-1", "ses-01", "1", 1):
            continue
        try:
            W = _prep(load_connectome(filepath, n_nodes), keep_idx)
        except Exception as e:  # pragma: no cover
            warnings.warn(f"hub-set: skipped {filepath}: {e}")
            continue
        acc = W if acc is None else acc + W
        n += 1
    if acc is None or n == 0:
        raise RuntimeError("No ses-1 connectomes found to define the hub set")
    mean_W = acc / n
    strength = mean_W.sum(axis=0)
    k = max(1, int(round(hub_frac * len(strength))))
    hub_idx = np.argsort(strength)[::-1][:k]
    return sorted(int(i) for i in hub_idx), n


def compute_hub_measures(W: np.ndarray, hub_idx: np.ndarray) -> dict:
    out = {key: np.nan for key in HUB_KEYS}
    if not np.any(W):
        return out
    n = W.shape[0]
    hub_mask = np.zeros(n, dtype=bool)
    hub_mask[hub_idx] = True

    strength = W.sum(axis=0)
    total_strength = strength.sum()
    if total_strength > 0:
        out["hub_strength_frac"] = float(strength[hub_mask].sum() / total_strength)

    triu = np.triu(np.ones((n, n), dtype=bool), 1)
    total_w = W[triu].sum()
    if total_w > 0:
        hh = np.outer(hub_mask, hub_mask) & triu
        hl = (np.outer(hub_mask, ~hub_mask) | np.outer(~hub_mask, hub_mask)) & triu
        ll = np.outer(~hub_mask, ~hub_mask) & triu
        out["richclub_edge_frac"] = float(W[hh].sum() / total_w)
        out["feeder_edge_frac"] = float(W[hl].sum() / total_w)
        out["local_edge_frac"] = float(W[ll].sum() / total_w)

    W_bin = (W > 0).astype(float)
    try:
        deg = bct.degrees_und(W_bin)
        k_hub = int(deg[hub_idx].min())
        rc = bct.rich_club_bu(W_bin, k_hub)[0]
        val = rc[k_hub - 1] if 0 < k_hub <= len(rc) else np.nan
        out["richclub_coef"] = float(val) if np.isfinite(val) else np.nan
    except Exception as e:  # pragma: no cover
        warnings.warn(f"rich-club failed: {e}")

    try:
        ci = bct.community_louvain(W, seed=42)[0]
        part = bct.participation_coef(W, ci)
        out["hub_participation_mean"] = float(np.mean(part[hub_mask]))
        out["connector_hub_count"] = float(
            np.sum(part[hub_mask] > PARTICIPATION_CONNECTOR_THRESHOLD))
    except Exception as e:  # pragma: no cover
        warnings.warn(f"participation failed: {e}")

    return out


def process_directory(data_dir, metadata, n_nodes, output_dir=None,
                      exclude_nodes=None, hub_frac=0.15):
    exclude_nodes = sorted(set(exclude_nodes or []))
    keep_idx = [i for i in range(n_nodes) if i not in exclude_nodes] if exclude_nodes else None

    files = find_connectome_files(data_dir, metadata)
    if len(files) == 0:
        warnings.warn(f"No connectome files found in {data_dir}")
        return pd.DataFrame()

    hub_idx, n_baseline = compute_hub_set(files, n_nodes, keep_idx, hub_frac)
    hub_idx = np.asarray(hub_idx)
    print(f"Hub set: {len(hub_idx)} nodes ({hub_frac:.0%}) from {n_baseline} ses-1 connectomes")

    rows = []
    for participant_id, session, filepath in tqdm(files, desc="Hub measures"):
        try:
            W = _prep(load_connectome(filepath, n_nodes), keep_idx)
            measures = compute_hub_measures(W, hub_idx)
        except Exception as e:
            warnings.warn(f"Failed to process {filepath}: {e}")
            measures = {k: np.nan for k in HUB_KEYS}
        rows.append({"participant_id": participant_id, "session": session, **measures})

    results = metadata.merge(pd.DataFrame(rows), on=["participant_id", "session"], how="left")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_dir / "hub_measures_results.csv", index=False)
        try:
            results.to_parquet(output_dir / "hub_measures_results.parquet", index=False)
        except ImportError:
            pass
        (output_dir / "hub_nodes.json").write_text(json.dumps(
            {"hub_frac": hub_frac, "n_baseline_connectomes": n_baseline,
             "hub_node_indices_post_exclusion": [int(i) for i in hub_idx]}, indent=2))

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--metadata-file", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-nodes", type=int, required=True)
    ap.add_argument("--exclude-nodes", default=None,
                    help="Comma-separated 0-indexed node indices to drop first")
    ap.add_argument("--hub-frac", type=float, default=0.15)
    args = ap.parse_args()

    exclude_nodes = None
    if args.exclude_nodes:
        exclude_nodes = [int(x) for x in args.exclude_nodes.split(",") if x.strip()]

    metadata = pd.read_csv(args.metadata_file)
    res = process_directory(
        data_dir=args.data_dir, metadata=metadata, n_nodes=args.n_nodes,
        output_dir=args.output_dir, exclude_nodes=exclude_nodes, hub_frac=args.hub_frac,
    )
    n_ok = res[HUB_KEYS].notna().all(axis=1).sum() if len(res) else 0
    print(f"Hub measures complete: {n_ok}/{len(res)} subject-sessions fully computed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
