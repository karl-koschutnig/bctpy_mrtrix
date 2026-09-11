#!/usr/bin/env python3
"""
Build the AAL3 analysis table for the factorial re-analysis.

Addresses three problems found in the exploratory sweep:

1. **Tractography yield.** `density` correlates r = .61 with the raw streamline
   total, so a group difference in density can be a difference in how many
   streamlines the tractography returned rather than in anatomy. Total
   streamline count is carried through as a covariate.

2. **Redundant measures.** The whole-brain battery has an effective
   dimensionality of ~3 (density / global efficiency / path length /
   subgraph centrality intercorrelate at |r| > .94; clustering and local
   efficiency at r = 1.00). Treating them as independent outcomes turns one
   finding into four. A PCA on the battery gives orthogonal components that
   are tested instead.

3. **Edge-set instability.** Which edges exist at all depends on the
   tractography's relative threshold. A consistency-thresholded *backbone*
   (edges present in >= 75% of all scans) fixes the edge set for every scan,
   so density is constant by construction and only edge weights can move.
   Measures recomputed on the backbone isolate weight change from yield change.

Also recodes the five arms into the orthogonal factorial the design actually
has: running vs control, group vs solo, 4 weeks vs 2 weeks, and their
interaction.

Usage:
    python aal3_prepare.py --data-dir .../bct_input/AAL3 \
        --metadata-file data/processed/metadata.csv \
        --output-dir outputs/aal3_reanalysis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **k):
        return it

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.organizational_measures import compute_organizational_measures
from scripts.small_worldness import find_connectome_files, load_connectome

# Orthogonal planned contrasts over (ctrl, solo_2w, solo_4w, group_2w, group_4w).
# Each row is a single degree of freedom the design was built to answer.
ARM_ORDER = ["ctrl", "solo_2w", "solo_4w", "group_2w", "group_4w"]
CONTRASTS = {
    "c_running":     [-1.0, 0.25, 0.25, 0.25, 0.25],   # any running vs control
    "c_group":       [0.0, -0.5, -0.5, 0.5, 0.5],      # group vs solo running
    "c_duration":    [0.0, -0.5, 0.5, -0.5, 0.5],      # 4 weeks vs 2 weeks
    "c_group_x_dur": [0.0, 0.5, -0.5, -0.5, 0.5],      # social x duration
}
BLIND_TO_ARM = {"ctrl": "ctrl", "g1_2w": "solo_2w", "g1_4w": "solo_4w",
                "g2_2w": "group_2w", "g2_4w": "group_4w"}

PCA_MEASURES = ["density", "path_length", "global_efficiency", "modularity",
                "core_periphery", "strength_mean", "local_efficiency_mean",
                "clustering_coef_mean", "betweenness_mean",
                "subgraph_centrality_log10_mean", "small_worldness"]


def load_pair(filepath: Path, n_nodes: int, keep: list[int]):
    """ROI-normalized matrix plus its raw streamline-count counterpart."""
    roi = load_connectome(filepath, n_nodes)[np.ix_(keep, keep)]
    raw_fp = Path(str(filepath).replace(".count.roi_normalized.csv", ".count.csv"))
    raw = (load_connectome(raw_fp, n_nodes)[np.ix_(keep, keep)]
           if raw_fp.exists() else None)
    np.fill_diagonal(roi, 0.0)
    if raw is not None:
        np.fill_diagonal(raw, 0.0)
    return roi, raw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--metadata-file", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-nodes", type=int, default=166)
    ap.add_argument("--exclude-nodes", default="128")
    ap.add_argument("--consistency", type=float, default=0.75,
                    help="Edge must appear in this fraction of scans to enter the backbone")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    exclude = [int(x) for x in args.exclude_nodes.split(",") if x.strip()]
    keep = [i for i in range(args.n_nodes) if i not in exclude]

    meta = pd.read_csv(args.metadata_file)
    files = find_connectome_files(args.data_dir, meta)
    print(f"{len(files)} AAL3 connectomes")

    # -------- pass 1: load, yield metrics, edge-presence for the backbone
    rois, ids, yields = [], [], []
    for pid, ses, fp in tqdm(files, desc="load"):
        roi, raw = load_pair(Path(fp), args.n_nodes, keep)
        rois.append(roi)
        ids.append({"participant_id": pid, "session": ses})
        tri = np.triu_indices(roi.shape[0], 1)
        yields.append({
            "total_streamlines": float(raw[tri].sum()) if raw is not None else np.nan,
            "n_edges_present": int((roi[tri] > 0).sum()),
        })
    rois = np.asarray(rois)

    presence = (rois > 0).mean(axis=0)
    backbone = presence >= args.consistency
    np.fill_diagonal(backbone, False)
    tri = np.triu_indices(rois.shape[1], 1)
    print(f"backbone: {int(backbone[tri].sum())} edges "
          f"({backbone[tri].mean():.1%} of possible) at >= {args.consistency:.0%} consistency")

    # -------- pass 2: measures on the full graph and on the fixed backbone
    rows = []
    for k in tqdm(range(len(rois)), desc="measures"):
        full = compute_organizational_measures(rois[k])
        back = compute_organizational_measures(rois[k] * backbone)
        rec = {**ids[k], **yields[k]}
        rec.update({f"full_{m}": v for m, v in full.items()})
        rec.update({f"bb_{m}": v for m, v in back.items()})
        w = (rois[k] * backbone)[tri]
        rec["bb_total_weight"] = float(w.sum())
        rec["bb_mean_weight"] = float(w[w > 0].mean()) if (w > 0).any() else np.nan
        rows.append(rec)

    df = pd.DataFrame(rows)

    # small-worldness comes from its own validated pipeline; merge if present
    sw = Path(args.output_dir).parent / "temporal_analysis/AAL3/small_worldness/small_worldness_results.csv"
    if sw.exists():
        s = pd.read_csv(sw)[["participant_id", "session", "small_worldness"]]
        df = df.merge(s.rename(columns={"small_worldness": "full_small_worldness"}),
                      on=["participant_id", "session"], how="left")

    df = meta.merge(df, on=["participant_id", "session"], how="left")

    # -------- factorial coding
    df["arm"] = df["group"].map(BLIND_TO_ARM)
    for name, weights in CONTRASTS.items():
        df[name] = df["arm"].map(dict(zip(ARM_ORDER, weights)))
    df["social"] = df["arm"].map({"solo_2w": "solo", "solo_4w": "solo",
                                  "group_2w": "group", "group_4w": "group"})
    df["duration"] = df["arm"].map({"solo_2w": "2w", "group_2w": "2w",
                                    "solo_4w": "4w", "group_4w": "4w"})

    # -------- PCA of the whole-brain battery (z-scored, complete cases)
    cols = [f"full_{m}" for m in PCA_MEASURES if f"full_{m}" in df.columns]
    sub = df[cols].dropna()
    Z = (sub - sub.mean()) / sub.std(ddof=0)
    evals, evecs = np.linalg.eigh(np.cov(Z.values, rowvar=False))
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    scores = Z.values @ evecs[:, :3]
    for i in range(3):
        df.loc[sub.index, f"PC{i+1}"] = scores[:, i]

    var = evals / evals.sum()
    eff_dim = evals.sum() ** 2 / (evals ** 2).sum()
    pd.DataFrame({"component": [f"PC{i+1}" for i in range(len(evals))],
                  "eigenvalue": evals, "prop_variance": var,
                  "cum_variance": np.cumsum(var)}).to_csv(out / "pca_variance.csv", index=False)
    pd.DataFrame(evecs[:, :3], index=cols,
                 columns=["PC1", "PC2", "PC3"]).to_csv(out / "pca_loadings.csv")
    print(f"PCA: PC1 {var[0]:.1%}, PC2 {var[1]:.1%}, PC3 {var[2]:.1%}; "
          f"effective dimensionality {eff_dim:.2f} of {len(cols)}")

    df.to_csv(out / "aal3_analysis_table.csv", index=False)
    np.save(out / "backbone_mask.npy", backbone)
    print(f"wrote {out/'aal3_analysis_table.csv'}  ({len(df)} rows, {df.shape[1]} cols)")
    print(df.groupby("arm")["participant_id"].nunique().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
