#!/usr/bin/env python3
"""
Benjamini-Hochberg FDR correction across the temporal-analysis output.

Walks outputs/temporal_analysis[_3group]/<atlas>/{mixed_model_results,sem_results,
difference_scores}/ and adds a `q_value` column to the p-value tables, correcting
*within a family*. A family here = (method, design, term-type) pooled across the
5 atlases and 13 measures - that is the set of tests you'd scan together looking
for "which measure in which atlas moved", so that's the set the correction has
to cover.

Writes <name>_fdr.csv next to each source table and a combined
outputs/temporal_analysis_fdr_summary.csv.

Usage:
    python fdr_correct.py [--outputs-root outputs] [--alpha 0.05]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DESIGNS = {"temporal_analysis": "5-arm", "temporal_analysis_3group": "3-group"}
ATLASES = ["AAL3", "Gordon333", "HCP-MMP", "Schaefer200", "Schaefer400"]

# (subdir, filename, p-column, term-column, keep-only-these-terms or None)
SOURCES = [
    ("mixed_model_results", "mixed_model_statistics.csv", "Pr(>F)", "term",
     {"group", "session", "group:session"}),
    ("mixed_model_results_adj", "mixed_model_statistics.csv", "Pr(>F)", "term",
     {"group", "session", "group:session"}),
    ("mixed_model_results_baseline", "mixed_model_statistics.csv", "Pr(>F)", "term",
     {"group", "session", "group:session"}),
    ("sem_results", "sem_growth_parameters.csv", "pvalue", None, None),          # group->i/s paths
    ("sem_results_adj", "sem_growth_parameters.csv", "pvalue", None, None),
    ("sem_results_latentbasis", "sem_growth_parameters.csv", "pvalue", None, None),
    ("sem_results", "sem_slope_lrt.csv", "p_slope_improves_fit", None, None),
    ("difference_scores", "difference_score_coefficients.csv", "Pr...t..", "term", None),
]


def bh(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return q
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    ranked = p[order] * m / (np.arange(1, m + 1))
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.clip(ranked, 0, 1)
    return q


def term_type(row, term_col):
    """Family label for one test row, or None if the row is not a hypothesis
    test we want corrected (latent means/variances, the intercept path, ...)."""
    if term_col is None:
        if "p_slope_improves_fit" in row.index:
            return "sem_slope_lrt"
        lhs, op, rhs = row.get("lhs"), row.get("op"), str(row.get("rhs"))
        if op != "~":                       # skip ~1 (means) and ~~ (variances)
            return None
        if lhs not in ("i", "s"):
            return None
        if rhs.startswith("grp_"):
            return f"sem_{lhs}_group"        # i/s regressed on a group dummy
        if rhs in ("age", "sex"):
            return f"sem_{lhs}_covariate"
        return None
    t = str(row[term_col])
    if t == "(Intercept)":
        return None
    if t.startswith("group") and t not in ("group", "group:session"):
        return "diff_group_contrast"
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    collected = []
    for design_dir, design in DESIGNS.items():
        for subdir, fname, pcol, tcol, keep in SOURCES:
            frames = []
            for atlas in ATLASES:
                f = args.outputs_root / design_dir / atlas / subdir / fname
                if not f.exists():
                    continue
                d = pd.read_csv(f)
                if pcol not in d.columns:
                    continue
                d["_atlas"] = atlas
                d["_src"] = str(f)
                frames.append(d)
            if not frames:
                continue
            alld = pd.concat(frames, ignore_index=True)
            if keep is not None and tcol is not None:
                alld = alld[alld[tcol].isin(keep)].copy()
            if alld.empty:
                continue
            alld["_tt"] = alld.apply(lambda r: term_type(r, tcol), axis=1)
            alld = alld[alld["_tt"].notna()].copy()
            if alld.empty:
                continue
            alld["_family"] = f"{design}|{subdir}|" + alld["_tt"]
            alld["q_value"] = np.nan
            for fam, grp in alld.groupby("_family"):
                alld.loc[grp.index, "q_value"] = bh(grp[pcol].to_numpy())

            for src, grp in alld.groupby("_src"):
                out = Path(src).with_name(Path(src).stem + "_fdr.csv")
                grp.drop(columns=["_src"]).to_csv(out, index=False)

            hit = alld[alld["q_value"] < args.alpha]
            for _, r in hit.iterrows():
                collected.append({
                    "design": design, "analysis": subdir, "table": fname,
                    "atlas": r["_atlas"], "family": r["_family"],
                    "metric": r.get("metric"),
                    "term": r[tcol] if tcol else f'{r.get("lhs")}~{r.get("rhs")}',
                    "p_value": r[pcol], "q_value": r["q_value"],
                })

    summary = pd.DataFrame(collected).sort_values(["design", "analysis", "q_value"])
    out = args.outputs_root / "temporal_analysis_fdr_summary.csv"
    summary.to_csv(out, index=False)
    print(f"{len(summary)} results survive FDR at q < {args.alpha}")
    if len(summary):
        print(summary.to_string(index=False))
    print(f"\nwrote {out} and per-table *_fdr.csv files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
