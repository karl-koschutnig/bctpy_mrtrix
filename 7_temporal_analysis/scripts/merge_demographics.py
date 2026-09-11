#!/usr/bin/env python3
"""
Add age / sex columns to a temporal-analysis metadata CSV by joining the
study demographics spreadsheet.

The demographics file uses encoded IDs (sub-1292001); the pipeline metadata
uses sub-001. They join on the last three digits of the numeric part. Two
demographics rows collide on that key (sub-*044, sub-*164) with identical
age/sex, so the collision is harmless - we keep the first and warn.

Usage:
    python merge_demographics.py \
        --demographics data/raw/participants_new_groups_02.04.25.xlsx \
        --metadata data/processed/metadata.csv \
        [--output data/processed/metadata.csv]   # default: overwrite --metadata
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

KEEP = ["age", "sex", "height_cm", "weight_kg"]
SRC_RENAME = {"size": "height_cm", "weight": "weight_kg"}


def short_id(raw: str) -> str | None:
    m = re.match(r"sub-(\d+)$", str(raw).strip())
    if not m:
        return None
    return "sub-" + m.group(1)[-3:]


def load_demographics(path: Path) -> pd.DataFrame:
    d = pd.read_excel(path)
    d = d.rename(columns=SRC_RENAME)
    d["participant_id"] = d["participant_id"].map(short_id)
    d = d.dropna(subset=["participant_id"])
    have = [c for c in KEEP if c in d.columns]
    d = d[["participant_id", *have]].copy()

    dup = d["participant_id"].duplicated(keep=False)
    if dup.any():
        collisions = sorted(d.loc[dup, "participant_id"].unique())
        for cid in collisions:
            rows = d[d["participant_id"] == cid]
            if rows[have].nunique().le(1).all():
                print(f"note: {cid} has {len(rows)} demographics rows with identical "
                      f"values - keeping one", file=sys.stderr)
            else:
                print(f"WARNING: {cid} has conflicting demographics rows:\n{rows}",
                      file=sys.stderr)
        d = d.drop_duplicates(subset="participant_id", keep="first")
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demographics", required=True, type=Path)
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    demo = load_demographics(args.demographics)
    meta = pd.read_csv(args.metadata)

    drop = [c for c in demo.columns if c != "participant_id" and c in meta.columns]
    if drop:
        meta = meta.drop(columns=drop)

    merged = meta.merge(demo, on="participant_id", how="left")

    subjects = merged["participant_id"].nunique()
    matched = merged.dropna(subset=["age"])["participant_id"].nunique()
    print(f"{matched}/{subjects} subjects matched to demographics")
    unmatched = sorted(set(merged.loc[merged["age"].isna(), "participant_id"]))
    if unmatched:
        print(f"unmatched: {unmatched}", file=sys.stderr)

    out = args.output or args.metadata
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"wrote {out}  ({len(merged)} rows, cols: {list(merged.columns)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
