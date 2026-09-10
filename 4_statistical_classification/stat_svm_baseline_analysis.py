#!/usr/bin/env python3
"""
SCRIPT: SVM Analysis with Different Study Design Variants
=========================================================

PURPOSE:
    Tests various grouping strategies using SVM classification:
    1. Social effect: alone vs group vs control
    2. Duration effect: 2w vs 4w vs control
    3. Intervention vs Control: binary classification

    Input is either global_metrics.parquet or node_level_metrics.parquet
    from previous pipeline steps. Groups are auto-detected from labels.

USAGE:
    python svm_analysis.py CLI flags
    python svm_analysis.py --metrics-file /path/to/metrics.parquet
                           --metadata /path/to/participants.tsv
                           --output-dir /path/to/output

AUTHOR: Analysis Pipeline
VERSION: 2.0 (CLI-driven, auto-detection)
"""

from __future__ import annotations
from datetime import datetime

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# Local
sys.path.append(str(Path(__file__).resolve().parent.parent / "1_utilities"))
from group_detection import detect_or_ask_groups

# ============================================================================
# CLI
# ============================================================================


def _progress(current, total, desc):
    """Simple progress display without external dependencies."""
    pct = current / total * 100 if total > 0 else 0
    bar_len = 30
    filled = int(bar_len * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"  {desc}: |{bar}| {current}/{total} ({pct:.0f}%)", end="\r", flush=True)
    if current == total:
        print()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SVM baseline analysis — all inputs via CLI flags.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--metrics-file",  required=True, help="Path to metrics .parquet file")
    parser.add_argument("--output-dir",    required=True, help="Directory where results are saved")
    parser.add_argument("--metadata",      default=None,  help="Participant metadata file (CSV or TSV)")
    parser.add_argument("--control-group", default=None,  help="Control group label (auto-detected if omitted)")
    parser.add_argument("--alone-groups",  default=None,  nargs="+", help="Alone group labels")
    parser.add_argument("--group-groups",  default=None,  nargs="+", help="Social group labels")
    parser.add_argument("--short-groups",  default=None,  nargs="+", help="Short duration group labels")
    parser.add_argument("--long-groups",   default=None,  nargs="+", help="Long duration group labels")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    """Validate paths and return config dict."""
    metrics_file = Path(args.metrics_file).expanduser().resolve()
    output_dir   = Path(args.output_dir).expanduser().resolve()
    if not metrics_file.exists():
        sys.exit(f"x Metrics file not found: {metrics_file}")
    return {
        "metrics_file":  metrics_file,
        "output_dir":    output_dir,
        "metadata_file": Path(args.metadata).expanduser().resolve() if args.metadata else None,
        "control_group": args.control_group,
        "alone_groups":  args.alone_groups,
        "group_groups":  args.group_groups,
        "short_groups":  args.short_groups,
        "long_groups":   args.long_groups,
    }


# ============================================================================
# AUTO-DETECTION
# ============================================================================

def detect_columns(df: pd.DataFrame) -> dict:
    """Auto-detect subject, session, group, sex, age columns."""
    cols_lower = {c.lower(): c for c in df.columns}

    def find(candidates):
        for c in candidates:
            if c in cols_lower:
                return cols_lower[c]
        for c in candidates:
            for cl, co in cols_lower.items():
                if c in cl:
                    return co
        return None

    exclude = {"subject", "participant_id", "session", "ses", "session_id",
               "node", "group", "condition", "arm", "intervention",
               "sex", "gender", "age", "atlas", "hub_type", "community",
               "nr_sessions", "dsi_metric", "matrix_type",
               "qc_passed", "qc_warnings"}

    detected = {
        "subject_col": find(["participant_id", "subject_id", "subject", "participant"]),
        "session_col": find(["session_id", "session", "ses", "timepoint", "visit"]),
        "group_col":   find(["group", "condition", "arm", "intervention"]),
        "sex_col":     find(["sex", "gender"]),
        "age_col":     find(["age"]),
        "metric_cols": [c for c in df.columns
                        if c.lower() not in exclude
                        and pd.api.types.is_numeric_dtype(df[c])
                        and c.lower() not in ("node",)],
    }

    print("  Auto-detected columns:")
    for k, v in detected.items():
        if k != "metric_cols":
            print(f"    {k}: {v if v else '⚠ not found'}")
        else:
            print(f"    metric_cols: {len(v)} columns")
    return detected


def detect_groups(df: pd.DataFrame, group_col: str, config: dict) -> dict:
    """Auto-detect control, alone, social, short, long groups from labels."""
    all_groups   = df[group_col].dropna().unique().tolist()
    group_counts = df.groupby(group_col)[df.columns[0]].count()

    # Control
    control = config.get("control_group") or group_counts.idxmin()

    intervention = [g for g in all_groups if g != control]

    def match(groups, keywords):
        return [g for g in groups if any(k in str(g).lower() for k in keywords)]

    alone  = config.get("alone_groups")  or match(intervention, ["alone", "individual", "solo"])
    social = config.get("group_groups")  or match(intervention, ["group", "social", "team"])
    short  = config.get("short_groups")  or match(intervention, ["2w", "short", "2week", "week2"])
    long_  = config.get("long_groups")   or match(intervention, ["4w", "long",  "4week", "week4"])

    # Fallback split
    if not alone and not social and len(intervention) >= 2:
        half   = len(intervention) // 2
        alone  = intervention[:half]
        social = intervention[half:]
        print("  ! Alone/group labels not detected — splitting intervention equally")

    if not short and not long_ and len(intervention) >= 2:
        half  = len(intervention) // 2
        short = intervention[:half]
        long_ = intervention[half:]
        print("  ! Short/long labels not detected — splitting intervention equally")

    groups = {
        "all":          all_groups,
        "control":      control,
        "intervention": intervention,
        "alone":        alone  or [],
        "social":       social or [],
        "short":        short  or [],
        "long":         long_  or [],
    }

    print(f"  Control group:       {control}")
    print(f"  Intervention groups: {intervention}")
    print(f"  Alone groups:        {groups['alone']}")
    print(f"  Social groups:       {groups['social']}")
    print(f"  Short groups:        {groups['short']}")
    print(f"  Long groups:         {groups['long']}")

    # Validate — warn if key groups are empty
    warnings_list = []
    if not groups["alone"] and not groups["social"]:
        warnings_list.append(
            "Could not detect 'alone' or 'social' groups for social effect analysis.\n"
            "  Expected labels containing: alone/individual/solo and group/social/team\n"
            f"  Found labels: {all_groups}\n"
            "  Fix: add 'alone_groups' and 'group_groups' to CLI flags"
        )
    if not groups["short"] and not groups["long"]:
        warnings_list.append(
            "Could not detect 'short' or 'long' duration groups for duration analysis.\n"
            "  Expected labels containing: 2w/short/2week and 4w/long/4week\n"
            f"  Found labels: {all_groups}\n"
            "  Fix: add 'short_groups' and 'long_groups' to CLI flags"
        )
    for w in warnings_list:
        print(f"\n  ⚠ WARNING: {w}")

    return groups


def detect_atlas(df: pd.DataFrame) -> str | None:
    """Auto-detect atlas column and return most common atlas."""
    if "atlas" in df.columns:
        atlas = df["atlas"].value_counts().idxmax()
        print(f"  + Atlas detected: {atlas}")
        return atlas
    return None


# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

def load_metrics(metrics_file: Path, metadata_file: Path | None,
                 cols: dict) -> pd.DataFrame:
    """Load metrics parquet and optionally merge with metadata."""
    df = pd.read_parquet(metrics_file)
    print(f"  + Loaded metrics: {len(df)} rows, {len(df.columns)} columns")

    if metadata_file and metadata_file.exists():
        sep  = "\t" if metadata_file.suffix == ".tsv" else ","
        meta = pd.read_csv(metadata_file, sep=sep)
        print(f"  + Loaded metadata: {len(meta)} rows")

        subj_col = cols["subject_col"]
        ses_col  = cols["session_col"]

        # Try to merge on subject + session
        merge_keys = [k for k in [subj_col, ses_col] if k and k in meta.columns and k in df.columns]
        if merge_keys:
            df = df.merge(meta, on=merge_keys, how="left", suffixes=("", "_meta"))
            print(f"  + Merged on: {merge_keys}")
        else:
            print("  ⚠ Could not merge metadata — no common columns found")

    return df


def aggregate_nodal(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """If nodal data (has 'node' column), aggregate to subject-session level."""
    if "node" not in df.columns:
        return df

    print("  + Nodal data detected — aggregating to subject-session level")
    group_keys = [c for c in [cols["subject_col"], cols["session_col"],
                               cols["group_col"], cols["sex_col"], cols["age_col"]]
                  if c and c in df.columns]
    metric_cols = cols["metric_cols"]
    available   = [c for c in metric_cols if c in df.columns]
    agg_df      = df.groupby(group_keys)[available].mean().reset_index()
    print(f"  + Aggregated to {len(agg_df)} subject-session rows")
    return agg_df


# ============================================================================
# SLOPE CALCULATION
# ============================================================================

def calculate_slopes(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """Calculate temporal slopes per subject per metric."""
    subject_col = cols["subject_col"]
    session_col = cols["session_col"]
    group_col   = cols["group_col"]
    sex_col     = cols["sex_col"]
    age_col     = cols["age_col"]
    metric_cols = [c for c in cols["metric_cols"] if c in df.columns]

    records = []
    _subjects_list = list(df.groupby(subject_col))
    _total_s = len(_subjects_list)
    for _i_s, (subject, subj_data) in enumerate(_subjects_list):
        _g = subj_data[group_col].iloc[0] if group_col and group_col in subj_data.columns else ""
        _progress(_i_s + 1, _total_s, "Calculating slopes")
        print(f"    {subject} [{_g}] fitting slopes across {len(subj_data)} sessions...  ", end="\n", flush=True)
        subj_data = subj_data.sort_values(session_col)
        if len(subj_data) < 2:
            continue

        # Extract session numbers
        sessions = subj_data[session_col].values
        try:
            sessions = pd.to_numeric(
                pd.Series(sessions).str.extract(r"(\d+)")[0].values
            )
        except Exception:
            sessions = np.arange(len(sessions), dtype=float)

        rec = {subject_col: subject}
        if group_col and group_col in subj_data.columns:
            rec[group_col] = subj_data[group_col].iloc[0]
        if sex_col and sex_col in subj_data.columns:
            rec[sex_col] = subj_data[sex_col].iloc[0]
        if age_col and age_col in subj_data.columns:
            rec[age_col] = subj_data[age_col].iloc[0]

        for metric in metric_cols:
            values = pd.to_numeric(subj_data[metric], errors="coerce").values
            valid  = ~np.isnan(values)
            if valid.sum() >= 2:
                try:
                    slope = np.polyfit(sessions[valid].astype(float), values[valid], 1)[0]
                    rec[f"{metric}_slope"] = slope
                except Exception:
                    pass

        records.append(rec)

    slopes_df = pd.DataFrame(records)
    print(f"  + Slopes calculated for {len(slopes_df)} subjects")
    return slopes_df


# ============================================================================
# GROUP REMAPPING
# ============================================================================

def remap_groups(group_values: pd.Series, group_a: list, group_b: list,
                 label_a: str, label_b: str, control: str) -> pd.Series:
    """Generic group remapping."""
    def _map(g):
        if g in group_a:   return label_a
        if g in group_b:   return label_b
        if g == control:   return "control"
        return np.nan
    return group_values.apply(_map)


def encode_labels(labels: np.ndarray, mapping: dict) -> np.ndarray:
    return np.array([mapping.get(l, np.nan) for l in labels], dtype=float)


# ============================================================================
# FEATURE PREPARATION
# ============================================================================

def prepare_features(slopes_df: pd.DataFrame, group_col: str, sex_col: str | None,
                     age_col: str | None, group_new: pd.Series,
                     label_mapping: dict) -> tuple:
    """Build feature matrix X and label vector y."""
    slopes_df = slopes_df.copy()
    slopes_df["_group_new"] = group_new.values

    # Encode sex
    if sex_col and sex_col in slopes_df.columns:
        slopes_df["sex_encoded"] = slopes_df[sex_col].map({"M": 1, "F": 0, 1: 1, 2: 0})
    else:
        slopes_df["sex_encoded"] = 0

    slope_cols = [c for c in slopes_df.columns if c.endswith("_slope")]
    extra_cols = [c for c in ["sex_encoded"] + ([age_col] if age_col and age_col in slopes_df.columns else [])
                  if c in slopes_df.columns]
    feature_cols = slope_cols + extra_cols

    X     = slopes_df[feature_cols].values.astype(float)
    y_raw = slopes_df["_group_new"].values
    y     = encode_labels(y_raw, label_mapping)

    valid = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    return X[valid], y[valid].astype(int), feature_cols, slopes_df[valid].copy()


# ============================================================================
# SVM TRAINING
# ============================================================================

def train_svm(X: np.ndarray, y: np.ndarray, label_type: str,
              target_names: list) -> dict:
    """Train SVM with cross-validation."""
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm      = SVC(kernel="rbf", C=1.0, gamma="scale",
                   class_weight="balanced", random_state=42)

    cv_scores = cross_val_score(svm, X_scaled, y, cv=min(5, len(np.unique(y))))
    svm.fit(X_scaled, y)
    y_pred = svm.predict(X_scaled)
    cm     = confusion_matrix(y, y_pred)

    print(f"\nCV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"Train Accuracy: {(y_pred == y).mean():.3f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=target_names, digits=2))
    print("Confusion Matrix:")
    print(cm)

    return {
        "svm": svm, "scaler": scaler,
        "cv_scores": cv_scores,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "train_acc": float((y_pred == y).mean()),
        "confusion_matrix": cm,
    }


def stratified_analysis(X: np.ndarray, y: np.ndarray,
                         sex_values: np.ndarray, label_type: str) -> dict:
    """Run SVM per gender."""
    results = {}
    for sex_label, sex_code in [("Male", 1), ("Female", 0)]:
        mask  = sex_values == sex_code
        X_sex = X[mask]
        y_sex = y[mask]

        if len(np.unique(y_sex)) < 2:
            print(f"  ⚠ {sex_label}: not enough classes — skipping")
            continue

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_sex)
        svm      = SVC(kernel="rbf", C=1.0, gamma="scale",
                       class_weight="balanced", random_state=42)
        cv       = cross_val_score(svm, X_scaled, y_sex, cv=min(5, len(np.unique(y_sex))))
        svm.fit(X_scaled, y_sex)
        y_pred   = svm.predict(X_scaled)

        print(f"  {sex_label} (n={len(X_sex)}): CV={cv.mean():.3f}±{cv.std():.3f}  Train={((y_pred==y_sex).mean()):.3f}")
        results[sex_label] = {
            "n": len(X_sex),
            "cv_mean": float(cv.mean()),
            "cv_std": float(cv.std()),
            "train_acc": float((y_pred == y_sex).mean()),
        }
    return results


def feature_importance_anova(X: np.ndarray, y: np.ndarray,
                              feature_cols: list) -> pd.DataFrame:
    """ANOVA F-score per feature."""
    scores = []
    for i, feat in enumerate(feature_cols):
        groups = [X[y == lbl, i] for lbl in np.unique(y)]
        try:
            f_stat, p_val = f_oneway(*groups)
        except Exception:
            f_stat, p_val = np.nan, np.nan
        scores.append({"feature": feat, "f_score": f_stat, "p_value": p_val})
    return pd.DataFrame(scores).sort_values("f_score", ascending=False)


# ============================================================================
# RUN ONE VARIANT
# ============================================================================

def run_variant(slopes_df: pd.DataFrame, group_col: str, sex_col: str | None,
                age_col: str | None, groups: dict,
                variant: str, output_dir: Path) -> dict:
    """Run one SVM variant (social / duration / intervention)."""

    print(f"\n{'='*60}")
    print(f"VARIANT: {variant.upper()}")
    print("="*60)

    if variant == "social":
        if not groups["alone"] or not groups["social"]:
            print(f"  x Skipping social variant — alone or social groups not detected")
            print(f"    Detected groups: {groups['all']}")
            print(f"    Fix: add 'alone_groups' and 'group_groups' to CLI flags")
            return {}
        group_new    = remap_groups(slopes_df[group_col], groups["alone"],
                                    groups["social"], "alone", "group",
                                    groups["control"])
        label_map    = {"alone": 0, "group": 1, "control": 2}
        target_names = ["Alone", "Group", "Control"]

    elif variant == "duration":
        if not groups["short"] or not groups["long"]:
            print(f"  x Skipping duration variant — short or long groups not detected")
            print(f"    Detected groups: {groups['all']}")
            print(f"    Fix: add 'short_groups' and 'long_groups' to CLI flags")
            return {}
        group_new    = remap_groups(slopes_df[group_col], groups["short"],
                                    groups["long"], "2w", "4w",
                                    groups["control"])
        label_map    = {"2w": 0, "4w": 1, "control": 2}
        target_names = ["2 weeks", "4 weeks", "Control"]

    elif variant == "intervention":
        if not groups["intervention"]:
            print(f"  x Skipping intervention variant — no intervention groups detected")
            return {}
        group_new    = remap_groups(slopes_df[group_col], groups["intervention"],
                                    [], "intervention", "_none_",
                                    groups["control"])
        label_map    = {"intervention": 0, "control": 1}
        target_names = ["Intervention", "Control"]

    else:
        sys.exit(f"x Unknown variant: {variant}")

    X, y, feature_cols, slopes_filt = prepare_features(
        slopes_df, group_col, sex_col, age_col, group_new, label_map
    )

    if len(X) == 0 or len(np.unique(y)) < 2:
        print(f"  x Not enough data for {variant} variant")
        return {}

    print(f"  Samples: {len(X)}, Features: {len(feature_cols)}, Classes: {np.unique(y)}")

    # SVM
    svm_results = train_svm(X, y, variant, target_names)

    # Feature importance
    feat_df = feature_importance_anova(X, y, feature_cols)
    print("\nTop 10 Features:")
    print(feat_df.head(10).to_string(index=False))

    # Stratified
    print(f"\nGender-Stratified:")
    if sex_col and "sex_encoded" in slopes_filt.columns:
        sex_vals = slopes_filt["sex_encoded"].values.astype(float)
    else:
        sex_vals = np.zeros(len(X))
    strat = stratified_analysis(X, y, sex_vals, variant)

    # Save
    slopes_filt.to_csv(output_dir / f"slopes_{variant}.csv", index=False)
    feat_df.to_csv(output_dir / f"feature_importance_{variant}.csv", index=False)
    print(f"  + Saved: slopes_{variant}.csv, feature_importance_{variant}.csv")

    return {
        "cv_mean":          svm_results["cv_mean"],
        "cv_std":           svm_results["cv_std"],
        "train_acc":        svm_results["train_acc"],
        "confusion_matrix": svm_results["confusion_matrix"].tolist(),
        "stratified":       strat,
        "top_features":     feat_df.head(10).to_dict("records"),
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args   = parse_args()
    config = build_config(args)

    metrics_file  = config["metrics_file"]
    metadata_file = config.get("metadata_file")
    output_dir    = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    _start_time = datetime.now()
    print("  Started:  " + _start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("SVM ANALYSIS")
    print("=" * 70)
    print(f"Metrics file: {metrics_file}")
    print(f"Output dir:   {output_dir}")
    print()

    # ── Load & detect ──────────────────────────────────────────────────────
    print("Loading data...")
    raw_df = pd.read_parquet(metrics_file)

    print("\nAuto-detecting columns...")
    cols = detect_columns(raw_df)

    # Merge metadata if provided
    if metadata_file:
        df = load_metrics(metrics_file, metadata_file, cols)
        cols = detect_columns(df)
    else:
        df = raw_df

    # Aggregate nodal → subject-session if needed
    df = aggregate_nodal(df, cols)

    # Re-detect after aggregation
    cols = detect_columns(df)

    group_col = cols["group_col"]
    sex_col = cols["sex_col"]
    age_col = cols["age_col"]
    subject_col = cols["subject_col"]

    if not group_col:
        sys.exit(
            "x Could not detect group column.\n"
            f"  Available columns: {list(df.columns)}\n"
            "  Expected a column named: group, condition, arm, or intervention"
        )

    print("\nAuto-detecting groups...")
    run_spec_path = None
    groups = detect_or_ask_groups(df, group_col, config, run_spec_path)
    print()
    # ── Slopes ─────────────────────────────────────────────────────────────
    print("Calculating slopes...")
    slopes_df = calculate_slopes(df, cols)

    if len(slopes_df) == 0:
        sys.exit("x No slopes could be calculated — check that subjects have ≥2 sessions")

    # ── Run variants ───────────────────────────────────────────────────────
    all_results = {}

    for variant in ["social", "duration", "intervention"]:
        result = run_variant(slopes_df, group_col, sex_col, age_col,
                             groups, variant, output_dir)
        if result:
            all_results[variant] = result

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for variant, res in all_results.items():
        print(f"  {variant:15s}: CV={res['cv_mean']:.3f}±{res['cv_std']:.3f}  Train={res['train_acc']:.3f}")

    # Save summary
    summary_file = output_dir / "svm_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n+ Saved: svm_summary.json")
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()