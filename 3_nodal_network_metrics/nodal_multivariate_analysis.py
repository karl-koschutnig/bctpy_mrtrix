#!/usr/bin/env python3
"""
SCRIPT: Node-Level Comprehensive Analysis
==========================================

PURPOSE:
    Full spectrum of statistical analyses at node level:
    1. 5-group intervention comparison (ANOVA)
    2. Social effects (alone vs group)
    3. Duration effects (2w vs 4w)
    4. Intervention vs control (binary)
    5. Gender effects
    6. Age correlations

    Groups, metrics, and columns are auto-detected from the data.
    Input is the node_level_metrics.parquet from 02_nodal_metrics.py.

USAGE:
    python 04_nodal_comprehensive_analysis.py run_spec.json
    python 04_nodal_comprehensive_analysis.py --node-metrics-dir /path/to/dir --output-dir /path/to/out

AUTHOR: Analysis Pipeline
VERSION: 1.0 (Auto-detection, run_spec driven)
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
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, pearsonr, spearmanr

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Local
sys.path.append(str(Path(__file__).resolve().parent.parent / "1_utilities"))
from group_detection import detect_or_ask_groups


# ============================================================================
# CLI / run_spec LOADING
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
        description="Node-level comprehensive analysis. Pass run_spec.json or explicit paths."
    )
    parser.add_argument("run_spec",            nargs="?", help="Path to run_spec.json")
    parser.add_argument("--node-metrics-dir",  help="Directory with node_level_metrics.parquet")
    parser.add_argument("--output-dir",        help="Output directory")
    parser.add_argument("--control-group",     help="Control group label (auto-detected if omitted)")
    parser.add_argument("--alone-groups",      nargs="+", help="Alone group labels (auto-detected if omitted)")
    parser.add_argument("--group-groups",      nargs="+", help="Group intervention labels (auto-detected if omitted)")
    parser.add_argument("--short-groups",      nargs="+", help="Short duration labels (auto-detected if omitted)")
    parser.add_argument("--long-groups",       nargs="+", help="Long duration labels (auto-detected if omitted)")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    config: dict = {}

    if args.run_spec:
        spec_path = Path(args.run_spec).expanduser().resolve()
        if not spec_path.exists():
            sys.exit(f"x run_spec not found: {spec_path}")
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        inputs  = spec.get("inputs", {})
        outputs = spec.get("outputs", {})
        config["node_metrics_dir"] = inputs.get("node_metrics_dir")
        config["output_dir"]       = outputs.get("output_dir")
        config["control_group"]    = spec.get("control_group",  None)
        config["alone_groups"]     = spec.get("alone_groups",   None)
        config["group_groups"]     = spec.get("group_groups",   None)
        config["short_groups"]     = spec.get("short_groups",   None)
        config["long_groups"]      = spec.get("long_groups",    None)

    if args.node_metrics_dir: config["node_metrics_dir"] = args.node_metrics_dir
    if args.output_dir:       config["output_dir"]       = args.output_dir
    if args.control_group:    config["control_group"]    = args.control_group
    if args.alone_groups:     config["alone_groups"]     = args.alone_groups
    if args.group_groups:     config["group_groups"]     = args.group_groups
    if args.short_groups:     config["short_groups"]     = args.short_groups
    if args.long_groups:      config["long_groups"]      = args.long_groups

    missing = [k for k in ("node_metrics_dir", "output_dir") if not config.get(k)]
    if missing:
        sys.exit(
            f"x Missing required config: {', '.join(missing)}\n"
            "  Provide via run_spec.json or CLI flags."
        )

    config["node_metrics_dir"] = Path(config["node_metrics_dir"]).expanduser().resolve()
    config["output_dir"]       = Path(config["output_dir"]).expanduser().resolve()
    return config


# ============================================================================
# AUTO-DETECTION
# ============================================================================

def detect_columns(node_df: pd.DataFrame) -> dict:
    """Auto-detect group, session, sex, age, subject, metric columns."""
    cols_lower = {c.lower(): c for c in node_df.columns}

    def find(candidates):
        for c in candidates:
            if c in cols_lower:
                return cols_lower[c]
        for c in candidates:
            for cl, co in cols_lower.items():
                if c in cl:
                    return co
        return None

    exclude = {"subject", "session", "ses", "node", "group", "condition",
               "arm", "intervention", "sex", "gender", "age", "atlas",
               "hub_type", "community"}

    detected = {
        "subject_col": find(["subject", "participant_id", "participant"]),
        "session_col": find(["session", "ses", "timepoint", "visit"]),
        "group_col":   find(["group", "condition", "arm", "intervention"]),
        "sex_col":     find(["sex", "gender"]),
        "age_col":     find(["age"]),
        "metric_cols": [c for c in node_df.columns
                        if c.lower() not in exclude
                        and pd.api.types.is_numeric_dtype(node_df[c])
                        and c.lower() != "node"],
    }

    print("  Auto-detected columns:")
    for k, v in detected.items():
        print(f"    {k}: {v}")
    return detected


def detect_groups(node_df: pd.DataFrame, group_col: str, config: dict):
    """
    Auto-detect control, alone, social, short, long groups from labels or size.
    Falls back to config overrides where provided.
    """
    all_groups = node_df[group_col].dropna().unique().tolist()
    group_counts = node_df.groupby(group_col)["node"].count()

    # Control: smallest group or config override
    if config.get("control_group") is not None:
        control = config["control_group"]
    else:
        control = group_counts.idxmin()

    intervention = [g for g in all_groups if g != control]

    # Try to infer alone/group/short/long from labels
    def match(groups, keywords):
        return [g for g in groups if any(k in str(g).lower() for k in keywords)]

    alone  = config.get("alone_groups")  or match(intervention, ["alone", "individual", "solo"])
    social = config.get("group_groups")  or match(intervention, ["group", "social", "team"])
    short  = config.get("short_groups")  or match(intervention, ["2w", "short", "2week", "week2"])
    long_  = config.get("long_groups")   or match(intervention, ["4w", "long",  "4week", "week4"])

    # Fallback: split intervention in half by position
    if not alone and not social and len(intervention) >= 2:
        half   = len(intervention) // 2
        alone  = intervention[:half]
        social = intervention[half:]
        print("  ! Could not detect alone/group labels — splitting intervention groups equally")

    if not short and not long_ and len(intervention) >= 2:
        half  = len(intervention) // 2
        short = intervention[:half]
        long_ = intervention[half:]
        print("  ! Could not detect short/long labels — splitting intervention groups equally")

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
    return groups


def detect_n_nodes(node_df: pd.DataFrame) -> int:
    n = int(node_df["node"].max())
    print(f"  n_nodes: {n}")
    return n


def normalize_sex(node_df: pd.DataFrame, sex_col: str) -> pd.DataFrame:
    """Normalize sex column to 'M'/'F' strings."""
    if sex_col and sex_col in node_df.columns:
        node_df[sex_col] = node_df[sex_col].replace({1: "M", 2: "F"})
    return node_df


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def run_five_group_anova(node_df, metric_cols, group_col, groups, n_nodes):
    records = []
    all_groups = groups["all"]

    for node in range(1, n_nodes + 1):
        _progress(node, n_nodes, "Running analyses")
        print(f"    node {node}/{n_nodes}: running ANOVA + t-tests + correlations...  ", end="\n", flush=True)
        nd = node_df[node_df["node"] == node]
        for metric in metric_cols:
            data = nd[nd[metric].notna()]
            groups_data = [data[data[group_col] == g][metric].values
                           for g in all_groups if len(data[data[group_col] == g]) > 0]
            if len(groups_data) < 2:
                continue
            try:
                f_stat, p_val = f_oneway(*groups_data)
                grand_mean = np.mean(np.concatenate(groups_data))
                ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups_data)
                ss_total   = sum(np.sum((g - grand_mean) ** 2) for g in groups_data)
                eta2       = ss_between / ss_total if ss_total > 0 else 0
                group_means = {f"mean_{g}": np.mean(d) for g, d in zip(all_groups, groups_data)}
                records.append({"node": node, "metric": metric, "f_statistic": f_stat,
                                 "p_value": p_val, "eta_squared": eta2,
                                 "significant": p_val < 0.05, **group_means})
            except Exception:
                continue

    return pd.DataFrame(records)


def run_ttest_analysis(node_df, metric_cols, group_col, group_a, group_b,
                       label_a, label_b, n_nodes):
    """Generic t-test between two sets of groups."""
    records = []
    for node in range(1, n_nodes + 1):
        nd = node_df[node_df["node"] == node]
        for metric in metric_cols:
            a_data = nd[nd[group_col].isin(group_a) & nd[metric].notna()][metric].values
            b_data = nd[nd[group_col].isin(group_b) & nd[metric].notna()][metric].values
            if len(a_data) < 3 or len(b_data) < 3:
                continue
            try:
                t_stat, p_val  = ttest_ind(a_data, b_data)
                pooled_std     = np.sqrt((a_data.std() ** 2 + b_data.std() ** 2) / 2)
                cohens_d       = (a_data.mean() - b_data.mean()) / pooled_std if pooled_std > 0 else 0
                records.append({
                    "node": node, "metric": metric,
                    "t_statistic": t_stat, "p_value": p_val,
                    "cohens_d": cohens_d, "abs_cohens_d": abs(cohens_d),
                    f"mean_{label_a}": a_data.mean(),
                    f"mean_{label_b}": b_data.mean(),
                    "direction": f"{label_a}>{label_b}" if cohens_d > 0 else f"{label_b}>{label_a}",
                    "significant": p_val < 0.05,
                    f"n_{label_a}": len(a_data),
                    f"n_{label_b}": len(b_data),
                })
            except Exception:
                continue
    return pd.DataFrame(records)


def run_age_correlations(node_df, metric_cols, age_col, n_nodes):
    records = []
    for node in range(1, n_nodes + 1):
        nd = node_df[node_df["node"] == node]
        for metric in metric_cols:
            data = nd[nd[metric].notna() & nd[age_col].notna()]
            if len(data) < 10:
                continue
            ages   = data[age_col].values
            values = data[metric].values
            try:
                r_p, p_p = pearsonr(ages, values)
                r_s, p_s = spearmanr(ages, values)
                records.append({
                    "node": node, "metric": metric,
                    "r_pearson": r_p,  "p_pearson": p_p,
                    "r_spearman": r_s, "p_spearman": p_s,
                    "abs_r_pearson": abs(r_p),
                    "direction": "positive" if r_p > 0 else "negative",
                    "significant_pearson":  p_p < 0.05,
                    "significant_spearman": p_s < 0.05,
                    "n_samples": len(ages),
                })
            except Exception:
                continue
    return pd.DataFrame(records)


# ============================================================================
# PLOTS
# ============================================================================

def make_significance_heatmaps(results: dict, output_dir: Path) -> None:
    analyses = [
        ("5-Group",      results["five_group"],  "significant",         "5-Group ANOVA (p<0.05)"),
        ("Social",       results["social"],       "significant",         "Social Effects (Alone vs Group)"),
        ("Duration",     results["duration"],     "significant",         "Duration Effects (Short vs Long)"),
        ("Intervention", results["binary"],       "significant",         "Intervention vs Control"),
        ("Gender",       results["gender"],       "significant",         "Gender Effects"),
        ("Age",          results["age"],          "significant_pearson", "Age Correlations"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for idx, (name, df, sig_col, title) in enumerate(analyses):
        ax = axes[idx // 3, idx % 3]
        if len(df) > 0 and sig_col in df.columns:
            mat = df.pivot_table(index="node", columns="metric",
                                  values=sig_col, aggfunc="sum", fill_value=0)
            sns.heatmap(mat, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Significant"})
        else:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=14)
        ax.set_title(title, fontweight="bold", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "significance_heatmaps.png", dpi=150)
    plt.close()
    print("  + significance_heatmaps.png")


def make_effect_distributions(results: dict, metric_cols: list, output_dir: Path) -> None:
    effect_analyses = [
        ("Social",       results["social"],   "abs_cohens_d"),
        ("Duration",     results["duration"], "abs_cohens_d"),
        ("Intervention", results["binary"],   "abs_cohens_d"),
        ("Gender",       results["gender"],   "abs_cohens_d"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (name, df, col) in enumerate(effect_analyses):
        ax = axes[idx // 2, idx % 2]
        if len(df) > 0 and col in df.columns:
            available = [m for m in metric_cols if m in df["metric"].unique()]
            if available:
                bp = ax.boxplot([df[df["metric"] == m][col].values for m in available],
                                 labels=available, patch_artist=True)
                for patch in bp["boxes"]:
                    patch.set_facecolor("lightblue")
                ax.axhline(0.8, color="red",    linestyle="--", alpha=0.5, label="Large (0.8)")
                ax.axhline(0.5, color="orange", linestyle="--", alpha=0.5, label="Medium (0.5)")
                ax.legend(fontsize=8)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        else:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=14)
        ax.set_ylabel("Effect Size |d|", fontweight="bold")
        ax.set_title(f"{name} Effect Sizes", fontweight="bold")
        ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "effect_size_distributions.png", dpi=150)
    plt.close()
    print("  + effect_size_distributions.png")


def make_top_nodes_summary(results: dict, output_dir: Path) -> None:
    top_analyses = [
        ("5-Group",      results["five_group"],  "eta_squared",    "η²"),
        ("Social",       results["social"],       "abs_cohens_d",  "|d|"),
        ("Duration",     results["duration"],     "abs_cohens_d",  "|d|"),
        ("Intervention", results["binary"],       "abs_cohens_d",  "|d|"),
        ("Gender",       results["gender"],       "abs_cohens_d",  "|d|"),
        ("Age",          results["age"],          "abs_r_pearson", "|r|"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    for idx, (name, df, col, label) in enumerate(top_analyses):
        ax = axes[idx // 2, idx % 2]
        if len(df) > 0 and col in df.columns:
            top = df.nlargest(min(20, len(df)), col)
            colors = plt.cm.viridis(np.linspace(0, 1, len(top)))
            ax.barh(range(len(top)), top[col].values, color=colors)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(
                [f"Node {int(r['node'])} ({r['metric']})" for _, r in top.iterrows()],
                fontsize=7
            )
        else:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=14)
        ax.set_xlabel(f"Effect Size ({label})", fontweight="bold")
        ax.set_title(f"Top 20 Nodes: {name}", fontweight="bold", fontsize=10)
        ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_dir / "top_nodes_summary.png", dpi=150)
    plt.close()
    print("  + top_nodes_summary.png")


# ============================================================================
# SUMMARY PRINT
# ============================================================================

def print_summary(results: dict) -> None:
    print("\n" + "=" * 70)
    print("COMPREHENSIVE NODE-LEVEL ANALYSIS SUMMARY")
    print("=" * 70)

    entries = [
        ("5-Group ANOVA",        results["five_group"],  "significant",         "eta_squared",    0.14),
        ("Social Effects",       results["social"],       "significant",         "abs_cohens_d",   0.8),
        ("Duration Effects",     results["duration"],     "significant",         "abs_cohens_d",   0.8),
        ("Intervention/Control", results["binary"],       "significant",         "abs_cohens_d",   0.8),
        ("Gender Effects",       results["gender"],       "significant",         "abs_cohens_d",   0.8),
        ("Age Correlations",     results["age"],          "significant_pearson", "abs_r_pearson",  0.3),
    ]

    for name, df, sig_col, eff_col, threshold in entries:
        total = len(df)
        sig   = df[sig_col].sum()   if total > 0 and sig_col in df.columns else 0
        large = (df[eff_col] > threshold).sum() if total > 0 and eff_col in df.columns else 0
        pct_s = 100 * sig   / total if total > 0 else 0
        pct_l = 100 * large / total if total > 0 else 0
        print(f"\n{name}:")
        print(f"  Total:       {total}")
        print(f"  Significant: {sig} ({pct_s:.1f}%)")
        print(f"  Large effect (>{threshold}): {large} ({pct_l:.1f}%)")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args   = parse_args()
    config = load_config(args)

    node_metrics_dir = config["node_metrics_dir"]
    output_dir       = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    _start_time = datetime.now()
    print("  Started:  " + _start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("NODE-LEVEL COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    print(f"Input:  {node_metrics_dir}")
    print(f"Output: {output_dir}")
    print()

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading node-level metrics...")
    node_file = node_metrics_dir / "node_level_metrics.parquet"
    if not node_file.exists():
        sys.exit(f"x node_level_metrics.parquet not found in: {node_metrics_dir}")
    node_df = pd.read_parquet(node_file)
    print(f"  + {len(node_df)} records loaded")

    # ── Auto-detection ─────────────────────────────────────────────────────
    print("\nAuto-detecting data structure...")
    cols = detect_columns(node_df)
    group_col = cols["group_col"]
    sex_col = cols["sex_col"]
    age_col = cols["age_col"]
    metric_cols = cols["metric_cols"]
    n_nodes = detect_n_nodes(node_df)
    run_spec_path = Path(args.run_spec) if args.run_spec else None
    groups = detect_or_ask_groups(node_df, group_col, config, run_spec_path)
    node_df = normalize_sex(node_df, sex_col)
    print()

    results = {}

    # ── Analysis 1: 5-group ANOVA ──────────────────────────────────────────
    print("=" * 70)
    print("ANALYSIS 1: 5-GROUP ANOVA")
    print("=" * 70)
    results["five_group"] = run_five_group_anova(node_df, metric_cols, group_col, groups, n_nodes)
    df = results["five_group"]
    print(f"Analyzed {len(df)} node-metric combinations")
    if len(df) > 0:
        print(f"Significant (p<0.05): {df['significant'].sum()}")
        print(f"Strong effects (η²>0.14): {(df['eta_squared'] > 0.14).sum()}")
    df.to_parquet(output_dir / "five_group_anova.parquet", index=False)
    print("+ Saved: five_group_anova.parquet\n")

    # ── Analysis 2: Social effects ─────────────────────────────────────────
    print("=" * 70)
    print("ANALYSIS 2: SOCIAL EFFECTS (ALONE VS GROUP)")
    print("=" * 70)
    results["social"] = run_ttest_analysis(
        node_df, metric_cols, group_col,
        groups["alone"], groups["social"], "alone", "group", n_nodes
    )
    df = results["social"]
    print(f"Analyzed {len(df)} node-metric combinations")
    if len(df) > 0:
        print(f"Significant (p<0.05): {df['significant'].sum()}")
        print(f"Large effects (|d|>0.8): {(df['abs_cohens_d'] > 0.8).sum()}")
    df.to_parquet(output_dir / "social_effects.parquet", index=False)
    print("+ Saved: social_effects.parquet\n")

    # ── Analysis 3: Duration effects ───────────────────────────────────────
    print("=" * 70)
    print("ANALYSIS 3: DURATION EFFECTS (SHORT VS LONG)")
    print("=" * 70)
    results["duration"] = run_ttest_analysis(
        node_df, metric_cols, group_col,
        groups["short"], groups["long"], "short", "long", n_nodes
    )
    df = results["duration"]
    print(f"Analyzed {len(df)} node-metric combinations")
    if len(df) > 0:
        print(f"Significant (p<0.05): {df['significant'].sum()}")
        print(f"Large effects (|d|>0.8): {(df['abs_cohens_d'] > 0.8).sum()}")
    df.to_parquet(output_dir / "duration_effects.parquet", index=False)
    print("+ Saved: duration_effects.parquet\n")

    # ── Analysis 4: Intervention vs control ───────────────────────────────
    print("=" * 70)
    print("ANALYSIS 4: INTERVENTION VS CONTROL")
    print("=" * 70)
    results["binary"] = run_ttest_analysis(
        node_df, metric_cols, group_col,
        groups["intervention"], [groups["control"]], "intervention", "control", n_nodes
    )
    df = results["binary"]
    print(f"Analyzed {len(df)} node-metric combinations")
    if len(df) > 0:
        print(f"Significant (p<0.05): {df['significant'].sum()}")
        print(f"Large effects (|d|>0.8): {(df['abs_cohens_d'] > 0.8).sum()}")
    df.to_parquet(output_dir / "intervention_vs_control.parquet", index=False)
    print("+ Saved: intervention_vs_control.parquet\n")

    # ── Analysis 5: Gender effects ─────────────────────────────────────────
    print("=" * 70)
    print("ANALYSIS 5: GENDER EFFECTS")
    print("=" * 70)
    if sex_col:
        results["gender"] = run_ttest_analysis(
            node_df, metric_cols, sex_col,
            ["M"], ["F"], "male", "female", n_nodes
        )
        df = results["gender"]
        print(f"Analyzed {len(df)} node-metric combinations")
        if len(df) > 0:
            print(f"Significant (p<0.05): {df['significant'].sum()}")
            print(f"Large effects (|d|>0.8): {(df['abs_cohens_d'] > 0.8).sum()}")
        df.to_parquet(output_dir / "gender_effects.parquet", index=False)
        print("+ Saved: gender_effects.parquet")
    else:
        print("x Sex column not detected — skipping gender analysis")
        results["gender"] = pd.DataFrame()
    print()

    # ── Analysis 6: Age correlations ───────────────────────────────────────
    print("=" * 70)
    print("ANALYSIS 6: AGE CORRELATIONS")
    print("=" * 70)
    if age_col:
        results["age"] = run_age_correlations(node_df, metric_cols, age_col, n_nodes)
        df = results["age"]
        print(f"Analyzed {len(df)} node-metric combinations")
        if len(df) > 0:
            print(f"Significant (p<0.05, Pearson): {df['significant_pearson'].sum()}")
            print(f"Strong correlations (|r|>0.3): {(df['abs_r_pearson'] > 0.3).sum()}")
        df.to_parquet(output_dir / "age_correlations.parquet", index=False)
        print("+ Saved: age_correlations.parquet")
    else:
        print("x Age column not detected — skipping age analysis")
        results["age"] = pd.DataFrame()
    print()

    # ── Plots ──────────────────────────────────────────────────────────────
    print("Creating plots...")
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    make_significance_heatmaps(results, plot_dir)
    make_effect_distributions(results, metric_cols, plot_dir)
    make_top_nodes_summary(results, plot_dir)

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary(results)
    _end_time = datetime.now()
    _duration = _end_time - _start_time
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("  Started:  " + _start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("  Finished: " + _end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("  Duration: " + str(_duration).split(".")[0])
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()