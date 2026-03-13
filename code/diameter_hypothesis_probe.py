#!/usr/bin/env python3
"""
Diameter Hypothesis Probe: Long-range evidence probe #2

Hypothesis: Transformer attention helps more on graphs with larger estimated diameter,
because larger diameter implies longer paths where local message passing takes more
hops to propagate information (vs. global attention which captures it directly).

This script runs targeted experiments to test this hypothesis with statistical tests.
"""
import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats as scipy_stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DEFAULT_SLICE_PATH = os.path.join(
    WORK_ROOT, "results", "gps_artifacts", "improve_round", "graph_slices",
    "sage_baseline_vs_performer", "graph_slice_metrics_by_fold.csv"
)
DEFAULT_OUT_DIR = os.path.join(
    WORK_ROOT, "results", "gps_artifacts", "improve_round", "diameter_hypothesis"
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_fold_metrics(path: str) -> List[Dict[str, Any]]:
    """Load per-fold slice metrics."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def analyze_diameter_hypothesis(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze whether transformer helps more on high-diameter graphs.
    
    Returns statistical tests comparing:
    - Baseline vs Transformer F1 on high-diameter graphs
    - Baseline vs Transformer F1 on low-diameter graphs
    - Transformer improvement on high vs low diameter
    """
    # Filter for diameter slices
    diameter_rows = [r for r in rows if r.get("slice_type") == "est_diameter"]
    
    # Organize by bucket and variant
    data: Dict[str, Dict[str, List[float]]] = {
        "baseline": {"low": [], "mid": [], "high": []},
        "transformer": {"low": [], "mid": [], "high": []},
    }
    
    for row in diameter_rows:
        variant = row["variant"]
        bucket = row["bucket"]
        f1 = float(row["f1_macro"])
        if variant in data and bucket in data[variant]:
            data[variant][bucket].append(f1)
    
    results = {
        "summary": {},
        "hypothesis_tests": [],
    }
    
    # Summary statistics
    for variant in ["baseline", "transformer"]:
        for bucket in ["low", "mid", "high"]:
            vals = data[variant][bucket]
            if vals:
                results["summary"][f"{variant}_{bucket}_f1_mean"] = np.mean(vals)
                results["summary"][f"{variant}_{bucket}_f1_std"] = np.std(vals)
                results["summary"][f"{variant}_{bucket}_n"] = len(vals)
    
    # Test 1: Transformer vs Baseline on HIGH diameter
    baseline_high = data["baseline"]["high"]
    trans_high = data["transformer"]["high"]
    if len(baseline_high) >= 2 and len(trans_high) >= 2:
        stat, pval = scipy_stats.ttest_ind(trans_high, baseline_high, equal_var=False)
        diff = np.mean(trans_high) - np.mean(baseline_high)
        results["hypothesis_tests"].append({
            "test": "Transformer vs Baseline on HIGH diameter",
            "transformer_mean": np.mean(trans_high),
            "baseline_mean": np.mean(baseline_high),
            "difference": diff,
            "t_statistic": stat,
            "p_value": pval,
            "significant_0.10": pval < 0.10,
            "direction": "transformer_better" if diff > 0 else "baseline_better",
        })
    
    # Test 2: Transformer vs Baseline on LOW diameter
    baseline_low = data["baseline"]["low"]
    trans_low = data["transformer"]["low"]
    if len(baseline_low) >= 2 and len(trans_low) >= 2:
        stat, pval = scipy_stats.ttest_ind(trans_low, baseline_low, equal_var=False)
        diff = np.mean(trans_low) - np.mean(baseline_low)
        results["hypothesis_tests"].append({
            "test": "Transformer vs Baseline on LOW diameter",
            "transformer_mean": np.mean(trans_low),
            "baseline_mean": np.mean(baseline_low),
            "difference": diff,
            "t_statistic": stat,
            "p_value": pval,
            "significant_0.10": pval < 0.10,
            "direction": "transformer_better" if diff > 0 else "baseline_better",
        })
    
    # Test 3: Is transformer improvement LARGER on high diameter vs low diameter?
    # (Interaction effect test)
    if len(baseline_high) >= 2 and len(trans_high) >= 2 and len(baseline_low) >= 2 and len(trans_low) >= 2:
        improvement_high = np.mean(trans_high) - np.mean(baseline_high)
        improvement_low = np.mean(trans_low) - np.mean(baseline_low)
        
        # Bootstrap to get CI on the interaction
        n_bootstrap = 1000
        np.random.seed(42)
        interaction_samples = []
        for _ in range(n_bootstrap):
            bh = np.random.choice(baseline_high, size=len(baseline_high), replace=True)
            th = np.random.choice(trans_high, size=len(trans_high), replace=True)
            bl = np.random.choice(baseline_low, size=len(baseline_low), replace=True)
            tl = np.random.choice(trans_low, size=len(trans_low), replace=True)
            imp_h = np.mean(th) - np.mean(bh)
            imp_l = np.mean(tl) - np.mean(bl)
            interaction_samples.append(imp_h - imp_l)
        
        interaction_mean = np.mean(interaction_samples)
        interaction_ci_low = np.percentile(interaction_samples, 5)
        interaction_ci_high = np.percentile(interaction_samples, 95)
        
        results["hypothesis_tests"].append({
            "test": "Interaction: Transformer improvement HIGH vs LOW diameter",
            "improvement_on_high": improvement_high,
            "improvement_on_low": improvement_low,
            "interaction_effect": improvement_high - improvement_low,
            "bootstrap_mean": interaction_mean,
            "ci_90_low": interaction_ci_low,
            "ci_90_high": interaction_ci_high,
            "hypothesis_supported": interaction_ci_low > 0,  # CI above 0 = transformer helps more on high diameter
            "interpretation": (
                "Transformer helps MORE on high-diameter graphs" if interaction_ci_low > 0 else
                "NO evidence transformer helps more on high-diameter graphs"
            ),
        })
    
    return results


def analyze_node_count_hypothesis(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Secondary analysis: Does transformer help more on larger graphs (more nodes)?
    Larger graphs may benefit more from global attention.
    """
    node_rows = [r for r in rows if r.get("slice_type") == "num_nodes"]
    
    data: Dict[str, Dict[str, List[float]]] = {
        "baseline": {"low": [], "mid": [], "high": []},
        "transformer": {"low": [], "mid": [], "high": []},
    }
    
    for row in node_rows:
        variant = row["variant"]
        bucket = row["bucket"]
        f1 = float(row["f1_macro"])
        if variant in data and bucket in data[variant]:
            data[variant][bucket].append(f1)
    
    results = {"tests": []}
    
    # Test: Transformer improvement on high vs low node count
    baseline_high = data["baseline"]["high"]
    trans_high = data["transformer"]["high"]
    baseline_low = data["baseline"]["low"]
    trans_low = data["transformer"]["low"]
    
    if all(len(d) >= 2 for d in [baseline_high, trans_high, baseline_low, trans_low]):
        improvement_high = np.mean(trans_high) - np.mean(baseline_high)
        improvement_low = np.mean(trans_low) - np.mean(baseline_low)
        
        results["tests"].append({
            "test": "Transformer improvement: HIGH vs LOW node count",
            "improvement_on_high_nodes": improvement_high,
            "improvement_on_low_nodes": improvement_low,
            "interaction_effect": improvement_high - improvement_low,
            "interpretation": (
                "Transformer helps MORE on larger graphs" if improvement_high > improvement_low else
                "Transformer helps MORE on smaller graphs (unexpected)" if improvement_low > improvement_high else
                "No difference"
            ),
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Diameter hypothesis probe")
    parser.add_argument("--slice-path", default=DEFAULT_SLICE_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    
    ensure_dir(args.out_dir)
    
    if not os.path.exists(args.slice_path):
        print(f"Error: Slice metrics file not found at {args.slice_path}")
        print("Run analyze_graph_slices.py first to generate slice metrics.")
        sys.exit(1)
    
    print(f"Loading slice metrics from {args.slice_path}...")
    rows = load_fold_metrics(args.slice_path)
    print(f"Loaded {len(rows)} rows")
    
    print("\n" + "="*70)
    print("DIAMETER HYPOTHESIS ANALYSIS")
    print("="*70)
    
    diameter_results = analyze_diameter_hypothesis(rows)
    
    print("\nSummary Statistics:")
    for key, val in diameter_results["summary"].items():
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
    
    print("\nHypothesis Tests:")
    for test in diameter_results["hypothesis_tests"]:
        print(f"\n  Test: {test['test']}")
        for k, v in test.items():
            if k != "test":
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")
    
    print("\n" + "="*70)
    print("NODE COUNT HYPOTHESIS ANALYSIS")
    print("="*70)
    
    node_results = analyze_node_count_hypothesis(rows)
    for test in node_results["tests"]:
        print(f"\n  Test: {test['test']}")
        for k, v in test.items():
            if k != "test":
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")
    
    # Save results
    combined_results = {
        "diameter_analysis": diameter_results,
        "node_count_analysis": node_results,
    }
    
    # Write summary
    summary_path = os.path.join(args.out_dir, "diameter_hypothesis_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("DIAMETER HYPOTHESIS PROBE RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write("HYPOTHESIS: Transformer attention helps more on graphs with larger\n")
        f.write("estimated diameter, because larger diameter implies longer paths\n")
        f.write("where local message passing takes more hops to propagate information.\n\n")
        
        f.write("SUMMARY STATISTICS:\n")
        for key, val in diameter_results["summary"].items():
            f.write(f"  {key}: {val:.4f}\n" if isinstance(val, float) else f"  {key}: {val}\n")
        
        f.write("\nHYPOTHESIS TESTS:\n")
        for test in diameter_results["hypothesis_tests"]:
            f.write(f"\n  {test['test']}:\n")
            for k, v in test.items():
                if k != "test":
                    if isinstance(v, float):
                        f.write(f"    {k}: {v:.4f}\n")
                    else:
                        f.write(f"    {k}: {v}\n")
        
        f.write("\n\nCONCLUSION:\n")
        interaction_test = [t for t in diameter_results["hypothesis_tests"] if "Interaction" in t["test"]]
        if interaction_test:
            t = interaction_test[0]
            f.write(f"  {t['interpretation']}\n")
            f.write(f"  Interaction effect: {t['interaction_effect']:.4f}\n")
            f.write(f"  90% CI: [{t['ci_90_low']:.4f}, {t['ci_90_high']:.4f}]\n")
    
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
