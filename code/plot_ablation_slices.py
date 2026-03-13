#!/usr/bin/env python3
"""Plot ablation and slice metrics from gps_artifacts CSV outputs.

Usage:
  python plot_ablation_slices.py
  python plot_ablation_slices.py --run-id 20260205_192420
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WORK_ROOT = SCRIPT_DIR.parent
ART = WORK_ROOT / "results" / "gps_artifacts"


def latest_file(prefix: str) -> Path:
    files = sorted(ART.glob(f"{prefix}_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No files found for {prefix}_*.csv in {ART}")
    return files[-1]


def file_for_run(prefix: str, run_id: str | None) -> Path:
    if run_id:
        p = ART / f"{prefix}_{run_id}.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    return latest_file(prefix)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def weighted_slice(rows, metric: str):
    agg = defaultdict(lambda: {"num": 0.0, "den": 0})
    for r in rows:
        key = (r["model"], r["slice_type"], r["bucket"])
        support = int(r["support"])
        v = float(r[metric])
        if np.isnan(v):
            continue
        agg[key]["num"] += v * support
        agg[key]["den"] += support
    out = {}
    for k, d in agg.items():
        out[k] = d["num"] / max(1, d["den"])
    return out


def plot_overview(summary_rows, out_path: Path):
    by_model = {r["model"]: r for r in summary_rows}
    models = ["baseline_sage", "baseline_plus_gps_transformer"]
    labels = ["Baseline SAGE", "SAGE + GPS"]
    metrics = ["accuracy", "f1_macro", "roc_auc", "pr_auc"]

    x = np.arange(len(metrics))
    width = 0.36

    vals = []
    errs = []
    for m in models:
        row = by_model.get(m)
        vals.append([float(row[f"{k}_mean"]) for k in metrics])
        errs.append([float(row[f"{k}_std"]) for k in metrics])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, vals[0], width, yerr=errs[0], capsize=3, label=labels[0])
    ax.bar(x + width / 2, vals[1], width, yerr=errs[1], capsize=3, label=labels[1])
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Macro-F1", "ROC-AUC", "PR-AUC"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Overview (mean ± std)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_slice_metric(slice_rows, slice_type: str, metric: str, out_path: Path):
    order = ["low", "mid", "high", "unknown"]
    agg = weighted_slice([r for r in slice_rows if r["slice_type"] == slice_type], metric)

    buckets = [b for b in order if ("baseline_sage", slice_type, b) in agg or ("baseline_plus_gps_transformer", slice_type, b) in agg]
    if not buckets:
        return

    if len(buckets) == 1:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.axis("off")
        ax.text(0.02, 0.6, f"Only one '{slice_type}' bucket present: {buckets[0]}", fontsize=12)
        ax.text(0.02, 0.35, "This slice is not discriminative for this run.", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return

    base = [agg.get(("baseline_sage", slice_type, b), np.nan) for b in buckets]
    gps = [agg.get(("baseline_plus_gps_transformer", slice_type, b), np.nan) for b in buckets]

    x = np.arange(len(buckets))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(x - width / 2, base, width, label="Baseline SAGE")
    ax.bar(x + width / 2, gps, width, label="SAGE + GPS")
    ax.set_xticks(x)
    ax.set_xticklabels([b.title() for b in buckets])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} by {slice_type.title()} Bucket")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None, help="Run id suffix, e.g. 20260205_192420")
    args = ap.parse_args()

    summary_path = file_for_run("ablation_summary", args.run_id)
    run_id = summary_path.stem.replace("ablation_summary_", "")
    slice_path = file_for_run("slice_metrics", run_id)

    summary_rows = read_csv(summary_path)
    slice_rows = read_csv(slice_path)

    out_dir = ART / f"plots_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_overview(summary_rows, out_dir / "ablation_overview.png")
    plot_slice_metric(slice_rows, "degree", "f1_macro", out_dir / "slice_degree_f1.png")
    plot_slice_metric(slice_rows, "degree", "pr_auc", out_dir / "slice_degree_pr_auc.png")
    plot_slice_metric(slice_rows, "purity", "f1_macro", out_dir / "slice_purity_f1.png")

    print(out_dir)


if __name__ == "__main__":
    main()
