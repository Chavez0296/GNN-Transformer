#!/usr/bin/env python3
"""Visualize model leaderboard results for graph-level runs.

Examples:
  python plot_gnn_model_comparison.py --artifacts-root GNN_Fall_Artifacts
  python plot_gnn_model_comparison.py --artifacts-root GNN_Fall_Artifacts_multiseed_final
  python plot_gnn_model_comparison.py --leaderboard-json GNN_Fall_Artifacts/leaderboard.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_model_name(name: str) -> str:
    if name.startswith("mlp::"):
        return "mlp"
    return name


def load_single_rows(path: Path) -> List[Dict[str, float]]:
    payload = read_json(path)
    rows = payload.get("results", [])
    out = []
    for row in rows:
        out.append(
            {
                "model": normalize_model_name(str(row["model"])),
                "val_accuracy": float(row["val_accuracy"]),
                "test_accuracy": float(row["test_accuracy"]),
                "val_f1_macro": float(row["val_f1_macro"]),
                "test_f1_macro": float(row["test_f1_macro"]),
                "val_roc_auc": float(row["val_roc_auc"]),
                "test_roc_auc": float(row["test_roc_auc"]),
                "is_aggregate": 0.0,
            }
        )
    return out


def load_aggregate_rows(path: Path) -> List[Dict[str, float]]:
    payload = read_json(path)
    rows = payload.get("results", [])
    out = []
    for row in rows:
        out.append(
            {
                "model": normalize_model_name(str(row["model"])),
                "val_accuracy": float(row["val_accuracy_mean"]),
                "test_accuracy": float(row["test_accuracy_mean"]),
                "val_f1_macro": float(row["val_f1_macro_mean"]),
                "test_f1_macro": float(row["test_f1_macro_mean"]),
                "val_roc_auc": float(row["val_roc_auc_mean"]),
                "test_roc_auc": float(row["test_roc_auc_mean"]),
                "val_accuracy_std": float(row.get("val_accuracy_std", 0.0)),
                "test_accuracy_std": float(row.get("test_accuracy_std", 0.0)),
                "val_f1_macro_std": float(row.get("val_f1_macro_std", 0.0)),
                "test_f1_macro_std": float(row.get("test_f1_macro_std", 0.0)),
                "val_roc_auc_std": float(row.get("val_roc_auc_std", 0.0)),
                "test_roc_auc_std": float(row.get("test_roc_auc_std", 0.0)),
                "is_aggregate": 1.0,
            }
        )
    return out


def sort_rows(rows: List[Dict[str, float]], key: str = "test_f1_macro") -> List[Dict[str, float]]:
    return sorted(rows, key=lambda r: float(r.get(key, 0.0)), reverse=True)


def grouped_metric_plot(rows: List[Dict[str, float]], split: str, out_path: Path) -> None:
    metrics = [f"{split}_accuracy", f"{split}_f1_macro", f"{split}_roc_auc"]
    labels = ["Accuracy", "Macro-F1", "ROC-AUC"]
    models = [str(r["model"]) for r in rows]
    x = np.arange(len(models), dtype=float)
    width = 0.24

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.2), 5.5))
    for i, (metric, metric_label) in enumerate(zip(metrics, labels)):
        vals = [float(r.get(metric, 0.0)) for r in rows]
        std_key = f"{metric}_std"
        errs = [float(r.get(std_key, 0.0)) for r in rows]
        use_err = any(e > 0 for e in errs)
        ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            yerr=errs if use_err else None,
            capsize=3 if use_err else 0,
            label=metric_label,
        )

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison ({split.title()})")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def by_seed_boxplot(by_seed_path: Path, metric: str, out_path: Path) -> None:
    payload = read_json(by_seed_path)
    runs = payload.get("runs", [])
    per_model: Dict[str, List[float]] = {}

    for run in runs:
        for row in run.get("results", []):
            model = normalize_model_name(str(row["model"]))
            per_model.setdefault(model, []).append(float(row[metric]))

    if not per_model:
        return

    items = sorted(per_model.items(), key=lambda kv: np.mean(kv[1]), reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 5.5))
    ax.boxplot(values, tick_labels=labels, showmeans=True)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} Across Seeds")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def resolve_inputs(args: argparse.Namespace) -> Tuple[List[Dict[str, float]], Path, Path | None]:
    if args.leaderboard_json:
        leaderboard_path = Path(args.leaderboard_json)
        out_dir = Path(args.out_dir) if args.out_dir else leaderboard_path.parent / "comparison_charts"
        rows = load_single_rows(leaderboard_path)
        return rows, out_dir, None

    root = Path(args.artifacts_root)
    agg = root / "leaderboard_aggregate.json"
    by_seed = root / "leaderboard_by_seed.json"
    single = root / "leaderboard.json"

    if agg.exists():
        out_dir = Path(args.out_dir) if args.out_dir else root / "comparison_charts"
        return load_aggregate_rows(agg), out_dir, by_seed if by_seed.exists() else None
    if single.exists():
        out_dir = Path(args.out_dir) if args.out_dir else root / "comparison_charts"
        return load_single_rows(single), out_dir, None

    raise FileNotFoundError(f"Could not find leaderboard JSON in: {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot comparison charts for graph-level model runs")
    parser.add_argument("--artifacts-root", type=str, default="GNN_Fall_Artifacts")
    parser.add_argument("--leaderboard-json", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    rows, out_dir, by_seed_path = resolve_inputs(args)
    rows = sort_rows(rows, key="test_f1_macro")
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped_metric_plot(rows, split="test", out_path=out_dir / "model_comparison_test_metrics.png")
    grouped_metric_plot(rows, split="val", out_path=out_dir / "model_comparison_val_metrics.png")

    if by_seed_path is not None:
        by_seed_boxplot(by_seed_path, metric="test_f1_macro", out_path=out_dir / "model_test_f1_across_seeds.png")
        by_seed_boxplot(by_seed_path, metric="test_roc_auc", out_path=out_dir / "model_test_rocauc_across_seeds.png")

    print(f"[done] Wrote comparison charts to: {out_dir}")


if __name__ == "__main__":
    main()
