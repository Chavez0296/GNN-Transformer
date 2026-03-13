#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_OUT_DIR = os.path.join(WORK_ROOT, "results", "gps_artifacts", "improve_round", "graph_slices")


def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def load_prediction_rows(pattern: str, variant: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(glob.glob(pattern)):
        for row in read_rows(path):
            rows.append(
                {
                    "variant": variant,
                    "fold": row["fold"],
                    "graph_index": int(row["graph_index"]),
                    "true_label": int(row["true_label"]),
                    "pred_label": int(row["pred_label"]),
                    "pred_prob_pos": safe_float(row["pred_prob_pos"]),
                    "num_nodes": safe_float(row["num_nodes"]),
                    "num_edges": safe_float(row["num_edges"]),
                    "density": safe_float(row["density"]),
                    "est_diameter": safe_float(row["est_diameter"]),
                }
            )
    return rows


def make_bucket_edges(values: Sequence[float]) -> List[float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return [0.0, 1.0, 2.0, 3.0]
    q1, q2 = np.quantile(arr, [1.0 / 3.0, 2.0 / 3.0])
    if q1 == q2:
        q2 = q1 + 1e-9
    return [float(np.min(arr)), float(q1), float(q2), float(np.max(arr)) + 1e-9]


def bucket_label(value: float, edges: Sequence[float]) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value <= edges[1]:
        return "low"
    if value <= edges[2]:
        return "mid"
    return "high"


def bucket_rows(rows: List[Dict[str, object]], metric_name: str, edges: Sequence[float]) -> None:
    key = f"{metric_name}_bucket"
    for row in rows:
        row[key] = bucket_label(float(row[metric_name]), edges)


def compute_metrics(rows: List[Dict[str, object]]) -> Dict[str, float]:
    y_true = np.asarray([int(r["true_label"]) for r in rows], dtype=int)
    y_pred = np.asarray([int(r["pred_label"]) for r in rows], dtype=int)
    y_prob = np.asarray([float(r["pred_prob_pos"]) for r in rows], dtype=float)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "support": float(y_true.shape[0]),
    }
    if np.unique(y_true).size >= 2 and np.isfinite(y_prob).all():
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["roc_auc"] = float("nan")
    return out


def summarize_slices(rows: List[Dict[str, object]], slice_key: str) -> List[Dict[str, object]]:
    by_fold_bucket: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_fold_bucket[(str(row["variant"]), str(row["fold"]), str(row[slice_key]))].append(row)

    fold_rows: List[Dict[str, object]] = []
    for (variant, fold, bucket), items in sorted(by_fold_bucket.items()):
        metrics = compute_metrics(items)
        fold_rows.append(
            {
                "variant": variant,
                "fold": fold,
                "slice_type": slice_key.replace("_bucket", ""),
                "bucket": bucket,
                **metrics,
            }
        )

    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in fold_rows:
        grouped[(str(row["variant"]), str(row["bucket"]))].append(row)

    summary_rows: List[Dict[str, object]] = []
    for (variant, bucket), items in sorted(grouped.items()):
        for metric in ("accuracy", "f1_macro", "roc_auc", "support"):
            vals = np.asarray([float(x[metric]) for x in items], dtype=float)
            mean = float(np.nanmean(vals))
            std = float(np.nanstd(vals))
            if metric == "accuracy":
                row = {
                    "variant": variant,
                    "slice_type": slice_key.replace("_bucket", ""),
                    "bucket": bucket,
                    "accuracy_mean": mean,
                    "accuracy_std": std,
                }
                summary_rows.append(row)
        last = summary_rows[-1]
        last["f1_macro_mean"] = float(np.nanmean([float(x["f1_macro"]) for x in items]))
        last["f1_macro_std"] = float(np.nanstd([float(x["f1_macro"]) for x in items]))
        last["roc_auc_mean"] = float(np.nanmean([float(x["roc_auc"]) for x in items]))
        last["roc_auc_std"] = float(np.nanstd([float(x["roc_auc"]) for x in items]))
        last["support_mean"] = float(np.nanmean([float(x["support"]) for x in items]))
    return fold_rows, summary_rows


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_slice(summary_rows: List[Dict[str, object]], slice_type: str, metric: str, out_path: str) -> None:
    order = ["low", "mid", "high", "unknown"]
    rows = [r for r in summary_rows if r["slice_type"] == slice_type and r["bucket"] in order]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: (order.index(str(r["bucket"])), str(r["variant"])))
    buckets = [b for b in order if any(r["bucket"] == b for r in rows)]
    base = [next((float(r[f"{metric}_mean"]) for r in rows if r["variant"] == "baseline" and r["bucket"] == b), np.nan) for b in buckets]
    gps = [next((float(r[f"{metric}_mean"]) for r in rows if r["variant"] == "transformer" and r["bucket"] == b), np.nan) for b in buckets]
    x = np.arange(len(buckets), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(x - width / 2, base, width, color="#8f9491", label="Baseline")
    ax.bar(x + width / 2, gps, width, color="#1f77b4", label="Transformer")
    ax.set_xticks(x)
    ax.set_xticklabels([b.title() for b in buckets])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} by {slice_type.replace('_', ' ').title()} Bucket")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize graph-level slice metrics")
    parser.add_argument("--baseline-glob", required=True)
    parser.add_argument("--transformer-glob", required=True)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_prediction_rows(args.baseline_glob, "baseline") + load_prediction_rows(args.transformer_glob, "transformer")
    if not rows:
        raise SystemExit("No prediction rows found.")

    node_edges = make_bucket_edges([float(r["num_nodes"]) for r in rows])
    density_edges = make_bucket_edges([float(r["density"]) for r in rows])
    diameter_edges = make_bucket_edges([float(r["est_diameter"]) for r in rows])
    bucket_rows(rows, "num_nodes", node_edges)
    bucket_rows(rows, "density", density_edges)
    bucket_rows(rows, "est_diameter", diameter_edges)

    all_fold_rows: List[Dict[str, object]] = []
    all_summary_rows: List[Dict[str, object]] = []
    for slice_key in ("num_nodes_bucket", "density_bucket", "est_diameter_bucket"):
        fold_rows, summary_rows = summarize_slices(rows, slice_key)
        all_fold_rows.extend(fold_rows)
        all_summary_rows.extend(summary_rows)

    write_csv(os.path.join(args.out_dir, "graph_slice_metrics_by_fold.csv"), all_fold_rows)
    write_csv(os.path.join(args.out_dir, "graph_slice_metrics_summary.csv"), all_summary_rows)
    write_csv(
        os.path.join(args.out_dir, "graph_slice_bucket_edges.csv"),
        [
            {"slice_type": "num_nodes", "low_max": node_edges[1], "mid_max": node_edges[2], "high_max": node_edges[3]},
            {"slice_type": "density", "low_max": density_edges[1], "mid_max": density_edges[2], "high_max": density_edges[3]},
            {"slice_type": "est_diameter", "low_max": diameter_edges[1], "mid_max": diameter_edges[2], "high_max": diameter_edges[3]},
        ],
    )

    plot_slice(all_summary_rows, "num_nodes", "f1_macro", os.path.join(args.out_dir, "slice_num_nodes_f1.png"))
    plot_slice(all_summary_rows, "density", "f1_macro", os.path.join(args.out_dir, "slice_density_f1.png"))
    plot_slice(all_summary_rows, "est_diameter", "f1_macro", os.path.join(args.out_dir, "slice_est_diameter_f1.png"))
    plot_slice(all_summary_rows, "num_nodes", "roc_auc", os.path.join(args.out_dir, "slice_num_nodes_auc.png"))
    plot_slice(all_summary_rows, "density", "roc_auc", os.path.join(args.out_dir, "slice_density_auc.png"))
    plot_slice(all_summary_rows, "est_diameter", "roc_auc", os.path.join(args.out_dir, "slice_est_diameter_auc.png"))
    print("[done]", args.out_dir)


if __name__ == "__main__":
    main()
