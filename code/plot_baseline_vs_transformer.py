#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS_ROOT = os.path.join(WORK_ROOT, "results", "gps_artifacts")


def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def ci95(std: float, n: int) -> float:
    if n <= 0 or math.isnan(std):
        return float("nan")
    return float(1.96 * std / math.sqrt(n))


def ref_by_variant_model(path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {"baseline": {}, "gps": {}}
    for row in read_rows(path):
        variant = row["variant"]
        model = row["model"]
        n = int(float(row["n_seeds"]))
        out[variant][model] = {
            "f1_mean": safe_float(row["f1_mean"]),
            "f1_std": safe_float(row["f1_std"]),
            "f1_ci95": ci95(safe_float(row["f1_std"]), n),
            "auc_mean": safe_float(row["auc_mean"]),
            "auc_std": safe_float(row["auc_std"]),
            "auc_ci95": ci95(safe_float(row["auc_std"]), n),
            "n_seeds": float(n),
        }
    return out


def track_by_model(path: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for row in read_rows(path):
        model = row["model"]
        n = int(float(row["n_seeds"]))
        f1_std = safe_float(row.get("f1_std", "nan"))
        auc_std = safe_float(row.get("auc_std", "nan"))
        out[model] = {
            "f1_mean": safe_float(row["f1_mean"]),
            "f1_std": f1_std,
            "f1_ci95": ci95(f1_std, n),
            "auc_mean": safe_float(row["auc_mean"]),
            "auc_std": auc_std,
            "auc_ci95": ci95(auc_std, n),
            "n_seeds": float(n),
        }
    return out


def plot_baseline_vs_initial_transformer(models: List[str], baseline: Dict[str, Dict[str, float]], gps_init: Dict[str, Dict[str, float]], out_path: str) -> None:
    x = np.arange(len(models), dtype=float)
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharex=True)
    for ax, metric_key, metric_label in [
        (axes[0], "f1", "Macro-F1"),
        (axes[1], "auc", "ROC-AUC"),
    ]:
        base_vals = [baseline[m][f"{metric_key}_mean"] for m in models]
        base_err = [baseline[m][f"{metric_key}_ci95"] for m in models]
        gps_vals = [gps_init[m][f"{metric_key}_mean"] for m in models]
        gps_err = [gps_init[m][f"{metric_key}_ci95"] for m in models]

        ax.bar(x - width / 2, base_vals, width, yerr=base_err, color="#8f9491", label="Baseline", capsize=3)
        ax.bar(x + width / 2, gps_vals, width, yerr=gps_err, color="#1f77b4", label="Transformer (initial GPS)", capsize=3)

        for i, (b, g) in enumerate(zip(base_vals, gps_vals)):
            d = g - b
            ax.text(i + width / 2, g + 0.012, f"{d:+.03f}", ha="center", va="bottom", fontsize=8)

        ax.set_ylim(0.42, 0.82)
        ax.set_ylabel(metric_label)
        ax.set_title(f"Baseline vs Initial Transformer: {metric_label}")
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha="right")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_track_vs_baseline(
    models: List[str],
    baseline: Dict[str, Dict[str, float]],
    track: Dict[str, Dict[str, float]],
    track_label: str,
    track_color: str,
    out_path: str,
) -> None:
    x = np.arange(len(models), dtype=float)
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharex=True)
    for ax, metric_key, metric_label in [
        (axes[0], "f1", "Macro-F1"),
        (axes[1], "auc", "ROC-AUC"),
    ]:
        base_vals = [baseline[m][f"{metric_key}_mean"] for m in models]
        base_err = [baseline[m][f"{metric_key}_ci95"] for m in models]
        tr_vals = [track[m][f"{metric_key}_mean"] for m in models]
        tr_err = [track[m][f"{metric_key}_ci95"] for m in models]

        ax.bar(x - width / 2, base_vals, width, yerr=base_err, color="#8f9491", label="Baseline", capsize=3)
        ax.bar(x + width / 2, tr_vals, width, yerr=tr_err, color=track_color, label=track_label, capsize=3)

        for i, (b, t) in enumerate(zip(base_vals, tr_vals)):
            d = t - b
            ax.text(i + width / 2, t + 0.012, f"{d:+.03f}", ha="center", va="bottom", fontsize=8)

        ax.set_ylim(0.42, 0.82)
        ax.set_ylabel(metric_label)
        ax.set_title(f"Baseline vs {track_label}: {metric_label}")
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha="right")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_tradeoff_scatter(
    models: List[str],
    baseline: Dict[str, Dict[str, float]],
    gps_init: Dict[str, Dict[str, float]],
    f1_track: Dict[str, Dict[str, float]],
    auc_track: Dict[str, Dict[str, float]],
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.3))

    palette = {
        "baseline": "#8f9491",
        "initial": "#1f77b4",
        "f1": "#2ca02c",
        "auc": "#ff7f0e",
    }

    for m in models:
        b = baseline[m]
        i = gps_init[m]
        f = f1_track[m]
        a = auc_track[m]

        ax.plot([b["auc_mean"], i["auc_mean"]], [b["f1_mean"], i["f1_mean"]], color="#b9bdbb", lw=1.2, alpha=0.8)
        ax.plot([i["auc_mean"], f["auc_mean"]], [i["f1_mean"], f["f1_mean"]], color="#9ed99e", lw=1.0, alpha=0.9)
        ax.plot([i["auc_mean"], a["auc_mean"]], [i["f1_mean"], a["f1_mean"]], color="#ffd2a6", lw=1.0, alpha=0.9)

        ax.scatter(b["auc_mean"], b["f1_mean"], s=40, c=palette["baseline"], marker="o")
        ax.scatter(i["auc_mean"], i["f1_mean"], s=45, c=palette["initial"], marker="^")
        ax.scatter(f["auc_mean"], f["f1_mean"], s=48, c=palette["f1"], marker="s")
        ax.scatter(a["auc_mean"], a["f1_mean"], s=48, c=palette["auc"], marker="D")

        ax.text(f["auc_mean"] + 0.0018, f["f1_mean"] + 0.0012, m, fontsize=8)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=palette["baseline"], markersize=7, label="Baseline"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=palette["initial"], markersize=7, label="Initial transformer"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=palette["f1"], markersize=7, label="F1-track pick"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=palette["auc"], markersize=7, label="AUC-track pick"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False)
    ax.set_xlabel("ROC-AUC")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Model Tradeoff Map: Baseline vs Transformer Tracks")
    ax.grid(alpha=0.25)
    ax.set_xlim(0.54, 0.73)
    ax.set_ylim(0.45, 0.70)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_delta_heatmap(
    models: List[str],
    baseline: Dict[str, Dict[str, float]],
    gps_init: Dict[str, Dict[str, float]],
    f1_track: Dict[str, Dict[str, float]],
    auc_track: Dict[str, Dict[str, float]],
    out_path: str,
) -> None:
    cols = [
        "init dF1",
        "init dAUC",
        "F1-track dF1",
        "F1-track dAUC",
        "AUC-track dF1",
        "AUC-track dAUC",
    ]

    mat = np.zeros((len(models), len(cols)), dtype=float)
    for i, m in enumerate(models):
        b = baseline[m]
        g = gps_init[m]
        f = f1_track[m]
        a = auc_track[m]
        mat[i, 0] = g["f1_mean"] - b["f1_mean"]
        mat[i, 1] = g["auc_mean"] - b["auc_mean"]
        mat[i, 2] = f["f1_mean"] - b["f1_mean"]
        mat[i, 3] = f["auc_mean"] - b["auc_mean"]
        mat[i, 4] = a["f1_mean"] - b["f1_mean"]
        mat[i, 5] = a["auc_mean"] - b["auc_mean"]

    vmax = float(np.max(np.abs(mat)))
    fig, ax = plt.subplots(figsize=(11.8, 4.6))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Delta vs baseline")

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models)
    ax.set_title("Performance Deltas vs Baseline")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.03f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot baseline vs transformer comparisons")
    parser.add_argument(
        "--seed-stability",
        default=os.path.join(RESULTS_ROOT, "final_sel", "reports", "seed_stability_summary.csv"),
    )
    parser.add_argument(
        "--f1-track",
        default=os.path.join(RESULTS_ROOT, "improve_round", "reports", "final_track_f1_first.csv"),
    )
    parser.add_argument(
        "--auc-track",
        default=os.path.join(RESULTS_ROOT, "improve_round", "reports", "final_track_auc_first.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(RESULTS_ROOT, "improve_round", "reports", "figures"),
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    refs = ref_by_variant_model(args.seed_stability)
    baseline = refs["baseline"]
    gps_init = refs["gps"]
    f1_track = track_by_model(args.f1_track)
    auc_track = track_by_model(args.auc_track)

    models = sorted(set(baseline.keys()) & set(gps_init.keys()) & set(f1_track.keys()) & set(auc_track.keys()))

    plot_baseline_vs_initial_transformer(
        models,
        baseline,
        gps_init,
        os.path.join(args.out_dir, "baseline_vs_initial_transformer.png"),
    )
    plot_track_vs_baseline(
        models,
        baseline,
        f1_track,
        track_label="Transformer F1-Track",
        track_color="#2ca02c",
        out_path=os.path.join(args.out_dir, "baseline_vs_transformer_f1_track.png"),
    )
    plot_track_vs_baseline(
        models,
        baseline,
        auc_track,
        track_label="Transformer AUC-Track",
        track_color="#ff7f0e",
        out_path=os.path.join(args.out_dir, "baseline_vs_transformer_auc_track.png"),
    )
    plot_tradeoff_scatter(
        models,
        baseline,
        gps_init,
        f1_track,
        auc_track,
        os.path.join(args.out_dir, "transformer_tradeoff_scatter.png"),
    )
    plot_delta_heatmap(
        models,
        baseline,
        gps_init,
        f1_track,
        auc_track,
        os.path.join(args.out_dir, "transformer_delta_heatmap.png"),
    )

    print("[done] Wrote figures to:", args.out_dir)


if __name__ == "__main__":
    main()
