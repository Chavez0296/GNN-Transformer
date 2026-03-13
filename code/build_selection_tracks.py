#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_RESULTS_ROOT = os.path.join(WORK_ROOT, "results", "gps_artifacts", "final_sel")


def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: str, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def aggregate_seed_rows(rows: List[Dict[str, str]], variant: str) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        model = row.get("model", "unknown")
        global_attn = row.get("global_attn", "none") if variant == "gps" else "none"
        key = (variant, model, global_attn)
        grouped[key].append(row)

    out: List[Dict[str, float]] = []
    for (var, model, global_attn), items in grouped.items():
        f1_vals = np.asarray([safe_float(r.get("f1_macro_mean", "nan")) for r in items], dtype=float)
        auc_vals = np.asarray([safe_float(r.get("roc_auc_mean", "nan")) for r in items], dtype=float)
        n = int(len(items))
        f1_mean = float(np.nanmean(f1_vals))
        f1_std = float(np.nanstd(f1_vals))
        auc_mean = float(np.nanmean(auc_vals))
        auc_std = float(np.nanstd(auc_vals))
        f1_ci95 = float(1.96 * f1_std / max(math.sqrt(n), 1.0))
        auc_ci95 = float(1.96 * auc_std / max(math.sqrt(n), 1.0))
        out.append(
            {
                "variant": var,
                "model": model,
                "global_attn": global_attn,
                "n_seeds": n,
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "f1_ci95": f1_ci95,
                "auc_mean": auc_mean,
                "auc_std": auc_std,
                "auc_ci95": auc_ci95,
                "stability_score": f1_mean - 0.5 * f1_std,
            }
        )
    return out


def write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build F1-first and AUC-first model selection tracks")
    parser.add_argument(
        "--baseline-seed-results",
        default=os.path.join(DEFAULT_RESULTS_ROOT, "baseline_seed_results.csv"),
    )
    parser.add_argument(
        "--gps-seed-results",
        default=os.path.join(DEFAULT_RESULTS_ROOT, "gps_seed_results.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(DEFAULT_RESULTS_ROOT, "reports"),
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    baseline_rows = read_rows(args.baseline_seed_results)
    gps_rows = read_rows(args.gps_seed_results)

    agg = aggregate_seed_rows(baseline_rows, "baseline") + aggregate_seed_rows(gps_rows, "gps")

    f1_overall = sorted(agg, key=lambda r: (r["f1_mean"], r["auc_mean"], -r["f1_std"]), reverse=True)
    auc_overall = sorted(agg, key=lambda r: (r["auc_mean"], r["f1_mean"], -r["auc_std"]), reverse=True)

    by_model: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for row in agg:
        by_model[str(row["model"])].append(row)

    f1_per_model: List[Dict[str, float]] = []
    auc_per_model: List[Dict[str, float]] = []
    for model in sorted(by_model.keys()):
        items = by_model[model]
        best_f1 = max(items, key=lambda r: (r["f1_mean"], r["auc_mean"], -r["f1_std"]))
        best_auc = max(items, key=lambda r: (r["auc_mean"], r["f1_mean"], -r["auc_std"]))
        f1_per_model.append(best_f1)
        auc_per_model.append(best_auc)

    write_csv(os.path.join(args.out_dir, "track_f1_first_overall.csv"), f1_overall)
    write_csv(os.path.join(args.out_dir, "track_auc_first_overall.csv"), auc_overall)
    write_csv(os.path.join(args.out_dir, "track_f1_first_per_model.csv"), f1_per_model)
    write_csv(os.path.join(args.out_dir, "track_auc_first_per_model.csv"), auc_per_model)

    print("[saved]", os.path.join(args.out_dir, "track_f1_first_overall.csv"))
    print("[saved]", os.path.join(args.out_dir, "track_auc_first_overall.csv"))
    print("[saved]", os.path.join(args.out_dir, "track_f1_first_per_model.csv"))
    print("[saved]", os.path.join(args.out_dir, "track_auc_first_per_model.csv"))


if __name__ == "__main__":
    main()
