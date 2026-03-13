#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS_ROOT = os.path.join(WORK_ROOT, "results", "gps_artifacts")


def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: str, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def aggregate_mean_std(rows: Iterable[Dict[str, str]], f1_key: str, auc_key: str) -> Tuple[float, float, float, float, int]:
    f1_vals = np.asarray([safe_float(r.get(f1_key, "nan")) for r in rows], dtype=float)
    auc_vals = np.asarray([safe_float(r.get(auc_key, "nan")) for r in rows], dtype=float)
    n = int(f1_vals.shape[0])
    return (
        float(np.nanmean(f1_vals)),
        float(np.nanstd(f1_vals)),
        float(np.nanmean(auc_vals)),
        float(np.nanstd(auc_vals)),
        n,
    )


def baseline_and_initial_gps_candidates(path: str) -> List[Dict[str, object]]:
    rows = read_rows(path)
    out: List[Dict[str, object]] = []
    for row in rows:
        variant = row["variant"].strip()
        source = "baseline_seed_stability" if variant == "baseline" else "initial_gps_seed_stability"
        out.append(
            {
                "candidate_id": f"{variant}_{row['model']}_seed10",
                "model": row["model"],
                "global_attn": "none" if variant == "baseline" else "as-run",
                "variant": variant,
                "source": source,
                "status": "accepted_reference",
                "n_seeds": int(float(row["n_seeds"])),
                "f1_mean": safe_float(row["f1_mean"]),
                "f1_std": safe_float(row["f1_std"]),
                "auc_mean": safe_float(row["auc_mean"]),
                "auc_std": safe_float(row["auc_std"]),
                "stability_score": safe_float(row.get("stability_score", "nan")),
                "notes": "Final selection seed protocol reference.",
            }
        )
    return out


def s2_gatedgraph_candidates(path: str) -> List[Dict[str, object]]:
    rows = read_rows(path)
    by_attn: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_attn[row["global_attn"]].append(row)

    out: List[Dict[str, object]] = []
    for attn, items in sorted(by_attn.items()):
        f1_mean, f1_std, auc_mean, auc_std, n = aggregate_mean_std(items, "f1_macro_mean", "roc_auc_mean")
        status = "accepted" if attn == "performer" else "accepted_reference"
        notes = "Lower-LR/dropout A/B run; performer improves AUC while F1 remains slightly below baseline-none."
        out.append(
            {
                "candidate_id": f"s2_gatedgraph_{attn}_seed{n}",
                "model": "gatedgraph",
                "global_attn": attn,
                "variant": "gps" if attn != "none" else "baseline_like",
                "source": "s2_gatedgraph_ab",
                "status": status,
                "n_seeds": n,
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "auc_mean": auc_mean,
                "auc_std": auc_std,
                "stability_score": f1_mean - 0.5 * f1_std,
                "notes": notes,
            }
        )
    return out


def s3_confirm_candidates(path: str) -> List[Dict[str, object]]:
    rows = [r for r in read_rows(path) if r.get("phase") == "confirm"]
    by_model: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    out: List[Dict[str, object]] = []
    for model, items in sorted(by_model.items()):
        f1_mean, f1_std, auc_mean, auc_std, n = aggregate_mean_std(items, "f1_macro_mean", "roc_auc_mean")
        global_attn = items[0].get("global_attn", "as-run")
        ad = items[0].get("attn_dropout", "")
        wd = items[0].get("weight_decay", "")
        out.append(
            {
                "candidate_id": f"s3_{model}_{global_attn}_seed{n}",
                "model": model,
                "global_attn": global_attn,
                "variant": "gps",
                "source": "s3_auc_tune_confirm",
                "status": "accepted",
                "n_seeds": n,
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "auc_mean": auc_mean,
                "auc_std": auc_std,
                "stability_score": f1_mean - 0.5 * f1_std,
                "notes": f"AUC-focused tuning confirm; attn_dropout={ad}, weight_decay={wd}.",
            }
        )
    return out


def s5_confirm_candidates(path: str) -> List[Dict[str, object]]:
    rows = read_rows(path)
    by_model: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    out: List[Dict[str, object]] = []
    for model, items in sorted(by_model.items()):
        f1_mean, f1_std, auc_mean, auc_std, n = aggregate_mean_std(items, "f1_macro_mean", "roc_auc_mean")
        global_attn = items[0].get("global_attn", "as-run")
        out.append(
            {
                "candidate_id": f"s5_{model}_{global_attn}_seed{n}",
                "model": model,
                "global_attn": global_attn,
                "variant": "gps",
                "source": "s5_pna_gine_confirm",
                "status": "accepted",
                "n_seeds": n,
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "auc_mean": auc_mean,
                "auc_std": auc_std,
                "stability_score": f1_mean - 0.5 * f1_std,
                "notes": "Post-improvement confirmation run.",
            }
        )
    return out


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def best_by_model(candidates: List[Dict[str, object]], metric: str) -> List[Dict[str, object]]:
    by_model: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in candidates:
        if str(row["status"]).startswith("rejected"):
            continue
        by_model[str(row["model"])].append(row)

    out: List[Dict[str, object]] = []
    for model, rows in sorted(by_model.items()):
        if metric == "f1":
            best = max(rows, key=lambda r: (float(r["f1_mean"]), float(r["auc_mean"]), -float(r["f1_std"])))
        else:
            best = max(rows, key=lambda r: (float(r["auc_mean"]), float(r["f1_mean"]), -float(r["auc_std"])))
        out.append(best)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile final recommendation tracks from all completed rounds")
    parser.add_argument(
        "--seed-stability",
        default=os.path.join(RESULTS_ROOT, "final_sel", "reports", "seed_stability_summary.csv"),
    )
    parser.add_argument(
        "--s2-ab",
        default=os.path.join(RESULTS_ROOT, "improve_round", "s2_gatedgraph_ab", "ab_results.csv"),
    )
    parser.add_argument(
        "--s3-auc-tune",
        default=os.path.join(RESULTS_ROOT, "improve_round", "s3_auc_tune", "auc_tune_results.csv"),
    )
    parser.add_argument(
        "--s5-confirm",
        default=os.path.join(RESULTS_ROOT, "improve_round", "s5_pna_gine_confirm", "pna_gine_confirm.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(RESULTS_ROOT, "improve_round", "reports"),
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    candidates = []
    candidates.extend(baseline_and_initial_gps_candidates(args.seed_stability))
    candidates.extend(s2_gatedgraph_candidates(args.s2_ab))
    candidates.extend(s3_confirm_candidates(args.s3_auc_tune))
    candidates.extend(s5_confirm_candidates(args.s5_confirm))

    candidates = sorted(
        candidates,
        key=lambda r: (str(r["model"]), -float(r["f1_mean"]), -float(r["auc_mean"]), float(r["f1_std"])),
    )
    write_csv(os.path.join(args.out_dir, "all_candidates_compiled.csv"), candidates)

    f1_track = sorted(
        best_by_model(candidates, metric="f1"),
        key=lambda r: (float(r["f1_mean"]), float(r["auc_mean"]), -float(r["f1_std"])),
        reverse=True,
    )
    auc_track = sorted(
        best_by_model(candidates, metric="auc"),
        key=lambda r: (float(r["auc_mean"]), float(r["f1_mean"]), -float(r["auc_std"])),
        reverse=True,
    )

    write_csv(os.path.join(args.out_dir, "final_track_f1_first.csv"), f1_track)
    write_csv(os.path.join(args.out_dir, "final_track_auc_first.csv"), auc_track)

    print("[saved]", os.path.join(args.out_dir, "all_candidates_compiled.csv"))
    print("[saved]", os.path.join(args.out_dir, "final_track_f1_first.csv"))
    print("[saved]", os.path.join(args.out_dir, "final_track_auc_first.csv"))


if __name__ == "__main__":
    main()
