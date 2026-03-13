#!/usr/bin/env python3
"""
Small-World Edge Perturbation Probe: Long-range evidence probe #3

Hypothesis: Adding random long-range edges should help local GNNs more than
transformers, because local GNNs rely on edge structure for information
propagation while transformers already have global attention.

Experiments:
1. Baseline: Train on original graphs
2. +Shortcuts: Add random long-range edges (small-world style)
3. Compare improvement from shortcuts: baseline vs transformer

If hypothesis is correct:
- Baseline should improve MORE when shortcuts are added
- Transformer should improve LESS (already has global context)
"""
import argparse
import csv
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)

from gnn_transformer_lympho import (
    FoldSplit,
    GPSModel,
    compute_degree_hist,
    compute_feature_stats,
    ensure_dir,
    load_graphs,
    normalize_graphs,
    seed_everything,
    stratified_split,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATA_PATH = os.path.join(WORK_ROOT, "data", "lymphocyte_toy_data.pkl")
DEFAULT_OUT_DIR = os.path.join(WORK_ROOT, "results", "gps_artifacts", "improve_round", "smallworld_probe")


def add_random_shortcuts(graph: Data, num_shortcuts: int, seed: int = 42) -> Data:
    """
    Add random long-range edges (shortcuts) to a graph.
    These connect random pairs of nodes that are not already connected.
    """
    rng = random.Random(seed)
    
    clone = graph.clone()
    edge_index = clone.edge_index.cpu().numpy()
    num_nodes = clone.num_nodes
    
    if num_nodes < 2:
        return clone
    
    # Build existing edge set
    existing_edges = set()
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        existing_edges.add((min(u, v), max(u, v)))
    
    # Add random shortcuts
    new_edges = []
    attempts = 0
    max_attempts = num_shortcuts * 10
    
    while len(new_edges) < num_shortcuts and attempts < max_attempts:
        u = rng.randint(0, num_nodes - 1)
        v = rng.randint(0, num_nodes - 1)
        if u == v:
            attempts += 1
            continue
        edge_key = (min(u, v), max(u, v))
        if edge_key in existing_edges:
            attempts += 1
            continue
        existing_edges.add(edge_key)
        new_edges.append([u, v])
        new_edges.append([v, u])  # Undirected
        attempts += 1
    
    if new_edges:
        new_edges_tensor = torch.tensor(new_edges, dtype=torch.long).T
        clone.edge_index = torch.cat([clone.edge_index, new_edges_tensor], dim=1)
        
        # Update edge_attr if present
        if hasattr(clone, "edge_attr") and clone.edge_attr is not None:
            num_new = new_edges_tensor.shape[1]
            new_attr = torch.ones((num_new, clone.edge_attr.shape[1]), dtype=clone.edge_attr.dtype)
            clone.edge_attr = torch.cat([clone.edge_attr, new_attr], dim=0)
    
    return clone


def add_shortcuts_to_dataset(dataset: List[Data], shortcut_frac: float, seed: int) -> List[Data]:
    """Add shortcuts to all graphs in dataset."""
    perturbed = []
    for i, graph in enumerate(dataset):
        num_shortcuts = max(1, int(graph.num_nodes * shortcut_frac))
        perturbed.append(add_random_shortcuts(graph, num_shortcuts, seed + i))
    return perturbed


def train_and_evaluate(
    dataset: List[Data],
    labels: np.ndarray,
    local_gnn: str,
    global_attn: str,
    seed: int,
    hidden_dim: int = 64,
    gps_layers: int = 2,
    epochs: int = 60,
    patience: int = 15,
    lr: float = 1e-3,
    dropout: float = 0.2,
    batch_size: int = 8,
    folds: int = 3,
) -> Dict[str, float]:
    """Train model and return metrics."""
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    indices = np.arange(len(labels))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    
    f1_scores = []
    auc_scores = []
    
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(indices, labels), start=1):
        train_idx, val_idx = stratified_split(train_val_idx, labels, 0.2, seed + fold)
        
        train_ds = [dataset[i] for i in train_idx]
        val_ds = [dataset[i] for i in val_idx]
        test_ds = [dataset[i] for i in test_idx]
        
        mean, std = compute_feature_stats(train_ds)
        train_ds = normalize_graphs(train_ds, mean, std)
        val_ds = normalize_graphs(val_ds, mean, std)
        test_ds = normalize_graphs(test_ds, mean, std)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        input_dim = train_ds[0].num_node_features
        pna_deg = compute_degree_hist(train_ds) if local_gnn == "pna" else None
        
        model = GPSModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=int(labels.max()) + 1,
            gps_layers=gps_layers,
            local_gnn=local_gnn,
            global_attn=global_attn,
            heads=4,
            dropout=dropout,
            attn_dropout=0.1,
            norm="layer",
            struct_features="none",
            lap_pe_encoder="raw",
            lap_pe_dim=0,
            lap_pe_out=0,
            rwse_encoder="raw",
            rwse_dim=0,
            rwse_out=0,
            graphormer_max_dist=5,
            edge_dim=None,
            pna_deg=pna_deg,
            pool="mean",
            graph_feat_dim=0,
            edge_dropout=0.0,
        ).to(device)
        
        class_counts = np.bincount(labels[train_idx])
        inv = class_counts.sum() / np.clip(class_counts, 1.0, None)
        weights = torch.tensor(inv / inv.mean(), dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        best_state = None
        best_val_f1 = -1.0
        no_improve = 0
        
        for epoch in range(1, epochs + 1):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                logits = model(batch)
                loss = criterion(logits, batch.y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            model.eval()
            val_labels_list = []
            val_preds = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    logits = model(batch)
                    preds = logits.argmax(dim=1)
                    val_labels_list.extend(batch.y.view(-1).cpu().numpy().tolist())
                    val_preds.extend(preds.cpu().numpy().tolist())
            
            val_f1 = f1_score(val_labels_list, val_preds, average="macro", zero_division=0)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                break
        
        if best_state is not None:
            model.load_state_dict(best_state)
        
        model.eval()
        test_labels_list = []
        test_preds = []
        test_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                test_labels_list.extend(batch.y.view(-1).cpu().numpy().tolist())
                test_preds.extend(preds.cpu().numpy().tolist())
                test_probs.extend(probs[:, 1].cpu().numpy().tolist())
        
        f1 = f1_score(test_labels_list, test_preds, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(test_labels_list, test_probs)
        except ValueError:
            auc = float("nan")
        
        f1_scores.append(f1)
        if not np.isnan(auc):
            auc_scores.append(auc)
    
    return {
        "f1_mean": float(np.mean(f1_scores)),
        "f1_std": float(np.std(f1_scores)),
        "auc_mean": float(np.mean(auc_scores)) if auc_scores else float("nan"),
        "auc_std": float(np.std(auc_scores)) if auc_scores else float("nan"),
    }


def run_smallworld_probe(
    local_gnn: str,
    seeds: List[int],
    shortcut_fracs: List[float],
    out_dir: str,
    data_path: str,
) -> None:
    """Run small-world perturbation experiment."""
    ensure_dir(out_dir)
    
    print(f"Loading dataset from {data_path}...")
    original_dataset, labels = load_graphs(data_path)
    print(f"Loaded {len(original_dataset)} graphs")
    
    results: List[Dict[str, Any]] = []
    
    for global_attn in ["none", "performer"]:
        for shortcut_frac in shortcut_fracs:
            print(f"\n{'='*60}")
            print(f"Config: local_gnn={local_gnn}, global_attn={global_attn}, shortcuts={shortcut_frac}")
            print(f"{'='*60}")
            
            # Prepare dataset
            if shortcut_frac == 0:
                dataset = original_dataset
            else:
                dataset = add_shortcuts_to_dataset(original_dataset, shortcut_frac, seed=42)
            
            seed_f1s = []
            seed_aucs = []
            
            for seed in seeds:
                print(f"  Running seed {seed}...")
                result = train_and_evaluate(
                    dataset=dataset,
                    labels=labels,
                    local_gnn=local_gnn,
                    global_attn=global_attn,
                    seed=seed,
                )
                seed_f1s.append(result["f1_mean"])
                if not np.isnan(result["auc_mean"]):
                    seed_aucs.append(result["auc_mean"])
            
            row = {
                "local_gnn": local_gnn,
                "global_attn": global_attn,
                "shortcut_frac": shortcut_frac,
                "f1_mean": float(np.mean(seed_f1s)),
                "f1_std": float(np.std(seed_f1s)),
                "auc_mean": float(np.mean(seed_aucs)) if seed_aucs else float("nan"),
                "auc_std": float(np.std(seed_aucs)) if seed_aucs else float("nan"),
                "num_seeds": len(seeds),
            }
            results.append(row)
            print(f"  F1: {row['f1_mean']:.4f}+-{row['f1_std']:.4f}, "
                  f"AUC: {row['auc_mean']:.4f}+-{row['auc_std']:.4f}")
    
    # Save results
    results_path = os.path.join(out_dir, f"smallworld_probe_{local_gnn}.csv")
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {results_path}")
    
    # Analyze and plot
    analyze_and_plot(results, local_gnn, out_dir)


def analyze_and_plot(results: List[Dict[str, Any]], local_gnn: str, out_dir: str) -> None:
    """Analyze smallworld perturbation results and create plots."""
    
    baseline = [r for r in results if r["global_attn"] == "none"]
    transformer = [r for r in results if r["global_attn"] == "performer"]
    
    baseline = sorted(baseline, key=lambda x: x["shortcut_frac"])
    transformer = sorted(transformer, key=lambda x: x["shortcut_frac"])
    
    fracs = [r["shortcut_frac"] for r in baseline]
    
    # Compute improvements from shortcuts (relative to no shortcuts)
    baseline_no_shortcut = baseline[0]["f1_mean"]
    transformer_no_shortcut = transformer[0]["f1_mean"]
    
    baseline_improvements = [(r["f1_mean"] - baseline_no_shortcut) for r in baseline]
    transformer_improvements = [(r["f1_mean"] - transformer_no_shortcut) for r in transformer]
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Absolute F1
    ax = axes[0]
    ax.errorbar(fracs, [r["f1_mean"] for r in baseline],
                yerr=[r["f1_std"] for r in baseline],
                marker='o', label='Baseline (none)', capsize=3, color='#1f77b4')
    ax.errorbar(fracs, [r["f1_mean"] for r in transformer],
                yerr=[r["f1_std"] for r in transformer],
                marker='s', label='Transformer (performer)', capsize=3, color='#ff7f0e')
    ax.set_xlabel("Shortcut Fraction (edges per node)")
    ax.set_ylabel("F1 Macro")
    ax.set_title(f"{local_gnn.upper()}: F1 vs Shortcut Fraction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: F1 Improvement from Shortcuts
    ax = axes[1]
    ax.plot(fracs, baseline_improvements, marker='o', label='Baseline Improvement', color='#1f77b4')
    ax.plot(fracs, transformer_improvements, marker='s', label='Transformer Improvement', color='#ff7f0e')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Shortcut Fraction (edges per node)")
    ax.set_ylabel("F1 Improvement from Shortcuts")
    ax.set_title(f"{local_gnn.upper()}: Improvement from Adding Shortcuts")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"smallworld_probe_{local_gnn}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_path}")
    
    # Write analysis summary
    summary_path = os.path.join(out_dir, f"smallworld_analysis_{local_gnn}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("SMALL-WORLD PERTURBATION PROBE ANALYSIS\n")
        f.write("="*60 + "\n\n")
        
        f.write("HYPOTHESIS:\n")
        f.write("Adding random shortcuts should help baseline GNNs more than\n")
        f.write("transformers, because baseline GNNs rely on edge structure\n")
        f.write("while transformers already have global attention.\n\n")
        
        f.write("RESULTS:\n")
        f.write(f"  Baseline (no shortcuts): F1 = {baseline_no_shortcut:.4f}\n")
        f.write(f"  Transformer (no shortcuts): F1 = {transformer_no_shortcut:.4f}\n\n")
        
        for frac, b_imp, t_imp in zip(fracs, baseline_improvements, transformer_improvements):
            if frac > 0:
                f.write(f"  At {frac} shortcuts/node:\n")
                f.write(f"    Baseline improvement: {b_imp:+.4f}\n")
                f.write(f"    Transformer improvement: {t_imp:+.4f}\n")
                f.write(f"    Difference (baseline - transformer): {b_imp - t_imp:+.4f}\n\n")
        
        # Final conclusion
        max_frac_idx = -1
        b_imp_max = baseline_improvements[max_frac_idx]
        t_imp_max = transformer_improvements[max_frac_idx]
        
        f.write("CONCLUSION:\n")
        if b_imp_max > t_imp_max:
            f.write("  HYPOTHESIS SUPPORTED: Baseline improves more from shortcuts than transformer.\n")
            f.write(f"  This suggests baseline benefits from structural shortcuts because it lacks\n")
            f.write(f"  the global attention that transformers have built-in.\n")
        else:
            f.write("  HYPOTHESIS NOT SUPPORTED: Transformer improves more (or equally) from shortcuts.\n")
            f.write(f"  This may indicate that shortcuts provide additional useful structure\n")
            f.write(f"  that benefits both architectures equally.\n")
    
    print(f"Analysis saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Small-world perturbation probe")
    parser.add_argument("--local-gnn", default="sage",
                        choices=["sage", "gcn", "gin", "pna", "gine"])
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds")
    parser.add_argument("--shortcut-fracs", default="0,0.1,0.2,0.3",
                        help="Comma-separated shortcut fractions")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    args = parser.parse_args()
    
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    shortcut_fracs = [float(f.strip()) for f in args.shortcut_fracs.split(",")]
    
    run_smallworld_probe(
        local_gnn=args.local_gnn,
        seeds=seeds,
        shortcut_fracs=shortcut_fracs,
        out_dir=args.out_dir,
        data_path=args.data_path,
    )


if __name__ == "__main__":
    main()
