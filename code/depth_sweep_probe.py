#!/usr/bin/env python3
"""
Depth Sweep Probe: Long-range evidence probe #1

Tests the hypothesis that transformer attention helps mitigate oversmoothing
at deeper layers by comparing baseline GNN vs GPS-transformer across depths.

Metrics collected:
- F1 macro, ROC-AUC at each depth
- Embedding variance (proxy for oversmoothing) at each depth

Usage:
    python depth_sweep_probe.py --local-gnn sage --seeds 42,123,456
"""
import argparse
import csv
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)

from gnn_transformer_lympho import (
    FoldSplit,
    GPSModel,
    compute_degree_hist,
    compute_feature_stats,
    compute_metrics,
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
DEFAULT_OUT_DIR = os.path.join(WORK_ROOT, "results", "gps_artifacts", "improve_round", "depth_sweep")


class GPSModelWithEmbeddingCapture(nn.Module):
    """
    Wrapper around GPSModel that captures per-layer embeddings for variance analysis.
    """
    def __init__(self, base_model: GPSModel):
        super().__init__()
        self.base_model = base_model
        self.layer_embeddings: List[torch.Tensor] = []

    def forward(self, data: Data) -> torch.Tensor:
        self.layer_embeddings = []
        
        # Replicate forward pass with embedding capture
        if data.x is None or data.edge_index is None:
            raise ValueError("data.x and data.edge_index are required.")
        x = data.x
        feats = [x]

        if self.base_model.lap_pe_encoder is not None and hasattr(data, "lap_pe"):
            lap_pe = data.lap_pe.to(x.device)
            feats.append(self.base_model.lap_pe_encoder(lap_pe))
        if self.base_model.rwse_encoder is not None and hasattr(data, "rwse"):
            rwse = data.rwse.to(x.device)
            feats.append(self.base_model.rwse_encoder(rwse))
        
        from gnn_transformer_lympho import build_struct_features, apply_edge_dropout
        struct = build_struct_features(data.edge_index, x.size(0), self.base_model.struct_features)
        if struct is not None:
            feats.append(struct.to(x.device))
        x = torch.cat(feats, dim=1)
        x = self.base_model.node_encoder(x)

        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
        if edge_attr is None and self.base_model.needs_edge_attr:
            edge_attr = torch.ones((data.edge_index.size(1), 1), device=x.device)
        edge_index = data.edge_index
        edge_index, edge_attr = apply_edge_dropout(edge_index, edge_attr, self.base_model.edge_dropout, self.training)
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else None
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Capture embeddings at each layer
        self.layer_embeddings.append(x.detach().clone())
        
        if self.base_model.global_attn == "none":
            for conv in self.base_model.layers:
                if self.base_model.pass_edge_attr and edge_attr is not None:
                    x = conv(x, edge_index, edge_attr)
                else:
                    x = conv(x, edge_index)
                x = F.relu(x)
                self.layer_embeddings.append(x.detach().clone())
            x = self.base_model.norm(x)
        elif self.base_model.global_attn == "graphormer":
            for layer in self.base_model.layers:
                x = layer(x, edge_index, batch, edge_attr)
                self.layer_embeddings.append(x.detach().clone())
        else:
            for layer in self.base_model.layers:
                if self.base_model.pass_edge_attr and edge_attr is not None:
                    x = layer(x, edge_index, batch=batch, edge_attr=edge_attr)
                else:
                    x = layer(x, edge_index, batch=batch)
                self.layer_embeddings.append(x.detach().clone())

        if self.base_model.pool == "sum":
            from torch_geometric.nn import global_add_pool
            graph_emb = global_add_pool(x, batch)
        elif self.base_model.pool == "attention":
            graph_emb = self.base_model.pooler(x, batch)
        else:
            graph_emb = global_mean_pool(x, batch)
        graph_emb = self.base_model._append_graph_features(graph_emb, data)
        return self.base_model.head(graph_emb)


def compute_embedding_variance(embeddings: List[torch.Tensor]) -> List[float]:
    """
    Compute per-layer variance of node embeddings (across feature dimension).
    Lower variance at deeper layers indicates oversmoothing.
    """
    variances = []
    for emb in embeddings:
        # Compute variance across feature dimension, then average across nodes
        var = emb.var(dim=1).mean().item()
        variances.append(var)
    return variances


def train_and_evaluate(
    dataset: List[Data],
    labels: np.ndarray,
    local_gnn: str,
    global_attn: str,
    gps_layers: int,
    seed: int,
    hidden_dim: int = 64,
    epochs: int = 60,
    patience: int = 15,
    lr: float = 1e-3,
    dropout: float = 0.2,
    batch_size: int = 8,
    folds: int = 3,
) -> Dict[str, Any]:
    """
    Train model and collect metrics + embedding variances.
    """
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    indices = np.arange(len(labels))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    
    all_metrics: List[Dict[str, float]] = []
    all_variances: List[List[float]] = []
    
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(indices, labels), start=1):
        # Split train_val into train and val
        train_idx, val_idx = stratified_split(train_val_idx, labels, 0.2, seed + fold)
        
        train_ds = [dataset[i] for i in train_idx]
        val_ds = [dataset[i] for i in val_idx]
        test_ds = [dataset[i] for i in test_idx]
        
        # Normalize features
        mean, std = compute_feature_stats(train_ds)
        train_ds = normalize_graphs(train_ds, mean, std)
        val_ds = normalize_graphs(val_ds, mean, std)
        test_ds = normalize_graphs(test_ds, mean, std)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        input_dim = train_ds[0].num_node_features
        pna_deg = compute_degree_hist(train_ds) if local_gnn == "pna" else None
        
        base_model = GPSModel(
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
        
        model = GPSModelWithEmbeddingCapture(base_model)
        
        # Class weights
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
            
            # Validate
            model.eval()
            val_labels = []
            val_preds = []
            val_probs = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    logits = model(batch)
                    probs = torch.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1)
                    val_labels.extend(batch.y.view(-1).cpu().numpy().tolist())
                    val_preds.extend(preds.cpu().numpy().tolist())
                    val_probs.extend(probs[:, 1].cpu().numpy().tolist())
            
            val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                break
        
        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)
        
        # Evaluate on test set and collect embedding variances
        model.eval()
        test_labels = []
        test_preds = []
        test_probs = []
        batch_variances: List[List[float]] = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                test_labels.extend(batch.y.view(-1).cpu().numpy().tolist())
                test_preds.extend(preds.cpu().numpy().tolist())
                test_probs.extend(probs[:, 1].cpu().numpy().tolist())
                
                # Collect embedding variances
                batch_variances.append(compute_embedding_variance(model.layer_embeddings))
        
        test_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
        try:
            test_auc = roc_auc_score(test_labels, test_probs)
        except ValueError:
            test_auc = float("nan")
        
        # Average variances across batches
        avg_variances = np.mean(batch_variances, axis=0).tolist()
        
        all_metrics.append({"f1_macro": test_f1, "roc_auc": test_auc})
        all_variances.append(avg_variances)
    
    # Aggregate across folds
    f1_vals = [m["f1_macro"] for m in all_metrics]
    auc_vals = [m["roc_auc"] for m in all_metrics if not np.isnan(m["roc_auc"])]
    
    # Average variances across folds (need to handle different lengths)
    max_layers = max(len(v) for v in all_variances)
    padded_variances = []
    for v in all_variances:
        padded = v + [v[-1]] * (max_layers - len(v))
        padded_variances.append(padded)
    avg_variances = np.mean(padded_variances, axis=0).tolist()
    
    return {
        "f1_mean": float(np.mean(f1_vals)),
        "f1_std": float(np.std(f1_vals)),
        "auc_mean": float(np.mean(auc_vals)) if auc_vals else float("nan"),
        "auc_std": float(np.std(auc_vals)) if auc_vals else float("nan"),
        "layer_variances": avg_variances,
        "final_layer_variance": avg_variances[-1] if avg_variances else float("nan"),
    }


def run_depth_sweep(
    local_gnn: str,
    seeds: List[int],
    depths: List[int],
    out_dir: str,
    data_path: str,
) -> None:
    """
    Run depth sweep for baseline and transformer configurations.
    """
    ensure_dir(out_dir)
    
    print(f"Loading dataset from {data_path}...")
    dataset, labels = load_graphs(data_path)
    print(f"Loaded {len(dataset)} graphs")
    
    results: List[Dict[str, Any]] = []
    
    for global_attn in ["none", "performer"]:
        for depth in depths:
            print(f"\n{'='*60}")
            print(f"Config: local_gnn={local_gnn}, global_attn={global_attn}, depth={depth}")
            print(f"{'='*60}")
            
            seed_results = []
            for seed in seeds:
                print(f"  Running seed {seed}...")
                result = train_and_evaluate(
                    dataset=dataset,
                    labels=labels,
                    local_gnn=local_gnn,
                    global_attn=global_attn,
                    gps_layers=depth,
                    seed=seed,
                )
                seed_results.append(result)
            
            # Aggregate across seeds
            f1_means = [r["f1_mean"] for r in seed_results]
            auc_means = [r["auc_mean"] for r in seed_results if not np.isnan(r["auc_mean"])]
            final_vars = [r["final_layer_variance"] for r in seed_results]
            
            # Get all layer variances (from first seed as representative)
            layer_variances = seed_results[0]["layer_variances"]
            
            row = {
                "local_gnn": local_gnn,
                "global_attn": global_attn,
                "depth": depth,
                "f1_mean": float(np.mean(f1_means)),
                "f1_std": float(np.std(f1_means)),
                "auc_mean": float(np.mean(auc_means)) if auc_means else float("nan"),
                "auc_std": float(np.std(auc_means)) if auc_means else float("nan"),
                "final_layer_variance": float(np.mean(final_vars)),
                "num_seeds": len(seeds),
            }
            
            # Add per-layer variances
            for i, var in enumerate(layer_variances):
                row[f"layer_{i}_variance"] = var
            
            results.append(row)
            print(f"  F1: {row['f1_mean']:.4f}+-{row['f1_std']:.4f}, "
                  f"AUC: {row['auc_mean']:.4f}+-{row['auc_std']:.4f}, "
                  f"Final Var: {row['final_layer_variance']:.4f}")
    
    # Save results
    results_path = os.path.join(out_dir, f"depth_sweep_{local_gnn}.csv")
    fieldnames = ["local_gnn", "global_attn", "depth", "f1_mean", "f1_std", 
                  "auc_mean", "auc_std", "final_layer_variance", "num_seeds"]
    # Add layer variance columns
    max_layers = max(len(r.get("layer_variances", []) if "layer_variances" in seed_results[0] else []) 
                     for r in results for seed_results in [[r]])
    for i in range(max_layers + 2):  # +2 for input + layers
        fieldnames.append(f"layer_{i}_variance")
    
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[fn for fn in fieldnames if any(fn in r for r in results)], 
                               extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {results_path}")
    
    # Generate plots
    plot_depth_sweep(results, local_gnn, out_dir)


def plot_depth_sweep(results: List[Dict[str, Any]], local_gnn: str, out_dir: str) -> None:
    """Generate depth sweep visualization plots."""
    
    baseline = [r for r in results if r["global_attn"] == "none"]
    transformer = [r for r in results if r["global_attn"] == "performer"]
    
    baseline = sorted(baseline, key=lambda x: x["depth"])
    transformer = sorted(transformer, key=lambda x: x["depth"])
    
    depths_b = [r["depth"] for r in baseline]
    depths_t = [r["depth"] for r in transformer]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Plot 1: F1 vs Depth
    ax = axes[0]
    ax.errorbar(depths_b, [r["f1_mean"] for r in baseline], 
                yerr=[r["f1_std"] for r in baseline],
                marker='o', label='Baseline (none)', capsize=3, color='#1f77b4')
    ax.errorbar(depths_t, [r["f1_mean"] for r in transformer],
                yerr=[r["f1_std"] for r in transformer],
                marker='s', label='Transformer (performer)', capsize=3, color='#ff7f0e')
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("F1 Macro")
    ax.set_title(f"{local_gnn.upper()}: F1 vs Depth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: AUC vs Depth
    ax = axes[1]
    ax.errorbar(depths_b, [r["auc_mean"] for r in baseline],
                yerr=[r["auc_std"] for r in baseline],
                marker='o', label='Baseline (none)', capsize=3, color='#1f77b4')
    ax.errorbar(depths_t, [r["auc_mean"] for r in transformer],
                yerr=[r["auc_std"] for r in transformer],
                marker='s', label='Transformer (performer)', capsize=3, color='#ff7f0e')
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"{local_gnn.upper()}: ROC-AUC vs Depth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Embedding Variance vs Depth (oversmoothing indicator)
    ax = axes[2]
    ax.plot(depths_b, [r["final_layer_variance"] for r in baseline],
            marker='o', label='Baseline (none)', color='#1f77b4')
    ax.plot(depths_t, [r["final_layer_variance"] for r in transformer],
            marker='s', label='Transformer (performer)', color='#ff7f0e')
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("Final Layer Embedding Variance")
    ax.set_title(f"{local_gnn.upper()}: Oversmoothing Indicator")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"depth_sweep_{local_gnn}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Depth sweep probe for long-range evidence")
    parser.add_argument("--local-gnn", default="sage", 
                        choices=["sage", "gcn", "gin", "pna", "gine"])
    parser.add_argument("--seeds", default="42,123,456", help="Comma-separated seeds")
    parser.add_argument("--depths", default="1,2,3,4", help="Comma-separated depths to test")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    args = parser.parse_args()
    
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    depths = [int(d.strip()) for d in args.depths.split(",")]
    
    run_depth_sweep(
        local_gnn=args.local_gnn,
        seeds=seeds,
        depths=depths,
        out_dir=args.out_dir,
        data_path=args.data_path,
    )


if __name__ == "__main__":
    main()
