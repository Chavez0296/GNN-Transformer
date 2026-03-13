#!/usr/bin/env python3
"""
Standalone fixed ablation runner for lymphocyte node-level training.
- No argparse
- No imports from gnn_transformer_lympho.py
- Fixed tuned recipe
- Writes all artifacts under gps_work/results/gps_artifacts/

Run:
    python best_gnn_transformer.py
"""

from __future__ import annotations

import csv
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GPSConv, SAGEConv
from torch_geometric.utils import to_undirected

from shared_lympho_utils import class_weights_from_labels, load_lympho_dataset


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

SEED = 42
DATA_PATH = os.path.join(WORK_ROOT, "data", "lymphocyte_toy_data.pkl")
OUT_DIR = os.path.join(WORK_ROOT, "results", "gps_artifacts", "best_gnn_transformer")

FOLDS = 5
VAL_FRAC = 0.2

EPOCHS = 80
PATIENCE = 40
LR = 0.003
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 1.0

NODE_BATCH_SIZE = 2048
NODE_NEIGHBORS = [25, 15]

HIDDEN_DIM = 256
GPS_LAYERS = 2
HEADS = 4
DROPOUT = 0.5
ATTN_TYPE = "multihead"

LABEL_SMOOTHING = 0.05
CLASS_WEIGHT_SMOOTHING = 0.3

TOP_ERROR_PER_FOLD = 50


@dataclass
class FoldSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


class SAGENodeModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.drop = nn.Dropout(DROPOUT)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x = self.conv1(data.x, data.edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.drop(x)
        return self.conv2(x, data.edge_index)


class GPSNodeModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.bn_in = nn.BatchNorm1d(hidden_dim)
        self.drop = nn.Dropout(DROPOUT)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(GPS_LAYERS):
            local_conv = SAGEConv(hidden_dim, hidden_dim)
            self.layers.append(
                GPSConv(
                    hidden_dim,
                    conv=local_conv,
                    heads=HEADS,
                    dropout=DROPOUT,
                    attn_type=ATTN_TYPE,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.lin_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x = self.lin_in(data.x)
        x = self.bn_in(x)
        x = torch.relu(x)
        x = self.drop(x)
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else None
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for i, layer in enumerate(self.layers):
            x = layer(x, data.edge_index, batch=batch)
            if i < len(self.layers) - 1:
                x = self.bns[i](x)
                x = torch.relu(x)
                x = self.drop(x)
        return self.lin_out(x)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_splits(labels: np.ndarray, folds: int, val_frac: float, seed: int) -> List[FoldSplit]:
    indices = np.arange(len(labels))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    val_ratio = val_frac / (1.0 - 1.0 / folds)
    out: List[FoldSplit] = []
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(indices, labels), start=1):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed + fold)
        tr_sub, va_sub = next(sss.split(train_val_idx, labels[train_val_idx]))
        out.append(FoldSplit(train_val_idx[tr_sub], train_val_idx[va_sub], test_idx))
    return out


def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thr = 0.5
    best_score = -1.0
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        score = f1_score(y_true, preds, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr, float(best_score)


def safe_roc_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, probs))
    except ValueError:
        return float("nan")


def safe_pr_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, probs))
    except ValueError:
        return float("nan")


def model_factory(model_name: str, in_dim: int, out_dim: int) -> nn.Module:
    if model_name == "baseline_sage":
        return SAGENodeModel(in_dim, HIDDEN_DIM, out_dim)
    if model_name == "baseline_plus_gps_transformer":
        return GPSNodeModel(in_dim, HIDDEN_DIM, out_dim)
    raise ValueError(model_name)

def compute_degree(edge_index_np: np.ndarray, num_nodes: int) -> np.ndarray:
    deg = np.zeros(num_nodes, dtype=np.int64)
    src, dst = edge_index_np[0], edge_index_np[1]
    np.add.at(deg, src, 1)
    np.add.at(deg, dst, 1)
    return deg


def build_adj_list(edge_index_np: np.ndarray, num_nodes: int) -> List[np.ndarray]:
    src, dst = edge_index_np[0], edge_index_np[1]
    neigh: List[List[int]] = [[] for _ in range(num_nodes)]
    for u, v in zip(src.tolist(), dst.tolist()):
        neigh[u].append(v)
        neigh[v].append(u)
    return [np.asarray(x, dtype=np.int64) for x in neigh]


def compute_purity(adj: List[np.ndarray], labels: np.ndarray) -> np.ndarray:
    """Compute 2-hop label agreement purity per node.

    If 1-hop purity is degenerate on this dataset (all 1.0), 2-hop tends to
    provide more variation for slice analysis.
    """
    n = labels.shape[0]
    purity = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        one_hop = adj[i]
        if one_hop.size == 0:
            continue
        # Collect 1-hop + 2-hop neighborhood (unique), excluding self.
        neighborhood = set(one_hop.tolist())
        for j in one_hop.tolist():
            neighborhood.update(adj[j].tolist())
        neighborhood.discard(i)
        if not neighborhood:
            continue
        neigh_idx = np.fromiter(neighborhood, dtype=np.int64)
        purity[i] = float(np.mean(labels[neigh_idx] == labels[i]))
    return purity


def compute_structural_purity(adj: List[np.ndarray]) -> np.ndarray:
    """Fallback purity proxy when label-agreement purity is degenerate.

    For each node, compute mean Jaccard overlap between its 1-hop neighbors and
    each neighbor's 1-hop neighbors. Higher means more structurally coherent
    local neighborhoods.
    """
    n = len(adj)
    out = np.full(n, np.nan, dtype=np.float32)
    neigh_sets = [set(a.tolist()) for a in adj]
    for i in range(n):
        nbs = adj[i]
        if nbs.size == 0:
            continue
        si = neigh_sets[i]
        vals = []
        for j in nbs.tolist():
            sj = neigh_sets[j]
            inter = len(si & sj)
            union = len(si | sj)
            if union > 0:
                vals.append(inter / union)
        if vals:
            out[i] = float(np.mean(vals))
    return out


def quantile_edges(values: np.ndarray) -> Tuple[float, float]:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return 0.0, 1.0
    q1 = float(np.quantile(valid, 0.33))
    q2 = float(np.quantile(valid, 0.66))
    if q1 == q2:
        q2 = q1 + 1e-9
    return q1, q2


def bucket(v: float, q1: float, q2: float) -> str:
    if not np.isfinite(v):
        return "unknown"
    if v <= q1:
        return "low"
    if v <= q2:
        return "mid"
    return "high"


def save_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def save_history_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_learning_curves(history_paths: List[str], out_path: str) -> None:
    if not history_paths:
        return
    train_loss_by_epoch: Dict[int, List[float]] = {}
    val_f1_by_epoch: Dict[int, List[float]] = {}
    val_pr_by_epoch: Dict[int, List[float]] = {}
    for p in history_paths:
        with open(p, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                ep = int(row["epoch"])
                train_loss_by_epoch.setdefault(ep, []).append(float(row["train_loss"]))
                val_f1_by_epoch.setdefault(ep, []).append(float(row["val_f1"]))
                val_pr_by_epoch.setdefault(ep, []).append(float(row["val_pr_auc"]))
    epochs = sorted(train_loss_by_epoch.keys())
    if not epochs:
        return
    mean_train_loss = [float(np.mean(train_loss_by_epoch[e])) for e in epochs]
    mean_val_f1 = [float(np.mean(val_f1_by_epoch[e])) for e in epochs]
    mean_val_pr = [float(np.mean(val_pr_by_epoch[e])) for e in epochs]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(epochs, mean_train_loss, color="#4C78A8")
    axes[1].plot(epochs, mean_val_f1, color="#F58518")
    axes[2].plot(epochs, mean_val_pr, color="#54A24B")
    axes[0].set_ylabel("Train Loss")
    axes[1].set_ylabel("Val F1")
    axes[2].set_ylabel("Val PR-AUC")
    axes[2].set_xlabel("Epoch")
    axes[0].set_title("Learning Curves")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_roc_auc_curves(history_paths: List[str], out_path: str) -> None:
    if not history_paths:
        return
    roc_by_epoch: Dict[int, List[float]] = {}
    for p in history_paths:
        with open(p, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                v = float(row["val_roc_auc"])
                if np.isnan(v):
                    continue
                ep = int(row["epoch"])
                roc_by_epoch.setdefault(ep, []).append(v)
    epochs = sorted(roc_by_epoch.keys())
    if not epochs:
        return
    mean_roc = [float(np.mean(roc_by_epoch[e])) for e in epochs]
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(epochs, mean_roc, color="#54A24B")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val ROC-AUC")
    ax.set_title("Validation ROC-AUC Over Epochs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def train_fold(model_name: str, fold_id: int, data: Data, labels: np.ndarray, split: FoldSplit, device: str, history_path: str) -> Dict[str, object]:
    seed_everything(SEED + fold_id)
    model = model_factory(model_name, int(data.num_node_features), int(labels.max()) + 1).to(device)
    weights = class_weights_from_labels(labels[split.train_idx], smoothing=CLASS_WEIGHT_SMOOTHING).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_loader = NeighborLoader(
        data,
        num_neighbors=NODE_NEIGHBORS,
        input_nodes=torch.as_tensor(split.train_idx, dtype=torch.long),
        batch_size=NODE_BATCH_SIZE,
        shuffle=True,
    )

    full_data = data.to(device)
    val_idx = torch.as_tensor(split.val_idx, dtype=torch.long, device=device)
    test_idx = torch.as_tensor(split.test_idx, dtype=torch.long, device=device)

    best_state = None
    best_val_pr_auc = -1.0
    no_improve = 0
    history_rows: List[Dict[str, float]] = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        total_samples = 0
        train_preds, train_true = [], []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)[: batch.batch_size]
            yb = batch.y[: batch.batch_size]
            loss = criterion(logits, yb)
            loss.backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            bsz = int(batch.batch_size)
            total_loss += float(loss.item()) * bsz
            total_samples += bsz
            train_preds.append(logits.argmax(1).detach().cpu().numpy())
            train_true.append(yb.cpu().numpy())

        train_loss = total_loss / max(1, total_samples)
        if train_preds:
            yp_tr = np.concatenate(train_preds)
            yt_tr = np.concatenate(train_true)
            tr_acc = float(accuracy_score(yt_tr, yp_tr))
            tr_f1 = float(f1_score(yt_tr, yp_tr, average="macro", zero_division=0))
        else:
            tr_acc = 0.0
            tr_f1 = 0.0

        model.eval()
        with torch.no_grad():
            logits_full = model(full_data)
            val_logits = logits_full[val_idx]
            val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
            val_preds = val_logits.argmax(1).cpu().numpy()
            val_true = labels[split.val_idx]

        val_acc = float(accuracy_score(val_true, val_preds))
        val_f1 = float(f1_score(val_true, val_preds, average="macro", zero_division=0))
        val_roc_auc = safe_roc_auc(val_true, val_probs)
        val_pr_auc = safe_pr_auc(val_true, val_probs)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": tr_acc,
                "train_f1": tr_f1,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_roc_auc": val_roc_auc,
                "val_pr_auc": val_pr_auc,
                "epoch_sec": time.time() - t0,
            }
        )

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"[{model_name}][fold{fold_id}] epoch {epoch:03d} val_pr_auc={val_pr_auc:.4f}")

        if no_improve >= PATIENCE:
            break

    save_history_csv(history_path, history_rows)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        logits_full = model(full_data)
        val_logits = logits_full[val_idx]
        val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
        val_true = labels[split.val_idx]

    best_thr, best_thr_f1 = find_best_threshold(val_true, val_probs)

    with torch.no_grad():
        test_logits = logits_full[test_idx]
        test_probs = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
    test_true = labels[split.test_idx]
    test_preds = (test_probs >= best_thr).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(test_true, test_preds)),
        "f1_macro": float(f1_score(test_true, test_preds, average="macro", zero_division=0)),
        "val_thr": float(best_thr),
        "val_thr_f1": float(best_thr_f1),
        "roc_auc": safe_roc_auc(test_true, test_probs),
        "pr_auc": safe_pr_auc(test_true, test_probs),
    }

    return {
        "metrics": metrics,
        "test_node_ids": split.test_idx,
        "test_true": test_true,
        "test_pred": test_preds,
        "test_prob": test_probs,
    }


def compute_slice_rows(model_name: str, fold_id: int, test_node_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, degree: np.ndarray, purity: np.ndarray, deg_q1: float, deg_q2: float, pur_q1: float, pur_q2: float) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    deg_bucket = np.array([bucket(float(degree[n]), deg_q1, deg_q2) for n in test_node_ids], dtype=object)
    pur_bucket = np.array([bucket(float(purity[n]), pur_q1, pur_q2) for n in test_node_ids], dtype=object)

    for slice_type, buckets in (("degree", deg_bucket), ("purity", pur_bucket)):
        for b in ("low", "mid", "high", "unknown"):
            m = buckets == b
            if int(m.sum()) == 0:
                continue
            yt, yp, ys = y_true[m], y_pred[m], y_prob[m]
            rows.append(
                {
                    "model": model_name,
                    "fold": fold_id,
                    "slice_type": slice_type,
                    "bucket": b,
                    "support": int(m.sum()),
                    "accuracy": float(accuracy_score(yt, yp)),
                    "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
                    "roc_auc": safe_roc_auc(yt, ys),
                    "pr_auc": safe_pr_auc(yt, ys),
                }
            )
    return rows


def compute_error_rows(model_name: str, fold_id: int, test_node_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, degree: np.ndarray, purity: np.ndarray, top_k: int) -> List[Dict[str, object]]:
    wrong = y_true != y_pred
    if int(wrong.sum()) == 0:
        return []
    node_wrong = test_node_ids[wrong]
    true_wrong = y_true[wrong]
    pred_wrong = y_pred[wrong]
    prob_wrong = y_prob[wrong]
    conf_wrong = np.where(pred_wrong == 1, prob_wrong, 1.0 - prob_wrong)

    order = np.argsort(-conf_wrong)[: min(top_k, conf_wrong.shape[0])]
    out: List[Dict[str, object]] = []
    for i in order:
        nid = int(node_wrong[i])
        out.append(
            {
                "model": model_name,
                "fold": fold_id,
                "node_id": nid,
                "true_label": int(true_wrong[i]),
                "pred_label": int(pred_wrong[i]),
                "prob_pos": float(prob_wrong[i]),
                "confidence": float(conf_wrong[i]),
                "degree": int(degree[nid]),
                "purity": float(purity[nid]) if np.isfinite(purity[nid]) else float("nan"),
            }
        )
    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    seed_everything(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, (torch.cuda.get_device_name(0) if device == "cuda" else "CPU"))

    x_all, y_all, edge_index = load_lympho_dataset(DATA_PATH)
    edge_tensor = to_undirected(torch.tensor(edge_index, dtype=torch.long))
    edge_np = edge_tensor.cpu().numpy()

    data = Data(
        x=torch.tensor(x_all, dtype=torch.float32),
        y=torch.tensor(y_all, dtype=torch.long),
        edge_index=edge_tensor,
    )

    degree = compute_degree(edge_np, y_all.shape[0])
    adj = build_adj_list(edge_np, y_all.shape[0])
    purity = compute_purity(adj, y_all)
    valid_p = purity[np.isfinite(purity)]
    if valid_p.size > 0 and np.allclose(valid_p, valid_p[0]):
        print("[warn] Label-agreement purity is degenerate; using structural purity proxy for slices.")
        purity = compute_structural_purity(adj)

    splits = build_splits(y_all, FOLDS, VAL_FRAC, SEED)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_names = ["baseline_sage", "baseline_plus_gps_transformer"]

    all_summary_rows: List[Dict[str, object]] = []
    all_per_fold_rows: List[Dict[str, object]] = []
    all_slice_rows: List[Dict[str, object]] = []
    all_error_rows: List[Dict[str, object]] = []

    for model_name in model_names:
        print(f"\n===== Running {model_name} =====")
        metrics_list: List[Dict[str, float]] = []
        history_paths: List[str] = []

        for fold_id, split in enumerate(splits, start=1):
            history_path = os.path.join(OUT_DIR, f"history_{run_id}_{model_name}_fold{fold_id}.csv")
            history_paths.append(history_path)

            result = train_fold(model_name, fold_id, data, y_all, split, device, history_path)
            metrics = result["metrics"]
            metrics_list.append(metrics)

            per_fold_row = {"model": model_name, "fold": fold_id}
            per_fold_row.update(metrics)
            all_per_fold_rows.append(per_fold_row)

            deg_q1, deg_q2 = quantile_edges(degree[split.train_idx].astype(np.float32))
            pur_q1, pur_q2 = quantile_edges(purity[split.train_idx].astype(np.float32))

            all_slice_rows.extend(
                compute_slice_rows(
                    model_name,
                    fold_id,
                    result["test_node_ids"],
                    result["test_true"],
                    result["test_pred"],
                    result["test_prob"],
                    degree,
                    purity,
                    deg_q1,
                    deg_q2,
                    pur_q1,
                    pur_q2,
                )
            )

            all_error_rows.extend(
                compute_error_rows(
                    model_name,
                    fold_id,
                    result["test_node_ids"],
                    result["test_true"],
                    result["test_pred"],
                    result["test_prob"],
                    degree,
                    purity,
                    TOP_ERROR_PER_FOLD,
                )
            )

            print(f"[{model_name}][fold{fold_id}] metrics: {metrics}")

        keys = sorted({k for m in metrics_list for k in m})
        summary = {"model": model_name}
        for k in keys:
            vals = [m[k] for m in metrics_list if k in m]
            summary[f"{k}_mean"] = float(np.nanmean(vals))
            summary[f"{k}_std"] = float(np.nanstd(vals))
        all_summary_rows.append(summary)

        plot_learning_curves(history_paths, os.path.join(OUT_DIR, f"learning_curves_{run_id}_{model_name}.png"))
        plot_roc_auc_curves(history_paths, os.path.join(OUT_DIR, f"roc_auc_curves_{run_id}_{model_name}.png"))

    save_csv(os.path.join(OUT_DIR, f"ablation_summary_{run_id}.csv"), all_summary_rows)
    save_csv(os.path.join(OUT_DIR, f"ablation_per_fold_{run_id}.csv"), all_per_fold_rows)
    save_csv(os.path.join(OUT_DIR, f"slice_metrics_{run_id}.csv"), all_slice_rows)
    save_csv(os.path.join(OUT_DIR, f"error_audit_{run_id}.csv"), all_error_rows)

    print("\n===== Ablation Summary =====")
    for row in all_summary_rows:
        print(row)


if __name__ == "__main__":
    main()
