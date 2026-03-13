#!/usr/bin/env python3
import argparse
import csv
import math
import os
import pickle
import random
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    GINEConv,
    GPSConv,
    PNAConv,
    SAGEConv,
    TransformerConv,
    global_add_pool,
    global_mean_pool,
)
from torch_geometric.utils import degree, dropout_edge, subgraph, to_dense_batch, to_undirected

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared_lympho_utils import class_weights_from_labels, load_lympho_dataset


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_DATA_PATH = os.path.join(WORK_ROOT, "data", "lymphocyte_toy_data.pkl")
DEFAULT_OUT_DIR = os.path.join(WORK_ROOT, "results", "gps_artifacts")

try:
    from torch_geometric.nn.aggr import AttentionalAggregation
except ImportError:
    from torch_geometric.nn import GlobalAttention as AttentionalAggregation

try:
    from torch_geometric.nn import GatedGraphConv
except ImportError:
    GatedGraphConv = None

try:
    from torch_geometric.transforms import VirtualNode
except Exception:
    VirtualNode = None


@dataclass
class FoldSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_path_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value).strip().lower())
    token = token.strip("_")
    return token or "run"


def make_run_output_dir(base_out_dir: str, local_gnn: str, global_attn: str, history_tag: str = "") -> Tuple[str, str]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    gnn_dir = os.path.join(base_out_dir, sanitize_path_token(local_gnn))
    ensure_dir(gnn_dir)
    name_parts = [run_id, sanitize_path_token(global_attn)]
    if history_tag:
        name_parts.append(sanitize_path_token(history_tag))
    run_dir = os.path.join(gnn_dir, "__".join(name_parts))
    ensure_dir(run_dir)
    return run_dir, run_id


def to_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_edge_index(edge_index: Any) -> Optional[np.ndarray]:
    edge_np = to_numpy(edge_index)
    if edge_np is None or edge_np.ndim != 2:
        return None
    if edge_np.shape[0] == 2:
        return edge_np.astype(np.int64)
    if edge_np.shape[1] == 2:
        return edge_np.T.astype(np.int64)
    return None



def compute_laplacian_pe(edge_index: np.ndarray, num_nodes: int, k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros((num_nodes, 0), dtype=np.float32)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj[edge_index[1], edge_index[0]] = 1.0
    deg = adj.sum(axis=1)
    deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    lap = np.eye(num_nodes, dtype=np.float32) - (deg_inv_sqrt[:, None] * adj) * deg_inv_sqrt[None, :]
    eigvals, eigvecs = np.linalg.eigh(lap)
    idx = np.argsort(eigvals)
    eigvecs = eigvecs[:, idx]
    start = 1 if eigvecs.shape[1] > 1 else 0
    pe = eigvecs[:, start : start + k]
    if pe.shape[1] < k:
        pad = np.zeros((num_nodes, k - pe.shape[1]), dtype=np.float32)
        pe = np.concatenate([pe, pad], axis=1)
    return pe.astype(np.float32)


def compute_rwse(edge_index: np.ndarray, num_nodes: int, k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros((num_nodes, 0), dtype=np.float32)
    if num_nodes == 0:
        return np.zeros((0, k), dtype=np.float32)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj[edge_index[1], edge_index[0]] = 1.0
    adj += np.eye(num_nodes, dtype=np.float32)
    deg = adj.sum(axis=1)
    deg[deg == 0] = 1.0
    trans = adj / deg[:, None]
    rwse = np.zeros((num_nodes, k), dtype=np.float32)
    walk = trans.copy()
    for step in range(k):
        rwse[:, step] = np.diag(walk)
        walk = walk @ trans
    return rwse.astype(np.float32)


def compute_graph_features(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    num_nodes = max(int(num_nodes), 1)
    num_edges = int(edge_index.shape[1]) if edge_index.size else 0
    avg_degree = (2.0 * num_edges) / float(num_nodes)
    density = num_edges / float(num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0
    return np.array(
        [math.log1p(num_nodes), math.log1p(num_edges), avg_degree, density],
        dtype=np.float32,
    )


def undirected_edge_count(edge_index: np.ndarray) -> int:
    if edge_index.size == 0:
        return 0
    pairs = {
        (int(min(u, v)), int(max(u, v)))
        for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist())
        if int(u) != int(v)
    }
    return len(pairs)


def estimate_graph_diameter(edge_index: np.ndarray, num_nodes: int) -> int:
    if num_nodes <= 1:
        return 0
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        u_i = int(u)
        v_i = int(v)
        if u_i == v_i:
            continue
        adj[u_i].append(v_i)
        adj[v_i].append(u_i)

    def bfs_farthest(start: int) -> Tuple[int, int, np.ndarray]:
        dist = np.full(num_nodes, -1, dtype=np.int32)
        queue = [start]
        dist[start] = 0
        q_idx = 0
        farthest = start
        while q_idx < len(queue):
            node = queue[q_idx]
            q_idx += 1
            farthest = node
            for nb in adj[node]:
                if dist[nb] != -1:
                    continue
                dist[nb] = dist[node] + 1
                queue.append(nb)
        return farthest, int(dist[farthest]), dist

    visited = np.zeros(num_nodes, dtype=bool)
    diameter = 0
    for start in range(num_nodes):
        if visited[start]:
            continue
        _, _, dist0 = bfs_farthest(start)
        component_nodes = np.where(dist0 >= 0)[0]
        visited[component_nodes] = True
        if component_nodes.size <= 1:
            continue
        farthest = int(component_nodes[np.argmax(dist0[component_nodes])])
        _, comp_diameter, _ = bfs_farthest(farthest)
        diameter = max(diameter, comp_diameter)
    return int(diameter)


def graph_stats_dict(edge_index: np.ndarray, num_nodes: int) -> Dict[str, float]:
    num_nodes = int(num_nodes)
    num_edges = int(undirected_edge_count(edge_index))
    density = num_edges / float(num_nodes * (num_nodes - 1) / 2.0) if num_nodes > 1 else 0.0
    return {
        "num_nodes": float(num_nodes),
        "num_edges": float(num_edges),
        "density": float(density),
        "est_diameter": float(estimate_graph_diameter(edge_index, num_nodes)),
    }



def compute_spd_matrix(edge_index: torch.Tensor, batch: torch.Tensor, max_dist: int) -> torch.Tensor:
    device = edge_index.device
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    counts = torch.bincount(batch, minlength=num_graphs) if num_graphs > 0 else torch.tensor([], device=device)
    max_nodes = int(counts.max().item()) if counts.numel() > 0 else 0
    spd = torch.full(
        (num_graphs, max_nodes, max_nodes),
        max_dist + 1,
        dtype=torch.long,
        device=device,
    )
    for g in range(num_graphs):
        n = int(counts[g].item())
        if n == 0:
            continue
        node_idx = (batch == g).nonzero(as_tuple=False).view(-1)
        local_map = -torch.ones(batch.size(0), dtype=torch.long, device=device)
        local_map[node_idx] = torch.arange(n, device=device)
        edges = local_map[edge_index]
        valid = (edges[0] >= 0) & (edges[1] >= 0)
        edges = edges[:, valid].detach().cpu().numpy()
        adj = [set() for _ in range(n)]
        for u, v in edges.T:
            if u == v:
                continue
            adj[u].add(v)
            adj[v].add(u)
        spd_np = np.full((n, n), max_dist + 1, dtype=np.int64)
        for i in range(n):
            spd_np[i, i] = 0
            dist = [-1] * n
            dist[i] = 0
            queue = [i]
            for u in queue:
                if dist[u] >= max_dist:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for j, d in enumerate(dist):
                if 0 <= d <= max_dist:
                    spd_np[i, j] = d
        spd[g, :n, :n] = torch.tensor(spd_np, device=device)
    return spd


def compute_degree_hist(graphs: List[Data]) -> torch.Tensor:
    degrees: List[torch.Tensor] = []
    for graph in graphs:
        if graph.edge_index is None:
            raise ValueError("edge_index is required for degree histogram.")
        deg = degree(graph.edge_index[0], num_nodes=graph.num_nodes, dtype=torch.long)
        degrees.append(deg)
    if not degrees:
        return torch.tensor([1], dtype=torch.float32)
    all_deg = torch.cat(degrees, dim=0)
    return torch.bincount(all_deg, minlength=int(all_deg.max().item()) + 1).float()


def compute_feature_stats(graphs: List[Data]) -> Tuple[torch.Tensor, torch.Tensor]:
    count = 0
    sum_x = None
    sum_sq = None
    for graph in graphs:
        if graph.x is None:
            raise ValueError("data.x is required for feature normalization.")
        x = graph.x
        count += x.size(0)
        if sum_x is None:
            sum_x = x.sum(dim=0)
            sum_sq = (x * x).sum(dim=0)
        else:
            sum_x = sum_x + x.sum(dim=0)
            sum_sq = sum_sq + (x * x).sum(dim=0)
    if count == 0 or sum_x is None or sum_sq is None:
        raise ValueError("No node features available for normalization.")
    mean = sum_x / float(count)
    var = sum_sq / float(count) - mean * mean
    std = torch.sqrt(var.clamp(min=1e-12))
    return mean, std


def normalize_graphs(graphs: List[Data], mean: torch.Tensor, std: torch.Tensor) -> List[Data]:
    normalized: List[Data] = []
    for graph in graphs:
        clone = graph.clone()
        if clone.x is None:
            raise ValueError("data.x is required for feature normalization.")
        clone.x = (clone.x - mean) / std
        normalized.append(clone)
    return normalized


def normalize_graphs_per_graph(graphs: List[Data]) -> List[Data]:
    normalized: List[Data] = []
    for graph in graphs:
        clone = graph.clone()
        if clone.x is None:
            raise ValueError("data.x is required for feature normalization.")
        mean = clone.x.mean(dim=0)
        std = clone.x.std(dim=0, unbiased=False).clamp(min=1e-12)
        clone.x = (clone.x - mean) / std
        normalized.append(clone)
    return normalized



def get_attr_or_key(obj: Any, keys: Iterable[str]) -> Any:
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            return obj[key]
        if hasattr(obj, key):
            return getattr(obj, key)
    return None


def load_graphs(
    path: str,
    lap_pe_dim: int = 0,
    rwse_dim: int = 0,
    add_virtual_node: bool = False,
    graph_features: str = "none",
    edge_features: str = "none",
) -> Tuple[List[Data], np.ndarray]:
    with open(path, "rb") as handle:
        data = pickle.load(handle)

    graphs = None
    if isinstance(data, dict):
        for key in ("X_graph", "X", "x"):
            if key in data:
                graphs = data[key]
                break
    elif isinstance(data, (list, tuple)):
        graphs = data

    if graphs is None:
        raise ValueError("Could not find graphs under X_graph/X/x in pickle.")

    graph_labels = None
    if isinstance(data, dict):
        for key in ("y_graph", "graph_y", "graph_labels", "graph_label", "y", "Y"):
            if key in data:
                graph_labels = data[key]
                break

    graphs = list(graphs)
    labels_out = []
    dataset: List[Data] = []

    virtual_node = VirtualNode() if add_virtual_node and VirtualNode is not None else None

    for idx, graph in enumerate(graphs):
        if isinstance(graph, Data):
            x = graph.x
            edge_index = graph.edge_index
            graph_y = graph.y
        elif isinstance(graph, dict):
            x = get_attr_or_key(graph, ("x", "X", "features", "feat"))
            edge_index = get_attr_or_key(graph, ("edge_index", "edges", "edge_idx", "adj"))
            graph_y = get_attr_or_key(graph, ("y", "label", "graph_y", "graph_label"))
        else:
            raise ValueError(f"Unsupported graph type at index {idx}: {type(graph)}")

        if x is None or edge_index is None:
            raise ValueError(f"Missing x or edge_index for graph {idx}.")

        edge_np = normalize_edge_index(edge_index)
        if edge_np is None:
            raise ValueError(f"Unsupported edge_index shape for graph {idx}.")

        if graph_labels is not None:
            y_val = to_numpy(graph_labels)[idx]
        else:
            y_val = graph_y

        y_np = to_numpy(y_val)
        if y_np is None:
            raise ValueError(f"Missing graph label for graph {idx}.")
        y_np = np.asarray(y_np).reshape(-1)
        y_scalar = int(y_np[0])

        x_tensor = torch.tensor(to_numpy(x), dtype=torch.float32)
        edge_tensor = torch.tensor(edge_np, dtype=torch.long)
        y_tensor = torch.tensor([y_scalar], dtype=torch.long)
        data_item = Data(x=x_tensor, edge_index=edge_tensor, y=y_tensor)
        stats = graph_stats_dict(edge_np, x_tensor.size(0))
        data_item.graph_id = torch.tensor([idx], dtype=torch.long)
        data_item.graph_num_nodes = torch.tensor([stats["num_nodes"]], dtype=torch.float32)
        data_item.graph_num_edges = torch.tensor([stats["num_edges"]], dtype=torch.float32)
        data_item.graph_density = torch.tensor([stats["density"]], dtype=torch.float32)
        data_item.graph_est_diameter = torch.tensor([stats["est_diameter"]], dtype=torch.float32)

        if lap_pe_dim > 0:
            lap_pe = compute_laplacian_pe(edge_np, x_tensor.size(0), lap_pe_dim)
            data_item.lap_pe = torch.tensor(lap_pe, dtype=torch.float32)
        if rwse_dim > 0:
            rwse = compute_rwse(edge_np, x_tensor.size(0), rwse_dim)
            data_item.rwse = torch.tensor(rwse, dtype=torch.float32)
        if graph_features != "none":
            graph_feat = compute_graph_features(edge_np, x_tensor.size(0))
            data_item.graph_feat = torch.tensor(graph_feat, dtype=torch.float32).unsqueeze(0)
        if edge_features == "ones":
            data_item.edge_attr = torch.ones((edge_tensor.size(1), 1), dtype=torch.float32)

        if virtual_node is not None:
            data_item = virtual_node(data_item)

        dataset.append(data_item)
        labels_out.append(y_scalar)

    return dataset, np.asarray(labels_out, dtype=int)


def stratified_split(
    indices: np.ndarray,
    labels: np.ndarray,
    val_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_sub, val_sub = next(splitter.split(indices, labels[indices]))
    return indices[train_sub], indices[val_sub]


def compute_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "f1_macro": float(f1_score(true_labels, pred_labels, average="macro", zero_division=0)),
    }


def compute_roc_auc(true_labels: np.ndarray, probabilities: np.ndarray) -> Optional[float]:
    if probabilities.shape[1] == 2:
        return float(roc_auc_score(true_labels, probabilities[:, 1]))
    return float(roc_auc_score(true_labels, probabilities, multi_class="ovr", average="macro"))


def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thr = 0.5
    best_score = -1.0
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        score = f1_score(y_true, preds, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_thr = thr
    return best_thr, best_score


def struct_feature_dim(mode: str) -> int:
    if mode == "degree":
        return 1
    if mode == "degree_log":
        return 1
    if mode == "degree_and_log":
        return 2
    return 0


def build_struct_features(edge_index: torch.Tensor, num_nodes: int, mode: str) -> Optional[torch.Tensor]:
    if mode == "none":
        return None
    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float32)
    if mode == "degree":
        return deg.unsqueeze(1)
    if mode == "degree_log":
        return torch.log1p(deg).unsqueeze(1)
    if mode == "degree_and_log":
        return torch.stack([deg, torch.log1p(deg)], dim=1)
    raise ValueError(f"Unsupported struct feature mode: {mode}")



class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SignNetEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.mlp = MLP(in_dim, hidden_dim, out_dim, dropout)

    def forward(self, pe: torch.Tensor) -> torch.Tensor:
        return self.mlp(pe) + self.mlp(-pe)


class DeepSetEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.phi = MLP(in_dim, hidden_dim, hidden_dim, dropout)
        self.rho = MLP(hidden_dim, hidden_dim, out_dim, dropout)

    def forward(self, pe: torch.Tensor) -> torch.Tensor:
        return self.rho(self.phi(pe))



class GraphormerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("Hidden dimension must be divisible by heads.")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias.unsqueeze(1).to(attn_scores.dtype)
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(bsz, seq_len, hidden)
        out = self.proj_dropout(self.proj(out))
        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x


class GraphormerEncoder(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, ffn_dim: int, dropout: float, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [GraphormerEncoderLayer(hidden_dim, heads, ffn_dim, dropout) for _ in range(layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_bias, key_padding_mask)
        return x



def build_local_conv(
    conv_type: str,
    hidden_dim: int,
    heads: int,
    dropout: float,
    edge_dim: Optional[int],
    pna_deg: Optional[torch.Tensor],
) -> nn.Module:
    if conv_type == "gin":
        mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        return GINConv(mlp)
    if conv_type == "gine":
        mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        return GINEConv(mlp, edge_dim=edge_dim or 1)
    if conv_type == "sage":
        return SAGEConv(hidden_dim, hidden_dim)
    if conv_type == "gcn":
        return GCNConv(hidden_dim, hidden_dim)
    if conv_type == "pna":
        if pna_deg is None:
            raise ValueError("PNA requires degree histogram from training data.")
        return PNAConv(
            hidden_dim,
            hidden_dim,
            aggregators=["mean", "min", "max", "std"],
            scalers=["identity", "amplification", "attenuation"],
            deg=pna_deg,
            edge_dim=edge_dim,
        )
    if conv_type == "transformerconv":
        if hidden_dim % heads != 0:
            raise ValueError("Hidden dimension must be divisible by heads for TransformerConv.")
        return TransformerConv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
        )
    if conv_type == "gatedgraph":
        if GatedGraphConv is None:
            raise ValueError("GatedGraphConv is not available in this torch_geometric version.")
        return GatedGraphConv(hidden_dim, num_layers=1)
    raise ValueError(f"Unsupported local GNN type: {conv_type}")


def apply_edge_dropout(
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
    p: float,
    training: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if p <= 0 or not training:
        return edge_index, edge_attr
    dropped_edge_index, edge_mask = dropout_edge(edge_index, p=p, force_undirected=False, training=True)
    if edge_attr is None:
        return dropped_edge_index, None
    return dropped_edge_index, edge_attr[edge_mask]


class GraphormerGPSLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        dropout: float,
        ffn_dim: int,
        local_conv: nn.Module,
        max_dist: int,
    ) -> None:
        super().__init__()
        self.local_conv = local_conv
        self.graphormer = GraphormerEncoder(hidden_dim, heads, ffn_dim, dropout, layers=1)
        self.graphormer_bias = nn.Embedding(max_dist + 2, 1)
        self.max_dist = max_dist
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_attr is not None and isinstance(self.local_conv, (GINEConv, PNAConv, TransformerConv)):
            x_local = self.local_conv(x, edge_index, edge_attr)
        else:
            x_local = self.local_conv(x, edge_index)
        dense_x, mask = to_dense_batch(x, batch)
        spd = compute_spd_matrix(edge_index, batch, self.max_dist)
        attn_bias = self.graphormer_bias(spd).squeeze(-1)
        dense_x = self.graphormer(dense_x, attn_bias, key_padding_mask=~mask)
        x_global = dense_x[mask]
        x = self.norm1(x + self.dropout(x_local + x_global))
        x = self.norm2(x + self.ffn(x))
        return x



class GPSModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        gps_layers: int,
        local_gnn: str,
        global_attn: str,
        heads: int,
        dropout: float,
        attn_dropout: float,
        norm: str,
        struct_features: str,
        lap_pe_encoder: str,
        lap_pe_dim: int,
        lap_pe_out: int,
        rwse_encoder: str,
        rwse_dim: int,
        rwse_out: int,
        graphormer_max_dist: int,
        edge_dim: Optional[int],
        pna_deg: Optional[torch.Tensor],
        pool: str,
        graph_feat_dim: int,
        edge_dropout: float,
    ) -> None:
        super().__init__()
        self.struct_features = struct_features
        self.edge_dropout = float(edge_dropout)
        self.pool = pool
        self.graph_feat_dim = graph_feat_dim
        self.needs_edge_attr = local_gnn == "gine"
        self.pass_edge_attr = local_gnn in ("gine", "pna", "transformerconv")

        pe_out_dim = lap_pe_out if lap_pe_out > 0 else lap_pe_dim
        se_out_dim = rwse_out if rwse_out > 0 else rwse_dim

        self.lap_pe_encoder = None
        if lap_pe_dim > 0:
            if lap_pe_encoder == "raw":
                self.lap_pe_encoder = nn.Identity()
            elif lap_pe_encoder == "linear":
                self.lap_pe_encoder = nn.Linear(lap_pe_dim, pe_out_dim)
            elif lap_pe_encoder == "signnet":
                self.lap_pe_encoder = SignNetEncoder(lap_pe_dim, hidden_dim, pe_out_dim, dropout)
            elif lap_pe_encoder == "deepset":
                self.lap_pe_encoder = DeepSetEncoder(lap_pe_dim, hidden_dim, pe_out_dim, dropout)
            else:
                raise ValueError(f"Unsupported lap_pe_encoder: {lap_pe_encoder}")

        self.rwse_encoder = None
        if rwse_dim > 0:
            if rwse_encoder == "raw":
                self.rwse_encoder = nn.Identity()
            elif rwse_encoder == "linear":
                self.rwse_encoder = nn.Linear(rwse_dim, se_out_dim)
            else:
                raise ValueError(f"Unsupported rwse_encoder: {rwse_encoder}")

        struct_dim = struct_feature_dim(struct_features)
        input_dim = input_dim + struct_dim + pe_out_dim + se_out_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.layers = nn.ModuleList()
        self.global_attn = global_attn
        if global_attn == "none":
            for _ in range(gps_layers):
                conv = build_local_conv(local_gnn, hidden_dim, heads, dropout, edge_dim, pna_deg)
                self.layers.append(conv)
            self.norm = nn.LayerNorm(hidden_dim)
        elif global_attn == "graphormer":
            for _ in range(gps_layers):
                conv = build_local_conv(local_gnn, hidden_dim, heads, dropout, edge_dim, pna_deg)
                self.layers.append(
                    GraphormerGPSLayer(
                        hidden_dim,
                        heads,
                        dropout,
                        hidden_dim * 2,
                        conv,
                        graphormer_max_dist,
                    )
                )
        else:
            attn_type = global_attn
            for _ in range(gps_layers):
                conv = build_local_conv(local_gnn, hidden_dim, heads, dropout, edge_dim, pna_deg)
                try:
                    layer = GPSConv(
                        hidden_dim,
                        conv=conv,
                        heads=heads,
                        dropout=dropout,
                        norm=norm,
                        attn_type=attn_type,
                        attn_kwargs={"dropout": attn_dropout},
                    )
                except ValueError as exc:
                    if attn_type == "bigbird":
                        layer = GPSConv(
                            hidden_dim,
                            conv=conv,
                            heads=heads,
                            dropout=dropout,
                            norm=norm,
                            attn_type="performer",
                            attn_kwargs={"dropout": attn_dropout},
                        )
                        print("[warn] bigbird not supported; using performer instead.")
                    else:
                        raise exc
                self.layers.append(layer)

        if pool == "attention":
            self.pooler = AttentionalAggregation(gate_nn=nn.Sequential(nn.Linear(hidden_dim, 1)))
        else:
            self.pooler = None
        if graph_feat_dim > 0:
            self.graph_feat_norm = nn.LayerNorm(graph_feat_dim)
        else:
            self.graph_feat_norm = None
        head_in = hidden_dim + graph_feat_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, max(1, head_in // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(1, head_in // 2), num_classes),
        )

    def _append_graph_features(self, graph_emb: torch.Tensor, data: Data) -> torch.Tensor:
        if self.graph_feat_dim <= 0:
            return graph_emb
        if not hasattr(data, "graph_feat") or data.graph_feat is None:
            raise ValueError("graph_feat is required when graph_features are enabled")
        graph_feat = data.graph_feat
        if graph_feat.dim() == 1:
            graph_feat = graph_feat.unsqueeze(0)
        graph_feat = graph_feat.to(graph_emb.device)
        if self.graph_feat_norm is not None:
            graph_feat = self.graph_feat_norm(graph_feat)
        return torch.cat([graph_emb, graph_feat], dim=1)

    def forward(self, data: Data) -> torch.Tensor:
        if data.x is None or data.edge_index is None:
            raise ValueError("data.x and data.edge_index are required.")
        x = data.x
        feats = [x]

        if self.lap_pe_encoder is not None and hasattr(data, "lap_pe"):
            lap_pe = data.lap_pe.to(x.device)
            feats.append(self.lap_pe_encoder(lap_pe))
        if self.rwse_encoder is not None and hasattr(data, "rwse"):
            rwse = data.rwse.to(x.device)
            feats.append(self.rwse_encoder(rwse))
        struct = build_struct_features(data.edge_index, x.size(0), self.struct_features)
        if struct is not None:
            feats.append(struct.to(x.device))
        x = torch.cat(feats, dim=1)
        x = self.node_encoder(x)

        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
        if edge_attr is None and self.needs_edge_attr:
            edge_attr = torch.ones((data.edge_index.size(1), 1), device=x.device)
        edge_index = data.edge_index
        edge_index, edge_attr = apply_edge_dropout(edge_index, edge_attr, self.edge_dropout, self.training)
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else None
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if self.global_attn == "none":
            for conv in self.layers:
                if self.pass_edge_attr and edge_attr is not None:
                    x = conv(x, edge_index, edge_attr)
                else:
                    x = conv(x, edge_index)
                x = F.relu(x)
            x = self.norm(x)
        elif self.global_attn == "graphormer":
            for layer in self.layers:
                x = layer(x, edge_index, batch, edge_attr)
        else:
            for layer in self.layers:
                if self.pass_edge_attr and edge_attr is not None:
                    x = layer(x, edge_index, batch=batch, edge_attr=edge_attr)
                else:
                    x = layer(x, edge_index, batch=batch)

        if self.pool == "sum":
            graph_emb = global_add_pool(x, batch)
        elif self.pool == "attention":
            graph_emb = self.pooler(x, batch)
        else:
            graph_emb = global_mean_pool(x, batch)
        graph_emb = self._append_graph_features(graph_emb, data)
        return self.head(graph_emb)


class GPSNodeModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        gps_layers: int,
        local_gnn: str,
        global_attn: str,
        heads: int,
        dropout: float,
        attn_dropout: float,
        norm: str,
        struct_features: str,
        lap_pe_encoder: str,
        lap_pe_dim: int,
        lap_pe_out: int,
        rwse_encoder: str,
        rwse_dim: int,
        rwse_out: int,
        graphormer_max_dist: int,
        edge_dim: Optional[int],
        pna_deg: Optional[torch.Tensor],
        edge_dropout: float,
    ) -> None:
        super().__init__()
        self.struct_features = struct_features
        self.edge_dropout = float(edge_dropout)
        self.needs_edge_attr = local_gnn == "gine"
        self.pass_edge_attr = local_gnn in ("gine", "pna", "transformerconv")

        pe_out_dim = lap_pe_out if lap_pe_out > 0 else lap_pe_dim
        se_out_dim = rwse_out if rwse_out > 0 else rwse_dim

        self.lap_pe_encoder = None
        if lap_pe_dim > 0:
            if lap_pe_encoder == "raw":
                self.lap_pe_encoder = nn.Identity()
            elif lap_pe_encoder == "linear":
                self.lap_pe_encoder = nn.Linear(lap_pe_dim, pe_out_dim)
            elif lap_pe_encoder == "signnet":
                self.lap_pe_encoder = SignNetEncoder(lap_pe_dim, hidden_dim, pe_out_dim, dropout)
            elif lap_pe_encoder == "deepset":
                self.lap_pe_encoder = DeepSetEncoder(lap_pe_dim, hidden_dim, pe_out_dim, dropout)
            else:
                raise ValueError(f"Unsupported lap_pe_encoder: {lap_pe_encoder}")

        self.rwse_encoder = None
        if rwse_dim > 0:
            if rwse_encoder == "raw":
                self.rwse_encoder = nn.Identity()
            elif rwse_encoder == "linear":
                self.rwse_encoder = nn.Linear(rwse_dim, se_out_dim)
            else:
                raise ValueError(f"Unsupported rwse_encoder: {rwse_encoder}")

        struct_dim = struct_feature_dim(struct_features)
        input_dim = input_dim + struct_dim + pe_out_dim + se_out_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.layers = nn.ModuleList()
        self.global_attn = global_attn
        if global_attn == "none":
            for _ in range(gps_layers):
                conv = build_local_conv(local_gnn, hidden_dim, heads, dropout, edge_dim, pna_deg)
                self.layers.append(conv)
            self.norm = nn.LayerNorm(hidden_dim)
        elif global_attn == "graphormer":
            for _ in range(gps_layers):
                conv = build_local_conv(local_gnn, hidden_dim, heads, dropout, edge_dim, pna_deg)
                self.layers.append(
                    GraphormerGPSLayer(
                        hidden_dim,
                        heads,
                        dropout,
                        hidden_dim * 2,
                        conv,
                        graphormer_max_dist,
                    )
                )
        else:
            attn_type = global_attn
            for _ in range(gps_layers):
                conv = build_local_conv(local_gnn, hidden_dim, heads, dropout, edge_dim, pna_deg)
                try:
                    layer = GPSConv(
                        hidden_dim,
                        conv=conv,
                        heads=heads,
                        dropout=dropout,
                        norm=norm,
                        attn_type=attn_type,
                        attn_kwargs={"dropout": attn_dropout},
                    )
                except ValueError as exc:
                    if attn_type == "bigbird":
                        layer = GPSConv(
                            hidden_dim,
                            conv=conv,
                            heads=heads,
                            dropout=dropout,
                            norm=norm,
                            attn_type="performer",
                            attn_kwargs={"dropout": attn_dropout},
                        )
                        print("[warn] bigbird not supported; using performer instead.")
                    else:
                        raise exc
                self.layers.append(layer)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max(1, hidden_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(1, hidden_dim // 2), num_classes),
        )

    def forward(self, data: Data) -> torch.Tensor:
        if data.x is None or data.edge_index is None:
            raise ValueError("data.x and data.edge_index are required.")
        x = data.x
        feats = [x]

        if self.lap_pe_encoder is not None and hasattr(data, "lap_pe"):
            lap_pe = data.lap_pe.to(x.device)
            feats.append(self.lap_pe_encoder(lap_pe))
        if self.rwse_encoder is not None and hasattr(data, "rwse"):
            rwse = data.rwse.to(x.device)
            feats.append(self.rwse_encoder(rwse))
        struct = build_struct_features(data.edge_index, x.size(0), self.struct_features)
        if struct is not None:
            feats.append(struct.to(x.device))
        x = torch.cat(feats, dim=1)
        x = self.node_encoder(x)

        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
        if edge_attr is None and self.needs_edge_attr:
            edge_attr = torch.ones((data.edge_index.size(1), 1), device=x.device)
        edge_index = data.edge_index
        edge_index, edge_attr = apply_edge_dropout(edge_index, edge_attr, self.edge_dropout, self.training)
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else None
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if self.global_attn == "none":
            for conv in self.layers:
                if self.pass_edge_attr and edge_attr is not None:
                    x = conv(x, edge_index, edge_attr)
                else:
                    x = conv(x, edge_index)
                x = F.relu(x)
            x = self.norm(x)
        elif self.global_attn == "graphormer":
            for layer in self.layers:
                x = layer(x, edge_index, batch, edge_attr)
        else:
            for layer in self.layers:
                if self.pass_edge_attr and edge_attr is not None:
                    x = layer(x, edge_index, batch=batch, edge_attr=edge_attr)
                else:
                    x = layer(x, edge_index, batch=batch)

        return self.head(x)


class GPSNetNode(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        gps_layers: int,
        local_gnn: str,
        global_attn: str,
        heads: int,
        dropout: float,
        edge_dim: Optional[int],
        pna_deg: Optional[torch.Tensor],
    ) -> None:
        super().__init__()
        self.needs_edge_attr = local_gnn == "gine"
        self.pass_edge_attr = local_gnn in ("gine", "pna", "transformerconv")
        self.lin_in = nn.Linear(input_dim, hidden_dim)
        self.bn_in = nn.BatchNorm1d(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        if global_attn == "none":
            for _ in range(gps_layers):
                conv = build_local_conv(local_gnn, hidden_dim, heads, dropout, edge_dim, pna_deg)
                self.layers.append(conv)
                self.bns.append(nn.BatchNorm1d(hidden_dim))
        else:
            for _ in range(gps_layers):
                conv = build_local_conv(local_gnn, hidden_dim, heads, dropout, edge_dim, pna_deg)
                self.layers.append(
                    GPSConv(
                        hidden_dim,
                        conv=conv,
                        heads=heads,
                        dropout=dropout,
                        attn_type=global_attn,
                    )
                )
                self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.lin_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        if data.x is None or data.edge_index is None:
            raise ValueError("data.x and data.edge_index are required.")
        x = self.lin_in(data.x)
        x = self.bn_in(x)
        x = torch.relu(x)
        x = self.drop(x)

        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
        if edge_attr is None and self.needs_edge_attr:
            edge_attr = torch.ones((data.edge_index.size(1), 1), device=x.device)
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else None
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for idx, layer in enumerate(self.layers):
            if isinstance(layer, GPSConv):
                if self.pass_edge_attr and edge_attr is not None:
                    x = layer(x, edge_index, batch=batch, edge_attr=edge_attr)
                else:
                    x = layer(x, edge_index, batch=batch)
            else:
                if self.pass_edge_attr and edge_attr is not None:
                    x = layer(x, edge_index, edge_attr=edge_attr)
                else:
                    x = layer(x, edge_index)
            if idx < len(self.layers) - 1:
                x = self.bns[idx](x)
                x = torch.relu(x)
                x = self.drop(x)

        return self.lin_out(x)



def collect_eval_outputs(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            all_logits.append(logits.cpu())
            all_labels.append(batch.y.view(-1).cpu())
    if not all_logits:
        return np.asarray([], dtype=int), np.zeros((0, 2), dtype=np.float32)
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    probs = torch.softmax(logits, dim=1).numpy()
    return labels.numpy(), probs


def evaluate_model(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    labels, probs = collect_eval_outputs(model, loader, device)
    if labels.size == 0:
        return {"accuracy": 0.0, "f1_macro": 0.0}
    preds = probs.argmax(axis=1)
    metrics = compute_metrics(labels, preds)
    try:
        metrics["roc_auc"] = compute_roc_auc(labels, probs)
    except ValueError as exc:
        print("[warn] ROC-AUC unavailable:", exc)
    return metrics


def export_graph_predictions(
    path: str,
    run_name: str,
    graph_indices: np.ndarray,
    dataset_split: List[Data],
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    rows: List[Dict[str, Any]] = []
    binary_probs = probabilities[:, 1] if probabilities.ndim == 2 and probabilities.shape[1] == 2 else np.full(len(true_labels), np.nan)
    for pos, graph_idx in enumerate(graph_indices.tolist()):
        item = dataset_split[pos]
        rows.append(
            {
                "fold": run_name,
                "graph_index": int(graph_idx),
                "true_label": int(true_labels[pos]),
                "pred_label": int(pred_labels[pos]),
                "pred_prob_pos": float(binary_probs[pos]) if pos < len(binary_probs) else float("nan"),
                "num_nodes": float(item.graph_num_nodes.view(-1)[0].item()) if hasattr(item, "graph_num_nodes") else float("nan"),
                "num_edges": float(item.graph_num_edges.view(-1)[0].item()) if hasattr(item, "graph_num_edges") else float("nan"),
                "density": float(item.graph_density.view(-1)[0].item()) if hasattr(item, "graph_density") else float("nan"),
                "est_diameter": float(item.graph_est_diameter.view(-1)[0].item()) if hasattr(item, "graph_est_diameter") else float("nan"),
            }
        )
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def augment_graph_for_training(data: Data, feature_mask_prob: float, node_drop_prob: float) -> Data:
    out = data.clone()

    if feature_mask_prob > 0.0 and hasattr(out, "x") and out.x is not None:
        mask = torch.rand_like(out.x) < feature_mask_prob
        out.x = out.x.masked_fill(mask, 0.0)

    if node_drop_prob <= 0.0:
        return out

    num_nodes = int(out.num_nodes)
    if num_nodes <= 2:
        return out

    keep_mask = torch.rand(num_nodes) > node_drop_prob
    if int(keep_mask.sum().item()) < 2:
        keep_mask[torch.randint(0, num_nodes, (2,))] = True
    keep_idx = keep_mask.nonzero(as_tuple=False).view(-1)

    edge_attr = out.edge_attr if hasattr(out, "edge_attr") else None
    new_edge_index, new_edge_attr = subgraph(
        keep_idx,
        out.edge_index,
        edge_attr=edge_attr,
        relabel_nodes=True,
        num_nodes=num_nodes,
    )

    for key in list(out.keys()):
        if key in ("edge_index", "edge_attr", "y", "graph_feat"):
            continue
        value = getattr(out, key)
        if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == num_nodes:
            setattr(out, key, value[keep_idx])

    out.edge_index = new_edge_index
    if edge_attr is not None:
        out.edge_attr = new_edge_attr
    out.num_nodes = int(keep_idx.numel())
    return out


def train_one_split(
    dataset: List[Data],
    labels: np.ndarray,
    args: argparse.Namespace,
    device: str,
    split: FoldSplit,
    run_name: str,
    history_path: Optional[str] = None,
    prediction_path: Optional[str] = None,
) -> Dict[str, float]:
    train_ds = [dataset[i] for i in split.train_idx]
    val_ds = [dataset[i] for i in split.val_idx]
    test_ds = [dataset[i] for i in split.test_idx]

    if args.feature_norm == "standard":
        mean, std = compute_feature_stats(train_ds)
        train_ds = normalize_graphs(train_ds, mean, std)
        val_ds = normalize_graphs(val_ds, mean, std)
        test_ds = normalize_graphs(test_ds, mean, std)
    elif args.feature_norm == "per-graph":
        train_ds = normalize_graphs_per_graph(train_ds)
        val_ds = normalize_graphs_per_graph(val_ds)
        test_ds = normalize_graphs_per_graph(test_ds)

    train_labels = labels[split.train_idx]
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = train_ds[0].num_node_features

    edge_dim = None
    if hasattr(train_ds[0], "edge_attr") and train_ds[0].edge_attr is not None:
        edge_dim = int(train_ds[0].edge_attr.size(1)) if train_ds[0].edge_attr.dim() > 1 else 1
    graph_feat_dim = 0
    if hasattr(train_ds[0], "graph_feat") and train_ds[0].graph_feat is not None:
        graph_feat_dim = int(train_ds[0].graph_feat.numel())
    pna_deg = compute_degree_hist(train_ds) if args.local_gnn == "pna" else None

    model = GPSModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=int(labels.max()) + 1,
        gps_layers=args.gps_layers,
        local_gnn=args.local_gnn,
        global_attn=args.global_attn,
        heads=args.heads,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        norm=args.norm,
        struct_features=args.struct_features,
        lap_pe_encoder=args.lap_pe_encoder,
        lap_pe_dim=args.lap_pe_dim,
        lap_pe_out=args.lap_pe_out_dim,
        rwse_encoder=args.rwse_encoder,
        rwse_dim=args.rwse_dim,
        rwse_out=args.rwse_out_dim,
        graphormer_max_dist=args.graphormer_max_dist,
        edge_dim=edge_dim,
        pna_deg=pna_deg,
        pool=args.pool,
        graph_feat_dim=graph_feat_dim,
        edge_dropout=args.edge_dropout,
    ).to(device)

    if args.disable_class_weights:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        class_counts = np.bincount(labels[split.train_idx])
        inv = class_counts.sum() / np.clip(class_counts, 1.0, None)
        weights = torch.tensor(inv / inv.mean(), dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_val_score = -1.0
    no_improve = 0

    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        epoch_train_ds = train_ds
        if args.graph_aug_feature_mask > 0.0 or args.graph_aug_node_drop > 0.0:
            epoch_train_ds = [
                augment_graph_for_training(graph, args.graph_aug_feature_mask, args.graph_aug_node_drop)
                for graph in train_ds
            ]

        if args.balanced_sampler:
            class_counts = np.bincount(train_labels)
            class_weights = 1.0 / np.clip(class_counts, 1.0, None)
            sample_weights = class_weights[train_labels]
            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.double),
                num_samples=len(epoch_train_ds),
                replacement=True,
            )
            train_loader = DataLoader(epoch_train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=False)
        else:
            train_loader = DataLoader(epoch_train_ds, batch_size=args.batch_size, shuffle=True)

        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            batch_size = int(batch.y.view(-1).size(0))
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
        train_loss = total_loss / max(total_samples, 1)
        val_metrics = evaluate_model(model, val_loader, device)
        val_f1 = val_metrics.get("f1_macro", 0.0)
        val_score = float(val_metrics.get(args.selection_metric, float("nan")))
        if not np.isfinite(val_score):
            val_score = -1.0
        if history_path is not None:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_accuracy": float(val_metrics.get("accuracy", 0.0)),
                    "val_f1": float(val_f1),
                    "val_roc_auc": float(val_metrics.get("roc_auc", float("nan"))),
                    "val_selection": float(val_score),
                }
            )
        if val_score > best_val_score:
            best_val_score = val_score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= args.patience:
            break
        if epoch == 1 or epoch % 10 == 0:
            print(f"[{run_name}] epoch {epoch:03d} val_f1={val_f1:.4f}")

    if history_path is not None and history:
        with open(history_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_model(model, test_loader, device)
    test_true, test_probs = collect_eval_outputs(model, test_loader, device)
    test_preds = test_probs.argmax(axis=1) if test_probs.size > 0 else np.asarray([], dtype=int)
    if args.threshold_calibration != "none":
        val_true, val_probs = collect_eval_outputs(model, val_loader, device)
        if val_probs.shape[1] == 2 and test_probs.shape[1] == 2 and val_true.size > 0 and test_true.size > 0:
            best_thr, best_thr_f1 = find_best_threshold(val_true, val_probs[:, 1])
            test_preds = (test_probs[:, 1] >= best_thr).astype(int)
            test_metrics = compute_metrics(test_true, test_preds)
            try:
                test_metrics["roc_auc"] = float(roc_auc_score(test_true, test_probs[:, 1]))
            except ValueError as exc:
                print("[warn] ROC-AUC unavailable:", exc)
            test_metrics["val_thr"] = float(best_thr)
            test_metrics["val_thr_f1"] = float(best_thr_f1)
        else:
            print("[warn] Threshold calibration skipped (requires non-empty binary outputs).")
    if prediction_path is not None and test_true.size > 0 and test_probs.size > 0:
        export_graph_predictions(prediction_path, run_name, split.test_idx, test_ds, test_true, test_preds, test_probs)
    return test_metrics


def train_node_split(
    data: Data,
    labels: np.ndarray,
    args: argparse.Namespace,
    device: str,
    split: FoldSplit,
    run_name: str,
    history_path: Optional[str] = None,
) -> Dict[str, float]:
    from torch_geometric.loader import NeighborLoader

    edge_dim = data.edge_attr.size(1) if hasattr(data, "edge_attr") and data.edge_attr is not None else None
    pna_deg = None
    if args.local_gnn == "pna":
        deg = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        pna_deg = torch.bincount(deg, minlength=int(deg.max()) + 1).float()

    if args.node_arch == "gps_net":
        model = GPSNetNode(
            input_dim=data.num_node_features,
            hidden_dim=args.hidden_dim,
            num_classes=int(labels.max()) + 1,
            gps_layers=args.gps_layers,
            local_gnn=args.local_gnn,
            global_attn=args.global_attn,
            heads=args.heads,
            dropout=args.dropout,
            edge_dim=edge_dim,
            pna_deg=pna_deg,
        ).to(device)
    else:
        model = GPSNodeModel(
            input_dim=data.num_node_features,
            hidden_dim=args.hidden_dim,
            num_classes=int(labels.max()) + 1,
            gps_layers=args.gps_layers,
            local_gnn=args.local_gnn,
            global_attn=args.global_attn,
            heads=args.heads,
            dropout=args.dropout,
            attn_dropout=args.attn_dropout,
            norm=args.norm,
            struct_features=args.struct_features,
            lap_pe_encoder=args.lap_pe_encoder,
            lap_pe_dim=args.lap_pe_dim,
            lap_pe_out=args.lap_pe_out_dim,
            rwse_encoder=args.rwse_encoder,
            rwse_dim=args.rwse_dim,
            rwse_out=args.rwse_out_dim,
            graphormer_max_dist=args.graphormer_max_dist,
            edge_dim=edge_dim,
            pna_deg=pna_deg,
            edge_dropout=args.edge_dropout,
        ).to(device)

    if args.disable_class_weights:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        weights = class_weights_from_labels(
            labels[split.train_idx],
            smoothing=args.class_weight_smoothing,
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_val_pr_auc = -1.0
    no_improve = 0

    history: List[Dict[str, float]] = []

    node_neighbors = parse_list(args.node_neighbors, int)
    train_loader = NeighborLoader(
        data,
        num_neighbors=node_neighbors,
        input_nodes=torch.as_tensor(split.train_idx, dtype=torch.long),
        batch_size=args.node_batch_size,
        shuffle=True,
    )

    full_data = data.to(device)
    val_idx = torch.as_tensor(split.val_idx, dtype=torch.long, device=device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        train_preds: List[np.ndarray] = []
        train_targets: List[np.ndarray] = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            logits = logits[: batch.batch_size]
            loss = criterion(logits, batch.y[: batch.batch_size])
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batch_size = int(batch.batch_size)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            train_preds.append(logits.argmax(1).detach().cpu().numpy())
            train_targets.append(batch.y[: batch.batch_size].cpu().numpy())

        train_loss = total_loss / max(total_samples, 1)
        if train_preds:
            yp_tr = np.concatenate(train_preds)
            yt_tr = np.concatenate(train_targets)
            tr_acc = accuracy_score(yt_tr, yp_tr)
            tr_f1 = f1_score(yt_tr, yp_tr, average="macro")
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

        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average="macro")
        try:
            val_roc_auc = roc_auc_score(val_true, val_probs)
        except ValueError:
            val_roc_auc = float("nan")
        try:
            val_pr_auc = average_precision_score(val_true, val_probs)
        except ValueError:
            val_pr_auc = 0.0

        if history_path is not None:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": float(tr_acc),
                    "train_f1": float(tr_f1),
                    "val_acc": float(val_acc),
                    "val_f1": float(val_f1),
                    "val_roc_auc": float(val_roc_auc),
                    "val_pr_auc": float(val_pr_auc),
                }
            )

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.patience:
            break
        if epoch == 1 or epoch % 10 == 0:
            print(f"[{run_name}] epoch {epoch:03d} val_pr_auc={val_pr_auc:.4f}")

    if history_path is not None and history:
        with open(history_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits_full = model(full_data)
        val_logits = logits_full[val_idx]
        val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
        val_true = labels[split.val_idx]

    best_thr, best_thr_f1 = find_best_threshold(val_true, val_probs)

    test_idx = torch.as_tensor(split.test_idx, dtype=torch.long, device=device)
    with torch.no_grad():
        test_logits = logits_full[test_idx]
        test_probs = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
        test_preds = (test_probs >= best_thr).astype(int)
        test_true = labels[split.test_idx]

    metrics = compute_metrics(test_true, test_preds)
    try:
        metrics["roc_auc"] = float(roc_auc_score(test_true, test_probs))
    except ValueError as exc:
        print("[warn] ROC-AUC unavailable:", exc)
        metrics["roc_auc"] = float("nan")
    try:
        metrics["pr_auc"] = float(average_precision_score(test_true, test_probs))
    except ValueError as exc:
        print("[warn] PR-AUC unavailable:", exc)
        metrics["pr_auc"] = float("nan")
    metrics["val_thr_f1"] = float(best_thr_f1)
    metrics["val_thr"] = float(best_thr)
    return metrics


def build_splits(labels: np.ndarray, args: argparse.Namespace) -> List[FoldSplit]:
    indices = np.arange(len(labels))
    if args.fixed_split:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
        train_idx, test_idx = next(splitter.split(indices, labels))
        train_idx, val_idx = stratified_split(train_idx, labels, args.val_frac, args.seed + 1)
        return [FoldSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)]
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits: List[FoldSplit] = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(indices, labels), start=1):
        train_idx, val_idx = stratified_split(train_idx, labels, args.val_frac, args.seed + fold)
        splits.append(FoldSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx))
    return splits


def run_node_experiment(
    args: argparse.Namespace,
    device: str,
    run_id: Optional[str],
    history_tag: str,
) -> Dict[str, float]:
    x_all, y_all, edge_index = load_lympho_dataset(args.path)
    edge_np = normalize_edge_index(edge_index)
    if edge_np is None:
        raise ValueError("Unsupported edge_index shape in stitched graph.")

    edge_tensor = torch.tensor(edge_np, dtype=torch.long)
    edge_tensor = to_undirected(edge_tensor)
    edge_np = edge_tensor.cpu().numpy()

    data = Data(
        x=torch.tensor(x_all, dtype=torch.float32),
        y=torch.tensor(y_all, dtype=torch.long),
        edge_index=edge_tensor,
    )

    if args.lap_pe_dim > 0:
        lap_pe = compute_laplacian_pe(edge_np, data.num_nodes, args.lap_pe_dim)
        data.lap_pe = torch.tensor(lap_pe, dtype=torch.float32)
    if args.rwse_dim > 0:
        rwse = compute_rwse(edge_np, data.num_nodes, args.rwse_dim)
        data.rwse = torch.tensor(rwse, dtype=torch.float32)
    if args.edge_features == "ones":
        data.edge_attr = torch.ones((data.edge_index.size(1), 1), dtype=torch.float32)

    indices = np.arange(len(y_all))
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    val_ratio = args.val_frac / (1.0 - 1.0 / args.folds)

    metrics_list: List[Dict[str, float]] = []
    history_paths: List[str] = []

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(indices, y_all), start=1):
        train_idx, val_idx = stratified_split(train_val_idx, y_all, val_ratio, args.seed + fold)
        split = FoldSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
        run_name = f"fold{fold}"
        history_path = None
        if args.save_history and run_id is not None:
            tag = f"_{history_tag}" if history_tag else ""
            history_path = os.path.join(args.out_dir, f"history_{run_id}{tag}_{run_name}.csv")
            history_paths.append(history_path)
        metrics = train_node_split(data, y_all, args, device, split, run_name, history_path)
        metrics_list.append(metrics)
        print(f"[{run_name}] metrics: {metrics}")

    keys = sorted({k for m in metrics_list for k in m})
    summary: Dict[str, float] = {}
    for key in keys:
        vals = [m[key] for m in metrics_list if key in m]
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))

    if args.save_history and run_id is not None and history_paths:
        tag = f"_{history_tag}" if history_tag else ""
        plot_learning_curves(
            history_paths,
            os.path.join(args.out_dir, f"learning_curves_{run_id}{tag}.png"),
        )
        plot_roc_auc_curves(
            history_paths,
            os.path.join(args.out_dir, f"roc_auc_curves_{run_id}{tag}.png"),
        )

    return summary


def parse_list(values: str, cast_type=str) -> List[Any]:
    if values is None:
        return []
    parts = [item.strip() for item in values.split(",") if item.strip()]
    return [cast_type(item) for item in parts]


def plot_learning_curves(history_paths: List[str], out_path: str) -> None:
    if not history_paths:
        return
    train_loss_by_epoch: Dict[int, List[float]] = {}
    val_f1_by_epoch: Dict[int, List[float]] = {}
    for path in history_paths:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                epoch = int(row["epoch"])
                train_loss_by_epoch.setdefault(epoch, []).append(float(row["train_loss"]))
                val_f1_by_epoch.setdefault(epoch, []).append(float(row["val_f1"]))
    epochs = sorted(train_loss_by_epoch.keys())
    if not epochs:
        return
    mean_train_loss = [float(np.mean(train_loss_by_epoch[ep])) for ep in epochs]
    mean_val_f1 = [float(np.mean(val_f1_by_epoch.get(ep, []))) for ep in epochs]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(epochs, mean_train_loss, color="#4C78A8")
    axes[1].plot(epochs, mean_val_f1, color="#F58518")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Learning Curves (Train Loss)")
    axes[1].set_ylabel("Val F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Learning Curves (Validation F1)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_roc_auc_curves(history_paths: List[str], out_path: str) -> None:
    if not history_paths:
        return
    roc_auc_by_epoch: Dict[int, List[float]] = {}
    for path in history_paths:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                value = float(row["val_roc_auc"])
                if value != value:
                    continue
                epoch = int(row["epoch"])
                roc_auc_by_epoch.setdefault(epoch, []).append(value)
    epochs = sorted(roc_auc_by_epoch.keys())
    if not epochs:
        return
    mean_roc_auc = [float(np.mean(roc_auc_by_epoch[ep])) for ep in epochs]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(epochs, mean_roc_auc, color="#54A24B")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val ROC-AUC")
    ax.set_title("Validation ROC-AUC Over Epochs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_experiment(
    dataset: List[Data],
    labels: np.ndarray,
    args: argparse.Namespace,
    run_id: Optional[str] = None,
    history_tag: str = "",
) -> Dict[str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, (torch.cuda.get_device_name(0) if device == "cuda" else "CPU"))
    splits = build_splits(labels, args)
    metrics_list: List[Dict[str, float]] = []
    history_paths: List[str] = []
    for idx, split in enumerate(splits, start=1):
        run_name = f"fold{idx}"
        history_path = None
        prediction_path = None
        if args.save_history and run_id is not None:
            tag = f"_{history_tag}" if history_tag else ""
            history_path = os.path.join(args.out_dir, f"history_{run_id}{tag}_{run_name}.csv")
            history_paths.append(history_path)
        if run_id is not None:
            tag = f"_{history_tag}" if history_tag else ""
            prediction_path = os.path.join(args.out_dir, f"test_predictions_{run_id}{tag}_{run_name}.csv")
        metrics = train_one_split(dataset, labels, args, device, split, run_name, history_path, prediction_path)
        metrics_list.append(metrics)
        print(f"[{run_name}] metrics: {metrics}")

    keys = sorted({k for m in metrics_list for k in m})
    summary: Dict[str, float] = {}
    for key in keys:
        vals = [m[key] for m in metrics_list if key in m]
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))
    if args.save_history and run_id is not None and history_paths:
        tag = f"_{history_tag}" if history_tag else ""
        plot_learning_curves(
            history_paths,
            os.path.join(args.out_dir, f"learning_curves_{run_id}{tag}.png"),
        )
        plot_roc_auc_curves(
            history_paths,
            os.path.join(args.out_dir, f"roc_auc_curves_{run_id}{tag}.png"),
        )
    return summary



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPS-style GNN/Transformer experiments on lymphocyte data")
    parser.add_argument("--path", default=DEFAULT_DATA_PATH, help="Path to dataset pickle")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fixed-split", action="store_true")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--edge-dropout", type=float, default=0.0)
    parser.add_argument("--graph-aug-feature-mask", type=float, default=0.0)
    parser.add_argument("--graph-aug-node-drop", type=float, default=0.0)
    parser.add_argument("--feature-norm", choices=("none", "standard", "per-graph"), default="none")
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--disable-class-weights", action="store_true")
    parser.add_argument("--class-weight-smoothing", type=float, default=0.0)
    parser.add_argument("--selection-metric", default="f1_macro", choices=("f1_macro", "roc_auc"))
    parser.add_argument("--threshold-calibration", default="none", choices=("none", "f1_macro"))
    parser.add_argument(
        "--save-history",
        action="store_true",
        default=True,
        help="Save per-epoch train/val metrics (default: on)",
    )
    parser.add_argument("--history-tag", default="", help="Optional tag for history files")
    parser.add_argument("--node-level", action="store_true", help="Run node-level CV on stitched graph")
    parser.add_argument("--node-batch-size", type=int, default=2048)
    parser.add_argument("--node-neighbors", default="25,15")
    parser.add_argument("--node-eval-batch-size", type=int, default=4096)
    parser.add_argument("--node-arch", default="gps_model", choices=("gps_model", "gps_net"))
    parser.add_argument(
        "--gps-preset",
        default="none",
        choices=("none", "paper_cifar", "lympho_small", "lympho_node"),
        help="Apply a GPS Study preset (approx. CIFAR10 config).",
    )

    parser.add_argument(
        "--local-gnn",
        default="sage",
        choices=("sage", "gin", "gine", "gcn", "pna", "transformerconv", "gatedgraph"),
    )
    parser.add_argument(
        "--global-attn",
        default="performer",
        choices=("none", "multihead", "performer", "bigbird", "graphormer"),
    )
    parser.add_argument("--gps-layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    parser.add_argument("--norm", default="batch_norm", choices=("batch_norm", "layer_norm", "none"))
    parser.add_argument("--pool", default="mean", choices=("mean", "sum", "attention"))
    parser.add_argument("--struct-features", default="none", choices=("none", "degree", "degree_log", "degree_and_log"))

    parser.add_argument("--lap-pe-dim", type=int, default=0)
    parser.add_argument("--lap-pe-encoder", default="raw", choices=("raw", "linear", "signnet", "deepset"))
    parser.add_argument("--lap-pe-out-dim", type=int, default=0)
    parser.add_argument("--rwse-dim", type=int, default=0)
    parser.add_argument("--rwse-encoder", default="raw", choices=("raw", "linear"))
    parser.add_argument("--rwse-out-dim", type=int, default=0)
    parser.add_argument("--graphormer-max-dist", type=int, default=5)

    parser.add_argument("--graph-features", default="none", choices=("none", "basic"))
    parser.add_argument("--edge-features", default="none", choices=("none", "ones"))
    parser.add_argument("--virtual-node", action="store_true")

    parser.add_argument("--sweep", action="store_true", help="Run a sweep over method combinations")
    parser.add_argument("--sweep-local", default="sage,gin,pna,transformerconv,gine")
    parser.add_argument("--sweep-global", default="none,multihead,performer")
    parser.add_argument("--sweep-lap", default="0,8")
    parser.add_argument("--sweep-rwse", default="0,8")
    parser.add_argument("--sweep-pe-encoder", default="raw,signnet")
    parser.add_argument("--sweep-se-encoder", default="raw,linear")
    return parser


def apply_gps_preset(args: argparse.Namespace) -> None:
    if args.gps_preset == "paper_cifar":
        args.local_gnn = "gine"
        args.global_attn = "multihead"
        args.gps_layers = 3
        args.hidden_dim = 52
        args.heads = 4
        args.dropout = 0.0
        args.attn_dropout = 0.5
        args.pool = "mean"
        args.lap_pe_dim = 8
        args.lap_pe_encoder = "deepset"
        args.lap_pe_out_dim = 0
        args.rwse_dim = 0
        args.rwse_encoder = "raw"
        args.rwse_out_dim = 0
        args.struct_features = "none"
        args.edge_features = "none"
        args.graph_features = "none"
        args.lr = 0.001
        args.weight_decay = 1e-5
        args.epochs = 100
        args.batch_size = 16
        args.norm = "batch_norm"
        if args.history_tag:
            args.history_tag = f"{args.history_tag}_paper_cifar"
        else:
            args.history_tag = "paper_cifar"
    elif args.gps_preset == "lympho_small":
        args.local_gnn = "sage"
        args.global_attn = "performer"
        args.gps_layers = 2
        args.hidden_dim = 128
        args.heads = 4
        args.dropout = 0.2
        args.attn_dropout = 0.2
        args.pool = "mean"
        args.lap_pe_dim = 8
        args.lap_pe_encoder = "signnet"
        args.lap_pe_out_dim = 0
        args.rwse_dim = 8
        args.rwse_encoder = "linear"
        args.rwse_out_dim = 0
        args.struct_features = "degree_log"
        args.edge_features = "none"
        args.graph_features = "none"
        args.lr = 0.002
        args.weight_decay = 5e-4
        args.epochs = 120
        args.batch_size = 16
        args.norm = "batch_norm"
        if args.history_tag:
            args.history_tag = f"{args.history_tag}_lympho_small"
        else:
            args.history_tag = "lympho_small"
    elif args.gps_preset == "lympho_node":
        args.node_level = True
        args.node_arch = "gps_net"
        args.local_gnn = "sage"
        args.global_attn = "multihead"
        args.gps_layers = 2
        args.hidden_dim = 256
        args.heads = 4
        args.dropout = 0.5
        args.attn_dropout = 0.0
        args.pool = "mean"
        args.lap_pe_dim = 0
        args.lap_pe_encoder = "raw"
        args.lap_pe_out_dim = 0
        args.rwse_dim = 0
        args.rwse_encoder = "raw"
        args.rwse_out_dim = 0
        args.struct_features = "none"
        args.edge_features = "none"
        args.graph_features = "none"
        args.lr = 0.003
        args.weight_decay = 5e-4
        args.epochs = 80
        args.node_batch_size = 2048
        args.node_neighbors = "25,15"
        args.node_eval_batch_size = 4096
        args.norm = "batch_norm"
        args.label_smoothing = 0.05
        args.class_weight_smoothing = 0.3
        args.disable_class_weights = False
        if args.history_tag:
            args.history_tag = f"{args.history_tag}_lympho_node"
        else:
            args.history_tag = "lympho_node"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.gps_preset != "none":
        apply_gps_preset(args)
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, (torch.cuda.get_device_name(0) if device == "cuda" else "CPU"))

    base_out_dir = args.out_dir
    ensure_dir(base_out_dir)
    mode_out_dir = os.path.join(base_out_dir, "node_level" if args.node_level else "graph_level")
    ensure_dir(mode_out_dir)

    if args.node_level:
        if args.sweep:
            local_list = parse_list(args.sweep_local, str)
            global_list = parse_list(args.sweep_global, str)
            lap_list = parse_list(args.sweep_lap, int)
            rwse_list = parse_list(args.sweep_rwse, int)
            pe_enc_list = parse_list(args.sweep_pe_encoder, str)
            se_enc_list = parse_list(args.sweep_se_encoder, str)

            sweep_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = os.path.join(mode_out_dir, f"gps_sweep_summary_{sweep_run_id}.csv")
            rows: List[Dict[str, Any]] = []

            for local_gnn, global_attn, lap_dim, rwse_dim, pe_enc, se_enc in product(
                local_list, global_list, lap_list, rwse_list, pe_enc_list, se_enc_list
            ):
                sweep_args = argparse.Namespace(**vars(args))
                sweep_args.local_gnn = local_gnn
                sweep_args.global_attn = global_attn
                sweep_args.lap_pe_dim = lap_dim
                sweep_args.rwse_dim = rwse_dim
                sweep_args.lap_pe_encoder = pe_enc
                sweep_args.rwse_encoder = se_enc
                sweep_args.history_tag = f"{local_gnn}_{global_attn}_lap{lap_dim}_rwse{rwse_dim}"
                sweep_args.out_dir, run_id = make_run_output_dir(
                    mode_out_dir,
                    local_gnn,
                    global_attn,
                    sweep_args.history_tag,
                )
                print(
                    f"[sweep-node] local={local_gnn} global={global_attn} lap={lap_dim}/{pe_enc} rwse={rwse_dim}/{se_enc}"
                )
                summary = run_node_experiment(sweep_args, device, run_id, sweep_args.history_tag)
                run_summary_path = os.path.join(sweep_args.out_dir, f"gps_summary_{run_id}.csv")
                with open(run_summary_path, "w", encoding="utf-8") as handle:
                    header = ",".join(summary.keys())
                    handle.write(header + "\n")
                    handle.write(",".join(f"{summary[k]:.6f}" for k in summary.keys()) + "\n")
                row = {
                    "local_gnn": local_gnn,
                    "global_attn": global_attn,
                    "lap_pe_dim": lap_dim,
                    "lap_pe_encoder": pe_enc,
                    "rwse_dim": rwse_dim,
                    "rwse_encoder": se_enc,
                    "out_dir": sweep_args.out_dir,
                }
                row.update(summary)
                rows.append(row)

            if rows:
                with open(summary_path, "w", encoding="utf-8") as handle:
                    keys = list(rows[0].keys())
                    handle.write(",".join(keys) + "\n")
                    for row in rows:
                        handle.write(",".join(str(row[k]) for k in keys) + "\n")
                print("[saved]", summary_path)
            return

        run_args = argparse.Namespace(**vars(args))
        run_args.out_dir, run_id = make_run_output_dir(
            mode_out_dir,
            run_args.local_gnn,
            run_args.global_attn,
            run_args.history_tag,
        )
        summary_path = os.path.join(run_args.out_dir, f"gps_summary_{run_id}.csv")
        summary = run_node_experiment(run_args, device, run_id, run_args.history_tag)
        with open(summary_path, "w", encoding="utf-8") as handle:
            header = ",".join(summary.keys())
            handle.write(header + "\n")
            handle.write(",".join(f"{summary[k]:.6f}" for k in summary.keys()) + "\n")
        print("[summary]", summary)
        print("[saved]", summary_path)
        return

    dataset, labels = load_graphs(
        args.path,
        lap_pe_dim=args.lap_pe_dim,
        rwse_dim=args.rwse_dim,
        add_virtual_node=args.virtual_node,
        graph_features=args.graph_features,
        edge_features=args.edge_features,
    )
    print(f"[info] Loaded {len(dataset)} graphs with labels: {np.bincount(labels)}")

    if not args.sweep:
        run_args = argparse.Namespace(**vars(args))
        run_args.out_dir, run_id = make_run_output_dir(
            mode_out_dir,
            run_args.local_gnn,
            run_args.global_attn,
            run_args.history_tag,
        )
        summary_path = os.path.join(run_args.out_dir, f"gps_summary_{run_id}.csv")
        summary = run_experiment(dataset, labels, run_args, run_id, run_args.history_tag)
        with open(summary_path, "w", encoding="utf-8") as handle:
            header = ",".join(summary.keys())
            handle.write(header + "\n")
            handle.write(",".join(f"{summary[k]:.6f}" for k in summary.keys()) + "\n")
        print("[summary]", summary)
        print("[saved]", summary_path)
        return

    local_list = parse_list(args.sweep_local, str)
    global_list = parse_list(args.sweep_global, str)
    lap_list = parse_list(args.sweep_lap, int)
    rwse_list = parse_list(args.sweep_rwse, int)
    pe_enc_list = parse_list(args.sweep_pe_encoder, str)
    se_enc_list = parse_list(args.sweep_se_encoder, str)

    sweep_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(mode_out_dir, f"gps_sweep_summary_{sweep_run_id}.csv")

    rows: List[Dict[str, Any]] = []
    for local_gnn, global_attn, lap_dim, rwse_dim, pe_enc, se_enc in product(
        local_list, global_list, lap_list, rwse_list, pe_enc_list, se_enc_list
    ):
        sweep_args = argparse.Namespace(**vars(args))
        sweep_args.local_gnn = local_gnn
        sweep_args.global_attn = global_attn
        sweep_args.lap_pe_dim = lap_dim
        sweep_args.rwse_dim = rwse_dim
        sweep_args.lap_pe_encoder = pe_enc
        sweep_args.rwse_encoder = se_enc
        sweep_args.history_tag = f"{local_gnn}_{global_attn}_lap{lap_dim}_rwse{rwse_dim}"
        sweep_args.out_dir, run_id = make_run_output_dir(
            mode_out_dir,
            local_gnn,
            global_attn,
            sweep_args.history_tag,
        )
        print(
            f"[sweep] local={local_gnn} global={global_attn} lap={lap_dim}/{pe_enc} rwse={rwse_dim}/{se_enc}"
        )
        dataset, labels = load_graphs(
            args.path,
            lap_pe_dim=lap_dim,
            rwse_dim=rwse_dim,
            add_virtual_node=args.virtual_node,
            graph_features=args.graph_features,
            edge_features=args.edge_features,
        )
        print(f"[info] Loaded {len(dataset)} graphs with labels: {np.bincount(labels)}")
        summary = run_experiment(dataset, labels, sweep_args, run_id, sweep_args.history_tag)
        run_summary_path = os.path.join(sweep_args.out_dir, f"gps_summary_{run_id}.csv")
        with open(run_summary_path, "w", encoding="utf-8") as handle:
            header = ",".join(summary.keys())
            handle.write(header + "\n")
            handle.write(",".join(f"{summary[k]:.6f}" for k in summary.keys()) + "\n")
        row = {
            "local_gnn": local_gnn,
            "global_attn": global_attn,
            "lap_pe_dim": lap_dim,
            "lap_pe_encoder": pe_enc,
            "rwse_dim": rwse_dim,
            "rwse_encoder": se_enc,
            "out_dir": sweep_args.out_dir,
        }
        row.update(summary)
        rows.append(row)

    if rows:
        with open(summary_path, "w", encoding="utf-8") as handle:
            keys = list(rows[0].keys())
            handle.write(",".join(keys) + "\n")
            for row in rows:
                handle.write(",".join(str(row[k]) for k in keys) + "\n")
        print("[saved]", summary_path)


if __name__ == "__main__":
    main()
