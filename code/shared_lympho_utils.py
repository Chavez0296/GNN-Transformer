#!/usr/bin/env python3
# shared_lympho_utils.py — common utilities for lymphocyte baselines (patched)

import os, random, warnings, pickle, numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
OUTDIR     = os.path.join(WORK_ROOT, "results", "gps_artifacts", "legacy_tests")

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- optional PyG convenience (safe if PyG isn't present) ---
try:
    from torch_geometric.utils import dropout_edge as _pyg_dropout_edge
except Exception:
    _pyg_dropout_edge = None

def dropout_edge(edge_index: np.ndarray, p: float = 0.0, training: bool = True):
    """Mirror torch_geometric.utils.dropout_edge (returns numpy)."""
    if p <= 0 or not training or edge_index is None or edge_index.size == 0:
        return edge_index
    if _pyg_dropout_edge is None:
        m = edge_index.shape[1]
        keep = np.random.rand(m) > p
        return edge_index[:, keep]
    else:
        ei_t = torch.as_tensor(edge_index, dtype=torch.long)
        ei_out, _ = _pyg_dropout_edge(ei_t, p=float(p), force_undirected=False, training=training)
        return ei_out.cpu().numpy()

def setup(seed=SEED, out_dir=OUTDIR):
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, (torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try: torch.set_float32_matmul_precision('high')
    except Exception: pass
    return device

def as_numpy(x):
    if x is None: return None
    try: return x.cpu().numpy() if hasattr(x, 'cpu') else np.asarray(x)
    except Exception: return np.asarray(x)

def round_int_labels(a):
    arr = np.asarray(a)
    if arr.ndim > 1: arr = arr.reshape(-1)
    return np.round(arr).astype(int)

def get_graphs(dct):
    if 'X_graph' in dct: return dct['X_graph']
    if 'x' in dct:       return dct['x']
    if 'X' in dct:       return dct['X']
    raise ValueError("Could not find graphs under 'X_graph', 'x', or 'X'.")

def extract_fields(g):
    feats = None
    if hasattr(g, 'x'): feats = as_numpy(g.x).astype(float)
    elif isinstance(g, dict) and 'x' in g: feats = as_numpy(g['x']).astype(float)
    if feats is not None and feats.ndim != 2: feats = None

    y = None
    gy = getattr(g, 'y', None) if hasattr(g, 'y') else (g.get('y') if isinstance(g, dict) else None)
    if gy is not None:
        gy = round_int_labels(as_numpy(gy))
        if feats is not None:
            if gy.size == feats.shape[0]: y = gy
            elif gy.size == 1: y = np.full(feats.shape[0], int(gy[0]), dtype=int)

    edges = None
    if hasattr(g, 'edge_index'): edges = as_numpy(g.edge_index)
    elif isinstance(g, dict) and 'edge_index' in g: edges = as_numpy(g['edge_index'])
    if edges is not None:
        if edges.ndim == 2 and edges.shape[0] == 2: edges = edges.astype(np.int64)
        elif edges.ndim == 2 and edges.shape[1] == 2: edges = edges.T.astype(np.int64)
        else: edges = None
    return feats, y, edges

def stitch_graphs(graphs):
    xs, ys, eis = [], [], []
    node_offset = 0
    for g in graphs:
        x, y, ei = extract_fields(g)
        if x is None or x.shape[0]==0: continue
        n = x.shape[0]
        if y is None: continue
        if y.size != n:
            if y.size == 1: y = np.full(n, int(y[0]), dtype=int)
            else: continue
        xs.append(x); ys.append(y.astype(int))
        if ei is not None and ei.size: eis.append(ei + node_offset)
        node_offset += n
    if not xs: raise RuntimeError('No usable graphs with features+labels.')
    x_all = np.vstack(xs); y_all = np.concatenate(ys)
    edge_index = np.concatenate(eis, axis=1) if eis else np.zeros((2,0), dtype=np.int64)
    return x_all, y_all, edge_index

def stratified_node_splits(y_all, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=SEED):
    N = y_all.shape[0]; idx = np.arange(N)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0-train_frac), random_state=seed)
    train_idx, hold_idx = next(sss1.split(idx, y_all))
    hold_y = y_all[hold_idx]
    val_ratio = val_frac / max(1e-9, (1.0-train_frac))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0-val_ratio), random_state=seed+1)
    val_subidx, test_subidx = next(sss2.split(hold_idx, hold_y))
    train_mask = np.zeros(N, dtype=bool); train_mask[train_idx]=True
    val_mask   = np.zeros(N, dtype=bool); val_mask[hold_idx[val_subidx]]=True
    test_mask  = np.zeros(N, dtype=bool); test_mask[hold_idx[test_subidx]]=True
    return train_mask, val_mask, test_mask

# ---------- Connected-component ("leave-graphs-out") split ----------
def _connected_components_undirected(edge_index: np.ndarray, num_nodes: int):
    """Union-Find CCs on an undirected view of edge_index -> comp_id[0..N-1]."""
    parent = np.arange(num_nodes)
    rank = np.zeros(num_nodes, dtype=int)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb: return
        if rank[ra] < rank[rb]: parent[ra] = rb
        elif rank[ra] > rank[rb]: parent[rb] = ra
        else: parent[rb] = ra; rank[ra] += 1

    if edge_index is not None and edge_index.size:
        for u, v in edge_index.T:
            u = int(u); v = int(v)
            if u == v: continue
            union(u, v); union(v, u)

    for i in range(num_nodes): parent[i] = find(i)
    _, comp_id = np.unique(parent, return_inverse=True)
    return comp_id

def leave_graphs_out_splits_stratified(x_or_y,
                                       y_or_ei,
                                       edge_index=None,
                                       *,
                                       train_frac: float = None,
                                       val_frac:   float = None,   # alias supported
                                       valid_frac: float = None,   # alias supported
                                       test_frac:  float = None,
                                       seed: int = SEED):
    """
    Leave-whole-connected-components together and split *components* with
    stratification by majority class of each component.

    Accepts either:
      - (X_all, y_all, edge_index, valid_frac=..., test_frac=..., seed=...)
      - (y_all, edge_index, train_frac=..., val_frac=..., seed=...)
    and returns (train_mask, val_mask, test_mask) over nodes.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit

    # ---- resolve which signature we received, and safely convert to numpy on CPU ----
    if edge_index is None:
        # signature: (y_all, edge_index, ...)
        y_all = _to_numpy_cpu(x_or_y).reshape(-1)
        edge_index = _to_numpy_cpu(y_or_ei).astype(np.int64, copy=False)
    else:
        # signature: (X_all, y_all, edge_index, ...)
        y_all = _to_numpy_cpu(y_or_ei).reshape(-1)
        edge_index = _to_numpy_cpu(edge_index).astype(np.int64, copy=False)

    # ensure integer labels
    if not np.issubdtype(y_all.dtype, np.integer):
        y_all = y_all.astype(np.int64, copy=False)

    # ---- resolve fractions ----
    if valid_frac is not None and val_frac is None:
        val_frac = float(valid_frac)
    if train_frac is None and val_frac is None and test_frac is None:
        train_frac = TRAIN_FRAC
        val_frac   = VAL_FRAC
        test_frac  = max(0.0, 1.0 - train_frac - val_frac)
    else:
        if val_frac is None:
            val_frac = VAL_FRAC
        if train_frac is None:
            train_frac = max(0.0, 1.0 - val_frac - (test_frac if test_frac is not None else 0.0))
        if test_frac is None:
            test_frac = max(0.0, 1.0 - train_frac - val_frac)

    # normalize if needed
    s = train_frac + val_frac + test_frac
    if s > 1.0 + 1e-6:
        train_frac, val_frac, test_frac = (train_frac/s, val_frac/s, test_frac/s)

    # ---- connected components on undirected view ----
    N = int(y_all.shape[0])
    comp_id = _connected_components_undirected(edge_index, N)
    C = int(comp_id.max()) + 1

    # majority label per component (for stratification)
    n_classes = int(y_all.max()) + 1
    comp_labels = np.zeros(C, dtype=np.int64)
    for cid in range(C):
        nodes = (comp_id == cid)
        y_c = y_all[nodes]
        if y_c.size:
            comp_labels[cid] = int(np.argmax(np.bincount(y_c, minlength=n_classes)))
        else:
            comp_labels[cid] = 0

    rng_seed = int(seed)
    comp_idx = np.arange(C)

    # train vs hold-out
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - train_frac), random_state=rng_seed)
    train_c, hold_c = next(sss1.split(comp_idx, comp_labels))
    hold_labels = comp_labels[hold_c]

    # val vs test inside hold-out
    hold_total = 1.0 - train_frac
    val_ratio_within_hold = (val_frac / hold_total) if hold_total > 0 else 0.0
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_ratio_within_hold),
                                  random_state=rng_seed + 1)
    val_subc, test_subc = next(sss2.split(hold_c, hold_labels))

    train_mask = np.zeros(N, dtype=bool)
    val_mask   = np.zeros(N, dtype=bool)
    test_mask  = np.zeros(N, dtype=bool)

    for cid in comp_idx[train_c]:
        train_mask[comp_id == cid] = True
    for cid in hold_c[val_subc]:
        val_mask[comp_id == cid] = True
    for cid in hold_c[test_subc]:
        test_mask[comp_id == cid] = True

    # ---- pretty print (matches your logs) ----
    from collections import Counter
    def _counts(mask): return dict(Counter(y_all[mask].tolist()))
    print("[split] Graph-level (by connected components):")
    print(f"  train: nodes={int(train_mask.sum())} comps={len(train_c)} class_counts={_counts(train_mask)}")
    print(f"  valid: nodes={int(val_mask.sum())} comps={len(hold_c[val_subc])} class_counts={_counts(val_mask)}")
    print(f"  test:  nodes={int(test_mask.sum())} comps={len(hold_c[test_subc])} class_counts={_counts(test_mask)}")
        # return dict so callers can use splits["train"] / ["valid"] / ["test"]
    return {
        "train": train_mask,
        "valid": val_mask,
        "test":  test_mask,
    }



def _to_numpy_cpu(x):
    """Convert lists / numpy / torch (CPU or CUDA) to a host NumPy array."""
    import numpy as np
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

# ---------- Induced subgraph ----------
def induce_subgraph(x_all: np.ndarray,
                    y_all: np.ndarray,
                    edge_index: np.ndarray,
                    mask_or_idx,
                    make_contiguous: bool = True):
    """
    Build a node-induced subgraph. Returns (x_sub, y_sub, edge_index_sub, kept_idx).
    """
    N = x_all.shape[0]
    if isinstance(mask_or_idx, np.ndarray) and mask_or_idx.dtype==bool:
        idx = np.nonzero(mask_or_idx)[0]
    else:
        idx = np.asarray(mask_or_idx, dtype=int).reshape(-1)
    idx_sorted = np.sort(idx)
    x_sub = x_all[idx_sorted]
    y_sub = y_all[idx_sorted]

    if edge_index is None or edge_index.size == 0:
        return x_sub, y_sub, np.zeros((2,0), dtype=np.int64), idx_sorted

    keep = np.isin(edge_index[0], idx_sorted) & np.isin(edge_index[1], idx_sorted)
    ei = edge_index[:, keep]
    if make_contiguous:
        remap = -np.ones(N, dtype=int)
        remap[idx_sorted] = np.arange(idx_sorted.shape[0], dtype=int)
        ei = remap[ei]
    return x_sub, y_sub, ei, idx_sorted

# ---------- Class weights & focal cross-entropy ----------
def class_weights_from_labels(y: np.ndarray, smoothing: float = 0.0):
    """
    Per-class weights (inverse frequency) for CE. Optional smoothing to
    avoid extreme weights.
    """
    y = np.asarray(y).astype(int).reshape(-1)
    n_classes = int(y.max()) + 1
    counts = np.bincount(y, minlength=n_classes).astype(float)
    if smoothing > 0:
        mu = counts.mean()
        counts = (1.0 - smoothing) * counts + smoothing * mu
    eps = 1e-8
    inv = (counts.sum()) / (counts + eps)
    w = inv / inv.mean()
    return torch.as_tensor(w, dtype=torch.float32)

class FocalCE(nn.Module):
    """Multi-class focal cross-entropy with optional class weights."""
    def __init__(self, alpha=None, gamma: float = 2.0,
                 reduction: str = "mean", label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("alpha", alpha if isinstance(alpha, torch.Tensor)
                             else (torch.as_tensor(alpha, dtype=torch.float32) if alpha is not None else None))
        self.gamma = float(gamma)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        ce = F.cross_entropy(logits, target, weight=self.alpha,
                             reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":  return loss.mean()
        if self.reduction == "sum":   return loss.sum()
        return loss

# ---------- Existing loader + quick EDA ----------
def load_lympho_dataset(path):
    with open(path, 'rb') as f: data = pickle.load(f)
    graphs = get_graphs(data)
    print(f'[info] Loaded {len(graphs)} graphs.')
    x_all, y_all, edge_index = stitch_graphs(graphs)
    print('[shape] X_all:', x_all.shape, '  y_all:', y_all.shape, '  edge_index:', edge_index.shape)
    return x_all, y_all, edge_index

def eda(out_dir, X_train, y_train, x_all, edge_index):
    os.makedirs(out_dir, exist_ok=True)
    # class distribution
    n_classes = int(np.max(y_train))+1
    counts = np.bincount(y_train, minlength=n_classes)
    plt.figure(figsize=(10,4)); plt.bar(np.arange(n_classes), counts)
    plt.title('Class Distribution (Train)'); plt.xlabel('Class'); plt.ylabel('Count')
    plt.tight_layout(); plt.savefig(f'{out_dir}/eda_class_distribution_lymph.png', dpi=150); plt.close()
    # PCA train
    pca = PCA(n_components=2, random_state=42); pca.fit(X_train)
    Z = pca.transform(X_train); labs = y_train
    plt.figure(figsize=(8,6)); sc = plt.scatter(Z[:,0], Z[:,1], c=labs, s=6, alpha=0.85, cmap='tab20')
    plt.title('PCA (2D) — Train'); plt.xlabel('PC1'); plt.ylabel('PC2'); plt.colorbar(sc, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(f'{out_dir}/pca_train_lymph.png', dpi=150); plt.close()
    # degree histogram
    dst = edge_index[1] if edge_index.size else np.array([], dtype=int)
    if dst.size: deg = np.bincount(dst, minlength=int(dst.max())+1)
    else: deg = np.zeros(x_all.shape[0], dtype=int)
    deg = deg[:x_all.shape[0]]
    plt.figure(figsize=(8,5))
    bins = np.arange(0, np.percentile(deg, 99)+2) if deg.size else 10
    if (isinstance(bins, np.ndarray) and len(bins)<2) or (not isinstance(bins, np.ndarray)):
        bins = max(10, int(deg.max())+2) if deg.size else 10
    plt.hist(deg, bins=bins, edgecolor='k', alpha=0.8)
    plt.xlabel('In-degree'); plt.ylabel('Nodes'); plt.title('Graph In-degree Histogram (trim @99th pct)')
    plt.tight_layout(); plt.savefig(f'{out_dir}/degree_histogram_lymph.png', dpi=150); plt.close()

def _plot_learning_curves(out_dir, history, title, fname, ema: float = 0.0):
    def _ema(series, alpha: float):
        if not series:
            return series
        out = [series[0]]
        for v in series[1:]:
            out.append(alpha * v + (1.0 - alpha) * out[-1])
        return out

    def _maybe_smooth(series):
        if ema and ema > 0 and len(series) > 1:
            return _ema(series, float(ema))
        return series

    epochs = list(range(1, len(history.get('val_f1', []))+1))
    plt.figure(figsize=(9,5))
    if history.get('train_acc'):
        plt.plot(epochs, _maybe_smooth(history['train_acc']), marker='o', label='Train Acc')
    if history.get('val_acc'):
        plt.plot(epochs, _maybe_smooth(history['val_acc']), marker='o', label='Val Acc')
    if history.get('train_f1'):
        plt.plot(epochs, _maybe_smooth(history['train_f1']), marker='o', label='Train F1')
    if history.get('val_f1'):
        plt.plot(epochs, _maybe_smooth(history['val_f1']), marker='o', label='Val F1')
    if history.get('val_pr_auc'):
        plt.plot(epochs, _maybe_smooth(history['val_pr_auc']), marker='o', label='Val PR-AUC')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(f"{out_dir}/{fname}", dpi=150); plt.close()
