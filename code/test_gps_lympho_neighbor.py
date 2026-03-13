#!/usr/bin/env python3
# test_gps_lympho_neighbor.py - GPSConv neighbor-sampling baseline

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score, roc_curve
from torch_geometric.data import Data
from torch_geometric.nn import GPSConv, SAGEConv
from torch_geometric.utils import to_undirected

from shared_lympho_utils import (
    _plot_learning_curves,
    class_weights_from_labels,
    load_lympho_dataset,
    setup,
    stratified_node_splits,
)

# -------------------- Config --------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS_ROOT = os.path.join(WORK_ROOT, "results", "gps_artifacts")

out_dir = os.path.join(RESULTS_ROOT, "legacy_tests", "gps_neighbor")
seed = 42
data_path = os.path.join(WORK_ROOT, "data", "lymphocyte_toy_data.pkl")
epochs = 80
lr = 0.003
wd = 5e-4
patience = 40
hidden = 256
heads = 4
dropout = 0.5
train_batch_size = 2048
train_num_neighbors = [25, 15]
grad_clip = 1.0
use_class_weights = True
class_weight_smoothing = 0.3
label_smoothing = 0.05
curve_ema = 0.6
sanity_label_shuffle = True
repeat_runs = 3

# -------------------- Model --------------------


class GPSNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4, p_drop=0.5):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.drop1 = nn.Dropout(p_drop)
        self.gps1 = GPSConv(
            hidden_dim,
            conv=SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
            heads=heads,
            dropout=p_drop,
            attn_type="multihead",
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.drop2 = nn.Dropout(p_drop)
        self.gps2 = GPSConv(
            hidden_dim,
            conv=SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
            heads=heads,
            dropout=p_drop,
            attn_type="multihead",
        )
        self.lin_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch=None):
        x = self.lin_in(x)
        x = torch.relu(self.bn1(x))
        x = self.drop1(x)
        x = self.gps1(x, edge_index, batch=batch)
        x = torch.relu(self.bn2(x))
        x = self.drop2(x)
        x = self.gps2(x, edge_index, batch=batch)
        x = self.lin_out(x)
        return x


def plot_roc(out_dir, fpr, tpr, auc, fname):
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"GPS (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("GPS ROC Curve (Test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{fname}", dpi=150)
    plt.close()


def train_neighbor(model, data, tr_mask, va_mask, te_mask, class_weights=None):
    from torch_geometric.loader import NeighborLoader

    dl = Data(x=data.x, y=data.y, edge_index=data.edge_index)

    seeds_tr = torch.as_tensor(np.where(tr_mask)[0], dtype=torch.long)
    seeds_va = torch.as_tensor(np.where(va_mask)[0], dtype=torch.long)
    seeds_te = torch.as_tensor(np.where(te_mask)[0], dtype=torch.long)

    tl = NeighborLoader(
        dl,
        num_neighbors=train_num_neighbors,
        input_nodes=seeds_tr,
        batch_size=train_batch_size,
        shuffle=True,
    )
    vl = NeighborLoader(
        dl,
        num_neighbors=[-1, -1],
        input_nodes=seeds_va,
        batch_size=4096,
        shuffle=False,
    )
    te_l = NeighborLoader(
        dl,
        num_neighbors=[-1, -1],
        input_nodes=seeds_te,
        batch_size=4096,
        shuffle=False,
    )

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    best_val_f1 = -1.0
    best_state = None
    no_improve = 0

    history = {"train_acc": [], "val_acc": [], "train_f1": [], "val_f1": []}
    epoch_times = []
    epoch_thrpt = []

    print("Backend: NeighborLoader")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        nodes_seen = 0
        running_loss = 0.0
        total_labeled = 0
        train_preds = []
        train_targets = []

        for batch in tl:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            batch_ids = batch.batch if hasattr(batch, "batch") else None
            out = model(batch.x, batch.edge_index, batch=batch_ids)
            logits = out[: batch.batch_size]
            loss = crit(logits, batch.y[: batch.batch_size])
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            bsz = int(batch.batch_size)
            nodes_seen += bsz
            running_loss += loss.item() * bsz
            total_labeled += bsz
            train_preds.append(logits.argmax(1).detach().cpu().numpy())
            train_targets.append(batch.y[: batch.batch_size].cpu().numpy())

        if train_preds:
            yp_tr = np.concatenate(train_preds)
            yt_tr = np.concatenate(train_targets)
            tr_acc = accuracy_score(yt_tr, yp_tr)
            tr_f1 = f1_score(yt_tr, yp_tr, average="macro")
        else:
            tr_acc = 0.0
            tr_f1 = 0.0

        t1 = time.time()
        epoch_times.append(t1 - t0)
        epoch_thrpt.append(nodes_seen / max(1e-9, t1 - t0))

        # validation (full neighbors)
        model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for batch in vl:
                batch = batch.to(device)
                batch_ids = batch.batch if hasattr(batch, "batch") else None
                out = model(batch.x, batch.edge_index, batch=batch_ids)
                logits = out[: batch.batch_size]
                all_p.append(logits.argmax(1).cpu().numpy())
                all_t.append(batch.y[: batch.batch_size].cpu().numpy())
        yp = np.concatenate(all_p)
        yt = np.concatenate(all_t)
        va_acc = accuracy_score(yt, yp)
        va_f1 = f1_score(yt, yp, average="macro")

        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if ep % 5 == 0 or ep == 1:
            print(
                f"[GPS][{ep:03d}/{epochs}] loss={running_loss / max(1, total_labeled):.4f} "
                f"tr_acc={tr_acc:.4f} tr_f1={tr_f1:.4f} "
                f"val_acc={va_acc:.4f} val_f1={va_f1:.4f} "
                f"thrpt={epoch_thrpt[-1]:.0f}/s"
            )

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    _plot_learning_curves(
        out_dir,
        history,
        "GPS (Neighbor) Learning Curves",
        "gps_neighbor_learning_curves_lymph.png",
        ema=curve_ema,
    )

    # test metrics
    model.eval()
    all_p, all_t, all_scores = [], [], []
    with torch.no_grad():
        for batch in te_l:
            batch = batch.to(device)
            batch_ids = batch.batch if hasattr(batch, "batch") else None
            out = model(batch.x, batch.edge_index, batch=batch_ids)
            logits = out[: batch.batch_size]
            probs_pos = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_scores.append(probs_pos)
            all_p.append(logits.argmax(1).cpu().numpy())
            all_t.append(batch.y[: batch.batch_size].cpu().numpy())

    yp = np.concatenate(all_p)
    yt = np.concatenate(all_t)
    y_score = np.concatenate(all_scores)

    print("\n[GPS] Test accuracy:", accuracy_score(yt, yp))
    print(
        "[GPS] Test classification report:\n",
        classification_report(yt, yp, digits=3, zero_division=0),
    )

    try:
        auc = roc_auc_score(yt, y_score)
        fpr, tpr, _ = roc_curve(yt, y_score)
        print(f"[GPS] Test ROC-AUC: {auc:.4f}")
        plot_roc(out_dir, fpr, tpr, auc, "gps_roc_lymph.png")
    except ValueError as e:
        print("[GPS] ROC could not be computed:", e)


# -------------------- Run --------------------

device = setup(seed, out_dir)

x_all, y_all, edge_index = load_lympho_dataset(data_path)
tr_mask, va_mask, te_mask = stratified_node_splits(y_all)

# -------------------- Sanity checks --------------------

def _counts(mask):
    counts = np.bincount(y_all[mask], minlength=int(y_all.max()) + 1)
    return {i: int(c) for i, c in enumerate(counts)}

overlap_tr_va = int(np.logical_and(tr_mask, va_mask).sum())
overlap_tr_te = int(np.logical_and(tr_mask, te_mask).sum())
overlap_va_te = int(np.logical_and(va_mask, te_mask).sum())

print("[split] sizes:", int(tr_mask.sum()), int(va_mask.sum()), int(te_mask.sum()))
print("[split] overlaps:", overlap_tr_va, overlap_tr_te, overlap_va_te)
print("[split] train label counts:", _counts(tr_mask))
print("[split] val label counts:", _counts(va_mask))
print("[split] test label counts:", _counts(te_mask))

class_weights = None
if use_class_weights:
    class_weights = class_weights_from_labels(
        y_all[tr_mask],
        smoothing=class_weight_smoothing,
    ).to(device)

data = Data(
    x=torch.tensor(x_all, dtype=torch.float32),
    y=torch.tensor(y_all, dtype=torch.long),
    edge_index=torch.tensor(
        to_undirected(torch.tensor(edge_index, dtype=torch.long)),
        dtype=torch.long,
    ),
)

def run_once(run_tag, y_override=None):
    if y_override is not None:
        y_tensor = torch.tensor(y_override, dtype=torch.long)
    else:
        y_tensor = torch.tensor(y_all, dtype=torch.long)

    local_data = Data(
        x=torch.tensor(x_all, dtype=torch.float32),
        y=y_tensor,
        edge_index=torch.tensor(
            to_undirected(torch.tensor(edge_index, dtype=torch.long)),
            dtype=torch.long,
        ),
    )

    local_model = GPSNet(
        in_dim=local_data.num_node_features,
        hidden_dim=hidden,
        out_dim=int(y_all.max()) + 1,
        heads=heads,
        p_drop=dropout,
    ).to(device)

    print(f"\n[run] {run_tag}")
    train_neighbor(local_model, local_data, tr_mask, va_mask, te_mask, class_weights=class_weights)


if sanity_label_shuffle:
    rng = np.random.default_rng(seed)
    y_shuffled = y_all.copy()
    y_shuffled[tr_mask] = rng.permutation(y_shuffled[tr_mask])
    run_once("label_shuffle_train_only", y_override=y_shuffled)

for i in range(repeat_runs):
    torch.manual_seed(seed + i)
    np.random.seed(seed + i)
    run_once(f"repeat_{i+1}")

print("[done]")
