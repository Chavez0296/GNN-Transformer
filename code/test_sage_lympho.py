#!/usr/bin/env python3
# test_sage_lympho.py — Week-7-style GraphSAGE baseline (patched with loss + ROC)

import os
import time, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.nn import SAGEConv
import matplotlib.pyplot as plt

from shared_lympho_utils import *
from shared_lympho_utils import _plot_learning_curves

# -------------------- Config --------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS_ROOT = os.path.join(WORK_ROOT, "results", "gps_artifacts")

out_dir = os.path.join(RESULTS_ROOT, "legacy_tests", "sage")
seed = 42
data_path = os.path.join(WORK_ROOT, "data", "lymphocyte_toy_data.pkl")
epochs = 80
lr = 0.003
wd = 5e-4
patience = 40
train_batch_size = 2048
train_num_neighbors = [25, 15]
grad_clip = 1.0
use_neighbor_backend = True
use_class_weights = True
class_weight_smoothing = 0.3
use_focal_loss = False
focal_gamma = 2.0
label_smoothing = 0.05
curve_ema = 0.6
run_tag = "step3_ls05_prauc"

# -------------------- Data prep --------------------

device = setup(seed, out_dir)

x_all, y_all, edge_index = load_lympho_dataset(data_path)
tr_mask, va_mask, te_mask = stratified_node_splits(y_all)
class_weights = None
if use_class_weights:
    class_weights = class_weights_from_labels(
        y_all[tr_mask],
        smoothing=class_weight_smoothing,
    ).to(device)

# simple EDA plots (PCA etc.)
eda(out_dir, x_all[tr_mask], y_all[tr_mask], x_all, edge_index)

data = Data(
    x=torch.tensor(x_all, dtype=torch.float32),
    y=torch.tensor(y_all, dtype=torch.long),
    edge_index=torch.tensor(
        to_undirected(torch.tensor(edge_index, dtype=torch.long)),
        dtype=torch.long,
    ),
)

# -------------------- Model --------------------


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=2, p_drop=0.5, use_bn=True):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden, aggr="mean")
        self.bn1 = nn.BatchNorm1d(hidden) if use_bn else nn.Identity()
        self.drop1 = nn.Dropout(p_drop)
        self.conv2 = SAGEConv(hidden, out_dim, aggr="mean")

    def forward(self, x, ei):
        x = self.conv1(x, ei)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.drop1(x)
        x = self.conv2(x, ei)
        return x


def has_neighbor_backend():
    """Return True if NeighborLoader backends (pyg_lib or torch_sparse) are available."""
    try:
        import pyg_lib  # noqa: F401

        return True
    except Exception:
        try:
            import torch_sparse  # noqa: F401

            return True
        except Exception:
            return False


# -------------------- Full-batch training (fallback) --------------------


def train_fullbatch(model, data, tr_mask, va_mask, te_mask, class_weights=None):
    x = data.x.to(device)
    y = data.y.to(device)
    ei = data.edge_index.to(device)

    tr_idx = torch.as_tensor(np.where(tr_mask)[0], dtype=torch.long, device=device)
    va_idx = torch.as_tensor(np.where(va_mask)[0], dtype=torch.long, device=device)
    te_idx = torch.as_tensor(np.where(te_mask)[0], dtype=torch.long, device=device)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if use_focal_loss:
        crit = FocalCE(alpha=class_weights, gamma=focal_gamma, label_smoothing=label_smoothing)
    else:
        crit = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    best_val_pr_auc = -1.0
    best_state = None
    no_improve = 0

    history = {
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
        "val_pr_auc": [],
    }

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(x, ei)
        loss = crit(logits[tr_idx], y[tr_idx])
        loss.backward()
        opt.step()

        # evaluate on train + val
        model.eval()
        with torch.no_grad():
            logits = model(x, ei)
            pr_tr = logits[tr_idx].argmax(1).cpu().numpy()
            gt_tr = y[tr_idx].cpu().numpy()
            va_logits = logits[va_idx]
            pr_va = va_logits.argmax(1).cpu().numpy()
            gt_va = y[va_idx].cpu().numpy()
            va_probs = torch.softmax(va_logits, dim=1)[:, 1].cpu().numpy()

        tr_acc = accuracy_score(gt_tr, pr_tr)
        tr_f1 = f1_score(gt_tr, pr_tr, average="macro")
        va_acc = accuracy_score(gt_va, pr_va)
        va_f1 = f1_score(gt_va, pr_va, average="macro")
        try:
            va_pr_auc = average_precision_score(gt_va, va_probs)
        except ValueError:
            va_pr_auc = 0.0

        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)
        history["val_pr_auc"].append(va_pr_auc)

        if va_pr_auc > best_val_pr_auc:
            best_val_pr_auc = va_pr_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if ep % 5 == 0 or ep == 1:
            print(
                f"[SAGE-FB][{ep:03d}/{epochs}] "
                f"loss={loss.item():.4f} tr_acc={tr_acc:.4f} "
                f"va_acc={va_acc:.4f} val_f1={va_f1:.4f} "
                f"val_pr_auc={va_pr_auc:.4f}"
            )

        if no_improve >= patience:
            break

    # restore best weights
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # learning curves
    _plot_learning_curves(
        out_dir,
        history,
        "GraphSAGE (Full-batch) Learning Curves",
        f"sage_learning_curves_lymph_{run_tag}.png",
        ema=curve_ema,
    )

    # test metrics + ROC
    model.eval()
    with torch.no_grad():
        logits = model(x, ei)
        pr_te = logits[te_idx].argmax(1).cpu().numpy()
        gt_te = y[te_idx].cpu().numpy()
        probs_pos = torch.softmax(logits[te_idx], dim=1)[:, 1].cpu().numpy()

    print("\n[SAGE-FB] Test accuracy:", accuracy_score(gt_te, pr_te))
    print(
        "[SAGE-FB] Test classification report:\n",
        classification_report(gt_te, pr_te, digits=3, zero_division=0),
    )

    try:
        auc = roc_auc_score(gt_te, probs_pos)
        fpr, tpr, _ = roc_curve(gt_te, probs_pos)
        print(f"[SAGE-FB] Test ROC-AUC: {auc:.4f}")
        try:
            pr_auc = average_precision_score(gt_te, probs_pos)
            print(f"[SAGE-FB] Test PR-AUC: {pr_auc:.4f}")
        except ValueError as e:
            print("[SAGE-FB] Test PR-AUC unavailable:", e)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"GraphSAGE-FB (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("GraphSAGE ROC Curve (Test, Full-batch)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/sage_fb_roc_lymph_{run_tag}.png", dpi=150)
        plt.close()
    except ValueError as e:
        print("[SAGE-FB] ROC could not be computed:", e)


# -------------------- NeighborLoader training (main path) --------------------


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
    te_l = NeighborLoader(
        dl,
        num_neighbors=[-1, -1],
        input_nodes=seeds_te,
        batch_size=4096,
        shuffle=False,
    )

    x_full = data.x.to(device)
    y_full = data.y.to(device)
    ei_full = data.edge_index.to(device)
    va_idx = torch.as_tensor(np.where(va_mask)[0], dtype=torch.long, device=device)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if use_focal_loss:
        crit = FocalCE(alpha=class_weights, gamma=focal_gamma, label_smoothing=label_smoothing)
    else:
        crit = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    best_val_f1 = -1.0
    best_state = None
    no_improve = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_acc": [],
        "val_f1": [],
        "val_pr_auc": [],
    }
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
            out = model(batch.x, batch.edge_index)
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

        mean_loss = running_loss / max(1, total_labeled)
        history["train_loss"].append(mean_loss)
        if train_preds:
            yp_tr = np.concatenate(train_preds)
            yt_tr = np.concatenate(train_targets)
            tr_acc = accuracy_score(yt_tr, yp_tr)
            tr_f1 = f1_score(yt_tr, yp_tr, average="macro")
        else:
            tr_acc = 0.0
            tr_f1 = 0.0
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)

        t1 = time.time()
        epoch_times.append(t1 - t0)
        epoch_thrpt.append(nodes_seen / max(1e-9, t1 - t0))

        # validation
        model.eval()
        with torch.no_grad():
            logits_full = model(x_full, ei_full)
            va_logits = logits_full[va_idx]
            yp = va_logits.argmax(1).cpu().numpy()
            yt = y_full[va_idx].cpu().numpy()
            val_probs = torch.softmax(va_logits, dim=1)[:, 1].cpu().numpy()
        va_acc = accuracy_score(yt, yp)
        va_f1 = f1_score(yt, yp, average="macro")
        try:
            va_pr_auc = average_precision_score(yt, val_probs)
        except ValueError:
            va_pr_auc = 0.0

        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)
        history["val_pr_auc"].append(va_pr_auc)

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if ep % 5 == 0 or ep == 1:
            print(
                f"[SAGE][{ep:03d}/{epochs}] loss={mean_loss:.4f} "
                f"tr_acc={tr_acc:.4f} tr_f1={tr_f1:.4f} "
                f"val_acc={va_acc:.4f} val_f1={va_f1:.4f} "
                f"val_pr_auc={va_pr_auc:.4f} thrpt={epoch_thrpt[-1]:.0f}/s"
            )

        if no_improve >= patience:
            break

    # restore best weights
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # learning curves (val only; train_loss is in history if needed later)
    _plot_learning_curves(
        out_dir,
        history,
        "GraphSAGE (Neighbor) Learning Curves",
        f"sage_neighbor_learning_curves_lymph_{run_tag}.png",
        ema=curve_ema,
    )

    # test metrics + ROC
    model.eval()
    all_p, all_t, all_scores = [], [], []
    with torch.no_grad():
        for batch in te_l:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            logits = out[: batch.batch_size]
            probs_pos = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_scores.append(probs_pos)
            all_p.append(logits.argmax(1).cpu().numpy())
            all_t.append(batch.y[: batch.batch_size].cpu().numpy())

    yp = np.concatenate(all_p)
    yt = np.concatenate(all_t)
    y_score = np.concatenate(all_scores)

    print("\n[SAGE] Test accuracy:", accuracy_score(yt, yp))
    print(
        "[SAGE] Test classification report:\n",
        classification_report(yt, yp, digits=3, zero_division=0),
    )

    try:
        auc = roc_auc_score(yt, y_score)
        fpr, tpr, _ = roc_curve(yt, y_score)
        print(f"[SAGE] Test ROC-AUC: {auc:.4f}")
        try:
            pr_auc = average_precision_score(yt, y_score)
            print(f"[SAGE] Test PR-AUC: {pr_auc:.4f}")
        except ValueError as e:
            print("[SAGE] Test PR-AUC unavailable:", e)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"GraphSAGE (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("GraphSAGE ROC Curve (Test)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/sage_roc_lymph_{run_tag}.png", dpi=150)
        plt.close()
    except ValueError as e:
        print("[SAGE] ROC could not be computed:", e)


# -------------------- Run --------------------


sage = GraphSAGE(
    in_dim=data.num_node_features,
    hidden=256,
    out_dim=int(y_all.max()) + 1,
).to(device)

if use_neighbor_backend and has_neighbor_backend():
    train_neighbor(sage, data, tr_mask, va_mask, te_mask, class_weights=class_weights)
else:
    print("Neighbor sampling backend unavailable -> full-batch fallback.")
    train_fullbatch(sage, data, tr_mask, va_mask, te_mask, class_weights=class_weights)

print("[done]")
