#!/usr/bin/env python3
# test_gps_lympho_cv_prauc.py - GPSConv CV with PR-AUC early stopping

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch_geometric.data import Data
from torch_geometric.nn import GPSConv, SAGEConv
from torch_geometric.utils import to_undirected

from shared_lympho_utils import (
    _plot_learning_curves,
    class_weights_from_labels,
    load_lympho_dataset,
    setup,
)

# -------------------- Config --------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS_ROOT = os.path.join(WORK_ROOT, "results", "gps_artifacts")

out_dir = os.path.join(RESULTS_ROOT, "legacy_tests", "gps_cv_prauc")
seed = 42
data_path = os.path.join(WORK_ROOT, "data", "lymphocyte_toy_data.pkl")
epochs = 80
lr = 0.003
wd = 5e-4
patience = 40
cv_folds = 5
val_frac = 0.15
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

# -------------------- Helpers --------------------


def plot_roc_curve(out_dir, fpr, tpr, auc, title, fname):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()


def find_best_threshold(y_true, probs, *, metric="f1_macro"):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thr = 0.5
    best_score = -1.0
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        if metric == "f1_macro":
            score = f1_score(y_true, preds, average="macro", zero_division=0)
        else:
            score = accuracy_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_thr = thr
    return best_thr, best_score


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


def train_neighbor(
    model,
    data,
    tr_mask,
    va_mask,
    te_mask,
    fold_id,
    fold_dir,
    class_weights=None,
):
    from torch_geometric.loader import NeighborLoader

    dl = Data(x=data.x, y=data.y, edge_index=data.edge_index)
    seeds_tr = torch.as_tensor(np.where(tr_mask)[0], dtype=torch.long)

    tl = NeighborLoader(
        dl,
        num_neighbors=train_num_neighbors,
        input_nodes=seeds_tr,
        batch_size=train_batch_size,
        shuffle=True,
    )

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    best_val_pr_auc = -1.0
    best_state = None
    best_epoch = 0
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

    x_full = data.x.to(device)
    y_full = data.y.to(device)
    ei_full = data.edge_index.to(device)
    va_idx = torch.as_tensor(np.where(va_mask)[0], dtype=torch.long, device=device)

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

        # full-batch validation for early stopping
        model.eval()
        with torch.no_grad():
            logits_full = model(x_full, ei_full, batch=None)
            va_logits = logits_full[va_idx]
            yp = va_logits.argmax(1).cpu().numpy()
            yt = y_full[va_idx].cpu().numpy()
            va_probs = torch.softmax(va_logits, dim=1)[:, 1].cpu().numpy()

        va_acc = accuracy_score(yt, yp)
        va_f1 = f1_score(yt, yp, average="macro")
        try:
            va_pr_auc = average_precision_score(yt, va_probs)
        except ValueError:
            va_pr_auc = 0.0

        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)
        history["val_pr_auc"].append(va_pr_auc)

        if va_pr_auc > best_val_pr_auc:
            best_val_pr_auc = va_pr_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            no_improve = 0
        else:
            no_improve += 1

        if ep % 5 == 0 or ep == 1:
            print(
                f"[GPS][fold={fold_id}][{ep:03d}/{epochs}] loss={mean_loss:.4f} "
                f"tr_acc={tr_acc:.4f} tr_f1={tr_f1:.4f} "
                f"val_acc={va_acc:.4f} val_f1={va_f1:.4f} "
                f"val_pr_auc={va_pr_auc:.4f} thrpt={epoch_thrpt[-1]:.0f}/s"
            )

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # threshold tuning on full-batch validation
    model.eval()
    with torch.no_grad():
        logits_full = model(x_full, ei_full, batch=None)
        va_logits = logits_full[va_idx]
        va_probs = torch.softmax(va_logits, dim=1)[:, 1].cpu().numpy()
        yt_val = y_full[va_idx].cpu().numpy()

    best_thr, best_thr_f1 = find_best_threshold(yt_val, va_probs, metric="f1_macro")
    best_thr_acc = accuracy_score(yt_val, (va_probs >= best_thr).astype(int))

    _plot_learning_curves(
        fold_dir,
        history,
        f"GPS CV Fold {fold_id} Learning Curves",
        "gps_learning_curves.png",
        ema=curve_ema,
    )

    # test metrics on full graph
    te_idx = torch.as_tensor(np.where(te_mask)[0], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        logits_full = model(x_full, ei_full, batch=None)
        te_logits = logits_full[te_idx]
        y_score = torch.softmax(te_logits, dim=1)[:, 1].cpu().numpy()
        yp_te = (y_score >= best_thr).astype(int)
        yt_te = y_full[te_idx].cpu().numpy()

    acc = accuracy_score(yt_te, yp_te)
    f1 = f1_score(yt_te, yp_te, average="macro")
    try:
        roc_auc = roc_auc_score(yt_te, y_score)
    except ValueError:
        roc_auc = 0.0
    try:
        pr_auc = average_precision_score(yt_te, y_score)
    except ValueError:
        pr_auc = 0.0

    print(
        f"[GPS][fold={fold_id}] best_epoch={best_epoch} "
        f"val_thr={best_thr:.2f} val_thr_f1={best_thr_f1:.4f} val_thr_acc={best_thr_acc:.4f}"
    )
    print(f"[GPS][fold={fold_id}] test_acc={acc:.4f} test_f1={f1:.4f}")
    print(
        "[GPS][fold={}] Test classification report:\n".format(fold_id),
        classification_report(yt_te, yp_te, digits=3, zero_division=0),
    )
    print(f"[GPS][fold={fold_id}] Test ROC-AUC: {roc_auc:.4f}")
    print(f"[GPS][fold={fold_id}] Test PR-AUC: {pr_auc:.4f}")

    fpr, tpr, _ = roc_curve(yt_te, y_score)
    precision, recall, _ = precision_recall_curve(yt_te, y_score)
    if roc_auc > 0:
        plot_roc_curve(fold_dir, fpr, tpr, roc_auc, "GPS ROC Curve", "gps_roc.png")
    if pr_auc > 0:
        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, label=f"AP={pr_auc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("GPS PR Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, "gps_pr.png"), dpi=150)
        plt.close()

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_epoch": best_epoch,
        "best_thr": best_thr,
        "val_thr_f1": best_thr_f1,
        "val_thr_acc": best_thr_acc,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
    }


# -------------------- Run --------------------

device = setup(seed, out_dir)

x_all, y_all, edge_index = load_lympho_dataset(data_path)

data = Data(
    x=torch.tensor(x_all, dtype=torch.float32),
    y=torch.tensor(y_all, dtype=torch.long),
    edge_index=torch.tensor(
        to_undirected(torch.tensor(edge_index, dtype=torch.long)),
        dtype=torch.long,
    ),
)

run_id = time.strftime("%Y%m%d_%H%M%S")
run_root = os.path.join(out_dir, f"run_{run_id}")
os.makedirs(run_root, exist_ok=True)

skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

metrics = []
all_idx = np.arange(y_all.shape[0])
val_ratio = val_frac / (1.0 - 1.0 / cv_folds)

for fold_id, (train_val_idx, test_idx) in enumerate(skf.split(all_idx, y_all), start=1):
    fold_dir = os.path.join(run_root, f"fold_{fold_id}")
    os.makedirs(fold_dir, exist_ok=True)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed + fold_id)
    train_sub_idx, val_sub_idx = next(sss.split(train_val_idx, y_all[train_val_idx]))
    train_idx = train_val_idx[train_sub_idx]
    val_idx = train_val_idx[val_sub_idx]

    tr_mask = np.zeros(y_all.shape[0], dtype=bool)
    va_mask = np.zeros(y_all.shape[0], dtype=bool)
    te_mask = np.zeros(y_all.shape[0], dtype=bool)
    tr_mask[train_idx] = True
    va_mask[val_idx] = True
    te_mask[test_idx] = True

    class_weights = None
    if use_class_weights:
        class_weights = class_weights_from_labels(
            y_all[tr_mask],
            smoothing=class_weight_smoothing,
        ).to(device)

    torch.manual_seed(seed + fold_id)
    np.random.seed(seed + fold_id)

    model = GPSNet(
        in_dim=data.num_node_features,
        hidden_dim=hidden,
        out_dim=int(y_all.max()) + 1,
        heads=heads,
        p_drop=dropout,
    ).to(device)

    fold_metrics = train_neighbor(
        model,
        data,
        tr_mask,
        va_mask,
        te_mask,
        fold_id,
        fold_dir,
        class_weights=class_weights,
    )
    fold_metrics["fold_id"] = fold_id
    metrics.append(fold_metrics)

accs = [m["accuracy"] for m in metrics]
f1s = [m["f1_macro"] for m in metrics]
rocs = [m["roc_auc"] for m in metrics]
pras = [m["pr_auc"] for m in metrics]

summary_path = os.path.join(run_root, f"gps_cv_summary_{run_id}.csv")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("accuracy_mean,accuracy_std,f1_macro_mean,f1_macro_std,roc_auc_mean,roc_auc_std,pr_auc_mean,pr_auc_std\n")
    f.write(
        f"{np.mean(accs):.6f},{np.std(accs):.6f},"
        f"{np.mean(f1s):.6f},{np.std(f1s):.6f},"
        f"{np.mean(rocs):.6f},{np.std(rocs):.6f},"
        f"{np.mean(pras):.6f},{np.std(pras):.6f}\n"
    )

thresholds_path = os.path.join(run_root, f"gps_cv_thresholds_{run_id}.csv")
with open(thresholds_path, "w", encoding="utf-8") as f:
    f.write("fold,best_epoch,thr,val_thr_f1,val_thr_acc,test_acc,test_f1,roc_auc,pr_auc\n")
    for m in metrics:
        f.write(
            f"{m['fold_id']},{m['best_epoch']},{m['best_thr']:.2f},"
            f"{m['val_thr_f1']:.6f},{m['val_thr_acc']:.6f},"
            f"{m['accuracy']:.6f},{m['f1_macro']:.6f},"
            f"{m['roc_auc']:.6f},{m['pr_auc']:.6f}\n"
        )

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
for m in metrics:
    plt.plot(m["fpr"], m["tpr"], alpha=0.7, label=f"Fold {m['fold_id']}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("GPS CV ROC Curves")
plt.legend(fontsize=8)

plt.subplot(1, 2, 2)
for m in metrics:
    plt.plot(m["recall"], m["precision"], alpha=0.7, label=f"Fold {m['fold_id']}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("GPS CV PR Curves")
plt.legend(fontsize=8)

plt.tight_layout()
summary_plot_path = os.path.join(run_root, "gps_cv_roc_pr_summary.png")
plt.savefig(summary_plot_path, dpi=150)
plt.close()

print("[done] CV summary saved to:", summary_path)
print("[done] Thresholds saved to:", thresholds_path)
print("[done] Summary plot saved to:", summary_plot_path)
