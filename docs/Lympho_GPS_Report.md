# Lymphocyte Detection in Tissue Slides (GPS Node-Level Report)

## Abstract
This report explains how we achieved strong lymphocyte classification results using a GPS-style graph transformer. We walk through the dataset, the graph setup, the training pipeline, and the exact changes that made the final model reliable. The explanation is written for readers with little or no background in graph neural networks.

---

## 1) Problem Overview (Plain Language)
We want to label cells in tissue images as one of two classes. Each cell is a node in a graph, and edges describe which cells are connected (neighbors). A graph neural network (GNN) can learn patterns by letting each node look at its neighbors. A GPS model improves this by adding global attention so nodes can also use information from farther away.

---

## 2) Dataset and Graph Construction
**Data source**: `lymphocyte_toy_data.pkl`

There are **75 graphs**. For node-level training we merge ("stitch") them into one large graph:
- Node features: **7080 nodes x 1024 features**
- Labels: **7080 node labels**
- Edges: **~63k edges**

We convert the edges to **undirected** so information flows both ways.

This stitched graph is produced by:
- `shared_lympho_utils.load_lympho_dataset`

---

## 3) Key Concepts (Beginner-Friendly)

### Graph Neural Network (GNN)
A GNN updates each node by mixing information from its neighbors. This is like a social network where each person learns from their friends.

### GPS (Graph Transformer)
GPS combines two ideas:
1. **Local message passing** (neighbor information, like GraphSAGE)
2. **Global attention** (ability to pay attention to nodes beyond immediate neighbors)

### How a GPS layer works (step-by-step)
In each GPS layer, every node is updated in two parallel ways:
- **Local branch (SAGEConv):** aggregates neighbor features and updates the node with nearby context.
- **Global branch (multi-head attention):** computes attention scores so a node can weigh information from farther nodes.
- **Fusion:** combines local and global outputs, then applies normalization/dropout before the next layer.

Why this helps:
- Local branch captures short-range tissue structure.
- Global branch captures longer-range context and non-local dependencies.
- Combining both usually improves representation quality versus local-only models.

### Important terms for beginners
- **Head (attention head):** an independent attention channel; multiple heads let the model learn different interaction patterns.
- **Neighbor sampling:** train-time subgraph sampling for speed/memory efficiency.
- **Early stopping:** stop training when validation quality stops improving.
- **Threshold tuning:** convert probabilities to class labels using a validation-selected cutoff (instead of fixed 0.5).

### Neighbor Sampling
Instead of using the full graph every step, we train on small sampled neighborhoods. This is faster and scales better.

### Metrics
- **Accuracy**: percent of correct predictions.
- **Macro F1**: average F1 across classes (important when classes are imbalanced).
- **ROC-AUC**: ranking quality (how well positives rank above negatives).
- **PR-AUC**: precision-recall area (often best for imbalanced data).

---

## 4) Baselines and Progression

### 4.1 GraphSAGE baseline (node-level)
We started with a strong GraphSAGE baseline, then improved it with:
- Full-graph validation for stability
- PR-AUC tracking
- Cross-validation (CV)
- Per-fold threshold tuning

**Result (CV, threshold-tuned):**
- accuracy_mean 0.960593
- f1_macro_mean 0.954567
- roc_auc_mean 0.994345
- pr_auc_mean 0.990779

Source: `artifacts/sage_cv_prauc/run_20260203_201401/sage_cv_summary_20260203_201401.csv`

### 4.2 GPS baseline (neighbor sampling)
We built a GPSConv neighbor model and confirmed it was not leaking data:
- **Mask overlap check**: train/val/test masks were disjoint.
- **Label shuffle control**: ROC-AUC dropped to chance (~0.5).
- **Repeat runs**: metrics stayed very high.

Source script: `test_gps_lympho_neighbor.py`

### 4.3 GPS CV baseline
We matched the GraphSAGE CV pipeline and ran GPS with the same process:
- CV splits
- PR-AUC early stopping
- Threshold tuning

**Result (CV):**
- accuracy_mean 0.946610
- f1_macro_mean 0.945422
- roc_auc_mean 0.999981
- pr_auc_mean 0.999962

Source: `artifacts/gps_cv_prauc/run_20260203_204705/gps_cv_summary_20260203_204705.csv`

---

## 5) Final GPS Node-Level Pipeline (Best Results)
We integrated the node-level GPS pipeline into the main script and created a preset that reproduces the best results.

### 5.1 Where it lives
Main script: `gnn_transformer_lympho.py`

New preset:
```
--gps-preset lympho_node
```

### 5.2 Why this works
- **GPSConv + Neighbor Sampling** captures both local and global signals.
- **PR-AUC early stopping** is stable under class imbalance.
- **Threshold tuning** improves classification metrics.
- **Class-weight smoothing** avoids over-penalizing the minority class.

### 5.3 Command
```
python gnn_transformer_lympho.py --gps-preset lympho_node --out-dir "artifacts/gps_lympho_node"
```

### 5.4 Final results (mean over CV folds)
- accuracy_mean **0.987429**
- f1_macro_mean **0.986413**
- roc_auc_mean **0.999991**
- pr_auc_mean **0.999982**

Source: `artifacts/gps_lympho_node/gps_summary_20260203_213816.csv`

---

## 6) Learning Curves and ROC Curves
The run automatically saves aggregated curves:
- Learning curves: `artifacts/gps_lympho_node/learning_curves_20260203_213816_lympho_node.png`
- ROC-AUC curves: `artifacts/gps_lympho_node/roc_auc_curves_20260203_213816_lympho_node.png`

These plots show the model converges quickly and maintains strong validation ranking.

---

## 7) Files and Scripts Used

**Core GPS scripts**
- `gnn_transformer_lympho.py` (final node-level pipeline)
- `test_gps_lympho_neighbor.py` (neighbor baseline + sanity checks)
- `test_gps_lympho_cv_prauc.py` (GPS CV baseline)

**Core GraphSAGE scripts**
- `test_sage_lympho.py` (neighbor baseline)
- `test_sage_lympho_cv_prauc.py` (GraphSAGE CV baseline)

**Shared utilities**
- `shared_lympho_utils.py` (stitching, class weights)

---

## 8) Reproducibility Checklist
1. Ensure `lymphocyte_toy_data.pkl` exists in the repo.
2. Run the GPS node preset command (Section 5.3).
3. Check the summary CSV for metrics.

---

## 9) Final Takeaway
The GPS node-level pipeline is currently the strongest and most reliable model in this project. It improves over the GraphSAGE baseline in both classification metrics and ranking metrics when evaluated with the same CV and threshold tuning procedure.

If you want even stronger guarantees, we can repeat the CV with multiple random seeds and report mean/std across seeds.
