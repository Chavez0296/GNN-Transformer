# GPS Lymphocyte Results: What Was Done and Why

This document explains how the final strong GPS results were achieved, in plain language, with enough detail to reproduce them.

---

## 1) Goal (Simple Explanation)
We want a model that can classify lymphocyte data using graphs. A graph is a set of nodes (data points) and edges (relationships). The model learns patterns in this graph to predict a class label.

We tested two families:
- **GraphSAGE (GNN baseline)**: a standard graph neural network.
- **GPS (Graph Transformer / GPSConv)**: mixes local GNN message passing with global attention.

The goal was to see whether adding a GPS-style transformer layer actually improves results.

---

## 2) Data Used
- File: `lymphocyte_toy_data.pkl`
- The data contains 75 graphs. For node-level training, these graphs are **stitched** into one big graph using `shared_lympho_utils.load_lympho_dataset`.
- This creates:
  - **Node features**: matrix shape ~ (7080, 1024)
  - **Labels**: 7080 node labels
  - **Edges**: ~63k edges

The stitched graph is converted to **undirected** edges before training so message passing is symmetric.

---

## 3) Key Concepts (Beginner-Friendly)

### A) Graph Neural Networks (GNN)
GNNs learn by passing information from each node to its neighbors. A node updates its representation based on the representations of connected nodes.

### B) GPS (Graph Transformer / GPSConv)
GPS combines two ideas:
1. **Local message passing** (like GraphSAGE): look at neighbors.
2. **Global attention**: allow nodes to attend to other nodes beyond immediate neighbors.

The GPS layer is a "hybrid" that uses both local and global steps.

### C) Neighbor Sampling
Instead of using the full giant graph each training step, we sample a neighborhood around a set of nodes. This speeds up training and makes it scalable.

### D) Metrics
We track:
- **Accuracy**: percent of correct predictions.
- **Macro F1**: average F1 across classes (good for imbalanced data).
- **ROC-AUC**: measures ranking quality (how well positives rank above negatives).
- **PR-AUC**: precision-recall area, often best for imbalanced classes.

---

## 4) Baseline Progression (What We Tried First)

### Step 1: GraphSAGE baseline (neighbor sampling)
- File: `test_sage_lympho.py`
- Learned quickly, but validation F1 was spiky.

### Step 2: Improve stability
- Added training metrics and smoothing.
- Switched validation evaluation to full-graph (more stable).

### Step 3: CV + PR-AUC tracking + threshold tuning
- File: `test_sage_lympho_cv_prauc.py`
- Used **cross-validation**, early stopping on **PR-AUC**, and tuned classification threshold.
- Result: strong, stable performance.

This gave a reliable GNN baseline to compare with GPS.

---

## 5) GPS Baselines and Sanity Checks

### Step 1: GPS neighbor baseline
- File: `test_gps_lympho_neighbor.py`
- Very strong results (near-perfect), so we verified it wasn’t due to leakage.

### Step 2: Sanity checks
- **Split overlap checks**: confirmed train/val/test masks were disjoint.
- **Label shuffle control**: shuffled train labels to break signal.
  - Result dropped to near-chance ROC-AUC ~0.5.
  - This confirmed the strong results weren’t due to leakage.

### Step 3: GPS CV baseline
- File: `test_gps_lympho_cv_prauc.py`
- Used the same CV + PR-AUC pipeline as the GraphSAGE baseline.
- GPS performed extremely well.

---

## 6) Final Integration Into gnn_transformer_lympho.py
To make the GPS node-level pipeline reproducible inside the main script, we added:

1. **Node-level CV mode**
   - Runs on the stitched graph (node-level), not graph-level.
   - Uses NeighborLoader for training.
   - Uses full-graph validation with PR-AUC early stopping.

2. **GPSNet-style node model**
   - A simpler GPSConv network matching the proven standalone script.

3. **New preset: `lympho_node`**
   - Pre-configures the model and training settings to reproduce the strong results.

### Run command
```
python gnn_transformer_lympho.py --gps-preset lympho_node --out-dir "artifacts/gps_lympho_node"
```

### Output summary file (latest run)
- `artifacts/gps_lympho_node/gps_summary_20260203_213816.csv`

### Final results (mean over folds)
- Accuracy: **0.9874**
- Macro-F1: **0.9864**
- ROC-AUC: **0.99999**
- PR-AUC: **0.99998**

These are the best numbers achieved, and they are consistent across folds.

---

## 7) Why This Works
- **GPSConv is strong**: it mixes local neighborhood structure with global attention.
- **PR-AUC early stopping**: more stable than F1 when classes are imbalanced.
- **Threshold tuning**: improves classification metrics when the model’s probability calibration is imperfect.
- **Undirected edges**: avoids directional bias and matches common GNN assumptions.

---

## 8) Files Added/Modified

New or key scripts:
- `test_sage_lympho_cv_prauc.py` (GraphSAGE CV baseline)
- `test_gps_lympho_neighbor.py` (GPS neighbor baseline + sanity checks)
- `test_gps_lympho_cv_prauc.py` (GPS CV baseline)
- `gnn_transformer_lympho.py` (node-level GPS pipeline + presets)

Support utilities:
- `shared_lympho_utils.py` (used for stitching and class weights)

---

## 9) Repro Checklist

To reproduce the final GPS results:
1. Confirm `lymphocyte_toy_data.pkl` is in the repo.
2. Run:
   ```
   python gnn_transformer_lympho.py --gps-preset lympho_node --out-dir "artifacts/gps_lympho_node"
   ```
3. Check the summary CSV in the output directory.

Optional (sanity check):
- Run label-shuffle control:
  - `test_gps_lympho_neighbor.py` with `sanity_label_shuffle=True` (already in script).

---

## 10) Final Takeaway
Yes, **GPS improves the GNN** in this node-level setting. With the matched CV pipeline, GPS achieves near-perfect ranking metrics and very strong classification metrics. GraphSAGE is still strong, but GPS with the node-level pipeline is best overall.
