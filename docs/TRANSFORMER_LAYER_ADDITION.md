# Transformer/GPS Plan (Generalized)

This plan explains how to apply the GPS recipe (local GNN + global attention + PE/SE) to any graph dataset. It is written for readers new to graph transformers.

## What you are doing (plain language)
You will train models that read graphs (nodes + edges) and predict a label. The GPS recipe combines:
- A local GNN (learns from nearby neighbors)
- A global attention layer (lets every node see every other node)
- Positional/structural encodings (extra signals about graph structure)

## Inputs you need
- A dataset of graphs with node features, edges, and graph-level labels.
- A training script that supports:
  - Local GNNs (e.g., SAGE/GIN/PNA/TransformerConv)
  - Global attention (e.g., multihead, performer, graphormer bias)
  - PE/SE options (LapPE, RWSE)

## Quick glossary
- GNN: graph neural network (local message passing)
- GPS: “General, Powerful, Scalable” graph transformer recipe
- LapPE: Laplacian positional encodings (global structure signal)
- RWSE: random-walk structural encodings (local-global signal)
- F1 macro: average F1 across classes (good for imbalance)

## Step 1: Inspect the dataset
Check:
- Number of graphs
- Label distribution
- Node feature dimension
- Typical graph sizes (nodes/edges)

If your project has a helper script, run it. Otherwise, write a small loader that prints these stats.

## Step 2: Establish a baseline
Run a simple local GNN with no global attention or PE/SE. This sets a reference point.
Example pattern:
```bash
python gnn_transformer_lympho.py --path <DATASET_PATH>   --local-gnn sage --global-attn none --lap-pe-dim 0 --rwse-dim 0
```

## Step 3: Add global attention
Enable a global attention mechanism and compare to the baseline.
Example pattern:
```bash
python gnn_transformer_lympho.py --path <DATASET_PATH>   --local-gnn sage --global-attn multihead --lap-pe-dim 0 --rwse-dim 0
```
If memory is limited, try performer attention instead of multihead.

## Step 4: Add positional/structural encodings
Test LapPE and RWSE separately, then together.
Example pattern:
```bash
# LapPE only
python gnn_transformer_lympho.py --path <DATASET_PATH>   --local-gnn sage --global-attn multihead --lap-pe-dim 8 --rwse-dim 0

# RWSE only
python gnn_transformer_lympho.py --path <DATASET_PATH>   --local-gnn sage --global-attn multihead --lap-pe-dim 0 --rwse-dim 8

# Both
python gnn_transformer_lympho.py --path <DATASET_PATH>   --local-gnn sage --global-attn multihead --lap-pe-dim 8 --rwse-dim 8
```

## Step 5: Run controlled sweeps
Run small grids to compare methods without exploding runtime.
Recommended starting sweep:
- Local GNNs: sage, gin, pna
- Global attention: performer or multihead
- LapPE: 0, 8
- RWSE: 0, 8

Example sweep pattern:
```bash
python gnn_transformer_lympho.py --path <DATASET_PATH> --sweep   --sweep-local sage,gin,pna --sweep-global performer --sweep-lap 0,8 --sweep-rwse 0,8   --sweep-pe-encoder raw --sweep-se-encoder raw
```

## Step 6: Tune the best config
Pick the best model by macro-F1 (or your chosen metric) and tune:
- Learning rate
- Weight decay
- Dropout

Example pattern:
```bash
python gnn_transformer_lympho.py --path <DATASET_PATH>   --local-gnn <BEST_LOCAL> --global-attn <BEST_GLOBAL>   --lap-pe-dim <BEST_LAP> --rwse-dim <BEST_RWSE>   --lr <LR> --weight-decay <WD> --dropout <DROPOUT>
```

## How to read results
Each summary CSV should include:
- accuracy_mean
- f1_macro_mean (primary for imbalanced data)
- roc_auc_mean (secondary metric)

## Common warnings
- Missing torch-scatter: affects speed, not correctness.
- NumPy deprecation warnings: safe to ignore for now.

## What to do next
- If F1 is unstable, try:
  - More dropout
  - Lower LR
  - Feature normalization
- If attention is too slow, switch from multihead to performer.
- If PE/SE helps only a little, keep the smallest dimension that improves results.
