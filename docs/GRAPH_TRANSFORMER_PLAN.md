# Graph Transformer Commands

## Basics
- `python lymph_graph_transformer.py --help`
- `python lymph_graph_transformer.py --task graph`
- `python lymph_graph_transformer.py --task node-ssl`
- `python lymph_graph_transformer.py --task both`

## Common overrides
- `python lymph_graph_transformer.py --task graph --path lymphocyte_toy_data.pkl`
- `python lymph_graph_transformer.py --task graph --out-dir artifacts/graph_transformer`
- `python lymph_graph_transformer.py --task graph --seed 123`
- `python lymph_graph_transformer.py --task graph --folds 3 --epochs 50 --patience 10`
- `python lymph_graph_transformer.py --task graph --batch-size 16 --hidden-dim 256 --heads 8 --dropout 0.2`
- `python lymph_graph_transformer.py --task graph --lr 1e-3 --weight-decay 5e-4`
- `python lymph_graph_transformer.py --task graph --disable-gps`
- `python lymph_graph_transformer.py --task graph --pool mean`
- `python lymph_graph_transformer.py --task graph --gnn gin --gnn-layers 2 --pool mean`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --pool mean`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 3 --gnn-jk sum --pool mean`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --head linear --pool mean`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --pool mean --transformer-layers 2`
- `python lymph_graph_transformer.py --task graph --gnn gatv2_gps --pool attention --transformer-layers 2`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --pool cls --head linear --transformer-layers 2 --transformer-virtual-node --disable-class-weights`
- `python lymph_graph_transformer.py --task graph --lap-pe-dim 8`
- `python lymph_graph_transformer.py --task graph --struct-features degree_and_log`
- `python lymph_graph_transformer.py --task graph --virtual-node`

## DGI overrides
- `python lymph_graph_transformer.py --task node-ssl --ssl-epochs 100 --ssl-lr 1e-3`

## Sanity checks
- `python summarize_node_embeddings.py`
- `python summarize_node_embeddings.py --path artifacts/graph_transformer/dgi_YYYYMMDD_HHMMSS/node_embeddings.pkl`

## Full example
- `python lymph_graph_transformer.py --task graph --path data/your_dataset.pkl --out-dir artifacts/my_run --folds 5 --epochs 200`

## Baseline Dataset
- Current baseline dataset: `lymphocyte_toy_data_relabel.pkl` (top-10 high-confidence corrections applied).
- Use `--path lymphocyte_toy_data.pkl` to run against the original labels.

## Methods and Rationale (Summary)

### Baselines
was- Mean-pooled LR/MLP baselines provide a sanity check for signal in raw node features; strong baselines imply the dataset contains linear signal that GNNs should match or beat.

### GNN backbones
- GATv2/GPS, GraphSAGE, GIN, and GCN explore different message-passing biases (attention, neighborhood aggregation, injective pooling). This helps identify whether oversmoothing or insufficient inductive bias is limiting performance.

### Pooling strategies
- Mean pooling and attention pooling aggregate node features into graph-level embeddings; mean pooling is a stable baseline while attention can overfit on small datasets.

### Transformer integration
- A TransformerEncoder after the GNN lets the model reweight node embeddings globally. A CLS token (virtual node) provides a dedicated graph-level representation; this can outperform pooling when global context matters.

### Graph Transformer Convolution
- TransformerConv applies attention-based message passing at the edge level, acting like a transformer inside the GNN before global pooling. This can improve expressivity when neighbor interactions are complex.

### PNA (Principal Neighbourhood Aggregation)
- PNAConv mixes multiple aggregators and scalers using the graph's degree histogram to improve robustness across neighborhoods of varying sizes.

### Random-walk structural encodings (RWSE)
- RWSE injects k-step return probabilities from a random walk as node positional features, helping attention layers distinguish structural roles.

### Graphormer bias
- Graphormer-style attention bias adds a learned embedding of shortest-path distances directly to attention logits, letting the transformer prefer structurally closer nodes.

### Two-stage projection (GNN -> MLP -> Transformer)
- A projection MLP can compress GNN embeddings before attention, reducing noise and making the transformer focus on a lower-dimensional representation.

### Virtual node (PyG transform)
- Adds a learnable global node connected to all nodes, enabling long-range information flow without deep message passing.

### Structural features
- Degree or log-degree features inject topology cues that may be absent from raw node features. This can improve generalization if structure is predictive.

### Laplacian positional encodings
- LapPE provides structural positional information for Transformers/GNNs, potentially helping with global ordering and graph structure awareness.

### Class weighting
- Class-weighted loss can help imbalance but may destabilize small data; disabling it often improves macro-F1 if the model overcompensates.

### Normalization
- Feature normalization (train-split standardization) can stabilize optimization and help Transformers/GNNs converge. GNN norms (batch/layer/pair) address oversmoothing or scale drift.

### Regularization
- Dropout and edge dropout target overfitting. Edge dropout acts as data augmentation by randomly removing edges during training.

### Balanced sampling
- WeightedRandomSampler balances class frequency per batch, which can help minority-class recall when labels are imbalanced.

### Label smoothing
- Label smoothing softens hard targets to reduce overconfidence and help generalization on noisy labels.

### Label noise audit
- Cross-validated out-of-fold predictions highlight high-confidence misclassifications and high-entropy samples that may indicate mislabeled graphs.

### Label review checklist
- Generates a ranked list of high-confidence wrong predictions with basic graph stats to guide relabeling review.

### Label override application
- Applies manual label corrections from the checklist CSV into a new dataset pickle for retraining.
- Auto-flip option: fill review_label with model predictions for the top-N high-confidence wrong cases to test label-noise impact.

### Relation to GNN_Project.pdf
- The PDF focuses on knowledge-graph link prediction (OGBL-BioKG), which differs from graph-level classification, but its baseline/diagnostic mindset maps well: use strong baselines, slice analyses by degree/graph size, and inspect high-confidence errors. Evidence-grounded reranking can be adapted into explainability subgraphs for misclassified graphs if desired.

### Next Plan (Adapted)
- Confirm dataset schema (homogeneous vs typed edges/nodes) to choose between relational GNNs vs homogeneous diagnostics.
- Run slice-based diagnostics: graph size, density, and degree buckets; compare macro-F1 across slices.
- Complete label review from the checklist, apply overrides, and rerun the best configuration.
- If desired, add an explainability step: extract k-hop subgraphs around misclassified graphs for evidence visualization.

### Per-graph feature normalization
- Normalizing each graph individually can reduce scale drift between graphs when features are not globally standardized.

### Graph-level features
- Concatenating simple graph stats (log nodes/edges, avg degree, density) can provide global context that pooled node embeddings may miss.

### Optimization
- LR/weight decay sweeps and cosine schedules test sensitivity to optimization dynamics; improvements here indicate training stability issues rather than model capacity.

## Experiment Log (2026-01-25)

### Dataset + SSL + sanity
- `python lymph_graph_transformer.py --task graph`
- `python lymph_graph_transformer.py --task graph --folds 3 --epochs 50 --patience 10`
- `python lymph_graph_transformer.py --task both`
- `python lymph_graph_transformer.py --task node-ssl`
- `python lymph_graph_transformer.py --task node-ssl --ssl-epochs 50`
- `python lymph_graph_transformer.py --task node-ssl --ssl-epochs 200`
- `python summarize_node_embeddings.py`

### Baselines
- `python baseline_graph_mean_lr.py`
- `python baseline_graph_mean_lr.py --scan-c`
- `python baseline_graph_mean_lr.py --grid`
- `python baseline_graph_mean_lr.py --grid --pool-features mean_std_max`
- `python baseline_graph_mean_mlp.py --early-stopping`

### GATv2/GPS and ablations
- `python lymph_graph_transformer.py --task graph --disable-gps --dropout 0.1`
- `python lymph_graph_transformer.py --task graph --disable-gps --pool mean --dropout 0.1`
- `python lymph_graph_transformer.py --task graph --disable-gps --hidden-dim 128 --heads 4 --dropout 0.1`
- `python lymph_graph_transformer.py --task graph --disable-gps --hidden-dim 128 --heads 4 --dropout 0.3`
- `python lymph_graph_transformer.py --task graph --disable-gps --disable-class-weights --hidden-dim 32 --heads 2 --dropout 0.1 --folds 3 --epochs 50 --patience 10`
- `python lymph_graph_transformer.py --task graph --disable-gps --disable-class-weights --hidden-dim 64 --heads 2 --dropout 0.1 --folds 3 --epochs 50 --patience 10`
- `python lymph_graph_transformer.py --task graph --disable-gps --disable-class-weights --hidden-dim 32 --heads 2 --dropout 0.1`
- `python lymph_graph_transformer.py --task graph --disable-gps --disable-class-weights --hidden-dim 64 --heads 2 --dropout 0.1`

### GNN backbones (no transformer)
- `python lymph_graph_transformer.py --task graph --gnn gin --gnn-layers 2 --hidden-dim 64 --dropout 0.1 --pool mean`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --hidden-dim 64 --dropout 0.1 --pool mean`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --hidden-dim 128 --dropout 0.0 --pool mean`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --hidden-dim 128 --dropout 0.1 --pool attention --gnn-norm layer`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 3 --gnn-jk sum --hidden-dim 128 --dropout 0.1 --pool mean`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --hidden-dim 128 --dropout 0.0 --pool mean --head linear`

### Transformer + GNN
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --hidden-dim 128 --dropout 0.1 --pool mean --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256`
- `python lymph_graph_transformer.py --task graph --gnn gatv2_gps --hidden-dim 128 --heads 4 --dropout 0.3 --pool attention --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --hidden-dim 128 --dropout 0.1 --pool mean --transformer-layers 4 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 512`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --hidden-dim 128 --dropout 0.1 --pool cls --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --hidden-dim 128 --dropout 0.1 --pool mean --transformer-layers 1 --transformer-dropout 0.0 --transformer-ffn 256`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --hidden-dim 128 --dropout 0.1 --pool mean --transformer-layers 1 --transformer-dropout 0.1 --transformer-ffn 128`
- `python lymph_graph_transformer.py --task graph --gnn gatv2 --hidden-dim 128 --heads 4 --dropout 0.3 --pool mean --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256`

### CLS + Transformer variants
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --hidden-dim 128 --dropout 0.0 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --hidden-dim 64 --dropout 0.1 --pool cls --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --hidden-dim 128 --dropout 0.0 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.0 --transformer-ffn 256 --transformer-virtual-node`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --hidden-dim 128 --dropout 0.0 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 3 --hidden-dim 128 --dropout 0.0 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-norm layer --gnn-layers 2 --hidden-dim 128 --dropout 0.0 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --hidden-dim 128 --dropout 0.0 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --hidden-dim 128 --dropout 0.0 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.0 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --hidden-dim 128 --dropout 0.0 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --hidden-dim 128 --dropout 0.0 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --hidden-dim 128 --dropout 0.0 --pool cls --head linear --transformer-layers 3 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights`
- `python lymph_graph_transformer.py --task graph --gnn gatv2_gps --hidden-dim 128 --heads 4 --dropout 0.3 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --hidden-dim 256 --dropout 0.0 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --hidden-dim 128 --dropout 0.0 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights --struct-features degree_and_log`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --hidden-dim 128 --dropout 0.0 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights --virtual-node`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --hidden-dim 128 --dropout 0.0 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights --lap-pe-dim 8`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 128 --dropout 0.0 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 128 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 128 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.0 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 128 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.2 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 128 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights --lr 1e-3 --weight-decay 0`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 128 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights --lr 1e-3 --weight-decay 5e-5`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 128 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights --lr-schedule cosine --warmup-epochs 5 --min-lr 2e-4`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 128 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights --cls-mlp residual`
- `python lymph_graph_transformer.py --task graph --gnn gin --gnn-layers 1 --hidden-dim 128 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --gnn-norm batch --hidden-dim 128 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights --cls-residual mean`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --gnn-norm pair --hidden-dim 128 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --gnn-norm none --hidden-dim 128 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --gnn-norm layer --hidden-dim 128 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 128 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights --edge-dropout 0.2`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 128 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights --cls-proj-dim 256`
- `python lymph_graph_transformer.py --task graph --gnn gin --gnn-layers 2 --hidden-dim 128 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 128 --dropout 0.1 --pool cls --head mlp --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 128 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights --feature-norm standard`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 128 --dropout 0.1 --pool cls --head mlp --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights --feature-norm standard`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 128 --dropout 0.1 --pool cls --head linear --transformer-layers 3 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights --feature-norm standard`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 128 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights --feature-norm standard --cls-proj-dim 256`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 192 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 384 --transformer-virtual-node --disable-class-weights --feature-norm standard`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 3 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.0 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.2 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --feature-norm standard`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1e-3`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 3e-3`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 5e-4`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3 --weight-decay 0`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3 --weight-decay 5e-5`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3 --weight-decay 5e-4`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.0 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 384 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 4 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 1 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.05 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 2 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3`
- `python lymph_graph_transformer.py --task graph --gnn gatv2 --hidden-dim 256 --heads 8 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3`
- `python lymph_graph_transformer.py --task graph --gnn transformerconv --gnn-layers 1 --hidden-dim 256 --heads 8 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3`
- `python lymph_graph_transformer.py --task graph --gnn transformerconv --gnn-layers 2 --hidden-dim 256 --heads 8 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3`
- `python lymph_graph_transformer.py --task graph --gnn pna --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3`
- `python lymph_graph_transformer.py --task graph --gnn pna --gnn-layers 2 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3 --rwse-dim 8`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3 --graphormer --graphormer-max-dist 5`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool mean --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --pre-transformer-dim 128 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 256 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3 --balanced-sampler`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3 --label-smoothing 0.1`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm per-graph --lr 1.5e-3`
- `python audit_label_noise.py --disable-class-weights --feature-norm standard --lr 1.5e-3`
- `python label_review_checklist.py --min-confidence 0.8 --top-k 20`
- `python apply_label_overrides.py --csv artifacts/graph_transformer/label_review_checklist.csv --out lymphocyte_toy_data_relabel.pkl`
- `python lymph_graph_transformer.py --task graph --path lymphocyte_toy_data_relabel.pkl --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3`
- `python apply_label_overrides.py --csv artifacts/graph_transformer/label_review_checklist.csv --out lymphocyte_toy_data_relabel_top20.pkl`
- `python lymph_graph_transformer.py --task graph --path lymphocyte_toy_data_relabel_top20.pkl --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3`
- `python audit_label_noise.py --path lymphocyte_toy_data_relabel.pkl --disable-class-weights --feature-norm standard --lr 1.5e-3 --out artifacts/graph_transformer/label_noise_audit_relabel_top10.csv --summary-out artifacts/graph_transformer/label_noise_audit_relabel_top10_summary.json`
- `python label_review_checklist.py --path lymphocyte_toy_data_relabel.pkl --audit-csv artifacts/graph_transformer/label_noise_audit_relabel_top10.csv --min-confidence 0.9 --top-k 20 --out-md artifacts/graph_transformer/label_review_checklist_relabel_top10.md --out-csv artifacts/graph_transformer/label_review_checklist_relabel_top10.csv`
- `python apply_label_overrides.py --path lymphocyte_toy_data_relabel.pkl --csv artifacts/graph_transformer/label_review_checklist_relabel_top10.csv --out lymphocyte_toy_data_relabel_top10_pass2.pkl`
- `python lymph_graph_transformer.py --task graph --path lymphocyte_toy_data_relabel_top10_pass2.pkl --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3`
- `python lymph_graph_transformer.py --task graph --gnn sage --gnn-layers 1 --hidden-dim 256 --dropout 0.1 --pool cls --head linear --transformer-layers 2 --transformer-heads 8 --transformer-dropout 0.1 --transformer-ffn 512 --transformer-virtual-node --disable-class-weights --feature-norm standard --lr 1.5e-3 --graph-features basic`

## Experiment Log (2026-02-03 to 2026-02-05) - Node-Level GPS Path to Best Run

### Stage A - GraphSAGE stabilization and diagnostics
- `python test_sage_lympho.py` (baseline neighbor run; strong ROC-AUC but spiky val curves)
- Added train metrics + rerun: `python test_sage_lympho.py`
- Switched to full-graph val eval each epoch + rerun: `python test_sage_lympho.py`
- Added EMA plot smoothing + rerun: `python test_sage_lympho.py`
- Optimizer noise reduction trial (lower LR/higher WD) + rerun: `python test_sage_lympho.py`
- EMA-on-weights trial + rerun: `python test_sage_lympho.py`
- Added PR-AUC logging + test PR-AUC + rerun: `python test_sage_lympho.py`
- Early-stop on PR-AUC trial + rerun: `python test_sage_lympho.py`

### Stage B - Strong GraphSAGE reference with CV
- New script: `test_sage_lympho_cv_prauc.py`
- Full-batch val for early-stop + CV + PR-AUC tracking:
  - `python test_sage_lympho_cv_prauc.py`
- Added per-fold threshold tuning (val-threshold -> test):
  - `python test_sage_lympho_cv_prauc.py`
- Added per-fold thresholds CSV + ROC/PR summary figure:
  - `python test_sage_lympho_cv_prauc.py`
- Key reference output:
  - `artifacts/sage_cv_prauc/run_20260203_201401/sage_cv_summary_20260203_201401.csv`
  - Summary: acc 0.960593, f1 0.954567, roc_auc 0.994345, pr_auc 0.990779

### Stage C - GPS baselines + sanity controls
- New neighbor GPS script: `test_gps_lympho_neighbor.py`
- Initial GPS neighbor run:
  - `python test_gps_lympho_neighbor.py`
- Split integrity checks added (disjoint masks + class counts):
  - `python test_gps_lympho_neighbor.py`
- Label-shuffle control + 3 repeat runs (sanity):
  - `python test_gps_lympho_neighbor.py`
  - Shuffle run collapsed to chance ROC-AUC (~0.5), repeat runs stayed very high

### Stage D - Fair GPS vs SAGE comparison (same CV protocol)
- New CV GPS script: `test_gps_lympho_cv_prauc.py`
- Run with same setup (CV + PR-AUC early stop + threshold tuning):
  - `python test_gps_lympho_cv_prauc.py`
- Key output:
  - `artifacts/gps_cv_prauc/run_20260203_204705/gps_cv_summary_20260203_204705.csv`
  - Summary: acc 0.946610, f1 0.945422, roc_auc 0.999981, pr_auc 0.999962

### Stage E - gnn_transformer_lympho.py integration and tuning
- Added preset support and tested paper-like preset:
  - `python gnn_transformer_lympho.py --gps-preset paper_cifar --out-dir artifacts/gps_lympho_paper_cifar --history-tag paper_cifar`
  - Output: weaker on this dataset (acc ~0.72)
- Added small-dataset preset trial:
  - `python gnn_transformer_lympho.py --gps-preset lympho_small --out-dir artifacts/gps_lympho_preset_lympho_small --history-tag lympho_small`
  - Output: weaker (acc ~0.653)
- Added node-level path to `gnn_transformer_lympho.py` with fixed lympho_node knobs and GPSNet-style node architecture.
- First successful integrated node run:
  - `python gnn_transformer_lympho.py --gps-preset lympho_node --out-dir artifacts/gps_lympho_node`
  - Output: `artifacts/gps_lympho_node/gps_summary_20260203_213816.csv`
  - Summary: acc 0.987429, f1 0.986413, roc_auc 0.999991, pr_auc 0.999982
- Repro variability run:
  - `python gnn_transformer_lympho.py --gps-preset lympho_node --out-dir artifacts/gps_lympho_node`
  - Output: `artifacts/gps_lympho_node/gps_summary_20260205_154217.csv`

### Stage F - Standalone fixed script (no config dependency)
- New standalone file: `best_gnn_transformer.py`
- No CLI args, no import from `gnn_transformer_lympho.py`; hard-coded architecture and hyperparameters.
- Run command:
  - `python best_gnn_transformer.py`
- Output folder (fixed):
  - `gps_artifacts/`
