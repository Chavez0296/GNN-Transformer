# Slide 1 - From GNN to GPS
- **Project:** Lymphocyte node classification on tissue-cell graphs
- **Challenge:** Local-only GNNs can miss long-range context
- **Question:** Does GPS (GNN + Transformer attention) improve reliability and performance?
- **Goal:** Build a robust model with strong CV metrics and clear failure-mode evidence

Speaker notes:
- Open with the core story: we moved from standard GNNs to GPS to address context limitations.
- Emphasize that this is not only about higher scores, but trustworthy behavior.

---

# Slide 2 - What Is GPS?
- **GPS = Graph Positioning System layer** (Graph Transformer design)
- It combines two pathways in each layer:
  - **Local pathway:** message passing (e.g., SAGEConv)
  - **Global pathway:** multi-head self-attention
- Output of both pathways is fused to produce richer node representations

Speaker notes:
- Keep it simple: local branch learns neighborhood patterns, global branch learns non-local relationships.
- This is the key architectural difference from plain GraphSAGE.

---

# Slide 3 - How a GPS Layer Works (Intuition)
- Step 1: Node gathers nearby information from connected neighbors (local aggregation)
- Step 2: Node attends to broader graph context via attention scores (global mixing)
- Step 3: Combine local + global outputs, normalize, and pass to next layer
- Result: Better handling of both short-range and long-range structure

Speaker notes:
- You can describe this as "zoom lens" behavior: GPS sees both close-up and wide-angle context.
- This is especially useful when local neighborhoods alone are ambiguous.

---

# Slide 4 - Why GPS for This Lymphocyte Task
- Tissue-cell graphs contain structural signals beyond immediate neighbors
- Baseline GraphSAGE showed instability and weaker slice performance in hard regions
- GPS hypothesis: global attention should improve difficult slices while preserving ranking quality
- We evaluate this with the same CV protocol, same seeds, same training recipe

Speaker notes:
- This slide bridges method to problem: why GPS is not just fancy, but targeted.
- Highlight fairness: only architecture changes in ablation.

---

# Slide 5 - Metrics and Evaluation Approach
- **Primary metrics:** Macro-F1, ROC-AUC, PR-AUC, Accuracy
- **Why these metrics:** PR-AUC and Macro-F1 are robust for class imbalance; ROC-AUC captures ranking quality
- **Validation protocol:** 5-fold stratified CV, same seeds/splits for fair comparison
- **Training controls:** PR-AUC early stopping + validation threshold tuning
- **Ablation design:** Baseline SAGE vs Baseline+GPS with identical training recipe

Speaker notes:
- This slide tells the audience exactly how performance is judged.
- Emphasize fairness: only the model architecture changes in the ablation.
