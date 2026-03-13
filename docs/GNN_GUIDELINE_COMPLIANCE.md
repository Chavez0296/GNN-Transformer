# GNN Guideline Compliance Plan

This document is a working checklist to close the remaining gaps against `gps_work/docs/GNN guideline.pdf`.

## 1) Formal Hypothesis Block (Required Format)

### Hypothesis H1
- **Claim:** Baseline local message passing underperforms on graphs requiring long-range context; adding a transformer-style global attention block improves performance.
- **Evidence:** In our improvement runs, tuned transformer variants improved both macro-F1 and ROC-AUC for `gcn`, `gin`, and `sage` versus baseline references (see `gps_work/results/gps_artifacts/improve_round/s3_auc_tune/auc_tune_results.csv` and compiled tracks in `gps_work/results/gps_artifacts/improve_round/reports/`).
- **Prediction (signature):** Gains should be strongest on harder graphs (larger size / higher estimated diameter / lower local purity).

### Hypothesis H2
- **Claim:** Transformer gains depend on structural priors (PE/SE, graph-aware bias), not just adding attention.
- **Evidence:** Existing runs already show attention type sensitivity (`none` vs `performer` for `gatedgraph`) and stability differences (`s4_scheduler` regressions).
- **Prediction (signature):** `multihead/performer + LapPE/RWSE/graphormer-bias` should outperform plain global attention on high-diameter or structurally sparse slices.

## 2) Literature Mapping Table (Guideline Requirement)

| Method idea (paper) | Failure mode targeted | What to implement (repo flags) | Compute cost | Expected effect | Ablation to test |
|---|---|---|---|---|---|
| **GraphGPS** (Rampasek et al., NeurIPS 2022, arXiv:2205.12454) | Local-only MPNN misses long-range interactions / oversquashing | `--local-gnn {sage,gin,gcn,pna}` + `--global-attn {performer,multihead}` + optional `--lap-pe-dim`, `--rwse-dim` | Medium | Better macro-F1 and AUC on hard slices while keeping stable training | Baseline local-only vs same local backbone + GPS attention, same seeds |
| **Graphormer bias** (Ying et al., NeurIPS 2021, arXiv:2106.05234) | Naive global attention lacks graph-structure inductive bias | `--global-attn graphormer --graphormer-max-dist 5` (same local backbone) | Medium-high | AUC gain on larger/high-diameter graphs; stronger global reasoning | `multihead` vs `graphormer` (same local GNN, same seeds/splits) |
| **SAN + LapPE** (Dwivedi & Bresson, arXiv:2012.09699) | Transformer attention without positional structure underuses topology | `--global-attn multihead --lap-pe-dim 8 --lap-pe-encoder signnet` (or `raw`) | Medium | Better discrimination across graph-size/diameter slices | `multihead + no PE` vs `multihead + LapPE`, same seeds |

## 3) Guideline Compliance Status (UPDATED March 2026)

| Guideline item | Status | Artifact |
|---|---|---|
| Graph-task slice metrics by **#nodes, density, estimated diameter** | **COMPLETE** | `gps_work/results/gps_artifacts/improve_round/graph_slices/sage_baseline_vs_performer/` |
| Long-range evidence probe 1: Depth sweep + oversmoothing | **COMPLETE** | `gps_work/results/gps_artifacts/improve_round/depth_sweep/depth_sweep_sage.csv` |
| Long-range evidence probe 2: Diameter hypothesis | **COMPLETE** | `gps_work/results/gps_artifacts/improve_round/diameter_hypothesis/diameter_hypothesis_summary.txt` |
| Long-range evidence probe 3: Small-world perturbation | **COMPLETE** | `gps_work/results/gps_artifacts/improve_round/smallworld_probe/smallworld_analysis_sage.txt` |
| Formal hypothesis block (claim/evidence/prediction) | **COMPLETE** | Section 1 above, and `gps_work/docs/FINAL_REPORT.md` |
| Literature mapping table | **COMPLETE** | Section 2 above, and `gps_work/docs/FINAL_REPORT.md` |
| Final 10-15 min presentation deck | **COMPLETE** | `gps_work/docs/FINAL_PRESENTATION.md` |
| Final rubric-aligned written report | **COMPLETE** | `gps_work/docs/FINAL_REPORT.md` |

## 4) Starter Experiment Matrix for Remaining Diagnostics

| Probe | Baseline run | Transformer run | Output artifact |
|---|---|---|---|
| Performance vs graph size/diameter/density | `local-only` config | best `GPS` config | `slice_metrics_graph_size_density_diameter.csv` + plots |
| Depth sweep + oversmoothing indicator | `--gps-layers 1,2,3,4` local-only branch | same sweep with global attention enabled | `depth_sweep_summary.csv` + embedding collapse/variance plot |
| Small-world test (add/remove random long edges) | perturb baseline | perturb transformer | `small_world_perturbation_summary.csv` |

## 5) Final Deliverable Checklist (Rubric-aligned)

- **Slides (10-15 min):**
  - Problem framing + dataset
  - Baseline vs transformer key results
  - Behavioral diagnosis
  - Hypothesis + literature mapping table
  - Minimal implementation summary
  - Ablation proof (mean±std, same seeds)
  - Key plots (learning curves, slices, error audit)
  - Conclusion, limitations, next steps

- **Report (4-8 pages):**
  - Setup/split protocol + reproducibility
  - Methods (baseline + transformer additions)
  - Behavioral diagnosis + hypothesis block
  - Literature-to-design mapping table (Section 2)
  - Experiments/ablations with mean±std
  - Slice + error analysis tied to hypothesis
  - Discussion (worked / failed / why)
  - Appendix with commands, configs, artifact paths

## 6) Citation Snippets

1. Rampasek et al., *Recipe for a General, Powerful, Scalable Graph Transformer*, NeurIPS 2022. arXiv:2205.12454.
2. Ying et al., *Do Transformers Really Perform Badly for Graph Representation?*, NeurIPS 2021. arXiv:2106.05234.
3. Dwivedi & Bresson, *A Generalization of Transformer Networks to Graphs*, arXiv:2012.09699.
