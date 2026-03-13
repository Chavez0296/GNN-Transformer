Results Index

Use this as the quick map for ablations and final reporting.

Core results root
- `gps_work/results/gps_artifacts/`

Key experiment groups
- `gps_work/results/gps_artifacts/data_agumentation/`
  - locked standalone baselines: `.../locked_in/graph_level/`
  - GPS experiments: `.../gps_experiments/`
- `gps_work/results/gps_artifacts/final_sel/`
  - seed protocol raw CSVs:
    - `.../baseline_seed_results.csv`
    - `.../gps_seed_results.csv`
  - final comparison reports:
    - `.../reports/seed_stability_summary.csv`
    - `.../reports/gps_vs_baseline_seed_comparison.csv`
  - final visualization PNGs:
    - `.../reports/baseline_gnn_performance.png`
    - `.../reports/gps_gnn_performance.png`
- `gps_work/results/gps_artifacts/improve_round/`
  - suggestion 2 gatedgraph A/B:
    - `.../s2_gatedgraph_ab/ab_results.csv`
  - suggestion 3 AUC-focused tuning + confirms:
    - `.../s3_auc_tune/auc_tune_results.csv`
  - suggestion 4 scheduler trial (kept for audit, not adopted):
    - `.../s4_scheduler/scheduler_results.csv`
  - suggestion 5 pna/gine confirmations:
    - `.../s5_pna_gine_confirm/pna_gine_confirm.csv`
  - compiled final tracks:
    - `.../reports/all_candidates_compiled.csv`
    - `.../reports/final_track_f1_first.csv`
    - `.../reports/final_track_auc_first.csv`
  - baseline vs transformer visualizations:
    - `.../reports/figures/baseline_vs_initial_transformer.png`
    - `.../reports/figures/baseline_vs_transformer_f1_track.png`
    - `.../reports/figures/baseline_vs_transformer_auc_track.png`
    - `.../reports/figures/transformer_tradeoff_scatter.png`
    - `.../reports/figures/transformer_delta_heatmap.png`

Legacy/earlier sweeps
- `gps_work/results/gps_artifacts/graph_level/`
- `gps_work/results/gps_artifacts/node_level/`
- `gps_work/results/gps_artifacts/gnn_baseline_stability_20260207/`
- `gps_work/results/gps_artifacts/gnn_baseline_stable_final_20260207/`
