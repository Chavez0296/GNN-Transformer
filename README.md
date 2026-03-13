GPS Work Consolidation

This directory contains the working set for GPS ablations, tuning, and final reporting.

Structure
- `gps_work/code/`: Core training and plotting scripts.
- `gps_work/docs/`: Paper/guideline docs and project notes.
- `gps_work/data/`: Lymphocyte datasets used by experiments.
- `gps_work/results/gps_artifacts/`: All experiment outputs and summaries.

Primary script
- `gps_work/code/gnn_transformer_lympho.py`
- Legacy test scripts in `gps_work/code/` now default to writing under `gps_work/results/gps_artifacts/legacy_tests/`.

Reporting utilities
- `gps_work/code/build_selection_tracks.py` builds the initial F1-first and AUC-first tracks from `final_sel` seed-protocol CSVs.
- `gps_work/code/compile_final_recommendations.py` compiles accepted improvement-round candidates and writes:
  - `gps_work/results/gps_artifacts/improve_round/reports/all_candidates_compiled.csv`
  - `gps_work/results/gps_artifacts/improve_round/reports/final_track_f1_first.csv`
  - `gps_work/results/gps_artifacts/improve_round/reports/final_track_auc_first.csv`
- `gps_work/code/plot_baseline_vs_transformer.py` renders baseline-vs-transformer comparison figures under:
  - `gps_work/results/gps_artifacts/improve_round/reports/figures/`

Default paths in the primary script are now workspace-relative to this folder:
- data: `gps_work/data/lymphocyte_toy_data.pkl`
- outputs: `gps_work/results/gps_artifacts`

Example run
```bash
python "gps_work/code/gnn_transformer_lympho.py"
```

If you run from another location, you can still override paths explicitly:
```bash
python "gps_work/code/gnn_transformer_lympho.py" --path "gps_work/data/lymphocyte_toy_data.pkl" --out-dir "gps_work/results/gps_artifacts"
```

Compile the final recommendation tracks:
```bash
python "gps_work/code/compile_final_recommendations.py"
```

Render baseline vs transformer figures:
```bash
python "gps_work/code/plot_baseline_vs_transformer.py"
```
