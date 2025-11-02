# Plan Update (Post 24-epoch Cosine Run)

## Completed
- Implemented RMSE_kmh and MAE_kmh logging.
- Added best checkpoint saving (`gcnlstm_model_best.pt`).
- Early stopping (patience=5, min_delta=1e-4).
- Ran 24 epochs with cosine scheduler (`e24_schcosine_g32_l64`).
- Generated predictions and routing outputs; opened local preview for route map.
- Grid summary seeded at `outputs/reports/grid_summary.csv`.

## Immediate
- Launch grid search: `pwsh D:/PGT/my_project/run_grid.ps1`.
- Monitor `outputs/reports/grid_summary.csv` growth and scan for top RMSE/MAE.
- If time-constrained, prioritize configs:
  - `e24_schcosine_g64_l128`
  - `e24_schnone_g64_l128` (baseline)
  - `e24_schstep_g64_l128` (step=5, gamma=0.6)

## Short Term (1–2 days)
- Add `weight_decay` (1e-4 to 5e-4) and log impact.
- Try `train_split=0.8` and verify gen-gap changes.
- Save per-epoch LR trace for scheduler diagnostics.
- Export top-3 routes with different (`alpha,beta,gamma`) to compare sensitivity.

## Documentation & PR
- Update PR description to reference new metrics and artifacts.
- Include `run_grid.ps1` and enhanced `train_eval.py` changes.
- Attach `outputs/reports/README.md` to showcase reproducibility.

## Risks & Mitigations
- CPU-only training speed: keep grid modest; cache artifacts.
- Overfit risk: early stop and weight decay; monitor gen-gap.
- Data drift: confirm station/edge alignment stays consistent post-build.