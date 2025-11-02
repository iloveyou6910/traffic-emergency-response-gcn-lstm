# Training & Inference Report (e24_schcosine_g32_l64)

- Run name: `e24_schcosine_g32_l64`
- Config: `D:/PGT/my_project/config.yaml`
- Epochs: 24 (early stop at 23, patience=5)
- Scheduler: Cosine annealing
- Model: GCN hidden 32, LSTM hidden 64

## Key Metrics
- Best Eval MSE: 0.0086489
- Best Eval RMSE_kmh: 7.440
- Best Eval MAE_kmh: 5.764
- Last Train RMSE_kmh: 6.003
- Last Eval RMSE_kmh: 7.958
- Generalization gap (last): 1.956 km/h

## Artifacts
- Best checkpoint: `outputs/models/gcnlstm_model_best.pt`
- Last checkpoint: `outputs/models/gcnlstm_model.pt`
- Training log: `outputs/logs/train_log.json`
- Epoch events: `outputs/logs/train_events.log`
- Predictions CSV: `outputs/logs/predictions.csv`
- Route JSON: `outputs/reports/route.json`
- Route Map HTML: `outputs/plots/route_map.html` (preview via `http://localhost:8000/route_map.html` after running `python -m http.server` in `outputs/plots`)

## Observations
- Cosine schedule helped reduce eval RMSE to ~7.44 km/h.
- Early stop triggered at epoch 23, indicating diminishing returns.
- Remaining gap suggests further capacity or regularization tuning.

## Next Steps
- Expand grid search: `pwsh D:/PGT/my_project/run_grid.ps1`.
- Try `gcn_hidden=64`, `lstm_hidden=128`, and add `weight_decay=1e-4`.
- Consider train_split=0.8 and longer T_max for cosine.
- Add MAE/percentile metrics to routing cost sensitivity analysis.