# Grid search for GCNLSTM hyperparameters and schedulers
param(
    [string]$ConfigPath = "D:/PGT/my_project/config.yaml"
)

$epochsList = @(12, 24)
$schedulers = @("none", "cosine", "step")
$gcnList = @(32, 64)
$lstmList = @(64, 128)

$reportsDir = "D:/PGT/my_project/outputs/reports"
New-Item -ItemType Directory -Force -Path $reportsDir | Out-Null

foreach ($ep in $epochsList) {
  foreach ($sch in $schedulers) {
    foreach ($gcn in $gcnList) {
      foreach ($lstm in $lstmList) {
        $run = "e$ep" + "_" + "sch$sch" + "_" + "g$gcn" + "_" + "l$lstm"
        Write-Host "[grid] Running $run"
        python D:/PGT/my_project/src/train_eval.py --config $ConfigPath --epochs $ep --lr_scheduler $sch --step_size 5 --gamma 0.6 --run_name $run
      }
    }
  }
}

Write-Host "[grid] Summary written to $reportsDir/grid_summary.csv"