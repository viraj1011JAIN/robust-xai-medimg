Param(
  [string]$Config = "configs/tiny.yaml",
  [string]$Out = "results/metrics/eval_tiny.csv",
  [switch]$DryRun = $true
)
$dry = $DryRun.IsPresent ? "--dry-run" : ""
python -m src.train.evaluate --config $Config $dry --out $Out
