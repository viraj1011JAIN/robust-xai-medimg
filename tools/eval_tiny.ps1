Param(
  [string]$Config = "configs/tiny.yaml",
  [string]$Out    = "results/metrics/eval_tiny.csv",
  [switch]$DryRun = $true
)
# Build argument list safely
$ArgsList = @("-m","src.train.evaluate","--config",$Config,"--out",$Out)
if ($DryRun.IsPresent) { $ArgsList += "--dry-run" }
python @ArgsList
