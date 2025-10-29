param(
  [string]$Cfg = "configs\isic_baseline.yaml"
)

$ErrorActionPreference = "Stop"
$py = (Get-Command python).Source
if (-not $py) { throw "Activate .venv310 first." }

Write-Host "[tcav] Using config: $Cfg"
& $py tools/tcav_pipeline.py
