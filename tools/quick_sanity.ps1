Param(
  [string]$Config = "configs/tiny.yaml"
)
Write-Host ">>> Smoke..."
python -m src.train.baseline --smoke

Write-Host ">>> Tiny train (1 epoch)..."
python -m src.train.baseline --config $Config

Write-Host ">>> Eval (CSV)..."
New-Item -ItemType Directory -Force -Path results\metrics | Out-Null
python -m src.train.evaluate --config $Config --dry-run --out results/metrics/eval_tiny.csv

Write-Host ">>> Grad-CAM figure..."
New-Item -ItemType Directory -Force -Path docs\figures | Out-Null
powershell -ExecutionPolicy Bypass -File tools\make_gradcam.ps1 -Config $Config
