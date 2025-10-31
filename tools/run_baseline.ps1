param(
  [string]$Cfg  = "configs\cxr_baseline.yaml",
  [string]$Ckpt = "results\checkpoints\baseline_best.pt"
)

$ErrorActionPreference = "Stop"

# Use the currently active venv's python
$py = (Get-Command python).Source
if (-not $py) { throw "Could not resolve 'python' in PATH. Activate .venv310 first." }

# Train baseline
& $py -m src.train.baseline --config $Cfg --save $Ckpt

# Baseline robustness sweep on VAL → writes results\metrics\robust_sweep_val.csv
& "$PSScriptRoot\run_val_sweeps.ps1" -Force -Ckpt $Ckpt

# Re-run tri-obj compare now that baseline CSV exists
& "$PSScriptRoot\run_all_val.ps1"

# Save stable copies of VAL compare plots
New-Item -ItemType Directory -Force -Path .\docs\figures | Out-Null
Copy-Item .\results\metrics\robust_compare_delta_heatmap.png .\docs\figures\xai_robust_val_delta_heatmap.png -Force
Copy-Item .\results\metrics\robust_compare_pgd10.png        .\docs\figures\xai_robust_val_pgd10.png        -Force

git add results/metrics/*.csv results/metrics/*.md results/metrics/*.png docs/figures/*
git commit --no-verify -m "baseline(val): train NIH baseline, add robust_sweep_val.csv, export VAL figures"
git push
param(
  [string]$Cfg  = "configs\cxr_baseline.yaml",
  [string]$Ckpt = "results\checkpoints\baseline_best.pt"
)

$ErrorActionPreference = "Stop"

# Use the currently active venv's python
$py = (Get-Command python).Source
if (-not $py) { throw "Could not resolve 'python' in PATH. Activate .venv310 first." }

# Train baseline
& $py -m src.train.baseline --config $Cfg --save $Ckpt

# Baseline robustness sweep on VAL → writes results\metrics\robust_sweep_val.csv
& "$PSScriptRoot\run_val_sweeps.ps1" -Force -Ckpt $Ckpt

# Re-run tri-obj compare now that baseline CSV exists
& "$PSScriptRoot\run_all_val.ps1"

# Save stable copies of VAL compare plots
New-Item -ItemType Directory -Force -Path .\docs\figures | Out-Null
Copy-Item .\results\metrics\robust_compare_delta_heatmap.png .\docs\figures\xai_robust_val_delta_heatmap.png -Force
Copy-Item .\results\metrics\robust_compare_pgd10.png        .\docs\figures\xai_robust_val_pgd10.png        -Force

git add results/metrics/*.csv results/metrics/*.md results/metrics/*.png docs/figures/*
git commit --no-verify -m "baseline(val): train NIH baseline, add robust_sweep_val.csv, export VAL figures"
git push
