param(
  [string]$Cfg  = "configs\cxr_baseline.yaml",
  [string]$Ckpt = "results\checkpoints\best.pt"   # matches your trainer output
)

$ErrorActionPreference = "Stop"

# Resolve python from the active venv
$py = (Get-Command python).Source
if (-not $py) { throw "Could not resolve 'python' in PATH. Activate .venv310 first." }

# Resolve the directory that holds this script (works when executed from file)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $here) {
  # Fallback when someone pastes this script inline: assume scripts live in .\tools
  $here = Join-Path (Get-Location) "tools"
}

# 1) Train baseline
& $py -m src.train.baseline --config $Cfg --save $Ckpt

# 2) Baseline robustness sweep on VAL → writes results\metrics\robust_sweep_val.csv
& (Join-Path $here 'run_val_sweeps.ps1') -Force -Ckpt $Ckpt

# 3) Re-run tri-obj vs baseline comparison (plots + tables)
& (Join-Path $here 'run_all_val.ps1')

# 4) Save stable copies of the compare plots
New-Item -ItemType Directory -Force -Path .\docs\figures | Out-Null
Copy-Item .\results\metrics\robust_compare_delta_heatmap.png .\docs\figures\xai_robust_val_delta_heatmap.png -Force
Copy-Item .\results\metrics\robust_compare_pgd10.png        .\docs\figures\xai_robust_val_pgd10.png        -Force

# 5) Commit (skip hooks if pre-commit not installed)
git add results/metrics/*.csv results/metrics/*.md results/metrics/*.png docs/figures/*
git commit --no-verify -m "baseline(val): train NIH baseline, add robust_sweep_val.csv, export VAL figures"
git push
