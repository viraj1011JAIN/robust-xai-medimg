param(
  [string]$BaseCsv = "results\metrics\robust_sweep_val.csv",
  [string]$TriCsv  = "results\metrics\robust_sweep_val_triobj.csv",
  [string]$OutDir  = "results\metrics"
)

$ErrorActionPreference = "Stop"

$py = (Get-Command python).Source
if (-not $py) { throw "Could not resolve 'python' in PATH. Activate .venv310 first." }

if (-not (Test-Path $BaseCsv)) { throw "Baseline CSV not found: $BaseCsv (run run_val_sweeps.ps1 first)" }
if (-not (Test-Path $TriCsv))  { throw "Tri-objective CSV not found: $TriCsv" }

& $py -m src.eval.compare_robustness --base_csv $BaseCsv --tri_csv $TriCsv --outdir $OutDir

# Copy stable figure names (README unchanged)
.\tools\export_for_thesis.ps1 -NoReadme
