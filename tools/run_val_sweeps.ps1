param(
  [string]$Ckpt = "",
  [string]$BaseCsv = "results\metrics\robust_sweep_val.csv",
  [switch]$Force
)

$ErrorActionPreference = "Stop"
$PY = ".\.venv\Scripts\python.exe"

# ckpt autodetect (as you already have)
$ckptDir   = "results\checkpoints"
if (-not $Ckpt -or -not (Test-Path $Ckpt)) {
  $candidates = @("triobj_best.pt","best.pt","last.pt","best_weights.pt") | ForEach-Object { Join-Path $ckptDir $_ }
  foreach ($p in $candidates) { if (Test-Path $p) { $Ckpt = $p; break } }
}
if (-not $Ckpt -or -not (Test-Path $Ckpt)) { throw "No checkpoint found in $ckptDir." }
Write-Host "[ckpt] using $Ckpt"

# paths
$triCsv = "results\metrics\robust_sweep_val_triobj.csv"

# tri-obj sweep (skip if exists unless -Force)
if ($Force -or -not (Test-Path $triCsv)) {
  & $PY -m src.eval.robust_sweep `
    --csv "C:\Users\Viraj Jain\data\nih_cxr\val.csv" `
    --ckpt $Ckpt `
    --model resnet18 --bs 32 --attack_bs 8 `
    --eps "0,2,4,6" --steps "0,5,10" --alpha 1 `
    --out $triCsv
} else {
  Write-Host "[skip] tri-obj sweep exists: $triCsv (use -Force to recompute)"
}

# comparison (only if baseline exists)
if (-not (Test-Path $BaseCsv)) {
  Write-Warning "Baseline CSV not found at $BaseCsv — skipping comparison."
} else {
  & $PY -m src.eval.compare_robustness `
    --base_csv $BaseCsv `
    --tri_csv  $triCsv `
    --outdir   results\metrics
}

Write-Host "[done] Wrote tri-obj sweep + compare plots to results\metrics"

