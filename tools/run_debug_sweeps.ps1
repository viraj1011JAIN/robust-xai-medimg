param(
  [string]$Ckpt = ""
)

$ErrorActionPreference = "Stop"
$PY = ".\.venv\Scripts\python.exe"

# --- pick a checkpoint (honor -Ckpt if valid, else auto-detect) ---
if (-not $Ckpt -or -not (Test-Path $Ckpt)) {
  $ckptDir   = "results\checkpoints"
  $candidates = @("triobj_best.pt","best.pt","last.pt","best_weights.pt") |
    ForEach-Object { Join-Path $ckptDir $_ }
  $Ckpt = $null
  foreach ($p in $candidates) { if (Test-Path $p) { $Ckpt = $p; break } }
}
if (-not $Ckpt -or -not (Test-Path $Ckpt)) {
  Write-Error ("No checkpoint found. Tried: {0}" -f (("triobj_best.pt","best.pt","last.pt","best_weights.pt") -join ", "))
  exit 1
}
Write-Host "[ckpt] using $Ckpt"

# --- tri-objective sweep ---
& $PY -m src.eval.robust_sweep `
  --csv "C:\Users\Viraj Jain\data\nih_cxr\val_debug.csv" `
  --ckpt $Ckpt `
  --model resnet18 --bs 32 --attack_bs 8 `
  --eps "0,2,4,6" --steps "0,5,10" --alpha 1 `
  --out results\metrics\robust_sweep_val_debug_triobj.csv

# --- comparison plots (baseline vs tri-obj) ---
& $PY -m src.eval.compare_robustness `
  --base_csv results\metrics\robust_sweep_val_debug.csv `
  --tri_csv  results\metrics\robust_sweep_val_debug_triobj.csv `
  --outdir   results\metrics

Write-Host "[done] Wrote tri-obj sweep + compare plots to results\metrics"
