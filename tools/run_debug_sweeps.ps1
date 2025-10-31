Param(
  [string]$Config = "configs/tiny.yaml",
  [string]$Ckpt   = "",
  [string]$BaseCsv = "",     # optional: external baseline metrics to compare against
  [switch]$Force
)

# 1) Read dataset CSV (VAL split) from the config
$py = "from omegaconf import OmegaConf; cfg=OmegaConf.load(r'$Config'); print(cfg.data.val_csv)"
$valCsv = & python -c $py
if ($LASTEXITCODE -ne 0 -or -not $valCsv) { throw "Failed to read cfg.data.val_csv from $Config" }
if (-not (Test-Path $valCsv)) { throw "Dataset CSV not found: $valCsv" }

# 2) Pick a checkpoint (if not given)
$ckpt = $Ckpt
if (-not $ckpt) {
  $candidates = @(
    "results\checkpoints\triobj_best.pt",
    "results\checkpoints\best.pt",
    "results\checkpoints\last.pt",
    "results\checkpoints\best_weights.pt"
  )
  foreach ($c in $candidates) { if (Test-Path $c) { $ckpt = $c; break } }
}
if ($ckpt) { Write-Host "[ckpt] using $ckpt" } else { Write-Host "[ckpt] none found; proceeding without --ckpt" }

# 3) Output paths
New-Item -ItemType Directory -Force -Path results\metrics | Out-Null
$baselineCsv = "results\metrics\robust_sweep_val_debug.csv"
$triobjCsv   = "results\metrics\robust_sweep_val_debug_triobj.csv"

# 4) (Re)compute BASELINE sweep (clean model)
if ($Force -or -not (Test-Path $baselineCsv)) {
  & python -m src.eval.robust_sweep --csv $valCsv --out $baselineCsv --img-size 64
  if ($LASTEXITCODE -ne 0) { throw "Baseline sweep failed." }
} else {
  Write-Host "[skip] baseline exists: $baselineCsv"
}

# 5) (Re)compute TRI-OBJECTIVE sweep (uses checkpoint if provided)
if ($Force -or -not (Test-Path $triobjCsv)) {
  $args = @("--csv",$valCsv,"--out",$triobjCsv,"--img-size","64","--triobj")
  if ($ckpt) { $args += @("--ckpt",$ckpt) }
  & python -m src.eval.robust_sweep @args
  if ($LASTEXITCODE -ne 0) { throw "Tri-obj sweep failed." }
} else {
  Write-Host "[skip] tri-obj sweep exists: $triobjCsv (use -Force to recompute)"
}

# 6) Compare (tri-obj vs baseline). If -BaseCsv is passed, use it; else use our baselineCsv.
$useBase = if ($BaseCsv) { $BaseCsv } else { $baselineCsv }
if (-not (Test-Path $useBase)) { throw "Baseline CSV to compare not found: $useBase" }

& python -m src.eval.robust_compare --base $useBase --tri $triobjCsv --outdir results\metrics
if ($LASTEXITCODE -ne 0) { throw "robust_compare failed." }

Write-Host "[done] Wrote tri-obj sweep + compare plots to results\metrics"
