Param(
  [string]$Config      = "configs/tiny.yaml",
  [string]$BaseCkpt    = "",  # baseline ckpt (e.g., results\checkpoints\best.pt)
  [string]$TriObjCkpt  = "",  # tri-obj ckpt if you have one (e.g., results\checkpoints\triobj_best.pt)
  [string]$BaseCsv     = "",  # optional: external baseline CSV to compare against (skips baseline sweep if provided)
  [switch]$Force
)

function Pick-Ckpt([string[]]$candidates) {
  foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
  return ""
}

# 1) Read VAL CSV and img_size from the config
$pyCsv = "from omegaconf import OmegaConf; cfg=OmegaConf.load(r'$Config'); print(cfg.data.val_csv)"
$valCsv = & python -c $pyCsv
if ($LASTEXITCODE -ne 0 -or -not $valCsv) { throw "Failed to read cfg.data.val_csv from $Config" }
if (-not (Test-Path $valCsv))            { throw "Dataset CSV not found: $valCsv" }

$pySz = "from omegaconf import OmegaConf; cfg=OmegaConf.load(r'$Config'); print(int(getattr(getattr(cfg,'data',None),'img_size',64) or 64))"
$imgSize = & python -c $pySz
if ($LASTEXITCODE -ne 0 -or -not $imgSize) { $imgSize = 64 }

# 2) Decide checkpoints
$baselineCkpt = $BaseCkpt
if (-not $baselineCkpt) {
  $baselineCkpt = Pick-Ckpt @(
    "results\checkpoints\best.pt",
    "results\checkpoints\last.pt",
    "results\checkpoints\best_weights.pt"
  )
}
if (-not $baselineCkpt) { throw "No baseline checkpoint found. Pass -BaseCkpt or create results\checkpoints\best.pt" }
Write-Host "[baseline ckpt] $baselineCkpt"

$triobjCkpt = $TriObjCkpt
if (-not $triobjCkpt) {
  $triobjCkpt = Pick-Ckpt @(
    "results\checkpoints\triobj_best.pt",  # preferred if available
    $baselineCkpt                           # fallback to baseline if robust ckpt is missing
  )
}
if (-not $triobjCkpt) { throw "No tri-obj checkpoint found (and no baseline fallback). Pass -TriObjCkpt." }
Write-Host "[triobj ckpt] $triobjCkpt"

# 3) Output paths
New-Item -ItemType Directory -Force -Path results\metrics | Out-Null
$baselineCsvLocal = "results\metrics\robust_sweep_val_debug.csv"
$triobjCsv        = "results\metrics\robust_sweep_val_debug_triobj.csv"

# 4) Baseline sweep (skip if external BaseCsv provided)
$useBaselineCsv = $BaseCsv
if ($useBaselineCsv) {
  if (-not (Test-Path $useBaselineCsv)) { throw "Provided -BaseCsv not found: $useBaselineCsv" }
  Write-Host "[baseline csv] Using external CSV: $useBaselineCsv"
} else {
  $useBaselineCsv = $baselineCsvLocal
  if ($Force -or -not (Test-Path $useBaselineCsv)) {
    Write-Host "[baseline] running robust_sweep -> $useBaselineCsv"
    & python -m src.eval.robust_sweep --csv $valCsv --ckpt $baselineCkpt --img_size $imgSize --out $useBaselineCsv
    if ($LASTEXITCODE -ne 0) { throw "Baseline sweep failed." }
  } else {
    Write-Host "[skip] baseline exists: $useBaselineCsv"
  }
}

# 5) Tri-obj sweep (really: second sweep with triobj ckpt)
if ($Force -or -not (Test-Path $triobjCsv)) {
  Write-Host "[triobj] running robust_sweep -> $triobjCsv"
  & python -m src.eval.robust_sweep --csv $valCsv --ckpt $triobjCkpt --img_size $imgSize --out $triobjCsv
  if ($LASTEXITCODE -ne 0) { throw "Tri-obj sweep failed." }
} else {
  Write-Host "[skip] tri-obj sweep exists: $triobjCsv (use -Force to recompute)"
}

# 6) Compare CSVs
& python -m src.eval.robust_compare --base $useBaselineCsv --tri $triobjCsv --outdir results\metrics
if ($LASTEXITCODE -ne 0) { throw "robust_compare failed." }

Write-Host "[done] Wrote sweeps + compare plots to results\metrics"
