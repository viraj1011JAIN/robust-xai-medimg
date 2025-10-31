function Resolve-ValCsv {
  param([string]$Config = "configs/tiny.yaml")
  $py = "from omegaconf import OmegaConf; print(OmegaConf.load(r'$Config').data.val_csv)"
  $csv = & python -c $py
  if (-not $csv -or -not (Test-Path $csv)) { throw "Dataset CSV not found from $Config -> $csv" }
  return $csv
}

function Resolve-Ckpt {
  param(
    [string]$Desired = "",
    [string[]]$Fallbacks = @(
      "results\checkpoints\triobj_best.pt",
      "results\checkpoints\best.pt",
      "results\checkpoints\last.pt",
      "results\checkpoints\best_weights.pt"
    ),
    [string]$Role = "ckpt"
  )
  if ($Desired -and (Test-Path $Desired)) { return $Desired }
  foreach ($c in $Fallbacks) { if (Test-Path $c) { return $c } }
  throw "No $Role checkpoint found. Tried: $($Fallbacks -join ', ')"
}

function Invoke-RobustSweepGrid {
  param(
    [Parameter(Mandatory=$true)][string]$Csv,
    [Parameter(Mandatory=$true)][string]$Ckpt,
    [Parameter(Mandatory=$true)][string]$Out,
    [int]$ImgSize = 64,
    [double[]]$Eps = @(0,2,4),
    [int[]]$Steps = @(0,5,10),
    [switch]$Fresh
  )
  foreach ($e in $Eps) {
    foreach ($s in $Steps) {
      $args = @(
        "-m","src.eval.robust_sweep",
        "--csv",$Csv,"--ckpt",$Ckpt,"--img_size",$ImgSize,
        "--eps",$e,"--steps",$s,"--out",$Out
      )
      if ($Fresh) { $args += "--fresh" }
      python @args
    }
  }
}

function Compare-RobustSweeps {
  param(
    [Parameter(Mandatory=$true)][string]$BaseCsv,
    [Parameter(Mandatory=$true)][string]$TriCsv,
    [string]$OutDir="results/metrics"
  )
  python -m src.eval.robust_compare --base $BaseCsv --tri $TriCsv --outdir $OutDir
}

function Invoke-RobustCompare {
  param(
    [string]$Config = "configs/tiny.yaml",
    [string]$BaseCkpt = "",
    [string]$TriCkpt  = "",
    [int]$ImgSize = 64,
    [double[]]$Eps = @(0,2,4),
    [int[]]$Steps = @(0,5,10),
    [string]$OutDir = "results/metrics",
    [switch]$Fresh
  )
  $csv = Resolve-ValCsv -Config $Config
  $baseCsv = Join-Path $OutDir "robust_sweep_val_debug.csv"
  $triCsv  = Join-Path $OutDir "robust_sweep_val_debug_triobj.csv"
  $b = Resolve-Ckpt -Desired $BaseCkpt -Role "baseline"
  $t = Resolve-Ckpt -Desired $TriCkpt  -Fallbacks @($b) -Role "tri-obj"

  Invoke-RobustSweepGrid -Csv $csv -Ckpt $b -Out $baseCsv -ImgSize $ImgSize -Eps $Eps -Steps $Steps -Fresh:$Fresh
  Invoke-RobustSweepGrid -Csv $csv -Ckpt $t -Out $triCsv  -ImgSize $ImgSize -Eps $Eps -Steps $Steps -Fresh:$Fresh
  Compare-RobustSweeps -BaseCsv $baseCsv -TriCsv $triCsv -OutDir $OutDir
}

function Show-RobustHeads {
  param([int]$Lines = 8, [string]$OutDir = "results/metrics")
  & powershell -ExecutionPolicy Bypass -File "tools\show_csv_heads.ps1" `
    -Lines $Lines `
    (Join-Path $OutDir "robust_sweep_val_debug.csv") `
    (Join-Path $OutDir "robust_sweep_val_debug_triobj.csv")
}

Export-ModuleMember -Function Resolve-ValCsv,Resolve-Ckpt,Invoke-RobustSweepGrid,Compare-RobustSweeps,Invoke-RobustCompare,Show-RobustHeads
