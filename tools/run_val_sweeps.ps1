param(
  [string]$Ckpt   = "results\checkpoints\best.pt",
  [string]$ValCsv = "C:\Users\Viraj Jain\data\nih_cxr\val_debug.csv",  # ← set this
  [string]$Out    = "results\metrics\robust_sweep_val.csv",
  [string]$Model  = "resnet18",
  [int]$ImgSize   = 224,
  [string]$Eps    = "0,2,4,6",
  [int]$Steps     = 10,
  [int]$Bs        = 32,
  [int]$AttackBs  = 16,
  [switch]$Fresh
)

$ErrorActionPreference = "Stop"

# Use python from the active venv
$py = (Get-Command python).Source
if (-not $py) { throw "Could not resolve 'python' in PATH. Activate .venv310 first." }

if (-not (Test-Path $ValCsv)) { throw "VAL CSV not found: $ValCsv" }
New-Item -ItemType Directory -Force -Path (Split-Path $Out) | Out-Null

# Build args
$args = @(
  "-m","src.eval.robust_sweep",
  "--csv",$ValCsv,
  "--ckpt",$Ckpt,
  "--model",$Model,
  "--bs",$Bs.ToString(),
  "--attack_bs",$AttackBs.ToString(),
  "--img_size",$ImgSize.ToString(),
  "--eps",$Eps,
  "--steps",$Steps.ToString(),
  "--out",$Out
)
if ($Fresh) { $args += "--fresh" }

& $py @args
