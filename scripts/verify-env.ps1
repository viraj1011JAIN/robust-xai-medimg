Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
function Fail($m){ Write-Host "FAIL: $m" -ForegroundColor Red; exit 1 }
function Ok($m){ Write-Host "OK:   $m" -ForegroundColor Green }

Write-Host "=== Robust-XAI env verification ===`n" -ForegroundColor Cyan

# Helper: run inline Python by writing a temp file
function Invoke-Py {
  param([Parameter(Mandatory)][string]$Code, [Parameter(Mandatory)][string]$Step)
  $tmp = [System.IO.Path]::GetTempFileName() + ".py"
  Set-Content -LiteralPath $tmp -Value $Code -Encoding UTF8
  & python $tmp
  $code = $LASTEXITCODE
  Remove-Item $tmp -Force
  if ($code -ne 0) { Fail "$Step failed (code $code)" }
}

$py = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $py) { Fail "python not found on PATH" } else { Ok "python: $($py.Source)" }

Invoke-Py -Step "Python version" -Code @"
import sys; print("python:", sys.version.replace("\n"," "))
"@; Ok "Printed Python version"

Invoke-Py -Step "Core py libs check" -Code @"
import importlib.util, sys
try:
    import numpy, scipy, cv2
except Exception as e:
    print("IMPORT ERROR:", e); sys.exit(2)
spec = importlib.util.find_spec("cv2")
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("cv2  :", getattr(spec,"origin",None))
info = getattr(cv2,"getBuildInformation", lambda: "")()
print("opencv headless:", ("HIGHGUI" not in info))
"@; Ok "NumPy/Scipy/OpenCV look good"

Invoke-Py -Step "PyTorch check" -Code @"
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
"@; Ok "PyTorch loaded"

if (Test-Path .\tools\smoke_gradcam.py) {
  & python .\tools\smoke_gradcam.py --device cpu
  if ($LASTEXITCODE -ne 0) { Fail "smoke_gradcam.py failed (code $LASTEXITCODE)" }
  if (-not (Test-Path .\gradcam_smoke.png)) { Fail "gradcam_smoke.png not created" }
  Ok "Grad-CAM smoke test passed (gradcam_smoke.png)"
} else {
  Write-Host "WARN: tools\smoke_gradcam.py not found, skipping smoke test" -ForegroundColor Yellow
}
Write-Host "`nAll checks passed." -ForegroundColor Green
