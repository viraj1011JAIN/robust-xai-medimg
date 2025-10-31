param(
  [string]$Python="python",
  [string]$Req="requirements-lock.txt"
)
if (-not (Test-Path .venv)) { & $Python -m venv .venv }
& .\.venv\Scripts\python -m pip install --upgrade pip
& .\.venv\Scripts\pip install -r $Req
Write-Host "Env ready: .venv"
