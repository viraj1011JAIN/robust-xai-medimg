Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
powershell -ExecutionPolicy Bypass -File .\tools\run_debug_sweeps.ps1 -Force -BaseCsv .\results\metrics\robust_sweep_val_debug.csv
.\tools\export_for_thesis.ps1
Get-ChildItem .\results\metrics | Sort-Object LastWriteTime -Descending | Select LastWriteTime,Name -First 8
