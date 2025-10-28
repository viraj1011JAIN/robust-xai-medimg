$ErrorActionPreference = "Stop"
$PY = ".\.venv\Scripts\python.exe"

# Tri-objective sweep
& $PY -m src.eval.robust_sweep `
  --csv "C:\Users\Viraj Jain\data\nih_cxr\val_debug.csv" `
  --ckpt results\checkpoints\best.pt `
  --model resnet18 --bs 32 --attack_bs 8 `
  --eps "0,2,4,6" --steps "0,5,10" --alpha 1 `
  --out results\metrics\robust_sweep_val_debug_triobj.csv

# Comparison plots (baseline vs tri-obj)
& $PY -m src.eval.compare_robustness `
  --base_csv results\metrics\robust_sweep_val_debug.csv `
  --tri_csv  results\metrics\robust_sweep_val_debug_triobj.csv `
  --outdir   results\metrics

Write-Host "[done] Wrote tri-obj sweep + compare plots to results\metrics"
