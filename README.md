# From repo root
$md = @'
## Debug robustness sweeps (quick start)

Run the tri-objective robustness sweep on the debug split and generate comparison plots vs the baseline.

```powershell
# Use newest checkpoint found in results/checkpoints (triobj_best.pt, best.pt, last.pt, best_weights.pt)
powershell -ExecutionPolicy Bypass -File tools\run_debug_sweeps.ps1

# Force recompute and/or point at a specific baseline CSV
powershell -ExecutionPolicy Bypass -File tools\run_debug_sweeps.ps1 -Force -BaseCsv results\metrics\robust_sweep_val_debug.csv

# (Optional) point at a specific checkpoint explicitly
powershell -ExecutionPolicy Bypass -File tools\run_debug_sweeps.ps1 -Ckpt results\checkpoints\triobj_best.pt
rn
