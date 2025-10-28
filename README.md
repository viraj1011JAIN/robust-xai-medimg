## Debug robustness sweeps (quick start)

Run the tri-objective robustness sweep on the debug split and generate comparison plots vs the baseline.

```powershell
# Use the newest checkpoint in results/checkpoints
# (tries: triobj_best.pt, best.pt, last.pt, best_weights.pt)
powershell -ExecutionPolicy Bypass -File tools\run_debug_sweeps.ps1

# Force recompute and/or point at a specific baseline CSV
powershell -ExecutionPolicy Bypass -File tools\run_debug_sweeps.ps1 -Force -BaseCsv results\metrics\robust_sweep_val_debug.csv

# (Optional) point at a specific checkpoint explicitly
powershell -ExecutionPolicy Bypass -File tools\run_debug_sweeps.ps1 -Ckpt results\checkpoints\triobj_best.pt

## Debug robustness sweeps (quick start)

Run the tri-objective robustness sweep on the debug split and generate comparison plots vs the baseline.

```powershell
# Use the newest checkpoint in results/checkpoints
# (tries: triobj_best.pt, best.pt, last.pt, best_weights.pt)
powershell -ExecutionPolicy Bypass -File tools\run_debug_sweeps.ps1

# Force recompute and/or point at a specific baseline CSV
powershell -ExecutionPolicy Bypass -File tools\run_debug_sweeps.ps1 -Force -BaseCsv results\metrics\robust_sweep_val_debug.csv

# (Optional) point at a specific checkpoint explicitly
powershell -ExecutionPolicy Bypass -File tools\run_debug_sweeps.ps1 -Ckpt results\checkpoints\triobj_best.pt


## Debug robustness sweeps (quick start)

Run the tri-objective robustness sweep on the debug split and generate comparison plots vs the baseline.

```powershell
# Use the newest checkpoint in results/checkpoints
# (tries: triobj_best.pt, best.pt, last.pt, best_weights.pt)
powershell -ExecutionPolicy Bypass -File tools\run_debug_sweeps.ps1

# Force recompute and/or point at a specific baseline CSV
powershell -ExecutionPolicy Bypass -File tools\run_debug_sweeps.ps1 -Force -BaseCsv results\metrics\robust_sweep_val_debug.csv

# (Optional) point at a specific checkpoint explicitly
powershell -ExecutionPolicy Bypass -File tools\run_debug_sweeps.ps1 -Ckpt results\checkpoints\triobj_best.pt

```
```

### Debug sweep outputs

**Tri-objective vs baseline (debug split)**  
![Delta heatmap](results/metrics/robust_compare_delta_heatmap.png)
![PGD10 comparison](results/metrics/robust_compare_pgd10.png)

**Tables & Data**
- Delta table (MD): [robust_compare_delta_table.md](results/metrics/robust_compare_delta_table.md)
- Tri-obj sweep CSV: [robust_sweep_val_debug_triobj.csv](results/metrics/robust_sweep_val_debug_triobj.csv)
- Tri-obj table (MD): [robust_sweep_val_debug_triobj_table.md](results/metrics/robust_sweep_val_debug_triobj_table.md)
