# VAL Robustness — Baseline vs Tri-Objective

**Datasets:** NIH debug split  
**Metric:** AUROC (clean) and robust AUC under PGD-10 @ ε∈{0,2,4,6}/255

![Delta Heatmap](./figures/xai_robust_val_delta_heatmap.png)

![PGD10 Lines](./figures/xai_robust_val_pgd10.png)

See detailed table: results/metrics/robust_compare_delta_table.md
### RQ1 — Adversarial Robustness (VAL, NIH Debug)

We compare a standard baseline against our tri-objective model under PGD-10
with ε ∈ {0, 2, 4, 6}/255. Figures \ref{fig:val-delta-heatmap} and
\ref{fig:val-pgd10} show consistent robustness gains across ε values with 
minimal impact on clean AUROC.

See Table: results/metrics/robust_compare_delta_table.md
