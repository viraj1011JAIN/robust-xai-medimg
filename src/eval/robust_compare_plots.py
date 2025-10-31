import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def _load_sweep(p: str) -> pd.DataFrame:
    """
    Accepts either our 'new' schema (attack, eps_255, steps, AUC_adv, ...)
    or the 'legacy' schema (eps, steps, auroc). Returns columns: eps_255, steps, AUC_adv.
    De-dups by (eps_255, steps) with mean if needed.
    """
    df = pd.read_csv(p)
    cols = set(df.columns)

    # Normalize column names
    if "eps_255" not in cols and "eps" in cols:
        df = df.rename(columns={"eps": "eps_255"})
    if "AUC_adv" not in cols and "auroc" in cols:
        df = df.rename(columns={"auroc": "AUC_adv"})

    needed = {"eps_255", "steps", "AUC_adv"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{p} is missing required columns {needed}, has {list(df.columns)}")

    out = df[["eps_255", "steps", "AUC_adv"]].copy()
    out = out.groupby(["eps_255", "steps"], as_index=False)["AUC_adv"].mean()
    return out.sort_values(["eps_255", "steps"]).reset_index(drop=True)


def _align(baseline: pd.DataFrame, triobj: pd.DataFrame) -> pd.DataFrame:
    """
    Inner-merge on (eps_255, steps). Result columns:
      eps_255, steps, AUC_adv_base, AUC_adv_tri
    """
    key = ["eps_255", "steps"]
    merged = baseline.merge(triobj, on=key, how="inner", suffixes=("_base", "_tri"))
    return merged


def _save(fig_path: str):
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[saved] {fig_path}")


def plot_delta_heatmap(both: pd.DataFrame, out_png: str) -> None:
    """
    both has columns: eps_255, steps, AUC_adv_base, AUC_adv_tri
    Saves heatmap of tri - base (Î”AUROC_adv).
    """
    both = both.copy()
    both["delta"] = both["AUC_adv_tri"] - both["AUC_adv_base"]
    pv = both.pivot(index="eps_255", columns="steps", values="delta").sort_index()
    plt.figure(figsize=(5.5, 3.8))
    im = plt.imshow(pv.values, aspect="auto")
    plt.xticks(range(len(pv.columns)), pv.columns)
    plt.yticks(range(len(pv.index)), pv.index)
    plt.xlabel("steps")
    plt.ylabel("Îµ (/255)")
    plt.title("Î”AUROC (tri âˆ’ baseline)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    _save(out_png)


def _plot_line_for_steps(both: pd.DataFrame, steps_val: int, title: str, out_png: str) -> None:
    s = both[both["steps"] == steps_val].sort_values("eps_255")
    if s.empty:
        print(f"[warn] no rows for steps={steps_val}; skip {out_png}")
        return
    plt.figure(figsize=(6, 4))
    plt.plot(s["eps_255"], s["AUC_adv_base"], marker="o", label="BASE")
    plt.plot(s["eps_255"], s["AUC_adv_tri"], marker="o", label="TRI")
    plt.xlabel("Îµ (/255)")
    plt.ylabel("AUC_adv")
    plt.title(title)
    plt.legend()
    _save(out_png)


def write_delta_table(both: pd.DataFrame, out_md: str) -> None:
    both = both.copy()
    both["delta"] = both["AUC_adv_tri"] - both["AUC_adv_base"]
    both = both.sort_values(["steps", "eps_255"])
    lines = [
        "| steps | Îµ (/255) | AUROC_base | AUROC_tri | Î” (tri - base) |",
        "|-----:|---------:|-----------:|----------:|---------------:|",
    ]
    for _, r in both.iterrows():
        lines.append(
            f"| {int(r['steps'])} | {int(r['eps_255'])} | "
            f"{r['AUC_adv_base']:.3f} | {r['AUC_adv_tri']:.3f} | {r['delta']:.3f} |"
        )
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {out_md}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_csv", required=True, help="baseline sweep CSV")
    ap.add_argument("--tri_csv", required=True, help="tri-objective sweep CSV")
    ap.add_argument("--outdir", default="results/metrics", help="directory for figures/tables")
    args = ap.parse_args()

    base = _load_sweep(args.base_csv)
    tri = _load_sweep(args.tri_csv)
    merged = _align(base, tri)

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Heatmap of Î”
    plot_delta_heatmap(merged, os.path.join(args.outdir, "robust_compare_delta_heatmap.png"))

    # 2) Line plots at specific steps
    _plot_line_for_steps(
        merged, 0, "FGSM (steps=0): AUC_adv vs Îµ", os.path.join(args.outdir, "robust_compare_fgsm.png")
    )
    _plot_line_for_steps(merged, 5, "PGD-5: AUC_adv vs Îµ", os.path.join(args.outdir, "robust_compare_pgd5.png"))
    _plot_line_for_steps(merged, 10, "PGD-10: AUC_adv vs Îµ", os.path.join(args.outdir, "robust_compare_pgd10.png"))
    _plot_line_for_steps(merged, 20, "PGD-20: AUC_adv vs Îµ", os.path.join(args.outdir, "robust_compare_pgd20.png"))

    # 3) Markdown delta table
    write_delta_table(merged, os.path.join(args.outdir, "robust_compare_delta_table.md"))

    print(f"[done] Wrote plots and table to: {args.outdir}")


if __name__ == "__main__":
    main()
