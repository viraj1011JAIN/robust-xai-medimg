import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _load_sweep(p: str) -> pd.DataFrame:
    """
    Accepts either our 'new' schema (attack, eps_255, steps, AUC_adv, ...)
    or the 'legacy' schema (eps, steps, auroc). Returns columns: eps_255, steps, AUC_adv.
    """
    df = pd.read_csv(p)
    if {"eps_255", "steps", "AUC_adv"}.issubset(df.columns):
        out = df[["eps_255", "steps", "AUC_adv"]].copy()
    else:
        # legacy
        df = df.rename(columns={"eps": "eps_255", "auroc": "AUC_adv"})
        out = df[["eps_255", "steps", "AUC_adv"]].copy()
    # de-dup per (eps_255, steps)
    out = out.groupby(["eps_255", "steps"], as_index=False)["AUC_adv"].mean()
    return out.sort_values(["eps_255", "steps"]).reset_index(drop=True)


def _align(baseline: pd.DataFrame, triobj: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    key = ["eps_255", "steps"]
    merged = baseline.merge(triobj, on=key, how="inner", suffixes=("_base", "_tri"))
    return merged[[*key, "AUC_adv_base"]], merged[[*key, "AUC_adv_tri"]]


def plot_delta_heatmap(both: pd.DataFrame, out_png: str) -> None:
    """
    both has columns: eps_255, steps, AUC_adv_base, AUC_adv_tri
    Saves heatmap of tri - base (ΔAUROC_adv).
    """
    both = both.copy()
    both["delta"] = both["AUC_adv_tri"] - both["AUC_adv_base"]
    pv = both.pivot(index="eps_255", columns="steps", values="delta").sort_index()
    plt.figure(figsize=(5, 3.5), dpi=160)
    im = plt.imshow(pv.values, aspect="auto")
    plt.xticks(range(len(pv.columns)), pv.columns)
    plt.yticks(range(len(pv.index)), pv.index)
    plt.xlabel("steps")
    plt.ylabel("epsilon (/255)")
    plt.title("ΔAUROC (tri-objective − baseline)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)


def plot_pgd10_lines(both: pd.DataFrame, out_png: str) -> None:
    """
    Line plot at steps==10 (PGD-10): AUROC vs ε for baseline vs tri-obj.
    """
    s = both[both["steps"] == 10].sort_values("eps_255")
    plt.figure(figsize=(6, 4), dpi=160)
    plt.plot(s["eps_255"], s["AUC_adv_base"], marker="o", label="baseline")
    plt.plot(s["eps_255"], s["AUC_adv_tri"], marker="o", label="tri-objective")
    plt.xlabel("epsilon (/255)")
    plt.ylabel("AUROC (adv)")
    plt.title("PGD-10: AUROC vs epsilon")
    plt.legend()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)


def write_delta_table(both: pd.DataFrame, out_md: str) -> None:
    both = both.copy()
    both["delta"] = both["AUC_adv_tri"] - both["AUC_adv_base"]
    both = both.sort_values(["steps", "eps_255"])
    lines = [
        "| steps | ε (/255) | AUROC_base | AUROC_tri | Δ (tri - base) |",
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_csv", required=True, help="baseline sweep CSV")
    ap.add_argument("--tri_csv", required=True, help="tri-objective sweep CSV")
    ap.add_argument(
        "--outdir",
        default="results/metrics",
        help="output directory for figures/tables",
    )
    args = ap.parse_args()

    base = _load_sweep(args.base_csv)
    tri = _load_sweep(args.tri_csv)
    merged = base.merge(tri, on=["eps_255", "steps"], suffixes=("_base", "_tri"))

    os.makedirs(args.outdir, exist_ok=True)
    plot_delta_heatmap(merged, os.path.join(args.outdir, "robust_compare_delta_heatmap.png"))
    plot_pgd10_lines(merged, os.path.join(args.outdir, "robust_compare_pgd10.png"))
    write_delta_table(merged, os.path.join(args.outdir, "robust_compare_delta_table.md"))
    print("[saved] delta heatmap, PGD10 lines, and table in", args.outdir)


if __name__ == "__main__":
    main()
