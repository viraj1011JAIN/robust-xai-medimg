import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def _load_sweep(path: str) -> pd.DataFrame:
    """
    Load either:
      - NEW schema: columns include ['attack','eps_255','steps','AUC_adv',...]
      - LEGACY     : columns include ['eps','steps','auroc',...]
    Returns a tidy frame with columns: ['eps_255','steps','AUC_adv'] deduped.
    """
    df = pd.read_csv(path)

    # Normalize column names we need
    if {"eps_255", "steps", "AUC_adv"}.issubset(df.columns):
        out = df[["eps_255", "steps", "AUC_adv"]].copy()
    elif {"eps", "steps", "auroc"}.issubset(df.columns):
        out = df.rename(columns={"eps": "eps_255", "auroc": "AUC_adv"})[["eps_255", "steps", "AUC_adv"]].copy()
    else:
        raise ValueError(f"CSV {path} does not contain required columns. " f"Found: {list(df.columns)}")

    # Ensure numeric
    out["eps_255"] = pd.to_numeric(out["eps_255"], errors="coerce")
    out["steps"] = pd.to_numeric(out["steps"], errors="coerce")
    out["AUC_adv"] = pd.to_numeric(out["AUC_adv"], errors="coerce")

    # De-dup in case multiple rows per (eps_255, steps) exist (take mean)
    out = (
        out.groupby(["eps_255", "steps"], as_index=False)["AUC_adv"]
        .mean()
        .sort_values(["eps_255", "steps"])
        .reset_index(drop=True)
    )
    return out


def _align(base: pd.DataFrame, tri: pd.DataFrame) -> pd.DataFrame:
    """
    Align on (eps_255, steps) and return a single DataFrame with:
      ['eps_255','steps','AUC_adv_base','AUC_adv_tri']
    """
    m = base.merge(tri, on=["eps_255", "steps"], how="inner", suffixes=("_base", "_tri"))
    # numeric and sorted
    m = m.sort_values(["steps", "eps_255"]).reset_index(drop=True)
    return m


def plot_delta_heatmap(both: pd.DataFrame, out_png: str) -> None:
    """
    both has columns: eps_255, steps, AUC_adv_base, AUC_adv_tri
    Saves heatmap of tri - base (Î”AUROC_adv).
    """
    both = both.copy()
    both["delta"] = both["AUC_adv_tri"] - both["AUC_adv_base"]
    pv = both.pivot(index="eps_255", columns="steps", values="delta").sort_index()

    plt.figure(figsize=(5.6, 3.8), dpi=160)
    im = plt.imshow(pv.values, aspect="auto")
    plt.xticks(range(len(pv.columns)), pv.columns)
    plt.yticks(range(len(pv.index)), pv.index)
    plt.xlabel("steps")
    plt.ylabel("Îµ (/255)")
    plt.title("Î”AUROC_adv (tri âˆ’ base)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def _plot_lines_for_steps(both: pd.DataFrame, steps: int, out_png: str, title: str) -> None:
    """Helper to draw a 2-line chart (base vs tri) at a fixed steps value."""
    s = both[both["steps"] == steps].sort_values("eps_255")
    plt.figure(figsize=(6.2, 4.2), dpi=160)
    plt.plot(s["eps_255"], s["AUC_adv_base"], marker="o", label="BASE")
    plt.plot(s["eps_255"], s["AUC_adv_tri"], marker="o", label="TRI")
    plt.xlabel("Îµ (/255)")
    plt.ylabel("AUC_adv")
    plt.title(title)
    plt.legend()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_pgd10_lines(both: pd.DataFrame, out_png: str) -> None:
    _plot_lines_for_steps(both, steps=10, out_png=out_png, title="PGD-10: AUC_adv vs Îµ")


def plot_fgsm_lines(both: pd.DataFrame, out_png: str) -> None:
    _plot_lines_for_steps(both, steps=0, out_png=out_png, title="FGSM: AUC_adv vs Îµ")


def write_delta_table(both: pd.DataFrame, out_md: str) -> None:
    both = both.copy()
    both["delta"] = both["AUC_adv_tri"] - both["AUC_adv_base"]
    both = both.sort_values(["steps", "eps_255"])

    lines = [
        "| steps | Îµ (/255) | AUROC_base | AUROC_tri | Î” (tri âˆ’ base) |",
        "|-----:|---------:|-----------:|----------:|---------------:|",
    ]
    for _, r in both.iterrows():
        lines.append(
            f"| {int(r['steps'])} | {int(r['eps_255'])} | "
            f"{r['AUC_adv_base']:.3f} | {r['AUC_adv_tri']:.3f} | "
            f"{(r['AUC_adv_tri']-r['AUC_adv_base']):.3f} |"
        )
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_csv", required=True, help="baseline sweep CSV")
    ap.add_argument("--tri_csv", required=True, help="tri-objective sweep CSV")
    ap.add_argument("--outdir", default="results/metrics", help="where to write figures/tables")
    args = ap.parse_args()

    base = _load_sweep(args.base_csv)
    tri = _load_sweep(args.tri_csv)
    both = _align(base, tri)  # columns: eps_255, steps, AUC_adv_base, AUC_adv_tri

    os.makedirs(args.outdir, exist_ok=True)
    plot_delta_heatmap(both, os.path.join(args.outdir, "robust_compare_delta_heatmap.png"))
    plot_pgd10_lines(both, os.path.join(args.outdir, "robust_compare_pgd10.png"))
    plot_fgsm_lines(both, os.path.join(args.outdir, "robust_compare_fgsm.png"))
    write_delta_table(both, os.path.join(args.outdir, "robust_compare_delta_table.md"))
    print("[saved] heatmap + PGD10 + FGSM plots and delta table in", args.outdir)


if __name__ == "__main__":
    main()
