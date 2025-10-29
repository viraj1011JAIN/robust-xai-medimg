import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # unify schema
    if {"eps_255", "steps", "AUC_adv"}.issubset(df.columns):
        eps_col, steps_col, y_col = "eps_255", "steps", "AUC_adv"
    elif {"eps", "steps", "auroc"}.issubset(df.columns):  # old schema
        eps_col, steps_col, y_col = "eps", "steps", "auroc"
    else:
        raise ValueError("CSV does not contain expected columns.")

    pivot = df.pivot_table(index=eps_col, columns=steps_col, values=y_col, aggfunc="mean").sort_index()

    plt.figure(figsize=(6, 4.5))
    for steps in pivot.columns:
        plt.plot(pivot.index, pivot[steps], marker="o", label=f"steps={steps}")
    plt.xlabel("epsilon (/255)")
    plt.ylabel("AUROC (adv)")
    plt.title("Robustness sweep (AUROC vs ε)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
