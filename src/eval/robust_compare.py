"""
Robustness comparison tool for adversarial evaluation sweeps.

This module compares baseline and tri-objective model performance across
different adversarial attack configurations (epsilon values and steps).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_sweep(path: str | Path) -> pd.DataFrame:
    """
    Load a robustness sweep CSV with flexible schema support.

    Supports both new schema (eps_255, steps, AUC_adv) and
    legacy schema (eps, steps, auroc).

    Args:
        path: Path to CSV file

    Returns:
        DataFrame with standardized columns: eps_255, steps, AUC_adv

    Raises:
        ValueError: If required columns are missing
    """
    df = pd.read_csv(path)

    # Drop attack column if present (not needed for comparison)
    if "attack" in df.columns:
        df = df.drop(columns=["attack"])

    # Standardize column names
    column_mapping = {}

    # Handle epsilon column
    if "eps_255" not in df.columns:
        if "eps" in df.columns:
            column_mapping["eps"] = "eps_255"
        else:
            raise ValueError(f"CSV at {path} missing required columns: need 'eps' or 'eps_255'")

    # Handle AUC/AUROC column
    if "AUC_adv" not in df.columns:
        if "auroc" in df.columns:
            column_mapping["auroc"] = "AUC_adv"
        elif "auc" in df.columns:
            column_mapping["auc"] = "AUC_adv"
        else:
            raise ValueError(
                f"CSV at {path} missing required columns: need 'AUC_adv', 'auroc', or 'auc'"
            )

    # Check for steps column
    if "steps" not in df.columns:
        raise ValueError(f"CSV at {path} missing required columns: need 'steps'")

    # Apply column renaming
    if column_mapping:
        df = df.rename(columns=column_mapping)

    # Keep only required columns
    df = df[["eps_255", "steps", "AUC_adv"]]

    # Handle duplicates by averaging
    df = df.groupby(["eps_255", "steps"], as_index=False).mean()

    return df


def _align(baseline: pd.DataFrame, triobj: pd.DataFrame) -> pd.DataFrame:
    """
    Align baseline and tri-objective DataFrames on (eps_255, steps).

    Args:
        baseline: Baseline model results
        triobj: Tri-objective model results

    Returns:
        DataFrame with columns: eps_255, steps, AUC_adv_base, AUC_adv_tri
    """
    # Merge on eps_255 and steps (inner join keeps only matching rows)
    merged = baseline.merge(
        triobj, on=["eps_255", "steps"], suffixes=("_base", "_tri"), how="inner"
    )

    return merged


def plot_delta_heatmap(both: pd.DataFrame, out_path: str | Path) -> None:
    """
    Create a heatmap showing delta (tri - base) across epsilon and steps.

    Args:
        both: Aligned DataFrame with AUC_adv_base and AUC_adv_tri columns
        out_path: Output path for PNG file
    """
    import matplotlib.pyplot as plt

    # Ensure output directory exists
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate delta
    both = both.copy()
    both["delta"] = both["AUC_adv_tri"] - both["AUC_adv_base"]

    # Create pivot table for heatmap
    pivot = both.pivot(index="steps", columns="eps_255", values="delta")

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", interpolation="nearest")

    # Set ticks and labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(x)}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{int(x)}" for x in pivot.index])

    ax.set_xlabel("ε (/255)")
    ax.set_ylabel("Steps")
    ax.set_title("AUROC Improvement (Tri-obj - Baseline)")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Δ AUROC", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_line_for_steps(
    both: pd.DataFrame, steps_val: int, title: str, out_path: str | Path
) -> None:
    """
    Create a line plot comparing baseline vs tri-objective for a specific steps value.

    Args:
        both: Aligned DataFrame with AUC_adv_base and AUC_adv_tri columns
        steps_val: Filter for this specific steps value
        title: Plot title
        out_path: Output path for PNG file
    """
    import matplotlib.pyplot as plt

    # Filter for specific steps value
    subset = both[both["steps"] == steps_val].copy()

    if subset.empty:
        print(f"[warn] no rows for steps={steps_val}")
        return

    # Sort by epsilon for smooth line
    subset = subset.sort_values("eps_255")

    # Ensure output directory exists
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Create line plot
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(subset["eps_255"], subset["AUC_adv_base"], marker="o", label="Baseline", linewidth=2)
    ax.plot(
        subset["eps_255"], subset["AUC_adv_tri"], marker="s", label="Tri-objective", linewidth=2
    )

    ax.set_xlabel("ε (/255)")
    ax.set_ylabel("AUROC")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_delta_table(both: pd.DataFrame, out_path: str | Path) -> None:
    """
    Write a markdown table showing the comparison results.

    Args:
        both: Aligned DataFrame with AUC_adv_base and AUC_adv_tri columns
        out_path: Output path for markdown file
    """
    # Ensure output directory exists
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate delta
    both = both.copy()
    both["delta"] = both["AUC_adv_tri"] - both["AUC_adv_base"]

    # Sort for consistent output
    both = both.sort_values(["steps", "eps_255"])

    # Write markdown table
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Robustness Comparison Results\n\n")
        f.write("| steps | ε (/255) | AUROC_base | AUROC_tri | Δ (tri - base) |\n")
        f.write("|-----:|---------:|-----------:|----------:|---------------:|\n")

        for _, row in both.iterrows():
            f.write(
                f"| {int(row['steps']):5d} | "
                f"{int(row['eps_255']):8d} | "
                f"{row['AUC_adv_base']:10.3f} | "
                f"{row['AUC_adv_tri']:9.3f} | "
                f"{row['delta']:14.3f} |\n"
            )


def main() -> None:
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Compare baseline and tri-objective robustness sweeps"
    )
    parser.add_argument("--base_csv", required=True, help="Path to baseline sweep CSV")
    parser.add_argument("--tri_csv", required=True, help="Path to tri-objective sweep CSV")
    parser.add_argument(
        "--outdir", default="results/metrics", help="Output directory for plots and tables"
    )

    args = parser.parse_args()

    # Load data
    baseline = _load_sweep(args.base_csv)
    triobj = _load_sweep(args.tri_csv)

    # Align datasets
    both = _align(baseline, triobj)

    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Generate outputs
    plot_delta_heatmap(both, outdir / "robust_compare_delta_heatmap.png")
    write_delta_table(both, outdir / "robust_compare_delta_table.md")

    # Generate line plots for common step values
    step_configs = [
        (0, "FGSM (steps=0)", "robust_compare_fgsm.png"),
        (5, "PGD-5 Comparison", "robust_compare_pgd5.png"),
        (10, "PGD-10 Comparison", "robust_compare_pgd10.png"),
        (20, "PGD-20 Comparison", "robust_compare_pgd20.png"),
    ]

    for steps_val, title, filename in step_configs:
        _plot_line_for_steps(both, steps_val, title, outdir / filename)

    print(f"[done] Wrote plots and table to: {outdir}")


if __name__ == "__main__":  # pragma: no cover
    main()
