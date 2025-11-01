import argparse
import os
import sys

import pandas as pd

REQUIRED_COLS = {"attack", "eps_255", "steps", "AUC_clean", "AUC_adv", "AUC_drop"}


def _require_cols(df: pd.DataFrame, name: str):
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _load(path: str, tag: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{tag} file not found: {path}")
    df = pd.read_csv(path)
    _require_cols(df, tag)
    return df


def main():
    ap = argparse.ArgumentParser(
        description="Merge two robustness CSVs and write a markdown comparison."
    )
    ap.add_argument(
        "--base", required=True, help="CSV from model A (e.g., best.pt sweep)"
    )
    ap.add_argument(
        "--tri", required=True, help="CSV from model B (e.g., last.pt sweep)"
    )
    ap.add_argument("--out", required=True, help="Output markdown path")
    ap.add_argument("--title", default="Robustness compare", help="Markdown title")
    ap.add_argument(
        "--ascii", action="store_true", help="Use ASCII column names (no Δ)"
    )
    ap.add_argument(
        "--round",
        dest="round_ndigits",
        type=int,
        default=3,
        help="Round numeric columns to N digits",
    )
    args = ap.parse_args()

    base = _load(args.base, "base")
    tri = _load(args.tri, "tri")

    # Merge on sweep keys
    keys = ["attack", "eps_255", "steps"]
    m = base.merge(tri, on=keys, suffixes=("_base", "_tri"))

    # Compute deltas
    if args.ascii:
        d_adv_col = "DeltaAUC_adv"
        d_drop_col = "DeltaDrop"
    else:
        d_adv_col = "ΔAUC_adv"
        d_drop_col = "Δdrop"

    m[d_adv_col] = m["AUC_adv_tri"] - m["AUC_adv_base"]
    m[d_drop_col] = m["AUC_drop_tri"] - m["AUC_drop_base"]

    # Sort nicely
    m = m.sort_values(["eps_255", "steps", "attack"]).reset_index(drop=True)

    # Select & round for the table
    cols = [
        "attack",
        "eps_255",
        "steps",
        "AUC_adv_base",
        "AUC_adv_tri",
        d_adv_col,
        d_drop_col,
    ]
    out_df = m[cols].copy()
    out_df = out_df.round(args.round_ndigits)

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.normpath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Write markdown (UTF-8 so Unicode works on Windows too)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# {args.title}\n\n")
        f.write(out_df.to_markdown(index=False))

    # Also write a sidecar CSV with the same data (easier for spreadsheets)
    sidecar_csv = os.path.splitext(args.out)[0] + "_delta.csv"
    out_df.to_csv(sidecar_csv, index=False, encoding="utf-8")

    print(f"[report] wrote markdown: {args.out}")
    print(f"[report] wrote csv:      {sidecar_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[markdown_report] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
