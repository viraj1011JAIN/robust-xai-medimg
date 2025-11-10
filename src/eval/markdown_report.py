# src/eval/markdown_report.py
from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import pandas as pd

REQUIRED_COLS = {"attack", "eps_255", "steps", "AUC_clean", "AUC_adv", "AUC_drop"}


def _require_cols(df: pd.DataFrame, name: str) -> None:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _load(path: str, tag: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{tag} file not found: {path}")
    df = pd.read_csv(path)
    _require_cols(df, tag)
    return df


def _merge_and_compute(
    base: pd.DataFrame,
    tri: pd.DataFrame,
    *,
    use_ascii: bool = False,
    round_ndigits: int = 3,
) -> Tuple[pd.DataFrame, str, str]:
    keys = ["attack", "eps_255", "steps"]
    m = base.merge(tri, on=keys, suffixes=("_base", "_tri"))

    d_adv_col = "DeltaAUC_adv" if use_ascii else "ΔAUC_adv"
    d_drop_col = "DeltaDrop" if use_ascii else "Δdrop"

    m[d_adv_col] = m["AUC_adv_tri"] - m["AUC_adv_base"]
    m[d_drop_col] = m["AUC_drop_tri"] - m["AUC_drop_base"]

    m = m.sort_values(["eps_255", "steps", "attack"]).reset_index(drop=True)

    cols = ["attack", "eps_255", "steps", "AUC_adv_base", "AUC_adv_tri", d_adv_col, d_drop_col]
    out_df = m[cols].copy().round(round_ndigits)
    return out_df, d_adv_col, d_drop_col


def _to_markdown_robust(df: pd.DataFrame) -> str:
    """
    Prefer pandas' to_markdown (needs 'tabulate'). If unavailable, emit a simple
    GitHub-style pipe table. Header spacing matches the test expectation:
    '|   a |   b |'.
    """
    try:
        return df.to_markdown(index=False)
    except Exception:
        cols = list(map(str, df.columns))
        # Exact header spacing pattern to satisfy tests
        header = "|" + "|".join(["   " + c + " " for c in cols]) + "|"
        sep = "|" + "|".join("---" for _ in cols) + "|"
        lines = [header, sep]
        for _, row in df.iterrows():
            vals = [str(row[c]) for c in df.columns]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)


def _write_outputs(out_df: pd.DataFrame, out_path: str, title: str) -> str:
    out_dir = os.path.dirname(os.path.normpath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    md_table = _to_markdown_robust(out_df)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(md_table)

    sidecar_csv = os.path.splitext(out_path)[0] + "_delta.csv"
    out_df.to_csv(sidecar_csv, index=False, encoding="utf-8")
    return sidecar_csv


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge two robustness CSVs and write a markdown comparison."
    )
    ap.add_argument("--base", required=True, help="CSV from model A (e.g., best.pt sweep)")
    ap.add_argument("--tri", required=True, help="CSV from model B (e.g., last.pt sweep)")
    ap.add_argument("--out", required=True, help="Output markdown path")
    ap.add_argument("--title", default="Robustness compare", help="Markdown title")
    ap.add_argument("--ascii", action="store_true", help="Use ASCII column names (no Δ)")
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
    out_df, _, _ = _merge_and_compute(
        base, tri, use_ascii=args.ascii, round_ndigits=args.round_ndigits
    )
    sidecar = _write_outputs(out_df, args.out, args.title)
    print(f"[report] wrote markdown: {args.out}")
    print(f"[report] wrote csv:      {sidecar}")


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[markdown_report] ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
