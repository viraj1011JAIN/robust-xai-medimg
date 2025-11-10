# tests/test_markdown_report.py
"""
Tests for src/eval/markdown_report.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.eval.markdown_report import (
    REQUIRED_COLS,
    _load,
    _merge_and_compute,
    _require_cols,
    _write_outputs,
    main,
)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _toy_df(rows):
    # Minimal valid schema
    return pd.DataFrame.from_records(
        rows,
        columns=["attack", "eps_255", "steps", "AUC_clean", "AUC_adv", "AUC_drop"],
    )


def test_require_cols_ok():
    df = _toy_df([("pgd", 2, 10, 0.9, 0.8, 0.1)])
    # should not raise
    _require_cols(df, "ok")


def test_require_cols_missing_raises():
    df = pd.DataFrame({"attack": ["pgd"]})
    with pytest.raises(ValueError) as ei:
        _require_cols(df, "bad")
    msg = str(ei.value)
    assert "bad is missing required columns" in msg
    # at least one known missing col mentioned
    assert "eps_255" in msg or "AUC_adv" in msg


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        _load("nope.csv", "base")


def test_merge_and_compute_unicode_and_rounding():
    base = _toy_df(
        [
            ("pgd", 2, 10, 0.95, 0.85, 0.10),
            ("pgd", 4, 10, 0.92, 0.80, 0.12),
            ("pgd", 2, 20, 0.93, 0.83, 0.10),
        ]
    )
    tri = _toy_df(
        [
            ("pgd", 2, 10, 0.95, 0.87, 0.08),
            ("pgd", 4, 10, 0.92, 0.82, 0.09),
            ("pgd", 2, 20, 0.93, 0.85, 0.06),
        ]
    )

    out_df, d_adv_col, d_drop_col = _merge_and_compute(base, tri, use_ascii=False, round_ndigits=3)
    # Column names with Δ
    assert d_adv_col in out_df.columns and d_drop_col in out_df.columns
    assert "ΔAUC_adv" in out_df.columns and "Δdrop" in out_df.columns

    # Correct deltas
    # For eps=2,steps=10: adv 0.87-0.85 = 0.02 ; drop 0.08-0.10 = -0.02
    row = out_df[(out_df["eps_255"] == 2) & (out_df["steps"] == 10)].iloc[0]
    assert abs(row["ΔAUC_adv"] - 0.02) < 1e-6
    assert abs(row["Δdrop"] - (-0.02)) < 1e-6

    # Sorted (eps, steps, attack)
    values = out_df[["eps_255", "steps", "attack"]].to_records(index=False)
    assert tuple(values[0]) <= tuple(values[-1])


def test_merge_and_compute_ascii_headers():
    base = _toy_df([("pgd", 2, 10, 0.95, 0.85, 0.10)])
    tri = _toy_df([("pgd", 2, 10, 0.95, 0.87, 0.08)])

    out_df, d_adv_col, d_drop_col = _merge_and_compute(base, tri, use_ascii=True, round_ndigits=2)
    assert d_adv_col == "DeltaAUC_adv" and d_drop_col == "DeltaDrop"
    assert set(["DeltaAUC_adv", "DeltaDrop"]).issubset(out_df.columns)
    # values rounded to 2 decimals
    assert out_df["DeltaAUC_adv"].iloc[0] == 0.02


def test_write_outputs_creates_files(temp_dir):
    out_md = os.path.join(temp_dir, "sub/rep.md")
    out_df = pd.DataFrame({"a": [1], "b": [2]})
    sidecar = _write_outputs(out_df, out_md, "Title")

    assert os.path.exists(out_md)
    assert os.path.exists(sidecar)

    with open(out_md, "r", encoding="utf-8") as f:
        text = f.read()
    assert "# Title" in text
    assert "|   a |   b |" in text or "a   b" in text  # markdown table header presence


def test_main_end_to_end_unicode(temp_dir, monkeypatch, capsys):
    # Build CSVs
    base = _toy_df(
        [
            ("pgd", 2, 10, 0.95, 0.85, 0.10),
            ("pgd", 4, 10, 0.92, 0.80, 0.12),
        ]
    )
    tri = _toy_df(
        [
            ("pgd", 2, 10, 0.95, 0.87, 0.08),
            ("pgd", 4, 10, 0.92, 0.82, 0.09),
        ]
    )
    base_csv = os.path.join(temp_dir, "base.csv")
    tri_csv = os.path.join(temp_dir, "tri.csv")
    out_md = os.path.join(temp_dir, "report.md")
    base.to_csv(base_csv, index=False)
    tri.to_csv(tri_csv, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "markdown_report.py",
            "--base",
            base_csv,
            "--tri",
            tri_csv,
            "--out",
            out_md,
            "--title",
            "My Report",
        ],
    )
    main()

    # outputs
    assert os.path.exists(out_md)
    sidecar_csv = os.path.splitext(out_md)[0] + "_delta.csv"
    assert os.path.exists(sidecar_csv)

    # Check Δ headers appear (unicode mode)
    with open(out_md, "r", encoding="utf-8") as f:
        md = f.read()
    assert "ΔAUC_adv" in md and "Δdrop" in md

    # stdout lines
    out = capsys.readouterr().out
    assert "[report] wrote markdown:" in out
    assert "[report] wrote csv:" in out


def test_main_ascii_headers(temp_dir, monkeypatch):
    base = _toy_df([("pgd", 2, 10, 0.95, 0.85, 0.10)])
    tri = _toy_df([("pgd", 2, 10, 0.95, 0.87, 0.08)])
    base_csv = os.path.join(temp_dir, "base.csv")
    tri_csv = os.path.join(temp_dir, "tri.csv")
    out_md = os.path.join(temp_dir, "report.md")
    base.to_csv(base_csv, index=False)
    tri.to_csv(tri_csv, index=False)

    monkeypatch.setattr(
        "sys.argv",
        ["markdown_report.py", "--base", base_csv, "--tri", tri_csv, "--out", out_md, "--ascii"],
    )
    main()

    with open(out_md, "r", encoding="utf-8") as f:
        md = f.read()
    assert "DeltaAUC_adv" in md and "DeltaDrop" in md
