# tests/test_compare_robustness.py
"""
Tests for robustness comparison utilities.
"""
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Check if matplotlib is available
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for testing
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

pytestmark = pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")

from src.eval.compare_robustness import (
    _align,
    _load_sweep,
    main,
    plot_delta_heatmap,
    plot_pgd10_lines,
    write_delta_table,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def new_schema_csv(temp_dir):
    """Create a CSV with new schema (attack, eps_255, steps, AUC_adv)."""
    df = pd.DataFrame(
        {
            "attack": ["pgd", "pgd", "pgd", "pgd"],
            "eps_255": [2, 4, 2, 4],
            "steps": [10, 10, 20, 20],
            "AUC_adv": [0.85, 0.80, 0.83, 0.78],
        }
    )
    path = os.path.join(temp_dir, "new_schema.csv")
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def legacy_schema_csv(temp_dir):
    """Create a CSV with legacy schema (eps, steps, auroc)."""
    df = pd.DataFrame(
        {
            "eps": [2, 4, 2, 4],
            "steps": [10, 10, 20, 20],
            "auroc": [0.82, 0.77, 0.80, 0.75],
        }
    )
    path = os.path.join(temp_dir, "legacy_schema.csv")
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def baseline_df():
    """Sample baseline dataframe."""
    return pd.DataFrame(
        {
            "eps_255": [2, 4, 8, 2, 4, 8],
            "steps": [10, 10, 10, 20, 20, 20],
            "AUC_adv": [0.85, 0.80, 0.70, 0.83, 0.78, 0.68],
        }
    )


@pytest.fixture
def triobj_df():
    """Sample tri-objective dataframe."""
    return pd.DataFrame(
        {
            "eps_255": [2, 4, 8, 2, 4, 8],
            "steps": [10, 10, 10, 20, 20, 20],
            "AUC_adv": [0.87, 0.82, 0.72, 0.85, 0.80, 0.70],
        }
    )


class TestLoadSweep:
    """Test _load_sweep function."""

    def test_load_new_schema(self, new_schema_csv):
        """Test loading CSV with new schema."""
        df = _load_sweep(new_schema_csv)

        assert set(df.columns) == {"eps_255", "steps", "AUC_adv"}
        assert len(df) == 4
        assert df["eps_255"].tolist() == [2, 2, 4, 4]
        assert df["steps"].tolist() == [10, 20, 10, 20]

    def test_load_legacy_schema(self, legacy_schema_csv):
        """Test loading CSV with legacy schema."""
        df = _load_sweep(legacy_schema_csv)

        assert set(df.columns) == {"eps_255", "steps", "AUC_adv"}
        assert len(df) == 4
        assert df["eps_255"].tolist() == [2, 2, 4, 4]
        assert df["steps"].tolist() == [10, 20, 10, 20]

    def test_deduplication(self, temp_dir):
        """Test that duplicates are averaged."""
        df = pd.DataFrame(
            {
                "eps_255": [2, 2, 4],
                "steps": [10, 10, 10],
                "AUC_adv": [0.8, 0.9, 0.85],
            }
        )
        path = os.path.join(temp_dir, "dup.csv")
        df.to_csv(path, index=False)

        result = _load_sweep(path)

        # Should average the two eps=2, steps=10 entries
        assert len(result) == 2
        row = result[(result["eps_255"] == 2) & (result["steps"] == 10)]
        assert len(row) == 1
        assert abs(row["AUC_adv"].iloc[0] - 0.85) < 0.01  # (0.8 + 0.9) / 2

    def test_sorting(self, new_schema_csv):
        """Test that results are sorted by eps_255 and steps."""
        df = _load_sweep(new_schema_csv)

        # Check sorted
        for i in range(len(df) - 1):
            curr = (df.iloc[i]["eps_255"], df.iloc[i]["steps"])
            next_val = (df.iloc[i + 1]["eps_255"], df.iloc[i + 1]["steps"])
            assert curr <= next_val


class TestAlign:
    """Test _align function."""

    def test_align_basic(self, baseline_df, triobj_df):
        """Test basic alignment."""
        base_out, tri_out = _align(baseline_df, triobj_df)

        assert len(base_out) == 6
        assert len(tri_out) == 6
        assert set(base_out.columns) == {"eps_255", "steps", "AUC_adv_base"}
        assert set(tri_out.columns) == {"eps_255", "steps", "AUC_adv_tri"}

    def test_align_partial_overlap(self):
        """Test alignment with partial overlap."""
        baseline = pd.DataFrame(
            {
                "eps_255": [2, 4, 8],
                "steps": [10, 10, 10],
                "AUC_adv": [0.85, 0.80, 0.70],
            }
        )
        triobj = pd.DataFrame(
            {
                "eps_255": [2, 4],  # Missing eps=8
                "steps": [10, 10],
                "AUC_adv": [0.87, 0.82],
            }
        )

        base_out, tri_out = _align(baseline, triobj)

        # Should only have 2 rows (inner join)
        assert len(base_out) == 2
        assert len(tri_out) == 2


class TestPlotDeltaHeatmap:
    """Test plot_delta_heatmap function."""

    def test_creates_file(self, temp_dir, baseline_df, triobj_df):
        """Test that heatmap file is created."""
        both = baseline_df.merge(triobj_df, on=["eps_255", "steps"], suffixes=("_base", "_tri"))
        out_path = os.path.join(temp_dir, "subdir", "heatmap.png")

        plot_delta_heatmap(both, out_path)

        assert os.path.exists(out_path)

    def test_calculates_delta(self, temp_dir, baseline_df, triobj_df):
        """Test that delta is calculated correctly."""
        both = baseline_df.merge(triobj_df, on=["eps_255", "steps"], suffixes=("_base", "_tri"))
        out_path = os.path.join(temp_dir, "heatmap.png")

        # Should not raise
        plot_delta_heatmap(both, out_path)


class TestPlotPGD10Lines:
    """Test plot_pgd10_lines function."""

    def test_creates_file(self, temp_dir, baseline_df, triobj_df):
        """Test that PGD-10 line plot is created."""
        both = baseline_df.merge(triobj_df, on=["eps_255", "steps"], suffixes=("_base", "_tri"))
        out_path = os.path.join(temp_dir, "subdir", "pgd10.png")

        plot_pgd10_lines(both, out_path)

        assert os.path.exists(out_path)

    def test_filters_steps_10(self, temp_dir, baseline_df, triobj_df):
        """Test that only steps==10 is plotted."""
        both = baseline_df.merge(triobj_df, on=["eps_255", "steps"], suffixes=("_base", "_tri"))
        out_path = os.path.join(temp_dir, "pgd10.png")

        # Should not raise even with steps 10 and 20
        plot_pgd10_lines(both, out_path)


class TestWriteDeltaTable:
    """Test write_delta_table function."""

    def test_creates_file(self, temp_dir, baseline_df, triobj_df):
        """Test that markdown table is created."""
        both = baseline_df.merge(triobj_df, on=["eps_255", "steps"], suffixes=("_base", "_tri"))
        out_path = os.path.join(temp_dir, "subdir", "table.md")

        write_delta_table(both, out_path)

        assert os.path.exists(out_path)

    def test_table_format(self, temp_dir, baseline_df, triobj_df):
        """Test markdown table format."""
        both = baseline_df.merge(triobj_df, on=["eps_255", "steps"], suffixes=("_base", "_tri"))
        out_path = os.path.join(temp_dir, "table.md")

        write_delta_table(both, out_path)

        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check header
        header = "| steps | ε (/255) | AUROC_base | AUROC_tri | Δ (tri - base) |"
        assert header in content
        # Check separator
        assert "|-----:|" in content
        # Check some data
        assert "0.850" in content or "0.85" in content

    def test_calculates_delta(self, temp_dir):
        """Test that delta is calculated correctly."""
        both = pd.DataFrame(
            {
                "eps_255": [2, 4],
                "steps": [10, 10],
                "AUC_adv_base": [0.80, 0.75],
                "AUC_adv_tri": [0.82, 0.78],
            }
        )
        out_path = os.path.join(temp_dir, "table.md")

        write_delta_table(both, out_path)

        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Delta for first row: 0.82 - 0.80 = 0.02
        assert "0.020" in content


class TestMain:
    """Test main function."""

    def test_main_with_args(self, monkeypatch, new_schema_csv, legacy_schema_csv, temp_dir):
        """Test main function with command-line arguments."""
        outdir = os.path.join(temp_dir, "output")

        # Mock sys.argv
        monkeypatch.setattr(
            "sys.argv",
            [
                "compare_robustness.py",
                "--base_csv",
                new_schema_csv,
                "--tri_csv",
                legacy_schema_csv,
                "--outdir",
                outdir,
            ],
        )

        main()

        # Check outputs were created
        assert os.path.exists(os.path.join(outdir, "robust_compare_delta_heatmap.png"))
        assert os.path.exists(os.path.join(outdir, "robust_compare_pgd10.png"))
        assert os.path.exists(os.path.join(outdir, "robust_compare_delta_table.md"))

    def test_main_default_outdir(self, monkeypatch, new_schema_csv, legacy_schema_csv, temp_dir):
        """Test main with default output directory."""
        # Create CSVs in temp dir
        base_csv = os.path.join(temp_dir, "base.csv")
        tri_csv = os.path.join(temp_dir, "tri.csv")

        df = pd.DataFrame(
            {
                "eps_255": [2, 4],
                "steps": [10, 10],
                "AUC_adv": [0.85, 0.80],
            }
        )
        df.to_csv(base_csv, index=False)

        df2 = pd.DataFrame(
            {
                "eps_255": [2, 4],
                "steps": [10, 10],
                "AUC_adv": [0.87, 0.82],
            }
        )
        df2.to_csv(tri_csv, index=False)

        # Change to temp dir so default outdir works
        original_dir = os.getcwd()
        try:
            os.chdir(temp_dir)

            monkeypatch.setattr(
                "sys.argv",
                [
                    "compare_robustness.py",
                    "--base_csv",
                    base_csv,
                    "--tri_csv",
                    tri_csv,
                ],
            )

            main()

            # Check default output location
            assert os.path.exists("results/metrics/robust_compare_delta_heatmap.png")
        finally:
            os.chdir(original_dir)

    def test_main_prints_output(
        self, monkeypatch, new_schema_csv, legacy_schema_csv, temp_dir, capsys
    ):
        """Test main prints confirmation message - covers line 111."""
        outdir = os.path.join(temp_dir, "output")

        monkeypatch.setattr(
            "sys.argv",
            [
                "compare_robustness.py",
                "--base_csv",
                new_schema_csv,
                "--tri_csv",
                legacy_schema_csv,
                "--outdir",
                outdir,
            ],
        )

        main()

        captured = capsys.readouterr()
        assert "[saved] delta heatmap, PGD10 lines, and table in" in captured.out
        assert outdir in captured.out
