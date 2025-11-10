"""
Complete tests for src/eval/robust_compare_plots.py to achieve 100% coverage.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.eval.robust_compare_plots import (
    _align,
    _load_sweep,
    _plot_line_for_steps,
    _save,
    main,
    plot_delta_heatmap,
    write_delta_table,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_df():
    """Sample DataFrame with robustness sweep data."""
    return pd.DataFrame(
        {
            "eps_255": [2, 4, 8, 2, 4, 8],
            "steps": [10, 10, 10, 20, 20, 20],
            "AUC_adv": [0.85, 0.80, 0.70, 0.83, 0.78, 0.68],
        }
    )


@pytest.fixture
def aligned_df():
    """Sample aligned DataFrame for plotting."""
    return pd.DataFrame(
        {
            "eps_255": [2, 4, 8, 2, 4, 8],
            "steps": [10, 10, 10, 20, 20, 20],
            "AUC_adv_base": [0.85, 0.80, 0.70, 0.83, 0.78, 0.68],
            "AUC_adv_tri": [0.87, 0.82, 0.72, 0.85, 0.80, 0.70],
        }
    )


class TestLoadSweep:
    """Test _load_sweep function."""

    def test_load_new_schema(self, temp_dir):
        """Test loading CSV with new schema."""
        df = pd.DataFrame(
            {
                "attack": ["pgd", "pgd"],
                "eps_255": [2, 4],
                "steps": [10, 10],
                "AUC_adv": [0.85, 0.80],
            }
        )
        path = os.path.join(temp_dir, "new_schema.csv")
        df.to_csv(path, index=False)

        result = _load_sweep(path)

        assert set(result.columns) == {"eps_255", "steps", "AUC_adv"}
        assert len(result) == 2

    def test_load_legacy_schema(self, temp_dir):
        """Test loading CSV with legacy schema."""
        df = pd.DataFrame({"eps": [2, 4], "steps": [10, 10], "auroc": [0.82, 0.77]})
        path = os.path.join(temp_dir, "legacy_schema.csv")
        df.to_csv(path, index=False)

        result = _load_sweep(path)

        assert set(result.columns) == {"eps_255", "steps", "AUC_adv"}
        assert len(result) == 2
        assert result["eps_255"].tolist() == [2, 4]

    def test_load_missing_columns(self, temp_dir):
        """Test that missing required columns raises ValueError."""
        df = pd.DataFrame({"eps": [2, 4], "steps": [10, 10]})  # Missing AUC
        path = os.path.join(temp_dir, "bad_schema.csv")
        df.to_csv(path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            _load_sweep(path)

    def test_load_missing_eps_column(self, temp_dir):
        """Test that missing eps column raises ValueError."""
        df = pd.DataFrame({"steps": [10, 10], "AUC_adv": [0.82, 0.77]})
        path = os.path.join(temp_dir, "no_eps.csv")
        df.to_csv(path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            _load_sweep(path)

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

        assert len(result) == 2
        row = result[(result["eps_255"] == 2) & (result["steps"] == 10)]
        assert len(row) == 1
        assert abs(row["AUC_adv"].iloc[0] - 0.85) < 0.01

    def test_sorted_output(self, temp_dir):
        """Test that output is sorted by eps_255 and steps."""
        df = pd.DataFrame(
            {
                "eps_255": [8, 2, 4],
                "steps": [20, 10, 10],
                "AUC_adv": [0.70, 0.85, 0.80],
            }
        )
        path = os.path.join(temp_dir, "unsorted.csv")
        df.to_csv(path, index=False)

        result = _load_sweep(path)

        # Should be sorted
        assert result["eps_255"].tolist() == [2, 4, 8]
        assert result["steps"].tolist() == [10, 10, 20]


class TestAlign:
    """Test _align function."""

    def test_align_basic(self):
        """Test basic alignment."""
        baseline = pd.DataFrame(
            {
                "eps_255": [2, 4, 8],
                "steps": [10, 10, 10],
                "AUC_adv": [0.85, 0.80, 0.70],
            }
        )
        triobj = pd.DataFrame(
            {
                "eps_255": [2, 4, 8],
                "steps": [10, 10, 10],
                "AUC_adv": [0.87, 0.82, 0.72],
            }
        )

        result = _align(baseline, triobj)

        assert len(result) == 3
        assert set(result.columns) == {
            "eps_255",
            "steps",
            "AUC_adv_base",
            "AUC_adv_tri",
        }

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

        result = _align(baseline, triobj)

        # Should only have 2 rows (inner join)
        assert len(result) == 2


class TestSave:
    """Test _save function."""

    def test_save_creates_directory(self, temp_dir):
        """Test that _save creates parent directories."""
        import matplotlib.pyplot as plt

        fig_path = os.path.join(temp_dir, "subdir", "test.png")

        plt.figure()
        plt.plot([1, 2, 3], [1, 2, 3])

        _save(fig_path)

        assert os.path.exists(fig_path)
        assert os.path.exists(os.path.dirname(fig_path))

    def test_save_prints_message(self, temp_dir, capsys):
        """Test that _save prints confirmation message."""
        import matplotlib.pyplot as plt

        fig_path = os.path.join(temp_dir, "test.png")

        plt.figure()
        plt.plot([1, 2, 3], [1, 2, 3])

        _save(fig_path)

        captured = capsys.readouterr()
        assert "[saved]" in captured.out
        assert fig_path in captured.out


class TestPlotDeltaHeatmap:
    """Test plot_delta_heatmap function."""

    def test_creates_file(self, temp_dir, aligned_df, capsys):
        """Test that heatmap file is created."""
        out_path = os.path.join(temp_dir, "heatmap.png")

        plot_delta_heatmap(aligned_df, out_path)

        assert os.path.exists(out_path)
        captured = capsys.readouterr()
        assert "[saved]" in captured.out

    def test_creates_subdirectory(self, temp_dir, aligned_df):
        """Test that subdirectories are created."""
        out_path = os.path.join(temp_dir, "subdir", "heatmap.png")

        plot_delta_heatmap(aligned_df, out_path)

        assert os.path.exists(out_path)


class TestPlotLineForSteps:
    """Test _plot_line_for_steps function."""

    def test_creates_file_with_data(self, temp_dir, aligned_df, capsys):
        """Test that line plot is created when data exists."""
        out_path = os.path.join(temp_dir, "line.png")

        _plot_line_for_steps(aligned_df, 10, "Test Plot", out_path)

        assert os.path.exists(out_path)
        captured = capsys.readouterr()
        assert "[saved]" in captured.out

    def test_empty_dataframe_no_file(self, temp_dir, aligned_df, capsys):
        """Test that no file is created when DataFrame is empty for steps."""
        out_path = os.path.join(temp_dir, "line.png")

        # Request steps=99 which doesn't exist in the data
        _plot_line_for_steps(aligned_df, 99, "Test Plot", out_path)

        # File should not be created
        assert not os.path.exists(out_path)

        # Should print warning
        captured = capsys.readouterr()
        assert "[warn] no rows for steps=99" in captured.out

    def test_multiple_steps_values(self, temp_dir, aligned_df):
        """Test creating plots for different step values."""
        for steps_val in [10, 20]:
            out_path = os.path.join(temp_dir, f"line_{steps_val}.png")
            _plot_line_for_steps(aligned_df, steps_val, f"Steps={steps_val}", out_path)
            assert os.path.exists(out_path)


class TestWriteDeltaTable:
    """Test write_delta_table function."""

    def test_creates_file(self, temp_dir, aligned_df, capsys):
        """Test that markdown table is created."""
        out_path = os.path.join(temp_dir, "table.md")

        write_delta_table(aligned_df, out_path)

        assert os.path.exists(out_path)
        captured = capsys.readouterr()
        assert "[saved]" in captured.out

    def test_table_content(self, temp_dir, aligned_df):
        """Test markdown table content."""
        out_path = os.path.join(temp_dir, "table.md")

        write_delta_table(aligned_df, out_path)

        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check header (handle potential unicode encoding issues)
        assert "| steps |" in content
        assert "| AUROC_base | AUROC_tri |" in content
        # Check separator
        assert "|-----:|" in content
        # Check data presence
        assert "0.850" in content or "0.85" in content

    def test_delta_calculation(self, temp_dir):
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

    def test_creates_subdirectory(self, temp_dir, aligned_df):
        """Test that subdirectories are created."""
        out_path = os.path.join(temp_dir, "subdir", "table.md")

        write_delta_table(aligned_df, out_path)

        assert os.path.exists(out_path)


class TestMain:
    """Test main function."""

    def test_main_with_args(self, monkeypatch, temp_dir, capsys):
        """Test main function with command-line arguments."""
        # Create test CSVs
        base_csv = os.path.join(temp_dir, "base.csv")
        tri_csv = os.path.join(temp_dir, "tri.csv")
        outdir = os.path.join(temp_dir, "output")

        base_df = pd.DataFrame(
            {
                "eps_255": [2, 4, 2, 4],
                "steps": [10, 10, 20, 20],
                "AUC_adv": [0.85, 0.80, 0.83, 0.78],
            }
        )
        tri_df = pd.DataFrame(
            {
                "eps_255": [2, 4, 2, 4],
                "steps": [10, 10, 20, 20],
                "AUC_adv": [0.87, 0.82, 0.85, 0.80],
            }
        )
        base_df.to_csv(base_csv, index=False)
        tri_df.to_csv(tri_csv, index=False)

        # Mock sys.argv
        monkeypatch.setattr(
            "sys.argv",
            [
                "robust_compare_plots.py",
                "--base_csv",
                base_csv,
                "--tri_csv",
                tri_csv,
                "--outdir",
                outdir,
            ],
        )

        main()

        # Check outputs were created
        assert os.path.exists(os.path.join(outdir, "robust_compare_delta_heatmap.png"))
        assert os.path.exists(os.path.join(outdir, "robust_compare_pgd10.png"))
        assert os.path.exists(os.path.join(outdir, "robust_compare_delta_table.md"))

        captured = capsys.readouterr()
        assert "[done] Wrote plots and table to:" in captured.out

    def test_main_default_outdir(self, monkeypatch, temp_dir):
        """Test main with default output directory."""
        base_csv = os.path.join(temp_dir, "base.csv")
        tri_csv = os.path.join(temp_dir, "tri.csv")

        base_df = pd.DataFrame({"eps_255": [2, 4], "steps": [10, 10], "AUC_adv": [0.85, 0.80]})
        tri_df = pd.DataFrame({"eps_255": [2, 4], "steps": [10, 10], "AUC_adv": [0.87, 0.82]})
        base_df.to_csv(base_csv, index=False)
        tri_df.to_csv(tri_csv, index=False)

        # Change to temp dir for default outdir
        original_dir = os.getcwd()
        try:
            os.chdir(temp_dir)

            monkeypatch.setattr(
                "sys.argv",
                ["robust_compare_plots.py", "--base_csv", base_csv, "--tri_csv", tri_csv],
            )

            main()

            # Check default output location
            assert os.path.exists("results/metrics/robust_compare_delta_heatmap.png")
        finally:
            os.chdir(original_dir)

    def test_main_prints_completion(self, monkeypatch, temp_dir, capsys):
        """Test that main prints completion message."""
        base_csv = os.path.join(temp_dir, "base.csv")
        tri_csv = os.path.join(temp_dir, "tri.csv")
        outdir = os.path.join(temp_dir, "output")

        base_df = pd.DataFrame({"eps_255": [2], "steps": [10], "AUC_adv": [0.85]})
        tri_df = pd.DataFrame({"eps_255": [2], "steps": [10], "AUC_adv": [0.87]})
        base_df.to_csv(base_csv, index=False)
        tri_df.to_csv(tri_csv, index=False)

        monkeypatch.setattr(
            "sys.argv",
            [
                "robust_compare_plots.py",
                "--base_csv",
                base_csv,
                "--tri_csv",
                tri_csv,
                "--outdir",
                outdir,
            ],
        )

        main()

        captured = capsys.readouterr()
        assert "[done] Wrote plots and table to:" in captured.out

    def test_main_with_steps_0_5_20(self, monkeypatch, temp_dir):
        """Test main generates all plot variants including steps 0, 5, 20."""
        base_csv = os.path.join(temp_dir, "base.csv")
        tri_csv = os.path.join(temp_dir, "tri.csv")
        outdir = os.path.join(temp_dir, "output")

        # Create data with steps 0, 5, 10, 20
        base_df = pd.DataFrame(
            {
                "eps_255": [2, 4, 2, 4, 2, 4, 2, 4],
                "steps": [0, 0, 5, 5, 10, 10, 20, 20],
                "AUC_adv": [0.90, 0.88, 0.85, 0.83, 0.85, 0.80, 0.83, 0.78],
            }
        )
        tri_df = pd.DataFrame(
            {
                "eps_255": [2, 4, 2, 4, 2, 4, 2, 4],
                "steps": [0, 0, 5, 5, 10, 10, 20, 20],
                "AUC_adv": [0.92, 0.90, 0.87, 0.85, 0.87, 0.82, 0.85, 0.80],
            }
        )
        base_df.to_csv(base_csv, index=False)
        tri_df.to_csv(tri_csv, index=False)

        monkeypatch.setattr(
            "sys.argv",
            [
                "robust_compare_plots.py",
                "--base_csv",
                base_csv,
                "--tri_csv",
                tri_csv,
                "--outdir",
                outdir,
            ],
        )

        main()

        # Check all plots were created
        assert os.path.exists(os.path.join(outdir, "robust_compare_fgsm.png"))
        assert os.path.exists(os.path.join(outdir, "robust_compare_pgd5.png"))
        assert os.path.exists(os.path.join(outdir, "robust_compare_pgd10.png"))
        assert os.path.exists(os.path.join(outdir, "robust_compare_pgd20.png"))

    def test_main_missing_steps_warning(self, monkeypatch, temp_dir, capsys):
        """Test that missing steps values generate warnings."""
        base_csv = os.path.join(temp_dir, "base.csv")
        tri_csv = os.path.join(temp_dir, "tri.csv")
        outdir = os.path.join(temp_dir, "output")

        # Create data with only steps=10 (missing 0, 5, 20)
        base_df = pd.DataFrame({"eps_255": [2, 4], "steps": [10, 10], "AUC_adv": [0.85, 0.80]})
        tri_df = pd.DataFrame({"eps_255": [2, 4], "steps": [10, 10], "AUC_adv": [0.87, 0.82]})
        base_df.to_csv(base_csv, index=False)
        tri_df.to_csv(tri_csv, index=False)

        monkeypatch.setattr(
            "sys.argv",
            [
                "robust_compare_plots.py",
                "--base_csv",
                base_csv,
                "--tri_csv",
                tri_csv,
                "--outdir",
                outdir,
            ],
        )

        main()

        captured = capsys.readouterr()
        # Should have warnings for missing steps
        assert "[warn] no rows for steps=0" in captured.out
        assert "[warn] no rows for steps=5" in captured.out
        assert "[warn] no rows for steps=20" in captured.out

    def test_main_as_script(self, temp_dir, monkeypatch):
        """Test the __main__ block by executing the file."""
        import importlib.util

        base_csv = os.path.join(temp_dir, "base.csv")
        tri_csv = os.path.join(temp_dir, "tri.csv")
        outdir = os.path.join(temp_dir, "output")

        base_df = pd.DataFrame({"eps_255": [2], "steps": [10], "AUC_adv": [0.85]})
        tri_df = pd.DataFrame({"eps_255": [2], "steps": [10], "AUC_adv": [0.87]})
        base_df.to_csv(base_csv, index=False)
        tri_df.to_csv(tri_csv, index=False)

        # Mock sys.argv for __main__ execution
        monkeypatch.setattr(
            "sys.argv",
            [
                "robust_compare_plots.py",
                "--base_csv",
                base_csv,
                "--tri_csv",
                tri_csv,
                "--outdir",
                outdir,
            ],
        )

        # Load and execute the module as __main__
        spec = importlib.util.spec_from_file_location(
            "__main__", "src/eval/robust_compare_plots.py"
        )
        module = importlib.util.module_from_spec(spec)

        # Execute the module (this will trigger if __name__ == "__main__")
        try:
            spec.loader.exec_module(module)
        except SystemExit:
            pass  # main() might call sys.exit()

        # Verify output was created
        assert os.path.exists(os.path.join(outdir, "robust_compare_delta_heatmap.png"))
