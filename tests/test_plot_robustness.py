# tests/test_plot_robustness.py
import os
import runpy
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Use non-interactive backend for headless/CI
try:
    import matplotlib

    matplotlib.use("Agg")
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

pytestmark = pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")

from src.eval.plot_robustness import main as plot_main


@pytest.fixture
def tmpdir_path():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _run_and_assert(csv_path: Path, out_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["plot_robustness.py", "--csv", str(csv_path), "--out", str(out_path)],
    )
    plot_main()
    assert out_path.exists() and out_path.stat().st_size > 0
    out = capsys.readouterr().out
    assert "[saved]" in out and str(out_path) in out


def test_new_schema_branch(tmpdir_path, monkeypatch, capsys):
    df = pd.DataFrame(
        {
            "attack": ["pgd"] * 4,
            "eps_255": [2, 4, 2, 4],
            "steps": [10, 10, 20, 20],
            "AUC_adv": [0.85, 0.80, 0.83, 0.78],
        }
    )
    csv = tmpdir_path / "new.csv"
    df.to_csv(csv, index=False)
    out_png = tmpdir_path / "plot_new.png"
    _run_and_assert(csv, out_png, monkeypatch, capsys)


def test_legacy_schema_branch(tmpdir_path, monkeypatch, capsys):
    df = pd.DataFrame(
        {
            "eps": [2, 4, 2, 4],
            "steps": [10, 10, 20, 20],
            "auroc": [0.82, 0.77, 0.80, 0.75],
        }
    )
    csv = tmpdir_path / "legacy.csv"
    df.to_csv(csv, index=False)
    out_png = tmpdir_path / "plot_legacy.png"
    _run_and_assert(csv, out_png, monkeypatch, capsys)


def test_invalid_schema_raises(tmpdir_path, monkeypatch):
    df = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
    csv = tmpdir_path / "bad.csv"
    df.to_csv(csv, index=False)
    out_png = tmpdir_path / "plot_bad.png"

    monkeypatch.setattr(
        "sys.argv",
        ["plot_robustness.py", "--csv", str(csv), "--out", str(out_png)],
    )
    with pytest.raises(ValueError, match="does not contain expected columns"):
        plot_main()


def test_runs_via_module_main(tmpdir_path, monkeypatch, capsys):
    """
    Cover the __main__ guard in src.eval.plot_robustness
    (line 41 in your coverage report).
    """
    # Prepare a valid "new schema" CSV
    df = pd.DataFrame(
        {
            "attack": ["pgd"] * 3,
            "eps_255": [2, 4, 8],
            "steps": [10, 10, 10],
            "AUC_adv": [0.85, 0.80, 0.72],
        }
    )
    csv = tmpdir_path / "as_main.csv"
    df.to_csv(csv, index=False)
    out_png = tmpdir_path / "as_main.png"

    # Simulate running the module as a script
    monkeypatch.setattr(
        "sys.argv",
        ["plot_robustness.py", "--csv", str(csv), "--out", str(out_png)],
    )
    runpy.run_module("src.eval.plot_robustness", run_name="__main__")

    assert out_png.exists() and out_png.stat().st_size > 0
    out = capsys.readouterr().out
    assert "[saved]" in out and str(out_png) in out
