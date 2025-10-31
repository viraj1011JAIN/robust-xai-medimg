import subprocess
import sys

import pytest

# Skip as xfail if gradcam backend isn’t importable in this environment
try:
    from src.xai import gradcam  # noqa: F401

    _GRADCAM_OK = True
except Exception:
    _GRADCAM_OK = False

requires_gradcam = pytest.mark.xfail(
    not _GRADCAM_OK, reason="gradcam backend not available in this environment", strict=False
)


@requires_gradcam
def test_gradcam_export_png(tmp_path):
    out_png = tmp_path / "gc.png"
    cmd = [
        sys.executable,
        "-m",
        "src.xai.export",
        "--config",
        "configs/tiny.yaml",
        "--out",
        str(out_png),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    assert res.returncode == 0, res.stdout + res.stderr
    assert out_png.exists() and out_png.stat().st_size > 0
