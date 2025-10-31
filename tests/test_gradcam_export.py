import subprocess
import sys


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
    assert out_png.exists()
    # should be non-empty PNG
    assert out_png.stat().st_size > 0
