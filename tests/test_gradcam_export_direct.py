import pytest
import torch

from src.xai.export import build_model, save_gradcam_png

# Skip as xfail if gradcam backend isn’t importable in this environment
try:
    from src.xai import gradcam  # noqa: F401

    _GRADCAM_OK = True
except Exception:
    _GRADCAM_OK = False

requires_gradcam = pytest.mark.xfail(
    not _GRADCAM_OK,
    reason="gradcam backend not available in this environment",
    strict=False,
)


@requires_gradcam
def test_gradcam_save_direct(tmp_path):
    model = build_model("resnet18")
    x = torch.randn(2, 3, 64, 64)
    out_png = tmp_path / "gc_direct.png"
    save_gradcam_png(model, x, out_png)
    assert out_png.exists()
    assert out_png.stat().st_size > 0
