import torch

from src.xai.export import build_model, save_gradcam_png


def test_gradcam_save_direct(tmp_path):
    model = build_model("resnet18")
    x = torch.randn(2, 3, 64, 64)
    out_png = tmp_path / "gc_direct.png"
    save_gradcam_png(model, x, out_png)
    assert out_png.exists()
    assert out_png.stat().st_size > 0
