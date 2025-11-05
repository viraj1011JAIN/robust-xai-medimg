# tests/test_export_more_edges.py
"""Additional edge case tests for export module."""

import torch

from src.xai import export as E


def test_save_gradcam_with_batch(tmp_path):
    """Test save_gradcam_png with batch > 1."""
    model = E.build_model()
    x = torch.rand(2, 3, 224, 224)
    out = tmp_path / "cam_batch.png"

    E.save_gradcam_png(model, x, out)
    assert out.exists()


def test_save_gradcam_different_sizes(tmp_path):
    """Test save_gradcam_png with different input sizes."""
    model = E.build_model()

    for size in [128, 256]:
        x = torch.rand(1, 3, size, size)
        out = tmp_path / f"cam_{size}.png"
        E.save_gradcam_png(model, x, out)
        assert out.exists()


def test_run_generate_with_real_gradcam():
    """Test _run_generate with real GradCAM object."""
    from src.xai.gradcam import GradCAM

    model = E.build_model()
    gc = GradCAM(model, target_layer_name="layer4")
    x = torch.rand(1, 3, 224, 224)

    result = E._run_generate(gc, x)
    assert isinstance(result, torch.Tensor)
    assert result.shape[-2:] == (224, 224)
