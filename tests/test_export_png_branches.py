# tests/test_export_png_branches.py
"""Tests for save_gradcam_png branches."""

import torch

from src.xai import export as E


def test_export_gradcam_2d_heat(tmp_path):
    """Test save_gradcam_png with 2D heatmap."""
    model = E.build_model()
    x = torch.rand(1, 3, 224, 224)
    out = tmp_path / "cam_2d.png"

    E.save_gradcam_png(model, x, out)
    assert out.exists()


def test_export_gradcam_3d_heat(tmp_path):
    """Test save_gradcam_png with 3D heatmap (batch)."""
    model = E.build_model()
    x = torch.rand(2, 3, 224, 224)
    out = tmp_path / "cam_3d.png"

    E.save_gradcam_png(model, x, out)
    assert out.exists()


def test_export_gradcam_resize_needed(tmp_path):
    """Test save_gradcam_png when resize is needed."""
    model = E.build_model()
    x = torch.rand(1, 3, 128, 128)
    out = tmp_path / "cam_resize.png"

    E.save_gradcam_png(model, x, out)
    assert out.exists()


def test_export_gradcam_averaging_channels(tmp_path):
    """Test save_gradcam_png with multi-channel averaging."""
    model = E.build_model()
    x = torch.rand(1, 3, 256, 256)
    out = tmp_path / "cam_avg.png"

    E.save_gradcam_png(model, x, out)
    assert out.exists()
