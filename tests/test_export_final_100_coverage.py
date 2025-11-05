"""
Complete 100% coverage tests for src/xai/export.py
Covers all branches including lines 128-132, 152->149, 156-188, 214-215, 219, 223, 251, 257->259
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image

from src.xai import export as E
from src.xai.gradcam import GradCAM

# ============================================================================
# Lines 128-132: _make_gc alternative constructor signatures
# ============================================================================


def test_make_gc_with_target_layer_kwarg():
    """Test _make_gc when GradCAM uses 'target_layer' instead of 'target_layer_name'."""
    model = E.build_model()

    # Create a mock that fails with target_layer_name, succeeds with target_layer
    original_gradcam = E.gradcam.GradCAM

    class AlternativeGradCAM:
        def __init__(self, model, target_layer=None):
            if target_layer is None:
                raise TypeError("missing required keyword argument: 'target_layer'")
            self.model = model
            self.target_layer = target_layer

    # Temporarily replace GradCAM
    E.gradcam.GradCAM = AlternativeGradCAM

    try:
        gc = E._make_gc(model)
        assert hasattr(gc, "model")
    finally:
        E.gradcam.GradCAM = original_gradcam


def test_make_gc_with_no_kwargs():
    """Test _make_gc when GradCAM accepts no keyword arguments."""
    model = E.build_model()

    original_gradcam = E.gradcam.GradCAM

    class NoKwargsGradCAM:
        def __init__(self, model):
            self.model = model

    E.gradcam.GradCAM = NoKwargsGradCAM

    try:
        gc = E._make_gc(model)
        assert hasattr(gc, "model")
    finally:
        E.gradcam.GradCAM = original_gradcam


def test_make_gc_fallback_to_get_gradcam():
    """Test _make_gc falls back to get_gradcam when GradCAM is not callable."""
    model = E.build_model()

    original_gradcam_class = E.gradcam.GradCAM
    original_get_gradcam = E.gradcam.get_gradcam

    # Make GradCAM not callable
    E.gradcam.GradCAM = None

    # Ensure get_gradcam works
    mock_gc = MagicMock()
    mock_gc.model = model
    E.gradcam.get_gradcam = lambda mod, layer: mock_gc

    try:
        gc = E._make_gc(model)
        assert gc is not None
    finally:
        E.gradcam.GradCAM = original_gradcam_class
        E.gradcam.get_gradcam = original_get_gradcam


# ============================================================================
# Lines 156-188: _run_generate alternative paths
# ============================================================================


def test_run_generate_with_generate_method_no_class_idx():
    """Test _run_generate when generate doesn't accept class_idx."""
    model = E.build_model()
    x = torch.randn(1, 3, 224, 224)
    expected_result = torch.randn(1, 224, 224)

    mock_gc = MagicMock()

    # generate raises TypeError with class_idx, works without
    def generate_func(x_arg, **kwargs):
        if "class_idx" in kwargs:
            raise TypeError("unexpected keyword argument 'class_idx'")
        return expected_result

    mock_gc.generate = generate_func

    result = E._run_generate(mock_gc, x)
    assert torch.equal(result, expected_result)


def test_run_generate_with_model_attribute_and_args():
    """Test _run_generate when gc.generate needs model as first argument."""
    model = E.build_model()
    x = torch.randn(1, 3, 224, 224)
    expected_result = torch.randn(1, 224, 224)

    mock_gc = MagicMock()
    mock_gc.model = model

    # generate needs (model, x, class_idx)
    def generate_func(*args, **kwargs):
        if len(args) == 1:  # Only x
            raise TypeError("missing required argument")
        # args should be (model, x, None) or (model, x)
        return expected_result

    mock_gc.generate = generate_func

    result = E._run_generate(mock_gc, x)
    assert torch.equal(result, expected_result)


def test_run_generate_with_model_attribute_no_class_idx():
    """Test _run_generate when gc.generate needs model but no class_idx."""
    model = E.build_model()
    x = torch.randn(1, 3, 224, 224)
    expected_result = torch.randn(1, 224, 224)

    mock_gc = MagicMock()
    mock_gc.model = model

    def generate_func(*args, **kwargs):
        if len(args) == 1:
            raise TypeError("missing model argument")
        if len(args) == 3:  # (model, x, None)
            raise TypeError("too many arguments")
        # Should work with (model, x)
        return expected_result

    mock_gc.generate = generate_func

    result = E._run_generate(mock_gc, x)
    assert torch.equal(result, expected_result)


def test_run_generate_callable_object_with_class_idx():
    """Test _run_generate when gc is callable and accepts class_idx."""
    x = torch.randn(1, 3, 224, 224)
    expected_result = torch.randn(1, 224, 224)

    def callable_gc(x_arg, class_idx=None):
        return expected_result

    result = E._run_generate(callable_gc, x)
    assert torch.equal(result, expected_result)


def test_run_generate_callable_object_no_class_idx():
    """Test _run_generate when gc is callable but doesn't accept class_idx."""
    x = torch.randn(1, 3, 224, 224)
    expected_result = torch.randn(1, 224, 224)

    def callable_gc(x_arg):
        return expected_result

    # First try with class_idx (should raise TypeError)
    # Then try without class_idx (should succeed)
    result = E._run_generate(callable_gc, x)
    assert torch.equal(result, expected_result)


def test_run_generate_fallback_to_gradcam_function():
    """Test _run_generate falls back to gradcam.gradcam() function."""
    x = torch.randn(1, 3, 224, 224)
    expected_result = torch.randn(1, 224, 224)

    # Create non-callable gc without generate method
    mock_gc = object()

    original_gradcam_func = getattr(E.gradcam, "gradcam", None)

    def mock_gradcam_func(gc, x_arg, class_idx=None):
        return expected_result

    E.gradcam.gradcam = mock_gradcam_func

    try:
        result = E._run_generate(mock_gc, x)
        assert torch.equal(result, expected_result)
    finally:
        if original_gradcam_func:
            E.gradcam.gradcam = original_gradcam_func


def test_run_generate_fallback_gradcam_no_class_idx():
    """Test _run_generate fallback to gradcam.gradcam() without class_idx."""
    x = torch.randn(1, 3, 224, 224)
    expected_result = torch.randn(1, 224, 224)

    mock_gc = object()

    original_gradcam_func = getattr(E.gradcam, "gradcam", None)

    def mock_gradcam_func(gc, x_arg, **kwargs):
        if "class_idx" in kwargs:
            raise TypeError("unexpected keyword")
        return expected_result

    E.gradcam.gradcam = mock_gradcam_func

    try:
        result = E._run_generate(mock_gc, x)
        assert torch.equal(result, expected_result)
    finally:
        if original_gradcam_func:
            E.gradcam.gradcam = original_gradcam_func


def test_run_generate_returns_non_tensor_skips():
    """Test _run_generate skips non-Tensor results and continues fallback."""
    x = torch.randn(1, 3, 224, 224)
    expected_result = torch.randn(1, 224, 224)

    # Create a simple object that's not a valid GradCAM but is callable
    class CallableObject:
        def __init__(self):
            self.model = E.build_model()

        def generate(self, *args, **kwargs):
            # Returns non-Tensor (should be skipped)
            return "not a tensor"

        def __call__(self, *args, **kwargs):
            # This should be called as fallback
            return expected_result

    mock_gc = CallableObject()
    result = E._run_generate(mock_gc, x)
    assert torch.equal(result, expected_result)


# ============================================================================
# Lines 214-215, 219, 223: save_gradcam_png heat dimension handling
# ============================================================================


def test_save_gradcam_png_2d_heat(tmp_path):
    """Test save_gradcam_png with 2D heatmap (line 219)."""
    model = E.build_model()
    x = torch.randn(1, 3, 224, 224)
    out_path = tmp_path / "cam_2d.png"

    # Mock _run_generate to return 2D tensor
    original_run_generate = E._run_generate
    E._run_generate = lambda gc, x: torch.rand(224, 224)

    try:
        E.save_gradcam_png(model, x, out_path)
        assert out_path.exists()
    finally:
        E._run_generate = original_run_generate


def test_save_gradcam_png_3d_heat(tmp_path):
    """Test save_gradcam_png with 3D heatmap (line 214-215)."""
    model = E.build_model()
    x = torch.randn(1, 3, 224, 224)
    out_path = tmp_path / "cam_3d.png"

    # Mock _run_generate to return 3D tensor
    original_run_generate = E._run_generate
    E._run_generate = lambda gc, x: torch.rand(1, 224, 224)

    try:
        E.save_gradcam_png(model, x, out_path)
        assert out_path.exists()
    finally:
        E._run_generate = original_run_generate


def test_save_gradcam_png_multichannel_heat(tmp_path):
    """Test save_gradcam_png with multi-channel heat needing averaging (line 223)."""
    model = E.build_model()
    x = torch.randn(1, 3, 224, 224)
    out_path = tmp_path / "cam_multi.png"

    # Mock _run_generate to return multi-channel heat
    original_run_generate = E._run_generate
    E._run_generate = lambda gc, x: torch.rand(1, 5, 224, 224)  # 5 channels

    try:
        E.save_gradcam_png(model, x, out_path)
        assert out_path.exists()
    finally:
        E._run_generate = original_run_generate


# ============================================================================
# Line 251: _load_one_from_cfg with train split
# ============================================================================


def test_load_one_from_cfg_train_split(tmp_path):
    """Test _load_one_from_cfg uses train_csv for 'train' split (line 251)."""
    # Create image
    img_path = tmp_path / "train_img.jpg"
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(img_path)

    # Create train CSV
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image_path,label\ntrain_img.jpg,1.0\n")

    # Create val CSV (empty)
    val_csv = tmp_path / "val.csv"
    val_csv.write_text("image_path,label\n")

    # Create config with both CSVs
    cfg_path = tmp_path / "config.yaml"
    cfg = OmegaConf.create(
        {
            "data": {
                "img_size": 64,
                "train_csv": str(train_csv),
                "val_csv": str(val_csv),
            }
        }
    )
    OmegaConf.save(cfg, cfg_path)

    # Load with train split - should use train_csv
    result = E._load_one_from_cfg(str(cfg_path), split="train")
    assert result.shape == (1, 3, 64, 64)


def test_load_one_from_cfg_val_split_default(tmp_path):
    """Test _load_one_from_cfg uses val_csv for 'val' split."""
    # Create image
    img_path = tmp_path / "val_img.jpg"
    Image.new("RGB", (64, 64), color=(200, 100, 50)).save(img_path)

    # Create val CSV
    val_csv = tmp_path / "val.csv"
    val_csv.write_text("image_path,label\nval_img.jpg,0.0\n")

    # Create config
    cfg_path = tmp_path / "config.yaml"
    cfg = OmegaConf.create(
        {
            "data": {
                "img_size": 64,
                "val_csv": str(val_csv),
            }
        }
    )
    OmegaConf.save(cfg, cfg_path)

    result = E._load_one_from_cfg(str(cfg_path), split="val")
    assert result.shape == (1, 3, 64, 64)


def test_load_one_from_cfg_fallback_to_val(tmp_path):
    """Test _load_one_from_cfg falls back to val_csv when train_csv missing (line 251)."""
    # Create image
    img_path = tmp_path / "fallback_img.jpg"
    Image.new("RGB", (32, 32), color=(50, 50, 50)).save(img_path)

    # Only create val CSV, no train CSV
    val_csv = tmp_path / "val.csv"
    val_csv.write_text("image_path,label\nfallback_img.jpg,0.5\n")

    # Create config without train_csv
    cfg_path = tmp_path / "config.yaml"
    cfg = OmegaConf.create(
        {
            "data": {
                "img_size": 32,
                "val_csv": str(val_csv),
            }
        }
    )
    OmegaConf.save(cfg, cfg_path)

    # Request train split but should fallback to val_csv
    result = E._load_one_from_cfg(str(cfg_path), split="train")
    assert result.shape == (1, 3, 32, 32)


# ============================================================================
# Line 257->259: _load_one_from_cfg with 3D tensor input
# ============================================================================


def test_load_one_from_cfg_3d_tensor_unsqueeze(tmp_path):
    """Test _load_one_from_cfg unsqueezes 3D tensor to 4D (line 257->259)."""
    # Create image
    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (48, 48), color=(128, 64, 192)).save(img_path)

    # Create CSV
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("image_path,label\nimg.jpg,1.0\n")

    # Create config
    cfg_path = tmp_path / "config.yaml"
    cfg = OmegaConf.create(
        {
            "data": {
                "img_size": 48,
                "val_csv": str(csv_path),
            }
        }
    )
    OmegaConf.save(cfg, cfg_path)

    result = E._load_one_from_cfg(str(cfg_path), split="val")

    # Should be 4D: [1, 3, H, W]
    assert result.ndim == 4
    assert result.shape[0] == 1
    assert result.shape == (1, 3, 48, 48)


# ============================================================================
# Additional comprehensive tests
# ============================================================================


def test_save_gradcam_png_with_resize(tmp_path):
    """Test save_gradcam_png when heatmap size differs from input."""
    model = E.build_model()
    x = torch.randn(1, 3, 224, 224)
    out_path = tmp_path / "cam_resize.png"

    # Mock to return smaller heatmap
    original_run_generate = E._run_generate
    E._run_generate = lambda gc, x: torch.rand(1, 1, 112, 112)

    try:
        E.save_gradcam_png(model, x, out_path)
        assert out_path.exists()
    finally:
        E._run_generate = original_run_generate


def test_save_gradcam_png_complete_flow(tmp_path):
    """Test complete save_gradcam_png flow with real GradCAM."""
    model = E.build_model()
    x = torch.randn(2, 3, 224, 224)
    out_path = tmp_path / "cam_complete.png"

    E.save_gradcam_png(model, x, out_path)
    assert out_path.exists()


def test_load_one_from_cfg_empty_dataset_raises(tmp_path):
    """Test _load_one_from_cfg raises error for empty dataset."""
    # Create empty CSV
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("image_path,label\n")

    # Create config
    cfg_path = tmp_path / "config.yaml"
    cfg = OmegaConf.create(
        {
            "data": {
                "img_size": 224,
                "val_csv": str(csv_path),
            }
        }
    )
    OmegaConf.save(cfg, cfg_path)

    with pytest.raises(RuntimeError, match="Dataset is empty"):
        E._load_one_from_cfg(str(cfg_path), split="val")


def test_real_gradcam_integration():
    """Integration test with real GradCAM."""
    model = E.build_model()
    gc = GradCAM(model, target_layer_name="layer4")
    x = torch.randn(1, 3, 224, 224)

    result = E._run_generate(gc, x)
    assert isinstance(result, torch.Tensor)
    assert result.shape[-2:] == (224, 224)


def test_save_npy_and_load_npy(tmp_path):
    """Test save_npy and load_npy functions."""
    arr = np.random.rand(10, 10).astype(np.float32)
    path = tmp_path / "test.npy"

    saved_path = E.save_npy(arr, path)
    assert saved_path.exists()

    loaded = E.load_npy(path)
    np.testing.assert_array_almost_equal(arr, loaded)


def test_save_heatmap_2d_array(tmp_path):
    """Test save_heatmap with 2D array."""
    arr = np.random.rand(64, 64).astype(np.float32)
    path = tmp_path / "heatmap.png"

    saved_path = E.save_heatmap(arr, path)
    assert saved_path.exists()


def test_save_heatmap_3d_tensor(tmp_path):
    """Test save_heatmap with 3D tensor."""
    tensor = torch.rand(1, 64, 64)
    path = tmp_path / "heatmap_tensor.png"

    saved_path = E.save_heatmap(tensor, path)
    assert saved_path.exists()


def test_save_json_and_load_json(tmp_path):
    """Test save_json and load_json functions."""
    data = {"test": 123, "nested": {"value": 456}}
    path = tmp_path / "test.json"

    saved_path = E.save_json(data, path)
    assert saved_path.exists()

    loaded = E.load_json(path)
    assert loaded == data


def test_save_csv_and_load_csv(tmp_path):
    """Test save_csv and load_csv functions."""
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    path = tmp_path / "test.csv"

    saved_path = E.save_csv(df, path, index=False)
    assert saved_path.exists()

    loaded = E.load_csv(path)
    pd.testing.assert_frame_equal(df, loaded)


def test_ensure_dir_creates_nested_directories(tmp_path):
    """Test ensure_dir creates nested directories."""
    nested_path = tmp_path / "a" / "b" / "c"
    result = E.ensure_dir(nested_path)

    assert result.exists()
    assert result.is_dir()
    assert result == nested_path


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.xai.export", "--cov-report=term-missing"])
