# tests/test_export_missing_branches.py
"""Complete tests for src.xai.export module - 100% coverage."""

import io
import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.xai import export as E


def test_load_npy_and_constant_heatmap(tmp_path):
    """Test loading numpy arrays and saving constant heatmaps."""
    # Test load_npy
    arr = np.ones((8, 8), dtype="float32")
    p = tmp_path / "a.npy"
    np.save(p, arr)
    out = E.load_npy(p)
    assert out.shape == (8, 8), "Loaded array should be 8x8"

    # Test save_heatmap with constant array (min==max normalization branch)
    p_png = tmp_path / "h.png"
    E.save_heatmap(arr, p_png)
    img = Image.open(io.BytesIO(p_png.read_bytes()))
    assert img.size == (8, 8), "Heatmap should be 8x8"


def test_save_load_csv_without_pandas_guard(monkeypatch, tmp_path):
    """Test CSV operations when pandas is missing."""
    # Simulate missing pandas to hit guard branches
    original_pd = E.pd
    monkeypatch.setattr(E, "pd", None, raising=False)
    df = pd.DataFrame({"id": [1], "s": [0.5]})

    with pytest.raises(RuntimeError, match="pandas"):
        E.save_csv(df, tmp_path / "x.csv")

    with pytest.raises(RuntimeError, match="pandas"):
        E.load_csv(tmp_path / "x.csv")

    # Restore
    monkeypatch.setattr(E, "pd", original_pd, raising=False)


def test_save_heatmap_with_varying_values(tmp_path):
    """Test heatmap normalization with varying values."""
    arr = np.array([[0.0, 0.5], [0.8, 1.0]], dtype="float32")
    p = tmp_path / "varying.png"
    E.save_heatmap(arr, p)
    img = Image.open(p)
    assert img.size == (2, 2), "Heatmap should preserve dimensions"


def test_save_heatmap_with_torch_tensor(tmp_path):
    """Test heatmap with torch tensor input."""
    t = torch.rand(8, 8)
    p = tmp_path / "torch_hm.png"
    E.save_heatmap(t, p)
    assert p.exists(), "Heatmap file should exist"


def test_save_heatmap_with_3d_tensor(tmp_path):
    """Test heatmap with 3D tensor (1,H,W)."""
    t = torch.rand(1, 8, 8)
    p = tmp_path / "3d_hm.png"
    E.save_heatmap(t, p)
    assert p.exists(), "Heatmap file should exist"


def test_save_heatmap_invalid_shape_raises(tmp_path):
    """Test that invalid shape raises ValueError."""
    arr = np.random.rand(3, 8, 8)  # Invalid: 3 channels
    with pytest.raises(ValueError, match="expects"):
        E.save_heatmap(arr, tmp_path / "bad.png")


def test_load_json(tmp_path):
    """Test JSON load function."""
    data = {"test": 123, "nested": {"value": 456}}
    p = tmp_path / "test.json"
    E.save_json(data, p)

    loaded = E.load_json(p)
    assert loaded["test"] == 123
    assert loaded["nested"]["value"] == 456


def test_save_json_custom_indent(tmp_path):
    """Test JSON with custom indent."""
    data = {"a": 1, "b": 2}
    p = tmp_path / "indent.json"
    E.save_json(data, p, indent=4)

    content = p.read_text()
    assert "a" in content
    assert json.loads(content) == data


def test_build_model():
    """Test model building."""
    model = E.build_model("resnet18")
    assert model is not None
    assert hasattr(model, "fc")
    assert model.fc.out_features == 1


def test_save_gradcam_png_invalid_dims_raises(tmp_path):
    """Test that non-4D input raises ValueError."""
    model = E.build_model()
    x = torch.rand(224, 224)  # Wrong dims

    with pytest.raises(ValueError, match="must be NCHW"):
        E.save_gradcam_png(model, x, tmp_path / "cam.png")


def test_save_gradcam_png_basic(tmp_path):
    """Test basic Grad-CAM PNG generation."""
    model = E.build_model()
    x = torch.rand(1, 3, 224, 224)
    out_path = tmp_path / "gradcam.png"

    E.save_gradcam_png(model, x, out_path)
    assert out_path.exists(), "Grad-CAM PNG should be created"


def test_make_gc_various_signatures():
    """Test _make_gc with different GradCAM signatures."""
    model = E.build_model()

    # This should work with the actual GradCAM implementation
    gc = E._make_gc(model)
    assert gc is not None


def test_run_generate_callable():
    """Test _run_generate with callable object."""
    model = E.build_model()
    gc = E._make_gc(model)
    x = torch.rand(1, 3, 224, 224)

    result = E._run_generate(gc, x)
    assert result is not None
    assert isinstance(result, torch.Tensor)


def test_load_one_from_cfg_missing_dataset_raises(tmp_path):
    """Test that empty dataset raises error."""
    from omegaconf import OmegaConf

    # Create empty CSV
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("image_path,label\n")

    # Create config
    cfg_path = tmp_path / "test_config.yaml"
    cfg = OmegaConf.create(
        {
            "data": {
                "img_size": 224,
                "val_csv": str(csv_path),
                "train_csv": str(csv_path),
            }
        }
    )
    OmegaConf.save(cfg, cfg_path)

    with pytest.raises(RuntimeError, match="empty"):
        E._load_one_from_cfg(str(cfg_path), split="val")


def test_load_one_from_cfg_train_split(tmp_path):
    """Test loading from train split."""
    from omegaconf import OmegaConf

    # Create test image
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (32, 32), color=(128, 128, 128)).save(img_path)

    # Create CSV
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("image_path,label\ntest.jpg,0\n")

    # Create config
    cfg_path = tmp_path / "config.yaml"
    cfg = OmegaConf.create(
        {
            "data": {
                "img_size": 32,
                "val_csv": str(csv_path),
                "train_csv": str(csv_path),
            }
        }
    )
    OmegaConf.save(cfg, cfg_path)

    result = E._load_one_from_cfg(str(cfg_path), split="train")
    assert result.shape == (1, 3, 32, 32)


def test_save_csv_with_index(tmp_path):
    """Test CSV saving with index."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    p = tmp_path / "with_index.csv"
    E.save_csv(df, p, index=True)

    loaded = pd.read_csv(p, index_col=0)
    assert len(loaded) == 2


def test_save_gradcam_2d_heat(tmp_path):
    """Test Grad-CAM with 2D heatmap output."""
    model = E.build_model()
    x = torch.rand(2, 3, 224, 224)  # Batch of 2
    out_path = tmp_path / "cam_2d.png"

    E.save_gradcam_png(model, x, out_path)
    assert out_path.exists()


def test_save_heatmap_near_zero_max(tmp_path):
    """Test heatmap with very small max value."""
    arr = np.array([[1e-15, 2e-15], [3e-15, 4e-15]], dtype="float32")
    p = tmp_path / "small_max.png"
    E.save_heatmap(arr, p)
    assert p.exists()


def test_save_gradcam_batch_processing(tmp_path):
    """Test Grad-CAM with larger batch."""
    model = E.build_model()
    x = torch.rand(4, 3, 224, 224)  # Batch of 4
    out_path = tmp_path / "cam_batch.png"

    E.save_gradcam_png(model, x, out_path)
    assert out_path.exists()


def test_save_gradcam_different_sizes(tmp_path):
    """Test Grad-CAM with different input sizes that need resizing."""
    model = E.build_model()
    x = torch.rand(1, 3, 128, 128)  # Different from standard 224
    out_path = tmp_path / "cam_128.png"

    E.save_gradcam_png(model, x, out_path)
    assert out_path.exists()


def test_ensure_dir_nested(tmp_path):
    """Test ensure_dir creates nested directories."""
    nested = tmp_path / "a" / "b" / "c"
    result = E.ensure_dir(nested)
    assert result.exists()
    assert result.is_dir()


def test_save_csv_without_index(tmp_path):
    """Test CSV saving without index (default)."""
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    p = tmp_path / "no_index.csv"
    E.save_csv(df, p)  # Default index=False

    content = p.read_text()
    assert "x,y" in content
    assert ",0," not in content  # No index column


def test_load_csv_roundtrip(tmp_path):
    """Test CSV save and load roundtrip."""
    df = pd.DataFrame({"col1": [10, 20], "col2": [30, 40]})
    p = tmp_path / "roundtrip.csv"

    E.save_csv(df, p)
    loaded = E.load_csv(p)

    assert len(loaded) == 2
    assert list(loaded.columns) == ["col1", "col2"]
    assert loaded["col1"].tolist() == [10, 20]


def test_save_json_nested_structures(tmp_path):
    """Test JSON with complex nested structures."""
    data = {
        "level1": {"level2": {"level3": [1, 2, 3], "dict": {"a": 1, "b": 2}}},
        "list": [{"x": 1}, {"y": 2}],
    }
    p = tmp_path / "nested.json"
    E.save_json(data, p)

    loaded = E.load_json(p)
    assert loaded["level1"]["level2"]["level3"] == [1, 2, 3]
    assert loaded["list"][0]["x"] == 1


def test_save_npy_different_dtypes(tmp_path):
    """Test save_npy with different data types."""
    for dtype in [np.float32, np.float64, np.int32, np.int64]:
        arr = np.array([[1, 2], [3, 4]], dtype=dtype)
        p = tmp_path / f"arr_{dtype.__name__}.npy"

        E.save_npy(arr, p)
        loaded = E.load_npy(p)

        assert loaded.dtype == dtype
        assert np.array_equal(loaded, arr)


def test_save_heatmap_large_array(tmp_path):
    """Test heatmap with larger array."""
    arr = np.random.rand(256, 256).astype("float32")
    p = tmp_path / "large.png"
    E.save_heatmap(arr, p)

    img = Image.open(p)
    assert img.size == (256, 256)


def test_gradcam_png_multichannel_averaging(tmp_path):
    """Test Grad-CAM with heat that needs channel averaging."""
    model = E.build_model()
    x = torch.rand(1, 3, 224, 224)
    out_path = tmp_path / "cam_avg.png"

    # This will test the h.mean(dim=0, keepdim=True) path
    E.save_gradcam_png(model, x, out_path)
    assert out_path.exists()


def test_load_npy_pathlib(tmp_path):
    """Test load_npy with pathlib.Path."""
    from pathlib import Path

    arr = np.array([[5, 6], [7, 8]])
    p = Path(tmp_path) / "pathlib.npy"
    np.save(p, arr)

    loaded = E.load_npy(p)  # Pass Path object
    assert np.array_equal(loaded, arr)
