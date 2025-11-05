"""
Final tests to achieve 100% coverage for all remaining lines.
Covers:
- src/xai/export.py: lines 132, 183->180, 188, 251, 257->259
- src/xai/gradcam.py: line 52
- src/data/nih_binary.py: lines 208-209, 217->223, 225-228
- src/data/derm_datasets.py: line 113
"""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image

from src.data.derm_datasets import ISICDataset
from src.data.nih_binary import CSVImageDataset, NIHBinarizedDataset
from src.xai import export as E
from src.xai.gradcam import GradCAM

# ============================================================================
# src/xai/export.py - Line 132: _make_gc with positional-only GradCAM
# ============================================================================


def test_make_gc_positional_only():
    """Test _make_gc when GradCAM only accepts positional arguments - Line 132."""
    model = E.build_model()

    original_gradcam = E.gradcam.GradCAM

    class PositionalOnlyGradCAM:
        def __init__(self, model):
            self.model = model

    E.gradcam.GradCAM = PositionalOnlyGradCAM

    try:
        gc = E._make_gc(model)
        assert hasattr(gc, "model")
    finally:
        E.gradcam.GradCAM = original_gradcam


# ============================================================================
# src/xai/export.py - Lines 183->180, 188: _run_generate fallback paths
# ============================================================================


def test_run_generate_callable_returns_non_tensor_then_fallback():
    """Test _run_generate when callable returns non-Tensor, falls back to gradcam.gradcam - Lines 183->180."""
    x = torch.randn(1, 3, 224, 224)
    expected_result = torch.randn(1, 224, 224)

    # Create object with generate that returns non-Tensor
    class NonTensorCallable:
        def generate(self, *args, **kwargs):
            return "not a tensor"

        def __call__(self, *args, **kwargs):
            return "also not a tensor"

    mock_gc = NonTensorCallable()

    # Mock gradcam.gradcam to return actual tensor
    original_gradcam_func = getattr(E.gradcam, "gradcam", None)
    E.gradcam.gradcam = lambda gc, x, **kwargs: expected_result

    try:
        result = E._run_generate(mock_gc, x)
        assert torch.equal(result, expected_result)
    finally:
        if original_gradcam_func:
            E.gradcam.gradcam = original_gradcam_func


def test_run_generate_all_paths_fail_raises_error():
    """Test _run_generate raises RuntimeError when all paths fail - Line 188."""
    x = torch.randn(1, 3, 224, 224)

    # Create object that returns non-Tensor everywhere
    class AlwaysNonTensor:
        def generate(self, *args, **kwargs):
            return None

        def __call__(self, *args, **kwargs):
            return None

    mock_gc = AlwaysNonTensor()

    # Mock gradcam.gradcam to also return None
    original_gradcam_func = getattr(E.gradcam, "gradcam", None)
    E.gradcam.gradcam = lambda gc, x, **kwargs: None

    try:
        with pytest.raises(RuntimeError, match="Don't know how to invoke Grad-CAM"):
            E._run_generate(mock_gc, x)
    finally:
        if original_gradcam_func:
            E.gradcam.gradcam = original_gradcam_func


# ============================================================================
# src/xai/export.py - Line 251: _load_one_from_cfg train_csv usage
# ============================================================================


def test_load_one_from_cfg_uses_train_csv_for_train_split(tmp_path):
    """Test _load_one_from_cfg uses train_csv when split='train' - Line 251."""
    # Create image
    img_path = tmp_path / "train_image.jpg"
    Image.new("RGB", (48, 48), color=(50, 100, 150)).save(img_path)

    # Create train CSV
    train_csv = tmp_path / "train_data.csv"
    train_csv.write_text("image_path,label\ntrain_image.jpg,1.0\n")

    # Create config with train_csv
    cfg_path = tmp_path / "cfg.yaml"
    cfg = OmegaConf.create(
        {
            "data": {
                "img_size": 48,
                "train_csv": str(train_csv),
            }
        }
    )
    OmegaConf.save(cfg, cfg_path)

    result = E._load_one_from_cfg(str(cfg_path), split="train")
    assert result.shape == (1, 3, 48, 48)


# ============================================================================
# src/xai/export.py - Line 257->259: Tensor unsqueezing
# ============================================================================


def test_load_one_from_cfg_unsqueezes_3d_to_4d(tmp_path):
    """Test _load_one_from_cfg unsqueezes 3D tensor to 4D - Line 257->259."""
    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (32, 32), color=(128, 128, 128)).save(img_path)

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("image_path,label\nimg.jpg,0.5\n")

    cfg_path = tmp_path / "config.yaml"
    cfg = OmegaConf.create(
        {
            "data": {
                "img_size": 32,
                "val_csv": str(csv_path),
            }
        }
    )
    OmegaConf.save(cfg, cfg_path)

    result = E._load_one_from_cfg(str(cfg_path), split="val")

    # Should be 4D after unsqueezing
    assert result.ndim == 4
    assert result.shape[0] == 1


# ============================================================================
# src/xai/gradcam.py - Line 52: Invalid layer name
# ============================================================================


def test_gradcam_invalid_layer_raises_error():
    """Test GradCAM raises ValueError for invalid layer name - Line 52."""
    model = E.build_model()

    with pytest.raises(ValueError, match="Layer .* not found"):
        GradCAM(model, target_layer_name="nonexistent_layer")


# ============================================================================
# src/data/nih_binary.py - Lines 208-209, 217->223, 225-228
# ============================================================================


def test_csv_image_dataset_file_not_found():
    """Test CSVImageDataset with non-existent CSV - Lines 208-209."""
    with pytest.raises(FileNotFoundError):
        CSVImageDataset(csv_file="/path/to/nowhere.csv", img_size=224)


def test_csv_image_dataset_invalid_headers(tmp_path):
    """Test CSVImageDataset with invalid headers - Lines 217->223."""
    csv_path = tmp_path / "bad.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["col1", "col2"])
        writer.writeheader()
        writer.writerow({"col1": "a", "col2": "b"})

    with pytest.raises(ValueError, match="CSV must contain headers: image_path,label"):
        CSVImageDataset(csv_file=str(csv_path), img_size=224)


def test_csv_image_dataset_functional(tmp_path):
    """Test CSVImageDataset complete functionality - Lines 225-228."""
    # Create image
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(img_path)

    # Create CSV
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerow({"image_path": "test.jpg", "label": "1.0"})

    # Test without augmentation
    dataset = CSVImageDataset(csv_file=str(csv_path), img_size=64, augment=False)

    assert len(dataset) == 1
    x, y = dataset[0]
    assert x.shape == (3, 64, 64)
    assert y.item() == 1.0


def test_csv_image_dataset_with_augmentation(tmp_path):
    """Test CSVImageDataset with augmentation enabled."""
    img_path = tmp_path / "aug_test.jpg"
    Image.new("RGB", (128, 128), color=(50, 100, 150)).save(img_path)

    csv_path = tmp_path / "aug_data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerow({"image_path": "aug_test.jpg", "label": "0.0"})

    dataset = CSVImageDataset(csv_file=str(csv_path), img_size=96, augment=True)

    x, y = dataset[0]
    assert x.shape == (3, 96, 96)
    assert y.item() == 0.0


# ============================================================================
# src/data/derm_datasets.py - Line 113: Missing metadata columns
# ============================================================================


def test_isic_dataset_missing_metadata_returns_empty_strings(tmp_path):
    """Test ISICDataset returns empty strings for missing metadata columns - Line 113."""
    csv_path = tmp_path / "minimal.csv"

    # CSV with only required columns, no optional metadata
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerow({"image_path": "img.png", "label": "1.0"})

    img_path = tmp_path / "img.png"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(img_path)

    dataset = ISICDataset(csv_path=str(csv_path), images_root=str(tmp_path))

    x, y, meta = dataset[0]

    # All optional metadata should be empty strings
    assert meta["center"] == ""
    assert meta["age"] == ""
    assert meta["sex"] == ""
    assert meta["location"] == ""
    assert meta["image"] == "img.png"


def test_isic_dataset_partial_metadata(tmp_path):
    """Test ISICDataset with some metadata present."""
    csv_path = tmp_path / "partial.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "age", "sex"])
        writer.writeheader()
        writer.writerow(
            {"image_path": "img.png", "label": "1.0", "age": "45", "sex": "M"}
        )

    img_path = tmp_path / "img.png"
    Image.new("RGB", (64, 64), color=(100, 100, 100)).save(img_path)

    dataset = ISICDataset(csv_path=str(csv_path), images_root=str(tmp_path))

    x, y, meta = dataset[0]

    # Present metadata should have values
    assert meta["age"] == "45"
    assert meta["sex"] == "M"
    # Missing metadata should be empty
    assert meta["center"] == ""
    assert meta["location"] == ""


# ============================================================================
# Integration tests
# ============================================================================


def test_complete_export_pipeline(tmp_path):
    """Test complete export pipeline."""
    # Create image
    img_path = tmp_path / "pipeline.jpg"
    Image.new("RGB", (224, 224), color=(128, 128, 128)).save(img_path)

    # Create CSV
    csv_path = tmp_path / "pipeline.csv"
    csv_path.write_text("image_path,label\npipeline.jpg,1.0\n")

    # Create config
    cfg_path = tmp_path / "pipeline_cfg.yaml"
    cfg = OmegaConf.create(
        {
            "data": {
                "img_size": 224,
                "val_csv": str(csv_path),
            }
        }
    )
    OmegaConf.save(cfg, cfg_path)

    # Load image
    x = E._load_one_from_cfg(str(cfg_path), split="val")

    # Create model and generate Grad-CAM
    model = E.build_model()
    out_path = tmp_path / "gradcam_output.png"

    E.save_gradcam_png(model, x, out_path)

    assert out_path.exists()


if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
            "--cov=src.xai.export",
            "--cov=src.xai.gradcam",
            "--cov=src.data.nih_binary",
            "--cov=src.data.derm_datasets",
            "--cov-branch",
            "--cov-report=term-missing",
        ]
    )
