import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.cxr_datasets import (NIHChestXray, PadChestCXRBase, VinDrCXRBase,
                                   _albumentations_like_call, _CXRBase,
                                   _read_rgb_or_placeholder,
                                   _to_chw_float_tensor)

# ============================================================================
# Test _read_rgb_or_placeholder - Line 40-49 coverage
# ============================================================================


def test_read_rgb_or_placeholder_success(tmp_path):
    """Test successful image reading."""
    img_path = tmp_path / "test.png"
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    Image.fromarray(img_array).save(img_path)

    result = _read_rgb_or_placeholder(str(img_path))
    assert result.shape == (100, 100, 3)
    assert result.dtype == np.uint8


def test_read_rgb_or_placeholder_failure_nonexistent():
    """Test placeholder return on nonexistent file - covers line 45."""
    result = _read_rgb_or_placeholder("/nonexistent/path/image.png")
    assert result.shape == (64, 64, 3)
    assert result.dtype == np.uint8
    assert np.all(result == 0)


def test_read_rgb_or_placeholder_failure_corrupted(tmp_path):
    """Test placeholder return on corrupted file - covers line 45."""
    bad_file = tmp_path / "corrupted.png"
    bad_file.write_bytes(b"not an image")

    result = _read_rgb_or_placeholder(str(bad_file))
    assert result.shape == (64, 64, 3)
    assert result.dtype == np.uint8
    assert np.all(result == 0)


def test_read_rgb_or_placeholder_converts_grayscale(tmp_path):
    """Test RGB conversion from grayscale."""
    img_path = tmp_path / "gray.png"
    gray_img = Image.new("L", (50, 50), color=128)
    gray_img.save(img_path)

    result = _read_rgb_or_placeholder(str(img_path))
    assert result.shape == (50, 50, 3)
    assert result.dtype == np.uint8


# ============================================================================
# Test _albumentations_like_call - Line 43-49 coverage
# ============================================================================


def test_albumentations_like_call_with_dict_interface():
    """Test transform with Albumentations-style dict interface."""

    def albumentations_transform(image):
        return {"image": image * 2}

    img = np.random.randint(0, 127, (64, 64, 3), dtype=np.uint8)
    result = _albumentations_like_call(albumentations_transform, img)

    assert isinstance(result, dict)
    assert "image" in result
    assert np.array_equal(result["image"], img * 2)


def test_albumentations_like_call_with_positional_interface():
    """Test transform with positional interface - covers line 48 TypeError catch."""

    def positional_transform(img):
        return img * 2

    img = np.random.randint(0, 127, (64, 64, 3), dtype=np.uint8)
    result = _albumentations_like_call(positional_transform, img)

    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, img * 2)


def test_albumentations_like_call_returns_tensor():
    """Test transform that returns a torch.Tensor."""

    def tensor_transform(image):
        return torch.from_numpy(image).permute(2, 0, 1).float()

    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    result = _albumentations_like_call(tensor_transform, img)

    assert isinstance(result, torch.Tensor)


# ============================================================================
# Test _to_chw_float_tensor - Line 99 coverage (unexpected shapes)
# ============================================================================


def test_to_chw_float_tensor_unexpected_ndarray_shape():
    """Test error handling for unexpected ndarray shapes - covers line 99."""
    # 4D array (unexpected)
    bad_array = np.random.rand(10, 64, 64, 3)
    with pytest.raises(TypeError, match="Unexpected ndarray shape"):
        _to_chw_float_tensor(bad_array)

    # 1D array (unexpected)
    bad_array_1d = np.random.rand(100)
    with pytest.raises(TypeError, match="Unexpected ndarray shape"):
        _to_chw_float_tensor(bad_array_1d)

    # 3D with wrong channel dimension
    bad_array_channels = np.random.rand(64, 64, 5)
    with pytest.raises(TypeError, match="Unexpected ndarray shape"):
        _to_chw_float_tensor(bad_array_channels)


def test_to_chw_float_tensor_unexpected_tensor_shape():
    """Test error handling for unexpected tensor shapes - line 72."""
    # 4D tensor
    bad_tensor = torch.rand(2, 3, 64, 64)
    with pytest.raises(TypeError, match="Unexpected tensor shape"):
        _to_chw_float_tensor(bad_tensor)

    # 2D tensor
    bad_tensor_2d = torch.rand(64, 64)
    with pytest.raises(TypeError, match="Unexpected tensor shape"):
        _to_chw_float_tensor(bad_tensor_2d)

    # 3D tensor with wrong channel count
    bad_tensor_channels = torch.rand(64, 64, 5)
    with pytest.raises(TypeError, match="Unexpected tensor shape"):
        _to_chw_float_tensor(bad_tensor_channels)


def test_to_chw_float_tensor_valid_formats():
    """Test all valid input formats for completeness."""
    # HWC ndarray
    hwc_array = np.random.rand(64, 64, 3).astype(np.float32)
    result = _to_chw_float_tensor(hwc_array)
    assert result.shape == (3, 64, 64)
    assert result.dtype == torch.float32

    # CHW tensor
    chw_tensor = torch.rand(3, 64, 64)
    result = _to_chw_float_tensor(chw_tensor)
    assert result.shape == (3, 64, 64)

    # HWC tensor
    hwc_tensor = torch.rand(64, 64, 3)
    result = _to_chw_float_tensor(hwc_tensor)
    assert result.shape == (3, 64, 64)

    # Grayscale (H, W, 1)
    gray_array = np.random.rand(64, 64, 1).astype(np.float32)
    result = _to_chw_float_tensor(gray_array)
    assert result.shape == (1, 64, 64)

    # 2D grayscale converted to 3D
    gray_2d = np.random.rand(64, 64).astype(np.float32)
    result = _to_chw_float_tensor(gray_2d)
    assert result.shape == (1, 64, 64)


# ============================================================================
# Test _CXRBase - CSV validation (line 62-65)
# ============================================================================


def test_cxr_base_missing_image_path_column(tmp_path):
    """Test CSV validation when image_path column is missing."""
    csv_path = tmp_path / "bad.csv"
    df = pd.DataFrame({"Pneumonia": [0, 1], "Edema": [1, 0]})
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="CSV missing required column.*image_path"):
        _CXRBase(
            csv_path=str(csv_path),
            images_root=str(tmp_path),
            target_cols=["Pneumonia", "Edema"],
        )


def test_cxr_base_missing_target_columns(tmp_path):
    """Test CSV validation when target columns are missing."""
    csv_path = tmp_path / "bad.csv"
    df = pd.DataFrame({"image_path": ["img1.png", "img2.png"], "Pneumonia": [0, 1]})
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="CSV missing required column.*Edema, Mass"):
        _CXRBase(
            csv_path=str(csv_path),
            images_root=str(tmp_path),
            target_cols=["Pneumonia", "Edema", "Mass"],
        )


def test_cxr_base_missing_multiple_columns(tmp_path):
    """Test CSV validation with multiple missing columns."""
    csv_path = tmp_path / "bad.csv"
    df = pd.DataFrame({"other_col": [1, 2, 3]})
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="CSV missing required column"):
        _CXRBase(
            csv_path=str(csv_path),
            images_root=str(tmp_path),
            target_cols=["Pneumonia", "Edema"],
        )


# ============================================================================
# Integration tests for dataset classes
# ============================================================================


@pytest.fixture
def sample_dataset_files(tmp_path):
    """Create sample CSV and images for testing."""
    # Create CSV
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "image_path": ["img1.png", "img2.png", "img3.png"],
            "Pneumonia": [1.0, 0.0, 1.0],
            "Edema": [0.0, 1.0, np.nan],  # Test NaN handling
            "Atelectasis": [1.0, 1.0, 0.0],
        }
    )
    df.to_csv(csv_path, index=False)

    # Create images
    for img_name in ["img1.png", "img2.png", "img3.png"]:
        img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(tmp_path / img_name)

    return csv_path, tmp_path


def test_nih_chest_xray_with_transform(sample_dataset_files):
    """Test NIHChestXray with transform - verifies numpy output."""
    csv_path, images_root = sample_dataset_files

    def simple_transform(image):
        # Simulates Albumentations interface
        return {"image": image}

    dataset = NIHChestXray(
        csv_path=str(csv_path),
        images_root=str(images_root),
        transform=simple_transform,
        target_cols=["Pneumonia", "Edema", "Atelectasis"],
    )

    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, np.ndarray)
    assert y.dtype == np.float32


def test_nih_chest_xray_without_transform(sample_dataset_files):
    """Test NIHChestXray without transform - verifies tensor output."""
    csv_path, images_root = sample_dataset_files

    dataset = NIHChestXray(
        csv_path=str(csv_path),
        images_root=str(images_root),
        transform=None,
        target_cols=["Pneumonia", "Edema", "Atelectasis"],
    )

    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert y.dtype == torch.float32


def test_padchest_auto_target_selection(tmp_path):
    """Test PadChestCXRBase automatic target column selection."""
    csv_path = tmp_path / "padchest.csv"
    df = pd.DataFrame(
        {
            "image_path": ["img1.png", "img2.png"],
            "Pneumonia": [1.0, 0.0],
            "Edema": [0.0, 1.0],
            "SomeOtherDisease": [0.0, 0.0],
        }
    )
    df.to_csv(csv_path, index=False)

    # Create dummy image
    img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(img_array).save(tmp_path / "img1.png")

    dataset = PadChestCXRBase(
        csv_path=str(csv_path),
        images_root=str(tmp_path),
        target_cols=None,  # Auto-select
    )

    # Should select from NIH default targets that exist in CSV
    assert "Pneumonia" in dataset.target_cols
    assert "Edema" in dataset.target_cols


def test_vindrcxr_auto_target_selection(tmp_path):
    """Test VinDrCXRBase automatic target column selection."""
    csv_path = tmp_path / "vindrcxr.csv"
    df = pd.DataFrame({"image_path": ["img1.png"], "Atelectasis": [1.0], "Mass": [0.0]})
    df.to_csv(csv_path, index=False)

    img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(img_array).save(tmp_path / "img1.png")

    dataset = VinDrCXRBase(
        csv_path=str(csv_path), images_root=str(tmp_path), target_cols=None
    )

    assert "Atelectasis" in dataset.target_cols
    assert "Mass" in dataset.target_cols


def test_dataset_with_missing_image(tmp_path):
    """Test dataset handles missing images gracefully with placeholder."""
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({"image_path": ["missing.png"], "Pneumonia": [1.0]})
    df.to_csv(csv_path, index=False)

    dataset = NIHChestXray(
        csv_path=str(csv_path), images_root=str(tmp_path), target_cols=["Pneumonia"]
    )

    x, y = dataset[0]
    # Should return 64x64 placeholder
    assert x.shape[1:] == (64, 64)
    assert isinstance(x, torch.Tensor)


def test_transform_with_positional_only_interface(sample_dataset_files):
    """Test transform that only accepts positional arguments."""
    csv_path, images_root = sample_dataset_files

    class PositionalOnlyTransform:
        def __call__(self, img):
            # This will raise TypeError with image=img, forcing positional path
            return img // 2

    dataset = NIHChestXray(
        csv_path=str(csv_path),
        images_root=str(images_root),
        transform=PositionalOnlyTransform(),
        target_cols=["Pneumonia"],
    )

    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)


# ============================================================================
# Edge cases and additional coverage
# ============================================================================


def test_cxr_base_nan_handling(tmp_path):
    """Verify NaN values in target columns are filled with 0."""
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame(
        {"image_path": ["img1.png"], "Pneumonia": [np.nan], "Edema": [1.0]}
    )
    df.to_csv(csv_path, index=False)

    img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(img_array).save(tmp_path / "img1.png")

    dataset = _CXRBase(
        csv_path=str(csv_path),
        images_root=str(tmp_path),
        target_cols=["Pneumonia", "Edema"],
    )

    x, y = dataset[0]
    assert y[0].item() == 0.0  # NaN should be filled with 0
    assert y[1].item() == 1.0


def test_dataset_length(sample_dataset_files):
    """Test __len__ method."""
    csv_path, images_root = sample_dataset_files

    dataset = NIHChestXray(
        csv_path=str(csv_path), images_root=str(images_root), target_cols=["Pneumonia"]
    )

    assert len(dataset) == 3


if __name__ == "__main__":
    pytest.main(
        [__file__, "-v", "--cov=src.data.cxr_datasets", "--cov-report=term-missing"]
    )
