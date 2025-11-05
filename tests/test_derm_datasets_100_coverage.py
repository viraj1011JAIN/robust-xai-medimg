"""Tests for 100% coverage of src/data/derm_datasets.py - covers lines 79-80, 86, 109"""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.derm_datasets import ISICDataset, _read_rgb_or_placeholder, _to_chw_float01

# ============================================================================
# Line 79-80: TypeError for transform returning Tensor with unexpected shape
# ============================================================================


def test_isic_transform_tensor_unexpected_shape(tmp_path):
    """Test _apply_transform raises TypeError for Tensor with unexpected shape - Lines 79-80."""
    csv_path = tmp_path / "data.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerow({"image_path": "img1.png", "label": "1"})

    img_path = tmp_path / "img1.png"
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    img.save(img_path)

    # Transform that returns dict with Tensor of unexpected shape (4D)
    def bad_transform(image):
        return {"image": torch.rand(2, 3, 64, 64)}  # 4D tensor, unexpected

    dataset = ISICDataset(
        csv_path=str(csv_path), images_root=str(tmp_path), transform=bad_transform
    )

    with pytest.raises(TypeError, match="Transform returned Tensor with unexpected shape"):
        _ = dataset[0]


def test_isic_transform_tensor_wrong_channel_count(tmp_path):
    """Test _apply_transform with Tensor having wrong channel count."""
    csv_path = tmp_path / "data.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerow({"image_path": "img1.png", "label": "1"})

    img_path = tmp_path / "img1.png"
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    img.save(img_path)

    # Transform returning Tensor with 5 channels (invalid)
    def bad_transform(image):
        return {"image": torch.rand(64, 64, 5)}  # HWC with 5 channels

    dataset = ISICDataset(
        csv_path=str(csv_path), images_root=str(tmp_path), transform=bad_transform
    )

    with pytest.raises(TypeError, match="Transform returned Tensor with unexpected shape"):
        _ = dataset[0]


# ============================================================================
# Line 86: TypeError for transform returning ndarray with unexpected shape
# ============================================================================


def test_isic_transform_ndarray_unexpected_shape(tmp_path):
    """Test _apply_transform raises TypeError for ndarray with unexpected shape - Line 86."""
    csv_path = tmp_path / "data.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerow({"image_path": "img1.png", "label": "1"})

    img_path = tmp_path / "img1.png"
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    img.save(img_path)

    # Transform returning dict with ndarray of wrong shape
    def bad_transform(image):
        return {"image": np.random.rand(64, 64, 5)}  # 5 channels, invalid

    dataset = ISICDataset(
        csv_path=str(csv_path), images_root=str(tmp_path), transform=bad_transform
    )

    with pytest.raises(TypeError, match="Transform returned ndarray with unexpected shape"):
        _ = dataset[0]


def test_isic_transform_ndarray_4d(tmp_path):
    """Test _apply_transform with 4D ndarray."""
    csv_path = tmp_path / "data.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerow({"image_path": "img1.png", "label": "1"})

    img_path = tmp_path / "img1.png"
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    img.save(img_path)

    # Transform returning 4D ndarray
    def bad_transform(image):
        return {"image": np.random.rand(2, 64, 64, 3)}

    dataset = ISICDataset(
        csv_path=str(csv_path), images_root=str(tmp_path), transform=bad_transform
    )

    with pytest.raises(TypeError, match="Transform returned ndarray with unexpected shape"):
        _ = dataset[0]


# ============================================================================
# Line 109: Missing metadata columns return empty strings
# ============================================================================


def test_isic_missing_metadata_columns(tmp_path):
    """Test that missing metadata columns return empty strings - Line 109."""
    csv_path = tmp_path / "data.csv"

    # CSV with only required columns, missing optional metadata
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerow({"image_path": "img1.png", "label": "1.0"})

    img_path = tmp_path / "img1.png"
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    img.save(img_path)

    dataset = ISICDataset(csv_path=str(csv_path), images_root=str(tmp_path))

    x, y, meta = dataset[0]

    # All metadata fields should be empty strings
    assert meta["center"] == ""
    assert meta["age"] == ""
    assert meta["sex"] == ""
    assert meta["location"] == ""
    assert meta["image"] == "img1.png"


def test_isic_partial_metadata_columns(tmp_path):
    """Test with some metadata columns present, some missing."""
    csv_path = tmp_path / "data.csv"

    # CSV with some optional metadata
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "age", "sex"])
        writer.writeheader()
        writer.writerow({"image_path": "img1.png", "label": "1.0", "age": "45", "sex": "M"})

    img_path = tmp_path / "img1.png"
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    img.save(img_path)

    dataset = ISICDataset(csv_path=str(csv_path), images_root=str(tmp_path))

    x, y, meta = dataset[0]

    # Present columns should have values
    assert meta["age"] == "45"
    assert meta["sex"] == "M"
    # Missing columns should be empty strings
    assert meta["center"] == ""
    assert meta["location"] == ""


def test_isic_all_metadata_columns(tmp_path):
    """Test with all metadata columns present."""
    csv_path = tmp_path / "data.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "label", "center", "age", "sex", "location"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_path": "img1.png",
                "label": "1.0",
                "center": "Hospital_A",
                "age": "55",
                "sex": "F",
                "location": "back",
            }
        )

    img_path = tmp_path / "img1.png"
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    img.save(img_path)

    dataset = ISICDataset(csv_path=str(csv_path), images_root=str(tmp_path))

    x, y, meta = dataset[0]

    assert meta["center"] == "Hospital_A"
    assert meta["age"] == "55"
    assert meta["sex"] == "F"
    assert meta["location"] == "back"


# ============================================================================
# Additional coverage tests
# ============================================================================


def test_isic_uncertain_label_mapping(tmp_path):
    """Test uncertain label (-1) mapping."""
    csv_path = tmp_path / "data.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerow({"image_path": "img1.png", "label": "-1"})

    img_path = tmp_path / "img1.png"
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    img.save(img_path)

    # Test with uncertain_to=0
    dataset = ISICDataset(csv_path=str(csv_path), images_root=str(tmp_path), uncertain_to=0)

    x, y, meta = dataset[0]
    assert y.item() == 0.0

    # Test with uncertain_to=1
    dataset2 = ISICDataset(csv_path=str(csv_path), images_root=str(tmp_path), uncertain_to=1)

    x2, y2, meta2 = dataset2[0]
    assert y2.item() == 1.0


def test_isic_missing_csv_columns(tmp_path):
    """Test ValueError when required CSV columns are missing."""
    csv_path = tmp_path / "bad.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wrong_col"])
        writer.writeheader()

    with pytest.raises(ValueError, match="CSV must contain columns"):
        ISICDataset(csv_path=str(csv_path), images_root=str(tmp_path))


def test_isic_transform_returns_direct_tensor(tmp_path):
    """Test transform that returns Tensor directly (not in dict)."""
    csv_path = tmp_path / "data.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerow({"image_path": "img1.png", "label": "1"})

    img_path = tmp_path / "img1.png"
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    img.save(img_path)

    def tensor_transform(image):
        # Return CHW tensor directly
        return torch.from_numpy(image).permute(2, 0, 1).float()

    dataset = ISICDataset(
        csv_path=str(csv_path), images_root=str(tmp_path), transform=tensor_transform
    )

    x, y, meta = dataset[0]
    assert x.shape[0] == 3  # Channels


def test_isic_transform_returns_direct_ndarray(tmp_path):
    """Test transform that returns ndarray directly (not in dict)."""
    csv_path = tmp_path / "data.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerow({"image_path": "img1.png", "label": "1"})

    img_path = tmp_path / "img1.png"
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    img.save(img_path)

    def ndarray_transform(image):
        return image * 0.5  # Return ndarray directly

    dataset = ISICDataset(
        csv_path=str(csv_path), images_root=str(tmp_path), transform=ndarray_transform
    )

    x, y, meta = dataset[0]
    assert isinstance(x, torch.Tensor)


def test_isic_transform_direct_tensor_wrong_shape(tmp_path):
    """Test direct Tensor return with wrong shape."""
    csv_path = tmp_path / "data.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerow({"image_path": "img1.png", "label": "1"})

    img_path = tmp_path / "img1.png"
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    img.save(img_path)

    def bad_transform(image):
        return torch.rand(64, 64, 5)  # 5 channels, invalid

    dataset = ISICDataset(
        csv_path=str(csv_path), images_root=str(tmp_path), transform=bad_transform
    )

    with pytest.raises(TypeError, match="Transform returned Tensor with unexpected shape"):
        _ = dataset[0]


def test_isic_transform_direct_ndarray_wrong_shape(tmp_path):
    """Test direct ndarray return with wrong shape."""
    csv_path = tmp_path / "data.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerow({"image_path": "img1.png", "label": "1"})

    img_path = tmp_path / "img1.png"
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    img.save(img_path)

    def bad_transform(image):
        return np.random.rand(64, 64, 7)  # 7 channels, invalid

    dataset = ISICDataset(
        csv_path=str(csv_path), images_root=str(tmp_path), transform=bad_transform
    )

    with pytest.raises(TypeError, match="Transform returned ndarray with unexpected shape"):
        _ = dataset[0]


def test_to_chw_float01():
    """Test _to_chw_float01 utility function."""
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    result = _to_chw_float01(img)

    assert result.shape == (3, 64, 64)
    assert result.dtype == torch.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_isic_custom_target_col(tmp_path):
    """Test ISICDataset with custom target column name."""
    csv_path = tmp_path / "data.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "diagnosis"])
        writer.writeheader()
        writer.writerow({"image_path": "img1.png", "diagnosis": "1"})

    img_path = tmp_path / "img1.png"
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    img.save(img_path)

    dataset = ISICDataset(csv_path=str(csv_path), images_root=str(tmp_path), target_col="diagnosis")

    x, y, meta = dataset[0]
    assert y.item() == 1.0


def test_isic_dataset_len(tmp_path):
    """Test __len__ method."""
    csv_path = tmp_path / "data.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        for i in range(5):
            writer.writerow({"image_path": f"img{i}.png", "label": str(i % 2)})

    # Create dummy images
    for i in range(5):
        img_path = tmp_path / f"img{i}.png"
        img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        img.save(img_path)

    dataset = ISICDataset(csv_path=str(csv_path), images_root=str(tmp_path))

    assert len(dataset) == 5


def test_read_rgb_or_placeholder_success(tmp_path):
    """Test successful RGB image reading."""
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(img_path)

    result = _read_rgb_or_placeholder(str(img_path))
    assert result.shape == (100, 100, 3)
    assert result.dtype == np.uint8


def test_read_rgb_or_placeholder_failure(tmp_path):
    """Test placeholder return on failure."""
    result = _read_rgb_or_placeholder("/nonexistent/path.png")
    assert result.shape == (64, 64, 3)
    assert np.all(result == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
