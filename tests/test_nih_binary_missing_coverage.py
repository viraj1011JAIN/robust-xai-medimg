"""
Tests for 100% coverage of src/data/nih_binary.py
Covers missing lines 162-163, 208-228
"""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.nih_binary import CSVImageDataset, NIHBinarizedDataset

# ============================================================================
# Lines 162-163: Transform returning invalid type (not Tensor/ndarray/dict)
# ============================================================================


def test_nih_transform_returns_invalid_type(tmp_path):
    """Test NIHBinarizedDataset raises TypeError when transform returns invalid type - Lines 162-163."""
    csv_path = tmp_path / "data.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Image", "Pneumonia"])
        writer.writeheader()
        writer.writerow({"Image": "img1.png", "Pneumonia": "1"})

    img_path = tmp_path / "img1.png"
    img = Image.new("L", (64, 64), color=128)
    img.save(img_path)

    # Transform that returns a string (invalid type)
    def bad_transform(img):
        return "this is not valid"

    dataset = NIHBinarizedDataset(
        csv_path=str(csv_path),
        classes=["Pneumonia"],
        images_root=str(tmp_path),
        transform=bad_transform,
    )

    with pytest.raises(TypeError, match="Transform must return Tensor/ndarray"):
        _ = dataset[0]


def test_nih_transform_dict_with_invalid_image_type(tmp_path):
    """Test NIHBinarizedDataset when transform returns dict with invalid 'image' value."""
    csv_path = tmp_path / "data.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Image", "Pneumonia"])
        writer.writeheader()
        writer.writerow({"Image": "img1.png", "Pneumonia": "1"})

    img_path = tmp_path / "img1.png"
    img = Image.new("L", (64, 64), color=128)
    img.save(img_path)

    # Transform that returns dict with invalid image value
    def bad_dict_transform(img):
        return {"image": [1, 2, 3]}  # List is not valid

    dataset = NIHBinarizedDataset(
        csv_path=str(csv_path),
        classes=["Pneumonia"],
        images_root=str(tmp_path),
        transform=bad_dict_transform,
    )

    with pytest.raises(TypeError, match="Transform must return"):
        _ = dataset[0]


# ============================================================================
# Lines 208-228: CSVImageDataset functionality
# ============================================================================


def test_csv_image_dataset_missing_csv():
    """Test CSVImageDataset raises FileNotFoundError for missing CSV - Line 208."""
    with pytest.raises(FileNotFoundError):
        CSVImageDataset(csv_file="/nonexistent/file.csv", img_size=224)


def test_csv_image_dataset_missing_headers(tmp_path):
    """Test CSVImageDataset raises ValueError for missing required headers - Line 214-216."""
    csv_path = tmp_path / "bad_headers.csv"

    # Create CSV with wrong headers
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wrong1", "wrong2"])
        writer.writeheader()
        writer.writerow({"wrong1": "a", "wrong2": "b"})

    with pytest.raises(ValueError, match="CSV must contain headers: image_path,label"):
        CSVImageDataset(csv_file=str(csv_path), img_size=224)


def test_csv_image_dataset_missing_image_path_header(tmp_path):
    """Test CSVImageDataset when only 'label' header is present."""
    csv_path = tmp_path / "missing_image_path.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "other_col"])
        writer.writeheader()
        writer.writerow({"label": "1.0", "other_col": "x"})

    with pytest.raises(ValueError, match="CSV must contain headers: image_path,label"):
        CSVImageDataset(csv_file=str(csv_path), img_size=224)


def test_csv_image_dataset_missing_label_header(tmp_path):
    """Test CSVImageDataset when only 'image_path' header is present."""
    csv_path = tmp_path / "missing_label.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "other_col"])
        writer.writeheader()
        writer.writerow({"image_path": "img.png", "other_col": "y"})

    with pytest.raises(ValueError, match="CSV must contain headers: image_path,label"):
        CSVImageDataset(csv_file=str(csv_path), img_size=224)


def test_csv_image_dataset_valid_with_augment(tmp_path):
    """Test CSVImageDataset with augmentation enabled - Line 219-228."""
    # Create test image
    img_path = tmp_path / "test_img.jpg"
    img = Image.new("RGB", (256, 256), color=(128, 64, 192))
    img.save(img_path)

    # Create valid CSV
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerow({"image_path": "test_img.jpg", "label": "1.0"})

    # Test with augmentation
    dataset = CSVImageDataset(csv_file=str(csv_path), img_size=224, augment=True)

    assert len(dataset) == 1
    x, y = dataset[0]

    # Check output shape and type
    assert x.shape == (3, 224, 224)
    assert y.dtype == torch.float
    assert y.item() == 1.0


def test_csv_image_dataset_valid_without_augment(tmp_path):
    """Test CSVImageDataset without augmentation - Line 219-228."""
    # Create test image
    img_path = tmp_path / "test_img2.jpg"
    img = Image.new("RGB", (256, 256), color=(64, 128, 255))
    img.save(img_path)

    # Create valid CSV
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerow({"image_path": "test_img2.jpg", "label": "0.5"})

    # Test without augmentation
    dataset = CSVImageDataset(csv_file=str(csv_path), img_size=128, augment=False)

    assert len(dataset) == 1
    x, y = dataset[0]

    assert x.shape == (3, 128, 128)
    assert y.item() == 0.5


def test_csv_image_dataset_multiple_records(tmp_path):
    """Test CSVImageDataset with multiple records."""
    # Create test images
    for i in range(3):
        img_path = tmp_path / f"img{i}.jpg"
        img = Image.new("RGB", (128, 128), color=(i * 50, i * 80, i * 100))
        img.save(img_path)

    # Create CSV with multiple rows
    csv_path = tmp_path / "multi.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        for i in range(3):
            writer.writerow({"image_path": f"img{i}.jpg", "label": str(float(i) / 2)})

    dataset = CSVImageDataset(csv_file=str(csv_path), img_size=64, augment=False)

    assert len(dataset) == 3

    # Test each item
    for i in range(3):
        x, y = dataset[i]
        assert x.shape == (3, 64, 64)
        assert abs(y.item() - (i / 2)) < 0.01


def test_csv_image_dataset_path_resolution(tmp_path):
    """Test that CSVImageDataset correctly resolves image paths."""
    # Create nested directory structure
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    img_path = img_dir / "nested_img.jpg"
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    img.save(img_path)

    # CSV is in tmp_path, image is in tmp_path/images
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerow({"image_path": "images/nested_img.jpg", "label": "1.0"})

    dataset = CSVImageDataset(csv_file=str(csv_path), img_size=64, augment=False)

    x, y = dataset[0]
    assert x.shape == (3, 64, 64)


# ============================================================================
# Additional comprehensive tests
# ============================================================================


def test_nih_binarized_with_2d_transform_output(tmp_path):
    """Test NIHBinarizedDataset with transform returning 2D tensor."""
    csv_path = tmp_path / "data.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Image", "Pneumonia"])
        writer.writeheader()
        writer.writerow({"Image": "img1.png", "Pneumonia": "1"})

    img_path = tmp_path / "img1.png"
    img = Image.new("L", (64, 64), color=128)
    img.save(img_path)

    # Transform that returns 2D tensor (will be unsqueezed to 3D [1, H, W])
    def transform_2d(img):
        return torch.from_numpy(img / 255.0).float()

    dataset = NIHBinarizedDataset(
        csv_path=str(csv_path),
        classes=["Pneumonia"],
        images_root=str(tmp_path),
        transform=transform_2d,
    )

    x, y, meta = dataset[0]
    # The 2D tensor gets unsqueezed, but the actual image is RGB converted, so shape is [1, H, W]
    assert x.ndim == 3
    # Since grayscale image was loaded and converted, first dim should be 1
    assert x.shape[0] in [1, 3]  # Accept either grayscale (1) or RGB (3)


def test_nih_binarized_with_3d_ndarray_transform(tmp_path):
    """Test NIHBinarizedDataset with transform returning 3D ndarray."""
    csv_path = tmp_path / "data.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Image", "Pneumonia"])
        writer.writeheader()
        writer.writerow({"Image": "img1.png", "Pneumonia": "1"})

    img_path = tmp_path / "img1.png"
    img = Image.new("L", (64, 64), color=128)
    img.save(img_path)

    # Transform that returns 3D ndarray [C, H, W]
    def transform_3d_ndarray(img):
        return np.expand_dims(img / 255.0, axis=0).astype(np.float32)

    dataset = NIHBinarizedDataset(
        csv_path=str(csv_path),
        classes=["Pneumonia"],
        images_root=str(tmp_path),
        transform=transform_3d_ndarray,
    )

    x, y, meta = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert x.ndim == 3


def test_nih_binarized_labels_with_exception_handling(tmp_path):
    """Test NIHBinarizedDataset handles non-numeric label values gracefully."""
    csv_path = tmp_path / "data.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Image", "Pneumonia", "Edema"])
        writer.writeheader()
        writer.writerow({"Image": "img1.png", "Pneumonia": "invalid", "Edema": "1"})  # Non-numeric

    img_path = tmp_path / "img1.png"
    img = Image.new("L", (64, 64), color=128)
    img.save(img_path)

    dataset = NIHBinarizedDataset(
        csv_path=str(csv_path),
        classes=["Pneumonia", "Edema"],
        images_root=str(tmp_path),
    )

    x, y, meta = dataset[0]
    # Non-numeric should be treated as 0
    assert y[0].item() == 0
    assert y[1].item() == 1


def test_csv_image_dataset_float_label_precision(tmp_path):
    """Test CSVImageDataset preserves float label precision."""
    img_path = tmp_path / "img.jpg"
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    img.save(img_path)

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerow({"image_path": "img.jpg", "label": "0.123456"})

    dataset = CSVImageDataset(csv_file=str(csv_path), img_size=64, augment=False)

    x, y = dataset[0]
    assert abs(y.item() - 0.123456) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.data.nih_binary", "--cov-report=term-missing"])
