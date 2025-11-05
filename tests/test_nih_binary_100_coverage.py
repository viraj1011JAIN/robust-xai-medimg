# tests/test_nih_binary_100_coverage.py
"""Complete coverage tests for NIH binary dataset."""

import csv

import pytest
import torch
from PIL import Image

from src.data.nih_binary import CSVImageDataset, NIHBinarizedDataset


def test_csv_image_dataset_missing_file():
    """Test CSVImageDataset with non-existent file."""
    with pytest.raises(FileNotFoundError):
        CSVImageDataset(csv_file="/nonexistent/file.csv", img_size=32)


def test_csv_image_dataset_invalid_headers(tmp_path):
    """Test CSVImageDataset with wrong headers."""
    import pandas as pd

    csv_path = tmp_path / "bad.csv"
    pd.DataFrame([{"wrong1": "x", "wrong2": "y"}]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="image_path,label"):
        CSVImageDataset(csv_file=str(csv_path), img_size=32)


def test_nih_binarized_missing_csv():
    """Test NIHBinarizedDataset with missing CSV."""
    with pytest.raises(FileNotFoundError):
        NIHBinarizedDataset(csv_path="/nonexistent.csv", classes=["A"])


def test_nih_binarized_uncertain_mapping_to_zero(tmp_path):
    """Test uncertain value (-1) mapping to 0."""
    csv_path = tmp_path / "data.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Image", "Pneumonia", "Edema"])
        writer.writeheader()
        writer.writerow({"Image": "img1.png", "Pneumonia": "-1", "Edema": "1"})

    img_path = tmp_path / "img1.png"
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    img.save(img_path)

    # Test with uncertain_to=0 (default)
    dataset = NIHBinarizedDataset(
        csv_path=str(csv_path),
        classes=["Pneumonia", "Edema"],
        images_root=str(tmp_path),
        uncertain_to=0,
    )

    x, y, meta = dataset[0]
    assert y[0].item() == 0.0  # -1 mapped to 0


def test_nih_binarized_uncertain_mapping_to_one(tmp_path):
    """Test uncertain value (-1) mapping to 1."""
    csv_path = tmp_path / "data.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Image", "Pneumonia"])
        writer.writeheader()
        writer.writerow({"Image": "img1.png", "Pneumonia": "-1"})

    img_path = tmp_path / "img1.png"
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    img.save(img_path)

    # Test with uncertain_to=1
    dataset = NIHBinarizedDataset(
        csv_path=str(csv_path),
        classes=["Pneumonia"],
        images_root=str(tmp_path),
        uncertain_to=1,
    )

    x, y, meta = dataset[0]
    assert y[0].item() == 1.0  # -1 mapped to 1


def test_nih_binarized_with_dict_transform(tmp_path):
    """Test NIHBinarizedDataset with dict-returning transform."""
    csv_path = tmp_path / "data.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Image", "A"])
        writer.writeheader()
        writer.writerow({"Image": "img.png", "A": "1"})

    img_path = tmp_path / "img.png"
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    img.save(img_path)

    def dict_transform(img):
        # Return dict with image key
        return {"image": torch.from_numpy(img).permute(2, 0, 1).float() / 255.0}

    dataset = NIHBinarizedDataset(
        csv_path=str(csv_path),
        classes=["A"],
        images_root=str(tmp_path),
        transform=dict_transform,
    )

    x, y, meta = dataset[0]
    assert x.shape[0] == 3  # RGB channels
    assert y[0].item() == 1.0


def test_csv_image_dataset_basic(tmp_path):
    """Test CSVImageDataset basic functionality."""
    import pandas as pd

    # Create image
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (32, 32), color=(100, 100, 100)).save(img_path)

    # Create CSV
    csv_path = tmp_path / "data.csv"
    pd.DataFrame([{"image_path": "test.jpg", "label": 1}]).to_csv(csv_path, index=False)

    ds = CSVImageDataset(str(csv_path), img_size=32, augment=False)
    assert len(ds) == 1

    x, y = ds[0]
    assert x.shape[-2:] == (32, 32)
    assert y.item() == 1.0


def test_csv_image_dataset_with_augment(tmp_path):
    """Test CSVImageDataset with augmentation."""
    import pandas as pd

    # Create image
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (32, 32), color=(150, 150, 150)).save(img_path)

    # Create CSV
    csv_path = tmp_path / "data.csv"
    pd.DataFrame([{"image_path": "test.jpg", "label": 0}]).to_csv(csv_path, index=False)

    ds = CSVImageDataset(str(csv_path), img_size=32, augment=True)
    x, y = ds[0]
    assert x.shape[-2:] == (32, 32)


def test_nih_with_img_root_fallback(tmp_path):
    """Test NIHBinarizedDataset with img_root parameter."""
    csv_path = tmp_path / "data.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Image", "A"])
        writer.writeheader()
        writer.writerow({"Image": "img.png", "A": "1"})

    img_path = tmp_path / "img.png"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(img_path)

    # Use img_root instead of images_root
    dataset = NIHBinarizedDataset(
        csv_path=str(csv_path), classes=["A"], img_root=str(tmp_path)
    )

    x, y, meta = dataset[0]
    assert x is not None


def test_nih_with_none_roots(tmp_path):
    """Test NIHBinarizedDataset with no explicit roots (uses CSV dir)."""
    csv_path = tmp_path / "data.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Image", "A"])
        writer.writeheader()
        writer.writerow({"Image": "img.png", "A": "0"})

    img_path = tmp_path / "img.png"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(img_path)

    # Neither images_root nor img_root specified
    dataset = NIHBinarizedDataset(csv_path=str(csv_path), classes=["A"])

    x, y, meta = dataset[0]
    assert x is not None
