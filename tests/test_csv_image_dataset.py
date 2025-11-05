# tests/test_csv_image_dataset.py
import csv
from pathlib import Path

import torch
from PIL import Image

from src.data.nih_binary import CSVImageDataset


def _write_img(p: Path, size=(16, 16), color=(128, 128, 128)):
    """Helper to create a test image."""
    img = Image.new("RGB", size, color)
    img.save(p)


def test_csv_image_dataset_happy_path(tmp_path):
    """Test CSVImageDataset loads images and labels correctly."""
    root = Path(tmp_path)
    # Create test images
    img1 = root / "a.jpg"
    img2 = root / "b.jpg"
    _write_img(img1)
    _write_img(img2)

    # Create CSV with image_path and label columns
    csv_path = root / "data.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "label"])
        w.writeheader()
        w.writerow({"image_path": "a.jpg", "label": 0})
        w.writerow({"image_path": "b.jpg", "label": 1})

    ds = CSVImageDataset(str(csv_path), img_size=32, augment=False)
    assert len(ds) == 2

    x0, y0 = ds[0]
    assert isinstance(x0, torch.Tensor), "Image should be a tensor"
    assert y0.dtype.is_floating_point, "Label should be float"
    assert x0.shape[-1] == 32 and x0.shape[-2] == 32, "Image should be 32x32"

    # Test second sample
    x1, y1 = ds[1]
    assert isinstance(x1, torch.Tensor)
    assert y1.dtype.is_floating_point


def test_csv_image_dataset_missing_headers_raises(tmp_path):
    """Test that missing required headers raises ValueError."""
    root = Path(tmp_path)
    bad_csv = root / "bad.csv"
    with bad_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["foo", "bar"])
        w.writeheader()
        w.writerow({"foo": "a.jpg", "bar": 0})

    try:
        _ = CSVImageDataset(str(bad_csv), img_size=32, augment=False)
        raised = False
    except ValueError as e:
        raised = "image_path,label" in str(e) or "image_path" in str(e)

    assert raised, "Should raise ValueError for missing headers"
