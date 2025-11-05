# tests/test_derm_isic_len.py
"""Tests for dermatology (ISIC) dataset - comprehensive coverage."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.derm_datasets import ISICDataset


def test_isic_len_exercised(tmp_path):
    """Test ISIC dataset __len__ and __getitem__ methods."""
    img = tmp_path / "imgs" / "a.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.random.rand(16, 16, 3) * 255).astype("uint8")).save(img)

    csv = tmp_path / "isic.csv"
    pd.DataFrame(
        [
            {
                "image_path": "imgs/a.png",
                "label": 1,
                "center": "c",
                "age": 30,
                "sex": "m",
                "location": "b",
            }
        ]
    ).to_csv(csv, index=False)

    ds = ISICDataset(csv_path=str(csv), images_root=str(tmp_path))
    assert len(ds) == 1, "Dataset should have 1 sample"
    x, y, meta = ds[0]

    # Verify image format
    if isinstance(x, torch.Tensor):
        assert x.shape[0] == 3, "Should have 3 channels in CHW format"
    else:
        assert 3 in x.shape, "Should have 3 channels"

    # Verify label is binary
    if isinstance(y, torch.Tensor):
        y_val = float(y.item())
    else:
        y_val = float(y)

    assert y_val in (0.0, 1.0), f"Label should be 0 or 1, got {y_val}"

    # Verify metadata keys
    assert "center" in meta, "Should have center metadata"
    assert "age" in meta, "Should have age metadata"
    assert "sex" in meta, "Should have sex metadata"
    assert "location" in meta, "Should have location metadata"
    assert "image" in meta, "Should have image path in metadata"


def test_isic_uncertain_label_mapping(tmp_path):
    """Test ISIC dataset uncertain label mapping."""
    img = tmp_path / "imgs" / "test.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.random.rand(16, 16, 3) * 255).astype("uint8")).save(img)

    csv = tmp_path / "isic.csv"
    pd.DataFrame(
        [
            {
                "image_path": "imgs/test.png",
                "label": -1,  # Uncertain label
                "center": "hosp",
                "age": 40,
                "sex": "f",
                "location": "leg",
            }
        ]
    ).to_csv(csv, index=False)

    # Test uncertain_to=0
    ds = ISICDataset(csv_path=str(csv), images_root=str(tmp_path), uncertain_to=0)
    x, y, meta = ds[0]

    y_val = float(y.item()) if isinstance(y, torch.Tensor) else float(y)
    assert y_val == 0.0, "Uncertain -1 should map to 0"

    # Test uncertain_to=1
    ds = ISICDataset(csv_path=str(csv), images_root=str(tmp_path), uncertain_to=1)
    x, y, meta = ds[0]

    y_val = float(y.item()) if isinstance(y, torch.Tensor) else float(y)
    assert y_val == 1.0, "Uncertain -1 should map to 1"


def test_isic_missing_image_placeholder(tmp_path):
    """Test ISIC dataset with missing image (should use placeholder)."""
    csv = tmp_path / "isic.csv"
    pd.DataFrame(
        [
            {
                "image_path": "nonexistent.png",
                "label": 0,
                "center": "test",
                "age": 25,
                "sex": "m",
                "location": "arm",
            }
        ]
    ).to_csv(csv, index=False)

    ds = ISICDataset(csv_path=str(csv), images_root=str(tmp_path))
    x, y, meta = ds[0]

    # Should get placeholder image
    assert x is not None, "Should return placeholder for missing image"
    assert x.shape[0] == 3, "Placeholder should have 3 channels"


def test_isic_invalid_csv_raises(tmp_path):
    """Test ISIC dataset with invalid CSV."""
    csv = tmp_path / "bad.csv"
    pd.DataFrame([{"wrong_col": "value"}]).to_csv(csv, index=False)

    with pytest.raises(ValueError, match="image_path"):
        ISICDataset(csv_path=str(csv), images_root=str(tmp_path))


def test_isic_transform_tensor_output(tmp_path):
    """Test ISIC dataset with transform returning tensor."""
    img = tmp_path / "imgs" / "test.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.random.rand(32, 32, 3) * 255).astype("uint8")).save(img)

    csv = tmp_path / "isic.csv"
    pd.DataFrame(
        [
            {
                "image_path": "imgs/test.png",
                "label": 1,
                "center": "c",
                "age": 35,
                "sex": "f",
                "location": "back",
            }
        ]
    ).to_csv(csv, index=False)

    # Transform that returns tensor directly
    def tensor_transform(img):
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    ds = ISICDataset(
        csv_path=str(csv), images_root=str(tmp_path), transform=tensor_transform
    )
    x, y, meta = ds[0]

    assert isinstance(x, torch.Tensor), "Should return tensor"
    assert x.shape[0] == 3, "Should be CHW format"


def test_isic_transform_ndarray_output(tmp_path):
    """Test ISIC dataset with transform returning ndarray."""
    img = tmp_path / "imgs" / "test.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.random.rand(32, 32, 3) * 255).astype("uint8")).save(img)

    csv = tmp_path / "isic.csv"
    pd.DataFrame(
        [
            {
                "image_path": "imgs/test.png",
                "label": 0,
                "center": "c",
                "age": 50,
                "sex": "m",
                "location": "chest",
            }
        ]
    ).to_csv(csv, index=False)

    # Transform that returns ndarray
    def array_transform(img):
        return img  # Return as-is

    ds = ISICDataset(
        csv_path=str(csv), images_root=str(tmp_path), transform=array_transform
    )
    x, y, meta = ds[0]

    assert isinstance(x, torch.Tensor), "Should convert ndarray to tensor"
