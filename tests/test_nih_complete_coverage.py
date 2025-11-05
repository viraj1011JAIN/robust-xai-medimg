# tests/test_nih_complete_coverage.py
"""Complete coverage tests for NIH binary dataset."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

import src.data.nih_binary as N


def test_csv_image_dataset_basic(tmp_path):
    """Test CSVImageDataset basic functionality."""
    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (32, 32), color=(100, 100, 100)).save(img_path)

    csv_path = tmp_path / "data.csv"
    pd.DataFrame([{"image_path": "img.jpg", "label": 1}]).to_csv(csv_path, index=False)

    ds = N.CSVImageDataset(str(csv_path), img_size=32, augment=False)
    assert len(ds) == 1

    x, y = ds[0]
    assert x.shape[-2:] == (32, 32)


def test_csv_image_dataset_with_augment(tmp_path):
    """Test CSVImageDataset with augmentation."""
    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (32, 32), color=(100, 100, 100)).save(img_path)

    csv_path = tmp_path / "data.csv"
    pd.DataFrame([{"image_path": "img.jpg", "label": 0}]).to_csv(csv_path, index=False)

    ds = N.CSVImageDataset(str(csv_path), img_size=32, augment=True)
    x, y = ds[0]
    assert x.shape[-2:] == (32, 32)


def test_nih_binarized_basic(tmp_path, monkeypatch):
    """Test NIHBinarizedDataset basic functionality."""
    csv_path = tmp_path / "nih.csv"
    pd.DataFrame(
        [
            {
                "Image": "test.png",
                "Finding": "A",
                "A": 1,
                "B": 0,
                "PatientID": 123,
                "Site": "S1",
            }
        ]
    ).to_csv(csv_path, index=False)

    def _load_img(_):
        return (np.random.rand(224, 224) * 255).astype(np.uint8)

    monkeypatch.setattr(N, "_imread_gray", _load_img)

    ds = N.NIHBinarizedDataset(
        csv_path=str(csv_path),
        classes=["A", "B"],
        transform=None,
    )

    assert len(ds) == 1
    x, y, meta = ds[0]
    assert y.shape == (2,)
    assert "patient_id" in meta


def test_nih_with_transform_dict(tmp_path, monkeypatch):
    """Test NIH dataset with dict-returning transform."""
    csv_path = tmp_path / "nih.csv"
    pd.DataFrame(
        [
            {
                "Image": "test.png",
                "Finding": "A",
                "A": 1,
                "PatientID": 456,
                "Site": "S2",
            }
        ]
    ).to_csv(csv_path, index=False)

    def _load_img(_):
        return (np.random.rand(224, 224) * 255).astype(np.uint8)

    monkeypatch.setattr(N, "_imread_gray", _load_img)

    def dict_transform(img):
        return {"image": torch.from_numpy(img / 255.0).unsqueeze(0).float()}

    ds = N.NIHBinarizedDataset(
        csv_path=str(csv_path),
        classes=["A"],
        transform=dict_transform,
    )

    x, y, meta = ds[0]
    assert isinstance(x, torch.Tensor)


def test_nih_with_images_root(tmp_path, monkeypatch):
    """Test NIH dataset with images_root parameter."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    csv_path = tmp_path / "nih.csv"
    pd.DataFrame(
        [
            {
                "Image": "images/test.png",
                "Finding": "A",
                "A": 0,
                "PatientID": 789,
                "Site": "S3",
            }
        ]
    ).to_csv(csv_path, index=False)

    def _load_img(_):
        return (np.random.rand(224, 224) * 255).astype(np.uint8)

    monkeypatch.setattr(N, "_imread_gray", _load_img)

    ds = N.NIHBinarizedDataset(
        csv_path=str(csv_path),
        classes=["A"],
        images_root=str(tmp_path),
    )

    x, y, meta = ds[0]
    assert x is not None


def test_nih_uncertain_labels(tmp_path, monkeypatch):
    """Test NIH dataset with uncertain labels."""
    csv_path = tmp_path / "nih.csv"
    pd.DataFrame(
        [
            {
                "Image": "test.png",
                "Finding": "A",
                "A": -1,  # Uncertain
                "PatientID": 111,
                "Site": "S1",
            }
        ]
    ).to_csv(csv_path, index=False)

    def _load_img(_):
        return (np.random.rand(224, 224) * 255).astype(np.uint8)

    monkeypatch.setattr(N, "_imread_gray", _load_img)

    # Test uncertain_to=0
    ds = N.NIHBinarizedDataset(
        csv_path=str(csv_path),
        classes=["A"],
        uncertain_to=0,
    )

    x, y, meta = ds[0]
    assert y.numpy()[0] == 0


def test_nih_missing_csv_raises():
    """Test NIH dataset with missing CSV file."""
    with pytest.raises(FileNotFoundError):
        N.NIHBinarizedDataset(
            csv_path="nonexistent.csv",
            classes=["A"],
        )


def test_imread_gray_function(tmp_path):
    """Test _imread_gray helper function."""
    img_path = tmp_path / "gray.png"
    Image.fromarray((np.random.rand(16, 16) * 255).astype("uint8")).convert("L").save(
        img_path
    )

    result = N._imread_gray(str(img_path))
    assert result.ndim == 2
    assert result.shape == (16, 16)
