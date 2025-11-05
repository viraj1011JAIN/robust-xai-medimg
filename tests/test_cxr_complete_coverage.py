# tests/test_cxr_complete_coverage.py
"""Complete coverage tests for CXR datasets."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data import cxr_datasets as C


def test_nihchestxray_basic(tmp_path):
    """Test NIHChestXray basic functionality."""
    # Create test image
    img_path = tmp_path / "imgs" / "test.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.random.rand(32, 32, 3) * 255).astype("uint8")).save(img_path)

    # Create CSV with required columns
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        [
            {
                "image_path": "imgs/test.png",
                "Atelectasis": 1,
                "Cardiomegaly": 0,
            }
        ]
    ).to_csv(csv_path, index=False)

    ds = C.NIHChestXray(
        str(csv_path),
        str(tmp_path),
        target_cols=["Atelectasis", "Cardiomegaly"],
        transform=None,
    )

    assert len(ds) == 1
    x, y = ds[0]
    assert y.shape == (2,)


def test_nihchestxray_with_transform(tmp_path):
    """Test NIHChestXray with albumentations transform."""
    from src.data.transforms import cxr_val

    img_path = tmp_path / "imgs" / "test.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8")).save(img_path)

    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        [
            {
                "image_path": "imgs/test.png",
                "Atelectasis": 1,
                "Cardiomegaly": 0,
                "Effusion": 1,
            }
        ]
    ).to_csv(csv_path, index=False)

    ds = C.NIHChestXray(
        str(csv_path),
        str(tmp_path),
        target_cols=["Atelectasis", "Cardiomegaly", "Effusion"],
        transform=cxr_val(32),
    )

    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, 32, 32)
    assert y.shape == (3,)


def test_padchest_if_available(tmp_path):
    """Test PadChest dataset if available."""
    if not hasattr(C, "PadChestCXRBase"):
        pytest.skip("PadChestCXRBase not available")

    img_path = tmp_path / "imgs" / "test.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.random.rand(32, 32, 3) * 255).astype("uint8")).save(img_path)

    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        [
            {
                "image_path": "imgs/test.png",
                "Atelectasis": 0,
                "Cardiomegaly": 1,
            }
        ]
    ).to_csv(csv_path, index=False)

    ds = C.PadChestCXRBase(
        str(csv_path),
        str(tmp_path),
        target_cols=["Atelectasis", "Cardiomegaly"],
        transform=None,
    )

    assert len(ds) == 1


def test_vindr_if_available(tmp_path):
    """Test VinDr dataset if available."""
    if not hasattr(C, "VinDrCXRBase"):
        pytest.skip("VinDrCXRBase not available")

    img_path = tmp_path / "imgs" / "test.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.random.rand(32, 32, 3) * 255).astype("uint8")).save(img_path)

    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        [
            {
                "image_path": "imgs/test.png",
                "Atelectasis": 1,
                "Cardiomegaly": 0,
            }
        ]
    ).to_csv(csv_path, index=False)

    ds = C.VinDrCXRBase(
        str(csv_path),
        str(tmp_path),
        target_cols=["Atelectasis", "Cardiomegaly"],
        transform=None,
    )

    assert len(ds) == 1


def test_cxr_missing_image_placeholder(tmp_path):
    """Test CXR dataset with missing image."""
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        [
            {
                "image_path": "nonexistent.png",
                "Atelectasis": 1,
                "Cardiomegaly": 0,
            }
        ]
    ).to_csv(csv_path, index=False)

    ds = C.NIHChestXray(
        str(csv_path),
        str(tmp_path),
        target_cols=["Atelectasis", "Cardiomegaly"],
        transform=None,
    )

    x, y = ds[0]
    # Should get placeholder
    assert x is not None


def test_cxr_invalid_csv_raises(tmp_path):
    """Test CXR dataset with invalid CSV."""
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame([{"wrong": "columns"}]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        C.NIHChestXray(
            str(csv_path), str(tmp_path), target_cols=["Atelectasis"], transform=None
        )
