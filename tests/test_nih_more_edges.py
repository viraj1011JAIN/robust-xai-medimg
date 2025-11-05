# tests/test_nih_more_edges.py
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

import src.data.nih_binary as N


def test__imread_gray_reads_pil(tmp_path):
    """Test grayscale image reading."""
    p = tmp_path / "g.png"
    Image.fromarray((np.random.rand(8, 8) * 255).astype("uint8")).convert("L").save(p)
    arr = N._imread_gray(str(p))
    assert arr.ndim == 2, "Should be 2D grayscale"
    assert arr.shape == (8, 8), "Should be 8x8"


def test_nih_missing_csv_raises():
    """Test that missing CSV raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        _ = N.NIHBinarizedDataset(csv_path="__nope__.csv", classes=["A"])


def test_nih_transform_dict_branch_and_meta(tmp_path):
    """Test NIH dataset with dict-returning transform."""
    # Create test image
    imgp = tmp_path / "x.png"
    Image.fromarray((np.random.rand(16, 16) * 255).astype("uint8")).convert("L").save(imgp)

    # Create CSV
    csv = tmp_path / "nih.csv"
    pd.DataFrame(
        [
            {
                "Image": "x.png",
                "Finding": "A",
                "A": 1,
                "B": 0,
                "PatientID": 123,
                "Site": "S1",
            }
        ]
    ).to_csv(csv, index=False)

    # Transform that returns dict to hit special branch
    def _tfm(img):
        assert isinstance(img, np.ndarray), "Input should be numpy array"
        return {"image": torch.from_numpy(img / 255.0).unsqueeze(0).float()}

    ds = N.NIHBinarizedDataset(
        csv_path=str(csv), classes=["A", "B"], transform=_tfm, images_root=str(tmp_path)
    )
    x, y, meta = ds[0]

    assert isinstance(x, torch.Tensor), "Image should be tensor"
    assert x.ndim == 3, "Image should be 3D (CHW)"
    assert "patient_id" in meta and "site" in meta, "Should have metadata"
    assert y.shape[-1] == 2, "Should have 2 class labels"


def test_nih_with_none_transform(tmp_path):
    """Test NIH dataset with None transform."""
    imgp = tmp_path / "y.png"
    Image.fromarray((np.random.rand(16, 16) * 255).astype("uint8")).convert("L").save(imgp)

    csv = tmp_path / "nih2.csv"
    pd.DataFrame(
        [
            {
                "Image": "y.png",
                "Finding": "C",
                "C": 1,
                "PatientID": 456,
                "Site": "S2",
            }
        ]
    ).to_csv(csv, index=False)

    ds = N.NIHBinarizedDataset(
        csv_path=str(csv),
        classes=["C"],
        transform=None,
        images_root=str(tmp_path),
    )

    x, y, meta = ds[0]
    assert x.shape == (16, 16) or x.ndim == 3, "Should have valid image shape"
