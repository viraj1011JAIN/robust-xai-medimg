# tests/test_nih_label_mapping.py
import numpy as np
import pandas as pd
import torch

import src.data.nih_binary as nih


def test_nih_uncertain_mapping(monkeypatch, tmp_path):
    """Test NIH dataset uncertain label mapping."""
    # Create fake CSV with uncertain labels
    df = pd.DataFrame(
        {
            "Image": ["x1.png", "x2.png"],
            "Finding": ["Pneumonia", "Effusion"],
            "Pneumonia": [1, -1],  # Second row has uncertain label
            "Effusion": [0, 1],
            "PatientID": [111, 222],
            "Site": ["A", "B"],
        }
    )
    csv = tmp_path / "nih.csv"
    df.to_csv(csv, index=False)

    # Mock image loader to return 224x224 grayscale
    def _load_img(_):
        arr = (np.random.rand(224, 224) * 255).astype(np.uint8)
        return arr

    monkeypatch.setattr(nih, "_imread_gray", _load_img)

    # Create dataset with uncertain_to=0 to map -1 to 0
    ds = nih.NIHBinarizedDataset(
        csv_path=str(csv),
        classes=["Pneumonia", "Effusion"],
        uncertain_to=0,
        transform=None,
    )

    # Test first sample (no uncertain labels)
    x0, y0, meta0 = ds[0]
    assert y0.shape[-1] == 2, "Should have 2 class labels"
    assert (y0.numpy() == np.array([1, 0])).all(), "First sample labels should be [1,0]"

    # Test second sample (uncertain -1 mapped to 0)
    x1, y1, meta1 = ds[1]
    assert (y1.numpy() == np.array([0, 1])).all(), "Uncertain -1 should map to 0"

    # Verify metadata
    assert "patient_id" in meta0 and "site" in meta0, "Should have patient metadata"
    assert isinstance(x0, torch.Tensor), "Image should be tensor"
    assert x0.ndim >= 2, "Image should be at least 2D"


def test_nih_uncertain_mapping_to_one(monkeypatch, tmp_path):
    """Test uncertain label mapping to 1."""
    df = pd.DataFrame(
        {
            "Image": ["x.png"],
            "Finding": ["A"],
            "A": [-1],
            "PatientID": [999],
            "Site": ["X"],
        }
    )
    csv = tmp_path / "nih.csv"
    df.to_csv(csv, index=False)

    def _load_img(_):
        return (np.random.rand(224, 224) * 255).astype(np.uint8)

    monkeypatch.setattr(nih, "_imread_gray", _load_img)

    ds = nih.NIHBinarizedDataset(
        csv_path=str(csv),
        classes=["A"],
        uncertain_to=1,  # Map -1 to 1
        transform=None,
    )

    _, y, _ = ds[0]
    assert y.numpy()[0] == 1, "Uncertain label should map to 1"
